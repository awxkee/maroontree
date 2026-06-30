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

use crate::Speed;
use crate::dct::{
    adst4x4_t, adst4x8_t, adst8x8_t, adst16x16_t, adstdct4x4_t, adstdct4x8_t, adstdct8x8_t,
    adstdct16x16_t, dct16x8_t, dctadst4x4_t, dctadst4x8_t, dctadst8x8_t, dctadst16x16_t,
    fidentity8x8_t,
};
use crate::idct::{
    iadst_dequant_4x4, iadst_dequant_4x8, iadst_dequant_8x8, iadst_dequant_16x16,
    iadstdct_dequant_4x4, iadstdct_dequant_4x8, iadstdct_dequant_8x8, iadstdct_dequant_16x16,
    idct_dequant_4x4, idct_dequant_4x8, idct_dequant_8x8, idct_dequant_8x16, idct_dequant_16x8,
    idct_dequant_16x16, idct_dequant_16x32, idct_dequant_32x32, idctadst_dequant_4x4,
    idctadst_dequant_4x8, idctadst_dequant_8x8, idctadst_dequant_16x16, iidentity_dequant_8x8,
};
use crate::obu::{
    frame_header_lossy_multitile, frame_header_lossy_multitile_th, temporal_delimiter,
    wrap_obu_frame, wrap_obu_frame_split,
};
use crate::odec::OdEcEncoder;
#[cfg(test)]
pub(crate) static FORCE_SPLIT4: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(not(test))]
pub(crate) static FORCE_SPLIT4: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Runtime enable for the RD-selected 4x4 luma split (off by default; the path
/// is bit-exact but its quality benefit is still being measured).
pub(crate) static SPLIT4_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Test/debug gate: force every eligible 16x16 luma block to PARTITION_H (two
/// 16x8 sub-blocks). Used to validate the rectangular-partition path to
/// byte-exactness in isolation, exactly as FORCE_SPLIT4 did for the 4x4 split.
pub(crate) static FORCE_HORZ: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// Runtime enable for the RD-selected PARTITION_H (16x8) candidate. Off by
/// default; the path is byte-exact but its quality benefit is being measured.
pub(crate) static HORZ_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Partition decision for a 16x16 luma region.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part16 {
    None,
    Horz,
    Split,
}
use crate::trellis::{trellis_optimize, trellis_optimize_ctx};

use crate::coeffs::encode_tx16_coeffs_adapt;
use crate::coeffs::*;
use crate::cost::*;
use crate::intrapred::*;
use crate::quant::*;
use crate::tables::*;

/// AV1 chroma transform type derived from the intra mode (`Mode_To_Txfm`),
/// restricted to the kinds reachable from the directional chroma modes this
/// encoder offers. The decoder derives this from the signalled `uv_mode`, so the
/// encoder must use the matching forward/inverse pair for byte-exactness.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChromaTx {
    DctDct,
    AdstAdst,
    AdstDct,
    DctAdst,
}

/// `Mode_To_Txfm` for the chroma intra modes the encoder searches. SMOOTH/PAETH
/// (ADST_ADST), SMOOTH_V (ADST_DCT) and SMOOTH_H (DCT_ADST) need no angle_delta.
/// V_PRED (ADST_DCT) and H_PRED (DCT_ADST) are directional: they additionally
/// emit a chroma `angle_delta` symbol (delta 0), and are only offered where the
/// chroma block is >= 8x8 (4:4:4), matching AV1's `use_angle_delta`.
fn chroma_tx_for_mode(mode: usize) -> ChromaTx {
    match mode {
        m if m == PAETH_PRED || m == SMOOTH_PRED => ChromaTx::AdstAdst,
        m if m == SMOOTH_V_PRED || m == V_PRED => ChromaTx::AdstDct,
        m if m == SMOOTH_H_PRED || m == H_PRED => ChromaTx::DctAdst,
        _ => ChromaTx::DctDct,
    }
}

/// Forward transform + trellis quant for an 8x8 chroma block under the given
/// chroma tx kind. Returns levels + unrounded targets like the other `*_t`.
fn fwd_chroma_8x8(tx: ChromaTx, resid: &[i32; 64], q: &impl Dct) -> ([i32; 64], [f64; 64]) {
    match tx {
        ChromaTx::DctDct => forward_dct_quant_8x8_t(resid, q),
        ChromaTx::AdstAdst => adst8x8_t(resid, q),
        ChromaTx::AdstDct => adstdct8x8_t(resid, q),
        ChromaTx::DctAdst => dctadst8x8_t(resid, q),
    }
}

fn inv_chroma_8x8(tx: ChromaTx, levels: &[i32; 64], q: &impl Dct) -> [i32; 64] {
    match tx {
        ChromaTx::DctDct => idct_dequant_8x8(levels, q),
        ChromaTx::AdstAdst => iadst_dequant_8x8(levels, q),
        ChromaTx::AdstDct => iadstdct_dequant_8x8(levels, q),
        ChromaTx::DctAdst => idctadst_dequant_8x8(levels, q),
    }
}

fn fwd_chroma_4x4(tx: ChromaTx, resid: &[i32; 16], q: &impl Dct) -> ([i32; 16], [f64; 16]) {
    match tx {
        ChromaTx::DctDct => forward_dct_quant_4x4_t(resid, q),
        ChromaTx::AdstAdst => adst4x4_t(resid, q),
        ChromaTx::AdstDct => adstdct4x4_t(resid, q),
        ChromaTx::DctAdst => dctadst4x4_t(resid, q),
    }
}

fn inv_chroma_4x4(tx: ChromaTx, levels: &[i32; 16], q: &impl Dct) -> [i32; 16] {
    match tx {
        ChromaTx::DctDct => idct_dequant_4x4(levels, q),
        ChromaTx::AdstAdst => iadst_dequant_4x4(levels, q),
        ChromaTx::AdstDct => iadstdct_dequant_4x4(levels, q),
        ChromaTx::DctAdst => idctadst_dequant_4x4(levels, q),
    }
}

fn fwd_chroma_4x8(tx: ChromaTx, resid: &[i32; 32], q: &impl Dct) -> ([i32; 32], [f64; 32]) {
    match tx {
        ChromaTx::DctDct => forward_dct_quant_4x8_t(resid, q),
        ChromaTx::AdstAdst => adst4x8_t(resid, q),
        ChromaTx::AdstDct => adstdct4x8_t(resid, q),
        ChromaTx::DctAdst => dctadst4x8_t(resid, q),
    }
}

fn inv_chroma_4x8(tx: ChromaTx, levels: &[i32; 32], q: &impl Dct) -> [i32; 32] {
    match tx {
        ChromaTx::DctDct => idct_dequant_4x8(levels, q),
        ChromaTx::AdstAdst => iadst_dequant_4x8(levels, q),
        ChromaTx::AdstDct => iadstdct_dequant_4x8(levels, q),
        ChromaTx::DctAdst => idctadst_dequant_4x8(levels, q),
    }
}

/// Per-frame adaptive CDF state. dav1d adapts every symbol's CDF as it decodes
/// (when `disable_cdf_update = 0`); to stay bit-exact we hold the same mutable
/// CDFs (initialised from the qcat defaults, in `icdf` form with a trailing
/// adaptation count) and adapt them identically after each coded symbol via
/// `OdEcEncoder::encode_symbol`. Coef class index: 0 = `TX_4X4` (4:2:0 chroma),
/// 1 = `TX_8X8`/`RTX_4X8` (luma, 4:4:4 and 4:2:2 chroma). Plane index: 0 = luma,
/// 1 = chroma.
pub(crate) struct Cdfs {
    pub(crate) skip: Vec<Vec<u16>>,               // block skip [3 ctx]
    pub(crate) part_bl8: Vec<Vec<u16>>,           // PARTITION_NONE @ 8x8 [4 ctx]
    pub(crate) part_split: Vec<Vec<Vec<u16>>>,    // SPLIT [bl-1=0..3][4 ctx]
    pub(crate) kf_y: Vec<Vec<u16>>,               // kf_y_mode[5*5], index [above_ctx*5 + left_ctx]
    pub(crate) uv_mode: Vec<Vec<u16>>,            // uv_mode[2*13], index [cfl_allowed*13 + y_mode]
    pub(crate) angle_delta: Vec<Vec<u16>>,        // angle_delta[8 directional modes]
    pub(crate) cfl_sign: Vec<u16>,                // cfl joint-sign (8 symbols)
    pub(crate) cfl_alpha: Vec<Vec<u16>>,          // cfl alpha magnitude [6 ctx]
    pub(crate) txtp: Vec<Vec<u16>>,               // intra txtp TX_8X8 luma, per intra mode [13]
    pub(crate) txtp4: Vec<Vec<u16>>,              // intra txtp TX_4X4 luma, per intra mode [13]
    pub(crate) txtp16: Vec<Vec<u16>>,             // intra txtp TX_16X16 luma, per intra mode [13]
    pub(crate) txb_skip: [Vec<Vec<u16>>; 4],      // [class][13 ctx] (class 3 = TX_32X32)
    pub(crate) base_tok: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][41/42 ctx]
    pub(crate) br_tok: [[Vec<Vec<u16>>; 2]; 4],   // [class][plane][21 ctx]
    pub(crate) eob_base: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][4 ctx]
    pub(crate) eob_hi: [[Vec<Vec<u16>>; 2]; 4],   // [class][plane][11 bins], each a 2-sym CDF
    pub(crate) dc_sign: [Vec<Vec<u16>>; 2],       // [plane][3 ctx]
    pub(crate) eob_bin_16_c: Vec<u16>,            // chroma, 4x4
    pub(crate) eob_bin_16_l: Vec<u16>,            // luma, 4x4
    pub(crate) eob_bin_32_c: Vec<u16>,            // chroma, 4x8
    pub(crate) eob_bin_64_l: Vec<u16>,            // luma, 8x8
    pub(crate) eob_bin_64_c: Vec<u16>,            // chroma, 8x8
    pub(crate) eob_bin_256_l: Vec<u16>,           // luma, 16x16 (class 2)
    pub(crate) eob_bin_256_c: Vec<u16>,           // chroma, 16x16 (class 2)
    pub(crate) eob_bin_128_c: Vec<u16>,           // chroma, RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_128_l: Vec<u16>,           // luma, RTX_16X8/RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_1024_l: Vec<u16>,          // luma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_1024_c: Vec<u16>,          // chroma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_512_c: Vec<u16>,           // chroma, RTX_16X32 (class 3, 512 coeffs)
    pub(crate) delta_q: Vec<u16>,                 // superblock delta-q magnitude (4 symbols)
    pub(crate) wiener_restore: Vec<u16>,          // use_wiener flag (2-symbol)
}

impl Cdfs {
    fn new(qctx: usize) -> Self {
        use crate::coef_q as Q;
        let rows = |t: &[[u16; 3]]| t.iter().map(|r| icdf(r)).collect::<Vec<_>>();
        let rows2 = |t: &[[u16; 2]]| t.iter().map(|r| icdf(r)).collect::<Vec<_>>();
        let his = |t: &[u16]| t.iter().map(|&v| icdf(&[v])).collect::<Vec<_>>();
        // skip CDFs by tx class
        let txb_skip = [
            Q::SKIP_TX4[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
            Q::SKIP_TX8[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
            Q::SKIP_TX16[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
            Q::SKIP_TX32[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
        ];
        // base/br/eob_base/eob_hi per [class][plane]
        let base_tok = [
            [
                rows(&Q::BASE_TOK_TX4_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX4_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BASE_TOK_TX8_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX8_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BASE_TOK_TX16_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX16_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BASE_TOK_TX32_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX32_CHROMA_Q[qctx]),
            ],
        ];
        let br_tok = [
            [
                rows(&Q::BR_TOK_TX4_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX4_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BR_TOK_TX8_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX8_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BR_TOK_TX16_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX16_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BR_TOK_TX32_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX32_CHROMA_Q[qctx]),
            ],
        ];
        let eob_base = [
            [
                rows2(&Q::EOB_BASE_TX4_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX4_CHROMA_Q[qctx]),
            ],
            [
                rows2(&Q::EOB_BASE_TX8_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX8_CHROMA_Q[qctx]),
            ],
            [
                rows2(&Q::EOB_BASE_TX16_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX16_CHROMA_Q[qctx]),
            ],
            [
                rows2(&Q::EOB_BASE_TX32_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX32_CHROMA_Q[qctx]),
            ],
        ];
        let eob_hi = [
            [
                his(&Q::EOB_HI_TX4_LUMA[qctx]),
                his(&Q::EOB_HI_TX4_CHROMA[qctx]),
            ],
            [
                his(&Q::EOB_HI_TX8_LUMA[qctx]),
                his(&Q::EOB_HI_TX8_CHROMA[qctx]),
            ],
            [
                his(&Q::EOB_HI_TX16_LUMA[qctx]),
                his(&Q::EOB_HI_TX16_CHROMA[qctx]),
            ],
            [
                his(&Q::EOB_HI_TX32_LUMA[qctx]),
                his(&Q::EOB_HI_TX32_CHROMA[qctx]),
            ],
        ];
        Cdfs {
            skip: SKIP_CDF.iter().map(|&v| icdf(&[v])).collect(),
            part_bl8: PART_BL8_CDF.iter().map(|r| icdf(r)).collect(),
            part_split: PART_SPLIT_CDF
                .iter()
                .map(|lvl| lvl.iter().map(|r| icdf(r)).collect())
                .collect(),
            kf_y: {
                let mut v = Vec::with_capacity(25);
                #[allow(clippy::needless_range_loop)]
                for a in 0..5 {
                    for l in 0..5 {
                        v.push(icdf(&KF_Y_MODE_CDF[a][l]));
                    }
                }
                v
            },
            angle_delta: ANGLE_DELTA_CDF.iter().map(|r| icdf(r)).collect(),
            cfl_sign: icdf(&CFL_SIGN_CDF),
            cfl_alpha: CFL_ALPHA_CDF.iter().map(|r| icdf(r)).collect(),
            uv_mode: {
                let mut v = Vec::with_capacity(26);
                #[allow(clippy::needless_range_loop)]
                for m in 0..13 {
                    v.push(icdf(&UV_MODE_NOCFL_CDF[m]));
                }
                #[allow(clippy::needless_range_loop)]
                for m in 0..13 {
                    v.push(icdf(&UV_MODE_CFL_CDF[m]));
                }
                v
            },
            txtp: TXTP_INTRA1_TX8.iter().map(|r| icdf(r)).collect(),
            txtp16: TXTP_INTRA2_TX16.iter().map(|r| icdf(r)).collect(),
            txtp4: TXTP_INTRA1_TX4.iter().map(|r| icdf(r)).collect(),
            txb_skip,
            base_tok,
            br_tok,
            eob_base,
            eob_hi,
            dc_sign: [
                Q::DC_SIGN_Q[qctx][0].iter().map(|&v| icdf(&[v])).collect(),
                Q::DC_SIGN_Q[qctx][1].iter().map(|&v| icdf(&[v])).collect(),
            ],
            eob_bin_16_c: icdf(&Q::EOB_BIN_16_CHROMA[qctx]),
            eob_bin_16_l: icdf(&Q::EOB_BIN_16_LUMA[qctx]),
            eob_bin_32_c: icdf(&Q::EOB_BIN_32_CHROMA[qctx]),
            eob_bin_64_l: icdf(&Q::EOB_BIN_64_LUMA[qctx]),
            eob_bin_64_c: icdf(&Q::EOB_BIN_64_CHROMA[qctx]),
            eob_bin_256_l: icdf(&Q::EOB_BIN_256_LUMA[qctx]),
            eob_bin_256_c: icdf(&Q::EOB_BIN_256_CHROMA[qctx]),
            eob_bin_128_c: icdf(&Q::EOB_BIN_128_CHROMA[qctx]),
            eob_bin_128_l: icdf(&Q::EOB_BIN_128_LUMA[qctx]),
            eob_bin_1024_l: icdf(&Q::EOB_BIN_1024_LUMA[qctx]),
            eob_bin_1024_c: icdf(&Q::EOB_BIN_1024_CHROMA[qctx]),
            eob_bin_512_c: icdf(&Q::EOB_BIN_512_CHROMA[qctx]),
            // AV1 Default_Delta_Q_Cdf = AOM_CDF4(28160, 32120, 32677); a single
            // (context-free) 4-symbol CDF for the delta-q magnitude token. Adapts
            // like every other symbol via OdEcEncoder::encode_symbol.
            delta_q: icdf(&[28160, 32120, 32677]),
            // Default LrWiener (use_wiener) CDF (AV1 Default_Wiener_Restore_Cdf).
            wiener_restore: icdf(&[11570]),
        }
    }
}

// ============================ Adaptive quantization ==========================
//
// Variance-based adaptive quantization for AV1, signalled through the standard
// superblock delta-Q mechanism (frame header `delta_q_present = 1`, and a
// `read_delta_qindex()` token in the first block of every superblock). The
// quantizer is reallocated per 64x64 superblock from local luma activity: flat
// regions (smooth sky / water, where banding and blocking are most visible) get
// a finer quantizer, busy/textured regions (where quantization error is masked)
// get a coarser one. The deltas are centred on the per-tile mean activity, so
// the average quantizer still tracks the requested base_q_idx and the rate stays
// close to the non-AQ encode while perceived quality improves.
//
// This mirrors the AV2 path in `av2/aq.rs`; the only real difference is the
// transport (AV1's `read_delta_qindex` symbol vs AV2's delta-q signalling) and
// the fixed-point luma plane (`i32` here, `f32` there).

/// log-resolution of the delta-q step: the signalled `delta_q_res` is `1 << this`
/// (so a step of 4 qindex units). Matches `av2/aq.rs::AQ_RES_LOG2`.
const AQ_RES_LOG2: u8 = 2;
/// Same value, exposed for the frame-header writer (`delta_q_res`) so the
/// signalled resolution always matches the per-SB step used by the encoder.
pub(crate) const AQ_DELTA_Q_RES_LOG2: u8 = AQ_RES_LOG2;
/// Spec limit: a single `read_delta_qindex` token without the literal-extension
/// escape can carry magnitudes 0..=2 directly; we keep the per-SB step within a
/// modest range and never need the `DELTA_Q_SMALL` escape (3+).
const AQ_MAX_STEPS: i32 = 12;
/// qindex per unit of log-activity (how hard flat vs busy regions are pushed).
/// Tuned on photographic stills; override at run time with `AQ_SLOPE`.
const AQ_SLOPE: f32 = 5.0;
/// per-superblock qindex delta clamp, before res-quantization.
const AQ_MAX_DELTA: f32 = 28.0;

/// Variance Boost configuration, carried from the encoder option down to the
/// per-tile [`AqCtx`]. `enabled == false` selects the classic whole-SB AQ (the
/// shipping default); when enabled the per-SB target comes from the octile of the
/// 64 8x8-subblock variances instead. Knobs mirror `av2/mod.rs::Tuning`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VarianceBoost {
    pub enabled: bool,
    pub octile: u8,
    pub strength: f32,
    pub boost_only: bool,
}

impl VarianceBoost {
    /// Disabled — classic whole-SB AQ, byte-identical to the pre-VB encoder.
    pub(crate) const fn off() -> Self {
        VarianceBoost {
            enabled: false,
            octile: 6,
            strength: 1.0,
            boost_only: false,
        }
    }

    pub(crate) fn on() -> Self {
        VarianceBoost {
            enabled: true,
            octile: 6,
            strength: 0.6,
            boost_only: true,
        }
    }
}

/// Mean+variance of the (up to) 64x64 luma region whose top-left is
/// `(sb_x, sb_y)`, returned as `ln(1 + variance)` — a perceptually reasonable
/// "activity" that compresses the huge dynamic range of raw variance. Operates
/// on the padded `i32` luma plane of stride `pw`.
fn sb_activity(
    yp: &[i32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
) -> f32 {
    let h = height.saturating_sub(sb_y).min(64);
    let w = width.saturating_sub(sb_x).min(64);
    if h == 0 || w == 0 {
        return 0.0;
    }
    let mut sum = 0i64;
    let mut sum2 = 0i64;
    for r in 0..h {
        let base = (sb_y + r) * pw + sb_x;
        for &c in &yp[base..base + w] {
            let v = c as i64;
            sum += v;
            sum2 += v * v;
        }
    }
    let n = (h * w) as f64;
    let mean = sum as f64 / n;
    let var = (sum2 as f64 / n - mean * mean).max(0.0);
    (1.0 + var).ln() as f32
}

/// Mean activity over every superblock of the tile — the reference the per-SB
/// deltas are centred on, so the deltas are (approximately) zero-mean and the
/// average quantizer tracks the base.
fn tile_ref_activity(yp: &[i32], pw: usize, w: usize, h: usize) -> f32 {
    let mut sum = 0f32;
    let mut cnt = 0f32;
    for sb_y in (0..h).step_by(64) {
        for sb_x in (0..w).step_by(64) {
            sum += sb_activity(yp, pw, sb_y, sb_x, w, h);
            cnt += 1.0;
        }
    }
    if cnt > 0.0 { sum / cnt } else { 5.0 }
}

/// Map a superblock's `activity` to a target qindex, centred on the tile mean
/// `ref_act`. Below-average activity (flat) → finer quantizer (lower qindex);
/// above-average (busy) → coarser. Clamped to a sane qindex range.
///
/// The slope, max delta, and the asymmetry factor applied to the coarsening
/// (positive-delta) side are compile-time constants.
fn aq_params() -> (f32, f32, f32) {
    // (slope, max delta, coarsen scale). Coarsen scale 1.0 == pure variance.
    (AQ_SLOPE, AQ_MAX_DELTA, 1.0)
}

fn aq_target_qidx(base_q: i32, activity: f32, ref_act: f32) -> i32 {
    let (slope, maxd, coarsen) = aq_params();
    let mut delta = (activity - ref_act) * slope;
    if delta > 0.0 {
        // Busy/textured: quantization error is masked, so coarsening is "free"
        // perceptually. `coarsen` < 1 spends fewer of those saved bits there and
        // more refining flats (better for perceptual/SSIM); = 1 is pure variance.
        delta *= coarsen;
    }
    let delta = delta.clamp(-maxd, maxd);
    (base_q + delta.round() as i32).clamp(1, 255)
}

// ----------------------------------------------------------------------------
// Variance Boost (variance-adaptive delta-Q) — the AV2 `av2/aq.rs` scheme ported
// to the AV1 transport. Where the classic whole-SB AQ above uses a single 64x64
// variance, Variance Boost splits each 64x64 superblock into 64 8x8 subblocks,
// computes each subblock's variance, then picks ONE representative variance at a
// configurable octile of the sorted set. Low picked variance => low local
// contrast => the eye still resolves detail, so the quantizer is *lowered*
// (quality boosted); high picked variance => texture masks artifacts, so it is
// nudged coarser (rate-balanced mode) or left alone (boost-only mode). The octile
// controls selectivity: a low octile boosts a SB if only a fraction of it is
// low-variance (more bits), a high octile boosts only when (nearly) the whole SB
// is low-variance. This is the same curve and defaults as `av2/aq.rs`; only the
// luma plane type differs (`i32` fixed-point here, `f32` in the AV2 path).

/// Per-8x8-subblock variances of a 64x64 superblock, written into `out` (length
/// 64, row-major over the 8x8 grid). Subblocks outside the frame (partial edge SB)
/// are filled with the mean of the in-frame subblocks so they don't bias the
/// octile pick. Returns the count of in-frame subblocks (>=1 when the SB has any
/// pixels). Mirrors `av2/aq.rs::sb_subblock_variances`.
fn aq_sb_subblock_variances(
    yp: &[i32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    out: &mut [f32; 64],
) -> usize {
    let mut filled = 0usize;
    let mut acc = 0f64;
    for by in 0..8 {
        for bx in 0..8 {
            let y0 = sb_y + by * 8;
            let x0 = sb_x + bx * 8;
            let h = height.saturating_sub(y0).min(8);
            let w = width.saturating_sub(x0).min(8);
            let idx = by * 8 + bx;
            if h == 0 || w == 0 {
                out[idx] = f32::NAN; // out-of-frame, patched below
                continue;
            }
            let mut sum = 0i64;
            let mut sum2 = 0i64;
            for r in 0..h {
                let base = (y0 + r) * pw + x0;
                for &v in &yp[base..base + w] {
                    let v = v as i64;
                    sum += v;
                    sum2 += v * v;
                }
            }
            let n = (h * w) as f64;
            let mean = sum as f64 / n;
            let var = (sum2 as f64 / n - mean * mean).max(0.0) as f32;
            out[idx] = var;
            acc += var as f64;
            filled += 1;
        }
    }
    if filled == 0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    let mean = (acc / filled as f64) as f32;
    for v in out.iter_mut() {
        if v.is_nan() {
            *v = mean;
        }
    }
    filled
}

/// Representative variance at the requested `octile` (1..=8) of the 64 sorted 8x8
/// variances. Octile 1 = most low-variance-biased (boost readily), octile 8 = only
/// the maximum. Mirrors `av2/aq.rs::sb_octile_variance`.
fn aq_sb_octile_variance(subvars: &mut [f32; 64], octile: u8) -> f32 {
    subvars.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let o = octile.clamp(1, 8) as usize;
    let idx = (o * 8 - 1).min(63);
    subvars[idx]
}

/// Variance Boost qindex delta for one superblock. `picked_var` is the octile
/// variance; `ref_log` is the per-tile reference (mean whole-SB log-variance) used
/// to keep the coarse side roughly zero-mean. Operates in log-variance space.
/// Below `LOW_LOG` the SB is "low contrast": qindex is lowered up to `MAX_BOOST`.
/// Above the reference it is nudged coarser (bounded by `MAX_CUT`) unless
/// `boost_only`. `strength` scales the whole effect (1.0 = nominal). Mirrors
/// `av2/aq.rs::variance_boost_delta` exactly.
fn aq_variance_boost_delta(picked_var: f32, ref_log: f32, strength: f32, boost_only: bool) -> i32 {
    let v_log = (1.0 + picked_var).ln();
    const LOW_LOG: f32 = 5.549_076; // (1.0 + 256.0).ln()
    const MAX_BOOST: f32 = 18.0;
    const MAX_CUT: f32 = 10.0;
    const BOOST_SLOPE: f32 = 5.0;
    const CUT_SLOPE: f32 = 3.0;

    if v_log < LOW_LOG {
        let d = ((LOW_LOG - v_log) * BOOST_SLOPE * strength).min(MAX_BOOST);
        -(d.round() as i32)
    } else if boost_only {
        0
    } else {
        let over = (v_log - ref_log.max(LOW_LOG)).max(0.0);
        let d = (over * CUT_SLOPE * strength).min(MAX_CUT);
        d.round() as i32
    }
}

/// Per-tile adaptive-quantization state held on the [`LossyTile`]. When
/// `enabled` is false every method is a no-op and the tile quantizes at the
/// fixed base, byte-identical to the pre-AQ encoder.
struct AqCtx {
    enabled: bool,
    /// frame `base_q_idx`: the anchor the per-SB deltas are measured from and the
    /// value `CurrentQIndex` is reset to at tile start.
    base_q: u8,
    /// `delta_q_res` (log2): the per-SB step is `1 << res_log2` qindex units.
    res_log2: u8,
    /// decoder `CurrentQIndex`, updated by each signalled delta; reset to `base_q`
    /// at the start of the tile (delta-Q does not persist across tiles).
    cur_qidx: i32,
    /// tile mean activity, the zero-delta reference (see [`tile_ref_activity`]).
    ref_act: f32,
    /// armed at the start of each superblock; the first coded block emits the
    /// `read_delta_qindex` token and disarms it (spec `ReadDeltas`).
    read_deltas: bool,
    /// `reducedDeltaQIndex` (pre-`<<res`) to emit at the first block of the SB.
    pending: i32,
    /// When true, use the Variance Boost scheme (octile of 8x8 subblock variances)
    /// instead of the classic whole-SB variance for the per-SB target. Gated behind
    /// the `variance_boost` encoder option; off => classic AQ, byte-identical.
    vb_enabled: bool,
    /// Variance Boost selectivity octile (1..=8). Default 6 (SVT-AV1-PSY default).
    vb_octile: u8,
    /// Variance Boost strength multiplier (1.0 = nominal).
    vb_strength: f32,
    /// When true, only boost low-variance SBs (net-negative, spends bits). When
    /// false, also coarsen high-variance SBs to keep the rate roughly matched.
    vb_boost_only: bool,
}

impl AqCtx {
    fn off() -> Self {
        AqCtx {
            enabled: false,
            base_q: 0,
            res_log2: 0,
            cur_qidx: 0,
            ref_act: 0.0,
            read_deltas: false,
            pending: 0,
            vb_enabled: false,
            vb_octile: 6,
            vb_strength: 1.0,
            vb_boost_only: false,
        }
    }
}

/// Whole-frame lossy encoder state. Context arrays are indexed by absolute frame
/// coordinates: the above arrays persist down the superblock rows, and the left
/// arrays are naturally fresh per SB row (each row occupies a distinct
/// coordinate range), mirroring dav1d's per-SB-row left reset.
struct LossyTile<'a> {
    bd: u8,
    quant: Quant,
    cquant: Quant,
    w: usize,
    h: usize,
    cw: usize,   // chroma plane width (= w for 4:4:4, w/2 for 4:2:2 and 4:2:0)
    ss422: bool, // chroma horizontally subsampled (4:2:2)
    ss420: bool, // chroma horizontally + vertically subsampled (4:2:0)
    mono: bool,  // monochrome: code luma only (NumPlanes=1, no chroma syntax)
    src: &'a [Vec<i32>; 3],
    recon: [Vec<i32>; 3],
    a_coef: [Vec<u8>; 3], // len w/4, absolute bx4
    l_coef: [Vec<u8>; 3], // len h/4, absolute by4
    a_part: Vec<u8>,      // len w/8, absolute x8
    l_part: Vec<u8>,      // len h/8, absolute y8
    a_skip: Vec<u8>,      // block skip flag per 4x4 col, absolute bx4
    l_skip: Vec<u8>,      // block skip flag per 4x4 row, absolute by4
    a_mode: Vec<u8>,      // luma intra mode per 4x4 col (for kf y-mode context)
    l_mode: Vec<u8>,      // luma intra mode per 4x4 row
    blk4: Vec<u8>, // luma block WIDTH (in 4-sample units) per 4x4 luma unit; for the deblock filter (vertical edges)
    blk4h: Vec<u8>, // luma block HEIGHT (in 4-sample units) per 4x4 luma unit; for the deblock filter (horizontal edges)
    skip8: Vec<bool>, // per-8x8-luma-unit block skip flag (true = no coded coeffs); for CDEF
    enc: OdEcEncoder,
    cdfs: Cdfs,
    /// RDO effort: [`Speed::Slow`] (default) or [`Speed::Fast`] (winner-only
    /// RDOQ, DCT-only transform choice, reduced intra mode set).
    speed: Speed,
    /// Adaptive-quantization state; `AqCtx::off()` unless enabled per tile.
    aq: AqCtx,
    /// Global luma Wiener filter to signal per superblock (`read_lr`), or `None`
    /// for RESTORE_NONE (no per-SB LR syntax emitted). When set, every 64x64
    /// restoration unit codes `use_wiener = 1` with these taps, delta-coded
    /// against the running reference `lr_ref_*` (spec 5.11.58).
    wiener: Option<crate::wiener::WienerUnit>,
    /// Running Wiener tap reference for delta coding (horizontal, vertical), one
    /// per coded tap. Reset to the spec midpoints at the start of the tile.
    lr_ref_h: [i32; 3],
    lr_ref_v: [i32; 3],
    /// Frame-absolute luma pixel origin of this tile and the full frame luma
    /// dimensions. Loop-restoration units are frame-relative, so `read_lr` is
    /// computed in frame coordinates even though the tile encodes locally.
    /// (`0,0` and the tile's own size for a single-tile frame.)
    frame_x0: usize,
    frame_y0: usize,
    frame_w: usize,
    frame_h: usize,
}

impl<'a> LossyTile<'a> {
    fn new(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; w * h], vec![0; w * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
        }
    }

    /// Monochrome tile: codes the luma plane only (`NumPlanes = 1`). Only
    /// `src[0]` is used; the chroma reconstruction and context arrays are left
    /// empty so any stray chroma access panics instead of corrupting output.
    /// Forces 8x8 luma transforms (see `prefer_16x16`/`prefer_32x32`).
    fn new_mono(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: true,
            src,
            recon: [vec![0; w * h], Vec::new(), Vec::new()],
            a_coef: [vec![0x40; w / 4], Vec::new(), Vec::new()],
            l_coef: [vec![0x40; h / 4], Vec::new(), Vec::new()],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
        }
    }

    /// 4:2:2 tile: luma is full w x h, chroma planes are subsampled to (w/2) x h.
    /// `src[1]`/`src[2]` must already be the half-width chroma planes.
    fn new_422(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        let cw = w / 2;
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw,
            ss422: true,
            ss420: false,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; cw * h], vec![0; cw * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
        }
    }

    /// 4:2:0 tile: luma is full w x h, chroma planes are subsampled to
    /// (w/2) x (h/2). `src[1]`/`src[2]` must already be the quarter-size planes.
    fn new_420(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        let (cw, ch) = (w / 2, h / 2);
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw,
            ss422: false,
            ss420: true,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; cw * ch], vec![0; cw * ch]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; ch / 4], vec![0x40; ch / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
        }
    }

    fn skip_ctx(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0 // luma: TX size == block size -> ctx 0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = (a[bx4] != 0x40 || a[bx4 + 1] != 0x40) as usize;
            let cl = (l[by4] != 0x40 || l[by4 + 1] != 0x40) as usize;
            7 + ca + cl
        }
    }

    fn dc_sign_ctx(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma = (a[bx4] >> 6) as i32 + (a[bx4 + 1] >> 6) as i32;
        let suml = (l[by4] >> 6) as i32 + (l[by4 + 1] >> 6) as i32;
        let s = suma + suml - 4;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a 16x16 transform. Luma: tx == block size -> 0.
    /// Chroma (4:4:4, chroma tx == chroma block): `7 + above_nz + left_nz` over
    /// the 4-unit (16-sample) footprint (`get_txb_skip_ctx`, ctx_offset = 7).
    fn skip_ctx_16(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
            let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
            7 + ca + cl
        }
    }

    /// dc_sign context for a 16x16 transform (4-unit footprint, baseline -8).
    fn dc_sign_ctx_16(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 8;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a luma RTX_16X8 block coded as a single transform.
    /// dav1d `get_skip_ctx` returns 0 when the block size equals the transform
    /// size (which it does here: BS_16x8 == one RTX_16X8), same as square luma.
    fn skip_ctx_16x8_luma(&self) -> usize {
        0
    }

    /// dc_sign context for a luma RTX_16X8 transform: 4 above-units (16 wide) and
    /// 2 left-units (8 tall) top-bit sums, baseline -(4+2) = -6.
    fn dc_sign_ctx_16x8_luma(&self, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[0];
        let l = &self.l_coef[0];
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:4:4 chroma txb_skip context for an RTX_16X8 block (block == transform, so
    /// `not_one_blk` is 0): `7 + ca + cl` over 4 above-units and 2 left-units.
    fn skip_ctx_16x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:4:4 chroma dc_sign context for RTX_16X8: 4 above + 2 left, baseline -6.
    fn dc_sign_ctx_16x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// Decide whether to code the 16x16 region at (`x8`,`y8`) as a single
    /// TX_16X16 (PARTITION_NONE) vs splitting into four 8x8. This is a pure R-D
    /// proxy — the decoder follows whatever partition we signal, so the choice
    /// affects compression only, never correctness. Proxy: compare the summed
    /// absolute quantized luma levels of the one 16x16 transform (plus a small
    /// per-block overhead) against the four 8x8 transforms (each with its own
    /// overhead). Smooth regions compact into the 16x16 and win decisively.
    /// Peak-to-peak luma range of the `dim`x`dim` source block at 8-unit origin
    /// `(x8,y8)`. Smooth low-contrast blocks (small range) ring into visible
    /// low-frequency banding under large transforms, so the partitioner uses it
    /// to keep such blocks on small (8x8) transforms.
    fn block_luma_range(&self, x8: usize, y8: usize, dim: usize) -> i32 {
        let (px, py) = (x8 * 8, y8 * 8);
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for ry in 0..dim {
            let base = (py + ry) * self.w + px;
            for &s in &self.src[0][base..base + dim] {
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }
        }
        hi - lo
    }

    /// R-D proxy for coding an 8x8 luma region as one TX_8X8 (PARTITION_NONE)
    /// vs splitting into four BLOCK_4X4. Runs the real non-directional mode
    /// search (SSE + lambda*bits) for both options so the decision reflects
    /// 4x4's per-quadrant mode diversity, not just a DC estimate. Returns
    /// `true` to keep the 8x8 whole. Split is offered only for 4:2:0/4:4:4.
    fn prefer_8x8_none(&self, x8: usize, y8: usize) -> bool {
        if self.mono || self.ss422 {
            return true;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        let maxv = (1 << self.bd) - 1;
        let (dcq, acq) = (self.quant.dc_q() as f64, self.quant.ac_q() as f64);
        let lam = trellis_lambda();
        let mlam = mode_lambda() * acq * acq;
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // best non-directional cost for one 8x8 (DCT_DCT only; cheap proxy)
        let mut eff8 = f64::INFINITY;
        for &m in modes {
            let mut pred = [0i32; 64];
            if m == DC_PRED {
                let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    false,
                    false,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            for ry in 0..8 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for rx in 0..8 {
                    resid[ry * 8 + rx] = srow[rx] - pred[ry * 8 + rx];
                }
            }
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_8X8, lam);
            let rr = idct_dequant_8x8(&cf, &self.quant);
            let mut sse = 0i64;
            for ry in 0..8 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for rx in 0..8 {
                    let r = (pred[ry * 8 + rx] + rr[ry * 8 + rx]).clamp(0, maxv);
                    let d = srow[rx] - r;
                    sse += (d * d) as i64;
                }
            }
            let eff = sse as f64 + mlam * block_rate_bits(&cf, &SCAN_8X8);
            if eff < eff8 {
                eff8 = eff;
            }
        }
        // best cost for four 4x4 (DC-pred / nd; current recon, decision-only)
        let mut eff4_sum = mlam * 2.0; // PARTITION_SPLIT symbol allowance
        for (sx, sy) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)] {
            let (bx, by) = (px + sx, py + sy);
            let mut best = f64::INFINITY;
            for &m in modes {
                let mut pred = [0i32; 16];
                if m == DC_PRED {
                    let d = dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    intra_predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        false,
                        false,
                        self.w,
                        self.h,
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 16];
                for ry in 0..4 {
                    let srow = &self.src[0][(by + ry) * self.w + bx..];
                    for rx in 0..4 {
                        resid[ry * 4 + rx] = srow[rx] - pred[ry * 4 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_4x4_t(&resid, &self.quant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_4X4, lam);
                let rr = idct_dequant_4x4(&cf, &self.quant);
                let mut sse = 0i64;
                for ry in 0..4 {
                    let srow = &self.src[0][(by + ry) * self.w + bx..];
                    for rx in 0..4 {
                        let r = (pred[ry * 4 + rx] + rr[ry * 4 + rx]).clamp(0, maxv);
                        let d = srow[rx] - r;
                        sse += (d * d) as i64;
                    }
                }
                // +mode/skip signalling allowance per 4x4 sub-block
                let eff = sse as f64 + mlam * (block_rate_bits(&cf, &SCAN_4X4) + 4.0);
                if eff < best {
                    best = eff;
                }
            }
            eff4_sum += best;
        }
        eff8 <= eff4_sum
    }

    /// Mean and variance of a `w`x`h` luma source region at pixel origin
    /// (px, py). libaom's partition search uses exactly these per-candidate
    /// variance features (`block_var`, `horz_block_var[2]`, `sub_block_var[4]`)
    /// to steer and prune the decision before paying for full R-D.
    fn luma_variance(&self, px: usize, py: usize, w: usize, h: usize) -> f64 {
        let mut sum = 0i64;
        let mut sqsum = 0i64;
        for ry in 0..h {
            let row = &self.src[0][(py + ry) * self.w + px..];
            for &s in &row[..w] {
                sum += s as i64;
                sqsum += (s as i64) * (s as i64);
            }
        }
        let n = (w * h) as f64;
        let mean = sum as f64 / n;
        (sqsum as f64 / n) - mean * mean
    }

    /// Full square+rectangular partition decision for a 16x16 luma region.
    /// Mirrors libaom's `rd_pick_partition` strategy: compute variance features
    /// per candidate, use them to PRUNE (skip full R-D on a candidate that the
    /// features say can't win), then compare the surviving candidates by the same
    /// SSE + mlam*bits objective the emitter minimises. HORZ is only offered for
    /// 4:4:4 today (the only format whose 16x8 chroma path is implemented).
    fn partition_choice_16(&self, x8: usize, y8: usize) -> Part16 {
        if self.mono {
            return Part16::Split; // monochrome codes 8x8 luma blocks only
        }
        if self.ss422 {
            return Part16::Split;
        }
        if self.block_luma_range(x8, y8, 16) < LF_BAND_SMOOTH_RANGE {
            return Part16::Split;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        let acq = self.quant.ac_q() as f64;
        let part_lam = mode_lambda() * acq * acq * self.perceptual_rd_scale(px, py, 16);

        let horz_on = !self.ss420
            && !self.ss422
            && HORZ_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
            // Q-adaptive gate (libaom strategy): rectangular partitions deliver
            // their gain at lower bitrates. At high quality the DC-only 16x8
            // sub-blocks lose to the square path's full mode search, so only
            // offer HORZ once the quantiser is coarse enough (ac_q ~> q60).
            && self.quant.ac_q() >= AC_Q_HORZ_MIN;

        // --- libaom-style variance features ---
        // block_var: whole 16x16. horz halves: two 16x8. These tell us whether
        // splitting horizontally actually reduces intra-block variance (i.e.
        // whether the content is horizontally banded). If the two horizontal
        // halves are each nearly as varied as the whole block, HORZ buys nothing
        // and is pruned without spending R-D on it.
        let block_var = self.luma_variance(px, py, 16, 16);
        let mut prune_horz = !horz_on;
        if horz_on {
            let vh0 = self.luma_variance(px, py, 16, 8);
            let vh1 = self.luma_variance(px, py + 8, 16, 8);
            let mean_half = 0.5 * (vh0 + vh1);
            // If the halves don't reduce variance vs the whole block (ratio close
            // to 1), the rows are not horizontally separable -> prune HORZ. The
            // 0.92 threshold mirrors libaom's "rectangular split only helps when
            // the directional variance drops meaningfully" heuristic.
            if block_var <= 1.0 || mean_half >= 0.85 * block_var {
                prune_horz = true;
            }
        }

        // --- R-D on the surviving candidates ---
        let rd_none = self.rd_cost_square(px, py, 16, false, false);
        let mut rd_split = part_lam * SPLIT_SIGNAL_BITS;
        for (sx, sy) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)] {
            rd_split += self.rd_cost_square(px + sx, py + sy, 8, false, false);
        }
        let rd_horz = if prune_horz {
            f64::INFINITY
        } else {
            self.rd_cost_horz(px, py)
        };

        // Pick the minimum of whatever survived pruning.
        if rd_none <= rd_split && rd_none <= rd_horz {
            Part16::None
        } else if rd_horz < rd_none && rd_horz <= rd_split {
            Part16::Horz
        } else {
            Part16::Split
        }
    }

    /// Code a 16x16 region (4:4:4 only) as a single TX_16X16 block: luma +
    /// chroma DC prediction, forward DCT16 + quant, the TX_16X16 coefficient
    /// coder, and reconstruction via the exact integer inverse. Updates the
    /// 4-unit (16-sample) skip / coef neighbour-context footprint.
    /// Set the RDO effort level (0 = full, >= 1 = fast). Returns `self` for
    /// builder-style chaining at tile construction.
    fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    /// Turn on variance-based adaptive quantization for this tile. `ref_act` is
    /// the tile mean activity (see [`tile_ref_activity`]); the running
    /// `CurrentQIndex` starts at the frame base. Must be paired with a frame
    /// header that signals `delta_q_present = 1` and the matching `delta_q_res`.
    fn enable_aq(&mut self, base_q: u8, ref_act: f32, vb: &VarianceBoost) {
        self.aq = AqCtx {
            enabled: true,
            base_q,
            res_log2: AQ_RES_LOG2,
            cur_qidx: base_q as i32,
            ref_act,
            read_deltas: false,
            pending: 0,
            vb_enabled: vb.enabled,
            vb_octile: vb.octile.clamp(1, 8),
            vb_strength: vb.strength.max(0.0),
            vb_boost_only: vb.boost_only,
        };
    }

    /// Begin a superblock at luma pixel `(sb_x, sb_y)`: pick the SB's quantizer
    /// from its local activity, retarget `self.quant`/`self.cquant`, and arm the
    /// `read_delta_qindex` token that the SB's first coded block will emit. No-op
    /// when AQ is disabled, so the non-AQ path is byte-identical.
    /// Emit `read_lr` for one 64x64 luma restoration unit at the start of a
    /// superblock (spec 5.11.57/58). With `LoopRestorationSize = 64` there is
    /// exactly one unit per SB whose top-left is the SB origin. When `self.wiener`
    /// is `None` nothing is emitted (RESTORE_NONE planes have no LR syntax).
    /// Emit `read_lr` for the restoration units that start in the superblock at
    /// **tile-local** pixel `(sb_x, sb_y)` (spec 5.11.57). Loop-restoration units
    /// are frame-relative, so the superblock position and unit counts are
    /// computed in frame coordinates (`frame_x0/y0`, `frame_w/h`). With
    /// `LoopRestorationSize = 64` most superblocks contain exactly one luma unit;
    /// small partial edge superblocks merge into the previous unit (the unit
    /// count uses `(frame+unit/2)/unit`) and emit zero units. Because units are
    /// >= the 64x64 superblock size and aligned to the frame grid, each unit's
    /// top-left superblock lies in exactly one tile, so every unit is signalled
    /// exactly once across all tiles — no tile-boundary special-casing needed.
    /// Emits nothing when `self.wiener` is `None`.
    fn emit_lr_sb(&mut self, sb_x: usize, sb_y: usize) {
        let Some(unit) = self.wiener else {
            return;
        };
        const UNIT: usize = 64;
        const MI: usize = 4;
        let count_units = |frame: usize| -> usize { (1).max((frame + (UNIT >> 1)) / UNIT) };
        let unit_rows = count_units(self.frame_h);
        let unit_cols = count_units(self.frame_w);
        // Frame-absolute superblock position in 4x4 MI units (luma).
        let r = (self.frame_y0 + sb_y) / MI;
        let c = (self.frame_x0 + sb_x) / MI;
        let sb_mi = UNIT / MI; // 16
        let urs = (r * MI + UNIT - 1) / UNIT;
        let ure = unit_rows.min(((r + sb_mi) * MI + UNIT - 1) / UNIT);
        let ucs = (c * MI + UNIT - 1) / UNIT;
        let uce = unit_cols.min(((c + sb_mi) * MI + UNIT - 1) / UNIT);
        for _ur in urs..ure {
            for _uc in ucs..uce {
                self.emit_lr_unit(&unit);
            }
        }
    }

    /// Emit one `read_lr_unit` for a RESTORE_WIENER luma unit (spec 5.11.58).
    fn emit_lr_unit(&mut self, unit: &crate::wiener::WienerUnit) {
        use crate::wiener::{WIENER_TAPS_K, WIENER_TAPS_MAX, WIENER_TAPS_MIN};
        // use_wiener: always 1 (we filter every unit with the global filter).
        self.enc.encode_symbol(1, &mut self.cdfs.wiener_restore);
        // read_wiener_filter: vertical taps (pass 0) then horizontal (pass 1),
        // each signed-subexp coded with per-tap k against the running reference,
        // which then updates to the coded value.
        for axis in 0..2 {
            let (taps, refs) = if axis == 0 {
                (unit.v, &mut self.lr_ref_v)
            } else {
                (unit.h, &mut self.lr_ref_h)
            };
            for j in 0..3usize {
                let lo = WIENER_TAPS_MIN[j];
                let hi = WIENER_TAPS_MAX[j] + 1; // exclusive high
                let k = WIENER_TAPS_K[j] as u32;
                self.enc
                    .encode_signed_subexp_with_ref(taps[j], lo, hi, k, refs[j]);
                refs[j] = taps[j];
            }
        }
    }

    fn aq_begin_sb(&mut self, sb_x: usize, sb_y: usize) {
        if !self.aq.enabled {
            return;
        }
        let target = if self.aq.vb_enabled {
            // Variance Boost: pick the representative 8x8 variance at the configured
            // octile and map it to a qindex delta off the frame base. ref_act (the
            // tile mean whole-SB log-variance) anchors the coarse side, matching
            // `av2/aq.rs::AqState::per_sb`.
            let mut subvars = [0f32; 64];
            let filled = aq_sb_subblock_variances(
                &self.src[0],
                self.w,
                sb_y,
                sb_x,
                self.w,
                self.h,
                &mut subvars,
            );
            if filled == 0 {
                self.aq.base_q as i32
            } else {
                let picked = aq_sb_octile_variance(&mut subvars, self.aq.vb_octile);
                let delta = aq_variance_boost_delta(
                    picked,
                    self.aq.ref_act,
                    self.aq.vb_strength,
                    self.aq.vb_boost_only,
                );
                (self.aq.base_q as i32 + delta).clamp(1, 255)
            }
        } else {
            let act = sb_activity(&self.src[0], self.w, sb_y, sb_x, self.w, self.h);
            aq_target_qidx(self.aq.base_q as i32, act, self.aq.ref_act)
        };
        let step = 1i32 << self.aq.res_log2;
        let steps = (((target - self.aq.cur_qidx) as f32) / step as f32)
            .round()
            .clamp(-(AQ_MAX_STEPS as f32), AQ_MAX_STEPS as f32) as i32;
        // The decoder applies Clip3(1,255, cur + steps*step); mirror it exactly so
        // both sides agree on the new qindex even when the clamp bites.
        let newq = (self.aq.cur_qidx + steps * step).clamp(1, 255);
        self.aq.cur_qidx = newq;
        self.aq.pending = steps;
        self.aq.read_deltas = true;
        self.quant = Quant::new(newq as u8, self.bd);
        // The chroma-DC delta is a frame-level constant (DeltaQUDc, derived from
        // the frame base_q_idx and signalled once in the header). The decoder
        // forms the chroma-DC qindex as CurrentQIndex + DeltaQUDc, so apply the
        // frame-level delta to the AQ-adjusted qindex — not chroma_dc_delta(newq).
        let frame_dc_delta = chroma_dc_delta(self.aq.base_q);
        self.cquant = Quant::new_chroma_with_delta(newq as u8, frame_dc_delta, self.bd);
    }

    /// Emit the `read_delta_qindex()` token for the first block of a superblock,
    /// if armed (spec `ReadDeltas`). Codes the magnitude with the adaptive
    /// `delta_q` CDF, the `DELTA_Q_SMALL` literal escape for magnitudes >= 3, and
    /// the equiprobable sign bit. Called immediately after the block-skip symbol,
    /// matching `intra_frame_mode_info()` ordering (no in-block segment/cdef token
    /// precedes it here). No-op when AQ is off or already emitted for this SB.
    fn code_delta_q_if_armed(&mut self) {
        if !self.aq.read_deltas {
            return;
        }
        self.aq.read_deltas = false;
        let m = self.aq.pending.unsigned_abs() as i32;
        const DELTA_Q_SMALL: usize = 3;
        if m < DELTA_Q_SMALL as i32 {
            self.enc.encode_symbol(m as usize, &mut self.cdfs.delta_q);
        } else {
            self.enc
                .encode_symbol(DELTA_Q_SMALL, &mut self.cdfs.delta_q);
            // m = abs_bits + (1 << rem) + 1, rem = floor(log2(m-1)) >= 1.
            let v = (m - 1) as u32;
            let rem = 31 - v.leading_zeros(); // floor(log2(v))
            let abs_bits = v - (1 << rem);
            self.enc.encode_literal(rem - 1, 3); // delta_q_rem_bits (pre-increment)
            self.enc.encode_literal(abs_bits, rem); // delta_q_abs_bits
        }
        if m != 0 {
            self.enc.encode_literal((self.aq.pending < 0) as u32, 1); // sign
        }
    }

    /// Variance-weighted ("SSIM-style") RD scale for a luma block whose top-left
    /// source pixel is `(px, py)` and whose side is `dim`. The returned factor
    /// multiplies the block's rate weights — `lam` (coefficient trellis / RDOQ)
    /// and `mlam` (mode & partition search) — so a value above 1 spends fewer
    /// bits on the block and below 1 spends more. The block activity is
    /// `ln(1 + variance)` of the source luma (same measure as [`sb_activity`]),
    /// centred on the tile mean `aq.ref_act`, then mapped through
    /// `exp(K * (act - ref))` and clamped (see [`prdo_k`] / [`prdo_clamp`]).
    ///
    /// Returns 1.0 (no change) when `PRDO_K == 0` or when no activity reference
    /// is available — the AQ pre-pass populates `aq.ref_act`; without it there is
    /// nothing to centre on, so the adjustment is skipped.
    fn perceptual_rd_scale(&self, px: usize, py: usize, dim: usize) -> f64 {
        let k = prdo_k();
        if k == 0.0 {
            return 1.0;
        }
        let refa = self.aq.ref_act;
        if refa <= 0.0 {
            return 1.0;
        }
        let bw = dim.min(self.w.saturating_sub(px));
        let bh = dim.min(self.h.saturating_sub(py));
        if bw == 0 || bh == 0 {
            return 1.0;
        }
        let yp = &self.src[0];
        let (mut sum, mut sum2) = (0i64, 0i64);
        for r in 0..bh {
            let base = (py + r) * self.w + px;
            for &c in &yp[base..base + bw] {
                let v = c as i64;
                sum += v;
                sum2 += v * v;
            }
        }
        let n = (bw * bh) as f64;
        let mean = sum as f64 / n;
        let var = (sum2 as f64 / n - mean * mean).max(0.0);
        let act = (1.0 + var).ln() as f32;
        let c = prdo_clamp();
        ((k * (act - refa) as f64).exp()).clamp(1.0 / c, c)
    }

    /// True luma R-D cost of coding one square `dim`×`dim` region at `(px,py)`
    /// as a single transform block, in the same units the per-block mode search
    /// minimises (`SSE + mlam * (coeff_bits + mode_signal_bits)`). Measures only:
    /// it predicts from the current `self.recon`, runs a compact intra-mode
    /// search (DCT_DCT residual, the same candidate list the emitter uses), and
    /// reconstructs through the exact inverse to score real distortion. It does
    /// NOT emit to the bitstream or mutate any encoder state, so it is safe to
    /// call speculatively while deciding a partition. Chroma is excluded — the
    /// luma residual dominates the partition choice and including chroma would
    /// double the cost of the search for little decision benefit.
    fn rd_cost_square(&self, px: usize, py: usize, dim: usize, have_tr: bool, have_bl: bool) -> f64 {
        let acq = self.quant.ac_q() as f64;
        let dcq = self.quant.dc_q() as f64;
        let lam = trellis_lambda();
        let mlam = mode_lambda() * acq * acq;
        let prdo = self.perceptual_rd_scale(px, py, dim);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // Compact candidate set: DC plus the two non-directional modes that win
        // most often on photographic content. (The full 13-mode search is what
        // the emitter runs; for a partition *decision* this proxy tracks it
        // closely while staying cheap.)
        let modes: &[usize] = &[DC_PRED, SMOOTH_PRED, PAETH_PRED];
        let mut best = f64::INFINITY;
        match dim {
            8 => {
                let scan = &SCAN_8X8;
                for &m in modes {
                    let mut pred = [0i32; 64];
                    if m == DC_PRED {
                        let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 64];
                    } else {
                        intra_predict_nd(
                            m, &self.recon[0], self.w, px, py, 8, 8, have_tr, have_bl, self.w,
                            self.h, &mut pred, self.bd,
                        );
                    }
                    let mut resid = [0i32; 64];
                    for (ry, (rrow, prow)) in resid
                        .as_chunks_mut::<8>()
                        .0
                        .iter_mut()
                        .zip(pred.as_chunks::<8>().0.iter())
                        .enumerate()
                    {
                        let srow = &self.src[0][(py + ry) * self.w + px..];
                        for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                            *r = s - p;
                        }
                    }
                    let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                    let bits = block_rate_bits(&cf, scan) + mode_signal_bits(m);
                    let cost = sse as f64 + mlam * bits;
                    if cost < best {
                        best = cost;
                    }
                }
            }
            16 => {
                let scan = &SCAN_16X16;
                for &m in modes {
                    let mut pred = [0i32; 256];
                    if m == DC_PRED {
                        let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 256];
                    } else {
                        intra_predict_nd(
                            m, &self.recon[0], self.w, px, py, 16, 16, have_tr, have_bl, self.w,
                            self.h, &mut pred, self.bd,
                        );
                    }
                    let mut resid = [0i32; 256];
                    for (ry, (rrow, prow)) in resid
                        .as_chunks_mut::<16>()
                        .0
                        .iter_mut()
                        .zip(pred.as_chunks::<16>().0.iter())
                        .enumerate()
                    {
                        let srow = &self.src[0][(py + ry) * self.w + px..];
                        for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                            *r = s - p;
                        }
                    }
                    let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_16x16(&cf, &self.quant);
                    let sse =
                        sse_recon::<256, 16>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                    let bits = block_rate_bits(&cf, scan) + mode_signal_bits(m);
                    let cost = sse as f64 + mlam * bits;
                    if cost < best {
                        best = cost;
                    }
                }
            }
            32 => {
                let scan = &SCAN_32X32;
                for &m in modes {
                    let mut pred = [0i32; 1024];
                    if m == DC_PRED {
                        let d = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 1024];
                    } else {
                        intra_predict_nd(
                            m, &self.recon[0], self.w, px, py, 32, 32, have_tr, have_bl, self.w,
                            self.h, &mut pred, self.bd,
                        );
                    }
                    let mut resid = [0i32; 1024];
                    for (ry, (rrow, prow)) in resid
                        .as_chunks_mut::<32>()
                        .0
                        .iter_mut()
                        .zip(pred.as_chunks::<32>().0.iter())
                        .enumerate()
                    {
                        let srow = &self.src[0][(py + ry) * self.w + px..];
                        for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                            *r = s - p;
                        }
                    }
                    let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_32x32(&cf, &self.quant);
                    let sse =
                        sse_recon::<1024, 32>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                    let bits = block_rate_bits(&cf, scan) + mode_signal_bits(m);
                    let cost = sse as f64 + mlam * bits;
                    if cost < best {
                        best = cost;
                    }
                }
            }
            _ => unreachable!("rd_cost_square dim {}", dim),
        }
        best
    }

    /// R-D estimate for coding a 16x16 luma region as PARTITION_H: two stacked
    /// 16x8 sub-blocks, each DC-predicted + DCT and trellis-quantized through the
    /// exact inverse (matching what `code_block16_horz_444` emits). Returns
    /// SSE + mlam*bits summed over both halves plus the HORZ partition signal.
    fn rd_cost_horz(&self, px: usize, py: usize) -> f64 {
        let acq = self.quant.ac_q() as f64;
        let dcq = self.quant.dc_q() as f64;
        let lam = trellis_lambda();
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let (lam, mlam) = (lam * prdo, mode_lambda() * acq * acq * prdo);
        let maxv = (1 << self.bd) - 1;
        let mut total = mlam * SPLIT_SIGNAL_BITS; // HORZ costs a partition symbol like SPLIT
        for half in 0..2 {
            let sy = py + half * 8;
            let dc = dc_pred_16x8(&self.recon[0], self.w, px, sy, self.bd as i32);
            let mut resid = [0i32; 128];
            for ry in 0..8 {
                let srow = &self.src[0][(sy + ry) * self.w + px..];
                for cx in 0..16 {
                    resid[ry * 16 + cx] = srow[cx] - dc;
                }
            }
            let (mut cf, tf) = dct16x8_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_16X8, lam);
            let rr = idct_dequant_16x8(&cf, &self.quant);
            let mut sse = 0i64;
            for ry in 0..8 {
                let srow = &self.src[0][(sy + ry) * self.w + px..];
                for cx in 0..16 {
                    let r = (dc + rr[ry * 16 + cx]).clamp(0, maxv);
                    let d = (srow[cx] - r) as i64;
                    sse += d * d;
                }
            }
            let bits = block_rate_bits(&cf, &SCAN_16X8);
            total += sse as f64 + mlam * bits;
        }
        total
    }

    /// Code a 16x16 luma region as PARTITION_H: two stacked 16x8 sub-blocks.
    /// Minimal first implementation: 4:4:4 only, DC prediction, DCT_DCT, no CfL /
    /// SMOOTH_V / mode search. Each 16x8 sub-block is a full intra block (own skip,
    /// y_mode, uv_mode, luma RTX_16X8 coeffs + chroma RTX_16X8 coeffs). Mirrors the
    /// decoder's two `decode_b(PARTITION_H)` calls.
    fn code_block16_horz_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f64, self.quant.ac_q() as f64);
        let (cdcq, cacq) = (self.cquant.dc_q() as f64, self.cquant.ac_q() as f64);
        // Two sub-blocks: half = 0 (top, py), half = 1 (bottom, py+8).
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4); // luma 4-unit coords
            // --- luma 16x8: DC predict, residual, forward, trellis, dc-snap ---
            let lpred = dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let mut lresid = [0i32; 128];
            for ry in 0..8 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for cx in 0..16 {
                    lresid[ry * 16 + cx] = srow[cx] - lpred;
                }
            }
            let (mut lcf, ltf) = dct16x8_t(&lresid, &self.quant);
            trellis_optimize(&mut lcf, &ltf, dcq, acq, &SCAN_16X8, lam);
            let mean_l = lresid.iter().sum::<i32>() / 128;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            // --- chroma 16x8 (4:4:4): DC predict each plane ---
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_16x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                for ry in 0..8 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    for cx in 0..16 {
                        resid[ry * 16 + cx] = srow[cx] - dc;
                    }
                }
                let (mut q, qt) = dct16x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_16X8, lam);
                let mean_c = resid.iter().sum::<i32>() / 128;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            // block_skip iff all planes have no coefficients.
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            // --- header: skip, delta-q (once), y_mode (DC), uv_mode (DC) ---
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.enc
                .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
            self.code_delta_q_if_armed();
            // record the 16x8 footprint for the deblock filter: width 4 units,
            // height 2 units (vertical edges every 16, horizontal every 8).
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            self.emit_uv_mode(DC_PRED, DC_PRED, None);
            // footprint update: skip/mode over 4 wide x 2 tall units.
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + 2].fill(DC_PRED as u8);
            // --- luma coeffs (RTX_16X8) ---
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            // reconstruct luma
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                for cx in 0..16 {
                    drow[cx] = (lpred + lrr[ry * 16 + cx]).clamp(0, maxval);
                }
            }
            // --- chroma coeffs + reconstruct (4:4:4, both planes RTX_16X8) ---
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_16x8_chroma(plane, bx4, by4);
                    let ds = self.dc_sign_ctx_16x8_chroma(plane, bx4, by4);
                    encode_16x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                };
                self.a_coef[plane][bx4..bx4 + 4].fill(cres_ctx);
                self.l_coef[plane][by4..by4 + 2].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 128]
                } else {
                    idct_dequant_16x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    for cx in 0..16 {
                        drow[cx] = (cpred[ci] + rr[ry * 16 + cx]).clamp(0, maxval);
                    }
                }
            }
        }
    }

    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 4);
        let (px, py) = (x8 * 8, y8 * 8);
        // luma 16x16 (identical for all subsampling modes)
        // Luma 16x16: same non-directional intra mode search as the 8x8 path.
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let dcs16 = self.dc_sign_ctx_16(0, px / 4, py / 4);
        let mlam = mode_lambda() * acq * acq;
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let mut best_mode = DC_PRED;
        let mut txtp16: u8 = 0; // 0=DCT_DCT 1=ADST_ADST 2=ADST_DCT 3=DCT_ADST
        let mut lpred_arr = [0i32; 256];
        let mut lcf = [0i32; 256];
        let mut best_eff = f64::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f64;
        let mut ltf = [0f64; 256]; // winner transform coeffs (f64, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        for &m in modes {
            let mut pred = [0i32; 256];
            if m == DC_PRED {
                let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 256];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 256];
            for (ry, (rrow, prow)) in resid
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(pred.as_chunks::<16>().0.iter())
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let blk_sse16 = |rr: &[i32; 256]| -> i64 {
                let mut sse = 0i64;
                for (ry, (prow, rrow)) in pred
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(rr.as_chunks::<16>().0.iter())
                    .enumerate()
                {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        sse += (d * d) as i64;
                    }
                }
                sse
            };
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    &self.cdfs,
                    2,
                    0,
                    &self.cdfs.eob_bin_256_l,
                    dcs16,
                );
            }
            let sse = blk_sse16(&idct_dequant_16x16(&cf, &self.quant));
            let bits = block_rate_bits(&cf, &SCAN_16X16);
            let cost = sse as f64 + mlam * (bits + mode_signal_bits(m));
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
            }
        }
        // Angle-delta winner refinement (see code_block: diagonals only, -3..=3).
        let mut best_delta: i32 = 0;
        if angle_delta_enabled()
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let ad_cdf = self.cdfs.angle_delta[best_mode - V_PRED].clone();
            let mut best_ad_cost =
                best_dct_sse as f64 + mlam * (best_dct_bits + cdf_cost(&ad_cdf, 3));
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut pred = [0i32; 256];
                intra_predict_nd_ad(
                    best_mode,
                    d,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 256];
                for ry in 0..16 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..16 {
                        resid[ry * 16 + rx] = srow[rx] - pred[ry * 16 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        &self.cdfs,
                        2,
                        0,
                        &self.cdfs.eob_bin_256_l,
                        dcs16,
                    );
                }
                let rr = idct_dequant_16x16(&cf, &self.quant);
                let mut sse = 0i64;
                for ry in 0..16 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..16 {
                        let r =
                            (pred[ry * 16 + rx] + rr[ry * 16 + rx]).clamp(0, (1 << self.bd) - 1);
                        let dd = srow[rx] - r;
                        sse += (dd * dd) as i64;
                    }
                }
                let bits = block_rate_bits(&cf, &SCAN_16X16);
                let cost = sse as f64 + mlam * (bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    lpred_arr = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                }
            }
        }
        // Fast path: run RDOQ once, on the winning mode only (libaom
        // winner-mode coeff opt). The decision above used un-trellised costs.
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                &self.cdfs,
                2,
                0,
                &self.cdfs.eob_bin_256_l,
                dcs16,
            );
        }
        // Winner-only ADST_ADST refinement. Full and Medium try it; only Fast
        // prunes the transform-type search to DCT_DCT (libaom-style).
        if self.speed.try_adst() {
            let mut resid = [0i32; 256];
            for (ry, rrow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut acf, atf) = adst16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                &self.cdfs,
                2,
                0,
                &self.cdfs.eob_bin_256_l,
                dcs16,
            );
            let rr = iadst_dequant_16x16(&acf, &self.quant);
            let mut asse = 0i64;
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    asse += (d * d) as i64;
                }
            }
            let abits = block_rate_bits(&acf, &SCAN_16X16);
            // Quality guard: only accept ADST if it does not meaningfully worsen
            // SSE. At low quality lambda (~quantizer^2) is enormous, so a pure
            // RD test would pick ADST whenever it shaves a few bits even while
            // inflating distortion ~2x; that tanks perceptual quality (SSIMULACRA2)
            // for a trivial rate gain. Requiring SSE-non-worsening keeps the
            // genuine high-quality ADST wins (where it lowers SSE) and blocks the
            // low-quality "trade quality for bits" pathology.
            if asse <= best_dct_sse + (best_dct_sse >> 5)
                && asse as f64 + mlam * abits < best_dct_sse as f64 + mlam * best_dct_bits
            {
                lcf = acf;
                txtp16 = 1;
            }
        }
        // Asymmetric-ADST refinement (ADST_DCT / DCT_ADST) for TX_16X16, same
        // rationale as the 8x8 path. Competes with the running tx winner.
        if self.speed.try_adst() && asym_adst_enabled() {
            let mut best_txtp16_sse = if txtp16 == 1 { i64::MAX } else { best_dct_sse };
            let mut best_txtp16_bits = best_dct_bits;
            if txtp16 == 1 {
                // recompute the ADST_ADST winner cost as the bar to beat
                let rr = iadst_dequant_16x16(&lcf, &self.quant);
                let mut s = 0i64;
                for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                    for ((&p, &rv), &sv) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = sv - r;
                        s += (d * d) as i64;
                    }
                }
                best_txtp16_sse = s;
                best_txtp16_bits = block_rate_bits(&lcf, &SCAN_16X16);
            }
            for (fwd_dctadst, inv_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 256];
                for (ry, rrow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                    for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                        *r = s - p;
                    }
                }
                let (mut acf, atf) = if fwd_dctadst {
                    dctadst16x16_t(&resid, &self.quant)
                } else {
                    adstdct16x16_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    &self.cdfs,
                    2,
                    0,
                    &self.cdfs.eob_bin_256_l,
                    dcs16,
                );
                let rr = if inv_dctadst {
                    idctadst_dequant_16x16(&acf, &self.quant)
                } else {
                    iadstdct_dequant_16x16(&acf, &self.quant)
                };
                let mut asse = 0i64;
                for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        asse += (d * d) as i64;
                    }
                }
                let abits = block_rate_bits(&acf, &SCAN_16X16);
                if asse <= best_dct_sse + (best_dct_sse >> 5)
                    && asse as f64 + mlam * abits < best_txtp16_sse as f64 + mlam * best_txtp16_bits
                {
                    lcf = acf;
                    txtp16 = if inv_dctadst { 3 } else { 2 };
                    best_txtp16_sse = asse;
                    best_txtp16_bits = abits;
                }
            }
        }
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block16_420(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
            );
        } else if self.ss422 {
            self.code_block16_422(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
            );
        } else {
            self.code_block16_444(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
            );
        }
    }

    /// Shared header + luma for a TX_16X16 block: codes the block-level skip
    /// flag, `DC_PRED` y/uv modes, the luma TX_16X16 coefficients, updates the
    /// 4-unit (16-sample) luma skip/coef footprint, and reconstructs luma. The
    /// caller has already decided `block_skip` (needs all planes) and passes the
    /// luma coefficients + DC prediction.
    /// Emit the chroma `uv_mode` symbol: plain DC (`None`) or CfL (`Some(alphas)`),
    /// in which case also the joint-sign and per-plane magnitude symbols.
    fn emit_uv_mode(&mut self, y_mode: usize, uv_mode: usize, cfl: Option<[i32; 2]>) {
        match cfl {
            Some(a) => {
                let su = if a[0] == 0 {
                    0
                } else if a[0] < 0 {
                    1
                } else {
                    2
                };
                let sv = if a[1] == 0 {
                    0
                } else if a[1] < 0 {
                    1
                } else {
                    2
                };
                let sign = su * 3 + sv;
                if sign == 0 {
                    // Both alphas zero: zero-alpha CfL reconstructs exactly as DC
                    // prediction, so signal plain DC and avoid an invalid
                    // `sign - 1` joint-sign symbol.
                    self.enc
                        .encode_symbol(DC_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
                    return;
                }
                self.enc
                    .encode_symbol(CFL_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
                self.enc.encode_symbol(sign - 1, &mut self.cdfs.cfl_sign);
                if su != 0 {
                    let c = (su == 2) as usize * 3 + sv;
                    self.enc
                        .encode_symbol((a[0].abs() - 1) as usize, &mut self.cdfs.cfl_alpha[c]);
                }
                if sv != 0 {
                    let c = (sv == 2) as usize * 3 + su;
                    self.enc
                        .encode_symbol((a[1].abs() - 1) as usize, &mut self.cdfs.cfl_alpha[c]);
                }
            }
            None => {
                self.enc
                    .encode_symbol(uv_mode, &mut self.cdfs.uv_mode[13 + y_mode]);
                // Directional chroma modes (here only V_PRED/H_PRED, used at 8x8
                // 4:4:4 chroma where `use_angle_delta` holds) emit a chroma
                // angle_delta symbol. The encoder only offers delta 0, so emit the
                // centre bucket (delta + 3 == 3).
                if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
                    self.enc
                        .encode_symbol(3, &mut self.cdfs.angle_delta[uv_mode - V_PRED]);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_header_luma16(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        txtp16: u8,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        // AV1 read_delta_qindex(): first block of the SB emits the per-SB
        // delta-q token here (after skip, before the luma mode).
        self.code_delta_q_if_armed();
        self.mark_skip8(x8, y8, 2, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc.encode_symbol(
                (angle_delta + 3) as usize,
                &mut self.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        self.emit_uv_mode(y_mode, uv_mode, cfl);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 4].fill(sv);
        self.l_skip[by4..by4 + 4].fill(sv);
        self.a_mode[bx4..bx4 + 4].fill(mv);
        self.l_mode[by4..by4 + 4].fill(mv);
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_16(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_16(0, bx4, by4);
            encode_tx16_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                lcf,
                false,
                sk,
                ds,
                y_mode,
                match txtp16 {
                    1 => ADST_ADST_TX16_IDX,
                    2 => ADST_DCT_TX16_IDX,
                    3 => DCT_ADST_TX16_IDX,
                    _ => 1,
                },
            )
        };
        self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
        self.l_coef[0][by4..by4 + 4].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 256]
        } else {
            match txtp16 {
                1 => iadst_dequant_16x16(lcf, &self.quant),
                2 => iadstdct_dequant_16x16(lcf, &self.quant),
                3 => idctadst_dequant_16x16(lcf, &self.quant),
                _ => idct_dequant_16x16(lcf, &self.quant),
            }
        };
        for (ry, (prow, rrow)) in lpred
            .as_chunks::<16>()
            .0
            .iter()
            .zip(lrr.as_chunks::<16>().0.iter())
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block16_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        txtp16: u8,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let mut ccf = [[0i32; 256]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_16x16(&self.recon[plane], self.w, px, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 256];
            for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(py + ry) * self.w + px..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                &SCAN_16X16,
                trellis_lambda(),
            );
            let mean_resid_dc = resid.iter().sum::<i32>() / 256;
            if ccf[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // 4:4:4 CfL for the 16x16 chroma blocks (mirrors the 8x8 path).
        let mut cpred16 = [[0i32; 256]; 2];
        let mut cfl_opt: Option<[i32; 2]> = None;
        {
            let lrr_cfl = match txtp16 {
                1 => iadst_dequant_16x16(lcf, &self.quant),
                2 => iadstdct_dequant_16x16(lcf, &self.quant),
                3 => idctadst_dequant_16x16(lcf, &self.quant),
                _ => idct_dequant_16x16(lcf, &self.quant),
            };
            let mut luma_rec = [0i32; 256];
            for i in 0..256 {
                luma_rec[i] = (lpred[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 256];
            cfl_ac_444(&luma_rec, 16, 16, &mut ac);
            let (dcq, acq, lam) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let mlam = mode_lambda() * acq * acq;
            let mut cfl_ccf = [[0i32; 256]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f64; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f64; 2]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 256];
                for (ry, drow) in src.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..16]);
                }
                let dcrr = idct_dequant_16x16(&ccf[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..256 {
                    let r = (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s += (d * d) as i64;
                }
                dc_sse[ci] = s;
                dc_bits[ci] = block_rate_bits(&ccf[ci], &SCAN_16X16);
                let a = cfl_best_alpha(&ac, &src, dc, 256, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 256];
                let mut resid = [0i32; 256];
                for i in 0..256 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X16, lam);
                let rr = idct_dequant_16x16(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..256 {
                    let r = (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_sse[ci] = s2;
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_16X16);
                cpred16[ci] = cpr;
            }
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let dc_total = (dc_sse[0] + dc_sse[1]) as f64 + mlam * (dc_bits[0] + dc_bits[1]);
            let cfl_total =
                (cfl_sse[0] + cfl_sse[1]) as f64 + mlam * (cfl_bits[0] + cfl_bits[1] + sig);
            // Let the RD comparison decide DC-vs-CfL across the whole quality
            // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
            // where it helps most (high quality).
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                cfl_opt = Some(cfl_a);
                ccf[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred16[ci] = [cpred[ci]; 256];
                }
            }
        }
        // SMOOTH_V check: on smooth vertical gradients DC prediction rings at block
        // boundaries; SMOOTH_V interpolates from the top edge to the bottom-left corner,
        // giving near-zero residual on gradients and eliminating the green-lane stripes.
        let mut chosen_uv_16 = DC_PRED;
        // SMOOTH_V is only beneficial at low quality (ac_q > 300 ≈ quality ≤ 35).
        // At higher quality DC/CfL suffice and SMOOTH_V's <= tie-break causes
        // block-boundary colour mismatches across the whole image.
        if self.quant.ac_q() > 300 {
            let (dcq, acq, lam) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let mut sv_ccf16 = [[0i32; 256]; 2];
            let mut sv_preds16 = [[0i32; 256]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            for ci in 0..2 {
                let plane = ci + 1;
                for ry in 0..16 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &cpred16[ci][ry * 16..];
                    for (&srow, &prow) in srow[..16].iter().zip(prow[..16].iter()) {
                        let d = srow - prow;
                        sse_cur += (d * d) as i64;
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    false,
                    false,
                    self.w,
                    self.h,
                    &mut sv_preds16[ci],
                    self.bd,
                );
                let mut resid = [0i32; 256];
                for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds16[ci][ry * 16..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                // SMOOTH_V chroma derives ADST_DCT (dav1d_txtp_from_uvmode), so
                // the encoder must forward/inverse with ADST_DCT to match the
                // decoder. Using plain DCT desyncs at >8-bit.
                let (q, qt) = adstdct16x16_t(&resid, &self.cquant);
                sv_ccf16[ci] = q;
                trellis_optimize(&mut sv_ccf16[ci], &qt, dcq, acq, &SCAN_16X16, lam);
                let mean_resid_sv = resid.iter().sum::<i32>() / 256;
                if sv_ccf16[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    sv_ccf16[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
                for ry in 0..16 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds16[ci][ry * 16..];
                    for (&srow, &prow) in srow[..16].iter().zip(prow[..16].iter()) {
                        let d = srow - prow;
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv <= sse_cur {
                ccf[..2].copy_from_slice(&sv_ccf16[..2]);
                cpred16[..2].copy_from_slice(&sv_preds16[..2]);
                cfl_opt = None; // SMOOTH_V overrides CfL if it wins
                chosen_uv_16 = SMOOTH_V_PRED;
            }
        } // end if ac_q > 300 (SMOOTH_V)
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv_16,
            cfl_opt,
            txtp16,
            angle_delta,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_16(plane, bx4, by4);
                encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    0,
                    1,
                )
            };
            self.a_coef[plane][bx4..bx4 + 4].fill(res_ctx);
            self.l_coef[plane][by4..by4 + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else if chosen_uv_16 == SMOOTH_V_PRED {
                // ADST_DCT inverse to match the decoder's derived chroma txtp.
                iadstdct_dequant_16x16(&ccf[ci], &self.cquant)
            } else {
                idct_dequant_16x16(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if cfl_opt.is_some() || chosen_uv_16 == SMOOTH_V_PRED {
                    let prow = &cpred16[ci][ry * 16..];
                    for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                        *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                    }
                } else {
                    // Plain DC chroma: use the scalar predictor directly so recon
                    // never depends on the CfL block having populated `cpred16`.
                    for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                        *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block16_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        txtp16: u8,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f64,
            self.cquant.ac_q() as f64,
            trellis_lambda(),
        );
        let maxval = (1 << self.bd) - 1;
        // DC path
        let mut ccf_dc = [[0i32; 64]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 64];
            for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
            ccf_dc[ci] = q;
            trellis_optimize(&mut ccf_dc[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            let mean_resid_dc = resid.iter().sum::<i32>() / 64;
            if ccf_dc[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf_dc[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // SMOOTH_V: only at low quality (ac_q > 300 ≈ quality ≤ 35)
        let smooth_v_active = false; // chroma SMOOTH_V needs ADST_DCT (mode-derived tx for chroma); encoder only has DCT_DCT, so offering it desyncs dav1d. Disabled until ADST_DCT chroma is implemented.
        let mut ccf_sv = [[0i32; 64]; 2];
        let mut sv_preds = [[0i32; 64]; 2];
        if smooth_v_active {
            for ci in 0..2 {
                let plane = ci + 1;
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    8,
                    8,
                    false,
                    false,
                    self.cw,
                    self.h,
                    &mut sv_preds[ci],
                    self.bd,
                );
                let mut resid = [0i32; 64];
                for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds[ci][ry * 8..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                ccf_sv[ci] = q;
                trellis_optimize(&mut ccf_sv[ci], &qt, dcq, acq, &SCAN_8X8, lam);
                let mean_resid_sv = resid.iter().sum::<i32>() / 64;
                if ccf_sv[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    ccf_sv[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
            }
        } // end if smooth_v_active
        // RD: reconstruct both and compare SSE; cache inverse-transform for winner reuse
        let mut rr_dc = [[0i32; 64]; 2];
        let mut rr_sv = [[0i32; 64]; 2];
        let mut sse_dc = 0i64;
        let mut sse_sv = 0i64;
        for ci in 0..2 {
            let plane = ci + 1;
            rr_dc[ci] = idct_dequant_8x8(&ccf_dc[ci], &self.cquant);
            rr_sv[ci] = idct_dequant_8x8(&ccf_sv[ci], &self.cquant);
            let dc = dc_preds[ci];
            for (ry, (rd_row, rs_row)) in rr_dc[ci]
                .as_chunks::<8>()
                .0
                .iter()
                .zip(rr_sv[ci].as_chunks::<8>().0.iter())
                .enumerate()
            {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                let prow = &sv_preds[ci][ry * 8..];
                for (((&s, &prow), &rd), &rs) in srow[..8]
                    .iter()
                    .zip(prow[..8].iter())
                    .zip(rd_row[..8].iter())
                    .zip(rs_row[..8].iter())
                {
                    let d = s - (dc + rd).clamp(0, maxval);
                    let v = s - (prow + rs).clamp(0, maxval);
                    sse_dc += (d * d) as i64;
                    sse_sv += (v * v) as i64;
                }
            }
        }
        let use_sv = smooth_v_active && sse_sv <= sse_dc;
        let (chosen_uv, ccf, rr_cache) = if use_sv {
            (SMOOTH_V_PRED, ccf_sv, rr_sv)
        } else {
            (DC_PRED, ccf_dc, rr_dc)
        };
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv,
            None,
            txtp16,
            angle_delta,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx(plane, bx4c, by4c);
                encode_tx8_coeffs_adapt(&mut self.enc, &mut self.cdfs, &ccf[ci], true, sk, ds, 0, 1)
            };
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 2].fill(res_ctx);
            let rr = if block_skip { [0i32; 64] } else { rr_cache[ci] };
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                if use_sv {
                    let prow = &sv_preds[ci][ry * 8..];
                    for ((dv, &rv), &prow) in
                        drow[..8].iter_mut().zip(rrow.iter()).zip(prow[..8].iter())
                    {
                        *dv = (prow + rv).clamp(0, maxval);
                    }
                } else {
                    let dc = dc_preds[ci];
                    for (dv, &rv) in drow[..8].iter_mut().zip(rrow.iter()) {
                        *dv = (dc + rv).clamp(0, maxval);
                    }
                }
            }
        }
    }

    /// 4:2:2: a 16x16 luma region maps to an 8-wide x 16-tall chroma region per
    /// plane (`RTX_8X16`, coef-CDF class 2). Chroma is full-height, half-width, so
    /// the chroma block sits at `(cx, py)` with `cx = px/2` and spans 2 coef units
    /// horizontally and 4 vertically on the chroma grid.
    #[allow(clippy::too_many_arguments)]
    fn code_block16_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        txtp16: u8,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        let (bx4c, by4c) = (cx / 4, py / 4);
        let mut ccf = [[0i32; 128]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_8x16(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 128];
            for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_8x16_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                &SCAN_8X16,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            DC_PRED,
            None,
            txtp16,
            angle_delta,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_8x16_422(plane, bx4c, by4c);
                encode_8x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            // RTX_8X16: 2 coef-context units wide, 4 units tall.
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_8x16(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                }
            }
        }
    }

    /// Code an 8x8 luma region as PARTITION_SPLIT into four BLOCK_4X4 luma
    /// sub-blocks (z-order), with the shared 4:2:0 4x4 chroma attached to the
    /// bottom-right sub-block. DC-only luma + DC chroma for now: this is the
    /// bit-exactness scaffold for sub-8x8 luma; richer modes/CfL layer on once
    /// the entropy/recon path is verified against dav1d. Caller has already
    /// emitted the PARTITION_SPLIT symbol.
    fn code_block_split4_dc(&mut self, x8: usize, y8: usize) {
        let (px, py) = (x8 * 8, y8 * 8);
        let maxv = (1 << self.bd) - 1;
        let (dcq, acq) = (self.quant.dc_q() as f64, self.quant.ac_q() as f64);
        let lam = trellis_lambda();
        // mark all four 4x4 luma units for the deblock filter (blk4 == 1)
        let nc4 = self.w / 4;
        for uy in 0..2 {
            for ux in 0..2 {
                self.blk4[(y8 * 2 + uy) * nc4 + (x8 * 2 + ux)] = 1;
                self.blk4h[(y8 * 2 + uy) * nc4 + (x8 * 2 + ux)] = 1;
            }
        }
        // chroma origin (4:2:0): one 4x4 chroma block for the whole 8x8 region
        let (cx, cy) = (px / 2, py / 2);
        // z-order: TL, TR, BL, BR
        let sub = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)];
        for (si, &(sx, sy)) in sub.iter().enumerate() {
            let (bx, by) = (px + sx * 4, py + sy * 4);
            let (bx4, by4) = (bx / 4, by / 4);
            let has_chroma = si == 3;

            // --- luma 4x4: search the non-directional intra modes DC, SMOOTH
            // and PAETH. SMOOTH_V/SMOOTH_H are intentionally excluded: at the
            // 4x4 size their reconstruction does not match dav1d here, and their
            // win over plain SMOOTH is rare/small. These modes use only the
            // above row, left column and above-left corner, so top-right/
            // bottom-left availability is irrelevant; the tx-type is signalled
            // (DCT_DCT), so the mode choice never desyncs.
            let mlam = mode_lambda() * acq * acq;
            let modes = fast_nd_modes();
            let mut best_mode = DC_PRED;
            let mut lpred = [0i32; 16];
            let mut lcf = [0i32; 16];
            let mut best_eff = f64::INFINITY;
            for &m in modes {
                let mut pred = [0i32; 16];
                if m == DC_PRED {
                    let d = dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    intra_predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        false,
                        false,
                        self.w,
                        self.h,
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 16];
                for ry in 0..4 {
                    let srow = &self.src[0][(by + ry) * self.w + bx..];
                    for rx in 0..4 {
                        resid[ry * 4 + rx] = srow[rx] - pred[ry * 4 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_4x4_t(&resid, &self.quant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_4X4, lam);
                let rr = idct_dequant_4x4(&cf, &self.quant);
                let mut sse = 0i64;
                for ry in 0..4 {
                    let srow = &self.src[0][(by + ry) * self.w + bx..];
                    for rx in 0..4 {
                        let r = (pred[ry * 4 + rx] + rr[ry * 4 + rx]).clamp(0, maxv);
                        let d = srow[rx] - r;
                        sse += (d * d) as i64;
                    }
                }
                let bits = block_rate_bits(&cf, &SCAN_4X4);
                let eff = sse as f64 + mlam * bits;
                if eff < best_eff {
                    best_eff = eff;
                    best_mode = m;
                    lpred = pred;
                    lcf = cf;
                }
            }
            let luma_zero = lcf.iter().all(|&c| c == 0);

            // --- chroma (BR only): DC prediction + forward transform ---
            let mut ccf = [[0i32; 16]; 2];
            let mut cpred = [0i32; 2];
            let mut chroma_zero = true;
            if has_chroma && !self.mono {
                let (cdcq, cacq) = (self.cquant.dc_q() as f64, self.cquant.ac_q() as f64);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = dc_pred_4x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                    cpred[ci] = dc;
                    let mut cres = [0i32; 16];
                    for ry in 0..4 {
                        let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                        for rx in 0..4 {
                            cres[ry * 4 + rx] = srow[rx] - dc;
                        }
                    }
                    let (mut q, qt) = forward_dct_quant_4x4_t(&cres, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_4X4, lam);
                    ccf[ci] = q;
                    if !q.iter().all(|&c| c == 0) {
                        chroma_zero = false;
                    }
                }
            }

            let block_skip = if has_chroma {
                luma_zero && chroma_zero
            } else {
                luma_zero
            };

            // --- mode info: skip, y_mode (DC), [uv_mode (DC) if has_chroma] ---
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.enc
                .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
            // AV1 read_delta_qindex(): first block of the SB emits the per-SB
            // delta-q token here (after skip, before the luma mode).
            self.code_delta_q_if_armed();
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
            if has_chroma {
                // chroma stays DC (CfL/SMOOTH_V layered later); uv context uses
                // the luma mode of the chroma-bearing (bottom-right) sub-block.
                self.emit_uv_mode(best_mode, DC_PRED, None);
            }

            // --- residual: luma 4x4, then (BR) chroma U/V 4x4 ---
            let lres_ctx = if block_skip {
                0x40
            } else {
                let ds = self.dc_sign_ctx_420(0, bx4, by4);
                encode_tx4_luma_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &lcf,
                    0, // luma TX_4X4 (tx == block) -> txb_skip ctx 0
                    ds,
                    best_mode,
                    1, // DCT_DCT
                )
            };
            self.a_coef[0][bx4] = lres_ctx;
            self.l_coef[0][by4] = lres_ctx;

            // luma reconstruction
            let lrr = if block_skip {
                [0i32; 16]
            } else {
                idct_dequant_4x4(&lcf, &self.quant)
            };
            for ry in 0..4 {
                let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                for rx in 0..4 {
                    drow[rx] = (lpred[ry * 4 + rx] + lrr[ry * 4 + rx]).clamp(0, maxv);
                }
            }

            // chroma residual + reconstruction (BR only)
            if has_chroma && !self.mono {
                let (bx4c, by4c) = (cx / 4, cy / 4);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let res_ctx = if block_skip {
                        0x40
                    } else {
                        let sk = self.skip_ctx_420(plane, bx4c, by4c);
                        let ds = self.dc_sign_ctx_420(plane, bx4c, by4c);
                        encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                    };
                    self.a_coef[plane][bx4c] = res_ctx;
                    self.l_coef[plane][by4c] = res_ctx;
                    let rr = if block_skip {
                        [0i32; 16]
                    } else {
                        idct_dequant_4x4(&ccf[ci], &self.cquant)
                    };
                    for ry in 0..4 {
                        let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                        for rx in 0..4 {
                            drow[rx] = (cpred[ci] + rr[ry * 4 + rx]).clamp(0, maxv);
                        }
                    }
                }
            }

            // --- neighbour context updates for this 4x4 ---
            self.a_skip[bx4] = block_skip as u8;
            self.l_skip[by4] = block_skip as u8;
            self.a_mode[bx4] = best_mode as u8;
            self.l_mode[by4] = best_mode as u8;
        }
    }

    fn code_block(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 2);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let cx = px / 2; // chroma column for 4:2:2

        // Forward-transform/quantize all planes up front to decide block skip.
        // Luma is always 8x8; chroma is 8x8 (4:4:4) or 4x8 (4:2:2).
        // Luma 8x8: search the non-directional intra modes (DC + SMOOTH*/PAETH)
        // and keep the one minimizing pixel SSE + lambda * estimated bits. The
        // chosen prediction is per-pixel; reconstruction uses the same array so
        // the decoder (which re-derives the identical prediction) stays bit-exact.
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let mlam = mode_lambda() * acq * acq;
        let prdo = self.perceptual_rd_scale(px, py, 8);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let mut best_mode = DC_PRED;
        let mut best_is_adst = false;
        let mut best_is_idtx = false;
        let mut best_is_adstdct = false;
        let mut best_is_dctadst = false;
        let mut lpred_arr = [0i32; 64];
        let mut lcf = [0i32; 64];
        let mut best_eff = f64::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f64;
        let dc_sgn = self.dc_sign_ctx(0, px / 4, py / 4);
        let mut ltf = [0f64; 64]; // winner transform coeffs (f64, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        for &m in modes {
            let mut pred = [0i32; 64];
            if m == DC_PRED {
                let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            for (ry, (rrow, prow)) in resid
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(pred.as_chunks::<8>().0.iter())
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            // Mode decision uses DCT_DCT only (cheap); the ADST_ADST transform
            // choice is refined once for the winning mode after the loop.
            let blk_sse = |rr: &[i32; 64]| -> i64 {
                let mut sse = 0i64;
                for (ry, (prow, rrow)) in pred
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(rr.as_chunks::<8>().0.iter())
                    .enumerate()
                {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        sse += (d * d) as i64;
                    }
                }
                sse
            };
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    &self.cdfs,
                    1,
                    0,
                    &self.cdfs.eob_bin_64_l,
                    dc_sgn,
                );
            }
            let sse = blk_sse(&idct_dequant_8x8(&cf, &self.quant));
            let bits = block_rate_bits(&cf, &SCAN_8X8);
            let cost = sse as f64 + mlam * (bits + mode_signal_bits(m));
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
            }
        }
        // Angle-delta winner refinement: if the winning luma mode is one of the
        // six pure diagonals, try angle_delta in -3..=3 (3 deg steps) and keep the
        // best by SSE + lambda*(coeff bits + angle_delta symbol bits). V/H and the
        // non-directional modes stay at delta 0. ~6 extra predictions per block.
        let mut best_delta: i32 = 0;
        if angle_delta_enabled()
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let ad_cdf = self.cdfs.angle_delta[best_mode - V_PRED].clone();
            let mut best_ad_cost =
                best_dct_sse as f64 + mlam * (best_dct_bits + cdf_cost(&ad_cdf, 3));
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut pred = [0i32; 64];
                intra_predict_nd_ad(
                    best_mode,
                    d,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 64];
                for ry in 0..8 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..8 {
                        resid[ry * 8 + rx] = srow[rx] - pred[ry * 8 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_8X8,
                        lam,
                        8,
                        &self.cdfs,
                        1,
                        0,
                        &self.cdfs.eob_bin_64_l,
                        dc_sgn,
                    );
                }
                let rr = idct_dequant_8x8(&cf, &self.quant);
                let mut sse = 0i64;
                for ry in 0..8 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..8 {
                        let r = (pred[ry * 8 + rx] + rr[ry * 8 + rx]).clamp(0, (1 << self.bd) - 1);
                        let dd = srow[rx] - r;
                        sse += (dd * dd) as i64;
                    }
                }
                let bits = block_rate_bits(&cf, &SCAN_8X8);
                let cost = sse as f64 + mlam * (bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    lpred_arr = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                }
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                &self.cdfs,
                1,
                0,
                &self.cdfs.eob_bin_64_l,
                dc_sgn,
            );
        }
        // Current best transform-type cost (starts at the DCT winner); the ADST
        // refinement updates it so the IDTX refinement compares against whichever
        // of DCT/ADST is currently winning.
        let mut best_txtp_sse = best_dct_sse;
        let mut best_txtp_bits = best_dct_bits;
        // Per-block transform refinement: try ADST_ADST on the winning
        // prediction only and keep it if cheaper than that mode's DCT. This is
        // one extra transform+trellis per block instead of one per candidate
        // mode, which is where the encode-time regression came from.
        // Full and Medium try ADST; only Fast prunes the transform type to DCT_DCT.
        if self.speed.try_adst() {
            let mut resid = [0i32; 64];
            for (ry, rrow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut acf, atf) = adst8x8_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                &self.cdfs,
                1,
                0,
                &self.cdfs.eob_bin_64_l,
                dc_sgn,
            );
            let rr = iadst_dequant_8x8(&acf, &self.quant);
            let mut asse = 0i64;
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    asse += (d * d) as i64;
                }
            }
            let abits = block_rate_bits(&acf, &SCAN_8X8);
            // Quality guard (see 16x16 ADST note): block low-q distortion-for-rate trades.
            if asse <= best_dct_sse + (best_dct_sse >> 5)
                && asse as f64 + mlam * abits < best_dct_sse as f64 + mlam * best_dct_bits
            {
                lcf = acf;
                best_is_adst = true;
                best_txtp_sse = asse;
                best_txtp_bits = abits;
            }
        }
        // Per-block asymmetric-ADST refinement. Intra residual is anisotropic:
        // it grows away from the reference edge in one direction (wants ADST
        // there) and is flat across it (wants DCT). ADST_DCT = vertical ADST,
        // DCT_ADST = horizontal ADST. Each competes with the running tx winner.
        if self.speed.try_adst() && asym_adst_enabled() {
            for (fwd_t, inv_is_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 64];
                for (ry, rrow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                    for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                        *r = s - p;
                    }
                }
                let (mut acf, atf) = if fwd_t {
                    dctadst8x8_t(&resid, &self.quant)
                } else {
                    adstdct8x8_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    &self.cdfs,
                    1,
                    0,
                    &self.cdfs.eob_bin_64_l,
                    dc_sgn,
                );
                let rr = if inv_is_dctadst {
                    idctadst_dequant_8x8(&acf, &self.quant)
                } else {
                    iadstdct_dequant_8x8(&acf, &self.quant)
                };
                let mut asse = 0i64;
                for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        asse += (d * d) as i64;
                    }
                }
                let abits = block_rate_bits(&acf, &SCAN_8X8);
                if asse <= best_dct_sse + (best_dct_sse >> 5)
                    && asse as f64 + mlam * abits < best_txtp_sse as f64 + mlam * best_txtp_bits
                {
                    lcf = acf;
                    best_is_adst = false;
                    best_is_idtx = false;
                    best_is_adstdct = !inv_is_dctadst;
                    best_is_dctadst = inv_is_dctadst;
                    best_txtp_sse = asse;
                    best_txtp_bits = abits;
                }
            }
        }
        // Per-block IDTX refinement: the identity transform (no spatial
        // decorrelation) wins on sharp edges / screen-content-like residuals
        // where DCT/ADST spread a step across many coefficients. One extra
        // forward+inverse on the winning prediction; kept only if it beats the
        // current best tx by real recon SSE + estimated bits. Bit-exactness is
        // carried by `iidentity_dequant_8x8` (dav1d's exact TX_8X8 IDTX inverse);
        // the IDTX symbol is 0 in the 7-type intra ext-tx set.
        if self.speed.try_adst() {
            let mut resid = [0i32; 64];
            for (ry, rrow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (icf, _itf) = fidentity8x8_t(&resid, &self.quant);
            // No RDOQ on IDTX: because the identity transform spreads a residual
            // across many small coefficients, an aggressive trellis zeros them
            // all and the block-level bit term then picks the collapsed result.
            // Plain forward levels keep IDTX conservative (chosen only on a clear
            // real-SSE win); bit-exactness is carried by the inverse regardless.
            let rr = iidentity_dequant_8x8(&icf, &self.quant);
            let mut isse = 0i64;
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    isse += (d * d) as i64;
                }
            }
            let ibits = block_rate_bits(&icf, &SCAN_8X8);
            // Quality guard (see ADST note): identity spreads residual energy and
            // is cheap to code, so at low-q lambda a pure RD test over-selects it
            // and flattens detail. Require SSE-non-worsening vs the best real tx.
            if isse <= best_txtp_sse + (best_txtp_sse >> 5)
                && isse as f64 + mlam * ibits < best_txtp_sse as f64 + mlam * best_txtp_bits
            {
                lcf = icf;
                best_is_adst = false;
                best_is_idtx = true;
                best_is_adstdct = false;
                best_is_dctadst = false;
            }
        }
        let mut ccf8 = [[0i32; 64]; 2];
        let mut ccf48 = [[0i32; 32]; 2];
        let mut ccf44 = [[0i32; 16]; 2];
        let mut cpred = [0i32; 2];
        let cy = py / 2; // chroma row for 4:2:0
        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let pred = dc_pred_4x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 16];
                for (ry, drow) in resid.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                ccf44[ci] = q;
                trellis_optimize(&mut ccf44[ci], &qt, dcq, acq, &SCAN_4X4, lam);
            } else if self.ss422 {
                let pred = dc_pred_4x8(&self.recon[plane], self.cw, cx, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 32];
                for (ry, drow) in resid.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                ccf48[ci] = q;
                trellis_optimize(&mut ccf48[ci], &qt, dcq, acq, &SCAN_4X8, lam);
            } else {
                let pred = dc_pred_8x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 64];
                for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                ccf8[ci] = q;
                trellis_optimize(&mut ccf8[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            }
        }

        // 4:4:4 chroma-from-luma: try predicting U/V from the reconstructed luma
        // block (scaled, mean-removed) and pick CfL over plain DC per block.
        let mut cpred444 = [[0i32; 64]; 2];
        let mut cpred420 = [[0i32; 16]; 2];
        let mut cpred422 = [[0i32; 32]; 2];
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        if !self.mono && !self.ss420 && !self.ss422 {
            // CfL luma reference must use the SAME inverse transform the decoder
            // will apply (the signaled luma tx-type), or the chroma CfL prediction
            // desyncs. Previously this was unconditionally idct, which diverged
            // whenever the luma block won with ADST or IDTX.
            let lrr_cfl = if best_is_idtx {
                iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 64];
            cfl_ac_444(&luma_rec, 8, 8, &mut ac);
            let mut cfl_ccf = [[0i32; 64]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f64; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f64; 2]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 64];
                for (ry, drow) in src.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..8]);
                }
                // DC option distortion/rate (from the coeffs already computed)
                let dcrr = idct_dequant_8x8(&ccf8[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..64 {
                    let r = (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s += (d * d) as i64;
                }
                dc_sse[ci] = s;
                dc_bits[ci] = block_rate_bits(&ccf8[ci], &SCAN_8X8);
                // CfL option
                let a = cfl_best_alpha(&ac, &src, dc, 64, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 64];
                let mut resid = [0i32; 64];
                for i in 0..64 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                let rr = idct_dequant_8x8(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..64 {
                    let r = (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_sse[ci] = s2;
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_8X8);
                cpred444[ci] = cpr;
            }
            // joint signalling cost estimate (sign symbol + 1 magnitude per non-zero plane)
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let dc_total = (dc_sse[0] + dc_sse[1]) as f64 + mlam * (dc_bits[0] + dc_bits[1]);
            let cfl_total =
                (cfl_sse[0] + cfl_sse[1]) as f64 + mlam * (cfl_bits[0] + cfl_bits[1] + sig);
            // Let the RD comparison decide DC-vs-CfL across the whole quality
            // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
            // where it helps most (high quality).
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf8[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred444[ci] = [cpred[ci]; 64];
                }
            }
        }

        // 4:4:4 directional chroma: PAETH_PRED and SMOOTH_PRED, both mapped to
        // ADST_ADST (the decoder derives the chroma tx-type from uv_mode, so
        // signalling either selects ADST_ADST automatically). These track
        // edges/gradients that plain DC smooths away — exactly the over-smoothing
        // the chroma path suffers from. Only considered when CfL did not win, and
        // chosen on a real RD margin over DC.
        let mut chosen_uv_444 = if use_cfl { CFL_PRED } else { DC_PRED };
        let mut paeth_pred444 = [[0i32; 64]; 2];
        // NOTE: the 4:4:4 8x8 chroma path has a pre-existing reconstruction
        // divergence from the decoder at >8-bit (present in plain DC chroma,
        // independent of directional modes — luma and the 4:2:0/4:2:2 4x4/4x8
        // chroma paths are byte-exact at 10/12-bit). Directional modes propagate
        // that corrupted reconstruction, so restrict them to 8-bit here until the
        // baseline 4:4:4 high-bit-depth chroma issue is fixed. 4:2:0/4:2:2 below
        // are byte-exact at all bit depths and stay enabled.
        if !self.mono && !self.ss420 && !self.ss422 && !use_cfl {
            let maxv = (1 << self.bd) - 1;
            // DC reference cost (current `ccf8`).
            let mut dc_total = 0f64;
            let mut src_planes = [[0i32; 64]; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let mut src = [0i32; 64];
                for (ry, drow) in src.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..8]);
                }
                src_planes[ci] = src;
                let dcrr = idct_dequant_8x8(&ccf8[ci], &self.cquant);
                let mut sse = 0i64;
                for i in 0..64 {
                    let r = (cpred[ci] + dcrr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    sse += (d * d) as i64;
                }
                dc_total += sse as f64 + mlam * block_rate_bits(&ccf8[ci], &SCAN_8X8);
            }
            // Try each directional candidate with its mode-derived transform;
            // keep the best that also beats DC by the mode-signalling margin.
            // V/H additionally emit a chroma angle_delta symbol (only valid here
            // at 8x8 4:4:4 chroma), costed below.
            let mut best_total = dc_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf = ccf8;
            let mut best_pred = [[0i32; 64]; 2];
            for &cand in &[
                PAETH_PRED,
                SMOOTH_PRED,
                SMOOTH_V_PRED,
                SMOOTH_H_PRED,
                V_PRED,
                H_PRED,
            ] {
                let tx = chroma_tx_for_mode(cand);
                // mode symbol (~4 bits) + angle_delta symbol (~3 bits) for V/H
                let sig_bits = if cand == V_PRED || cand == H_PRED {
                    7.0
                } else {
                    4.0
                };
                let mut cand_ccf = [[0i32; 64]; 2];
                let mut cand_pred = [[0i32; 64]; 2];
                let mut cand_total = mlam * sig_bits;
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut pp = [0i32; 64];
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.w,
                        px,
                        py,
                        8,
                        8,
                        false,
                        false,
                        self.w,
                        self.h,
                        &mut pp,
                        self.bd,
                    );
                    let mut resid = [0i32; 64];
                    for i in 0..64 {
                        resid[i] = src_planes[ci][i] - pp[i];
                    }
                    let (mut q, qt) = fwd_chroma_8x8(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                    let rr = inv_chroma_8x8(tx, &q, &self.cquant);
                    let mut sse = 0i64;
                    for i in 0..64 {
                        let r = (pp[i] + rr[i]).clamp(0, maxv);
                        let d = src_planes[ci][i] - r;
                        sse += (d * d) as i64;
                    }
                    cand_total += sse as f64 + mlam * block_rate_bits(&q, &SCAN_8X8);
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                chosen_uv_444 = best_mode_uv;
                ccf8[..2].copy_from_slice(&best_ccf[..2]);
                paeth_pred444[..2].copy_from_slice(&best_pred[..2]);
            }
        }

        let chroma_zero = |ci: usize| {
            if self.ss420 {
                ccf44[ci].iter().all(|&c| c == 0)
            } else if self.ss422 {
                ccf48[ci].iter().all(|&c| c == 0)
            } else {
                ccf8[ci].iter().all(|&c| c == 0)
            }
        };
        let block_skip =
            lcf.iter().all(|&c| c == 0) && (self.mono || (chroma_zero(0) && chroma_zero(1)));

        // block-level mode info: skip (ctx = above_skip + left_skip), y/uv = DC
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        // AV1 read_delta_qindex(): first block of the SB emits the per-SB
        // delta-q token here (after skip, before the luma mode).
        self.code_delta_q_if_armed();
        self.mark_skip8(x8, y8, 1, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&best_mode) {
            // angle_delta refined for diagonals (above); V/H stay at delta 0.
            self.enc.encode_symbol(
                (best_delta + 3) as usize,
                &mut self.cdfs.angle_delta[best_mode - V_PRED],
            );
        }
        // SMOOTH_V check for 4:2:0 4x4 chroma: only at low quality (ac_q > 300)
        let smooth_v_active_ss420 = false; // see note: chroma SMOOTH_V -> ADST_DCT not implemented; would desync decoder
        let mut sv_preds_420 = [[0i32; 16]; 2];
        let mut chosen_uv_block = DC_PRED;
        // 4:2:0 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x4). Populated by
        // the search block below (after CfL); recon uses `iadst_dequant_4x4`.
        let mut chosen_uv_420 = DC_PRED;
        let mut paeth_pred420 = [[0i32; 16]; 2];
        // 4:2:2 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x8).
        let mut chosen_uv_422 = DC_PRED;
        let mut paeth_pred422 = [[0i32; 32]; 2];
        if !self.mono && self.ss420 && smooth_v_active_ss420 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let mut sv_ccf44_2 = [[0i32; 16]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                for ry in 0..4 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for &sr in srow[..4].iter() {
                        let d = sr - dc;
                        sse_cur += (d * d) as i64;
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    4,
                    4,
                    false,
                    false,
                    self.cw,
                    self.h,
                    &mut sv_preds_420[ci],
                    self.bd,
                );
                let mut resid = [0i32; 16];
                for (ry, drow) in resid.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds_420[ci][ry * 4..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                sv_ccf44_2[ci] = q;
                trellis_optimize(&mut sv_ccf44_2[ci], &qt, dcq2, acq2, &SCAN_4X4, lam2);
                for ry in 0..4 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds_420[ci][ry * 4..];
                    for j in 0..4 {
                        let d = srow[j] - prow[j];
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv < sse_cur {
                ccf44[..2].copy_from_slice(&sv_ccf44_2[..2]);
                chosen_uv_block = SMOOTH_V_PRED;
            }
        }
        // Note: SMOOTH_V for 4:4:4 8x8 (code_block small-block path) is intentionally
        // not added here — it introduces too many DC↔SV mode transitions at 8-row
        // boundaries that are visible as faint lines at quality 50-75.
        // 4:2:0 chroma-from-luma: predict the 4x4 U/V from the 2x2-subsampled
        // reconstructed luma of this 8x8 block (dav1d cfl_ac, ss_hor=ss_ver=1).
        // Competes with the current DC/SMOOTH_V choice on rate-distortion.
        if !self.mono && self.ss420 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let lrr = if best_is_idtx {
                iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 16];
            cfl_ac_sub(&luma_rec, 8, 4, 4, true, true, &mut ac);
            let mut cfl_ccf = [[0i32; 16]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
            let (mut cur_bits, mut cfl_bits) = (0f64, 0f64);
            let maxv = (1 << self.bd) - 1;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 16];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(cy + ry) * self.cw + cx..][..4]);
                }
                let curr = idct_dequant_4x4(&ccf44[ci], &self.cquant);
                for i in 0..16 {
                    let p = if chosen_uv_block == SMOOTH_V_PRED {
                        sv_preds_420[ci][i]
                    } else {
                        dc
                    };
                    let r = (p + curr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cur_sse += (d * d) as i64;
                }
                cur_bits += block_rate_bits(&ccf44[ci], &SCAN_4X4);
                let a = cfl_best_alpha(&ac, &src, dc, 16, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 16];
                let mut resid = [0i32; 16];
                for i in 0..16 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X4, lam2);
                let rr = idct_dequant_4x4(&q, &self.cquant);
                for i in 0..16 {
                    let r = (cpr[i] + rr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cfl_sse += (d * d) as i64;
                }
                cfl_bits += block_rate_bits(&q, &SCAN_4X4);
                cfl_ccf[ci] = q;
                cpred420[ci] = cpr;
            }
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let cur_total = cur_sse as f64 + mlam * cur_bits;
            let cfl_total = cfl_sse as f64 + mlam * (cfl_bits + sig);
            if cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf44[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        // 4:2:0 directional chroma: PAETH_PRED / SMOOTH_PRED (both -> ADST_ADST,
        // now available at 4x4). Same rationale and structure as the 4:4:4 path:
        // tracks chroma edges/gradients that plain DC over-smooths. Considered
        // only when CfL did not win; chosen on a real RD margin over DC.
        if !self.mono && self.ss420 && !use_cfl && self.cquant.ac_q() < 120 {
            let maxv = (1 << self.bd) - 1;
            let mut src_planes = [[0i32; 16]; 2];
            let mut dc_total = 0f64;
            for ci in 0..2 {
                let plane = ci + 1;
                let mut src = [0i32; 16];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(cy + ry) * self.cw + cx..][..4]);
                }
                src_planes[ci] = src;
                let dcrr = idct_dequant_4x4(&ccf44[ci], &self.cquant);
                let mut sse = 0i64;
                for i in 0..16 {
                    let r = (cpred[ci] + dcrr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    sse += (d * d) as i64;
                }
                dc_total += sse as f64 + mlam * block_rate_bits(&ccf44[ci], &SCAN_4X4);
            }
            let mut best_total = dc_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf = ccf44;
            let mut best_pred = [[0i32; 16]; 2];
            for &cand in &[PAETH_PRED, SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED] {
                let tx = chroma_tx_for_mode(cand);
                let mut cand_ccf = [[0i32; 16]; 2];
                let mut cand_pred = [[0i32; 16]; 2];
                let mut cand_total = mlam * 4.0; // non-DC uv_mode signalling
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut pp = [0i32; 16];
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                        false,
                        false,
                        self.cw,
                        self.h,
                        &mut pp,
                        self.bd,
                    );
                    let mut resid = [0i32; 16];
                    for i in 0..16 {
                        resid[i] = src_planes[ci][i] - pp[i];
                    }
                    let (mut q, qt) = fwd_chroma_4x4(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X4, lam);
                    let rr = inv_chroma_4x4(tx, &q, &self.cquant);
                    let mut sse = 0i64;
                    for i in 0..16 {
                        let r = (pp[i] + rr[i]).clamp(0, maxv);
                        let d = src_planes[ci][i] - r;
                        sse += (d * d) as i64;
                    }
                    cand_total += sse as f64 + mlam * block_rate_bits(&q, &SCAN_4X4);
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                chosen_uv_420 = best_mode_uv;
                ccf44[..2].copy_from_slice(&best_ccf[..2]);
                paeth_pred420[..2].copy_from_slice(&best_pred[..2]);
            }
        }
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0).
        if !self.mono && self.ss422 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let lrr = if best_is_idtx {
                iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 32];
            cfl_ac_sub(&luma_rec, 8, 4, 8, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 32]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
            let (mut cur_bits, mut cfl_bits) = (0f64, 0f64);
            let maxv = (1 << self.bd) - 1;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 32];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..4]);
                }
                let curr = idct_dequant_4x8(&ccf48[ci], &self.cquant);
                for i in 0..32 {
                    let r = (dc + curr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cur_sse += (d * d) as i64;
                }
                cur_bits += block_rate_bits(&ccf48[ci], &SCAN_4X8);
                let a = cfl_best_alpha(&ac, &src, dc, 32, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 32];
                let mut resid = [0i32; 32];
                for i in 0..32 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X8, lam2);
                let rr = idct_dequant_4x8(&q, &self.cquant);
                for i in 0..32 {
                    let r = (cpr[i] + rr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cfl_sse += (d * d) as i64;
                }
                cfl_bits += block_rate_bits(&q, &SCAN_4X8);
                cfl_ccf[ci] = q;
                cpred422[ci] = cpr;
            }
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let cur_total = cur_sse as f64 + mlam * cur_bits;
            let cfl_total = cfl_sse as f64 + mlam * (cfl_bits + sig);
            if cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf48[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        // 4:2:2 directional chroma: PAETH_PRED / SMOOTH_PRED (-> ADST_ADST 4x8).
        // Same rationale/structure as the 4:2:0 path; block is 4 wide x 8 tall at
        // chroma coords (cx, py). Gated to higher quality (chroma ac_q < 120) and
        // only when CfL did not win; chosen on a real RD margin over DC.
        if !self.mono && self.ss422 && !use_cfl && self.cquant.ac_q() < 120 {
            let maxv = (1 << self.bd) - 1;
            let mut src_planes = [[0i32; 32]; 2];
            let mut dc_total = 0f64;
            for ci in 0..2 {
                let plane = ci + 1;
                let mut src = [0i32; 32];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..4]);
                }
                src_planes[ci] = src;
                let dcrr = idct_dequant_4x8(&ccf48[ci], &self.cquant);
                let mut sse = 0i64;
                for i in 0..32 {
                    let r = (cpred[ci] + dcrr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    sse += (d * d) as i64;
                }
                dc_total += sse as f64 + mlam * block_rate_bits(&ccf48[ci], &SCAN_4X8);
            }
            let mut best_total = dc_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf = ccf48;
            let mut best_pred = [[0i32; 32]; 2];
            for &cand in &[PAETH_PRED, SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED] {
                let tx = chroma_tx_for_mode(cand);
                let mut cand_ccf = [[0i32; 32]; 2];
                let mut cand_pred = [[0i32; 32]; 2];
                let mut cand_total = mlam * 4.0;
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut pp = [0i32; 32];
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.cw,
                        cx,
                        py,
                        4,
                        8,
                        false,
                        false,
                        self.cw,
                        self.h,
                        &mut pp,
                        self.bd,
                    );
                    let mut resid = [0i32; 32];
                    for i in 0..32 {
                        resid[i] = src_planes[ci][i] - pp[i];
                    }
                    let (mut q, qt) = fwd_chroma_4x8(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X8, lam);
                    let rr = inv_chroma_4x8(tx, &q, &self.cquant);
                    let mut sse = 0i64;
                    for i in 0..32 {
                        let r = (pp[i] + rr[i]).clamp(0, maxv);
                        let d = src_planes[ci][i] - r;
                        sse += (d * d) as i64;
                    }
                    cand_total += sse as f64 + mlam * block_rate_bits(&q, &SCAN_4X8);
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                chosen_uv_422 = best_mode_uv;
                ccf48[..2].copy_from_slice(&best_ccf[..2]);
                paeth_pred422[..2].copy_from_slice(&best_pred[..2]);
            }
        }
        if !self.mono {
            // 4:4:4 uses the directional (PAETH) decision; 4:2:0/4:2:2 use their
            // own block-mode choice. CfL overrides via the alpha argument.
            let uv_mode_sym = if !self.ss420 && !self.ss422 {
                chosen_uv_444
            } else if self.ss420 {
                // 4:2:0: directional (PAETH/SMOOTH via ADST4) overrides DC; the
                // legacy SMOOTH_V path stays gated off (chosen_uv_block == DC).
                if chosen_uv_420 != DC_PRED {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                }
            } else if self.ss422 {
                if chosen_uv_422 != DC_PRED {
                    chosen_uv_422
                } else {
                    chosen_uv_block
                }
            } else {
                chosen_uv_block
            };
            self.emit_uv_mode(
                best_mode,
                uv_mode_sym,
                if use_cfl { Some(cfl_alpha_uv) } else { None },
            );
        }
        let sv = block_skip as u8;
        self.a_skip[bx4] = sv;
        self.a_skip[bx4 + 1] = sv;
        self.l_skip[by4] = sv;
        self.l_skip[by4 + 1] = sv;
        let mv = best_mode as u8;
        self.a_mode[bx4] = mv;
        self.a_mode[bx4 + 1] = mv;
        self.l_mode[by4] = mv;
        self.l_mode[by4 + 1] = mv;

        // luma (TX_8X8)
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx(0, bx4, by4, false);
            let ds = self.dc_sign_ctx(0, bx4, by4);
            encode_tx8_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                &lcf,
                false,
                sk,
                ds,
                best_mode,
                if best_is_idtx {
                    0
                } else if best_is_adst {
                    ADST_ADST_TX8_IDX
                } else if best_is_adstdct {
                    ADST_DCT_TX8_IDX
                } else if best_is_dctadst {
                    DCT_ADST_TX8_IDX
                } else {
                    1
                },
            )
        };
        self.a_coef[0][bx4] = lres_ctx;
        self.a_coef[0][bx4 + 1] = lres_ctx;
        self.l_coef[0][by4] = lres_ctx;
        self.l_coef[0][by4 + 1] = lres_ctx;
        let lrr = if block_skip {
            [0i32; 64]
        } else if best_is_idtx {
            iidentity_dequant_8x8(&lcf, &self.quant)
        } else if best_is_adst {
            iadst_dequant_8x8(&lcf, &self.quant)
        } else if best_is_adstdct {
            iadstdct_dequant_8x8(&lcf, &self.quant)
        } else if best_is_dctadst {
            idctadst_dequant_8x8(&lcf, &self.quant)
        } else {
            idct_dequant_8x8(&lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred_arr
            .as_chunks::<8>()
            .0
            .iter()
            .zip(lrr.as_chunks::<8>().0.iter())
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }

        // chroma U, V
        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let (bx4c, by4c) = (cx / 4, cy / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_420(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_420(plane, bx4c, by4c);
                    encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf44[ci], sk, ds)
                };
                // TX_4X4: 1 coef-context unit wide and tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                let paeth420 = chosen_uv_420 != DC_PRED;
                let rr = if block_skip {
                    [0i32; 16]
                } else if paeth420 {
                    inv_chroma_4x4(chroma_tx_for_mode(chosen_uv_420), &ccf44[ci], &self.cquant)
                } else {
                    idct_dequant_4x4(&ccf44[ci], &self.cquant)
                };
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    if use_cfl {
                        let prow = &cpred420[ci][ry * 4..];
                        for ((dv, &rv), &p) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if chosen_uv_block == SMOOTH_V_PRED {
                        let prow = &sv_preds_420[ci][ry * 4..];
                        for ((dv, &rv), &prow) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (prow + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if paeth420 {
                        let prow = &paeth_pred420[ci][ry * 4..];
                        for ((dv, &rv), &p) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            } else if self.ss422 {
                let (bx4c, by4c) = (cx / 4, py / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_422(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_422(plane, bx4c, by4c);
                    encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf48[ci], sk, ds)
                };
                // RTX_4X8: 1 coef-context unit wide, 2 units tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                self.l_coef[plane][by4c + 1] = res_ctx;
                let paeth422 = chosen_uv_422 != DC_PRED;
                let rr = if block_skip {
                    [0i32; 32]
                } else if paeth422 {
                    inv_chroma_4x8(chroma_tx_for_mode(chosen_uv_422), &ccf48[ci], &self.cquant)
                } else {
                    idct_dequant_4x8(&ccf48[ci], &self.cquant)
                };
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                    if use_cfl {
                        let prow = &cpred422[ci][ry * 4..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if paeth422 {
                        let prow = &paeth_pred422[ci][ry * 4..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            } else {
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx(plane, bx4, by4, true);
                    let ds = self.dc_sign_ctx(plane, bx4, by4);
                    encode_tx8_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &ccf8[ci],
                        true,
                        sk,
                        ds,
                        0,
                        1,
                    )
                };
                self.a_coef[plane][bx4] = res_ctx;
                self.a_coef[plane][bx4 + 1] = res_ctx;
                self.l_coef[plane][by4] = res_ctx;
                self.l_coef[plane][by4 + 1] = res_ctx;
                let paeth = chosen_uv_444 != DC_PRED && chosen_uv_444 != CFL_PRED;
                let rr = if block_skip {
                    [0i32; 64]
                } else if paeth {
                    // Directional chroma: tx derived from uv_mode (Mode_To_Txfm).
                    inv_chroma_8x8(chroma_tx_for_mode(chosen_uv_444), &ccf8[ci], &self.cquant)
                } else {
                    idct_dequant_8x8(&ccf8[ci], &self.cquant)
                };
                for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        let prow = &cpred444[ci][ry * 8..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if paeth {
                        let prow = &paeth_pred444[ci][ry * 8..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        // Plain DC chroma: use the scalar predictor directly so the
                        // reconstruction never depends on the CfL evaluation block
                        // having populated `cpred444`.
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            }
        }
    }

    /// 4:2:2 chroma txb_skip (all_zero) context for an RTX_4X8 block (1 unit
    /// wide, 2 units tall; `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = (a[bx4c] != 0x40) as usize;
        let cl = (l[by4c] != 0x40 || l[by4c + 1] != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_4X8: 1 unit wide, 2 tall, baseline -3.
    fn dc_sign_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 + (l[by4c + 1] >> 6) as i32 - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_8X16 block (2 units wide, 4 tall;
    /// chroma tx == chroma block so ctx_offset = 7): `7 + a_nz + l_nz`, where each
    /// term ORs over the units the block spans.
    fn skip_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = (a[bx4c] != 0x40 || a[bx4c + 1] != 0x40) as usize;
        let cl =
            (l[by4c] != 0x40 || l[by4c + 1] != 0x40 || l[by4c + 2] != 0x40 || l[by4c + 3] != 0x40)
                as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_8X16: 2 units wide, 4 tall, baseline -6.
    fn dc_sign_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32
            + (a[bx4c + 1] >> 6) as i32
            + (l[by4c] >> 6) as i32
            + (l[by4c + 1] >> 6) as i32
            + (l[by4c + 2] >> 6) as i32
            + (l[by4c + 3] >> 6) as i32
            - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_16X32 block (4 units wide, 8 tall).
    fn skip_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4c..bx4c + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4c..by4c + 8].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_16X32: 4 units wide, 8 tall, baseline -12.
    fn dc_sign_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4c..bx4c + 4].iter().map(|x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4c..by4c + 8].iter().map(|x| (x >> 6) as i32).sum();
        let s = suma + suml - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:0 chroma txb_skip context for a TX_4X4 block (1 unit wide and tall;
    /// `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        7 + (a[bx4c] != 0x40) as usize + (l[by4c] != 0x40) as usize
    }

    /// 4:2:0 chroma dc_sign context for TX_4X4: 1 unit each side, baseline -2.
    fn dc_sign_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 - 2;
        (s != 0) as usize + (s > 0) as usize
    }

    /// dc_sign context for a TX_32X32 (8-unit footprint, baseline -16).
    fn dc_sign_ctx_32(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = (0..8).map(|k| (a[bx4 + k] >> 6) as i32).sum();
        let suml: i32 = (0..8).map(|k| (l[by4 + k] >> 6) as i32).sum();
        let s = suma + suml - 16;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a TX_32X32 (8-unit footprint). Luma (max tx in a
    /// 32x32 block) is always ctx 0; chroma uses `7 + above_nz + left_nz`.
    fn skip_ctx_32(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = (0..8).any(|k| a[bx4 + k] != 0x40) as usize;
            let cl = (0..8).any(|k| l[by4 + k] != 0x40) as usize;
            7 + ca + cl
        }
    }

    fn prefer_32x32(&self, _x8: usize, _y8: usize) -> bool {
        let policy = tx32_policy();
        if policy == 0 || self.mono {
            return false;
        }
        if policy == 1 && self.block_luma_range(_x8, _y8, 32) < tx32_smooth_gate() {
            return false;
        }
        let (px, py) = (_x8 * 8, _y8 * 8);
        let lpred = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
        let mut r32 = [0i32; 1024];
        for (ry, drow) in r32.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let srow = &self.src[0][(py + ry) * self.w + px..];
            for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                *dv = s - lpred;
            }
        }
        forward_dct_quant_32x32(&mut r32, &self.quant);
        let cost32: u32 = est_block_bits(&r32, &SCAN_32X32) + OVERHEAD_16;
        let mut cost16 = 0u32;
        for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
            let pred = dc_pred_16x16(&self.recon[0], self.w, px + sx, py + sy, self.bd as i32);
            let mut r16 = [0i32; 256];
            for (ry, drow) in r16.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let srow = &self.src[0][(py + sy + ry) * self.w + px + sx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            forward_dct_quant_16x16(&mut r16, &self.quant);
            cost16 += est_block_bits(&r16, &SCAN_16X16) + OVERHEAD_16;
        }
        cost32 + (cost16 >> 4) <= cost16
    }

    /// Code a 32x32 region (4:4:4 only) as a single TX_32X32 block: DC-pred luma
    /// and both chroma planes, forward DCT32 + quant + trellis, the TX_32X32
    /// coefficient coder, and reconstruction via the exact integer inverse.
    /// Updates the 8-unit (32-sample) skip / mode / coef neighbour footprint.
    /// (DC-only for now; SMOOTH/PAETH/directional and CfL at 32x32 come next.)
    fn code_block32(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 8);
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let mlam = mode_lambda() * acq * acq;
        let prdo = self.perceptual_rd_scale(px, py, 32);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // luma intra mode search (non-directional + directional; the TX_32X32
        // residual transform is always DCT_DCT, so the mode affects prediction
        // only). Mirrors the 16x16 search.
        let mut best_mode = DC_PRED;
        let mut lpred = [0i32; 1024];
        let mut lcf = [0i32; 1024];
        let mut best_eff = f64::INFINITY;
        let mut ltf = [0f64; 1024]; // winner transform coeffs (f64, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        for &m in modes {
            let mut pred = [0i32; 1024];
            if m == DC_PRED {
                let d = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 1024];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 1024];
            for (ry, (rrow, prow)) in resid
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(pred.as_chunks::<32>().0.iter())
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    &self.cdfs,
                    3,
                    0,
                    &self.cdfs.eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, px / 4, py / 4),
                );
            }
            let rr = idct_dequant_32x32(&cf, &self.quant);
            let mut sse = 0i64;
            for (ry, (prow, rrow)) in pred
                .as_chunks::<32>()
                .0
                .iter()
                .zip(rr.as_chunks::<32>().0.iter())
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    sse += (d * d) as i64;
                }
            }
            let bits = block_rate_bits(&cf, &SCAN_32X32) + mode_signal_bits(m);
            let cost = sse as f64 + mlam * bits;
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred = pred;
                lcf = cf;
                ltf = tf;
            }
        }
        // Angle-delta winner refinement (see code_block: diagonals only, -3..=3).
        let mut best_delta: i32 = 0;
        if angle_delta_enabled()
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let ad_cdf = self.cdfs.angle_delta[best_mode - V_PRED].clone();
            let ds = self.dc_sign_ctx_32(0, px / 4, py / 4);
            let wrr = idct_dequant_32x32(&lcf, &self.quant);
            let mut wsse = 0i64;
            for ry in 0..32 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for rx in 0..32 {
                    let r = (lpred[ry * 32 + rx] + wrr[ry * 32 + rx]).clamp(0, (1 << self.bd) - 1);
                    let dd = srow[rx] - r;
                    wsse += (dd * dd) as i64;
                }
            }
            let wbits = block_rate_bits(&lcf, &SCAN_32X32);
            let mut best_ad_cost = wsse as f64 + mlam * (wbits + cdf_cost(&ad_cdf, 3));
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut pred = [0i32; 1024];
                intra_predict_nd_ad(
                    best_mode,
                    d,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 1024];
                for ry in 0..32 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..32 {
                        resid[ry * 32 + rx] = srow[rx] - pred[ry * 32 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_32X32,
                        lam,
                        32,
                        &self.cdfs,
                        3,
                        0,
                        &self.cdfs.eob_bin_1024_l,
                        ds,
                    );
                }
                let rr = idct_dequant_32x32(&cf, &self.quant);
                let mut sse = 0i64;
                for ry in 0..32 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..32 {
                        let r =
                            (pred[ry * 32 + rx] + rr[ry * 32 + rx]).clamp(0, (1 << self.bd) - 1);
                        let dd = srow[rx] - r;
                        sse += (dd * dd) as i64;
                    }
                }
                let bits = block_rate_bits(&cf, &SCAN_32X32);
                let cost = sse as f64 + mlam * (bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    lpred = pred;
                    lcf = cf;
                    ltf = tf;
                }
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                &self.cdfs,
                3,
                0,
                &self.cdfs.eob_bin_1024_l,
                self.dc_sign_ctx_32(0, px / 4, py / 4),
            );
        }
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block32_420(x8, y8, &lcf, &lpred, best_mode, luma_zero, best_delta);
        } else if self.ss422 {
            self.code_block32_422(x8, y8, &lcf, &lpred, best_mode, luma_zero, best_delta);
        } else {
            self.code_block32_444(x8, y8, &lcf, &lpred, best_mode, luma_zero, best_delta);
        }
    }

    /// Shared header + luma for a TX_32X32 block: block skip flag, y/uv modes
    /// (uv via `emit_uv_mode`, plain DC or CfL), `angle_delta` for directional
    /// luma modes, the TX_32X32 luma coefficients (no tx-type symbol), the
    /// 8-unit (32-sample) skip/mode/coef footprint, and luma reconstruction.
    #[allow(clippy::too_many_arguments)]
    fn code_header_luma32(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        // AV1 read_delta_qindex(): first block of the SB emits the per-SB
        // delta-q token here (after skip, before the luma mode).
        self.code_delta_q_if_armed();
        self.mark_skip8(x8, y8, 4, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc.encode_symbol(
                (angle_delta + 3) as usize,
                &mut self.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        self.emit_uv_mode(y_mode, uv_mode, cfl);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 8].fill(sv);
        self.l_skip[by4..by4 + 8].fill(sv);
        self.a_mode[bx4..bx4 + 8].fill(mv);
        self.l_mode[by4..by4 + 8].fill(mv);
        let lres = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_32(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_32(0, bx4, by4);
            encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, lcf, false, sk, ds)
        };
        self.a_coef[0][bx4..bx4 + 8].fill(lres);
        self.l_coef[0][by4..by4 + 8].fill(lres);
        let lrr = if block_skip {
            [0i32; 1024]
        } else {
            idct_dequant_32x32(lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred
            .as_chunks::<32>()
            .0
            .iter()
            .zip(lrr.as_chunks::<32>().0.iter())
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block32_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f64,
            self.cquant.ac_q() as f64,
            trellis_lambda(),
        );
        // plain-DC chroma
        let mut ccf = [[0i32; 1024]; 2];
        let mut cdc = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = dc_pred_32x32(&self.recon[plane], self.w, px, py, self.bd as i32);
            cdc[ci] = dc;
            let mut cresid = [0i32; 1024];
            for (ry, drow) in cresid.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(py + ry) * self.w + px..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_32x32_t(&cresid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(&mut ccf[ci], &qt, dcq, acq, &SCAN_32X32, lam);
            let mean_resid_dc = cresid.iter().sum::<i32>() / 1024;
            if ccf[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // CfL: predict chroma from the reconstructed luma AC.
        let mut cfl_ccf = [[0i32; 1024]; 2];
        let mut cfl_pred = [[0i32; 1024]; 2];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f64; 2], [0f64; 2]);
        let mlam = mode_lambda() * acq * acq;
        {
            let lrr_cfl = idct_dequant_32x32(lcf, &self.quant);
            let mut luma_rec = [0i32; 1024];
            for i in 0..1024 {
                luma_rec[i] = (lpred[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 1024];
            cfl_ac_444(&luma_rec, 32, 32, &mut ac);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cdc[ci];
                let mut src = [0i32; 1024];
                for (ry, drow) in src.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..32]);
                }
                let dcrr = idct_dequant_32x32(&ccf[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..1024 {
                    let d = src[i] - (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    s += (d * d) as i64;
                }
                dc_cost[ci] = s as f64 + mlam * block_rate_bits(&ccf[ci], &SCAN_32X32);
                let a = cfl_best_alpha(&ac, &src, dc, 1024, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 1024];
                let mut resid = [0i32; 1024];
                for i in 0..1024 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_32X32, lam);
                let rr = idct_dequant_32x32(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..1024 {
                    let d = src[i] - (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_pred[ci] = cpr;
                cfl_cost[ci] = s2 as f64 + mlam * block_rate_bits(&q, &SCAN_32X32);
            }
        }
        // CfL signaling costs extra (sign + per-plane alpha); only use it when
        // it beats plain DC on both planes' summed cost by that overhead.
        let cfl_sig =
            4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
        // Gate CfL on low quality: at high quality DC prediction is accurate and CfL
        // alpha varying block-to-block creates visible colour stripes at boundaries.
        let use_cfl = acq > 300.0
            && (cfl_a[0] != 0 || cfl_a[1] != 0)
            && cfl_cost[0] + cfl_cost[1] + mlam * cfl_sig < dc_cost[0] + dc_cost[1];
        #[allow(unused_mut)] // cfl_opt mutated in 'sv block when SMOOTH_V wins
        let (cf_use, pred_dc, mut cfl_opt): (
            &[[i32; 1024]; 2],
            [i32; 2],
            Option<[i32; 2]>,
        ) = if use_cfl {
            (&cfl_ccf, cdc, Some(cfl_a))
        } else {
            (&ccf, cdc, None)
        };
        // SMOOTH_V check on 32x32 chroma (same principle as 16x16 path).
        #[allow(unused_mut)] // assigned via break in 'sv labeled block
        let mut cf_use_owned: [[i32; 1024]; 2];
        let mut sv_preds32 = [[0i32; 1024]; 2];
        let (final_cf, chosen_uv_32) = 'sv: {
            // Gate SMOOTH_V on low quality only
            if self.quant.ac_q() <= 300 {
                break 'sv (cf_use, DC_PRED);
            }
            let mut sv_ccf32 = [[0i32; 1024]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            let dcq2 = self.cquant.dc_q() as f64;
            let acq2 = self.cquant.ac_q() as f64;
            let lam2 = trellis_lambda();
            for ci in 0..2 {
                let plane = ci + 1;
                // sse_cur: raw source vs current winner prediction (DC scalar or CfL pixels)
                for ry in 0..32 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        let prow = &cfl_pred[ci][ry * 32..];
                        for (&sr, &pr) in srow[..32].iter().zip(prow[..32].iter()) {
                            let d = sr - pr;
                            sse_cur += (d * d) as i64;
                        }
                    } else {
                        let dc = pred_dc[ci];
                        for &s in srow[..32].iter() {
                            let d = s - dc;
                            sse_cur += (d * d) as i64;
                        }
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    false,
                    false,
                    self.w,
                    self.h,
                    &mut sv_preds32[ci],
                    self.bd,
                );
                let mut resid = [0i32; 1024];
                for (ry, drow) in resid.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds32[ci][ry * 32..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                sv_ccf32[ci] = q;
                trellis_optimize(&mut sv_ccf32[ci], &qt, dcq2, acq2, &SCAN_32X32, lam2);
                let mean_resid_sv = resid.iter().sum::<i32>() / 1024;
                if sv_ccf32[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    sv_ccf32[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
                for ry in 0..32 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds32[ci][ry * 32..];
                    for (&srow, &prow) in srow[..32].iter().zip(prow[..32].iter()) {
                        let d = srow - prow;
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv < sse_cur {
                cfl_opt = None; // SMOOTH_V overrides CfL if it wins
                cf_use_owned = sv_ccf32;
                break 'sv (&cf_use_owned, SMOOTH_V_PRED);
            }
            (cf_use, DC_PRED)
        };
        let block_skip =
            luma_zero && final_cf[0].iter().all(|&c| c == 0) && final_cf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv_32,
            cfl_opt,
            angle_delta,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let cres = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_32(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_32(plane, bx4, by4);
                encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, &final_cf[ci], true, sk, ds)
            };
            self.a_coef[plane][bx4..bx4 + 8].fill(cres);
            self.l_coef[plane][by4..by4 + 8].fill(cres);
            let crr = if block_skip {
                [0i32; 1024]
            } else {
                idct_dequant_32x32(&final_cf[ci], &self.cquant)
            };
            for (ry, rrow) in crr.as_chunks::<32>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if chosen_uv_32 == SMOOTH_V_PRED {
                    let prow = &sv_preds32[ci][ry * 32..];
                    for (j, (dv, &rv)) in drow[..32].iter_mut().zip(rrow.iter()).enumerate() {
                        *dv = (prow[j] + rv).clamp(0, (1 << self.bd) - 1);
                    }
                } else {
                    let base = if use_cfl {
                        cfl_pred[ci][ry * 32..][0]
                    } else {
                        pred_dc[ci]
                    };
                    for (dv, (&cp, &rv)) in drow[..32]
                        .iter_mut()
                        .zip(cfl_pred[ci][ry * 32..].iter().zip(rrow.iter()))
                    {
                        let b = if use_cfl { cp } else { base };
                        *dv = (b + rv).clamp(0, (1 << self.bd) - 1);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block32_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f64,
            self.cquant.ac_q() as f64,
            trellis_lambda(),
        );
        let maxval = (1 << self.bd) - 1;
        let mut ccf_dc = [[0i32; 256]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = dc_pred_16x16(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 256];
            for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
            ccf_dc[ci] = q;
            trellis_optimize(&mut ccf_dc[ci], &qt, dcq, acq, &SCAN_16X16, lam);
            let mean_resid_dc = resid.iter().sum::<i32>() / 256;
            if ccf_dc[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf_dc[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        let smooth_v_active_32 = false; // see note: chroma SMOOTH_V -> ADST_DCT not implemented; would desync decoder
        let mut ccf_sv = [[0i32; 256]; 2];
        let mut sv_preds = [[0i32; 256]; 2];
        if smooth_v_active_32 {
            for ci in 0..2 {
                let plane = ci + 1;
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    16,
                    16,
                    false,
                    false,
                    self.cw,
                    self.h,
                    &mut sv_preds[ci],
                    self.bd,
                );
                let mut resid = [0i32; 256];
                for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds[ci][ry * 16..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
                ccf_sv[ci] = q;
                trellis_optimize(&mut ccf_sv[ci], &qt, dcq, acq, &SCAN_16X16, lam);
                let mean_resid_sv = resid.iter().sum::<i32>() / 256;
                if ccf_sv[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    ccf_sv[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
            }
        } // end if smooth_v_active_32
        let mut rr_dc = [[0i32; 256]; 2];
        let mut rr_sv = [[0i32; 256]; 2];
        let mut sse_dc = 0i64;
        let mut sse_sv = 0i64;
        for ci in 0..2 {
            let plane = ci + 1;
            rr_dc[ci] = idct_dequant_16x16(&ccf_dc[ci], &self.cquant);
            rr_sv[ci] = idct_dequant_16x16(&ccf_sv[ci], &self.cquant);
            let dc = dc_preds[ci];
            for (ry, (rd_row, rs_row)) in rr_dc[ci]
                .as_chunks::<16>()
                .0
                .iter()
                .zip(rr_sv[ci].as_chunks::<16>().0.iter())
                .enumerate()
            {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                let prow = &sv_preds[ci][ry * 16..];
                for (((&s, &prow), &rd), &rs) in srow[..16]
                    .iter()
                    .zip(prow[..16].iter())
                    .zip(rd_row[..16].iter())
                    .zip(rs_row[..16].iter())
                {
                    let d = s - (dc + rd).clamp(0, maxval);
                    let v = s - (prow + rs).clamp(0, maxval);
                    sse_dc += (d * d) as i64;
                    sse_sv += (v * v) as i64;
                }
            }
        }
        let use_sv = smooth_v_active_32 && sse_sv <= sse_dc;
        let (chosen_uv, ccf, rr_cache) = if use_sv {
            (SMOOTH_V_PRED, ccf_sv, rr_sv)
        } else {
            (DC_PRED, ccf_dc, rr_dc)
        };
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv,
            None,
            angle_delta,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx_16(plane, bx4c, by4c);
                encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    0,
                    1,
                )
            };
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else {
                rr_cache[ci]
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                if use_sv {
                    let prow = &sv_preds[ci][ry * 16..];
                    for ((dv, &rv), &prow) in drow[..16]
                        .iter_mut()
                        .zip(rrow[..16].iter())
                        .zip(prow[..16].iter())
                    {
                        *dv = (prow + rv).clamp(0, maxval);
                    }
                } else {
                    let dc = dc_preds[ci];
                    for (dv, &rv) in drow[..16].iter_mut().zip(rrow.iter()) {
                        *dv = (dc + rv).clamp(0, maxval);
                    }
                }
            }
        }
    }

    /// 4:2:2: a 32x32 luma region maps to a 16-wide x 32-tall chroma block per
    /// plane (`RTX_16X32`, coef-CDF class 3). DC-pred chroma.
    #[allow(clippy::too_many_arguments)]
    fn code_block32_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
        angle_delta: i32,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        let (bx4c, by4c) = (cx / 4, py / 4);
        let mut ccf = [[0i32; 512]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_16x32(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 512];
            for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_16x32_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                &SCAN_16X32,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            DC_PRED,
            None,
            angle_delta,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x32_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_16x32_422(plane, bx4c, by4c);
                encode_16x32_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 8].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 512]
            } else {
                idct_dequant_16x32(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                }
            }
        }
    }
    /// Record a square luma block's size (in 4-sample units) into the `blk4`
    /// grid for every 4x4 luma unit it covers. Used by the deblocking filter to
    /// locate block edges and pick filter widths. `(x8,y8)` is the 8-sample
    /// origin; `dim4` is the block size in 4-units (2=8px, 4=16px, 8=32px).
    /// Record a coding block's skip flag into the per-8x8 CDEF map. `dim8` is the
    /// block side in 8-pixel units (8x8 -> 1, 16x16 -> 2, 32x32 -> 4).
    fn mark_skip8(&mut self, x8: usize, y8: usize, dim8: usize, skip: bool) {
        let sb8w = self.w.div_ceil(8);
        let sb8h = self.h.div_ceil(8);
        for ry in 0..dim8 {
            for rx in 0..dim8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    /// Rectangular per-8x8 skip mark: `w8` x `h8` 8x8 luma units.
    fn mark_skip8_rect(&mut self, x8: usize, y8: usize, w8: usize, h8: usize, skip: bool) {
        let sb8w = self.w.div_ceil(8);
        let sb8h = self.h.div_ceil(8);
        for ry in 0..h8 {
            for rx in 0..w8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    fn record_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        let nc4 = self.w / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        let d = dim4 as usize;
        let nr4 = self.h / 4;
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.blk4[r * nc4 + c] = dim4;
                self.blk4h[r * nc4 + c] = dim4;
            }
        }
    }

    /// Rectangular block record for deblocking. `w4`/`h4` are the block width and
    /// height in 4-sample units. The luma deblock derives vertical-edge spacing
    /// from the width map (`blk4`) and horizontal-edge spacing from the height
    /// map (`blk4h`), so a non-square block stores its true width and height
    /// separately — no longer the conservative min().
    fn record_blk_rect(&mut self, x8: usize, y8: usize, w4: u8, h4: u8) {
        let nc4 = self.w / 4;
        let nr4 = self.h / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        for r in by4..(by4 + h4 as usize).min(nr4) {
            for c in bx4..(bx4 + w4 as usize).min(nc4) {
                self.blk4[r * nc4 + c] = w4;
                self.blk4h[r * nc4 + c] = h4;
            }
        }
    }

    fn decode_sb(&mut self, bl: usize, x8: usize, y8: usize, sz8: usize, thr: bool, lhb: bool) {
        if sz8 == 1 {
            // BL_8X8 leaf (always fully in-frame for multiple-of-8 dimensions):
            // emit PARTITION_NONE, then the block. When the split scaffold is
            // forced (test-only), emit PARTITION_SPLIT and code four BLOCK_4X4.
            let ctx = get_partition_ctx(&self.a_part, &self.l_part, 4, x8, y8);
            let split_eligible = !self.ss422 && !self.mono;
            let want_split = split_eligible
                && (FORCE_SPLIT4.load(std::sync::atomic::Ordering::Relaxed)
                    || (SPLIT4_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
                        && !self.prefer_8x8_none(x8, y8)));
            if want_split {
                self.enc.encode_symbol(3, &mut self.cdfs.part_bl8[ctx]); // SPLIT
                self.code_block_split4_dc(x8, y8);
                self.a_part[x8] = 0x1f;
                self.l_part[y8] = 0x1f;
                return;
            }
            self.enc.encode_symbol(0, &mut self.cdfs.part_bl8[ctx]);
            let have_tr = thr && y8 > 0 && (x8 * 8 + 8) < self.w;
            let have_bl = lhb && x8 > 0 && (y8 * 8 + 8) < self.h;
            self.code_block(x8, y8, have_tr, have_bl);
            self.a_part[x8] = 0x1e;
            self.l_part[y8] = 0x1e;
            return;
        }
        // BL_32X32: optionally code the whole 32x32 as one TX_32X32 block
        // (PARTITION_NONE) instead of splitting into four 16x16. 4:4:4 only for
        // now (prefer_32x32 returns false otherwise). Requires the full 32x32
        // in-frame.
        if sz8 == 4 {
            let full_h = (x8 + 4) * 8 <= self.w;
            let full_v = (y8 + 4) * 8 <= self.h;
            if full_h && full_v && self.prefer_32x32(x8, y8) {
                let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                self.enc
                    .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]); // NONE
                let have_tr = thr && y8 > 0 && (x8 * 8 + 32) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 32) < self.h;
                self.code_block32(x8, y8, have_tr, have_bl);
                self.a_part[x8..x8 + 4].fill(0x18);
                self.l_part[y8..y8 + 4].fill(0x18);
                return;
            }
        }
        // BL_16X16: optionally code the whole 16x16 as one TX_16X16 block
        // (PARTITION_NONE) instead of splitting into four 8x8. Enabled for all
        // subsampling modes: 4:4:4 (chroma 16x16), 4:2:0 (chroma TX_8X8) and
        // 4:2:2 (chroma RTX_8X16). Requires the full 16x16 to be in-frame
        // (have_h && have_v at hh=1 guarantees it, since the coded frame is
        // 8-aligned and the test is strict).
        if sz8 == 2 {
            let have_h = (x8 + 1) * 8 < self.w;
            let have_v = (y8 + 1) * 8 < self.h;
            if have_h && have_v {
                // FORCE_HORZ (test) overrides the RD decision for 4:4:4.
                let forced_horz = !self.ss420
                    && !self.ss422
                    && !self.mono
                    && FORCE_HORZ.load(std::sync::atomic::Ordering::Relaxed);
                let choice = if forced_horz {
                    Part16::Horz
                } else {
                    self.partition_choice_16(x8, y8)
                };
                match choice {
                    Part16::Horz => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(1, &mut self.cdfs.part_split[bl - 1][ctx]); // HORZ
                        self.code_block16_horz_444(x8, y8);
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1e);
                        return;
                    }
                    Part16::None => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]); // NONE
                        let have_tr = thr && y8 > 0 && (x8 * 8 + 16) < self.w;
                        let have_bl = lhb && x8 > 0 && (y8 * 8 + 16) < self.h;
                        self.code_block16(x8, y8, have_tr, have_bl);
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1c);
                        return;
                    }
                    Part16::Split => { /* fall through to the SPLIT path below */ }
                }
            }
        }
        let hh = sz8 / 2;
        // content past the horizontal / vertical midpoint of this block?
        let have_h = (x8 + hh) * 8 < self.w;
        let have_v = (y8 + hh) * 8 < self.h;
        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
        if have_h && have_v {
            // full PARTITION_SPLIT symbol -> adapt the partition CDF (dav1d uses
            // decode_symbol_adapt here).
            self.enc
                .encode_symbol(3, &mut self.cdfs.part_split[bl - 1][ctx]);
        } else if have_h {
            // edge: dav1d codes a NON-adapting bool from a gathered probability;
            // read the live (possibly already-adapted) icdf CDF, do not adapt it.
            let p = gather_split_prob_icdf(&self.cdfs.part_split[bl - 1][ctx], true);
            self.enc.encode_bool(true, p);
        } else if have_v {
            let p = gather_split_prob_icdf(&self.cdfs.part_split[bl - 1][ctx], false);
            self.enc.encode_bool(true, p);
        }
        // else: neither -> implicit split, no symbol
        // recurse the children whose top-left is in-frame, propagating the
        // intra-edge availability (dav1d intra-edge tree). z-order child index
        // n: 0=TL,1=TR,2=BL,3=BR -> (top_has_right, left_has_bottom):
        //   TL=(1,1)  TR=(parent_thr,0)  BL=(1,parent_lhb)  BR=(0,0)
        let children = [
            (x8, y8, true, true),
            (x8 + hh, y8, thr, false),
            (x8, y8 + hh, true, lhb),
            (x8 + hh, y8 + hh, false, false),
        ];
        for (cx, cy, cthr, clhb) in children {
            if cx * 8 < self.w && cy * 8 < self.h {
                self.decode_sb(bl + 1, cx, cy, hh, cthr, clhb);
            }
        }
    }
}

/// Sum of squared error between the source and the reconstruction
/// `clamp(pred + residual)` over a `D`×`D` luma block at `(px,py)`. `N = D*D`.
/// Used by the partition R-D estimator to score a candidate block size.
#[inline]
fn sse_recon<const N: usize, const D: usize>(
    pred: &[i32; N],
    resid: &[i32; N],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    bd: u8,
) -> i64 {
    let maxv = (1 << bd) - 1;
    let mut sse = 0i64;
    for ry in 0..D {
        let srow = &src[(py + ry) * stride + px..];
        for c in 0..D {
            let r = (pred[ry * D + c] + resid[ry * D + c]).clamp(0, maxv);
            let d = (srow[c] - r) as i64;
            sse += d * d;
        }
    }
    sse
}

fn tx32_policy() -> u32 {
    2
}

/// Asymmetric-ADST tx-type search. A/B knob for the ADST_DCT/DCT_ADST trials;
/// enabled by default.
fn asym_adst_enabled() -> bool {
    true
}

fn angle_delta_enabled() -> bool {
    // Diagonal angle_delta winner-refinement search. Measured ~no-op on
    // photographic content (diagonals rarely win the mode search), so it is held
    // off until intra edge-filter/upsampling make directional modes competitive.
    false
}

/// Strength (and sign) of the variance-weighted "SSIM-style" RD adjustment.
/// The per-block rate weight is scaled by
/// `exp(K * (block_activity - tile_mean_activity))`, clamped to `[1/C, C]`:
///   K > 0  → busy blocks get a larger rate weight (fewer bits there — visual
///            masking hides the error), flat blocks more bits (aom `tune=ssim`);
///   K < 0  → the opposite (protect texture, spend more bits on busy blocks);
///   K = 0  → disabled (no change).
/// Disabled by default.
fn prdo_k() -> f64 {
    0.0
}

/// Clamp `C` for the perceptual RD scale: the per-block scale is limited to
/// `[1/C, C]` so no block is starved or flooded.
fn prdo_clamp() -> f64 {
    2.0
}

#[inline]
fn tx32_smooth_gate() -> i32 {
    LF_BAND_SMOOTH_RANGE
}

const LF_BAND_SMOOTH_RANGE: i32 = 32;

/// Extra rate (in bits) attributed to a PARTITION_SPLIT decision over
/// PARTITION_NONE in the R-D partition search. Splitting signals one partition
/// symbol at the parent plus four child partition symbols and four sets of
/// per-block mode/skip headers; this lumped constant biases the search toward
/// the larger block on a tie, matching how a full RDO would price the extra
/// syntax. Tuned conservatively — too low over-splits (bloats rate), too high
/// under-splits (blurs detail).
const SPLIT_SIGNAL_BITS: f64 = 24.0;
/// Minimum ac quantiser for the rectangular PARTITION_H candidate. Below this
/// (high quality) the DC-only 16x8 sub-blocks lose to the square path's full
/// mode search, so HORZ is gated off — libaom's Q-adaptive partition strategy.
const AC_Q_HORZ_MIN: i32 = 100;

pub(crate) fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Pad a `w`×`h` plane to `w8`×`h8` (≥ originals) by replicating the last
/// in-frame row/column. AV1's coded block grid is always 8-pixel aligned
/// (`MiCols = ((w+7)>>3)<<1`), so frames whose dimensions are not multiples of 8
/// are coded on the padded grid and the decoder crops back to the signaled
/// frame size. Edge replication keeps the (cropped-away) padding cheap to code.
pub(crate) fn pad_to_mult8<T: Copy>(src: &[T], w: usize, h: usize, w8: usize, h8: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w8 * h8);
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        out.extend_from_slice(row);
        out.resize(out.len() + (w8 - w), row[w - 1]);
    }
    for _ in h..h8 {
        out.extend_from_within((h - 1) * w8..h * w8);
    }
    out
}

/// Smallest `k` such that `(blk << k) >= target` (AV1 spec `tile_log2`).
fn tile_log2(blk: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

/// AV1 `increment_*_log2` bit sequence signalling `target` to a decoder that
/// starts at `min` and reads bits while its running value is `< max`: a `1` for
/// each step up, then a terminating `0` when `target < max` (at `max` the
/// decoder's loop ends on its own and reads no further bit).
fn increment_bits(min: u32, max: u32, target: u32) -> Vec<bool> {
    let mut v = Vec::new();
    let mut cur = min;
    while cur < max {
        if cur < target {
            v.push(true);
            cur += 1;
        } else {
            v.push(false);
            break;
        }
    }
    v
}

/// Full tiling decision: the chosen `(TileColsLog2, TileRowsLog2)` plus the
/// `increment_*_log2` bit sequences the frame header must emit to signal them.
struct Tiling {
    tcl: u32,
    trl: u32,
    cols_incr: Vec<bool>,
    rows_incr: Vec<bool>,
}

/// Pick a tiling for a frame of `sb_cols` x `sb_rows` superblocks. It is always
/// at least the spec **minimum** the decoder derives (so large frames stay
/// valid), and is subdivided further toward `target_tiles` so tile-level threads
/// have independent work. `target_tiles == 1` yields exactly the spec minimum —
/// a single tile for small frames, byte-identical to the untiled path. Extra
/// tiles trade a little compression (each tile resets entropy contexts and can't
/// predict across its edges) for parallelism, splitting the longer side first so
/// tiles stay roughly square.
fn plan_tiling(sb_cols: u32, sb_rows: u32, target_tiles: usize) -> Tiling {
    const MAX_TILE_WIDTH_SB: u32 = 4096 / 64; // 64
    const MAX_TILE_AREA_SB: u32 = (4096 * 2304) / (64 * 64); // 2304
    let min_log2_tile_cols = tile_log2(MAX_TILE_WIDTH_SB, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(MAX_TILE_AREA_SB, sb_rows * sb_cols));

    // Start at the spec minimum.
    let mut tcl = min_log2_tile_cols.min(max_log2_tile_cols);
    let mut trl = min_log2_tiles.saturating_sub(tcl).min(max_log2_tile_rows);

    // Climb toward target_tiles, splitting whichever side currently has the
    // larger tiles so the grid stays balanced.
    let target = target_tiles.max(1) as u32;
    while (1u32 << (tcl + trl)) < target {
        let can_col = tcl < max_log2_tile_cols;
        let can_row = trl < max_log2_tile_rows;
        if !can_col && !can_row {
            break;
        }
        let col_span = sb_cols >> tcl; // SBs per tile column (approx)
        let row_span = sb_rows >> trl; // SBs per tile row (approx)
        if can_col && (!can_row || col_span >= row_span) {
            tcl += 1;
        } else {
            trl += 1;
        }
    }

    let cols_incr = increment_bits(min_log2_tile_cols, max_log2_tile_cols, tcl);
    // The decoder derives its row minimum from the (now decoded) TileColsLog2.
    let min_log2_tile_rows = min_log2_tiles.saturating_sub(tcl);
    let rows_incr = increment_bits(min_log2_tile_rows, max_log2_tile_rows, trl);
    Tiling {
        tcl,
        trl,
        cols_incr,
        rows_incr,
    }
}

/// Uniform-spacing tile start offsets (in SB units), matching the decoder's
/// `for (startSb = 0; startSb < sbs; startSb += sizeSb)` loop. The returned vec
/// has one entry per tile; the implied end of tile `i` is `starts[i+1]` (or
/// `sbs` for the last). The tile count may be **less** than `1 << log2`.
fn tile_starts_sb(sbs: u32, log2: u32) -> Vec<u32> {
    let size_sb = sbs.div_ceil(1 << log2);
    let mut starts = Vec::new();
    let mut s = 0;
    while s < sbs {
        starts.push(s);
        s += size_sb;
    }
    starts
}

fn crop_plane<T: Copy>(
    src: &[T],
    full_w: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(tw * th);
    for r in 0..th {
        let s = (y0 + r) * full_w + x0;
        out.extend_from_slice(&src[s..s + tw]);
    }
    out
}

fn stitch_plane(
    dst: &mut [i32],
    full_w: usize,
    x0: usize,
    y0: usize,
    tile: &[i32],
    tw: usize,
    th: usize,
) {
    for r in 0..th {
        let d = (y0 + r) * full_w + x0;
        dst[d..d + tw].copy_from_slice(&tile[r * tw..(r + 1) * tw]);
    }
}

/// Pixel rectangle of one tile, in both luma and (subsampled) chroma coords.
#[derive(Clone, Copy)]
struct TileRect {
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
    cx0: usize,
    cy0: usize,
    ctw: usize,
    cth: usize,
}

/// Encoded output of one tile: its entropy-coded payload plus the tile-local
/// reconstruction (luma `tw*th`, chroma `ctw*cth`). Owned + `Send`, so it can be
/// produced on a worker thread and moved back to the caller.
struct TileOut {
    payload: Vec<u8>,
    recon: [Vec<i32>; 3],
    skip8: Vec<bool>, // per-8x8 luma-unit skip flag (tile-local, row-major over ceil(tw/8))
    blk4: Vec<u8>,    // per-4x4 luma block WIDTH map (tile-local), for frame-level deblocking
    blk4h: Vec<u8>,   // per-4x4 luma block HEIGHT map (tile-local), for frame-level deblocking
}

/// Encode a single tile as an independent sub-frame. Pure function of its inputs
/// (no shared mutable state), so it is safe to run on any thread. When `mono`,
/// only the luma plane is coded (`src[1]`/`src[2]` ignored, chroma recon empty).
#[allow(clippy::too_many_arguments)]
fn encode_one_tile(
    base_q_idx: u8,
    bd: u8,
    full_w: usize,
    full_h: usize,
    cw8: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    src: &[Vec<i32>; 3],
    r: &TileRect,
    speed: Speed,
    aq: bool,
    vb: &VarianceBoost,
    wiener: Option<crate::wiener::WienerUnit>,
) -> TileOut {
    let tsrc = if mono {
        [
            crop_plane(&src[0], full_w, r.x0, r.y0, r.tw, r.th),
            Vec::new(),
            Vec::new(),
        ]
    } else {
        [
            crop_plane(&src[0], full_w, r.x0, r.y0, r.tw, r.th),
            crop_plane(&src[1], cw8, r.cx0, r.cy0, r.ctw, r.cth),
            crop_plane(&src[2], cw8, r.cx0, r.cy0, r.ctw, r.cth),
        ]
    };
    let mut tile = if mono {
        LossyTile::new_mono(base_q_idx, bd, r.tw, r.th, &tsrc)
    } else {
        match (sub_x, sub_y) {
            (0, 0) => LossyTile::new(base_q_idx, bd, r.tw, r.th, &tsrc),
            (1, 0) => LossyTile::new_422(base_q_idx, bd, r.tw, r.th, &tsrc),
            _ => LossyTile::new_420(base_q_idx, bd, r.tw, r.th, &tsrc),
        }
    }
    .with_speed(speed);
    tile.wiener = wiener;
    // Loop restoration is frame-relative: record this tile's frame-absolute luma
    // origin and the full frame luma size so `read_lr` is computed in frame
    // coordinates regardless of tiling.
    tile.frame_x0 = r.x0;
    tile.frame_y0 = r.y0;
    tile.frame_w = full_w;
    tile.frame_h = full_h;
    if aq {
        // Center the per-SB deltas on this tile's mean activity so the average
        // quantizer tracks base_q_idx (zero-mean deltas => ~rate-neutral).
        let ref_act = tile_ref_activity(&tile.src[0], tile.w, tile.w, tile.h);
        tile.enable_aq(base_q_idx, ref_act, vb);
    }
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            tile.emit_lr_sb(sb_x, sb_y);
            tile.aq_begin_sb(sb_x, sb_y);
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    // NOTE: the in-loop deblocking filter is deliberately NOT applied here.
    // In AV1 the deblocking filter is a frame-level post-filter that operates
    // ACROSS tile boundaries (only entropy coding and prediction are
    // tile-independent). Applying it per tile leaves the inter-tile edges
    // unfiltered, diverging from the decoder at every tile boundary. The filter
    // is instead applied once on the stitched frame in `encode_lossy_tilegroup`.
    // Intra prediction already used the unfiltered recon during `decode_sb`, so
    // deferring the filter does not change any coding decision.
    let skip8 = tile.skip8;
    let blk4 = tile.blk4;
    let blk4h = tile.blk4h;
    let payload = tile.enc.done();
    TileOut {
        payload,
        recon: tile.recon,
        skip8,
        blk4,
        blk4h,
    }
}

/// Resolve the requested thread count: `0` => all available cores (fallback 1),
/// otherwise the value as-is. The caller still caps this at the tile count.
fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

/// Encode `src` (already padded to `w8` x `h8`, chroma subsampled by
/// `sub_x`/`sub_y`) as one or more AV1 tiles and return the **tile-group
/// payload** (everything that follows the frame header inside `OBU_FRAME`), the
/// stitched full-frame reconstruction, and the chosen `(TileColsLog2,
/// TileRowsLog2)`.
///
/// Each tile is encoded as an independent sub-frame: the source is cropped to
/// the tile's pixel rectangle and handed to a fresh [`LossyTile`] whose origin
/// is the tile's top-left, so tile boundaries become frame boundaries and all
/// the existing prediction/availability/context logic applies unchanged (intra
/// prediction and entropy contexts never cross a tile edge, as the spec
/// requires). For a single tile the payload is just that tile's bytes —
/// byte-identical to the previous single-tile path.
///
/// `threads` controls tile-level parallelism (AV1's natural parallel unit, since
/// tiles share no state): `1` runs serially (no threads spawned), `0` uses all
/// available cores, `N` uses up to `N`; the effective count is capped at the
/// number of tiles. The output is byte-identical regardless of `threads` — the
/// thread count only decides which core encodes which tile.
#[allow(clippy::too_many_arguments)]
fn encode_lossy_tilegroup(
    base_q_idx: u8,
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i32>; 3],
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: &VarianceBoost,
    cdef_on: bool,
    wiener_on: bool,
) -> (
    Vec<u8>,
    [Vec<i32>; 3],
    Tiling,
    Option<crate::obu::CdefParams>,
    Option<crate::obu::LrParams>,
) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;

    // Aim for ~one tile per worker so small frames can be paralleled too.
    // `threads == 1` -> target 1 -> spec-minimum tiling (single tile for small
    // frames, byte-identical to the untiled output).
    let want = resolve_threads(threads);
    let plan = plan_tiling(sb_cols, sb_rows, want);
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    let (cw8, ch8) = (w8 >> sub_x, h8 >> sub_y);

    // Tile rectangles in raster order (top-to-bottom, left-to-right).
    let mut rects: Vec<TileRect> = Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        let th = y1 - y0;
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            let tw = x1 - x0;
            rects.push(TileRect {
                x0,
                y0,
                tw,
                th,
                cx0: x0 >> sub_x,
                cy0: y0 >> sub_y,
                ctw: tw >> sub_x,
                cth: th >> sub_y,
            });
        }
    }

    let n = rects.len();
    let nthreads = want.clamp(1, n.max(1));

    // Encode every tile. Serial when a single thread (or single tile) is asked
    // for; otherwise split the tiles into disjoint chunks, one scoped thread per
    // chunk (no shared mutable state, so no locks and no `unsafe`).
    // `wiener_unit` controls whether each SB emits `read_lr` Wiener syntax; it is
    // `None` on the first pass (coefficients not yet known) and the chosen global
    // filter on the optional second pass. Because LR is a post-filter, the
    // reconstruction is identical regardless, so only the payloads differ.
    let encode_all = |wiener_unit: Option<crate::wiener::WienerUnit>| -> Vec<TileOut> {
        if nthreads <= 1 || n <= 1 {
            rects
                .iter()
                .map(|r| {
                    encode_one_tile(
                        base_q_idx,
                        bd,
                        w8,
                        h8,
                        cw8,
                        sub_x,
                        sub_y,
                        mono,
                        src,
                        r,
                        speed,
                        aq,
                        vb,
                        wiener_unit,
                    )
                })
                .collect()
        } else {
            let mut slots: Vec<Option<TileOut>> = (0..n).map(|_| None).collect();
            let chunk = n.div_ceil(nthreads);
            std::thread::scope(|scope| {
                for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                    scope.spawn(move || {
                        for (r, o) in rs.iter().zip(os.iter_mut()) {
                            *o = Some(encode_one_tile(
                                base_q_idx,
                                bd,
                                w8,
                                h8,
                                cw8,
                                sub_x,
                                sub_y,
                                mono,
                                src,
                                r,
                                speed,
                                aq,
                                vb,
                                wiener_unit,
                            ));
                        }
                    });
                }
            });
            slots.into_iter().map(|o| o.unwrap()).collect()
        }
    };
    let outs: Vec<TileOut> = encode_all(None);

    // Stitch reconstructions and collect payloads (raster order, serial).
    // Monochrome has only a luma plane; chroma recon stays empty.
    let mut recon = if mono {
        [vec![0i32; w8 * h8], Vec::new(), Vec::new()]
    } else {
        [
            vec![0i32; w8 * h8],
            vec![0i32; cw8 * ch8],
            vec![0i32; cw8 * ch8],
        ]
    };
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(n);
    let sb8w = w8.div_ceil(8);
    let sb8h = h8.div_ceil(8);
    let mut skip8 = vec![true; sb8w * sb8h];
    // Frame-level luma block-size map (4x4 units), assembled from every tile so
    // the deblocking filter can run on the stitched frame (across tile edges).
    let nc4f = w8 / 4;
    let nr4f = h8 / 4;
    let mut blk4f = vec![0u8; nc4f * nr4f];
    let mut blk4hf = vec![0u8; nc4f * nr4f];
    for (r, out) in rects.iter().zip(outs) {
        // stitch this tile's per-8x8 skip map into the frame map
        let tsb8w = r.tw.div_ceil(8);
        let (ox8, oy8) = (r.x0 / 8, r.y0 / 8);
        for ty in 0..r.th.div_ceil(8) {
            for tx in 0..tsb8w {
                let (fx, fy) = (ox8 + tx, oy8 + ty);
                if fx < sb8w && fy < sb8h {
                    skip8[fy * sb8w + fx] = out.skip8[ty * tsb8w + tx];
                }
            }
        }
        stitch_plane(&mut recon[0], w8, r.x0, r.y0, &out.recon[0], r.tw, r.th);
        // stitch this tile's per-4x4 luma block-size map into the frame map
        let tnc4 = r.tw / 4;
        let (ox4, oy4) = (r.x0 / 4, r.y0 / 4);
        for ty in 0..(r.th / 4) {
            for tx in 0..tnc4 {
                let (fx, fy) = (ox4 + tx, oy4 + ty);
                if fx < nc4f && fy < nr4f {
                    blk4f[fy * nc4f + fx] = out.blk4[ty * tnc4 + tx];
                    blk4hf[fy * nc4f + fx] = out.blk4h[ty * tnc4 + tx];
                }
            }
        }
        if !mono {
            stitch_plane(
                &mut recon[1],
                cw8,
                r.cx0,
                r.cy0,
                &out.recon[1],
                r.ctw,
                r.cth,
            );
            stitch_plane(
                &mut recon[2],
                cw8,
                r.cx0,
                r.cy0,
                &out.recon[2],
                r.ctw,
                r.cth,
            );
        }
        payloads.push(out.payload);
    }

    // Frame-level in-loop deblocking filter, applied once on the stitched
    // reconstruction so that inter-tile edges are filtered exactly as the
    // decoder does (deblocking is not tile-independent in AV1). `filter_plane`
    // is a no-op when the derived level is 0 (e.g. lossless).
    let (lvl_y, lvl_uv) = crate::obu::loop_filter_levels(base_q_idx);
    frame_deblock(
        &mut recon, w8, h8, cw8, ch8, &blk4f, &blk4hf, nc4f, sub_x, sub_y, mono, lvl_y, lvl_uv, bd,
    );

    // Frame-level CDEF, applied after deblocking exactly as the decoder does.
    // The encoder searches a global strength against the source (cheap SSE, not
    // RD) and applies the matching filter so the reconstruction stays in sync
    // with what the decoder will produce from the signalled cdef_params.
    let cdef = if cdef_on && base_q_idx != 0 {
        frame_cdef(
            &mut recon, src, &skip8, sb8w, w8, h8, cw8, ch8, sub_x, sub_y, mono, base_q_idx, bd,
        )
    } else {
        None
    };

    // Frame-level luma Wiener loop restoration, applied after CDEF (the last
    // in-loop filter). Works with any tiling: `read_lr` is signalled in frame
    // coordinates so each restoration unit is emitted exactly once by whichever
    // tile contains its top-left superblock. The encoder searches one global
    // Wiener filter against the source over the CDEF'd luma; if it helps, it
    // re-encodes the tiles so each superblock emits the `read_lr` syntax (the
    // reconstruction is unchanged because LR is a post-filter — only the payload
    // gains the symbols), then applies the same filter to the reconstruction so
    // it stays in sync with the decoder.
    let lr = if wiener_on && base_q_idx != 0 {
        if let Some(unit) = frame_wiener_search(&recon[0], &src[0], w8, h8, bd) {
            // Re-encode tiles emitting the LR syntax (recon is unchanged because
            // LR is a post-filter; only the payload gains the read_lr symbols).
            let outs2 = encode_all(Some(unit));
            payloads = outs2.into_iter().map(|o| o.payload).collect();
            // Apply the Wiener filter to the luma reconstruction in place.
            apply_frame_wiener(&mut recon[0], w8, h8, &unit, bd);
            Some(crate::obu::LrParams { luma_wiener: true })
        } else {
            None
        }
    } else {
        None
    };

    let tilegroup = assemble_tilegroup(payloads);
    (tilegroup, recon, plan, cdef, lr)
}

/// CDEF damping derived from the base quantizer (spec range 3..=6); higher q ->
/// stronger ringing -> a touch more damping. Kept simple and deterministic.
fn cdef_damping(base_q_idx: u8) -> u8 {
    3 + ((base_q_idx as u32) / 64).min(3) as u8
}

/// Apply a single global luma Wiener filter to the whole plane in place. Because
/// every restoration unit uses the same coefficients, the per-unit boundary
/// handling collapses to a single edge-clamped pass over the frame, matching the
/// decoder's result for this configuration.
fn apply_frame_wiener(
    plane: &mut [i32],
    w: usize,
    h: usize,
    unit: &crate::wiener::WienerUnit,
    bd: u8,
) {
    use crate::wiener::{WienerKernel, wiener_filter_plane};
    let hk = WienerKernel::from_coded(unit.h);
    let vk = WienerKernel::from_coded(unit.v);
    let src = plane.to_vec();
    wiener_filter_plane(plane, &src, w, h, &hk, &vk, bd);
}

/// Search a single global luma Wiener filter by SSE against the source over the
/// CDEF'd reconstruction. Returns the chosen `WienerUnit`, or `None` when no
/// candidate beats "no restoration" (so the caller signals RESTORE_NONE). This
/// is the cheap, not-RDO decision the project uses elsewhere: a real distortion
/// metric over a small, sane candidate set rather than a full Wiener solve.
fn frame_wiener_search(
    recon: &[i32],
    src: &[i32],
    w: usize,
    h: usize,
    bd: u8,
) -> Option<crate::wiener::WienerUnit> {
    use crate::wiener::{WienerKernel, wiener_filter_plane};
    let sse = |a: &[i32]| -> i64 {
        let mut s = 0i64;
        for i in 0..w * h {
            let d = (a[i] - src[i]) as i64;
            s += d * d;
        }
        s
    };
    let base = sse(recon);
    // Small candidate set of separable low-pass Wiener kernels (coded taps
    // [t0,t1,t2]); the identity is implicit via `base`. These span gentle to
    // moderate smoothing within the spec tap ranges.
    const CANDS: [[i32; 3]; 4] = [[0, 0, 1], [-1, 2, 2], [0, 1, 3], [1, -3, 5]];
    let mut best: Option<(i64, crate::wiener::WienerUnit)> = None;
    let mut tmp = recon.to_vec();
    for &h_taps in &CANDS {
        for &v_taps in &CANDS {
            let hk = WienerKernel::from_coded(h_taps);
            let vk = WienerKernel::from_coded(v_taps);
            wiener_filter_plane(&mut tmp, recon, w, h, &hk, &vk, bd);
            let s = sse(&tmp);
            if s < base && best.as_ref().map_or(true, |b| s < b.0) {
                best = Some((
                    s,
                    crate::wiener::WienerUnit {
                        h: h_taps,
                        v: v_taps,
                    },
                ));
            }
        }
    }
    best.map(|b| b.1)
}

/// Search a single global CDEF strength (luma + chroma) by SSE against the
/// source and apply it to the deblocked reconstruction in place. Returns the
/// `CdefParams` to signal (`cdef_bits = 0`, one strength entry), or `None` when
/// the best choice is "no filtering" (so the caller can leave CDEF effectively
/// off while keeping the headers consistent). This is the cheap, not-RDO
/// decision the brief asked for: a real distortion metric (SSE) over a small
/// candidate set, not a one-tap proxy.
#[allow(clippy::too_many_arguments)]
fn frame_cdef(
    recon: &mut [Vec<i32>; 3],
    src: &[Vec<i32>; 3],
    skip8: &[bool],
    sb8w: usize,
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    base_q_idx: u8,
    bd: u8,
) -> Option<crate::obu::CdefParams> {
    use crate::cdef;
    // The signalled cdef_damping is `damping - 3` (obu.rs); the decoder
    // reconstructs it and adds `bitdepth_min_8` (= bd - 8) before filtering, so
    // the encoder filters with `signalled + (bd - 8)`. Chroma further uses
    // `damping - 1`.
    let signalled_damping = cdef_damping(base_q_idx) as i32;
    let damping = signalled_damping + (bd as i32 - 8);

    // Precompute per-8x8 luma directions on the deblocked recon.
    let nbx = w8.div_ceil(8);
    let nby = h8.div_ceil(8);
    let mut ldirs = vec![0usize; nbx * nby];
    for by in 0..nby {
        for bx in 0..nbx {
            if bx * 8 < w8 && by * 8 < h8 {
                let (d, _) = cdef::cdef_direction(&recon[0], w8, bx * 8, by * 8, bd);
                ldirs[by * nbx + bx] = d;
            }
        }
    }

    // Per-8x8 luma skip map: CDEF is NOT applied to skip blocks (the decoder
    // leaves them untouched), so the encoder must mirror that exactly. `skip8`
    // is already a per-8x8 frame map.
    let lskip: Vec<bool> = (0..nbx * nby)
        .map(|i| {
            let (bx, by) = (i % nbx, i / nbx);
            skip8.get(by * sb8w + bx).copied().unwrap_or(true)
        })
        .collect();

    let luma_margin: i64 = if base_q_idx >= 180 { 12 } else { 22 };
    let (yp, ys) = cdef_search_plane(
        &recon[0],
        &src[0],
        w8,
        h8,
        &ldirs,
        &lskip,
        nbx,
        damping,
        bd,
        luma_margin,
        1000,
    );

    // --- Chroma strength search. ---
    // Chroma reuses the LUMA direction (remapped via uv_dir) and damping-1, and
    // is filtered at chroma sub-block granularity tied to the covering luma 8x8:
    // 8x8 for 4:4:4, 4x8 for 4:2:2, 4x4 for 4:2:0 — exactly as the decoders do
    // (dav1d cdef.fb[I444 - layout]). No per-block variance scaling for chroma.
    let uv_dir: [usize; 8] = if sub_x == 1 && sub_y == 0 {
        [7, 0, 2, 4, 5, 6, 6, 6] // 4:2:2
    } else {
        [0, 1, 2, 3, 4, 5, 6, 7] // 4:2:0 / 4:4:4 (identity)
    };
    let chroma_damping = damping - 1;
    let (up, us) = if mono {
        (0, 0)
    } else {
        cdef_search_chroma(
            &recon[1], &src[1], cw8, ch8, &ldirs, &uv_dir, skip8, sb8w, nbx, nby, sub_x, sub_y,
            chroma_damping, bd,
        )
    };

    if yp == 0 && ys == 0 && up == 0 && us == 0 {
        return None;
    }

    // Apply luma.
    apply_cdef_plane(
        &mut recon[0],
        w8,
        h8,
        &ldirs,
        &lskip,
        nbx,
        yp,
        ys,
        damping,
        bd,
    );
    // Apply chroma (U, V) at sub-block granularity with remapped luma directions.
    if !mono && (up != 0 || us != 0) {
        for plane in 1..3 {
            apply_cdef_chroma(
                &mut recon[plane], cw8, ch8, &ldirs, &uv_dir, skip8, sb8w, nbx, nby, sub_x, sub_y,
                up, us, chroma_damping, bd,
            );
        }
    }

    Some(crate::obu::CdefParams {
        bits: 0,
        damping: signalled_damping as u8,
        strengths: vec![(yp as u8, ys as u8, up as u8, us as u8)],
    })
}

#[allow(clippy::too_many_arguments)]
fn cdef_search_plane(
    recon: &[i32],
    src: &[i32],
    w: usize,
    h: usize,
    dirs: &[usize],
    skip: &[bool],
    nbx: usize,
    damping: i32,
    bd: u8,
    margin_num: i64,
    margin_den: i64,
) -> (i32, i32) {
    use crate::cdef;
    // First measure the no-filter SSE baseline.
    let mut off_sse = 0i64;
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            off_sse += plane_block_sse(recon, src, w, h, x, y);
        }
    }
    let threshold = off_sse - (off_sse.saturating_mul(margin_num) / margin_den.max(1));
    let mut best = (0i32, 0i32);
    let mut best_sse = off_sse;
    let mut tmp = recon.to_vec();
    for &pri in &cdef::PRI_CANDIDATES {
        for &sec in &cdef::SEC_CANDIDATES {
            if pri == 0 && sec == 0 {
                continue; // baseline already measured
            }
            let mut sse = 0i64;
            for y in (0..h).step_by(8) {
                for x in (0..w).step_by(8) {
                    let bxi = x / 8;
                    let byi = y / 8;
                    if skip.get(byi * nbx + bxi).copied().unwrap_or(true) {
                        sse += plane_block_sse(recon, src, w, h, x, y);
                        continue;
                    }
                    let _ = dirs;
                    let (dir, var) = cdef::cdef_direction(recon, w, x, y, bd);
                    // adjust_pri must be applied to the bit-depth-shifted strength
                    // (matches the decoders' adjust_strength, which scales the
                    // already-shifted level); scaling then shifting does not
                    // commute because of the `+8 >> 4` rounding.
                    let apri = cdef::adjust_pri(pri << (bd - 8), var);
                    cdef::cdef_filter_8x8(
                        &mut tmp,
                        recon,
                        w,
                        x,
                        y,
                        apri,
                        sec << (bd - 8),
                        dir,
                        damping,
                        bd,
                    );
                    sse += plane_block_sse(&tmp, src, w, h, x, y);
                }
            }
            // Must beat the current best AND clear the improvement margin vs off.
            if sse < best_sse && sse <= threshold {
                best_sse = sse;
                best = (pri, sec);
            }
        }
    }
    best
}

/// SSE over one (possibly edge-clipped) 8x8 block of a plane.
fn plane_block_sse(a: &[i32], b: &[i32], w: usize, h: usize, x: usize, y: usize) -> i64 {
    let mut s = 0i64;
    let yh = (y + 8).min(h);
    let xw = (x + 8).min(w);
    for yy in y..yh {
        for xx in x..xw {
            let d = (a[yy * w + xx] - b[yy * w + xx]) as i64;
            s += d * d;
        }
    }
    s
}

/// Apply a single global CDEF strength to a whole plane in place, reading from a
/// pre-CDEF snapshot so every 8x8 filters the same source pixels (the decoder
/// likewise reads the deblocked frame, not partially-CDEF'd pixels).
#[allow(clippy::too_many_arguments)]
fn apply_cdef_plane(
    plane: &mut [i32],
    w: usize,
    h: usize,
    dirs: &[usize],
    skip: &[bool],
    nbx: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) {
    use crate::cdef;
    let snapshot = plane.to_vec();
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let bxi = x / 8;
            let byi = y / 8;
            // Skip blocks are left untouched, matching the decoder.
            if skip.get(byi * nbx + bxi).copied().unwrap_or(true) {
                continue;
            }
            // Recompute the direction and variance from the (pre-CDEF) snapshot
            // so the primary strength can be scaled per block exactly as the
            // decoder does. `dirs` is ignored here in favour of the fresh value.
            let _ = dirs;
            let (dir, var) = cdef::cdef_direction(&snapshot, w, x, y, bd);
            // adjust_pri on the bit-depth-shifted strength (see search above).
            let apri = cdef::adjust_pri(pri << (bd - 8), var);
            cdef::cdef_filter_8x8(
                plane,
                &snapshot,
                w,
                x,
                y,
                apri,
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    }
}

/// Per-luma-8x8 chroma CDEF. For each non-skip luma 8x8 block, the covering
/// chroma sub-block — 8x8 (4:4:4), 4x8 (4:2:2) or 4x4 (4:2:0) — is filtered with
/// the remapped luma direction (`uv_dir[ldir]`) and `damping` (already the luma
/// damping minus one). Chroma never applies the variance-based strength scaling.
/// `skip8`/`sb8w` are the luma per-8x8 skip map; an 8x8 luma block that is skip
/// leaves its chroma untouched, matching the decoders' noskip_mask.
#[allow(clippy::too_many_arguments)]
fn apply_cdef_chroma(
    plane: &mut [i32],
    cw: usize,
    ch: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    skip8: &[bool],
    sb8w: usize,
    nbx: usize,
    nby: usize,
    sub_x: usize,
    sub_y: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) {
    use crate::cdef;
    let snapshot = plane.to_vec();
    let cbw = 8 >> sub_x; // chroma sub-block width per luma 8x8
    let cbh = 8 >> sub_y;
    for lby in 0..nby {
        for lbx in 0..nbx {
            // Skip if the covering luma 8x8 is a skip block.
            if skip8.get(lby * sb8w + lbx).copied().unwrap_or(true) {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            let cy = (lby * 8) >> sub_y;
            if cx >= cw || cy >= ch {
                continue;
            }
            let ld = ldirs.get(lby * nbx + lbx).copied().unwrap_or(0);
            let dir = uv_dir[ld];
            cdef::cdef_filter_block(
                plane,
                &snapshot,
                cw,
                cx,
                cy,
                cbw,
                cbh,
                pri << (bd - 8),
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    }
}

/// Chroma CDEF strength search mirroring `apply_cdef_chroma`: it evaluates each
/// (pri, sec) candidate by filtering every non-skip chroma sub-block against the
/// source and keeping the lowest-SSE strength.
#[allow(clippy::too_many_arguments)]
fn cdef_search_chroma(
    recon: &[i32],
    src: &[i32],
    cw: usize,
    ch: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    skip8: &[bool],
    sb8w: usize,
    nbx: usize,
    nby: usize,
    sub_x: usize,
    sub_y: usize,
    damping: i32,
    bd: u8,
) -> (i32, i32) {
    use crate::cdef;
    let cbw = 8 >> sub_x;
    let cbh = 8 >> sub_y;
    // baseline (no filter) SSE over the full plane.
    let mut off_sse = 0i64;
    for y in (0..ch).step_by(8) {
        for x in (0..cw).step_by(8) {
            off_sse += plane_block_sse(recon, src, cw, ch, x, y);
        }
    }
    let mut best = (0i32, 0i32);
    let mut best_sse = off_sse;
    let mut tmp = recon.to_vec();
    for &pri in &cdef::PRI_CANDIDATES {
        for &sec in &cdef::SEC_CANDIDATES {
            if pri == 0 && sec == 0 {
                continue;
            }
            // Reset the scratch buffer to the unfiltered recon, filter all
            // non-skip chroma sub-blocks, then measure SSE over the whole plane.
            tmp.copy_from_slice(recon);
            for lby in 0..nby {
                for lbx in 0..nbx {
                    if skip8.get(lby * sb8w + lbx).copied().unwrap_or(true) {
                        continue;
                    }
                    let cx = (lbx * 8) >> sub_x;
                    let cy = (lby * 8) >> sub_y;
                    if cx >= cw || cy >= ch {
                        continue;
                    }
                    let ld = ldirs.get(lby * nbx + lbx).copied().unwrap_or(0);
                    let dir = uv_dir[ld];
                    cdef::cdef_filter_block(
                        &mut tmp,
                        recon,
                        cw,
                        cx,
                        cy,
                        cbw,
                        cbh,
                        pri << (bd - 8),
                        sec << (bd - 8),
                        dir,
                        damping,
                        bd,
                    );
                }
            }
            let mut sse = 0i64;
            for y in (0..ch).step_by(8) {
                for x in (0..cw).step_by(8) {
                    sse += plane_block_sse(&tmp, src, cw, ch, x, y);
                }
            }
            if sse < best_sse {
                best_sse = sse;
                best = (pri, sec);
            }
        }
    }
    best
}


#[allow(clippy::too_many_arguments)]
fn frame_deblock(
    recon: &mut [Vec<i32>; 3],
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    blk4: &[u8],  // luma block width map (vertical edges)
    blk4h: &[u8], // luma block height map (horizontal edges)
    nc4: usize, // luma 4-col count == w8/4
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    level_y: i32,
    level_uv: i32,
    bd: u8,
) {
    if level_y > 0 {
        crate::loopfilter::filter_plane(
            &mut recon[0],
            w8,
            h8,
            blk4,
            blk4h,
            nc4,
            level_y,
            true,
            16, // 64px superblock -> 16 4-unit rows
            bd,
        );
    }
    if mono || level_uv <= 0 {
        return;
    }
    let ss_hor = sub_x;
    let ss_ver = sub_y;
    let cw = cw8;
    let ch = ch8;
    let cnc4 = cw / 4;
    let cnr4 = ch / 4;
    let mut cbw4 = vec![0u8; cnc4 * cnr4];
    let mut cbh4 = vec![0u8; cnc4 * cnr4];
    for cr in 0..cnr4 {
        for cc in 0..cnc4 {
            let lr = cr << ss_ver;
            let lc = cc << ss_hor;
            let dw = blk4[lr * nc4 + lc];
            let dh = blk4h[lr * nc4 + lc];
            cbw4[cr * cnc4 + cc] = (dw >> ss_hor).max(1);
            cbh4[cr * cnc4 + cc] = (dh >> ss_ver).max(1);
        }
    }
    let csb = 16 >> ss_ver;
    #[allow(clippy::needless_range_loop)]
    for plane in 1..3 {
        crate::loopfilter::filter_plane(
            &mut recon[plane],
            cw,
            ch,
            &cbw4,
            &cbh4,
            cnc4,
            level_uv,
            false,
            csb,
            bd,
        );
    }
}

/// Concatenate per-tile payloads into a tile-group. A single tile is returned
/// verbatim (no header byte, no size prefix). For `NumTiles > 1` the spec
/// `tile_group_obu` prepends `tile_start_and_end_present_flag = 0` followed by
/// `byte_alignment()` (one `0x00` byte), then every tile except the last is
/// prefixed with `tile_size_minus_1` as `TileSizeBytes = 4` little-endian bytes.
fn assemble_tilegroup(payloads: Vec<Vec<u8>>) -> Vec<u8> {
    if payloads.len() == 1 {
        return payloads.into_iter().next().unwrap();
    }
    let mut out = Vec::new();
    out.push(0u8);
    let last = payloads.len() - 1;
    for (i, p) in payloads.iter().enumerate() {
        if i != last {
            let sz_minus_1 = (p.len() - 1) as u32; // TileSizeBytes = 4
            out.extend_from_slice(&sz_minus_1.to_le_bytes());
        }
        out.extend_from_slice(p);
    }
    out
}

/// Build the frame OBU(s) that follow the sequence header. A single tile is
/// emitted as one combined `OBU_FRAME` (type 6) — byte-identical to the previous
/// output. Multi-tile frames are emitted as a separate `OBU_FRAME_HEADER`
/// (type 3) + `OBU_TILE_GROUP` (type 4), which strict parsers (ffmpeg's
/// `av1_frame_merge` BSF) handle reliably where a multi-tile combined
/// `OBU_FRAME` does not.
fn assemble_frame_obus(
    base_q_idx: u8,
    plan: &Tiling,
    tilegroup: &[u8],
    mono: bool,
    aq: bool,
    cdef: Option<&crate::obu::CdefParams>,
    lr: Option<&crate::obu::LrParams>,
) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = frame_header_lossy_multitile_th(
            base_q_idx,
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            mono,
            aq,
            cdef,
            lr,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = frame_header_lossy_multitile(
            base_q_idx,
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            mono,
            aq,
            cdef,
            lr,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Encode one lossless 4:4:4 tile: crop the three full-resolution planes to the
/// tile's pixel rect and hand them to `encode_tile_lossless` (whose origin is the
/// tile, so the tile's top/left behave as frame edges — intra prediction and
/// entropy never cross a tile boundary). Pure function of its inputs, so it runs
/// on any thread. Lossless recon equals the source, so no reconstruction is
/// returned. `r` is `(x0, y0, tw, th)` in pixels.
/// Encode a lossless 4:4:4 frame to its OBU frame portion (an `OBU_FRAME` for a
/// single tile, or `OBU_FRAME_HEADER` + `OBU_TILE_GROUP` for multiple). `src` are
/// the three full-resolution `w8*h8` planes, already padded to a multiple of 8.
/// Tiling is chosen automatically (at least the spec minimum, so large frames
/// are valid); `threads` parallelises across tiles (`1` = serial, byte-identical
/// to thz old single-tile output for small frames). The caller prepends the
/// temporal delimiter, sequence header and any metadata OBUs.
pub(crate) fn encode_lossless_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
) -> Vec<u8> {
    let (tilegroup, plan) = encode_lossless_tilegroup(bd, w8, h8, src, threads);
    assemble_lossless_frame_obus(&plan, &tilegroup)
}

fn encode_one_lossless_tile(
    bd: u8,
    full_w: usize,
    src: &[Vec<i16>; 3],
    r: &(usize, usize, usize, usize),
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(&src[0], full_w, x0, y0, tw, th);
    let p1 = crop_plane(&src[1], full_w, x0, y0, tw, th);
    let p2 = crop_plane(&src[2], full_w, x0, y0, tw, th);
    crate::av1_tile::encode_tile_lossless(tw, th, bd, [&p0, &p1, &p2])
}

/// Encode a **lossless** 4:4:4 frame as a (possibly multi-tile) tile group,
/// mirroring the lossy tiling path. The frame is split into at least the spec
/// minimum tiling — so frames wider than 4096px or larger than the max tile area
/// stay valid (the previous single-tile lossless path mis-signalled these) — and
/// further toward `threads` tiles for parallelism. Each tile is encoded
/// independently; `threads` parallelises across tiles with scoped threads (no
/// shared mutable state, no locks). `threads == 1` yields the spec minimum: a
/// single tile for small frames, byte-identical to the untiled path. The output
/// is byte-identical regardless of thread count for a fixed tiling.
fn encode_lossless_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = resolve_threads(threads);
    let plan = plan_tiling(sb_cols, sb_rows, want);
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
    let nthreads = want.clamp(1, n.max(1));
    let payloads: Vec<Vec<u8>> = if nthreads <= 1 || n <= 1 {
        rects
            .iter()
            .map(|r| encode_one_lossless_tile(bd, w8, src, r))
            .collect()
    } else {
        let mut slots: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (r, o) in rs.iter().zip(os.iter_mut()) {
                        *o = Some(encode_one_lossless_tile(bd, w8, src, r));
                    }
                });
            }
        });
        slots.into_iter().map(|o| o.unwrap()).collect()
    };

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a lossless tile group with the matching frame header: a single tile uses
/// a combined `OBU_FRAME` (type 6); multiple tiles use a separate
/// `OBU_FRAME_HEADER` (type 3) + `OBU_TILE_GROUP` (type 4), the layout strict
/// parsers (ffmpeg's cbs_av1) accept.
fn assemble_lossless_frame_obus(plan: &Tiling, tilegroup: &[u8]) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh =
            crate::obu::frame_header_lossless_multitile(&plan.cols_incr, &plan.rows_incr, 0, 0);
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Crop the single luma plane to a tile rect and encode it as a mono lossless
/// tile. Pure function of its inputs (safe on any thread).
fn encode_one_lossless_tile_mono(
    bd: u8,
    full_w: usize,
    luma: &[i16],
    r: &(usize, usize, usize, usize),
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(luma, full_w, x0, y0, tw, th);
    crate::av1_tile::encode_tile_lossless_mono(tw, th, bd, &p0)
}

/// Monochrome counterpart of [`encode_lossless_tilegroup`]: a single full-res
/// luma plane (`w8*h8`, padded to a multiple of 8) tiled identically to the
/// 4:4:4 path. Byte-identical output for a fixed tiling regardless of thread
/// count.
fn encode_lossless_mono_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    luma: &[i16],
    threads: usize,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = resolve_threads(threads);
    let plan = plan_tiling(sb_cols, sb_rows, want);
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
    let nthreads = want.clamp(1, n.max(1));
    let payloads: Vec<Vec<u8>> = if nthreads <= 1 || n <= 1 {
        rects
            .iter()
            .map(|r| encode_one_lossless_tile_mono(bd, w8, luma, r))
            .collect()
    } else {
        let mut slots: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (r, o) in rs.iter().zip(os.iter_mut()) {
                        *o = Some(encode_one_lossless_tile_mono(bd, w8, luma, r));
                    }
                });
            }
        });
        slots.into_iter().map(|o| o.unwrap()).collect()
    };

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a mono lossless tile group with a `mono_chrome = 1` lossless frame
/// header (single tile ⇒ combined `OBU_FRAME`; multi-tile ⇒ `OBU_FRAME_HEADER` +
/// `OBU_TILE_GROUP`).
fn assemble_lossless_mono_frame_obus(plan: &Tiling, tilegroup: &[u8]) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_mono_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = crate::obu::frame_header_lossless_mono_multitile(
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Encode a monochrome lossless frame's OBU portion from a padded `w8*h8` luma
/// plane. Caller prepends temporal delimiter, sequence header, metadata.
pub(crate) fn encode_lossless_mono_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    luma: &[i16],
    threads: usize,
) -> Vec<u8> {
    let (tilegroup, plan) = encode_lossless_mono_tilegroup(bd, w8, h8, luma, threads);
    assemble_lossless_mono_frame_obus(&plan, &tilegroup)
}

/// Encode a full monochrome **lossless** AV1 still image: temporal delimiter +
/// monochrome sequence header (`mono_chrome = 1`) + lossless frame. `luma` is
/// `w*h` samples; it is padded to a multiple of 8 internally. Profile 0 for
/// 8/10-bit, profile 2 for 12-bit.
pub(crate) fn encode_av1_mono_lossless_image(
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i16],
    full_range: bool,
    threads: usize,
) -> Vec<u8> {
    assert_eq!(luma.len(), w * h, "luma plane must be w*h");
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let padded = pad_to_mult8(luma, w, h, w8, h8);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_mono(
        w as u32, h as u32, bd, full_range,
    ));
    bytes.extend_from_slice(&encode_lossless_mono_frame_obus(
        bd, w8, h8, &padded, threads,
    ));
    bytes
}

/// Lossy encoder with explicit color mode: `ycbcr=false` signals MC_IDENTITY
/// (planes coded as GBR); `ycbcr=true` signals full-range BT.601 so the decoder
/// converts the coded Y/Cb/Cr planes back to RGB (decorrelated -> smaller).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_cs(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Vec<u8> {
    encode_av1_lossy_image_cs_recon_dbg(
        base_q_idx, bd, w, h, luma, u, v, color, threads, speed, aq, &vb, cdef, wiener,
    )
    .0
}

/// Debug variant of [`encode_av1_lossy_image_cs`] (4:4:4) returning the padded
/// reconstruction and dims, for bit-exactness verification.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_cs_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: &VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> (Vec<u8>, [Vec<i32>; 3], (usize, usize)) {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let src = [
        pad_to_mult8(luma, w, h, w8, h8),
        pad_to_mult8(u, w, h, w8, h8),
        pad_to_mult8(v, w, h, w8, h8),
    ];
    let (payload, recon, plan, cdefp, lrp) = encode_lossy_tilegroup(
        base_q_idx, bd, w8, h8, &src, 0, 0, false, threads, speed, aq, vb, cdef, wiener,
    );
    let profile = if bd == 12 { 2 } else { 1 };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp(
        w as u32, h as u32, profile, bd, color,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(
        base_q_idx,
        &plan,
        &payload,
        false,
        aq,
        cdefp.as_ref(),
        lrp.as_ref(),
    ));
    (bytes, recon, (w8, h8))
}

/// Encode a **lossy 4:2:2** YCbCr still (profile 2). `luma` is `w*h`; `u`/`v`
/// are the horizontally-subsampled chroma planes, each `cw*h` with
/// `cw = (w+1)/2`. The decoder reconstructs full-resolution RGB via the
/// signalled BT.601 matrix and 4:2:2 upsampling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_422(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Vec<u8> {
    encode_av1_lossy_image_422_recon_dbg(
        base_q_idx, bd, w, h, luma, u, v, color, threads, speed, aq, vb, cdef, wiener,
    )
    .0
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn encode_av1_lossy_image_422_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> (Vec<u8>, [Vec<i32>; 3], (usize, usize, usize)) {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let cw = w.div_ceil(2);
    assert_eq!(u.len(), cw * h);
    assert_eq!(v.len(), cw * h);
    let (w8, h8) = (align8(w), align8(h));
    let cw8 = w8 / 2;
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| -> Vec<i32> { pad_to_mult8(p, cw, h, cw8, h8) };
    let src = [luma_p, pad_c(u), pad_c(v)];
    let (payload, recon, plan, cdefp, lrp) = encode_lossy_tilegroup(
        base_q_idx, bd, w8, h8, &src, 1, 0, false, threads, speed, aq, &vb, cdef, wiener,
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp_ss(
        w as u32, h as u32, 2, bd, color, 1, 0,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(
        base_q_idx,
        &plan,
        &payload,
        false,
        aq,
        cdefp.as_ref(),
        lrp.as_ref(),
    ));
    (bytes, recon, (w8, h8, cw8))
}

/// Encode a **lossy 4:2:0** YCbCr still (profile 0). `luma` is `w*h`; `u`/`v`
/// are the half-width, half-height chroma planes, each `cw*ch` with
/// `cw=(w+1)/2`, `ch=(h+1)/2`. Each 8x8 luma block carries a 4x4 (`TX_4X4`)
/// chroma block per plane. Reconstruction is bit-exact vs dav1d 1.4.1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_420(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Vec<u8> {
    encode_av1_lossy_image_420_recon_dbg(
        base_q_idx, bd, w, h, luma, u, v, color, threads, speed, aq, vb, cdef, wiener,
    )
    .0
}

/// Debug variant of [`encode_av1_lossy_image_420`] also returning the encoder's
/// padded reconstruction `[Y, U, V]` and the padded dims `(w8, h8, cw8, ch8)`,
/// for bit-exactness verification against the decoder.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn encode_av1_lossy_image_420_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> (Vec<u8>, [Vec<i32>; 3], (usize, usize, usize, usize)) {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    assert_eq!(u.len(), cw * ch);
    assert_eq!(v.len(), cw * ch);
    let (w8, h8) = (align8(w), align8(h));
    let (cw8, ch8) = (w8 / 2, h8 / 2);
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| -> Vec<i32> { pad_to_mult8(p, cw, ch, cw8, ch8) };
    let src = [luma_p, pad_c(u), pad_c(v)];
    let (payload, recon, plan, cdefp, lrp) = encode_lossy_tilegroup(
        base_q_idx, bd, w8, h8, &src, 1, 1, false, threads, speed, aq, &vb, cdef, wiener,
    );
    let profile = if bd == 12 { 2 } else { 0 };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp_ss(
        w as u32, h as u32, profile, bd, color, 1, 1,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(
        base_q_idx,
        &plan,
        &payload,
        false,
        aq,
        cdefp.as_ref(),
        lrp.as_ref(),
    ));
    (bytes, recon, (w8, h8, cw8, ch8))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_mono_image(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Vec<u8> {
    let (bytes, _recon, _w8, _h8) = encode_av1_mono_image_recon_dbg(
        base_q_idx, bd, w, h, luma, full_range, threads, speed, aq, vb, cdef, wiener,
    );
    bytes
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_mono_image_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> (Vec<u8>, Vec<i32>, usize, usize) {
    assert_eq!(luma.len(), w * h, "luma plane must be w*h");
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let src = [pad_to_mult8(luma, w, h, w8, h8), Vec::new(), Vec::new()];
    let (payload, recon, plan, cdefp, lrp) = encode_lossy_tilegroup(
        base_q_idx, bd, w8, h8, &src, 0, 0, true, threads, speed, aq, &vb, cdef, wiener,
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_mono(
        w as u32, h as u32, bd, full_range,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(
        base_q_idx,
        &plan,
        &payload,
        true,
        aq,
        cdefp.as_ref(),
        lrp.as_ref(),
    ));
    let [luma_recon, _, _] = recon;
    (bytes, luma_recon, w8, h8)
}
