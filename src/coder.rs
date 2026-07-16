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
use crate::chroma_rect::*;
use crate::dct::{
    adst4x4_t, adst4x8_t, adst8x8_t, adst16x16_t, adstdct4x4_t, adstdct4x8_t, adstdct8x8_t,
    adstdct16x16_t, dct4x8_t, dct8x4_t, dct8x8_t, dct8x16_t, dct16x8_t, dct16x32_t, dct32x16_t,
    dctadst4x4_t, dctadst4x8_t, dctadst8x8_t, dctadst16x16_t, fidentity8x8_t,
};
use crate::idct::{
    iadst_dequant_4x4, iadst_dequant_4x8, iadst_dequant_8x8, iadst_dequant_16x16,
    iadstdct_dequant_4x4, iadstdct_dequant_4x8, iadstdct_dequant_8x8, iadstdct_dequant_16x16,
    idct_dequant_4x4, idct_dequant_4x8, idct_dequant_8x4, idct_dequant_8x8, idct_dequant_8x16,
    idct_dequant_16x8, idct_dequant_16x16, idct_dequant_16x32, idct_dequant_32x16,
    idct_dequant_32x32, idctadst_dequant_4x4, idctadst_dequant_4x8, idctadst_dequant_8x8,
    idctadst_dequant_16x16, iidentity_dequant_8x8,
};
use crate::obu::{
    frame_header_lossy_multitile, frame_header_lossy_multitile_th, wrap_obu_frame,
    wrap_obu_frame_split,
};
use crate::odec::OdEcEncoder;
use crate::par::Pool;
use crate::quant::QmLevels;
#[cfg(test)]
pub(crate) static FORCE_SPLIT4: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(not(test))]
pub(crate) static FORCE_SPLIT4: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) static SPLIT4_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

pub(crate) static FORCE_HORZ: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub static HORZ_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub static VERT_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub(crate) static TUNE_SSIMULACRA2: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Partition decision for a 16x16 luma region.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part16 {
    None,
    Horz,
    Vert,
    Split,
    HorzA,
    HorzB,
    VertA,
    VertB,
}
use crate::aq_common::DarkAq;
use crate::trellis::{trellis_optimize, trellis_optimize_ctx};

use crate::coeffs::encode_tx16_coeffs_adapt;
use crate::coeffs::*;
use crate::cost::*;
use crate::intrapred::*;
use crate::quant::*;
use crate::tables::*;
use crate::util::FastRound;

/// AV1 chroma transform type derived from the intra mode (`Mode_To_Txfm`)
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
    // Mirrors dav1d_txtp_from_uvmode: the decoder derives the chroma transform
    // type from the uv_mode. (At tx sizes whose square is >= TX_32X32 the spec
    // forces DCT_DCT regardless; callers at those sizes ignore this and use DCT.)
    match mode {
        // ADST_ADST: SMOOTH, PAETH, D135 (DIAG_DOWN_RIGHT)
        m if m == PAETH_PRED || m == SMOOTH_PRED || m == D135_PRED => ChromaTx::AdstAdst,
        // ADST_DCT: SMOOTH_V, V, D113 (VERT_RIGHT), D67 (VERT_LEFT)
        m if m == SMOOTH_V_PRED || m == V_PRED || m == D113_PRED || m == VERT_LEFT_PRED => {
            ChromaTx::AdstDct
        }
        // DCT_ADST: SMOOTH_H, H, D157 (HOR_DOWN), D203 (HOR_UP)
        m if m == SMOOTH_H_PRED || m == H_PRED || m == D157_PRED || m == D203_PRED => {
            ChromaTx::DctAdst
        }
        // DCT_DCT: DC, D45 (DIAG_DOWN_LEFT), and anything else.
        _ => ChromaTx::DctDct,
    }
}

/// Forward transform + trellis quant for an 8x8 chroma block under the given
/// chroma tx kind. Returns levels + unrounded targets like the other `*_t`.
fn fwd_chroma_8x8(tx: ChromaTx, resid: &[i32; 64], q: &impl Dct) -> ([i32; 64], [f32; 64]) {
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

/// Forward transform + trellis quant for a 16x16 chroma block under the given
/// chroma tx kind (mirrors `fwd_chroma_8x8` at TX_16X16).
fn fwd_chroma_16x16(tx: ChromaTx, resid: &[i32; 256], q: &impl Dct) -> ([i32; 256], [f32; 256]) {
    match tx {
        ChromaTx::DctDct => forward_dct_quant_16x16_t(resid, q),
        ChromaTx::AdstAdst => adst16x16_t(resid, q),
        ChromaTx::AdstDct => adstdct16x16_t(resid, q),
        ChromaTx::DctAdst => dctadst16x16_t(resid, q),
    }
}

fn inv_chroma_16x16(tx: ChromaTx, levels: &[i32; 256], q: &impl Dct) -> [i32; 256] {
    match tx {
        ChromaTx::DctDct => idct_dequant_16x16(levels, q),
        ChromaTx::AdstAdst => iadst_dequant_16x16(levels, q),
        ChromaTx::AdstDct => iadstdct_dequant_16x16(levels, q),
        ChromaTx::DctAdst => idctadst_dequant_16x16(levels, q),
    }
}

fn fwd_chroma_4x4(tx: ChromaTx, resid: &[i32; 16], q: &impl Dct) -> ([i32; 16], [f32; 16]) {
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

fn fwd_chroma_4x8(tx: ChromaTx, resid: &[i32; 32], q: &impl Dct) -> ([i32; 32], [f32; 32]) {
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
    pub(crate) eob_bin_32_c: Vec<u16>,
    pub(crate) eob_bin_32_l: Vec<u16>,
    pub(crate) eob_bin_64_l: Vec<u16>,   // luma, 8x8
    pub(crate) eob_bin_64_c: Vec<u16>,   // chroma, 8x8
    pub(crate) eob_bin_256_l: Vec<u16>,  // luma, 16x16 (class 2)
    pub(crate) eob_bin_256_c: Vec<u16>,  // chroma, 16x16 (class 2)
    pub(crate) eob_bin_128_c: Vec<u16>,  // chroma, RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_128_l: Vec<u16>,  // luma, RTX_16X8/RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_1024_l: Vec<u16>, // luma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_1024_c: Vec<u16>, // chroma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_512_c: Vec<u16>,
    pub(crate) eob_bin_512_l: Vec<u16>,
    pub(crate) delta_q: Vec<u16>, // superblock delta-q magnitude (4 symbols)
    pub(crate) wiener_restore: Vec<u16>, // use_wiener flag (2-symbol)
}

impl Cdfs {
    pub(crate) fn new(qctx: usize) -> Self {
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
            eob_bin_32_l: icdf(&Q::EOB_BIN_32_LUMA[qctx]),
            eob_bin_64_l: icdf(&Q::EOB_BIN_64_LUMA[qctx]),
            eob_bin_64_c: icdf(&Q::EOB_BIN_64_CHROMA[qctx]),
            eob_bin_256_l: icdf(&Q::EOB_BIN_256_LUMA[qctx]),
            eob_bin_256_c: icdf(&Q::EOB_BIN_256_CHROMA[qctx]),
            eob_bin_128_c: icdf(&Q::EOB_BIN_128_CHROMA[qctx]),
            eob_bin_128_l: icdf(&Q::EOB_BIN_128_LUMA[qctx]),
            eob_bin_1024_l: icdf(&Q::EOB_BIN_1024_LUMA[qctx]),
            eob_bin_1024_c: icdf(&Q::EOB_BIN_1024_CHROMA[qctx]),
            eob_bin_512_c: icdf(&Q::EOB_BIN_512_CHROMA[qctx]),
            eob_bin_512_l: icdf(&Q::EOB_BIN_512_LUMA[qctx]),
            // AV1 Default_Delta_Q_Cdf = AOM_CDF4(28160, 32120, 32677); a single
            // (context-free) 4-symbol CDF for the delta-q magnitude token. Adapts
            // like every other symbol via OdEcEncoder::encode_symbol.
            delta_q: icdf(&[28160, 32120, 32677]),
            // Default LrWiener (use_wiener) CDF (AV1 Default_Wiener_Restore_Cdf).
            wiener_restore: wiener_restore_icdf(),
        }
    }
}

/// log-resolution of the delta-q step: the signaled `delta_q_res` is `1 << this`
/// (so a step of 4 qindex units). Matches `av2/aq.rs::AQ_RES_LOG2`.
const AQ_RES_LOG2: u8 = 2;
/// Same value, exposed for the frame-header writer (`delta_q_res`) so the
/// signaled resolution always matches the per-SB step used by the encoder.
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
    /// Dark-structured-detail protection, combined with the variance boost by `max`.
    /// Independent of `enabled`: fires in both the Variance Boost and classic-AQ paths.
    pub dark: DarkAq,
    pub qm: QmLevels,
}

impl VarianceBoost {
    /// Disabled — classic whole-SB AQ, byte-identical to the pre-VB encoder.
    pub(crate) fn off() -> Self {
        VarianceBoost {
            enabled: false,
            octile: 6,
            strength: 1.0,
            boost_only: false,
            dark: DarkAq::off(),
            qm: QmLevels::FLAT,
        }
    }

    pub(crate) fn on() -> Self {
        VarianceBoost {
            enabled: true,
            octile: 6,
            strength: 0.6,
            boost_only: true,
            dark: DarkAq::on(),
            qm: QmLevels::FLAT,
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
    let n = (h * w) as f32;
    let mean = sum as f32 / n;
    let var = (sum2 as f32 / n - mean * mean).max(0.0);
    (1.0 + var).ln()
}

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
    (base_q + delta.fast_round() as i32).clamp(1, 255)
}

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
    let mut acc = 0f32;
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
            let n = (h * w) as f32;
            let mean = sum as f32 / n;
            let var = (sum2 as f32 / n - mean * mean).max(0.0);
            out[idx] = var;
            acc += var;
            filled += 1;
        }
    }
    if filled == 0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    let mean = acc / filled as f32;
    for v in out.iter_mut() {
        if v.is_nan() {
            *v = mean;
        }
    }
    filled
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
    /// decoder `CurrentQIndex`, updated by each signaled delta; reset to `base_q`
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
    /// Dark-structured-detail protection (see [`DarkAq`]); independent of `vb_enabled`.
    dark: DarkAq,
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
            dark: DarkAq::off(),
        }
    }
}

/// Whole-frame lossy encoder state.
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
    a_uv_mode: Vec<u8>,   // chroma intra mode above each luma 4x4 column
    l_uv_mode: Vec<u8>,   // chroma intra mode left of each luma 4x4 row
    blk4: Vec<u8>, // luma block WIDTH (in 4-sample units) per 4x4 luma unit; for the deblock filter (vertical edges)
    blk4h: Vec<u8>, // luma block HEIGHT (in 4-sample units) per 4x4 luma unit; for the deblock filter (horizontal edges)
    blk4v: Vec<bool>, // true where a luma block starts at this 4x4 column
    blk4t: Vec<bool>, // true where a luma block starts at this 4x4 row
    skip8: Vec<bool>, // per-8x8-luma-unit block skip flag (true = no coded coeffs); for CDEF
    /// Whether the current superblock already recorded its `read_cdef()` trace
    /// point (the first non-skip block carries the per-unit `cdef_idx`).
    cdef_point_marked: bool,
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
    /// Base quant index this tile was built with. Stored so the R-D search can
    /// apply libaom's SSIMULACRA2 rdmult weight (see `cost::mode_lambda_aom` /
    /// `aom_ssimulacra2_rdmult_weight`), which is a function of qindex.
    base_q_idx: u8,
}

// Keep the state type and shared imports in this module while splitting its
// implementation by coding responsibility. `include!` preserves private field
// access without widening the encoder's internal API.
include!("coder/lossy_state.rs");
include!("coder/partition_search.rs");
include!("coder/block16.rs");
include!("coder/block8.rs");
include!("coder/block32.rs");
include!("coder/superblock.rs");

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
    debug_assert_eq!(N, D * D);
    crate::rd_sse::sse_recon(pred, resid, src, stride, px, py, D, D, bd)
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
    true
}

const DIRECTIONAL_RDO_TOP_K: usize = 3;

#[derive(Clone, Copy)]
struct DirectionalTopK {
    modes: [usize; DIRECTIONAL_RDO_TOP_K],
    costs: [u64; DIRECTIONAL_RDO_TOP_K],
    len: usize,
}

impl DirectionalTopK {
    #[inline]
    fn new() -> Self {
        Self {
            modes: [usize::MAX; DIRECTIONAL_RDO_TOP_K],
            costs: [u64::MAX; DIRECTIONAL_RDO_TOP_K],
            len: 0,
        }
    }

    #[inline]
    fn insert(&mut self, mode: usize, cost: u64) {
        let mut pos = self.len.min(DIRECTIONAL_RDO_TOP_K - 1);
        if self.len == DIRECTIONAL_RDO_TOP_K && cost >= self.costs[pos] {
            return;
        }
        if self.len < DIRECTIONAL_RDO_TOP_K {
            self.len += 1;
            pos = self.len - 1;
        }
        while pos > 0 && cost < self.costs[pos - 1] {
            self.costs[pos] = self.costs[pos - 1];
            self.modes[pos] = self.modes[pos - 1];
            pos -= 1;
        }
        self.costs[pos] = cost;
        self.modes[pos] = mode;
    }

    #[inline]
    fn contains(&self, mode: usize) -> bool {
        self.modes[..self.len].contains(&mode)
    }
}

#[inline]
fn is_directional_mode(mode: usize) -> bool {
    (V_PRED..=VERT_LEFT_PRED).contains(&mode)
}

/// Cheap first-stage directional ranking. SATD estimates transform-domain
/// sparsity while SAD prevents a transform-friendly but visibly biased predictor
/// from ranking too highly. The 4x4 Hadamard sum is normalized to SAD scale.
fn satd_sad_proxy(
    src: &[i32],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    #[inline]
    fn had4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
        let (e, f, g, h) = (a + c, a - c, b + d, b - d);
        [e + g, f + h, f - h, e - g]
    }

    debug_assert_eq!(w & 3, 0);
    debug_assert_eq!(h & 3, 0);
    let mut sad = 0u64;
    let mut satd = 0u64;
    for ty in (0..h).step_by(4) {
        for tx in (0..w).step_by(4) {
            let mut rows = [[0i32; 4]; 4];
            for r in 0..4 {
                let sr = &src[(ty + r) * src_stride + tx..];
                let pr = &pred[(ty + r) * pred_stride + tx..];
                let d: [i32; 4] = std::array::from_fn(|x| sr[x] - pr[x]);
                sad += d.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>();
                rows[r] = had4(d[0], d[1], d[2], d[3]);
            }
            #[allow(clippy::needless_range_loop)]
            for x in 0..4 {
                let col = had4(rows[0][x], rows[1][x], rows[2][x], rows[3][x]);
                satd += col.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>();
            }
        }
    }
    sad + (satd >> 2)
}

/// Strength (and sign) of the variance-weighted "SSIM-style" RD adjustment.
/// The per-block rate weight is scaled by
/// `exp(K * (block_activity - tile_mean_activity))`, clamped to `[1/C, C]`:
///   K > 0  → busy blocks get a larger rate weight (fewer bits there — visual
///            masking hides the error), flat blocks more bits (aom `tune=ssim`);
///   K < 0  → the opposite (protect texture, spend more bits on busy blocks);
///   K = 0  → disabled (no change).
/// Disabled by default.
fn prdo_k() -> f32 {
    0.0
}

/// Clamp `C` for the perceptual RD scale: the per-block scale is limited to
/// `[1/C, C]` so no block is starved or flooded.
fn prdo_clamp() -> f32 {
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
const SPLIT_SIGNAL_BITS: f32 = 24.0;
const ASYM_PART_SIGNAL_BITS: f32 = SPLIT_SIGNAL_BITS;
/// Extra uncertainty charge for A partitions whose final rectangular leaf
/// predicts from siblings that have not yet been reconstructed during the
/// lightweight partition RDO. This keeps marginal wins from exploiting stale
/// edge pixels while preserving candidates with a clear distortion advantage.
const ASYM_DEPENDENT_RDO_BITS: f32 = 32.0;
/// Minimum ac quantizer for the rectangular PARTITION_H candidate. Below this
/// (high quality) the DC-only 16x8 sub-blocks lose to the square path's full
/// mode search, so HORZ is gated off — libaom's Q-adaptive partition strategy.
const AC_Q_HORZ_MIN: i32 = 100;
/// Unpruned 32x32 rectangle search is a strong medium/low-quality win but can
/// regress fine high-quality texture. Keep it to the measured coarse-q regime.
const UNPRUNED_RECT32_MIN_QINDEX: u8 = 128;
/// Weight of chroma in shared-tree partition R-D. Raw U/V SSE is too strong
/// relative to luma for perceptual RGB quality, especially after subsampling;
/// one eighth keeps color edges relevant without letting them dominate.
const CHROMA_PART_RD_WEIGHT: f32 = 0.125;

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
pub(crate) struct Tiling {
    tcl: u32,
    trl: u32,
    cols_incr: Vec<bool>,
    rows_incr: Vec<bool>,
}

/// Pick a tiling for a frame of `sb_cols` x `sb_rows` superblocks.
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

/// Default (untrained) inverse CDF for the `use_wiener` flag, shared by tile
/// entropy init and the LR replay path.
pub(crate) fn wiener_restore_icdf() -> Vec<u16> {
    icdf(&[11570])
}

/// `read_lr` symbols owed by the superblock at tile-local `(sb_x, sb_y)` (spec
/// 5.11.57); geometry rationale on `LossyTile::emit_lr_sb`.
#[allow(clippy::too_many_arguments)]
fn emit_lr_sb_syms(
    enc: &mut OdEcEncoder,
    wr_cdf: &mut [u16],
    lr_ref_v: &mut [i32; 3],
    lr_ref_h: &mut [i32; 3],
    unit: &crate::wiener::WienerUnit,
    frame_x0: usize,
    frame_y0: usize,
    frame_w: usize,
    frame_h: usize,
    sb_x: usize,
    sb_y: usize,
) {
    const UNIT: usize = 64;
    const MI: usize = 4;
    let count_units = |frame: usize| -> usize { (1).max((frame + (UNIT >> 1)) / UNIT) };
    let unit_rows = count_units(frame_h);
    let unit_cols = count_units(frame_w);
    // Frame-absolute superblock position in 4x4 MI units (luma).
    let r = (frame_y0 + sb_y) / MI;
    let c = (frame_x0 + sb_x) / MI;
    let sb_mi = UNIT / MI; // 16
    let urs = (r * MI).div_ceil(UNIT);
    let ure = unit_rows.min(((r + sb_mi) * MI).div_ceil(UNIT));
    let ucs = (c * MI).div_ceil(UNIT);
    let uce = unit_cols.min(((c + sb_mi) * MI).div_ceil(UNIT));
    for _ur in urs..ure {
        for _uc in ucs..uce {
            emit_lr_unit_syms(enc, wr_cdf, lr_ref_v, lr_ref_h, unit);
        }
    }
}

/// One `read_lr_unit` for a RESTORE_WIENER luma unit (spec 5.11.58): `use_wiener`,
/// then v/h taps signed-subexp coded against (and updating) the running refs.
fn emit_lr_unit_syms(
    enc: &mut OdEcEncoder,
    wr_cdf: &mut [u16],
    lr_ref_v: &mut [i32; 3],
    lr_ref_h: &mut [i32; 3],
    unit: &crate::wiener::WienerUnit,
) {
    use crate::wiener::{WIENER_TAPS_K, WIENER_TAPS_MAX, WIENER_TAPS_MIN};
    enc.encode_symbol(1, wr_cdf);
    for axis in 0..2 {
        let (taps, refs) = if axis == 0 {
            (unit.v, &mut *lr_ref_v)
        } else {
            (unit.h, &mut *lr_ref_h)
        };
        for j in 0..3usize {
            let lo = WIENER_TAPS_MIN[j];
            let hi = WIENER_TAPS_MAX[j] + 1; // exclusive high
            let k = WIENER_TAPS_K[j] as u32;
            enc.encode_signed_subexp_with_ref(taps[j], lo, hi, k, refs[j]);
            refs[j] = taps[j];
        }
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
    trace: Option<Box<crate::odec::SymbolTrace>>,
    recon: [Vec<i32>; 3],
    skip8: Vec<bool>, // per-8x8 luma-unit skip flag (tile-local, row-major over ceil(tw/8))
    blk4: Vec<u8>,    // per-4x4 luma block WIDTH map (tile-local), for frame-level deblocking
    blk4h: Vec<u8>,   // per-4x4 luma block HEIGHT map (tile-local), for frame-level deblocking
    blk4v: Vec<bool>, // per-4x4 actual luma vertical-edge map
    blk4t: Vec<bool>, // per-4x4 actual luma horizontal-edge map
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
    record: bool,
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
        LossyTile::new_mono(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm)
    } else {
        match (sub_x, sub_y) {
            (0, 0) => LossyTile::new(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm),
            (1, 0) => LossyTile::new_422(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm),
            _ => LossyTile::new_420(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm),
        }
    }
    .with_speed(speed);
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
    if record {
        tile.enc.begin_trace();
    }
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            // The mark sits exactly where a replay would interleave the LR
            // symbols owed by this superblock (`emit_lr_sb` is a no-op here).
            tile.enc.trace_mark();
            tile.cdef_point_marked = false;
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
    let blk4v = tile.blk4v;
    let blk4t = tile.blk4t;
    let trace = tile.enc.take_trace();
    let payload = tile.enc.done();
    TileOut {
        payload,
        trace,
        recon: tile.recon,
        skip8,
        blk4,
        blk4h,
        blk4v,
        blk4t,
    }
}

/// Per-64x64-unit CDEF replay context: the frame-level on/off grid plus its
/// column count. Each SB with a recorded `read_cdef()` point gets its unit's
/// index inserted there as a raw `L(1)` literal.
struct CdefReplay<'a> {
    grid: &'a [u8],
    unit_cols: usize,
    /// cdef_bits: width of the per-unit raw index literal (1..=3).
    bits: u32,
}

/// Replay a tile's recorded symbols with the frame-level filter syntax
/// interleaved: the Wiener `read_lr` symbols at each SB start (when `lr` is
/// set) and the per-unit `cdef_idx` literal at each SB's `read_cdef()` point
/// (when `cdef` is set). LR touches only its own CDF + raw bits and `cdef_idx`
/// is an equiprobable literal, so this is byte-identical to a re-encode.
fn replay_tile_with_filters(
    r: &TileRect,
    trace: &crate::odec::SymbolTrace,
    lr: Option<&crate::wiener::WienerUnit>,
    cdef: Option<&CdefReplay>,
    frame_w: usize,
    frame_h: usize,
) -> Vec<u8> {
    let mut enc = OdEcEncoder::new();
    let mut wr_cdf = wiener_restore_icdf();
    let mut lr_ref_v = crate::wiener::WIENER_TAPS_MID;
    let mut lr_ref_h = crate::wiener::WIENER_TAPS_MID;
    let mut i = 0usize;
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            if let Some(unit) = lr {
                emit_lr_sb_syms(
                    &mut enc,
                    &mut wr_cdf,
                    &mut lr_ref_v,
                    &mut lr_ref_h,
                    unit,
                    r.x0,
                    r.y0,
                    frame_w,
                    frame_h,
                    sb_x,
                    sb_y,
                );
            }
            match cdef {
                Some(c) => {
                    let (pre, post) = trace.sb_ops_split(i);
                    enc.replay(pre);
                    if let Some(post) = post {
                        // This SB has a non-skip block: its 64x64 unit carries a
                        // cdef_idx literal at the recorded read_cdef() point.
                        let u = ((r.y0 + sb_y) / 64) * c.unit_cols + (r.x0 + sb_x) / 64;
                        let idx = c.grid.get(u).copied().unwrap_or(0);
                        enc.encode_literal(idx as u32, c.bits);
                        enc.replay(post);
                    }
                }
                None => enc.replay(trace.sb_ops(i)),
            }
            i += 1;
        }
    }
    debug_assert_eq!(i, trace.sb_count(), "trace/SB iteration mismatch");
    enc.done()
}

/// Resolve the requested thread count: `0` => all available cores (fallback 1),
/// otherwise the value as-is. The caller still caps this at the tile count.
pub(crate) fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_tilegroup(
    base_q_idx: u8,
    bd: u8,
    w8: usize,
    h8: usize,
    disp_w: usize,
    disp_h: usize,
    src: &[Vec<i32>; 3],
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    vb: &VarianceBoost,
    cdef_on: bool,
    wiener_on: bool,
) -> (
    Vec<u8>,
    Tiling,
    Option<crate::obu::CdefParams>,
    Option<crate::obu::LrParams>,
) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;

    // Aim for ~one tile per worker so small frames can be paralleled too.
    // `threads == 1` -> target 1 -> spec-minimum tiling (single tile for small
    // frames, byte-identical to the untiled output).
    let want = pool.width();
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

    // Recording the symbol trace lets a winning Wiener unit or a per-unit CDEF
    // grid be signaled by a cheap replay instead of a second full encode of
    // every tile.
    let record = (wiener_on || cdef_on) && base_q_idx != 0;
    let mut outs: Vec<TileOut> = pool.map_indexed(nthreads, n, |i| {
        encode_one_tile(
            base_q_idx, bd, w8, h8, cw8, sub_x, sub_y, mono, src, &rects[i], speed, aq, vb, record,
        )
    });

    let mut payloads: Vec<Vec<u8>> = outs
        .iter_mut()
        .map(|o| std::mem::take(&mut o.payload))
        .collect();
    let traces: Vec<_> = outs.iter_mut().map(|o| o.trace.take()).collect();

    // Small per-8x8 / per-4x4 maps: stitched serially (they are tiny).
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
    let sb8w = w8.div_ceil(8);
    let sb8h = h8.div_ceil(8);
    let mut skip8 = vec![true; sb8w * sb8h];
    // Frame-level luma block-size map (4x4 units), assembled from every tile so
    // the deblocking filter can run on the stitched frame (across tile edges).
    let nc4f = w8 / 4;
    let nr4f = h8 / 4;
    let mut blk4f = vec![0u8; nc4f * nr4f];
    let mut blk4hf = vec![0u8; nc4f * nr4f];
    let mut blk4vf = vec![false; nc4f * nr4f];
    let mut blk4tf = vec![false; nc4f * nr4f];
    for (r, out) in rects.iter().zip(outs.iter()) {
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
        let tnc4 = r.tw / 4;
        let (ox4, oy4) = (r.x0 / 4, r.y0 / 4);
        for ty in 0..(r.th / 4) {
            for tx in 0..tnc4 {
                let (fx, fy) = (ox4 + tx, oy4 + ty);
                if fx < nc4f && fy < nr4f {
                    blk4f[fy * nc4f + fx] = out.blk4[ty * tnc4 + tx];
                    blk4hf[fy * nc4f + fx] = out.blk4h[ty * tnc4 + tx];
                    blk4vf[fy * nc4f + fx] = out.blk4v[ty * tnc4 + tx];
                    blk4tf[fy * nc4f + fx] = out.blk4t[ty * tnc4 + tx];
                }
            }
        }
    }

    // Pixel planes: every tile row owns a disjoint horizontal band of each
    // plane, so (plane, tile row) pairs stitch in parallel.
    let ncols = col_starts.len();
    {
        let mut items: Vec<(usize, usize, &mut [i32])> = Vec::new();
        for (pl, plane) in recon.iter_mut().enumerate() {
            if plane.is_empty() {
                continue;
            }
            let pw = if pl == 0 { w8 } else { cw8 };
            let mut rest = &mut plane[..];
            let mut consumed = 0usize;
            for ti in 0..row_starts.len() {
                let r0 = &rects[ti * ncols];
                let (py0, pth) = if pl == 0 {
                    (r0.y0, r0.th)
                } else {
                    (r0.cy0, r0.cth)
                };
                debug_assert_eq!(consumed, py0 * pw);
                let (band, r2) = std::mem::take(&mut rest).split_at_mut(pth * pw);
                rest = r2;
                consumed += band.len();
                items.push((pl, ti, band));
            }
        }
        pool.for_each(nthreads, items, |(pl, ti, band)| {
            for (r, out) in rects[ti * ncols..(ti + 1) * ncols]
                .iter()
                .zip(&outs[ti * ncols..(ti + 1) * ncols])
            {
                let (pw, px0, ptw, pth) = if pl == 0 {
                    (w8, r.x0, r.tw, r.th)
                } else {
                    (cw8, r.cx0, r.ctw, r.cth)
                };
                stitch_plane(band, pw, px0, 0, &out.recon[pl], ptw, pth);
            }
        });
    }
    drop(outs);

    // Frame-level in-loop deblocking filter, applied once on the stitched
    // reconstruction so that inter-tile edges are filtered exactly as the
    // decoder does (deblocking is not tile-independent in AV1). `filter_plane`
    // is a no-op when the derived level is 0 (e.g. lossless).
    let (lvl_y, lvl_uv) = crate::obu::loop_filter_levels(base_q_idx);
    frame_deblock(
        &mut recon, w8, h8, cw8, ch8, disp_w, disp_h, &blk4f, &blk4hf, &blk4vf, &blk4tf, nc4f,
        sub_x, sub_y, mono, lvl_y, lvl_uv, bd,
    );

    // Frame-level CDEF (R-D searched; may pick per-64x64-unit signaling).
    let cdef_decision = if cdef_on && base_q_idx != 0 {
        frame_cdef(
            &mut recon, src, &skip8, sb8w, w8, h8, cw8, ch8, disp_w, disp_h, sub_x, sub_y, mono,
            base_q_idx, bd, speed, pool,
        )
    } else {
        None
    };

    // Frame-level luma Wiener loop restoration (searched on the CDEF-filtered
    // recon, matching the decoder's filter order).
    let lr_unit = if wiener_on && base_q_idx != 0 {
        frame_wiener_search(&recon[0], &src[0], w8, h8, bd, pool)
    } else {
        None
    };

    // Signal the winning filter syntax by replaying each tile's recorded
    // symbols with the read_lr symbols / per-unit cdef_idx literals
    // interleaved (byte-identical to a full re-encode; recon is unchanged
    // because both only add their own symbols).
    let cdef_replay = cdef_decision
        .as_ref()
        .filter(|d| d.params.bits > 0)
        .map(|d| CdefReplay {
            grid: &d.grid,
            unit_cols: d.unit_cols,
            bits: d.params.bits as u32,
        });
    if lr_unit.is_some() || cdef_replay.is_some() {
        payloads = pool.map_indexed(nthreads, n, |i| {
            let trace = traces[i]
                .as_deref()
                .expect("trace recorded for filter replay");
            replay_tile_with_filters(
                &rects[i],
                trace,
                lr_unit.as_ref(),
                cdef_replay.as_ref(),
                w8,
                h8,
            )
        });
    }
    let lr = lr_unit.map(|_| crate::obu::LrParams { luma_wiener: true });

    // Conformance oracle: dump the final (deblock+CDEF) reconstruction so an
    // external decode (avifdec --raw y4m) can be byte-compared against it.
    // Planes are raw little-endian u16 at padded dims, Y then U then V.
    // Only meaningful with Wiener off (LR is not applied to this recon).
    if let Ok(path) = std::env::var("MT_AV1_DUMP_RECON") {
        let mut buf: Vec<u8> = Vec::new();
        for (pl, plane) in recon.iter().enumerate() {
            if mono && pl > 0 {
                break;
            }
            for &v in plane {
                buf.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        let _ = std::fs::write(path, buf);
    }

    let tilegroup = assemble_tilegroup(payloads);
    (tilegroup, plan, cdef_decision.map(|d| d.params), lr)
}

/// CDEF damping derived from the base quantizer (spec range 3..=6); higher q ->
/// stronger ringing -> a touch more damping. Kept simple and deterministic.
fn cdef_damping(base_q_idx: u8) -> u8 {
    3 + ((base_q_idx as u32) / 64).min(3) as u8
}

fn frame_wiener_search(
    recon: &[i32],
    src: &[i32],
    w: usize,
    h: usize,
    bd: u8,
    pool: &Pool,
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
    let cands: Vec<(&[i32; 3], &[i32; 3])> = CANDS
        .iter()
        .flat_map(|h_taps| CANDS.iter().map(move |v_taps| (h_taps, v_taps)))
        .collect();
    // Each candidate filters into its own buffer; the reduce below walks the
    // original (h, v) order so ties break exactly as the sequential loop did.
    let want = pool.width().min(cands.len());
    let sses: Vec<i64> = pool.map_indexed(want, cands.len(), |i| {
        let (h_taps, v_taps) = cands[i];
        let hk = WienerKernel::from_coded(*h_taps);
        let vk = WienerKernel::from_coded(*v_taps);
        let mut tmp = vec![0i32; w * h];
        wiener_filter_plane(&mut tmp, recon, w, h, &hk, &vk, bd);
        sse(&tmp)
    });
    let mut best: Option<(i64, crate::wiener::WienerUnit)> = None;
    for (&(h_taps, v_taps), &s) in cands.iter().zip(sses.iter()) {
        if s < base && best.as_ref().is_none_or(|b| s < b.0) {
            best = Some((
                s,
                crate::wiener::WienerUnit {
                    h: *h_taps,
                    v: *v_taps,
                },
            ));
        }
    }
    best.map(|b| b.1)
}

/// Default directional-variance gate threshold (see `frame_cdef`).
const UNIT_DIR_VAR_THRESH_DEFAULT: i64 = 15000;
/// Default per-mille per-unit margin (see `frame_cdef`).
const MARGIN_DEFAULT: i64 = 22;

/// Frame CDEF decision: header params plus, when `params.bits == 1`, the
/// per-64x64-unit on/off grid (frame unit raster order; 1 = index 1 = filtered).
/// Units whose 8x8 blocks are all skip carry no `cdef_idx` and are never
/// filtered; their grid entry stays 0.
pub(crate) struct CdefFrameDecision {
    pub(crate) params: crate::obu::CdefParams,
    pub(crate) grid: Vec<u8>,
    pub(crate) unit_cols: usize,
}

/// R-D CDEF search: one searched active strength (luma + chroma), then an exact
/// rate-distortion choice between OFF, a global strength (`cdef_bits = 0`), and
/// per-64x64-unit signaling (`cdef_bits = 1`, 2-entry table `[(0,0),(y,uv)]`,
/// one raw `cdef_idx` bit at each non-all-skip unit).
///
/// Distortion is libaom's perceptual `cdef_dist_8x8` for luma (variance-masked
/// SSE — the metric libaom tuned CDEF against, aligned with SSIMULACRA2) and
/// plain SSE for chroma. Rate is exact: the per-unit index is an equiprobable
/// literal (1 bit) and the second strength-table entry costs 12 header bits
/// (6 when monochrome); both indices of a signaled unit cost the same bit, so
/// the per-unit on/off choice reduces to min distortion while the rate only
/// arbitrates OFF vs global vs per-unit. lambda uses the same libaom-shaped
/// `mode_lambda_q(dc_q)` the mode search runs on, keeping CDEF on the encoder's
/// single (SSE, bits) axis.
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
    disp_w: usize,
    disp_h: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    base_q_idx: u8,
    bd: u8,
    speed: Speed,
    pool: &Pool,
) -> Option<CdefFrameDecision> {
    use crate::cdef;
    let signaled_damping = cdef_damping(base_q_idx) as i32;
    let damping = signaled_damping + (bd as i32 - 8);

    // The decoder reconstructs and reads REAL pixels across the whole
    // mi-aligned (coded) area — CDEF taps see the coded padding columns/rows as
    // ordinary neighbors, and only samples beyond the mi extent are
    // CDEF_VERY_LARGE (handled by the stride bounds in `sample`). The padding is
    // never deblocked (see `filter_plane`'s visible clip), so the snapshot below
    // matches the decoder's pre-CDEF frame everywhere. Distortion, however, is
    // measured over the VISIBLE window only — invisible pixels must not drive
    // the decision.
    let (cw_vis, ch_vis) = (disp_w.div_ceil(1 << sub_x), disp_h.div_ceil(1 << sub_y));
    let snap_y = recon[0].clone();

    let nbx = w8.div_ceil(8);
    let nby = h8.div_ceil(8);
    let mut ldirs = vec![0usize; nbx * nby];
    let mut lvars = vec![0i32; nbx * nby];
    {
        // Directions read the decoder's frame buffer, which contains REAL
        // reconstructed pixels across the whole mi-aligned (coded) area — the
        // padding columns/rows are coded and reconstructed, only CDEF taps
        // beyond them are masked. So the direction search uses the unmarked
        // recon.
        let luma = &recon[0];
        #[allow(clippy::type_complexity)]
        let items: Vec<(usize, (&mut [usize], &mut [i32]))> = ldirs
            .chunks_mut(nbx)
            .zip(lvars.chunks_mut(nbx))
            .enumerate()
            .collect();
        pool.for_each(pool.width(), items, |(by, (drow, vrow))| {
            for bx in 0..nbx {
                if bx * 8 < w8 && by * 8 < h8 {
                    let (d, v) = cdef::cdef_direction(luma, w8, bx * 8, by * 8, bd);
                    drow[bx] = d;
                    vrow[bx] = v;
                }
            }
        });
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

    // 64x64 CDEF unit grid; a unit is "signaled" (carries a cdef_idx literal)
    // iff it contains at least one non-skip 8x8.
    let uc = w8.div_ceil(64);
    let ur = h8.div_ceil(64);
    let n_units = uc * ur;
    let mut signaled = vec![false; n_units];
    for by in 0..nby {
        for bx in 0..nbx {
            if !lskip[by * nbx + bx] {
                signaled[(by / 8) * uc + bx / 8] = true;
            }
        }
    }
    let n_sig = signaled.iter().filter(|&&s| s).count();
    if n_sig == 0 {
        return None;
    }

    // Decision knobs, env-overridable while the RD recipe is being tuned:
    //   MT_CDEF_GATE   directional-variance gate threshold (0 disables)
    //   MT_CDEF_METRIC "sse" for plain SSE, default perceptual cdef_dist
    //   MT_CDEF_MARGIN per-mille distortion margin a unit must clear to filter
    let env_i64 = |k: &str, d: i64| -> i64 {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let gate_thresh = env_i64("MT_CDEF_GATE", UNIT_DIR_VAR_THRESH_DEFAULT);
    let perceptual = std::env::var("MT_CDEF_METRIC").map_or(true, |v| v != "sse");
    let margin = env_i64("MT_CDEF_MARGIN", MARGIN_DEFAULT);

    // Directional-variance gate for the GLOBAL (filter-everything) option only:
    // that mode has no per-unit off-switch, so it is offered solely when most
    // units show a dominant edge direction (where ringing lives — CDEF's
    // target). Isotropic fine detail (fractals) has no dominant direction;
    // there the variance-masked metric collapses to plain SSE and rewards
    // smoothing SSIMULACRA2 punishes — measured on the corpus as a −0.2 SS2
    // regression whenever global filtering engages on such content. Per-unit
    // decisions are NOT gated: the distortion margin protects them.
    let mut dv_sum = vec![0i64; n_units];
    let mut dv_cnt = vec![0i64; n_units];
    for by in 0..nby {
        for bx in 0..nbx {
            if !lskip[by * nbx + bx] {
                let u = (by / 8) * uc + bx / 8;
                dv_sum[u] += lvars[by * nbx + bx] as i64;
                dv_cnt[u] += 1;
            }
        }
    }
    let trusted: Vec<bool> = (0..n_units)
        .map(|u| gate_thresh == 0 || dv_sum[u] / dv_cnt[u].max(1) >= gate_thresh)
        .collect();
    let n_trusted = trusted.iter().filter(|&&t| t).count();

    // ---- Candidate strengths ------------------------------------------------
    // Slow: two-stage full search — all 15 primary-only strengths, then the
    // top-K primaries crossed with every secondary (libaom pickcdef coverage).
    // Fast: the small legacy set. Index 0 is always (0,0) = unfiltered.
    let luma_tab = |pri: i32, sec: i32| -> Vec<i64> {
        cdef_luma_unit_dists(
            &snap_y, &src[0], w8, h8, disp_w, disp_h, &ldirs, &lvars, &lskip, nbx, uc, n_units,
            pri, sec, damping, bd, perceptual,
        )
    };
    // A unit only counts as gained when filtering clears the margin: a small
    // guaranteed improvement, standing in for the SSE/SSIMULACRA2 mismatch.
    let clears = |off: i64, on: i64| -> i64 {
        let thr = off - off.saturating_mul(margin) / 1000;
        if on < thr { off - on } else { 0 }
    };

    let slow = speed == Speed::Slow;
    let mut cands: Vec<(i32, i32)> = vec![(0, 0)];
    let mut ly: Vec<Vec<i64>> = vec![luma_tab(0, 0)];
    let luma_off: Vec<i64> = ly[0].clone();
    if slow {
        // Stage A: primary-only sweep. Strengths above 4 are deliberately NOT
        // searched: the corpus shows the decision metric over-rates strong
        // smoothing (SSIMULACRA2 regressions up to −0.4 when pri 5..15 entries
        // were offered), matching the legacy candidate cap.
        let pri_list: Vec<i32> = vec![1, 2, 4];
        let want = pool.width().min(pri_list.len());
        let tabs = pool.map_indexed(want, pri_list.len(), |i| luma_tab(pri_list[i], 0));
        let mut ranked: Vec<(i64, usize)> = tabs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    (0..n_units).map(|u| clears(luma_off[u], t[u])).sum::<i64>(),
                    i,
                )
            })
            .collect();
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push((pri_list[i], 0));
            ly.push(t);
        }
        // Stage B: top-K primaries (plus pri 0) crossed with the secondaries.
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut stage_b: Vec<(i32, i32)> = Vec::new();
        for sec in [1, 2] {
            stage_b.push((0, sec));
            for &(gain, i) in ranked.iter().take(4) {
                if gain > 0 {
                    stage_b.push((pri_list[i], sec));
                }
            }
        }
        let want = pool.width().min(stage_b.len().max(1));
        let tabs = pool.map_indexed(want, stage_b.len(), |i| {
            luma_tab(stage_b[i].0, stage_b[i].1)
        });
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push(stage_b[i]);
            ly.push(t);
        }
    } else {
        let list: Vec<(i32, i32)> = cdef::PRI_CANDIDATES
            .iter()
            .flat_map(|&pri| cdef::SEC_CANDIDATES.iter().map(move |&sec| (pri, sec)))
            .filter(|&(pri, sec)| !(pri == 0 && sec == 0))
            .collect();
        let want = pool.width().min(list.len().max(1));
        let tabs = pool.map_indexed(want, list.len(), |i| luma_tab(list[i].0, list[i].1));
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push(list[i]);
            ly.push(t);
        }
    }

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
    let snap_uv: [Vec<i32>; 2] = if mono {
        [Vec::new(), Vec::new()]
    } else {
        [recon[1].clone(), recon[2].clone()]
    };
    // Chroma candidate space: (0,0) plus the same list as luma (its tables are
    // cheap relative to luma). Monochrome keeps only (0,0).
    let (c_cands, lc): (Vec<(i32, i32)>, Vec<Vec<i64>>) = if mono {
        (vec![(0, 0)], vec![vec![0; n_units]])
    } else {
        let unit_sse = |pri: i32, sec: i32| -> Vec<i64> {
            let u = cdef_chroma_unit_sse(
                &snap_uv[0],
                &src[1],
                cw8,
                ch8,
                cw_vis,
                ch_vis,
                &ldirs,
                &uv_dir,
                &lskip,
                nbx,
                nby,
                uc,
                n_units,
                sub_x,
                sub_y,
                pri,
                sec,
                chroma_damping,
                bd,
            );
            let v = cdef_chroma_unit_sse(
                &snap_uv[1],
                &src[2],
                cw8,
                ch8,
                cw_vis,
                ch_vis,
                &ldirs,
                &uv_dir,
                &lskip,
                nbx,
                nby,
                uc,
                n_units,
                sub_x,
                sub_y,
                pri,
                sec,
                chroma_damping,
                bd,
            );
            (0..n_units).map(|i| u[i] + v[i]).collect()
        };
        let want = pool.width().min(cands.len().max(1));
        let tabs = pool.map_indexed(want, cands.len(), |i| {
            if i == 0 {
                unit_sse(0, 0)
            } else {
                unit_sse(cands[i].0, cands[i].1)
            }
        });
        (cands.clone(), tabs)
    };
    let chroma_off: Vec<i64> = lc[0].clone();

    let d_off: Vec<i64> = (0..n_units).map(|u| luma_off[u] + chroma_off[u]).collect();
    // Margin threshold each unit must clear to filter (see `clears`).
    let thr: Vec<i64> = d_off
        .iter()
        .map(|&o| o - o.saturating_mul(margin) / 1000)
        .collect();

    // ---- Joint (luma, chroma) strength-set search ----------------------------
    // Greedy set construction over all strength PAIRS, exactly libaom's
    // `joint_strength_search_dual`: each added entry minimizes the frame total
    // when every unit picks its best entry (with the off-fallback margin rule
    // once the set contains the all-zero entry). The greedy is deterministic
    // and incremental, so one run to 8 entries yields the candidate sets for
    // every cdef_bits at its prefixes of size 1, 2, 4, 8.
    let nc = c_cands.len();
    let n_pairs = cands.len() * nc;
    let d_pair = |p: usize, u: usize| ly[p / nc][u] + lc[p % nc][u];
    let want = pool.width().min(n_pairs);

    // Global (cdef_bits = 0) candidate: the raw frame-total minimizer — every
    // signaled unit filters with it, no off-fallback.
    let raw_totals: Vec<i64> = pool.map_indexed(want, n_pairs, |p| {
        (0..n_units)
            .map(|u| if signaled[u] { d_pair(p, u) } else { d_off[u] })
            .sum()
    });
    let mut global_entry = 0usize;
    let mut prefix_tot = [i64::MAX; 4]; // totals at set sizes 1, 2, 4, 8
    for (p, &t) in raw_totals.iter().enumerate() {
        if t < prefix_tot[0] {
            prefix_tot[0] = t;
            global_entry = p;
        }
    }

    // Per-unit sets are seeded with the all-zero entry: a unit must always be
    // able to signal OFF (the Phase-1 {off, strength} table generalized), and
    // the margin rule sends units that don't clearly gain back to it.
    let mut cur_min: Vec<i64> = d_off.clone();
    let mut entries: Vec<usize> = vec![0];
    while entries.len() < 8 {
        let scores: Vec<i64> = pool.map_indexed(want, n_pairs, |p| {
            let mut tot = 0i64;
            for u in 0..n_units {
                if !signaled[u] {
                    tot += d_off[u];
                    continue;
                }
                let b = cur_min[u].min(d_pair(p, u));
                tot += if b >= thr[u] { d_off[u] } else { b };
            }
            tot
        });
        let mut best_p = 0usize;
        let mut best_tot = i64::MAX;
        for (p, &t) in scores.iter().enumerate() {
            if t < best_tot {
                best_tot = t;
                best_p = p;
            }
        }
        entries.push(best_p);
        for (u, m) in cur_min.iter_mut().enumerate() {
            *m = (*m).min(d_pair(best_p, u));
        }
        let n = entries.len();
        if n.is_power_of_two() && n <= 8 {
            prefix_tot[n.trailing_zeros() as usize] = best_tot;
        }
    }

    // ---- Per-unit assignment and exact-rate R-D choice of cdef_bits ----------
    // A unit does not take its raw metric-minimizing entry: with several
    // entries available, per-unit metric errors accumulate (measured as SS2
    // regressions on detail-rich content). Instead each unit takes the MILDEST
    // entry that still clears the margin against off — least filtering among
    // the options the metric is confident in. Units clearing nothing take the
    // all-zero entry when present (else the raw minimum). The R-D totals are
    // computed from this realized assignment, not the greedy's raw minimum.
    let strength_mag = |p: usize| -> i32 {
        let (yp, ys) = cands[p / nc];
        let (up, us) = c_cands[p % nc];
        yp + ys + up + us
    };
    let assign = |set: &[usize]| -> (Vec<u8>, i64) {
        let z_in_set = set.iter().position(|&p| p == 0);
        let mut total = 0i64;
        let grid: Vec<u8> = (0..n_units)
            .map(|u| {
                if !signaled[u] {
                    total += d_off[u];
                    return 0;
                }
                let mut pick: Option<(i32, i64, usize)> = None; // (mag, d, e)
                let mut raw_best = (i64::MAX, 0usize);
                for (e, &p) in set.iter().enumerate() {
                    let d = d_pair(p, u);
                    if d < raw_best.0 {
                        raw_best = (d, e);
                    }
                    if d < thr[u] {
                        let mag = strength_mag(p);
                        if pick.is_none_or(|(m, pd, _)| mag < m || (mag == m && d < pd)) {
                            pick = Some((mag, d, e));
                        }
                    }
                }
                match (pick, z_in_set) {
                    (Some((_, d, e)), _) => {
                        total += d;
                        e as u8
                    }
                    (None, Some(z)) => {
                        total += d_off[u];
                        z as u8
                    }
                    (None, None) => {
                        total += raw_best.0;
                        raw_best.1 as u8
                    }
                }
            })
            .collect();
        (grid, total)
    };

    // Rate deltas against the always-written noop CDEF header: each strength
    // entry past the first costs 12 header bits (6 mono), and every signaled
    // unit carries a raw `cdef_idx` literal of `cdef_bits` bits.
    let lambda = crate::cost::mode_lambda_q(crate::quant::dc_q(base_q_idx, bd) as f32);
    let total_off: i64 = d_off.iter().sum();
    let cost_off = total_off as f32;
    let per_entry_bits = if mono { 6.0f32 } else { 12.0f32 };
    let mut best_bits: Option<u8> = None;
    let mut best_cost = cost_off;
    let max_bits = env_i64("MT_CDEF_MAX_BITS", 3).clamp(0, 3) as u8;
    for b in 0..=max_bits {
        let nb = 1usize << b;
        let total = if b == 0 {
            // Global: every signaled unit filters with `global_entry`.
            prefix_tot[0]
        } else {
            assign(&entries[..nb]).1
        };
        if total == i64::MAX {
            continue;
        }
        // Global filtering (cdef_bits = 0) has no per-unit off-switch, so it is
        // offered only when most units show a dominant edge direction (the
        // regime the metric is trusted in) and the whole frame clears the
        // margin. Per-unit modes are protected by the margin rule instead.
        if b == 0
            && (global_entry == 0
                || n_trusted * 4 < n_sig * 3
                || total >= total_off - total_off.saturating_mul(margin) / 1000)
        {
            continue;
        }
        let cost =
            total as f32 + lambda * (per_entry_bits * (nb - 1) as f32 + b as f32 * n_sig as f32);
        if cost < best_cost {
            best_cost = cost;
            best_bits = Some(b);
        }
    }

    if std::env::var_os("MT_AV1_CDEF_DEBUG").is_some() {
        let named: Vec<String> = entries
            .iter()
            .map(|&p| format!("y{:?}c{:?}", cands[p / nc], c_cands[p % nc]))
            .collect();
        eprintln!(
            "CDEF q={base_q_idx} bits={best_bits:?} n_sig={n_sig}/{n_units} off={cost_off:.0} \
             prefix={prefix_tot:?} lambda={lambda:.2} entries=[{}]",
            named.join(" ")
        );
    }
    // Debug oracle: dump the pre-CDEF (deblocked) luma plane + per-8x8 dirs/vars
    // so an external harness can simulate filter variants against a decode.
    if let Ok(path) = std::env::var("MT_AV1_DUMP_PRECDEF") {
        let mut buf: Vec<u8> = Vec::new();
        for &v in &recon[0] {
            buf.extend_from_slice(&(v as u16).to_le_bytes());
        }
        for i in 0..nbx * nby {
            buf.extend_from_slice(&(ldirs[i] as u16).to_le_bytes());
            buf.extend_from_slice(&(lvars[i] as u32).to_le_bytes());
        }
        for &sk in lskip.iter().take(nbx * nby) {
            buf.push(sk as u8);
        }
        let _ = std::fs::write(path, buf);
    }

    // Test hook: force a specific cdef_bits to exercise wider index literals.
    if let Ok(v) = std::env::var("MT_CDEF_FORCE_BITS")
        && let Ok(b) = v.parse::<u8>()
        && b <= 3
    {
        best_bits = Some(b);
    }
    let bits = best_bits?;
    let nb = 1usize << bits;
    let global_set = [global_entry];
    let set: &[usize] = if bits == 0 {
        &global_set
    } else {
        &entries[..nb]
    };
    let grid: Vec<u8> = if bits == 0 {
        vec![0; n_units]
    } else {
        assign(set).0
    };

    let strengths: Vec<(u8, u8, u8, u8)> = set
        .iter()
        .map(|&p| {
            let (yp, ys) = cands[p / nc];
            let (up, us) = c_cands[p % nc];
            (yp as u8, ys as u8, up as u8, us as u8)
        })
        .collect();
    // Per-unit effective strengths for the apply pass (constant for bits == 0;
    // all-skip units never filter regardless).
    let unit_y: Vec<(i32, i32)> = (0..n_units)
        .map(|u| {
            let (yp, ys, _, _) = strengths[grid[u] as usize];
            (yp as i32, ys as i32)
        })
        .collect();
    let unit_uv: Vec<(i32, i32)> = (0..n_units)
        .map(|u| {
            let (_, _, up, us) = strengths[grid[u] as usize];
            (up as i32, us as i32)
        })
        .collect();

    let decision = CdefFrameDecision {
        params: crate::obu::CdefParams {
            bits,
            damping: signaled_damping as u8,
            strengths,
        },
        grid: if bits > 0 { grid } else { Vec::new() },
        unit_cols: uc,
    };

    // Apply the decision to the reconstruction, reading every tap from the
    // pre-CDEF snapshot exactly as the decoder does.
    if unit_y.iter().any(|&(p, s)| p != 0 || s != 0) {
        apply_cdef_plane(
            &mut recon[0],
            &snap_y,
            w8,
            h8,
            &ldirs,
            &lvars,
            &lskip,
            nbx,
            uc,
            &unit_y,
            damping,
            bd,
            pool,
        );
    }
    if !mono && unit_uv.iter().any(|&(p, s)| p != 0 || s != 0) {
        for plane in 1..3 {
            apply_cdef_chroma(
                &mut recon[plane],
                &snap_uv[plane - 1],
                cw8,
                ch8,
                &ldirs,
                &uv_dir,
                skip8,
                sb8w,
                nbx,
                nby,
                sub_x,
                sub_y,
                uc,
                &unit_uv,
                chroma_damping,
                bd,
                pool,
            );
        }
    }

    Some(decision)
}

/// Distortion of one (possibly partially visible) 8x8: the perceptual
/// `cdef_dist_8x8` when the block is fully inside the visible frame, otherwise
/// plain SSE over the visible `vis_w` x `vis_h` window (the x64 variance
/// calibration needs a full 8x8; matches the shared metric's edge fallback).
#[allow(clippy::too_many_arguments)]
fn cdef_block_dist_vis(
    src: &[i32],
    dst: &[i32],
    stride: usize,
    vis_w: usize,
    vis_h: usize,
    x: usize,
    y: usize,
    coeff_shift: u32,
    perceptual: bool,
) -> i64 {
    if perceptual && x + 8 <= vis_w && y + 8 <= vis_h {
        let rows = dst.len() / stride;
        return crate::cdef::cdef_dist_8x8(src, dst, stride, rows, x, y, coeff_shift);
    }
    let mut s = 0i64;
    for yy in y..(y + 8).min(vis_h) {
        for xx in x..(x + 8).min(vis_w) {
            let d = (dst[yy * stride + xx] - src[yy * stride + xx]) as i64;
            s += d * d;
        }
    }
    s
}

/// Per-64x64-unit luma perceptual CDEF distortion for one candidate strength
/// (`pri == sec == 0` measures the unfiltered baseline). `recon` is the MARKED
/// plane (out-of-frame samples = `CDEF_VERY_LARGE`); distortion counts only the
/// visible `disp_w` x `disp_h` window. Skip 8x8s are excluded entirely — the
/// decoder never filters them, so their distortion is identical across every
/// option and cancels out of the R-D comparison.
#[allow(clippy::too_many_arguments)]
fn cdef_luma_unit_dists(
    recon: &[i32],
    src: &[i32],
    w: usize,
    h: usize,
    disp_w: usize,
    disp_h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    n_units: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
    perceptual: bool,
) -> Vec<i64> {
    use crate::cdef;
    let coeff_shift = (bd - 8) as u32;
    let filtering = pri != 0 || sec != 0;
    let mut tmp = if filtering {
        recon.to_vec()
    } else {
        Vec::new()
    };
    let mut out = vec![0i64; n_units];
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let bi = (y / 8) * nbx + x / 8;
            if skip.get(bi).copied().unwrap_or(true) {
                continue;
            }
            if x >= disp_w || y >= disp_h {
                continue; // fully outside the visible frame
            }
            let u = (y / 64) * uc + x / 64;
            let dist = if filtering {
                // adjust_pri must be applied to the bit-depth-shifted strength
                // (matches the decoders' adjust_strength, which scales the
                // already-shifted level); scaling then shifting does not commute
                // because of the `+8 >> 4` rounding.
                let apri = cdef::adjust_pri(pri << (bd - 8), vars[bi]);
                cdef::cdef_filter_8x8(
                    &mut tmp,
                    recon,
                    w,
                    x,
                    y,
                    apri,
                    sec << (bd - 8),
                    // Decoders pass dir 0 when the SIGNALED pri strength is 0
                    // (`pri_strength ? dir : 0`) — sec-only filtering then runs
                    // on direction 0, not the block's estimated direction.
                    if pri == 0 { 0 } else { dirs[bi] },
                    damping,
                    bd,
                );
                cdef_block_dist_vis(src, &tmp, w, disp_w, disp_h, x, y, coeff_shift, perceptual)
            } else {
                cdef_block_dist_vis(src, recon, w, disp_w, disp_h, x, y, coeff_shift, perceptual)
            };
            out[u] += dist;
        }
    }
    out
}

/// Per-64x64-unit chroma SSE for one candidate strength (`0,0` = baseline),
/// accumulated into the covering LUMA unit (the cdef_idx is shared with luma).
/// One chroma sub-block per non-skip luma 8x8: 8x8 (4:4:4), 4x8 (4:2:2),
/// 4x4 (4:2:0). Plain SSE — chroma sub-blocks are too small for the 8x8
/// variance calibration of the perceptual metric (libaom also uses MSE here).
#[allow(clippy::too_many_arguments)]
fn cdef_chroma_unit_sse(
    recon: &[i32],
    src: &[i32],
    cw: usize,
    _ch: usize,
    cw_vis: usize,
    ch_vis: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    lskip: &[bool],
    nbx: usize,
    nby: usize,
    uc: usize,
    n_units: usize,
    sub_x: usize,
    sub_y: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) -> Vec<i64> {
    use crate::cdef;
    let cbw = 8 >> sub_x;
    let cbh = 8 >> sub_y;
    let filtering = pri != 0 || sec != 0;
    let mut tmp = if filtering {
        recon.to_vec()
    } else {
        Vec::new()
    };
    let mut out = vec![0i64; n_units];
    for lby in 0..nby {
        for lbx in 0..nbx {
            if lskip.get(lby * nbx + lbx).copied().unwrap_or(true) {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            let cy = (lby * 8) >> sub_y;
            if cx >= cw_vis || cy >= ch_vis {
                continue; // fully outside the visible chroma frame
            }
            let u = (lby / 8) * uc + lbx / 8;
            let cand: &[i32] = if filtering {
                // dir 0 when the signaled pri strength is 0 (see luma above).
                let dir = if pri == 0 {
                    0
                } else {
                    uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)]
                };
                cdef::cdef_filter_block(
                    &mut tmp,
                    0,
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
                &tmp
            } else {
                recon
            };
            let mut sse = 0i64;
            for yy in cy..(cy + cbh).min(ch_vis) {
                for xx in cx..(cx + cbw).min(cw_vis) {
                    let d = (cand[yy * cw + xx] - src[yy * cw + xx]) as i64;
                    sse += d * d;
                }
            }
            out[u] += sse;
        }
    }
    out
}

/// Apply the luma CDEF strength to a whole plane in place, reading from the
/// MARKED pre-CDEF snapshot (out-of-frame samples = `CDEF_VERY_LARGE`) so every
/// 8x8 filters the same source pixels the decoder does. With a per-unit `grid`
/// (`cdef_bits = 1`), only 8x8s inside on-units are filtered.
#[allow(clippy::too_many_arguments)]
fn apply_cdef_plane(
    plane: &mut [i32],
    snapshot: &[i32],
    w: usize,
    _h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    unit_pri_sec: &[(i32, i32)],
    damping: i32,
    bd: u8,
    pool: &Pool,
) {
    use crate::cdef;
    // Every 8x8 reads the pre-CDEF snapshot and writes only its own rows, so
    // 8-row bands filter in parallel with identical output to the serial pass.
    let items: Vec<(usize, &mut [i32])> = plane.chunks_mut(8 * w).enumerate().collect();
    pool.for_each(pool.width(), items, |(bi8, band)| {
        let y = bi8 * 8;
        for x in (0..w).step_by(8) {
            let bxi = x / 8;
            let byi = y / 8;
            let bi = byi * nbx + bxi;
            // Skip blocks are left untouched, matching the decoder.
            if skip.get(bi).copied().unwrap_or(true) {
                continue;
            }
            // The unit's signaled strength; (0,0) units stay untouched.
            let (pri, sec) = unit_pri_sec[(y / 64) * uc + x / 64];
            if pri == 0 && sec == 0 {
                continue;
            }
            // Use the precomputed direction/variance (same pre-CDEF snapshot the
            // search used). Identical to the old in-loop recompute, but free.
            // Decoders force dir 0 when the signaled pri strength is 0.
            let dir = if pri == 0 { 0 } else { dirs[bi] };
            let var = vars[bi];
            // adjust_pri on the bit-depth-shifted strength (see search above).
            let apri = cdef::adjust_pri(pri << (bd - 8), var);
            cdef::cdef_filter_block(
                band,
                y,
                snapshot,
                w,
                x,
                y,
                8,
                8,
                apri,
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    });
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
    snapshot: &[i32],
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
    uc: usize,
    unit_pri_sec: &[(i32, i32)],
    damping: i32,
    bd: u8,
    pool: &Pool,
) {
    use crate::cdef;
    let cbw = 8 >> sub_x; // chroma sub-block width per luma 8x8
    let cbh = 8 >> sub_y;
    // One band per luma block-row (cbh chroma rows): disjoint writes, shared
    // pre-CDEF snapshot reads — identical output to the serial pass.
    let items: Vec<(usize, &mut [i32])> = plane.chunks_mut(cbh * cw).enumerate().collect();
    pool.for_each(pool.width(), items, |(lby, band)| {
        if lby >= nby {
            return;
        }
        let cy = (lby * 8) >> sub_y;
        for lbx in 0..nbx {
            // Skip if the covering luma 8x8 is a skip block.
            if skip8.get(lby * sb8w + lbx).copied().unwrap_or(true) {
                continue;
            }
            // The unit's signaled chroma strength; (0,0) units stay untouched.
            let (pri, sec) = unit_pri_sec[(lby / 8) * uc + lbx / 8];
            if pri == 0 && sec == 0 {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            if cx >= cw || cy >= ch {
                continue;
            }
            // Decoders force dir 0 when the signaled pri strength is 0.
            let dir = if pri == 0 {
                0
            } else {
                uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)]
            };
            cdef::cdef_filter_block(
                band,
                cy,
                snapshot,
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
    });
}

#[allow(clippy::too_many_arguments)]
fn frame_deblock(
    recon: &mut [Vec<i32>; 3],
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    disp_w: usize,
    disp_h: usize,
    blk4: &[u8],    // luma block width map (vertical edges)
    blk4h: &[u8],   // luma block height map (horizontal edges)
    blk4v: &[bool], // luma block starts at this 4x4 column
    blk4t: &[bool], // luma block starts at this 4x4 row
    nc4: usize,     // luma 4-col count == w8/4
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
            disp_w,
            disp_h,
            blk4,
            blk4h,
            blk4v,
            blk4t,
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
    let cvis_w = disp_w.div_ceil(1 << ss_hor);
    let cvis_h = disp_h.div_ceil(1 << ss_ver);
    #[allow(clippy::needless_range_loop)]
    for plane in 1..3 {
        crate::loopfilter::filter_plane(
            &mut recon[plane],
            cw,
            ch,
            cvis_w,
            cvis_h,
            &cbw4,
            &cbh4,
            &[],
            &[],
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_frame_obus(
    base_q_idx: u8,
    qm: QmLevels,
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
            qm,
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
            qm,
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

pub(crate) fn encode_lossless_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
) -> Vec<u8> {
    let pool = Pool::new(threads);
    let (tilegroup, plan) = encode_lossless_tilegroup(bd, w8, h8, src, &pool);
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
    crate::tile::encode_tile_lossless(tw, th, bd, [&p0, &p1, &p2])
}

fn encode_lossless_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i16>; 3],
    pool: &Pool,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = pool.width();
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
    let payloads: Vec<Vec<u8>> = pool.map_indexed(nthreads, n, |i| {
        encode_one_lossless_tile(bd, w8, src, &rects[i])
    });

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
    crate::tile::encode_tile_lossless_mono(tw, th, bd, &p0)
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
    pool: &Pool,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = pool.width();
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
    let payloads: Vec<Vec<u8>> = pool.map_indexed(nthreads, n, |i| {
        encode_one_lossless_tile_mono(bd, w8, luma, &rects[i])
    });

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
    let pool = Pool::new(threads);
    let (tilegroup, plan) = encode_lossless_mono_tilegroup(bd, w8, h8, luma, &pool);
    assemble_lossless_mono_frame_obus(&plan, &tilegroup)
}

#[cfg(test)]
mod aq_tests {
    use super::*;

    #[test]
    fn directional_top_k_keeps_three_lowest_costs() {
        let mut top = DirectionalTopK::new();
        for (mode, cost) in [(1, 50), (2, 10), (3, 30), (4, 20), (5, 40)] {
            top.insert(mode, cost);
        }
        assert!(top.contains(2));
        assert!(top.contains(3));
        assert!(top.contains(4));
        assert!(!top.contains(1));
        assert!(!top.contains(5));
    }

    #[test]
    fn satd_sad_proxy_is_zero_only_for_equal_blocks() {
        let src = [7i32; 16];
        assert_eq!(satd_sad_proxy(&src, 4, &src, 4, 4, 4), 0);
        let mut pred = src;
        pred[5] += 3;
        assert!(satd_sad_proxy(&src, 4, &pred, 4, 4, 4) > 0);
    }

    /// Dark protection over a full 64×64 SB of the given i32 plane, with the AV1
    /// 8-bit normalization scale for bit depth `bd`.
    fn dark_prot(d: &DarkAq, base_q: i32, yp: &[i32], bd: u8) -> i32 {
        let scale = 1.0 / (1u32 << (bd - 8)) as f32;
        crate::aq_common::dark_protection(d, base_q, yp, 64, 0, 0, 64, 64, scale)
    }

    #[test]
    fn dark_protection_targets_dark_structure_only() {
        // 64x64 SBs at three luma levels, each with a coherent striped texture (survives
        // the 2x downsample, so it registers as structure not noise). Ported from
        // `av2/aq.rs`, on the i32 luma plane the AV1 encoder uses.
        let build = |mean: i32| -> Vec<i32> {
            let mut p = vec![0i32; 64 * 64];
            for r in 0..64 {
                for c in 0..64 {
                    // 4px-wide stripes: strong at both scales.
                    p[r * 64 + c] = mean + if (c / 4) % 2 == 0 { 12 } else { -12 };
                }
            }
            p
        };
        let d = DarkAq::on(); // enabled, min_q = 150
        let dark = build(36);
        let bright = build(180);
        let flat_dark = vec![24i32; 64 * 64]; // dark but no structure

        let base_q = 190; // in the gated range (>= 150)
        let d_dark = dark_prot(&d, base_q, &dark, 8);
        let d_bright = dark_prot(&d, base_q, &bright, 8);
        let d_flat = dark_prot(&d, base_q, &flat_dark, 8);
        assert!(
            d_dark > 0,
            "dark structured SB should be protected, got {d_dark}"
        );
        assert_eq!(d_bright, 0, "bright structured SB must not be protected");
        assert_eq!(
            d_flat, 0,
            "dark flat SB (no structure) must not be protected"
        );

        // Gate: disabled below min_q (high quality / low qindex).
        assert_eq!(dark_prot(&d, 100, &dark, 8), 0, "gated out below min_q");
        // Disabled config is always inert.
        assert_eq!(
            dark_prot(&DarkAq::off(), base_q, &dark, 8),
            0,
            "disabled dark AQ must not protect"
        );
    }

    #[test]
    fn dark_protection_bitdepth_normalized() {
        // A 10-bit dark structured SB (all values << 2) must score the SAME protection
        // as its 8-bit counterpart: the stats normalize native depth to 8-bit range.
        let build = |mean: i32, shift: u32| -> Vec<i32> {
            let mut p = vec![0i32; 64 * 64];
            for r in 0..64 {
                for c in 0..64 {
                    p[r * 64 + c] = (mean + if (c / 4) % 2 == 0 { 12 } else { -12 }) << shift;
                }
            }
            p
        };
        let d = DarkAq::on();
        let base_q = 190;
        let p8 = build(36, 0);
        let p10 = build(36, 2);
        let d8 = dark_prot(&d, base_q, &p8, 8);
        let d10 = dark_prot(&d, base_q, &p10, 10);
        assert!(d8 > 0);
        assert_eq!(d8, d10, "dark protection must be bit-depth invariant");
    }

    #[test]
    fn chroma_rd_can_favor_a_split_on_color_detail() {
        let mut src = [
            vec![128i32; 32 * 32],
            vec![128i32; 32 * 32],
            vec![128i32; 32 * 32],
        ];
        for y in 0..16 {
            for x in 0..16 {
                let quadrant = (x >= 8) as usize + 2 * (y >= 8) as usize;
                src[1][y * 32 + x] = [24, 232, 216, 40][quadrant];
                src[2][y * 32 + x] = [224, 48, 32, 240][quadrant];
            }
        }
        let tile = LossyTile::new(160, 8, 32, 32, &src, QmLevels::FLAT);
        let none = tile.rd_cost_chroma_partition(0, 0, 16, Part16::None, 1.0);
        let split = tile.rd_cost_chroma_partition(0, 0, 16, Part16::Split, 1.0);
        assert!(
            split < none,
            "four color-homogeneous chroma blocks should beat one mixed block: split={split}, none={none}"
        );
    }
}
