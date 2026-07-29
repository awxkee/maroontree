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

//! Rate-distortion cost matching AVM (av2/encoder/rd.{c,h}). Replaces the
//! fixed linear-q SSE thresholds in the inter paths with q^2 RD comparisons.

/// AVM low-delay rd-mult numerator (RDMULT_FROM_Q2_NUM_LD).
const RDMULT_Q2_NUM_LD: i64 = 82;
/// AVM rd-mult denominator (RDMULT_FROM_Q2_DEN).
const RDMULT_Q2_DEN: i64 = 32;

/// AVM rdmult from the dc quant step (low-delay curve). rdmult ~ q^2 * 82/32.
#[inline]
pub(crate) fn rd_mult(qstep: u32) -> f32 {
    let q = qstep as i64;
    (q * q * RDMULT_Q2_NUM_LD / RDMULT_Q2_DEN) as f32
}

/// AVM RDCOST: (rate_bits * rdmult) + distortion, in comparable f32 units
/// (dropping AVM's fixed >>9 / <<7 shifts as a common scale).
#[inline]
pub(crate) fn rd_cost(distortion: f32, rate_bits: f32, qstep: u32) -> f32 {
    const RECIP_512: f32 = 1. / 512.0;
    distortion + rate_bits * rd_mult(qstep) * RECIP_512
}

/// SS2-calibrated distortion weight applied uniformly to inter distortion so the
/// RD boundary is retuned against SSIMULACRA2 without re-deriving rdmult.
pub(crate) const SS2_INTER_DIST_W: f32 = 16.0;

/// Rate for the block-level inter syntax that is known before transform and
/// coefficient coding. CDF-coded fields use the tile's current probabilities,
/// including the exact QTR_PEL shell walk for NEWMV.
#[inline]
pub(crate) fn inter_syntax_bits(
    enc: &crate::av2::entropy::RangeEncoder,
    skip_ctx: usize,
    mode_ctx: usize,
    skip_txfm: bool,
    mode: usize,
    mvd: Option<crate::av2::video::mv::Mv>,
) -> f32 {
    let mut bits = enc.estimate_intra_inter_bits(1)
        + enc.estimate_skip_txfm_bits(skip_ctx, usize::from(skip_txfm))
        + enc.estimate_single_ref_bits(enc.ref_rank)
        + enc.estimate_inter_mode_bits(mode_ctx, mode);
    if mode != 1 {
        bits += enc.estimate_drl_bits(mode_ctx, 0);
    }
    if let Some(delta) = mvd {
        bits += crate::av2::video::mvd::estimate_mvd_qtr_bits(enc, delta.row, delta.col);
    }
    bits
}

/// Final live-CDF competition between the spatial NEARMV predictor and a
/// searched NEWMV. Distortions are already in the caller's chosen pixel-domain
/// metric; this helper only supplies the exact syntax-rate side of the decision.
pub(crate) struct NearMvRdSpec {
    pub(crate) skip_ctx: usize,
    pub(crate) mode_ctx: usize,
    pub(crate) skip_txfm: bool,
    pub(crate) near_distortion: f32,
    pub(crate) new_distortion: f32,
    pub(crate) new_mvd: crate::av2::video::mv::Mv,
    pub(crate) qstep: u32,
}

#[inline]
pub(crate) fn prefer_nearmv(enc: &crate::av2::entropy::RangeEncoder, spec: NearMvRdSpec) -> bool {
    let NearMvRdSpec {
        skip_ctx,
        mode_ctx,
        skip_txfm,
        near_distortion,
        new_distortion,
        new_mvd,
        qstep,
    } = spec;
    let near_rate = inter_syntax_bits(enc, skip_ctx, mode_ctx, skip_txfm, 0, None);
    let new_rate = inter_syntax_bits(enc, skip_ctx, mode_ctx, skip_txfm, 2, Some(new_mvd));
    rd_cost(near_distortion, near_rate, qstep) <= rd_cost(new_distortion, new_rate, qstep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::{entropy::RangeEncoder, video::mv::Mv};

    #[test]
    fn live_inter_syntax_rate_is_finite_and_charges_newmv() {
        let mut enc = RangeEncoder::new();
        enc.inter_tile = true;
        enc.enable_adaptive_cdf(1);
        let global = inter_syntax_bits(&enc, 0, 0, true, 1, None);
        let newmv = inter_syntax_bits(&enc, 0, 0, false, 2, Some(Mv { row: 8, col: -4 }));
        assert!(global.is_finite() && global > 0.0);
        assert!(newmv.is_finite() && newmv > global);
    }

    #[test]
    fn live_rate_can_prefer_a_slightly_worse_nearmv_predictor() {
        let mut enc = RangeEncoder::new();
        enc.inter_tile = true;
        enc.enable_adaptive_cdf(1);
        assert!(prefer_nearmv(
            &enc,
            NearMvRdSpec {
                skip_ctx: 0,
                mode_ctx: 0,
                skip_txfm: true,
                near_distortion: 101.0,
                new_distortion: 100.0,
                new_mvd: Mv { row: 24, col: -16 },
                qstep: 64,
            },
        ));
    }
}
