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

use crate::av2::entropy::RangeEncoder;
use crate::av2::quant::qstep;
use crate::util::FastRound;

pub(crate) const AQ_RES_LOG2: u8 = 2;

const AQ_MAX_SIGNALED: i32 = 6;

pub(crate) fn sb_activity(
    yp: &[f32],
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
    let mut sum = 0f64;
    let mut sum2 = 0f64;
    for r in 0..h {
        let base = (sb_y + r) * pw + sb_x;
        let yp = &yp[base..base + w];
        for &c in yp.iter() {
            let v = c as f64;
            sum += v;
            sum2 += v * v;
        }
    }
    let n = (h * w) as f64;
    let mean = sum / n;
    let var = (sum2 / n - mean * mean).max(0.0);
    (1.0 + var).ln() as f32
}

fn sb_subblock_variances(
    yp: &[f32],
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
                out[idx] = f32::NAN; // mark out-of-frame, patched below
                continue;
            }
            let mut sum = 0f64;
            let mut sum2 = 0f64;
            for r in 0..h {
                let base = (y0 + r) * pw + x0;
                for &v in &yp[base..base + w] {
                    let v = v as f64;
                    sum += v;
                    sum2 += v * v;
                }
            }
            let n = (h * w) as f64;
            let mean = sum / n;
            let var = (sum2 / n - mean * mean).max(0.0) as f32;
            out[idx] = var;
            acc += var as f64;
            filled += 1;
        }
    }
    if filled == 0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    // Patch out-of-frame subblocks with the in-frame mean (neutral for the octile).
    let mean = (acc / filled as f64) as f32;
    for v in out.iter_mut() {
        if v.is_nan() {
            *v = mean;
        }
    }
    filled
}

/// The representative variance of a superblock for Variance Boost: the value at the
/// requested `octile` (1..=8) of the 64 sorted 8x8 variances. Octile 1 = the most
/// low-variance-biased pick (boost readily), octile 8 = only the maximum (boost only
/// when the whole SB is low-variance).
fn sb_octile_variance(subvars: &mut [f32; 64], octile: u8) -> f32 {
    subvars.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Octile o in 1..=8 maps to sorted index o*8 - 1 (o=1 -> 7, o=4 -> 31 (median-ish),
    // o=6 -> 47 (SVT-AV1-PSY default), o=8 -> 63 (max)).
    let o = octile.clamp(1, 8) as usize;
    let idx = (o * 8 - 1).min(63);
    subvars[idx]
}

fn variance_boost_delta(picked_var: f32, ref_log: f32, strength: f32, boost_only: bool) -> i32 {
    // Work in log-variance: compresses the huge dynamic range of variance and matches
    // the reference (which is a mean of log-variances).
    let v_log = (1.0 + picked_var).ln();
    // Low-variance threshold (curve 0): ln(1 + 256).
    const LOW_LOG: f32 = 5.549_076; // (1.0 + 256.0).ln()
    const MAX_BOOST: f32 = 18.0; // max qindex *reduction* for the flattest SBs
    const MAX_CUT: f32 = 10.0; // max qindex *increase* for the busiest SBs
    // qindex per unit log-variance for each side.
    const BOOST_SLOPE: f32 = 5.0;
    const CUT_SLOPE: f32 = 3.0;

    if v_log < LOW_LOG {
        // Low contrast: boost (negative delta). Deeper below threshold => stronger.
        let d = ((LOW_LOG - v_log) * BOOST_SLOPE * strength).min(MAX_BOOST);
        -(d.fast_round() as i32)
    } else if boost_only {
        0
    } else {
        // Higher contrast: coarsen relative to the tile reference, capped. Using the
        // reference (not the threshold) keeps well-textured frames near zero-mean.
        let over = (v_log - ref_log.max(LOW_LOG)).max(0.0);
        let d = (over * CUT_SLOPE * strength).min(MAX_CUT);
        d.fast_round() as i32
    }
}

/// Mean activity over all superblocks of a (padded) luma plane — the per-tile
/// reference used to center the AQ deltas so they are zero-mean.
pub(crate) fn tile_ref_activity(
    yp: &[f32],
    pw: usize,
    sb_rows: usize,
    sb_cols: usize,
    width: usize,
    height: usize,
) -> f32 {
    let mut sum = 0f32;
    let mut cnt = 0f32;
    for row in 0..sb_rows {
        for col in 0..sb_cols {
            sum += sb_activity(yp, pw, row * 64, col * 64, width, height);
            cnt += 1.0;
        }
    }
    if cnt > 0.0 { sum / cnt } else { 5.0 }
}

/// Scale a residual block by `s` (identity-fast for `s == 1.0`). Used to quantize
/// at a per-SB qstep via the linear projection: `levels = project(resid * (qstep_base
/// / qstep_sb))` equals quantizing at `qstep_sb`.
pub(crate) fn scale_resid(v: &[f32], s: f32) -> Vec<f32> {
    if s == 1.0 {
        v.to_vec()
    } else {
        v.iter().map(|&x| x * s).collect()
    }
}

/// Per-tile adaptive-quantization state. Construct once per `encode_*_core`
/// invocation (one tile); call [`AqState::per_sb`] at the top of each superblock.
pub(crate) struct AqState {
    present: bool,
    base_q: i32,
    qstep_base: i32,
    ref_act: f32,
    /// Running qindex accumulator (decoder `ts.last_qidx`), reset to the frame
    /// base at each tile start.
    last_qidx: i32,
    /// Variance Boost selectivity octile (1..=8). Default 6 (SVT-AV1-PSY default).
    vb_octile: u8,
    /// Variance Boost strength multiplier (1.0 = nominal).
    vb_strength: f32,
    /// When true, only boost low-variance SBs (net-negative, spends bits). When false
    /// (default), also coarsen high-variance SBs to keep the rate roughly matched.
    vb_boost_only: bool,
}

impl AqState {
    /// `present` = frame enables delta-Q; `qstep_base` = qstep at the frame base
    /// qindex; `ref_act` = tile mean activity (see [`tile_ref_activity`]).
    pub(crate) fn new(present: bool, base_q: i32, qstep_base: i32, ref_act: f32) -> Self {
        AqState {
            present,
            base_q,
            qstep_base,
            ref_act,
            last_qidx: base_q,
            vb_octile: 6,
            vb_strength: 1.0,
            vb_boost_only: false,
        }
    }

    /// Override the Variance Boost knobs (octile selectivity, strength, boost-only).
    /// Returns `self` for chaining at the construction site.
    pub(crate) fn with_variance_boost(
        mut self,
        octile: u8,
        strength: f32,
        boost_only: bool,
    ) -> Self {
        self.vb_octile = octile.clamp(1, 8);
        // SS2-calibrated taper: boost pays at coarse q, is net-negative near-lossless.
        let taper = ((self.base_q as f32 - 30.0) / 40.0).clamp(0.0, 1.0);
        self.vb_strength = strength.max(0.0) * taper;
        self.vb_boost_only = boost_only;
        self
    }

    /// Decide this superblock's quantizer from its `activity`, set
    /// `enc.delta_q_signaled`, and return `(qstep_sb, resid_scale)` to apply to the
    /// SB's luma and chroma (pass `resid_scale` to `encode_luma_sb` / `scale_resid`
    /// and reconstruct at `qstep_sb`). When AQ is off this is `(qstep_base, 1.0)`
    /// and signals 0. The caller still arms `enc.delta_q_pending` before the mode.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn per_sb_probe(
        &self,
        yp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        width: usize,
        height: usize,
    ) -> (i32, f32) {
        if !self.present {
            return (self.qstep_base, 1.0);
        }
        let mut subvars = [0f32; 64];
        let filled = sb_subblock_variances(yp, pw, sb_y, sb_x, width, height, &mut subvars);
        let target = if filled == 0 {
            self.base_q
        } else {
            let picked = sb_octile_variance(&mut subvars, self.vb_octile);
            let delta =
                variance_boost_delta(picked, self.ref_act, self.vb_strength, self.vb_boost_only);
            (self.base_q + delta).clamp(1, 255)
        };
        let step = 1i32 << AQ_RES_LOG2;
        let sig = (((target - self.last_qidx) as f32) / step as f32)
            .fast_round()
            .clamp(-(AQ_MAX_SIGNALED as f32), AQ_MAX_SIGNALED as f32) as i32;
        let newq = (self.last_qidx + sig * step).clamp(1, 255);
        let qs = qstep(newq as u32) as i32;
        (qs, self.qstep_base as f32 / qs as f32)
    }

    /// Current accumulated (qstep, resid_scale) without signaling — for SBs that
    /// emit delta 0 but must code at the decoder's accumulated qindex.
    pub(crate) fn current(&self) -> (i32, f32) {
        if !self.present {
            return (self.qstep_base, 1.0);
        }
        let qs = qstep(self.last_qidx as u32) as i32;
        (qs, self.qstep_base as f32 / qs as f32)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn per_sb(
        &mut self,
        enc: &mut RangeEncoder,
        yp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        width: usize,
        height: usize,
    ) -> (i32, f32) {
        if !self.present {
            enc.delta_q_signaled = 0;
            return (self.qstep_base, 1.0);
        }
        // Variance Boost: pick the representative 8x8 variance at the configured
        // octile, map it to a qindex delta (boost low-contrast, coarsen high-contrast),
        // and signal it through the same accumulator/qstep machinery as classic AQ.
        let mut subvars = [0f32; 64];
        let filled = sb_subblock_variances(yp, pw, sb_y, sb_x, width, height, &mut subvars);
        let target = if filled == 0 {
            self.base_q
        } else {
            let picked = sb_octile_variance(&mut subvars, self.vb_octile);
            let delta =
                variance_boost_delta(picked, self.ref_act, self.vb_strength, self.vb_boost_only);
            (self.base_q + delta).clamp(1, 255)
        };
        let step = 1i32 << AQ_RES_LOG2;
        let sig = (((target - self.last_qidx) as f32) / step as f32)
            .fast_round()
            .clamp(-(AQ_MAX_SIGNALED as f32), AQ_MAX_SIGNALED as f32) as i32;
        let newq = (self.last_qidx + sig * step).clamp(1, 255);
        self.last_qidx = newq;
        enc.delta_q_signaled = sig;
        let qs = qstep(newq as u32) as i32;
        (qs, self.qstep_base as f32 / qs as f32)
    }
}
