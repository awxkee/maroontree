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

/// Map superblock activity to a target qindex, centered on the tile mean
/// `ref_act`. Flat regions (below-average activity, banding/blocking visible) get
/// a finer quantizer; busy/textured regions (artifacts masked) get a coarser one.
/// Zero-mean by construction, so the average quantizer tracks the frame base.
///
/// Tuning knobs: `SLOPE` (qindex per unit log-activity) and `MAX_DELTA` (per-SB
/// clamp). Lower both for a gentle effect, raise for aggressive redistribution.
fn aq_target_qidx(base_q: i32, activity: f32, ref_act: f32) -> i32 {
    const SLOPE: f32 = 4.0;
    const MAX_DELTA: f32 = 20.0;
    let delta = ((activity - ref_act) * SLOPE).clamp(-MAX_DELTA, MAX_DELTA);
    (base_q + delta.fast_round() as i32).clamp(1, 255)
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
        }
    }

    /// Decide this superblock's quantizer from its `activity`, set
    /// `enc.delta_q_signaled`, and return `(qstep_sb, resid_scale)` to apply to the
    /// SB's luma and chroma (pass `resid_scale` to `encode_luma_sb` / `scale_resid`
    /// and reconstruct at `qstep_sb`). When AQ is off this is `(qstep_base, 1.0)`
    /// and signals 0. The caller still arms `enc.delta_q_pending` before the mode.
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
        let act = sb_activity(yp, pw, sb_y, sb_x, width, height);
        let target = aq_target_qidx(self.base_q, act, self.ref_act);
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
