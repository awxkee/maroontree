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

// Adaptive quantization: variance boost, superblock activity, per-tile `AqCtx` and the precomputed `AqCell` grid.

const AQ_RES_LOG2: u8 = 2;
pub(crate) const AQ_DELTA_Q_RES_LOG2: u8 = AQ_RES_LOG2;
const AQ_MAX_STEPS: i32 = 12;

/// Base-shift experiment scaffolding. **Unreachable as shipped.**
///
/// `BASEQ_SHIFT_K` is a plain const with no setter, and `baseq_shift` returns 0
/// unconditionally when it is 0. So `dispatch.rs`'s `if shift != 0` block never
/// runs, `VarianceBoost::base_shift` is always 0, and the
/// `if delta < 0 && self.aq.base_shift > 0` re-anchoring branch in
/// `aq_sb_target` is dead with it.
///
/// Unlike `aq_slope`/`aq_max_delta` (revivable via
/// `EncodeConfig::with_variance_boost(false)`), there is NO configuration that
/// makes this live — verified by tracing every assignment to `base_shift`.
/// Safe to delete along with `baseq_shift`, the `base_shift` field, the
/// dispatch block and the `aq_sb_target` branch.
pub(crate) const BASEQ_SHIFT_K: i32 = 0;
pub(crate) fn baseq_shift(base_q_idx: u8) -> i32 {
    let qi = base_q_idx as i32;
    let k = BASEQ_SHIFT_K;
    if k == 0 || qi <= 40 {
        0
    } else if qi < 56 {
        k * (qi - 40) / 16
    } else if qi <= 112 {
        k
    } else if qi < 144 {
        k * (144 - qi) / 32
    } else {
        0
    }
}
/// qindex per unit of log-activity (how hard flat vs busy regions are pushed).
///
/// ONLY REACHED WITH VARIANCE BOOST DISABLED. Its sole consumer is
/// `aq_target_qidx`, which lives in the `else` of `if self.aq.vb_enabled` in
/// `aq_sb_target`; Variance Boost is on by default, so the default encoder
/// never evaluates this. A liveness sweep therefore reports it as dead — it is
/// not: `EncodeConfig::variance_boost(false)` makes this path live again.
/// Do not delete it, and do not tune it against default-config measurements.
fn aq_slope() -> f32 {
    crate::tuning::get().aq_slope
}
/// per-superblock qindex delta clamp, before res-quantization.
/// Same reachability caveat as [`aq_slope`].
fn aq_max_delta() -> f32 {
    crate::tuning::get().aq_max_delta
}

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
    /// Mid-band base-q shift this frame was encoded with (see [`baseq_shift`]);
    /// the AQ protection side anchors to `base_q - base_shift`.
    pub base_shift: i32,
    pub qm: QmLevels,
    /// 4:0:0 alpha profile: allow 16x16/32x32 luma leaves (bottom-up priced).
    /// Plain grey images keep the 8x8 leaves, which measured better there.
    pub mono_large: bool,
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
            base_shift: 0,
            qm: QmLevels::FLAT,
            mono_large: false,
        }
    }

    pub(crate) fn on() -> Self {
        VarianceBoost {
            enabled: true,
            octile: 6,
            strength: 1.0,
            boost_only: false,
            dark: DarkAq::on(),
            base_shift: 0,
            qm: QmLevels::FLAT,
            mono_large: false,
        }
    }
}

#[inline(never)]
pub(crate) fn sb_activity(
    yp: &[u16],
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
    let (sum, sum_sq) = block_moments_i32(yp, pw, sb_x, sb_y, w, h);
    let n = (h * w) as f32;
    let mean = sum as f32 / n;
    let var = (sum_sq as f32 / n - mean * mean).max(0.0);
    dirty_log1pf(var)
}

fn tile_ref_activity(yp: &[u16], pw: usize, w: usize, h: usize) -> f32 {
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

/// Tile mean of log1p(octile-6 8x8 subblock variance) — the cut-side anchor
/// in the SAME domain as the per-SB `picked` value.
fn tile_ref_picked(yp: &[u16], pw: usize, w: usize, h: usize, octile: u8, bd: u8) -> f32 {
    let var_scale = 1.0 / (1u32 << (2 * (bd - 8))) as f32;
    let mut sum = 0f32;
    let mut cnt = 0f32;
    for sb_y in (0..h).step_by(64) {
        for sb_x in (0..w).step_by(64) {
            let mut subvars = [0f32; 64];
            let filled = aq_sb_subblock_variances(yp, pw, sb_y, sb_x, w, h, &mut subvars);
            if filled == 0 {
                continue;
            }
            let picked = crate::aq_common::sb_octile_variance(&mut subvars, octile);
            sum += dirty_log1pf(picked * var_scale);
            cnt += 1.0;
        }
    }
    if cnt > 0.0 { sum / cnt } else { 5.0 }
}

fn aq_params() -> (f32, f32, f32) {
    // (slope, max delta, coarsen scale). Coarsen scale 1.0 == pure variance.
    (aq_slope(), aq_max_delta(), 1.0)
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
    yp: &[u16],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    out: &mut [f32; 64],
) -> usize {
    let mut filled = 0usize;
    let mut acc = 0f32;
    for (by, row) in out.as_chunks_mut::<8>().0.iter_mut().take(8).enumerate() {
        for (bx, out) in row.iter_mut().enumerate() {
            let y0 = sb_y + by * 8;
            let x0 = sb_x + bx * 8;
            let h = height.saturating_sub(y0).min(8);
            let w = width.saturating_sub(x0).min(8);
            if h == 0 || w == 0 {
                *out = f32::NAN; // out-of-frame, patched below
                continue;
            }
            let (sum, sum_sq) = block_moments_i32(yp, pw, x0, y0, w, h);
            let n = (h * w) as f32;
            let mean = sum as f32 / n;
            let var = (sum_sq as f32 / n - mean * mean).max(0.0);
            *out = var;
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
#[derive(Clone, PartialEq)]
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
    /// Decoder qindex on entry to the current superblock. A whole-SB skip does
    /// not execute `read_delta_qindex`, so its tentative AQ change is rolled
    /// back to this value.
    prev_qidx: i32,
    /// tile mean activity, the zero-delta reference (see [`tile_ref_activity`]).
    ref_act: f32,
    /// Same-domain cut reference: tile mean of log1p(octile-6 8x8 variance) —
    /// the anchor the coarsen side compares against (the whole-SB `ref_act`
    /// is systematically higher than the octile domain, which had silently
    /// disabled cuts; see `variance_boost_delta`).
    ref_pick: f32,
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
    /// Mid-band base-shift the frame was encoded with; protection deltas
    /// anchor to `base_q - base_shift` (see `aq_sb_target`).
    base_shift: i32,
}

impl AqCtx {
    fn off() -> Self {
        AqCtx {
            enabled: false,
            base_q: 0,
            res_log2: 0,
            cur_qidx: 0,
            prev_qidx: 0,
            ref_act: 0.0,
            ref_pick: 0.0,
            read_deltas: false,
            pending: 0,
            vb_enabled: false,
            vb_octile: 6,
            vb_strength: 1.0,
            vb_boost_only: false,
            dark: DarkAq::off(),
            base_shift: 0,
        }
    }
}

/// One superblock's precomputed AQ state: the post-clamp qindex the SB codes
/// with and the signaled `reducedDeltaQIndex` step count that got it there.
/// Produced in raster order by `precompute_aq_grid`; indexed `row * cols + col`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AqCell {
    /// Raster-chain qindex BEFORE this SB (the anchor). Cells must start from
    /// this — NOT the worker's own accumulator: the clamp(1, 255) at
    /// near-lossless qindex breaks the shared-lattice property that otherwise
    /// makes any-prev derivations converge (kodak20 q99 t8 silent corruption).
    prev: u8,
    newq: u8,
    steps: i32,
}
