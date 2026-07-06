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
use crate::av2::helpers::sum_sumsq_f32;
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
    let mut sum = 0f32;
    let mut sum2 = 0f32;
    for r in 0..h {
        let base = (sb_y + r) * pw + sb_x;
        let (row_sum, row_sum2) = sum_sumsq_f32(&yp[base..base + w]);
        sum += row_sum;
        sum2 += row_sum2;
    }
    let n = (h * w) as f32;
    let mean = sum / n;
    let var = (sum2 / n - mean * mean).max(0.0);
    (1.0 + var).ln()
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
    let mut acc = 0f32;
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
            let mut sum = 0f32;
            let mut sum2 = 0f32;
            for r in 0..h {
                let base = (y0 + r) * pw + x0;
                let (row_sum, row_sum2) = sum_sumsq_f32(&yp[base..base + w]);
                sum += row_sum;
                sum2 += row_sum2;
            }
            let n = (h * w) as f32;
            let mean = sum / n;
            let var = (sum2 / n - mean * mean).max(0.0);
            out[idx] = var;
            acc += var;
            filled += 1;
        }
    }
    if filled == 0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    // Patch out-of-frame subblocks with the in-frame mean (neutral for the octile).
    let mean = acc / filled as f32;
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

/// Mean luma and a two-scale, noise-suppressed mid-band energy for the SB. `mid_energy`
/// is the geometric mean of the full-resolution mean `|Laplacian|` and the mean
/// `|Laplacian|` of a 2× box-downsample. Structured detail (edges, rock texture) survives
/// the downsample and scores on BOTH scales; isolated one-pixel sensor noise averages
/// away at the coarse scale, so the geometric mean discounts it — unlike raw variance,
/// which would strongly over-protect dark noise.
fn dark_structure_stats(
    yp: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
) -> (f32, f32) {
    let h = height.saturating_sub(sb_y).min(64);
    let w = width.saturating_sub(sb_x).min(64);
    if h == 0 || w == 0 {
        return (0.0, 0.0);
    }
    let mut buf = [[0f32; 64]; 64];
    let mut sum = 0f32;
    for (r, row) in buf.iter_mut().enumerate().take(h) {
        let base = (sb_y + r) * pw + sb_x;
        for c in 0..w {
            let v = yp[base + c];
            row[c] = v;
            sum += v;
        }
    }
    let mean = sum / (h * w) as f32;
    if h < 3 || w < 3 {
        return (mean, 0.0);
    }
    // Full-resolution |Laplacian| (interior 3×3).
    let mut lap_full = 0f32;
    let mut nf = 0u32;
    for r in 1..h - 1 {
        for c in 1..w - 1 {
            let l = 4.0 * buf[r][c] - buf[r - 1][c] - buf[r + 1][c] - buf[r][c - 1] - buf[r][c + 1];
            lap_full += l.abs();
            nf += 1;
        }
    }
    let lap_full = lap_full / nf as f32;
    // 2× box downsample, then |Laplacian| at the coarse scale.
    let (hh, ww) = (h / 2, w / 2);
    if hh < 3 || ww < 3 {
        return (mean, 0.0);
    }
    let mut half = [[0f32; 32]; 32];
    for r in 0..hh {
        for c in 0..ww {
            half[r][c] = 0.25
                * (buf[2 * r][2 * c]
                    + buf[2 * r][2 * c + 1]
                    + buf[2 * r + 1][2 * c]
                    + buf[2 * r + 1][2 * c + 1]);
        }
    }
    let mut lap_half = 0f32;
    let mut nh = 0u32;
    for r in 1..hh - 1 {
        for c in 1..ww - 1 {
            let l = 4.0 * half[r][c]
                - half[r - 1][c]
                - half[r + 1][c]
                - half[r][c - 1]
                - half[r][c + 1];
            lap_half += l.abs();
            nh += 1;
        }
    }
    let lap_half = lap_half / nh as f32;
    (mean, (lap_full * lap_half).sqrt())
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
/// invocation (one tile). Inter-capable paths should use [`AqState::per_sb_probe`]
/// while choosing a mode and call [`AqState::per_sb`] only when the selected block
/// carries delta-Q syntax. A full-superblock `skip_txfm=1` block must not advance
/// `last_qidx`, because the decoder does not read delta-Q for that block.
/// One superblock's precomputed AQ decision. Wavefront blocker #1: the parallel
/// (diagonal-order) decide can't thread `per_sb`'s raster-serial `last_qidx`
/// accumulator, so a cheap serial pre-pass ([`AqState::precompute_grid`]) yields
/// this grid for the decide to read; `sig` feeds the serial emit's `delta_q`.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct AqCell {
    pub(crate) qs: i32,
    pub(crate) resid_scale: f32,
    pub(crate) qs_c: i32,
    pub(crate) resid_scale_c: f32,
    pub(crate) sig: i32,
    /// The committed qindex this SB codes at (`newq` for full-interior, the entering
    /// accumulator for edge/`!present`). Mirrors `current_qidx()` after this SB.
    pub(crate) qidx: i32,
    /// The accumulator qstep *entering* this SB, before it commits — i.e. what
    /// `aqs.current()` returns at the partition probe (the previous full-interior
    /// SB's `newq`). In a raster serial run this equals the prior cell's state; a
    /// wavefront decide reads it from the grid instead of a serial accumulator.
    pub(crate) qs_before: i32,
    pub(crate) resid_scale_before: f32,
}

/// Dark-structured-detail protection. An INDEPENDENT AQ signal (combined with the
/// flat-region variance boost by `max`) that gives extra qindex reduction to dark
/// superblocks carrying real structure — rock texture, boundaries — that raw variance
/// would coarsen and quantize to extinction at low quality. Gated to `base_q >= min_q`
/// (the low-quality range where qstep changes fastest), and driven by a two-scale
/// noise-suppressed energy rather than raw variance (which would over-protect noise).
#[derive(Clone, Copy, Debug)]
pub struct DarkAq {
    pub enabled: bool,
    /// Only active at base_q >= this (qstep-extinction range, roughly q<=45 in 8-bit).
    pub min_q: i32,
    pub mean_floor: f32,
    pub dark_ref: f32,
    pub gamma: f32,
    pub max_weight: f32,
    /// qindex units per unit of `log1p(mid_energy * dark_weight)`.
    pub scale: f32,
    /// Cap on the extra boost (qindex reduction).
    pub max_qidx: i32,
}

impl Default for DarkAq {
    fn default() -> Self {
        DarkAq {
            enabled: false,
            min_q: 150,
            mean_floor: 16.0,
            dark_ref: 56.0,
            gamma: 1.2,
            // `darkness = max_weight - 1` is the effective multiplier for the darkest SBs.
            max_weight: 4.5,
            scale: 4.0,
            max_qidx: 16,
        }
    }
}

pub(crate) struct AqState {
    present: bool,
    base_q: i32,
    qstep_base: i32,
    /// Frame-level chroma qindex delta (AVM base_uv_ac_delta_q); chroma qstep at an
    /// SB is `qstep((sb_qidx + uv_delta).clamp(1,255))`, matching decoder `get_q`.
    uv_delta: i32,
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
    dark: DarkAq,
}

impl AqState {
    /// `present` = frame enables delta-Q; `qstep_base` = qstep at the frame base
    /// qindex; `ref_act` = tile mean activity (see [`tile_ref_activity`]).
    pub(crate) fn new(
        present: bool,
        base_q: i32,
        qstep_base: i32,
        ref_act: f32,
        uv_delta: i32,
    ) -> Self {
        AqState {
            present,
            base_q,
            qstep_base,
            uv_delta,
            ref_act,
            last_qidx: base_q,
            vb_octile: 6,
            vb_strength: 1.0,
            vb_boost_only: false,
            dark: DarkAq::default(),
        }
    }

    /// Enable/configure the dark-structured-detail protection term (see [`DarkAq`]).
    pub(crate) fn with_dark_aq(mut self, dark: DarkAq) -> Self {
        self.dark = dark;
        self
    }

    /// Chroma (qstep, resid_scale) at qindex `q`: decoder chroma dequant is
    /// `get_q(clamp(q + uv_delta, 1, 255))` (equal ac/dc), residual pre-scale keeps
    /// the shared basis calibrated at `qstep_base`.
    fn chroma_at(&self, q: i32) -> (i32, f32) {
        let qc = qstep((q + self.uv_delta).clamp(1, 255) as u32) as i32;
        (qc, self.qstep_base as f32 / qc as f32)
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

    /// Extra qindex reduction (>= 0) for a dark, structured SB. 0 when disabled, out of
    /// the gated quality range, or the SB carries no cross-scale structure.
    fn dark_protection(
        &self,
        yp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        width: usize,
        height: usize,
    ) -> i32 {
        let d = &self.dark;
        if !d.enabled || self.base_q < d.min_q {
            return 0;
        }
        let (mean, mid_energy) = dark_structure_stats(yp, pw, sb_y, sb_x, width, height);
        if mid_energy <= 0.0 {
            return 0;
        }
        // Darker SBs get a heavier weight; `dark_weight` is 1.0 at/above `dark_ref` and
        // rises toward `max_weight` as the SB darkens. Subtracting 1 makes the protection
        // vanish for bright/mid SBs (so it doesn't just boost all texture) and scale with
        // how far below the reference the SB sits.
        let dark_weight = (((d.mean_floor + d.dark_ref) / (d.mean_floor + mean)).powf(d.gamma))
            .clamp(1.0, d.max_weight);
        let darkness = dark_weight - 1.0;
        if darkness <= 0.0 {
            return 0;
        }
        let dark_structure = (mid_energy * darkness).ln_1p();
        ((dark_structure * d.scale).min(d.max_qidx as f32))
            .max(0.0)
            .fast_round() as i32
    }

    /// The SB's target qindex: the flat-region variance boost combined with the
    /// dark-structured-detail protection. `protection = max(flat_boost, dark_boost)`;
    /// the coarsen (positive) side of the variance boost is kept only when neither
    /// protection signal fires. Shared by `per_sb`, `per_sb_probe` and `precompute_grid`
    /// so they stay bit-exact.
    fn sb_target(
        &self,
        yp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        width: usize,
        height: usize,
    ) -> i32 {
        let mut subvars = [0f32; 64];
        let filled = sb_subblock_variances(yp, pw, sb_y, sb_x, width, height, &mut subvars);
        if filled == 0 {
            return self.base_q;
        }
        let picked = sb_octile_variance(&mut subvars, self.vb_octile);
        let vb_delta =
            variance_boost_delta(picked, self.ref_act, self.vb_strength, self.vb_boost_only);
        let dark = self.dark_protection(yp, pw, sb_y, sb_x, width, height);
        let flat_boost = (-vb_delta).max(0);
        let protection = flat_boost.max(dark);
        let delta = if protection > 0 {
            -protection
        } else {
            vb_delta
        };
        (self.base_q + delta).clamp(1, 255)
    }

    pub(crate) fn current_qidx(&self) -> i32 {
        self.last_qidx
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn per_sb_probe(
        &self,
        yp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        width: usize,
        height: usize,
    ) -> (i32, f32, i32, f32) {
        if !self.present {
            let (qc, rc) = self.chroma_at(self.base_q);
            return (self.qstep_base, 1.0, qc, rc);
        }
        let target = self.sb_target(yp, pw, sb_y, sb_x, width, height);
        let step = 1i32 << AQ_RES_LOG2;
        let sig = (((target - self.last_qidx) as f32) / step as f32)
            .fast_round()
            .clamp(-(AQ_MAX_SIGNALED as f32), AQ_MAX_SIGNALED as f32) as i32;
        let newq = (self.last_qidx + sig * step).clamp(1, 255);
        let qs = qstep(newq as u32) as i32;
        let (qc, rc) = self.chroma_at(newq);
        (qs, self.qstep_base as f32 / qs as f32, qc, rc)
    }

    /// Current accumulated (qstep, resid_scale) without signaling — for SBs that
    /// emit delta 0 but must code at the decoder's accumulated qindex.
    pub(crate) fn current(&self) -> (i32, f32, i32, f32) {
        if !self.present {
            let (qc, rc) = self.chroma_at(self.base_q);
            return (self.qstep_base, 1.0, qc, rc);
        }
        let qs = qstep(self.last_qidx as u32) as i32;
        let (qc, rc) = self.chroma_at(self.last_qidx);
        (qs, self.qstep_base as f32 / qs as f32, qc, rc)
    }

    /// Serial raster-order pre-pass producing the per-SB AQ grid, **bit-exact** with
    /// the serial `per_sb`(full-interior SBs) / `current`(edge & `!present`) sequence
    /// the main loop runs. This removes the raster-serial `last_qidx` dependency so a
    /// wavefront (diagonal) decide can read `grid[r*sb_cols + c]` instead of calling
    /// `per_sb`; the serial emit pass then replays `sig` into `enc.delta_q_signaled`.
    /// `per_sb` is invoked ⟺ `full_interior` (whole-64 fast path OR `sb_use_split`);
    /// edge SBs (`sb_walk`) use `current()` — mirrored here exactly.
    #[allow(dead_code)]
    pub(crate) fn precompute_grid(
        &self,
        yp: &[f32],
        pw: usize,
        width: usize,
        height: usize,
        needs_partition: bool,
    ) -> Vec<AqCell> {
        let sb_cols = width.div_ceil(64);
        let sb_rows = height.div_ceil(64);
        let step = 1i32 << AQ_RES_LOG2;
        let mut last_qidx = self.last_qidx;
        let mut grid = Vec::with_capacity(sb_cols * sb_rows);
        for r in 0..sb_rows {
            for c in 0..sb_cols {
                let sb_y = r * 64;
                let sb_x = c * 64;
                // `per_sb` (accumulate) is called ⟺ the SB is NOT a native edge walk,
                // i.e. `!sb_walk` = `full_interior || !needs_partition`. When the frame
                // is padded (`!needs_partition`) even geometric-edge SBs take the
                // whole-64 path and accumulate; only native-partition edge SBs
                // (`needs_partition && !full_interior`) use `current()` (no accumulate).
                let full_interior = (sb_x + 64 <= width && sb_y + 64 <= height) || !needs_partition;
                // Accumulator qstep entering this SB (== `aqs.current()` at the probe).
                let qs_before = if self.present {
                    qstep(last_qidx as u32) as i32
                } else {
                    self.qstep_base
                };
                let resid_scale_before = self.qstep_base as f32 / qs_before as f32;
                let cell = if !self.present {
                    let (qc, rc) = self.chroma_at(self.base_q);
                    AqCell {
                        qs: self.qstep_base,
                        resid_scale: 1.0,
                        qs_c: qc,
                        resid_scale_c: rc,
                        sig: 0,
                        qidx: self.base_q,
                        qs_before,
                        resid_scale_before,
                    }
                } else if full_interior {
                    // Mirrors `per_sb`: variance → target → signalled delta → newq (accumulate).
                    let target = self.sb_target(yp, pw, sb_y, sb_x, width, height);
                    let sig = (((target - last_qidx) as f32) / step as f32)
                        .fast_round()
                        .clamp(-(AQ_MAX_SIGNALED as f32), AQ_MAX_SIGNALED as f32)
                        as i32;
                    let newq = (last_qidx + sig * step).clamp(1, 255);
                    last_qidx = newq;
                    let qs = qstep(newq as u32) as i32;
                    let (qc, rc) = self.chroma_at(newq);
                    AqCell {
                        qs,
                        resid_scale: self.qstep_base as f32 / qs as f32,
                        qs_c: qc,
                        resid_scale_c: rc,
                        sig,
                        qidx: newq,
                        qs_before,
                        resid_scale_before,
                    }
                } else {
                    // Mirrors `current()`: no accumulation, no signalled delta.
                    let qs = qstep(last_qidx as u32) as i32;
                    let (qc, rc) = self.chroma_at(last_qidx);
                    AqCell {
                        qs,
                        resid_scale: self.qstep_base as f32 / qs as f32,
                        qs_c: qc,
                        resid_scale_c: rc,
                        sig: 0,
                        qidx: last_qidx,
                        qs_before,
                        resid_scale_before,
                    }
                };
                grid.push(cell);
            }
        }
        grid
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
    ) -> (i32, f32, i32, f32) {
        if !self.present {
            enc.delta_q_signaled = 0;
            let (qc, rc) = self.chroma_at(self.base_q);
            return (self.qstep_base, 1.0, qc, rc);
        }
        // Variance Boost (flat-region protection) combined with dark-structured-detail
        // protection, signalled through the same accumulator/qstep machinery as classic AQ.
        let target = self.sb_target(yp, pw, sb_y, sb_x, width, height);
        let step = 1i32 << AQ_RES_LOG2;
        let sig = (((target - self.last_qidx) as f32) / step as f32)
            .fast_round()
            .clamp(-(AQ_MAX_SIGNALED as f32), AQ_MAX_SIGNALED as f32) as i32;
        let newq = (self.last_qidx + sig * step).clamp(1, 255);
        self.last_qidx = newq;
        enc.delta_q_signaled = sig;
        let qs = qstep(newq as u32) as i32;
        let (qc, rc) = self.chroma_at(newq);
        (qs, self.qstep_base as f32 / qs as f32, qc, rc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_protection_targets_dark_structure_only() {
        // 64x64 SBs at three luma levels, each with a coherent striped texture (survives
        // the 2x downsample, so it registers as structure not noise).
        let build = |mean: f32| -> Vec<f32> {
            let mut p = vec![0f32; 64 * 64];
            for r in 0..64 {
                for c in 0..64 {
                    // 4px-wide stripes: strong at both scales.
                    p[r * 64 + c] = mean + if (c / 4) % 2 == 0 { 12.0 } else { -12.0 };
                }
            }
            p
        };
        let base_q = 190; // in the gated range (>= 150)
        let st =
            AqState::new(true, base_q, qstep(base_q as u32) as i32, 10.0, 0).with_dark_aq(DarkAq {
                enabled: true,
                ..DarkAq::default()
            });
        let dark = build(36.0);
        let bright = build(180.0);
        let flat_dark = vec![24.0f32; 64 * 64]; // dark but no structure
        let d_dark = st.dark_protection(&dark, 64, 0, 0, 64, 64);
        let d_bright = st.dark_protection(&bright, 64, 0, 0, 64, 64);
        let d_flat = st.dark_protection(&flat_dark, 64, 0, 0, 64, 64);
        assert!(
            d_dark > 0,
            "dark structured SB should be protected, got {d_dark}"
        );
        assert_eq!(d_bright, 0, "bright structured SB must not be protected");
        assert_eq!(
            d_flat, 0,
            "dark flat SB (no structure) must not be protected"
        );
        // Gate: disabled below min_q.
        let st_hi = AqState::new(true, 100, qstep(100) as i32, 10.0, 0).with_dark_aq(DarkAq {
            enabled: true,
            ..DarkAq::default()
        });
        assert_eq!(
            st_hi.dark_protection(&dark, 64, 0, 0, 64, 64),
            0,
            "gated out below min_q"
        );
    }

    #[test]
    fn precompute_grid_matches_serial_per_sb() {
        // The wavefront AQ pre-pass must be bit-exact with the serial
        // `per_sb`(full-interior) / `current`(edge) sequence the main loop runs.
        let base_q = 120;
        let base_step = qstep(base_q as u32) as i32;
        // Non-SB-aligned dims so the grid mixes full-interior AND edge SBs.
        let (width, height) = (160usize, 96usize);
        let pw = crate::av2::helpers::sb_align(width);
        let ph = crate::av2::helpers::sb_align(height);
        let yp: Vec<f32> = (0..pw * ph).map(|i| (i * 37 % 251) as f32).collect();
        let mk = || AqState::new(true, base_q, base_step, 10.0, 0);
        let sb_cols = width.div_ceil(64);
        let sb_rows = height.div_ceil(64);

        // `needs_partition` selects the edge rule: `true` = native edge SBs use
        // `current()` (no accumulate); `false` = padded frame, edge SBs also
        // `per_sb`. The main loop's accumulate condition is `!sb_walk`
        // (`full_interior || !needs_partition`) — mirror it in the reference.
        for needs_partition in [true, false] {
            let grid = mk().precompute_grid(&yp, pw, width, height, needs_partition);
            let mut aq = mk();
            let mut enc = crate::av2::entropy::RangeEncoder::new();
            for r in 0..sb_rows {
                for c in 0..sb_cols {
                    let (sb_y, sb_x) = (r * 64, c * 64);
                    let full_interior =
                        (sb_x + 64 <= width && sb_y + 64 <= height) || !needs_partition;
                    let (qs, rs, qc, rc) = if full_interior {
                        aq.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height)
                    } else {
                        enc.delta_q_signaled = 0;
                        aq.current()
                    };
                    let cell = grid[r * sb_cols + c];
                    assert_eq!(cell.qs, qs, "qs at ({r},{c}) np={needs_partition}");
                    assert_eq!(cell.qs_c, qc, "qs_c at ({r},{c}) np={needs_partition}");
                    assert_eq!(cell.sig, enc.delta_q_signaled, "sig at ({r},{c})");
                    assert_eq!(cell.resid_scale.to_bits(), rs.to_bits(), "rs at ({r},{c})");
                    assert_eq!(cell.resid_scale_c.to_bits(), rc.to_bits(), "rs_c ({r},{c})");
                    assert_eq!(cell.qidx, aq.current_qidx(), "qidx at ({r},{c})");
                }
            }
        }
    }

    #[test]
    fn probe_does_not_advance_delta_q_accumulator() {
        let base_q = 120;
        let base_step = qstep(base_q as u32) as i32;
        let mut aq =
            AqState::new(true, base_q, base_step, 10.0, 0).with_variance_boost(6, 1.0, true);
        let plane = vec![128.0f32; 64 * 64];

        let before = aq.current();
        let probed = aq.per_sb_probe(&plane, 64, 0, 0, 64, 64);
        assert_ne!(probed.0, before.0, "flat block should request an AQ boost");
        assert_eq!(
            aq.current().0,
            before.0,
            "probe must preserve decoder qindex state"
        );

        let mut enc = RangeEncoder::new();
        enc.delta_q_present = true;
        let committed = aq.per_sb(&mut enc, &plane, 64, 0, 0, 64, 64);
        assert_eq!(committed.0, probed.0);
        assert_eq!(aq.current().0, probed.0);
        assert_ne!(enc.delta_q_signaled, 0);
    }
}
