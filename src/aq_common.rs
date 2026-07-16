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

use crate::util::FastRound;

pub(crate) trait AqLuma: Copy {
    fn to_f32(self) -> f32;
}
impl AqLuma for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}
impl AqLuma for i32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// The representative variance of a superblock for Variance Boost: the value at the
/// requested `octile` (1..=8) of the 64 sorted 8x8 variances. Octile 1 = the most
/// low-variance-biased pick (boost readily), octile 8 = only the maximum (boost only
/// when the whole SB is low-variance).
pub(crate) fn sb_octile_variance(subvars: &mut [f32; 64], octile: u8) -> f32 {
    subvars.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Octile o in 1..=8 maps to sorted index o*8 - 1 (o=1 -> 7, o=4 -> 31 (median-ish),
    // o=6 -> 47 (SVT-AV1-PSY default), o=8 -> 63 (max)).
    let o = octile.clamp(1, 8) as usize;
    let idx = (o * 8 - 1).min(63);
    subvars[idx]
}

/// Variance Boost qindex delta for one superblock. `picked_var` is the octile pick
/// (see [`sb_octile_variance`]); `ref_log` is the tile mean log-variance.
pub(crate) fn variance_boost_delta(
    picked_var: f32,
    ref_log: f32,
    strength: f32,
    boost_only: bool,
) -> i32 {
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

impl DarkAq {
    /// Disabled — no dark protection.
    pub(crate) fn off() -> Self {
        DarkAq {
            enabled: false,
            ..DarkAq::default()
        }
    }

    /// Enabled with the calibrated defaults.
    pub(crate) fn on() -> Self {
        DarkAq {
            enabled: true,
            ..DarkAq::default()
        }
    }
}

pub(crate) fn dark_structure_stats<T: AqLuma>(
    yp: &[T],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    scale: f32,
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
            let v = yp[base + c].to_f32() * scale;
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

/// Extra qindex reduction (>= 0) for a dark, structured SB. 0 when disabled, out of the
/// gated quality range, or the SB carries no cross-scale structure. `scale` normalizes
/// the plane to 8-bit range for [`dark_structure_stats`] (AV2: `1.0`; AV1: `1/(1<<(bd-8))`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dark_protection<T: AqLuma>(
    d: &DarkAq,
    base_q: i32,
    yp: &[T],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    scale: f32,
) -> i32 {
    if !d.enabled || base_q < d.min_q {
        return 0;
    }
    let (mean, mid_energy) = dark_structure_stats(yp, pw, sb_y, sb_x, width, height, scale);
    if mid_energy <= 0.0 {
        return 0;
    }
    // Darker SBs get a heavier weight; `dark_weight` is 1.0 at/above `dark_ref` and
    // rises toward `max_weight` as the SB darkens. Subtracting 1 makes the protection
    // vanish for bright/mid SBs (so it doesn't just boost all texture) and scale with
    // how far below the reference the SB sits.
    let dark_weight = ((d.mean_floor + d.dark_ref) / (d.mean_floor + mean))
        .powf(d.gamma)
        .clamp(1.0, d.max_weight);
    let darkness = dark_weight - 1.0;
    if darkness <= 0.0 {
        return 0;
    }
    let dark_structure = (mid_energy * darkness).ln_1p();
    (dark_structure * d.scale)
        .min(d.max_qidx as f32)
        .max(0.0)
        .fast_round() as i32
}
