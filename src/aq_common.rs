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
impl AqLuma for u16 {
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
    let o = octile.clamp(1, 8) as usize;
    let idx = (o * 8 - 1).min(63);
    subvars[idx]
}

#[inline(always)]
pub(crate) fn f_fmlaf(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    ))]
    {
        f32::mul_add(a, b, c)
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        target_arch = "aarch64"
    )))]
    {
        a * b + c
    }
}

#[inline]
pub(crate) fn dirty_log1pf(d: f32) -> f32 {
    let ix = (1.0 + d).to_bits();
    let exponent = ix & 0x7f80_0000;
    let n = (exponent >> 23) as i32 - 0x7f;

    // Replacing the exponent with 127 normalizes 1+d to 1+t without division.
    // For n=0 use d itself so tiny inputs do not lose precision in (1+d)-1.
    let mantissa = f32::from_bits((ix & 0x007f_ffff) | 0x3f80_0000);
    let t = if n == 0 { d } else { mantissa - 1.0 };

    // Direct minimax ln(1+t) ~= t*P(t), t in [0, 1]. See tools/log1p.sollya.
    let mut p = 0.014539075084030628;
    p = f_fmlaf(p, t, -0.0675969123840332);
    p = f_fmlaf(p, t, 0.15056970715522766);
    p = f_fmlaf(p, t, -0.23573730885982513);
    p = f_fmlaf(p, t, 0.33125850558280945);
    p = f_fmlaf(p, t, -0.4998837411403656);
    p = f_fmlaf(p, t, 0.999998927116394);
    f_fmlaf(n as f32, std::f32::consts::LN_2, t * p)
}

/// Variance Boost qindex delta for one superblock. `picked_var` is the octile pick
/// (see [`sb_octile_variance`]); `ref_log` is the tile mean log-variance.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn variance_boost_delta(
    picked_var: f32,
    ref_log: f32,
    ref_pick: f32,
    strength: f32,
    boost_only: bool,
    mid_t: f32,
    cut_wide_t: f32,
) -> i32 {
    // Work in log-variance: compresses the huge dynamic range of variance and matches
    // the reference (which is a mean of log-variances).
    let v_log = dirty_log1pf(picked_var);
    let low_log = f_fmlaf(0.5, mid_t, 5.549_076); // (1.0 + 256.0).ln() at mid_t=0
    let max_boost = 18.0f32;
    let max_cut = 16.0f32 + 24.0 * cut_wide_t;
    // qindex per unit log-variance for each side.
    let boost_slope = 5.0 + 1.5 * mid_t;
    let cut_slope = 6.0 + 2.0 * mid_t;

    if v_log < low_log {
        // Low contrast: boost (negative delta). Deeper below threshold => stronger.
        let d = ((low_log - v_log) * boost_slope * strength).min(max_boost);
        -(d.fast_round() as i32)
    } else if boost_only {
        0
    } else {
        let anchor = if ref_pick > 0.0 {
            ref_pick
        } else {
            ref_log * 0.95
        };
        let over = (v_log - anchor.max(low_log)).max(0.0);
        let d = (over * cut_slope * strength).min(max_cut);
        d.fast_round() as i32
    }
}

/// AV2 entry point: the pre-widening signature (cut_wide_t = 0, byte-identical).
pub(crate) fn variance_boost_delta_av2(
    picked_var: f32,
    ref_log: f32,
    ref_pick: f32,
    strength: f32,
    boost_only: bool,
    mid_t: f32,
) -> i32 {
    variance_boost_delta(
        picked_var, ref_log, ref_pick, strength, boost_only, mid_t, 0.0,
    )
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
            min_q: 90,
            ..DarkAq::default()
        }
    }
}

#[inline]
fn laplacian_abs_sum<const STRIDE: usize>(buf: &[[f32; STRIDE]], h: usize, w: usize) -> (f32, u32) {
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for rows in buf[..h].array_windows::<3>() {
        let [top, middle, bottom] = rows;
        for ((&up, &down), &[left, center, right]) in top[1..w - 1]
            .iter()
            .zip(bottom[1..w - 1].iter())
            .zip(middle[..w].array_windows::<3>())
        {
            let l = 4.0 * center - up - down - left - right;
            sum += l.abs();
            n += 1;
        }
    }
    (sum, n)
}

#[inline]
fn box_downsample_2x<const SRC_STRIDE: usize, const DST_STRIDE: usize>(
    src: &[[f32; SRC_STRIDE]],
    h: usize,
    w: usize,
    dst: &mut [[f32; DST_STRIDE]],
) -> (usize, usize) {
    let (hh, ww) = (h / 2, w / 2);
    for (dst_row, rows) in dst[..hh]
        .iter_mut()
        .zip(src[..h].array_windows::<2>().step_by(2))
    {
        let [top, bottom] = rows;
        for (out, (&[top_left, top_right], &[bottom_left, bottom_right])) in
            dst_row[..ww].iter_mut().zip(
                top[..w]
                    .array_windows::<2>()
                    .step_by(2)
                    .zip(bottom[..w].array_windows::<2>().step_by(2)),
            )
        {
            *out = 0.25 * (top_left + top_right + bottom_left + bottom_right);
        }
    }
    (hh, ww)
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
        for (yp, dst) in yp[base..base + w].iter().zip(row.iter_mut()) {
            let v = yp.to_f32() * scale;
            *dst = v;
            sum += v;
        }
    }
    let mean = sum / (h * w) as f32;
    if h < 3 || w < 3 {
        return (mean, 0.0);
    }
    // Full-resolution |Laplacian| (interior 3×3).
    let (lap_full, nf) = laplacian_abs_sum(&buf, h, w);
    let lap_full = lap_full / nf as f32;
    // 2× box downsample, then |Laplacian| at the coarse scale.
    let (hh, ww) = (h / 2, w / 2);
    if hh < 3 || ww < 3 {
        return (mean, 0.0);
    }
    let mut half = [[0f32; 32]; 32];
    let (hh, ww) = box_downsample_2x(&buf, h, w, &mut half);
    if hh < 3 || ww < 3 {
        return (mean, 0.0);
    }
    let (lap_half, nh) = laplacian_abs_sum(&half, hh, ww);
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
    let dark_structure = dirty_log1pf(mid_energy * darkness);
    (dark_structure * d.scale)
        .min(d.max_qidx as f32)
        .max(0.0)
        .fast_round() as i32
}
