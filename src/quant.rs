/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::qm_tables::AV1_IQM;
use crate::util::FastRound;

pub(crate) const QM_FLAT_LEVEL: u8 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QmLevels {
    pub y: u8,
    pub u: u8,
    pub v: u8,
}

impl QmLevels {
    pub(crate) const FLAT: Self = Self {
        y: QM_FLAT_LEVEL,
        u: QM_FLAT_LEVEL,
        v: QM_FLAT_LEVEL,
    };

    pub(crate) fn uniform(level: u8) -> Self {
        let level = level.min(QM_FLAT_LEVEL);
        Self {
            y: level,
            u: level,
            v: level,
        }
    }
}

#[inline]
fn qm_offset(w: usize, h: usize) -> usize {
    match (w, h) {
        (4, 4) => 0,
        (8, 8) => 16,
        (16, 16) => 80,
        (32, 32) => 336,
        (4, 8) => 1360,
        (8, 4) => 1392,
        (8, 16) => 1424,
        (16, 8) => 1552,
        (16, 32) => 1680,
        (32, 16) => 2192,
        (4, 16) => 2704,
        (16, 4) => 2768,
        (8, 32) => 2832,
        (32, 8) => 3088,
        _ => panic!("unsupported AV1 QM transform size {w}x{h}"),
    }
}

/// The inverse QM table row for one transform size, or `None` when flat.
/// Resolving the table offset ONCE per block instead of once per coefficient:
/// `qm_offset` is a 14-arm match that sat inside every dequant/trellis loop.
#[inline]
pub(crate) fn qm_row(level: u8, chroma: bool, w: usize, h: usize) -> Option<&'static [u8]> {
    if level >= QM_FLAT_LEVEL {
        return None;
    }
    let off = qm_offset(w, h);
    Some(&AV1_IQM[level as usize][chroma as usize][off..off + w * h])
}

#[inline]
pub(crate) fn qm_weight(level: u8, chroma: bool, rc: usize, w: usize, h: usize) -> u8 {
    if level >= QM_FLAT_LEVEL {
        return 32;
    }
    // AV1_IQM stores dav1d's runtime tables verbatim, which are already in
    // dav1d's transposed coefficient order == our x-major cf layout
    // (`rc = x*h + y`), so the index is linear — no remap.
    AV1_IQM[level as usize][chroma as usize][qm_offset(w, h) + rc]
}

/// DC dequant value `dav1d_dq_tbl[bd][q][0]` for bit_depth 8/10/12.
pub(crate) fn dc_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::DC_QLOOKUP_10,
        12 => &crate::coef_q::DC_QLOOKUP_12,
        _ => &crate::coef_q::DC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}
/// AC dequant value `dav1d_dq_tbl[bd][q][1]` for bit_depth 8/10/12.
pub(crate) fn ac_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::AC_QLOOKUP_10,
        12 => &crate::coef_q::AC_QLOOKUP_12,
        _ => &crate::coef_q::AC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}

pub(crate) fn top_ease_t(base_q_idx: u8) -> f32 {
    let t = crate::tuning::get();
    ((t.top_ease_knee - base_q_idx as f32) / t.top_ease_width).clamp(0.0, 1.0)
}

#[inline]
pub(crate) fn top_ease() -> f32 {
    crate::tuning::get().top_ease
}

#[inline]
fn qm_high_quality_t(base_q_idx: u8) -> f32 {
    let t = crate::tuning::get();
    ((t.qm_hq_knee - base_q_idx as f32) / t.qm_hq_width).clamp(0.0, 1.0)
}

pub(crate) fn qm_level_law(base_q_idx: u8, sub: usize) -> u8 {
    let subsampled = sub != 0;
    const HI: f32 = 48.0; // level 8 at this qindex
    const LO: f32 = 128.0; // level 0 at and above this qindex
    let qi = base_q_idx as f32;
    if qi <= 20.0 {
        if subsampled { 8 } else { 15 }
    } else if qi <= HI {
        let top = if subsampled { 8.0 } else { 15.0 };
        (top + (8.0 - top) * (qi - 20.0) / (HI - 20.0)).fast_round() as u8
    } else if qi >= LO {
        0
    } else {
        (8.0 * (LO - qi) / (LO - HI)).fast_round() as u8
    }
}

pub(crate) fn qm_chroma_level_law(base_q_idx: u8, sub: usize) -> u8 {
    if sub != 0 {
        return qm_level_law(base_q_idx, sub);
    }
    const TOP: u8 = 2; // chroma level from the top through the mid band
    let pinned = TOP.min(qm_level_law(base_q_idx, sub));
    let t = top_ease() * qm_high_quality_t(base_q_idx);
    if t > 0.0 {
        (pinned as f32 + (15.0 - pinned as f32) * t).round() as u8
    } else {
        pinned
    }
}

const UV420_MID_D: i32 = 28;
fn uv420_mid_delta(base_q_idx: u8) -> i32 {
    let qi = base_q_idx as i32;
    let d = UV420_MID_D;
    if qi <= 20 {
        0
    } else if qi < 64 {
        -(d * (qi - 20)) / 44
    } else if qi <= 96 {
        -d
    } else if qi < 136 {
        -(d * (136 - qi)) / 40
    } else {
        0
    }
}

fn top_444_uac() -> i32 {
    crate::tuning::get().top_444_uac as i32
}

fn uv422_mid_delta(base_q_idx: u8) -> i32 {
    let s = crate::tuning::get().uv422_mid_scale;
    (s * uv420_mid_delta(base_q_idx) as f32) as i32
}

pub(crate) fn chroma_ac_delta(base_q_idx: u8, sub: usize) -> i32 {
    match sub {
        2 => return uv420_mid_delta(base_q_idx),
        1 => return uv422_mid_delta(base_q_idx),
        _ => {}
    }
    let qi = base_q_idx as i32;
    if qi <= 20 {
        top_444_uac()
    } else if qi >= 96 {
        24
    } else {
        24 * (qi - 20) / 76
    }
}

/// Frame-level chroma-DC qindex delta (`DeltaQUDc`), same `sub` convention.
pub(crate) fn chroma_dc_delta(base_q_idx: u8, sub: usize) -> i32 {
    let q = base_q_idx as i32;
    let deep = if q <= 160 {
        0
    } else {
        -((q - 160) / 3).min(32)
    };
    if sub == 2 {
        deep + uv420_mid_delta(base_q_idx)
    } else if sub == 1 {
        deep + uv422_mid_delta(base_q_idx)
    } else {
        deep
    }
}

/// Inverse-transform clip bounds for `bit_depth`, matching dav1d's `itx_tmpl.c`:
/// returns `(row_min, row_max, col_min, col_max, cf_max)`. 8-bit uses `INT16`
/// for both row and col; for higher depth the row clip is `±2^(bd+7)`, the
/// column clip is `±2^(bd+5)`, and `cf_max == row_max == ~(~127 << bd)`.
pub(crate) fn itx_clips(bd: u8) -> (i32, i32, i32, i32, i32) {
    if bd <= 8 {
        let (mn, mx) = (i16::MIN as i32, i16::MAX as i32);
        (mn, mx, mn, mx, 32767)
    } else {
        let row_max = (1i32 << (bd + 7)) - 1;
        let col_max = (1i32 << (bd + 5)) - 1;
        (!row_max, row_max, !col_max, col_max, row_max)
    }
}

/// Source of the dequant coefficients and inverse-transform clip bounds the
/// DCT/IDCT drivers need. Implemented by [`Quant`], which computes them once
/// per (base_q_idx, bit_depth) so the transforms read them from `self` instead
/// of indexing `dav1d_dq_tbl` and recomputing the clips on every block.
pub(crate) trait Dct {
    /// DC dequant step (`dav1d_dq_tbl[bd][q][0]`). Used by the inverse transform
    /// and as the trellis distortion weight.
    fn dc_q(&self) -> i32;
    /// AC dequant step (`dav1d_dq_tbl[bd][q][1]`).
    fn ac_q(&self) -> i32;
    /// Inverse-transform clips `(row_min, row_max, col_min, col_max, cf_max)`.
    fn clips(&self) -> (i32, i32, i32, i32, i32);
    /// Forward-quantisation multiplier for DC, `round(65536 / dc_q)`, so that the
    /// integer forward DCT's `mul_q16(coeff, q_mult_dc()) ≈ coeff / dc_q` (the
    /// inverse multiplies the level back by `dc_q`, so this round-trips).
    fn q_mult_dc(&self) -> i32;
    /// Forward-quantisation multiplier for AC, `round(65536 / ac_q)`.
    fn q_mult_ac(&self) -> i32;
    fn qm_level(&self) -> u8;
    fn qm_chroma(&self) -> bool;

    #[inline]
    fn has_qmatrix(&self) -> bool {
        self.qm_level() < QM_FLAT_LEVEL
    }

    #[inline]
    fn dequant_step(&self, rc: usize, w: usize, h: usize) -> i32 {
        let base = if rc == 0 { self.dc_q() } else { self.ac_q() };
        if !self.has_qmatrix() {
            return base;
        }
        let weight = qm_weight(self.qm_level(), self.qm_chroma(), rc, w, h) as i32;
        (base * weight + 16) >> 5
    }

    #[cfg(test)]
    fn forward_qm_weight(&self, rc: usize, w: usize, h: usize) -> i32 {
        if !self.has_qmatrix() {
            return 32;
        }
        let iwt = qm_weight(self.qm_level(), self.qm_chroma(), rc, w, h) as i32;
        (1024 + iwt / 2) / iwt
    }
}

pub(crate) struct FlatDct<'a, T: Dct>(pub &'a T);

impl<T: Dct> Dct for FlatDct<'_, T> {
    fn dc_q(&self) -> i32 {
        self.0.dc_q()
    }
    fn ac_q(&self) -> i32 {
        self.0.ac_q()
    }
    fn clips(&self) -> (i32, i32, i32, i32, i32) {
        self.0.clips()
    }
    fn q_mult_dc(&self) -> i32 {
        self.0.q_mult_dc()
    }
    fn q_mult_ac(&self) -> i32 {
        self.0.q_mult_ac()
    }
    fn qm_level(&self) -> u8 {
        QM_FLAT_LEVEL
    }
    fn qm_chroma(&self) -> bool {
        self.0.qm_chroma()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Quant {
    dc: i32,
    ac: i32,
    q_mult_dc: i32,
    q_mult_ac: i32,
    rmin: i32,
    rmax: i32,
    cmin: i32,
    cmax: i32,
    cf_max: i32,
    qm_level: u8,
    qm_chroma: bool,
    /// AC-side qindex this quantizer was built from (chroma includes its
    /// frame-level AC delta); drives the coefficient-dropout thresholds.
    qidx: u8,
}

impl Quant {
    #[allow(dead_code)]
    pub(crate) fn new(base_q_idx: u8, bd: u8) -> Self {
        Self::new_with_qm(base_q_idx, bd, QM_FLAT_LEVEL)
    }

    pub(crate) fn luma_dc_delta_probe(base_q_idx: u8) -> i32 {
        let _ = base_q_idx;
        -8
    }

    pub(crate) fn new_with_qm(base_q_idx: u8, bd: u8, qm_level: u8) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        let dc_idx =
            (base_q_idx as i32 + Self::luma_dc_delta_probe(base_q_idx)).clamp(0, 255) as u8;
        let dc = dc_q(dc_idx, bd) as i32;
        let ac = ac_q(base_q_idx, bd) as i32;
        Quant {
            dc,
            ac,
            q_mult_dc: (65536.0_f64 / dc as f64).round() as i32,
            q_mult_ac: (65536.0_f64 / ac as f64).round() as i32,
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
            qm_level: qm_level.min(QM_FLAT_LEVEL),
            qm_chroma: false,
            qidx: base_q_idx,
        }
    }

    pub(crate) fn qidx(&self) -> u8 {
        self.qidx
    }

    pub(crate) fn new_chroma(base_q_idx: u8, bd: u8) -> Self {
        Self::new_chroma_with_delta_qm(
            base_q_idx,
            chroma_dc_delta(base_q_idx, 0),
            0,
            bd,
            QM_FLAT_LEVEL,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn new_chroma_with_delta(base_q_idx: u8, dc_delta: i32, bd: u8) -> Self {
        Self::new_chroma_with_delta_qm(base_q_idx, dc_delta, 0, bd, QM_FLAT_LEVEL)
    }

    pub(crate) fn new_chroma_with_delta_qm(
        base_q_idx: u8,
        dc_delta: i32,
        ac_delta: i32,
        bd: u8,
        qm_level: u8,
    ) -> Self {
        let dc_idx = (base_q_idx as i32 + dc_delta).clamp(0, 255) as u8;
        let ac_idx = (base_q_idx as i32 + ac_delta).clamp(0, 255) as u8;
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        let dc = dc_q(dc_idx, bd) as i32;
        let ac = ac_q(ac_idx, bd) as i32;
        Quant {
            dc,
            ac,
            q_mult_dc: (65536.0_f64 / dc as f64).round() as i32,
            q_mult_ac: (65536.0_f64 / ac as f64).round() as i32,
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
            qm_level: qm_level.min(QM_FLAT_LEVEL),
            qm_chroma: true,
            qidx: ac_idx,
        }
    }
}

impl Dct for Quant {
    #[inline]
    fn dc_q(&self) -> i32 {
        self.dc
    }
    #[inline]
    fn ac_q(&self) -> i32 {
        self.ac
    }
    #[inline]
    fn clips(&self) -> (i32, i32, i32, i32, i32) {
        (self.rmin, self.rmax, self.cmin, self.cmax, self.cf_max)
    }
    #[inline]
    fn q_mult_dc(&self) -> i32 {
        self.q_mult_dc
    }
    #[inline]
    fn q_mult_ac(&self) -> i32 {
        self.q_mult_ac
    }
    #[inline]
    fn qm_level(&self) -> u8 {
        self.qm_level
    }
    #[inline]
    fn qm_chroma(&self) -> bool {
        self.qm_chroma
    }
}

#[cfg(test)]
mod qm_tests {
    use super::*;

    #[test]
    fn high_quality_qm_taper_preserves_shaping_then_flattens() {
        assert_eq!(qm_high_quality_t(17), 0.0);
        assert_eq!(qm_high_quality_t(12), 0.0);
        assert!((qm_high_quality_t(10) - 0.4).abs() < f32::EPSILON);
        assert_eq!(qm_high_quality_t(7), 1.0);
        assert_eq!(qm_high_quality_t(1), 1.0);
    }

    #[test]
    fn high_quality_qm_taper_is_format_specific() {
        // Neither subsampled format flattens at the top any more: measured
        // 2026-07-27, flattening cost top-band holdout -1.78% at 420 and
        // -3.20% at 422. Both hold level 8 down to qindex 0.
        for sub in [1usize, 2] {
            for qi in [20u8, 12, 7, 1] {
                assert_eq!(qm_level_law(qi, sub), 8, "sub={sub} qi={qi}");
            }
        }
        // Full-resolution luma is unaffected and stays flat at the top.
        assert_eq!(qm_level_law(7, 0), QM_FLAT_LEVEL);

        // Full-resolution chroma follows the same bounded transition.
        assert_eq!(qm_chroma_level_law(12, 0), 2);
        assert_eq!(qm_chroma_level_law(7, 0), QM_FLAT_LEVEL);
    }

    #[test]
    fn flat_level_preserves_scalar_quantizers() {
        let q = Quant::new_with_qm(128, 8, 15);
        for rc in 0..64 {
            assert_eq!(
                q.dequant_step(rc, 8, 8),
                if rc == 0 { q.dc_q() } else { q.ac_q() }
            );
            assert_eq!(q.forward_qm_weight(rc, 8, 8), 32);
        }
    }

    #[test]
    fn normative_level_zero_4x4_weights_match_av1() {
        let expected = [
            32, 43, 73, 97, 43, 67, 94, 110, 73, 94, 137, 150, 97, 110, 150, 200,
        ];
        for y in 0..4 {
            for x in 0..4 {
                let rc = x * 4 + y;
                assert_eq!(qm_weight(0, false, rc, 4, 4), expected[y * 4 + x]);
            }
        }
    }

    #[test]
    fn forward_and_inverse_matrix_weights_are_reciprocals() {
        let q = Quant::new_with_qm(128, 8, 10);
        for rc in 0..256 {
            let inverse = qm_weight(10, false, rc, 16, 16) as i32;
            let forward = q.forward_qm_weight(rc, 16, 16);
            assert!((forward * inverse - 1024).abs() <= inverse / 2);
        }
    }
}
