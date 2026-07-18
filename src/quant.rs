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

use crate::dct::{
    dct4x4_t, dct4x8_t, dct8x8_t, dct8x16_t, dct16x16_t, dct16x32_t, dct32x32, dct32x32_t,
};
use crate::qm_tables::AV1_IQM;

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
        _ => panic!("unsupported AV1 QM transform size {w}x{h}"),
    }
}

#[inline]
pub(crate) fn qm_weight(level: u8, chroma: bool, rc: usize, w: usize, h: usize) -> u8 {
    if level >= QM_FLAT_LEVEL {
        return 32;
    }
    // Coefficients are stored x-major (`rc = x*h + y`), while the normative
    // matrix is row-major (`y*w + x`).
    let x = rc / h;
    let y = rc % h;
    AV1_IQM[level as usize][chroma as usize][qm_offset(w, h) + y * w + x]
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

pub(crate) fn chroma_dc_delta(base_q_idx: u8) -> i32 {
    let q = base_q_idx as i32;
    if q <= 160 {
        0
    } else {
        -((q - 160) / 3).min(32)
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

    #[inline]
    fn forward_qm_weight(&self, rc: usize, w: usize, h: usize) -> i32 {
        if !self.has_qmatrix() {
            return 32;
        }
        let iwt = qm_weight(self.qm_level(), self.qm_chroma(), rc, w, h) as i32;
        (1024 + iwt / 2) / iwt
    }
}

/// Precomputed dequant coefficients + inverse-transform clips for one
/// (base_q_idx, bit_depth). Cheap to copy; build once and hand to the transforms.
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
}

impl Quant {
    #[allow(dead_code)]
    pub(crate) fn new(base_q_idx: u8, bd: u8) -> Self {
        Self::new_with_qm(base_q_idx, bd, QM_FLAT_LEVEL)
    }

    pub(crate) fn new_with_qm(base_q_idx: u8, bd: u8, qm_level: u8) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        let dc = dc_q(base_q_idx, bd) as i32;
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
        }
    }

    pub(crate) fn new_chroma(base_q_idx: u8, bd: u8) -> Self {
        Self::new_chroma_with_delta_qm(base_q_idx, chroma_dc_delta(base_q_idx), bd, QM_FLAT_LEVEL)
    }

    /// As [`Self::new_chroma`] but with an explicit chroma-DC qindex delta. Under
    /// adaptive quantization the per-superblock qindex (`CurrentQIndex`) changes,
    /// but the chroma-DC delta is a frame-level constant signaled once in the
    /// header (`DeltaQUDc`, derived from the frame `base_q_idx`). The decoder
    /// forms the chroma-DC qindex as `CurrentQIndex + DeltaQUDc`, so the encoder
    /// must apply that same frame-level delta to the AQ-adjusted qindex rather
    /// than recomputing it from the local qindex.
    #[allow(dead_code)]
    pub(crate) fn new_chroma_with_delta(base_q_idx: u8, dc_delta: i32, bd: u8) -> Self {
        Self::new_chroma_with_delta_qm(base_q_idx, dc_delta, bd, QM_FLAT_LEVEL)
    }

    pub(crate) fn new_chroma_with_delta_qm(
        base_q_idx: u8,
        dc_delta: i32,
        bd: u8,
        qm_level: u8,
    ) -> Self {
        let dc_idx = (base_q_idx as i32 + dc_delta).clamp(0, 255) as u8;
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
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
            qm_chroma: true,
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

pub(crate) fn forward_dct_quant_8x8_t(
    residual: &[i32; 64],
    q: &impl Dct,
) -> ([i32; 64], [f32; 64]) {
    dct8x8_t(residual, q)
}

/// As [`forward_dct_quant_16x16`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_16x16_t(
    residual: &[i32; 256],
    q: &impl Dct,
) -> ([i32; 256], [f32; 256]) {
    dct16x16_t(residual, q)
}

#[allow(unused)]
pub(crate) fn forward_dct_quant_32x32(residual: &mut [i32; 1024], q: &impl Dct) {
    dct32x32(residual, q)
}

pub(crate) fn forward_dct_quant_32x32_t(
    residual: &[i32; 1024],
    q: &impl Dct,
) -> ([i32; 1024], [f32; 1024]) {
    dct32x32_t(residual, q)
}

/// As [`forward_dct_quant_4x8`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_4x8_t(
    residual: &[i32; 32],
    q: &impl Dct,
) -> ([i32; 32], [f32; 32]) {
    dct4x8_t(residual, q)
}

/// As [`forward_dct_quant_8x16`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_8x16_t(
    residual: &[i32; 128],
    q: &impl Dct,
) -> ([i32; 128], [f32; 128]) {
    dct8x16_t(residual, q)
}

pub(crate) fn forward_dct_quant_16x32_t(
    residual: &[i32; 512],
    q: &impl Dct,
) -> ([i32; 512], [f32; 512]) {
    dct16x32_t(residual, q)
}

pub(crate) fn forward_dct_quant_4x4_t(
    residual: &[i32; 16],
    q: &impl Dct,
) -> ([i32; 16], [f32; 16]) {
    dct4x4_t(residual, q)
}

#[cfg(test)]
mod qm_tests {
    use super::*;

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
