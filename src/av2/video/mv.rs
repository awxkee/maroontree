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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Mv {
    pub row: i32,
    pub col: i32,
}

impl Mv {
    pub const ZERO: Mv = Mv { row: 0, col: 0 };

    pub fn diff(self, r: Mv) -> Mv {
        Mv {
            row: self.row - r.row,
            col: self.col - r.col,
        }
    }
}

/// AVM `div_mult` reciprocal table for MV projection (mvref_common.h).
static DIV_MULT: [i64; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

/// AVM MAX_FRAME_DISTANCE ((1 << FRAME_OFFSET_BITS) - 1).
const MAX_FRAME_DISTANCE: i32 = 31;

/// AVM `get_mv_projection`: scale a neighbor MV taken against a reference at
/// distance `den` onto this block's reference at distance `num`. Feeds the
/// cross-reference derived DRL candidates on two-reference frames. Component
/// clamp is MV_LOW+1..=MV_UPP-1 (MV_IN_USE_BITS = 16).
pub(crate) fn mv_projection(mv: Mv, num: i32, den: i32) -> Mv {
    let den = den.clamp(1, MAX_FRAME_DISTANCE);
    let num = num.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    let round_signed = |value: i64| -> i32 {
        let rounded = if value < 0 {
            -(((-value) + (1 << 13)) >> 14)
        } else {
            (value + (1 << 13)) >> 14
        };
        rounded.clamp(-(1 << 16) + 1, (1 << 16) - 1) as i32
    };
    Mv {
        row: round_signed(mv.row as i64 * num as i64 * DIV_MULT[den as usize]),
        col: round_signed(mv.col as i64 * num as i64 * DIV_MULT[den as usize]),
    }
}

/// MV coding cost proxy: bits(mv - ref) scaled by lambda_mv.
/// AVM ref: `av2/encoder/mcomp.c` mv_cost = `lambda * (mvjoint + mvcomp bits)`.
/// Exact component-CDF cost is added once inter CDFs land; this Exp-Golomb
/// approximation is the search-time proxy, calibrated against SSIMULACRA2.
pub fn mv_bits(d: Mv) -> u32 {
    fn comp_bits(v: i32) -> u32 {
        // Exp-Golomb-like: 2*floor(log2(|v|+1)) + 1, plus a sign bit when nonzero.
        let a = v.unsigned_abs();
        let mag = 32 - (a + 1).leading_zeros();
        2 * mag + if v != 0 { 1 } else { 0 }
    }
    // MV-joint (~2 bits) + per-component magnitude.
    2 + comp_bits(d.row) + comp_bits(d.col)
}

/// Search-time MV cost in the same units as SAD-derived distortion.
#[inline]
pub fn mv_cost(d: Mv, lambda_mv: u32) -> u32 {
    mv_bits(d).saturating_mul(lambda_mv)
}
