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

//! AV2 (avm) quantizer-step lookup for 8-bit.
//!
//! avm replaced AV1's separate dc/ac 256-entry tables with one compact 25-entry
//! base table that doubles every 24 q-indices (`av2/common/quant_common.c`,
//! `ac_qlookup_QTX` + `qlookup`). With the frame's dc/ac delta-q at 0, the DC and
//! AC steps are identical, so a single `qstep` drives the whole frame.

static AC_QLOOKUP_QTX: [u32; 25] = [
    64, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78,
];

/// Dequant step for an 8-bit base_q_idx (delta_q = 0), matching avm `get_q`.
pub(crate) fn qstep(qindex: u32) -> u32 {
    if qindex == 0 {
        return AC_QLOOKUP_QTX[0];
    }
    let q = qindex.clamp(1, 255);
    if q < 25 {
        AC_QLOOKUP_QTX[q as usize]
    } else {
        AC_QLOOKUP_QTX[(((q - 1) % 24) + 1) as usize] << ((q - 1) / 24)
    }
}

/// base_q_idx the bundled bases were measured at, and its step (78 << 4 = 1248).
pub(crate) const BASE_Q: u32 = 120;
