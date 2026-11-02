/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! AV2 quantization matrices. The tables are AV1's (identical in AVM), held in
//! the decoder's transposed coefficient order: the weight of the coefficient at
//! (row, col) of a `cw` x `ch` coefficient grid (sides capped at 32) is
//! `table[col * ch + row]`, and the decoder dequantizes it with
//! `(dq * weight + 16) >> 5` (DC included; 2-D non-identity transforms only).
//!
//! The levels and the plane being coded are per-thread state: every SB decision
//! sets the frame's levels (luma plane), and the chroma coders (the SDP chroma
//! tree, the 4:4:4 probe's chroma) switch to the chroma plane. QM is only
//! enabled together with SDP, so chroma is never coded inside the luma pass.

use std::cell::Cell;
use std::sync::OnceLock;

/// Level 15 is the flat matrix (QM off for that plane).
pub(crate) const QM_FLAT: u8 = 15;

thread_local! {
    /// (luma level, chroma level, coding chroma).
    static CUR: Cell<(u8, u8, bool)> = const { Cell::new((QM_FLAT, QM_FLAT, false)) };
}

/// Scope of an SB decision: sets the frame's levels (luma plane) and restores the
/// previous state on drop, so nested decisions (SB trials) and other encoders on
/// the same thread are unaffected.
pub(crate) struct QmScope(u8, u8, bool);

impl QmScope {
    pub(crate) fn new(luma: u8, chroma: u8) -> Self {
        let prev = CUR.with(|c| c.replace((luma, chroma, false)));
        QmScope(prev.0, prev.1, prev.2)
    }
}

impl Drop for QmScope {
    fn drop(&mut self) {
        CUR.with(|c| c.set((self.0, self.1, self.2)));
    }
}

/// Select the chroma (`true`) or luma plane; returns the previous selection.
pub(crate) fn set_chroma(chroma: bool) -> bool {
    CUR.with(|c| {
        let (l, ch, prev) = c.get();
        c.set((l, ch, chroma));
        prev
    })
}

/// 4x32 / 32x4 tables (AV2 sizes AV1 lacks), derived from the 32x16 base table
/// as the decoder does: every row of the 32-wide table from offset 32, every 4th row.
fn derived_4x32() -> &'static [[[Vec<u8>; 2]; 2]; 15] {
    static T: OnceLock<[[[Vec<u8>; 2]; 2]; 15]> = OnceLock::new();
    T.get_or_init(|| {
        std::array::from_fn(|lvl| {
            std::array::from_fn(|pl| {
                // AV1's 16x32 entry is the raw (untransposed) 32x16 base table.
                let t32x16 = crate::quant::qm_row(lvl as u8, pl == 1, 16, 32).expect("qm level");
                let mut q32x4 = Vec::with_capacity(128);
                for y in (0..16).step_by(4) {
                    q32x4.extend_from_slice(&t32x16[32 + y * 32..32 + y * 32 + 32]);
                }
                let mut q4x32 = vec![0u8; 128];
                for y in 0..4 {
                    for x in 0..32 {
                        q4x32[x * 4 + y] = q32x4[y * 32 + x];
                    }
                }
                // [0] = TX 4x32 (stored as the 32x4 rows), [1] = TX 32x4.
                [q32x4, q4x32]
            })
        })
    })
}

/// The weight table for a `cw` x `ch` coefficient grid on the plane being coded,
/// or `None` when that plane's matrix is flat.
#[inline]
pub(crate) fn weights(cw: usize, ch: usize) -> Option<&'static [u8]> {
    let (l, c, chroma) = CUR.with(|s| s.get());
    let level = if chroma { c } else { l };
    if level >= QM_FLAT {
        return None;
    }
    match (cw, ch) {
        (4, 32) => Some(&derived_4x32()[level as usize][chroma as usize][0]),
        (32, 4) => Some(&derived_4x32()[level as usize][chroma as usize][1]),
        _ => crate::quant::qm_row(level, chroma, cw, ch),
    }
}

/// Weighted dequant step of the coefficient at grid index `idx` (`col * ch + row`).
#[inline]
pub(crate) fn step(tbl: Option<&[u8]>, qstep: i32, idx: usize) -> i64 {
    match tbl {
        Some(t) => ((qstep * t[idx] as i32 + 16) >> 5) as i64,
        None => qstep as i64,
    }
}

/// Code chroma while alive: selects the chroma plane, restores on drop.
pub(crate) struct ChromaScope(bool);

impl ChromaScope {
    pub(crate) fn new() -> Self {
        ChromaScope(set_chroma(true))
    }
}

impl Drop for ChromaScope {
    fn drop(&mut self) {
        set_chroma(self.0);
    }
}
