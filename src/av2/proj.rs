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

use crate::av2::quant::{BASE_Q, qstep};
use crate::util::FastRound;
use std::cell::RefCell;

pub(crate) struct Basis {
    /// quantizer rescale factor (qstep_target / qstep_measured), applied at run time.
    pub(crate) scale: f32,
    /// Max reconstructed sample value = (1 << bit_depth) - 1. Defaults to 8-bit (255).
    pub(crate) max_val: f32,
    /// Dequantizer step for the forward-DCT path (set in [`Bases::rescaled_to_q`]).
    pub(crate) qstep: i32,
    /// Which avm forward transform this basis uses (DCT size, or 16×16 ADST mix).
    pub(crate) fwd: crate::av2::fdct::FwdKind,
}

/// RD end-of-block (EOB) truncation threshold, in unrounded-coefficient-magnitude units.
pub(crate) const RDOQ_EOB_T: f32 = 0.9;
pub(crate) const DEFAULT_RDOQ_LAMBDA: f32 = 0.07;

pub(crate) fn hq_rdoq_lambda(base_q_idx: u32) -> f32 {
    const BASE: f32 = DEFAULT_RDOQ_LAMBDA;
    const PEAK: f32 = 0.13;
    if base_q_idx >= 50 || base_q_idx == 0 {
        return BASE;
    }
    let t = (50 - base_q_idx) as f32 / 46.0; // ramp toward the top
    let mut boost = PEAK * t.min(1.0);
    if base_q_idx <= 4 {
        // ease off near-lossless so the absolute top keeps full precision
        boost *= base_q_idx as f32 / 5.0;
    }
    BASE + boost
}

/// Zero trailing coefficients (scan order) whose unrounded magnitude `prm[k]` is below the
/// EOB threshold `t`, stopping at the first kept coefficient. `lev`/`prm` are in scan order.
fn rdoq_truncate_eob(lev: &mut [f32], prm: &[f32], t: f32) {
    if t <= 0.0 {
        return;
    }
    let mut k = lev.len().min(prm.len());
    while k > 0 {
        k -= 1;
        if lev[k] == 0.0 {
            continue; // already zero — keep walking back toward the last kept coeff
        }
        if prm[k] < t {
            lev[k] = 0.0; // trailing & weak → drop, shrinking EOB
        } else {
            break; // first coefficient worth keeping: stop
        }
    }
}

pub(crate) struct Bases {
    pub(crate) luma: Basis,
    pub(crate) chroma420: Basis,
    /// 4:2:2 chroma: a 32-wide × 64-tall (TX_32X64) transform. avmdec codes its
    pub(crate) chroma422: Basis,
    pub(crate) chroma444: Basis,
    /// 4:4:4 chroma for a bottom-edge 64×32 leaf: a 64-wide × 32-tall (TX_64X32)
    pub(crate) chroma444_64x32: Basis,
    /// 16-tap-family luma bases (residue-4 leaves). `luma16x64` = 16 wide × 64 tall
    /// (TX_16X64, SCAN16X32 coeff grid); `luma64x16` = 64 wide × 16 tall (TX_64X16,
    /// SCAN32X16); `luma16x16` = 16×16 (TX_16X16, SCAN16). The 16-tap 1D profile is
    /// derived analytically (DCT-II) from the luma 32-tap DC gain — adequate for
    /// bitstream validity; tune empirically against avmdec to remove drift.
    pub(crate) luma16x64: Basis,
    pub(crate) luma64x16: Basis,
    pub(crate) luma16x16: Basis,
    /// ADST_ADST (DST-VII both axes) 16×16 luma basis — the mode-dependent transform
    /// alternative to `luma16x16` (DCT_DCT) for native TX_16X16 intra leaves.
    pub(crate) luma16x16_adst: Basis,
    /// Mixed 1D transforms for native TX_16X16 intra leaves (DC mode, EXT_NEW_TX_SET):
    /// `luma16x16_adst_dct` = ADST_DCT (idx 2: ADST vertical / DST-VII column, DCT
    /// horizontal); `luma16x16_dct_adst` = DCT_ADST (idx 3: DCT vertical, ADST
    /// horizontal). Per avm, the row (horizontal) transform is g_hor_tx_type and the
    /// column (vertical) is g_ver_tx_type; from_1d_rect_scan's `h_vert` is the vertical
    /// (column-index) profile and `h_horiz` the horizontal (row-index) profile.
    pub(crate) luma16x16_adst_dct: Basis,
    pub(crate) luma16x16_dct_adst: Basis,
    /// 8-tap-family luma bases (residue-2 leaves). The right/bottom edges partition to
    /// 8×32 / 32×8 (BLOCK_8X64 would be 1:8 aspect, which is disallowed). `luma8x32` =
    /// 8 wide × 32 tall (TX_8X32, SCAN8X32); `luma32x8` = 32 wide × 8 tall (TX_32X8,
    /// SCAN32X8). TX_8X32/32X8 are entropy class 2 (reuse the LUMA16 token cdfs).
    pub(crate) luma8x32: Basis,
    pub(crate) luma32x8: Basis,
    /// 16-tap-family luma corner bases (residue-4 × residue-{6,8}). `luma16x32` =
    /// 16 wide × 32 tall (TX_16X32, coeff 16×32, SCAN16X32); `luma32x16` = 32 wide ×
    /// 16 tall (TX_32X16, coeff 32×16, SCAN32X16). Both are EXT_TX long-side-32, so
    /// the leaf coder emits a txtp_long32_dct flag (=1, DCT) + short-side idx 0.
    pub(crate) luma16x32: Basis,
    pub(crate) luma32x16: Basis,
    /// 4:2:2 chroma basis for a 32×32 luma leaf: chroma is 16 wide × 32 tall (TX_16X32,
    /// coeff region 16×32, SCAN16X32).
    pub(crate) c16x32: Basis,
    /// 4:2:2 chroma for 32×64 luma → 16×64 already covered by luma16x64. These three
    /// cover the 4:2:2 16-family: 16×64 luma→8×64 (TX_8X64), 64×16 luma→32×16
    /// (TX_32X16), 16×16 luma→8×16 (TX_8X16) chroma.
    pub(crate) c8x64: Basis,
    pub(crate) c32x16: Basis,
    pub(crate) c8x16: Basis,
    /// 8×8 chroma basis (TX_8X8) for the 4:2:0 16×16 luma corner leaf.
    pub(crate) c8x8: Basis,
    /// 4:2:2 8-family chroma: 8×32 luma→4×32 (TX_4X32) and 32×8 luma→16×8 (TX_16X8).
    pub(crate) c4x32: Basis,
    /// 4:2:0 chroma for 8×32 / 32×8 luma corners: 4×16 (TX_4X16) and 16×4 (TX_16X4).
    pub(crate) c4x16: Basis,
    /// 4:2:2 chroma for the 8×8 luma corner: 4×8 (TX_4X8).
    pub(crate) c4x8: Basis,
    /// 4:2:0 chroma for the 16×8 luma corner: 8×4 (TX_8X4).
    pub(crate) c8x4: Basis,
    pub(crate) c16x4: Basis,
    /// 4:2:0 chroma for the 8×8 luma corner: 4×4 (TX_4X4).
    pub(crate) c4x4: Basis,
    pub(crate) c16x8: Basis,
}

/// Thread-local forward-DCT input/output scratch: the i32 residual fed to the
/// transform and the coded coefficients fed to the quantizer. Reused across blocks.
struct FwdScratch {
    resid: Box<[i32; 4096]>,
    out: Box<[i32; 1024]>,
}
thread_local! {
    static FWD_SCRATCH: RefCell<FwdScratch> =
        RefCell::new(FwdScratch { resid: Box::new([0; 4096]), out: Box::new([0; 1024])});
}

impl Basis {
    /// Rescale to a different quantizer. The bases are the decoder's level-1
    /// reconstruction `qstep · T(k)` with a q-independent transform `T`, so the target
    /// quantizer is just a multiply by `f = qstep_target/qstep_measured`. We record it
    /// as a factor (applied in project/reconstruct) rather than touching the profiles.
    pub(crate) fn scale(&mut self, f: f32) {
        self.scale *= f;
    }
}

impl Bases {
    /// Rescale bases measured at `quant::BASE_Q` to an arbitrary 8-bit base_q_idx.
    /// Set the reconstruction clamp ceiling from the signaled bit depth (8/10/12).
    pub(crate) fn set_bit_depth(&mut self, bit_depth: u8) {
        let mv = ((1u32 << bit_depth) - 1) as f32;
        self.luma.max_val = mv;
        self.chroma420.max_val = mv;
        self.chroma422.max_val = mv;
        self.chroma444.max_val = mv;
        self.chroma444_64x32.max_val = mv;
        self.luma16x64.max_val = mv;
        self.luma64x16.max_val = mv;
        self.luma16x16.max_val = mv;
        self.luma16x16_adst.max_val = mv;
        self.luma16x16_adst_dct.max_val = mv;
        self.luma16x16_dct_adst.max_val = mv;
        self.luma8x32.max_val = mv;
        self.luma32x8.max_val = mv;
        self.c16x32.max_val = mv;
        self.c8x64.max_val = mv;
        self.c32x16.max_val = mv;
        self.c8x16.max_val = mv;
        self.c8x8.max_val = mv;
        self.c4x32.max_val = mv;
        self.c4x16.max_val = mv;
        self.c16x4.max_val = mv;
        self.c16x8.max_val = mv;
    }
    pub(crate) fn rescaled_to_q(mut self, base_q_idx: u32) -> Bases {
        let f = qstep(base_q_idx) as f32 / qstep(BASE_Q) as f32;
        if f != 1.0 {
            self.luma.scale(f);
            self.chroma420.scale(f);
            self.chroma422.scale(f);
            self.chroma444.scale(f);
            self.chroma444_64x32.scale(f);
            self.luma16x64.scale(f);
            self.luma64x16.scale(f);
            self.luma16x16.scale(f);
            self.luma16x16_adst.scale(f);
            self.luma16x16_adst_dct.scale(f);
            self.luma16x16_dct_adst.scale(f);
            self.luma8x32.scale(f);
            self.luma32x8.scale(f);
            self.luma16x32.scale(f);
            self.luma32x16.scale(f);
            self.c16x32.scale(f);
            self.c8x64.scale(f);
            self.c32x16.scale(f);
            self.c8x16.scale(f);
            self.c8x8.scale(f);
            self.c4x32.scale(f);
            self.c4x16.scale(f);
            self.c16x4.scale(f);
            self.c16x8.scale(f);
        }
        let qs = qstep(base_q_idx) as i32;
        self.luma.qstep = qs;
        self.chroma420.qstep = qs;
        self.chroma444.qstep = qs;
        self.luma16x16.qstep = qs;
        self.chroma422.qstep = qs;
        self.chroma444_64x32.qstep = qs;
        self.luma16x16_adst.qstep = qs;
        self.luma16x16_adst_dct.qstep = qs;
        self.luma16x16_dct_adst.qstep = qs;
        self.luma8x32.qstep = qs;
        self.luma32x8.qstep = qs;
        self.luma16x32.qstep = qs;
        self.luma32x16.qstep = qs;
        self.luma16x64.qstep = qs;
        self.luma64x16.qstep = qs;
        self.c16x32.qstep = qs;
        self.c8x64.qstep = qs;
        self.c32x16.qstep = qs;
        self.c8x16.qstep = qs;
        self.c8x8.qstep = qs;
        self.c4x32.qstep = qs;
        self.c4x16.qstep = qs;
        self.c16x4.qstep = qs;
        self.c16x8.qstep = qs;
        self
    }
}

impl Basis {
    /// Square separable basis. Equivalent to the rectangular builder with the same
    /// profile on both axes (`m[k][py,px] = h[c,py]·h[a,px]/dc`, c=SCAN[k]&31,
    /// a=SCAN[k]>>5).
    /// Minimal basis: only the fields the avm forward path needs. `.fwd` is set by
    /// the caller; `qstep` is overwritten by [`Bases::rescaled_to_q`].
    fn new() -> Basis {
        Basis {
            scale: 1.0,
            max_val: 255.0,
            qstep: qstep(BASE_Q) as i32,
            fwd: crate::av2::fdct::FwdKind::Proj,
        }
    }

    /// avm forward transform for this basis; returns scan-order `(levels, |unquantised|)`.
    /// Only the unset `Proj` sentinel returns `None` (no basis is left in that state).
    fn fwd_fast(&self, resid: &[f32], scan: &[u16], thresh: f32) -> Option<(Vec<f32>, Vec<f32>)> {
        use crate::av2::fdct::{FwdKind, dc_tx_scale, fdct_rect, fwd16x16, quantize_to_levels};
        let q = self.qstep;
        Some(match self.fwd {
            FwdKind::Proj => return None,
            // Every DCT_DCT size routes through the one general avm butterfly driver.
            FwdKind::Dct(w, h) => {
                let n = w * h;
                let cw = w.min(32);
                let factor = (1u32 << (3 + dc_tx_scale(w, h))) as f32;
                // Thread-local residual + coded-coeff scratch (reused, never re-zeroed).
                FWD_SCRATCH.with(|cell| {
                    let s = &mut *cell.borrow_mut();
                    for (d, &v) in s.resid[..n].iter_mut().zip(resid) {
                        *d = v.fast_round() as i32;
                    }
                    let nc = fdct_rect(&s.resid[..n], w, h, s.out.as_mut_slice());
                    quantize_to_levels(&s.out[..nc], cw, q, factor, thresh, scan)
                })
            }
            // 16×16 ADST/DCT mixes: fwd16x16(col_adst, row_adst), factor 8.
            FwdKind::AdstAdst16 | FwdKind::AdstDct16 | FwdKind::DctAdst16 => {
                let (col_adst, row_adst) = match self.fwd {
                    FwdKind::AdstAdst16 => (true, true),
                    FwdKind::AdstDct16 => (true, false), // ADST_DCT: col ADST, row DCT
                    _ => (false, true),                  // DCT_ADST: col DCT, row ADST
                };
                let mut r = [0i32; 256];
                for (d, &s) in r.iter_mut().zip(resid) {
                    *d = s.fast_round() as i32;
                }
                quantize_to_levels(&fwd16x16(&r, col_adst, row_adst), 16, q, 8.0, thresh, scan)
            }
        })
    }

    /// Forward-transform a residual block (row-major) → integer coefficient levels;
    /// |coef| < thresh is dropped.
    ///
    /// After quantization an RD end-of-block (EOB) truncation is applied: see
    /// [`rdoq_truncate_eob`].
    pub(crate) fn project(&self, resid: &[f32], thresh: f32) -> Vec<f32> {
        let (mut lev, prm) = self
            .fwd_fast(resid, &crate::av2::tables::SCAN, thresh)
            .expect("every basis carries a forward DCT");
        rdoq_truncate_eob(&mut lev, &prm, RDOQ_EOB_T);
        lev
    }

    /// Like [`project`] but returns the round-to-nearest levels together with the
    /// per-coefficient unquantised magnitudes `|pr|` (scan order), and applies no
    /// EOB truncation — trellis RDOQ ([`crate::av2::coder::rdoq_luma`]) consumes
    /// these and does its own level + EOB optimisation.
    pub(crate) fn project_with_prm(&self, resid: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.fwd_fast(resid, &crate::av2::tables::SCAN, 0.0)
            .expect("every basis carries a forward DCT")
    }

    /// Scan-parameterised forward transform (see `project`).
    pub(crate) fn project_scan(&self, resid: &[f32], thresh: f32, scan: &[u16]) -> Vec<f32> {
        let (mut lev, prm) = self
            .fwd_fast(resid, scan, thresh)
            .expect("every basis carries a forward DCT");
        rdoq_truncate_eob(&mut lev, &prm, RDOQ_EOB_T);
        lev
    }

    /// Scan-parameterized [`project_with_prm`]: round levels + `|pr|`, no EOB
    /// truncation, for the partition leaf paths (all TX_32X32-class).
    pub(crate) fn project_scan_with_prm(
        &self,
        resid: &[f32],
        scan: &[u16],
    ) -> (Vec<f32>, Vec<f32>) {
        self.fwd_fast(resid, scan, 0.0)
            .expect("every basis carries a forward DCT")
    }
}

/// Build the full transform-basis table. Every basis runs the avm forward DCT/ADST;
/// `.fwd` selects which, `qstep` is set later by [`Bases::rescaled_to_q`].
pub(crate) fn default_bases() -> Bases {
    use crate::av2::fdct::FwdKind::*;
    let mk = |fwd| {
        let mut b = Basis::new();
        b.fwd = fwd;
        b
    };
    Bases {
        luma: mk(Dct(32, 32)),
        chroma420: mk(Dct(32, 32)),
        chroma422: mk(Dct(32, 64)),
        chroma444: mk(Dct(64, 64)),
        chroma444_64x32: mk(Dct(64, 32)),
        luma16x64: mk(Dct(16, 64)),
        luma64x16: mk(Dct(64, 16)),
        luma16x16: mk(Dct(16, 16)),
        luma16x16_adst: mk(AdstAdst16),
        luma16x16_adst_dct: mk(AdstDct16),
        luma16x16_dct_adst: mk(DctAdst16),
        luma8x32: mk(Dct(8, 32)),
        luma32x8: mk(Dct(32, 8)),
        luma16x32: mk(Dct(16, 32)),
        luma32x16: mk(Dct(32, 16)),
        c16x32: mk(Dct(16, 32)),
        c8x64: mk(Dct(8, 64)),
        c32x16: mk(Dct(32, 16)),
        c8x16: mk(Dct(8, 16)),
        c8x8: mk(Dct(8, 8)),
        c4x32: mk(Dct(4, 32)),
        c4x16: mk(Dct(4, 16)),
        c4x8: mk(Dct(4, 8)),
        c8x4: mk(Dct(8, 4)),
        c16x4: mk(Dct(16, 4)),
        c4x4: mk(Dct(4, 4)),
        c16x8: mk(Dct(16, 8)),
    }
}
