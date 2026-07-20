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

//! Entropy-accurate block rate for R-D decisions.
//!
//! [`crate::cost::block_rate_bits`] estimates a coefficient block's rate as a
//! context-free sum of a static per-level table. That proxy is blind to the
//! three terms that actually dominate AV1 coefficient rate:
//!
//! 1. **Neighbor context.** A coefficient's base token is coded against
//!    `base_tok[ctx]`, where `ctx` comes from the already-coded neighbours
//!    (`get_lo_ctx_2d`). An isolated high-frequency level costs far more than
//!    the same level sitting in a cluster; the proxy prices them identically.
//! 2. **EOB position.** The proxy charges a flat 2 bits no matter where the
//!    last nonzero lands. The real coder pays an `eob_pt` symbol plus up to
//!    `log2(n)` extra bypass bits, so a single stray coefficient far out in the
//!    scan is expensive. Pricing that at 2 bits makes the search keep tails it
//!    should drop, and mis-ranks transform types and partitions.
//! 3. **Live CDF state.** Probabilities adapt as the tile codes; the proxy is
//!    static.
//!
//! This module mirrors `encode_tx*_coeffs_adapt` symbol for symbol, read-only
//! (it never adapts a CDF), so the number it returns is the rate the coder will
//! actually spend if this candidate is committed.
//!
//! Square transforms only (4x4/8x8/16x16/32x32) — the rectangular coders use a
//! different context-offset table; callers for those keep the proxy.

use crate::coder::Cdfs;
use crate::coeffs::get_lo_ctx_2d;
use crate::cost::{cdf_cost, hi_tok_cost};
use crate::tables::{LO_CTX_OFF, level_byte};

/// Everything the real coefficient coder consults for one square transform
/// block, gathered so [`real_block_bits`] can mirror it symbol for symbol.
pub(crate) struct RateCtx<'a> {
    pub(crate) cdfs: &'a Cdfs,
    /// Coefficient class: 0 = TX_4X4, 1 = TX_8X8, 2 = TX_16X16, 3 = TX_32X32.
    pub(crate) cls: usize,
    /// 0 = luma, 1 = chroma.
    pub(crate) plane: usize,
    /// Transform width; the scan is `w*w` and `w` is the `levels` stride.
    pub(crate) w: usize,
    /// `eob_bin_*` CDF for this size/plane (the coder picks it by size).
    pub(crate) eob_bin: &'a [u16],
    /// `txb_skip` context (neighbour coded-ness) for the all-zero flag.
    pub(crate) skip_ctx: usize,
    /// `dc_sign` context for the DC sign symbol.
    pub(crate) dcs_ctx: usize,
    /// Luma only: the per-intra-mode txtp CDF and the chosen type. Chroma codes
    /// no transform-type symbol, so it passes `None`.
    pub(crate) txtp: Option<(&'a [u16], usize)>,
}

/// Scratch `levels` plane, reused across calls (this runs in every R-D
/// comparison, so the buffer must not be reallocated per candidate).
fn with_levels<R>(w: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    thread_local! {
        static LEVELS: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    LEVELS.with_borrow_mut(|buf| {
        // `get_lo_ctx_2d` reads up to two rows/columns past a position, so the
        // plane is padded exactly as the coder's own `levels` arrays are.
        let need = w * (w + 4);
        buf.clear();
        buf.resize(need, 0);
        f(buf)
    })
}

/// Rate, in bits, that the coefficient coder will spend on `cf` — the same
/// symbols `encode_tx*_coeffs_adapt` would emit, costed against the live CDFs.
///
/// Mirrors the coder's structure exactly: all-zero flag, transform type,
/// `eob_pt` (+ hi bin + bypass extra bits), the eob coefficient's own token,
/// the reverse-scan interior tokens with their `get_lo_ctx_2d` contexts, the DC
/// token, and the sign/Golomb tails. `hi_tok_cost` already folds in the
/// Exp-Golomb residual for levels >= 15, so the sign loops add only sign bits.
pub(crate) fn real_block_bits(cf: &[i32], scan: &[u32], c: &RateCtx) -> f32 {
    let (cls, pl, w) = (c.cls, c.plane, c.w);
    let skip_cdf = &c.cdfs.txb_skip[cls][c.skip_ctx];

    let Some(eob) = scan.iter().rposition(|&rc| cf[rc as usize] != 0) else {
        // All-zero: the block costs exactly one `all_zero = 1` symbol.
        return cdf_cost(skip_cdf, 1);
    };

    let base_tok = &c.cdfs.base_tok[cls][pl];
    let br_tok = &c.cdfs.br_tok[cls][pl];
    let eob_base = &c.cdfs.eob_base[cls][pl];
    let eob_hi = &c.cdfs.eob_hi[cls][pl];
    let dc_sign = &c.cdfs.dc_sign[pl][c.dcs_ctx];

    let mut bits = cdf_cost(skip_cdf, 0);
    if let Some((txtp_cdf, txtp)) = c.txtp {
        bits += cdf_cost(txtp_cdf, txtp);
    }

    // eob == 0 is the DC-only tail: eob_pt(0), the DC base token off `eob_base`,
    // its base-range ladder, and the DC sign.
    if eob == 0 {
        let m = cf[0].unsigned_abs();
        bits += cdf_cost(c.eob_bin, 0);
        bits += cdf_cost(&eob_base[0], m.min(3) as usize - 1);
        if m.min(3) == 3 {
            bits += hi_tok_cost(m, &br_tok[0]);
        }
        bits += cdf_cost(dc_sign, (cf[0] < 0) as usize);
        return bits;
    }

    // eob_pt: the bin, then the hi bit and the remaining bypass bits that pin
    // the exact position. This is the term the proxy charges a flat 2 bits for.
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    bits += cdf_cost(c.eob_bin, eob_bin);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        bits += cdf_cost(&eob_hi[eob_bin], (eob >> nbits) & 1);
        bits += nbits as f32; // equiprobable extra bits
    }

    let n = scan.len();
    let log2w = w.trailing_zeros() as usize;
    let stride = w;

    with_levels(w, |levels| {
        // The eob coefficient uses the eob_base CDF (not base_tok) and a br
        // context that depends only on its own position.
        let ctx_e = 1 + (eob > (n >> 3)) as usize + (eob > (n >> 2)) as usize;
        let rc = scan[eob] as usize;
        let (ex, ey) = (rc >> log2w, rc & (w - 1));
        let m = cf[rc].unsigned_abs();
        let eob_tok = m.min(3) - 1;
        bits += cdf_cost(&eob_base[ctx_e], eob_tok as usize);
        if eob_tok == 2 {
            let bc = if (ex | ey) > 1 { 14 } else { 7 };
            bits += hi_tok_cost(m, &br_tok[bc]);
        }
        levels[ex * stride + ey] = level_byte(m);

        // Interior coefficients, reverse scan — each token's context comes from
        // the neighbours already written into `levels`.
        for i in (1..eob).rev() {
            let rc_i = scan[i] as usize;
            let (x, y) = (rc_i >> log2w, rc_i & (w - 1));
            let (ctx, hi_mag) = get_lo_ctx_2d(levels, x, y, &LO_CTX_OFF, stride);
            let m = cf[rc_i].unsigned_abs();
            let tok = m.min(3);
            bits += cdf_cost(&base_tok[ctx], tok as usize);
            if tok == 3 {
                let mag = hi_mag & 63;
                let bc =
                    (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
                bits += hi_tok_cost(m, &br_tok[bc as usize]);
            }
            levels[x * stride + y] = level_byte(m);
        }

        // DC token (context 0) and its base-range ladder.
        let dm = cf[0].unsigned_abs();
        let dc_tok = dm.min(3);
        bits += cdf_cost(&base_tok[0], dc_tok as usize);
        if dc_tok == 3 {
            let mag =
                (levels[1] as u32 + levels[stride] as u32 + levels[stride + 1] as u32) & 63;
            let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
            bits += hi_tok_cost(dm, &br_tok[bc as usize]);
        }
        if cf[0] != 0 {
            bits += cdf_cost(dc_sign, (cf[0] < 0) as usize);
        }

        // One bypass sign bit per nonzero AC (the Golomb tail is already folded
        // into `hi_tok_cost` above).
        for i in 1..=eob {
            if cf[scan[i] as usize] != 0 {
                bits += 1.0;
            }
        }
        bits
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::SCAN_8X8;

    fn ctx<'a>(cdfs: &'a Cdfs) -> RateCtx<'a> {
        RateCtx {
            cdfs,
            cls: 1,
            plane: 0,
            w: 8,
            eob_bin: &cdfs.eob_bin_64_l,
            skip_ctx: 0,
            dcs_ctx: 0,
            txtp: None,
        }
    }

    /// An all-zero block costs exactly the `all_zero = 1` flag — not the proxy's
    /// flat 1.0.
    #[test]
    fn all_zero_is_the_skip_flag() {
        let cdfs = Cdfs::new(0);
        let c = ctx(&cdfs);
        let got = real_block_bits(&[0i32; 64], &SCAN_8X8, &c);
        assert_eq!(got, cdf_cost(&cdfs.txb_skip[1][0], 1));
    }

    /// The whole point of the model: a lone coefficient far out in the scan must
    /// cost materially more than the same coefficient at DC, because eob_pt has
    /// to encode the position. The proxy prices these within 0.1 bits.
    #[test]
    fn far_eob_costs_more_than_dc() {
        let cdfs = Cdfs::new(0);
        let c = ctx(&cdfs);
        let mut dc_only = [0i32; 64];
        dc_only[0] = 3;
        let mut far = [0i32; 64];
        far[SCAN_8X8[40] as usize] = 3;

        let dc_bits = real_block_bits(&dc_only, &SCAN_8X8, &c);
        let far_bits = real_block_bits(&far, &SCAN_8X8, &c);
        assert!(
            far_bits > dc_bits + 4.0,
            "far eob {far_bits} should dwarf DC-only {dc_bits}"
        );
    }

    /// Clustered low-frequency energy is cheaper than the same number of levels
    /// scattered across the block — the neighbor-context effect the proxy is
    /// blind to.
    #[test]
    fn clustered_beats_scattered() {
        let cdfs = Cdfs::new(0);
        let c = ctx(&cdfs);
        let mut clustered = [0i32; 64];
        for &i in &[0usize, 1, 2, 3] {
            clustered[SCAN_8X8[i] as usize] = 2;
        }
        let mut scattered = [0i32; 64];
        for &i in &[0usize, 12, 24, 36] {
            scattered[SCAN_8X8[i] as usize] = 2;
        }
        assert!(
            real_block_bits(&clustered, &SCAN_8X8, &c)
                < real_block_bits(&scattered, &SCAN_8X8, &c)
        );
    }
}
