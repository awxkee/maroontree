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

use crate::coder::Cdfs;
use crate::coeffs::get_lo_ctx_2d;
#[cfg(test)]
use crate::cost::cdf_cost;
use crate::cost::{
    br_cum_row_with_table, cdf_cost_with_table, cost_q_table, golomb_cost, hi_tok_cost_with_table,
};
use crate::tables::{
    COEFF_BASE_RANGE, LO_CTX_OFF, LO_CTX_OFF_WGH, LO_CTX_OFF_WLH, NUM_BASE_LEVELS, level_byte,
};

/// Everything the real coefficient coder consults for one transform
/// block, gathered so [`real_block_bits`] can mirror it symbol for symbol.
pub(crate) struct RateCtx<'a> {
    pub(crate) cdfs: &'a Cdfs,
    /// Coefficient class: 0 = TX_4X4, 1 = TX_8X8, 2 = TX_16X16, 3 = TX_32X32.
    pub(crate) cls: usize,
    /// 0 = luma, 1 = chroma.
    pub(crate) plane: usize,
    /// Transform dimensions. Coefficients are stored column-major by the AV1
    /// transform helpers (`cf[x * h + y]`), so `h` is the levels stride.
    pub(crate) w: usize,
    pub(crate) h: usize,
    /// `eob_bin_*` CDF for this size/plane (the coder picks it by size).
    pub(crate) eob_bin: &'a [u16],
    /// `txb_skip` context (neighbor coded-ness) for the all-zero flag.
    pub(crate) skip_ctx: usize,
    /// `dc_sign` context for the DC sign symbol.
    pub(crate) dcs_ctx: usize,
    /// Luma only: the per-intra-mode txtp CDF and the chosen type. Chroma codes
    /// no transform-type symbol, so it passes `None`.
    pub(crate) txtp: Option<(&'a [u16], usize)>,
}

/// Scratch `levels` plane, reused across calls (this runs in every R-D
/// comparison, so the buffer must not be reallocated per candidate).
fn with_levels<R>(w: usize, h: usize, f: impl FnOnce(&mut [u8], &mut Vec<usize>) -> R) -> R {
    thread_local! {
        static LEVELS: std::cell::RefCell<(Vec<u8>, Vec<usize>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    LEVELS.with_borrow_mut(|scratch| {
        let (buf, dirty) = (&mut scratch.0, &mut scratch.1);
        // Only nonzero coefficient locations are ever dirty. Clearing those is
        // substantially cheaper than zeroing a padded TX32 plane for every R-D
        // candidate, and padding bytes remain zero for the lifetime of the TLS.
        for &idx in dirty.iter() {
            buf[idx] = 0;
        }
        dirty.clear();
        // `get_lo_ctx_2d` reads up to two rows/columns past a position, so the
        // plane is padded exactly as the coder's own `levels` arrays are.
        let need = (w + 2) * (h + 2);
        if buf.len() < need {
            buf.resize(need, 0);
        }
        f(&mut buf[..need], dirty)
    })
}

/// Rate, in bits, that the coefficient coder will spend on `cf` — the same
/// symbols `encode_tx*_coeffs_adapt` would emit, costed against the live CDFs.
pub(crate) fn real_block_bits(cf: &[i32], scan: &[u32], c: &RateCtx) -> f32 {
    real_block_bits_bounded(cf, scan, c, f32::INFINITY)
}

/// As [`real_block_bits`], but ABORTS (returning `f32::INFINITY`) as soon as
/// the accumulated bits exceed `bound`. Bits only ever accumulate, so an
/// abort is EXACT: the true rate is >= the partial sum > bound, and a caller
/// that only compares `rd_cost(sse, lam, bits)` against a best already below
/// the bound rejects the candidate either way — byte-identical decisions,
/// less CDF walking (the exact-rate twin family is ~17% of encode
/// self-time; trial losers pay most of it).
pub(crate) fn real_block_bits_bounded(cf: &[i32], scan: &[u32], c: &RateCtx, bound: f32) -> f32 {
    let cost_table = cost_q_table();
    let (cls, pl, w, h) = (c.cls, c.plane, c.w, c.h);
    let skip_cdf = &c.cdfs.txb_skip[cls][c.skip_ctx];

    let Some(eob) = scan.iter().rposition(|&rc| cf[rc as usize] != 0) else {
        // All-zero: the block costs exactly one `all_zero = 1` symbol.
        return cdf_cost_with_table(skip_cdf, 1, cost_table);
    };

    let base_tok = &c.cdfs.base_tok[cls][pl];
    let br_tok = &c.cdfs.br_tok[cls][pl];
    let eob_base = &c.cdfs.eob_base[cls][pl];
    let eob_hi = &c.cdfs.eob_hi[cls][pl];
    let dc_sign = &c.cdfs.dc_sign[pl][c.dcs_ctx];

    let mut bits = cdf_cost_with_table(skip_cdf, 0, cost_table);
    if let Some((txtp_cdf, txtp)) = c.txtp {
        bits += cdf_cost_with_table(txtp_cdf, txtp, cost_table);
    }

    // eob == 0 is the DC-only tail: eob_pt(0), the DC base token off `eob_base`,
    // its base-range ladder, and the DC sign.
    if eob == 0 {
        let m = cf[0].unsigned_abs();
        bits += cdf_cost_with_table(c.eob_bin, 0, cost_table);
        bits += cdf_cost_with_table(&eob_base[0], m.min(3) as usize - 1, cost_table);
        if m.min(3) == 3 {
            bits += hi_tok_cost_with_table(m, &br_tok[0], cost_table);
        }
        bits += cdf_cost_with_table(dc_sign, (cf[0] < 0) as usize, cost_table);
        return bits;
    }

    // eob_pt: the bin, then the hi bit and the remaining bypass bits that pin
    // the exact position. This is the term the proxy charges a flat 2 bits for.
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    bits += cdf_cost_with_table(c.eob_bin, eob_bin, cost_table);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        bits += cdf_cost_with_table(&eob_hi[eob_bin], (eob >> nbits) & 1, cost_table);
        bits += nbits as f32; // equiprobable extra bits
    }

    if bits > bound {
        return f32::INFINITY;
    }
    debug_assert_eq!(scan.len(), w * h);
    let n = scan.len();
    let log2h = h.trailing_zeros() as usize;
    let stride = h;
    let offsets = if w < h {
        &LO_CTX_OFF_WLH
    } else if w > h {
        &LO_CTX_OFF_WGH
    } else {
        &LO_CTX_OFF
    };

    thread_local! {
        static COST_CACHE: std::cell::RefCell<CostCache> =
            const { std::cell::RefCell::new(CostCache::new()) };
    }
    struct CostCache {
        bt_c: [[f32; 4]; 32],
        bt_key: [u64; 32],
        bt_valid: [bool; 32],
        br_c: [[f32; 13]; 32],
        br_key: [u64; 32],
        br_valid: [bool; 32],
    }
    impl CostCache {
        const fn new() -> Self {
            CostCache {
                bt_c: [[0.0; 4]; 32],
                bt_key: [0; 32],
                bt_valid: [false; 32],
                br_c: [[0.0; 13]; 32],
                br_key: [0; 32],
                br_valid: [false; 32],
            }
        }
    }
    #[inline]
    fn cdf_key(cdf: &[u16]) -> u64 {
        // Coefficient CDFs have four coded partitions; any adaptation counter
        // after them cannot affect `cdf_cost`. Keying the four partitions keeps
        // the cache exact even if a mutable CDF is later passed here.
        u64::from(cdf[0])
            | (u64::from(cdf[1]) << 16)
            | (u64::from(cdf[2]) << 32)
            | (u64::from(cdf[3]) << 48)
    }
    COST_CACHE.with_borrow_mut(|cc| {
        with_levels(w, h, |levels, dirty| {
            // The eob coefficient uses the eob_base CDF (not base_tok) and a br
            // context that depends only on its own position.
            let ctx_e = 1 + (eob > (n >> 3)) as usize + (eob > (n >> 2)) as usize;
            let rc = scan[eob] as usize;
            let (ex, ey) = (rc >> log2h, rc & (h - 1));
            let m = cf[rc].unsigned_abs();
            let eob_tok = m.min(3) - 1;
            bits += cdf_cost_with_table(&eob_base[ctx_e], eob_tok as usize, cost_table);
            if eob_tok == 2 {
                let bc = if (ex | ey) > 1 { 14 } else { 7 };
                bits += hi_tok_cost_with_table(m, &br_tok[bc], cost_table);
            }
            let eob_pos = ex * stride + ey;
            levels[eob_pos] = level_byte(m);
            dirty.push(eob_pos);

            // Interior coefficients, reverse scan — each token's context comes from
            // the neighbors already written into `levels`.
            let mut ac_nonzero = 1usize; // the nonzero EOB coefficient
            for i in (1..eob).rev() {
                if i & 15 == 0 && bits > bound {
                    return f32::INFINITY;
                }
                let rc_i = scan[i] as usize;
                let (x, y) = (rc_i >> log2h, rc_i & (h - 1));
                let (ctx, hi_mag) = get_lo_ctx_2d(levels, x, y, offsets, stride);
                let m = cf[rc_i].unsigned_abs();
                let tok = m.min(3);
                let r = &base_tok[ctx];
                let key = cdf_key(r);
                if !cc.bt_valid[ctx] || cc.bt_key[ctx] != key {
                    cc.bt_valid[ctx] = true;
                    cc.bt_key[ctx] = key;
                    cc.bt_c[ctx] = [
                        cdf_cost_with_table(r, 0, cost_table),
                        cdf_cost_with_table(r, 1, cost_table),
                        cdf_cost_with_table(r, 2, cost_table),
                        cdf_cost_with_table(r, 3, cost_table),
                    ];
                }
                bits += cc.bt_c[ctx][tok as usize];
                if tok == 3 {
                    let mag = hi_mag & 63;
                    let bc = (if (y | x) > 1 { 14 } else { 7 })
                        + if mag > 12 { 6 } else { (mag + 1) >> 1 };
                    let bc = bc as usize;
                    let r = &br_tok[bc];
                    let key = cdf_key(r);
                    if !cc.br_valid[bc] || cc.br_key[bc] != key {
                        cc.br_valid[bc] = true;
                        cc.br_key[bc] = key;
                        cc.br_c[bc] = br_cum_row_with_table(r, cost_table);
                    }
                    let total_br =
                        (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE) as usize;
                    bits += cc.br_c[bc][total_br];
                    if m >= 15 {
                        bits += golomb_cost(m - 15);
                    }
                }
                if m != 0 {
                    let pos = x * stride + y;
                    levels[pos] = level_byte(m);
                    dirty.push(pos);
                    ac_nonzero += 1;
                }
            }

            // DC token (context 0) and its base-range ladder.
            let dm = cf[0].unsigned_abs();
            let dc_tok = dm.min(3);
            bits += cdf_cost_with_table(&base_tok[0], dc_tok as usize, cost_table);
            if dc_tok == 3 {
                let mag =
                    (levels[1] as u32 + levels[stride] as u32 + levels[stride + 1] as u32) & 63;
                let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
                bits += hi_tok_cost_with_table(dm, &br_tok[bc as usize], cost_table);
            }
            if cf[0] != 0 {
                bits += cdf_cost_with_table(dc_sign, (cf[0] < 0) as usize, cost_table);
            }

            // One bypass sign bit per nonzero AC (the Golomb tail is already folded
            // into `hi_tok_cost` above). Keep the original repeated f32 additions,
            // but avoid a second random-access scan over the coefficients.
            for _ in 0..ac_nonzero {
                bits += 1.0;
            }
            bits
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{SCAN_4X8, SCAN_8X4, SCAN_8X8};

    fn ctx<'a>(cdfs: &'a Cdfs) -> RateCtx<'a> {
        RateCtx {
            cdfs,
            cls: 1,
            plane: 0,
            w: 8,
            h: 8,
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
            real_block_bits(&clustered, &SCAN_8X8, &c) < real_block_bits(&scattered, &SCAN_8X8, &c)
        );
    }

    /// The two orientations use different raster strides and normative
    /// lo-context offset tables. Keep golden costs for both so neither silently
    /// falls back to the square table.
    #[test]
    fn rectangular_orientations_use_their_own_contexts() {
        let cdfs = Cdfs::new(0);
        let mut a = [0i32; 32];
        let mut b = [0i32; 32];
        for &(i, level) in &[(0usize, 3), (1, -2), (4, 1), (17, 4)] {
            a[SCAN_4X8[i] as usize] = level;
            b[SCAN_8X4[i] as usize] = level;
        }
        let mk = |w, h| RateCtx {
            cdfs: &cdfs,
            cls: 1,
            plane: 1,
            w,
            h,
            eob_bin: &cdfs.eob_bin_32_c,
            skip_ctx: 7,
            dcs_ctx: 0,
            txtp: None,
        };
        let ar = real_block_bits(&a, &SCAN_4X8, &mk(4, 8));
        let br = real_block_bits(&b, &SCAN_8X4, &mk(8, 4));
        assert!((ar - 38.532_284).abs() < 1e-5, "4x8 rate changed: {ar}");
        assert!((br - 38.220_8).abs() < 1e-5, "8x4 rate changed: {br}");
        assert_ne!(ar, br);
    }
}
