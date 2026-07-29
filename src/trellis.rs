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
use crate::aq_common::f_fmlaf;
use crate::coder::*;
use crate::coeffs::get_lo_ctx_2d;
use crate::cost::*;
use crate::tables::{COEFF_BASE_RANGE, LO_CTX_OFF, NUM_BASE_LEVELS, level_byte};
use crate::trellis_dist::{
    trellis_dist_current_zero_scan, trellis_dist_one, trellis_round_down_scan,
};

const TRELLIS_AOM_CALIB: f32 = 0.045025873597302174 * 0.5525;

fn trellis_tilt_mag_cap() -> f32 {
    crate::tuning::get().trellis_tilt_mag_cap
}

#[inline]
fn trellis_lambda_aom(dc_q: f32, _ac_q: f32) -> f32 {
    TRELLIS_AOM_CALIB * dc_q * dc_q * (3.3 + 0.0015 * dc_q)
}

#[inline]
fn scaled_trellis_lambda(dc_q: f32, ac_q: f32, lambda0: f32) -> f32 {
    trellis_lambda_aom(dc_q, ac_q) * (lambda0 / TRELLIS_LAMBDA0)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn trellis_optimize_ctx(
    cf: &mut [i32],
    tf: &[f32],
    dc_q: f32,
    ac_q: f32,
    scan: &[u32],
    lambda0: f32,
    w: usize,
    h: usize,
    cdfs: &Cdfs,
    cls: usize,
    plane: usize,
    eob_bin_cdf: &[u16],
    dcs_ctx: usize,
    qm_level: u8,
    _qindex: i32,
) {
    if lambda0 <= 0.0 {
        return;
    }
    let n = scan.len();
    // Read straight from the static QM table: this used to heap-allocate a
    // `w*w` Vec of weights on EVERY call and resolve the table offset once per
    // coefficient. `qm_row` resolves the offset once; the per-access work is a
    // byte load and a multiply, so no precomputed buffer is warranted.
    let qm_table = crate::quant::qm_row(qm_level, plane != 0, w, h);
    let qm_full = qm_level >= 6;
    let qm_at = |rc: usize| -> f32 {
        let wq = qm_table.map_or(32.0, |t| t[rc] as f32) / 32.0;
        if qm_full { wq * wq } else { wq }
    };
    let lambda = scaled_trellis_lambda(dc_q, ac_q, lambda0);
    // Shape-generic coefficient geometry, identical to the rate twin in
    // `rate.rs`: positions decompose against the HEIGHT (`rc >> log2h`,
    // `rc & (h - 1)`) with `stride == h`, and the base-token position offsets
    // come from the aspect-specific table. For a square block this is exactly
    // the old `w`-only form.
    let log2h = h.trailing_zeros() as usize;
    let stride = h;
    let lo_off = if w < h {
        &crate::tables::LO_CTX_OFF_WLH
    } else if w > h {
        &crate::tables::LO_CTX_OFF_WGH
    } else {
        &LO_CTX_OFF
    };
    // Hoist the per-(class, plane) CDF tables once for clarity (and to avoid
    // re-walking the nested arrays on every coefficient).
    let base_tok = &cdfs.base_tok[cls][plane];
    let br_tok = &cdfs.br_tok[cls][plane];
    let eob_hi = &cdfs.eob_hi[cls][plane];
    let eob_base = &cdfs.eob_base[cls][plane];
    let dc_sign = &cdfs.dc_sign[plane];
    let cost_table = cost_q_table();
    let dq2_dc = dc_q * dc_q;
    let dq2_ac = ac_q * ac_q;
    let band_t = cdfs.band_tilt;
    let n_inv = 1.0 / n as f32;

    let distw = |idx: usize, rc: usize, lev: i32| -> f32 {
        let mut d = trellis_dist_one(tf, rc, lev, dq2_dc, dq2_ac);
        if qm_table.is_some() {
            d *= qm_at(rc);
        }
        if band_t != 0.0 && tf[rc].abs() < trellis_tilt_mag_cap() {
            d /= 1.0 + band_t * (idx as f32 * n_inv);
        }
        d
    };

    // Gate lowered 256 -> 64 with the shared `br_cum_row` builder (2026-07-26
    // cost-precache pass): the eager 21-row build is ~84 cdf_cost calls, paid
    // back by any block whose candidates take the hi_tok ladder more than a
    // few dozen times — true from 8x8 up. 4x4 keeps the direct path.
    let use_br_table = n >= 64;
    let mut br_cum = [[0f32; 13]; 21];
    if use_br_table {
        for (row, br) in br_cum.iter_mut().zip(br_tok.iter()) {
            *row = br_cum_row_with_table(br, cost_table);
        }
    }
    // Base-range tail cost for magnitude `m` (>= 3) in br context `bc`.
    let hi_cost = |m: u32, bc: usize| -> f32 {
        if use_br_table {
            let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
            let mut bits = br_cum[bc][total_br as usize];
            if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            bits
        } else {
            hi_tok_cost_with_table(m, &br_tok[bc], cost_table)
        }
    };
    let eob: i32 = scan
        .iter()
        .rposition(|&rc| cf[rc as usize] != 0)
        .map_or(-1, |i| i as i32);
    if eob < 0 {
        return;
    }
    let eu = eob as usize;

    thread_local! {
    static SCRATCH: std::cell::RefCell<(
        Vec<u8>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<usize>,
    )> = const {
        std::cell::RefCell::new((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    };
    }
    SCRATCH.with_borrow_mut(|scratch| {
        let (levels, pre, suf0, dist_cur, dist_zero, dirty) = (
            &mut scratch.0,
            &mut scratch.1,
            &mut scratch.2,
            &mut scratch.3,
            &mut scratch.4,
            &mut scratch.5,
        );
        // Context levels are sparse after quantization. Clear only the nonzero
        // positions left by the preceding call instead of memset'ing the whole
        // padded TX plane for every candidate.
        for &idx in dirty.iter() {
            levels[idx] = 0;
        }
        dirty.clear();
        let levels_need = (w + 2) * (h + 2);
        if levels.len() < levels_need {
            levels.resize(levels_need, 0);
        }
        // Positions past the last nonzero (eu) contribute the same zero-level
        // distortion constant to every EOB candidate and to skip, so all the
        // scans below truncate to m = eu + 1 — byte-identical decisions, but
        // a mid-q TX32 goes from 1024-length passes to a few tens. (`n` still
        // feeds ctx_e and the band-tilt normalizer, which are spec/n-based.)
        let m = eu + 1;
        pre.resize(m + 1, 0.0);
        suf0.resize(m + 1, 0.0);
        dist_cur.resize(m, 0.0);
        dist_zero.resize(m, 0.0);
        let set_level = |levels: &mut [u8], rc: usize, m: u32| {
            levels[(rc >> log2h) * stride + (rc & (h - 1))] = level_byte(m);
        };
        for &rc32 in &scan[..eu + 1] {
            let rc = rc32 as usize;
            let magnitude = cf[rc].unsigned_abs();
            if magnitude != 0 {
                let pos = (rc >> log2h) * stride + (rc & (h - 1));
                levels[pos] = level_byte(magnitude);
                dirty.push(pos);
            }
        }

        // Interior base-token context + br context for a position, from `levels`.
        let interior_ctx = |levels: &[u8], rc: usize| -> (usize, usize) {
            let (x, y) = (rc >> log2h, rc & (h - 1));
            let (ctx, hi_mag) = get_lo_ctx_2d(levels, x, y, lo_off, stride);
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            (ctx, bc as usize)
        };
        let dc_brc = |levels: &[u8]| -> usize {
            let mag = (levels[1] as u32 + levels[stride] as u32 + levels[stride + 1] as u32) & 63;
            if mag > 12 {
                6
            } else {
                ((mag + 1) >> 1) as usize
            }
        };
        // Rate of an interior coefficient at level k (base_tok + br + AC sign).
        let interior_rate = |ctx: usize, bc: usize, k: u32| -> f32 {
            if k == 0 {
                return cdf_cost_with_table(&base_tok[ctx], 0, cost_table);
            }
            let tok = k.min(3);
            let mut b = cdf_cost_with_table(&base_tok[ctx], tok as usize, cost_table);
            if tok == 3 {
                b += hi_cost(k, bc);
            }
            b + 1.0 // AC sign (bypass)
        };

        // Step A: reverse-scan per-coefficient RD-best level (interior), then DC.
        for i in (1..(eob as usize)).rev() {
            let rc = scan[i] as usize;
            let l = cf[rc].unsigned_abs();
            if l == 0 {
                continue;
            }
            let (ctx, bc) = interior_ctx(levels, rc);
            // Hoist the four base-token costs (tok 0..=3) out of the k-loop; only the
            // br/Golomb tail (k >= 3) and distortion vary per candidate. Float-op
            // order matches `interior_rate` exactly so the choice is unchanged.
            let bt = &base_tok[ctx];
            let bt0 = cdf_cost_with_table(bt, 0, cost_table);
            let bt1 = cdf_cost_with_table(bt, 1, cost_table);
            let bt2 = cdf_cost_with_table(bt, 2, cost_table);
            let bt3 = cdf_cost_with_table(bt, 3, cost_table);
            let rate_k = |k: u32| -> f32 {
                match k {
                    0 => bt0,
                    1 => bt1 + 1.0,
                    2 => bt2 + 1.0,
                    _ => (bt3 + hi_cost(k, bc)) + 1.0,
                }
            };
            let mut best_k = l;
            let mut best_c = distw(i, rc, l as i32) + rate_cost(lambda, rate_k(l));

            for k in (0..l).rev() {
                let dk = distw(i, rc, k as i32);
                // dist grows monotonically as k falls below l (l <= |tf|), and the
                // rate is non-negative, so once dist alone reaches best_c no smaller
                // level can win. Exact, just stops the scan early.
                if dk >= best_c {
                    break;
                }
                let c = dk + rate_cost(lambda, rate_k(k));
                if c < best_c {
                    best_c = c;
                    best_k = k;
                }
            }
            if best_k != l {
                cf[rc] = if cf[rc] < 0 {
                    -(best_k as i32)
                } else {
                    best_k as i32
                };
                set_level(levels, rc, best_k);
            }
        }
        {
            let rc = scan[0] as usize;
            let l = cf[rc].unsigned_abs();
            if l != 0 {
                let bc = dc_brc(levels);
                let sgn = (cf[rc] < 0) as usize;
                let dc_rate = |k: u32| -> f32 {
                    if k == 0 {
                        return cdf_cost_with_table(&base_tok[0], 0, cost_table);
                    }
                    let tok = k.min(3);
                    let mut b = cdf_cost_with_table(&base_tok[0], tok as usize, cost_table);
                    if tok == 3 {
                        b += hi_cost(k, bc);
                    }
                    b + cdf_cost_with_table(&dc_sign[dcs_ctx], sgn, cost_table)
                };
                let mut best_k = l;
                let mut best_c = distw(0, rc, l as i32) + rate_cost(lambda, dc_rate(l));
                for k in (0..l).rev() {
                    let dk = distw(0, rc, k as i32);
                    if dk >= best_c {
                        break;
                    }
                    let c = dk + rate_cost(lambda, dc_rate(k));
                    if c < best_c {
                        best_c = c;
                        best_k = k;
                    }
                }
                if best_k != l {
                    cf[rc] = if cf[rc] < 0 {
                        -(best_k as i32)
                    } else {
                        best_k as i32
                    };
                    set_level(levels, rc, best_k);
                }
            }
        }

        // Step B: EOB-position selection with accurate eob_pt / eob_base costs.
        let eob_pt_cost = |e: usize| -> f32 {
            let bin = if e < 2 {
                e
            } else {
                32 - (e as u32).leading_zeros() as usize
            };
            let mut c = cdf_cost_with_table(eob_bin_cdf, bin, cost_table);
            if bin > 1 {
                let nbits = bin - 2;
                c += cdf_cost_with_table(&eob_hi[bin], (e >> nbits) & 1, cost_table);
                c += nbits as f32; // remaining eob offset bits (bypass)
            }
            c
        };
        let eob_coeff_cost = |e: usize, m: u32| -> f32 {
            let ctx_e = 1 + (e > n / 8) as usize + (e > n / 4) as usize;
            let tok = m.min(3);
            let mut c = cdf_cost_with_table(&eob_base[ctx_e], tok as usize - 1, cost_table);
            if tok == 3 {
                let rc = scan[e] as usize;
                let (ex, ey) = (rc >> log2h, rc & (h - 1));
                let bc = if (ex | ey) > 1 { 14 } else { 7 };
                c += hi_cost(m, bc);
            }
            c + 1.0 // sign
        };

        // pre[e] = interior cost of positions [1, e-1]; an EOB candidate at e then
        // adds only e's own eob-coeff cost. dist precomputed in scan order.
        pre[0] = 0.0;
        pre[1] = 0.0;
        trellis_dist_current_zero_scan(dist_cur, dist_zero, tf, cf, &scan[..m], dq2_dc, dq2_ac);
        if qm_table.is_some() {
            for i in 0..m {
                let rc = scan[i] as usize;
                let w2 = qm_at(rc);
                dist_cur[i] *= w2;
                dist_zero[i] *= w2;
            }
        }
        if band_t != 0.0 {
            for i in 0..m {
                let rc = scan[i] as usize;
                if tf[rc].abs() < trellis_tilt_mag_cap() {
                    let wgt = 1.0 / (1.0 + band_t * (i as f32 * n_inv));
                    dist_zero[i] *= wgt;
                    dist_cur[i] *= wgt;
                }
            }
        }
        let mut acc = 0.0f32; // pre[1]: empty prefix

        // Interior positions 1..=eu written to pre[2..=eu+1]: pre[i+1]=sum_{1..i}.
        let scan_it = scan[1..=eu].iter();
        let dist_it = dist_cur[1..=eu].iter();
        let pre_it = pre[2..=eu + 1].iter_mut();
        for ((&rc_u32, &dist), out_pre) in scan_it.zip(dist_it).zip(pre_it) {
            let rc = rc_u32 as usize;
            let (ctx, bc) = interior_ctx(levels, rc);
            let r = interior_rate(ctx, bc, cf[rc].unsigned_abs());
            acc += rate_cost(lambda, r) + dist;
            *out_pre = acc;
        }

        let _ = acc; // prefix ends at eu (positions past eu are truncated)

        suf0[m] = 0.0;

        let mut sacc = 0.0f32;

        // Suffix over 1..m, reversed.
        for (&dist, out_suf) in dist_zero[1..m]
            .iter()
            .rev()
            .zip(suf0[1..m].iter_mut().rev())
        {
            sacc += dist;
            *out_suf = sacc;
        }
        // DC contribution (rate + distortion), constant across EOB choices ≥ 1.
        let dc_rc = scan[0] as usize;
        let dc_m = cf[dc_rc].unsigned_abs();
        let dc_cost = if dc_m == 0 {
            rate_cost(lambda, cdf_cost_with_table(&base_tok[0], 0, cost_table))
        } else {
            let bc = dc_brc(levels);
            let tok = dc_m.min(3);
            let mut b = cdf_cost_with_table(&base_tok[0], tok as usize, cost_table);
            if tok == 3 {
                b += hi_cost(dc_m, bc);
            }
            b += cdf_cost_with_table(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize, cost_table);
            rate_cost(lambda, b)
        } + dist_cur[0];

        let mut best_e: i32 = -1;
        let mut best_m: u32 = 0; // chosen level at the EOB coefficient
        let mut best_cost = f32::INFINITY;

        // eob_coeff cost + distortion for magnitude m at position e (m >= 1).
        let eob_cand = |e: usize, m: u32| -> f32 {
            rate_cost(lambda, eob_coeff_cost(e, m)) + distw(e, scan[e] as usize, m as i32)
        };

        for (e, (&pre_e, &suf_next)) in pre[..m].iter().zip(&suf0[1..=m]).enumerate().skip(1) {
            let rc = scan[e] as usize;
            let m = cf[rc].unsigned_abs();
            if m == 0 {
                continue; // EOB must land on a nonzero
            }
            let base = dc_cost + pre_e + rate_cost(lambda, eob_pt_cost(e)) + suf_next;
            // Try the EOB coefficient at its level and, like libaom, at level-1.
            let mut m_best = m;
            let mut c = base + eob_cand(e, m);
            if m >= 2 {
                let c_low = base + eob_cand(e, m - 1);
                if c_low < c {
                    c = c_low;
                    m_best = m - 1;
                }
            }
            if c < best_cost {
                best_cost = c;
                best_e = e as i32;
                best_m = m_best;
            }
        }
        // EOB at DC (only DC nonzero) and the all-zero (txb_skip) alternative.
        if dc_m != 0 {
            let ctx_e = 1usize; // e == 0
            let tok = dc_m.min(3);
            let mut c0 = cdf_cost_with_table(eob_bin_cdf, 0, cost_table)
                + cdf_cost_with_table(&eob_base[ctx_e], tok as usize - 1, cost_table);
            if tok == 3 {
                c0 += hi_cost(dc_m, dc_brc(levels));
            }
            c0 += cdf_cost_with_table(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize, cost_table);
            let total0 = rate_cost(lambda, c0) + dist_cur[0] + suf0[1];
            if total0 < best_cost {
                best_cost = total0;
                best_e = 0;
                best_m = dc_m;
            }
        }
        let skip_cost = suf0[1] + dist_zero[0] + rate_cost(lambda, 1.0f32);
        if best_e < 0 || skip_cost < best_cost {
            for &rc32 in scan[..m].iter() {
                cf[rc32 as usize] = 0;
            }
        } else {
            let e = best_e as usize;
            let rc = scan[e] as usize;
            if best_m != cf[rc].unsigned_abs() {
                cf[rc] = if cf[rc] < 0 {
                    -(best_m as i32)
                } else {
                    best_m as i32
                };
            }
            for &rc in scan[(e + 1)..m].iter() {
                cf[rc as usize] = 0;
            }
        }
    });
}

pub(crate) fn trellis_optimize(
    cf: &mut [i32],
    tf: &[f32],
    dc_q: f32,
    ac_q: f32,
    scan: &[u32],
    lambda0: f32,
) {
    if lambda0 <= 0.0 {
        return; // trellis disabled
    }
    let n = scan.len();
    let lambda = scaled_trellis_lambda(dc_q, ac_q, lambda0);
    let (dc_q2, ac_q2) = (dc_q * dc_q, ac_q * ac_q);

    let Some(eob_idx) = scan[..n].iter().rposition(|&rc| cf[rc as usize] != 0) else {
        return; // already all-zero
    };

    // Step A: per-coefficient round-down (toward zero) by local R-D.
    trellis_round_down_scan(cf, tf, &scan[..=eob_idx], dc_q2, ac_q2, lambda);

    // Everything past the last nonzero is already zero, and a zero coefficient
    // has dist_cur == dist_zero, so the tail past eob_idx adds the SAME
    // constant to every EOB candidate and to the skip cost — it cannot move
    // the argmin. Truncate every scan to m = eob_idx + 1 (a mid-q TX32 has
    // n = 1024 but eob ~ tens; decisions are byte-identical to the full scan).
    let m = eob_idx + 1;

    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    }
    SCRATCH.with_borrow_mut(|scratch| {
        let (suf0, dist_cur, dist_zero) = (&mut scratch.0, &mut scratch.1, &mut scratch.2);

        dist_cur.resize(m, 0.0);
        dist_zero.resize(m, 0.0);
        trellis_dist_current_zero_scan(dist_cur, dist_zero, tf, cf, &scan[..m], dc_q2, ac_q2);

        suf0.resize(m + 1, 0.0); // distortion of zeroing coeffs from i..m
        suf0[m] = 0.0; // cumulative seed (read as suf0[m]; not written by the loop)
        assert!(suf0.len() > m, "suf0 must be indexed up to m");
        assert!(dist_zero.len() >= m);
        assert!(dist_cur.len() >= m);
        for i in (0..m).rev() {
            suf0[i] = suf0[i + 1] + dist_zero[i];
        }
        let eob_sig = |e: usize| -> f32 {
            let bin = if e < 2 {
                e
            } else {
                (32 - (e as u32).leading_zeros()) as usize
            };
            let extra = if bin > 1 { bin - 2 } else { 0 };
            f_fmlaf(bin as f32, 0.9, extra as f32 + 2.0) // eob_pt + extra bits + eob_base token
        };

        let mut best_e: i32 = -1;
        let mut best_cost = f32::INFINITY;
        let mut pre = 0.0f32; // interior cost of coefficients strictly before e
        for (e, (&rc32, &suf0)) in scan[..m].iter().zip(suf0[1..].iter()).enumerate() {
            let rc = rc32 as usize;
            if cf[rc] != 0 {
                let c = pre + dist_cur[e] + rate_cost(lambda, eob_sig(e)) + suf0;
                if c < best_cost {
                    best_cost = c;
                    best_e = e as i32;
                }
            }
            // Preserve the original left-to-right f32 recurrence exactly while
            // folding the old prefix-building pass into candidate selection.
            pre = pre + dist_cur[e] + rate_cost(lambda, coef_rate_bits(cf[rc].unsigned_abs()));
        }
        let skip_cost = suf0[0] + rate_cost(lambda, 1.0f32); // zero everything + the txb_skip flag
        if best_e < 0 || skip_cost < best_cost {
            for &rc32 in scan[..m].iter() {
                cf[rc32 as usize] = 0;
            }
        } else {
            for &x32 in scan[(best_e as usize + 1)..m].iter() {
                cf[x32 as usize] = 0;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coder::Cdfs;
    use crate::tables::SCAN_16X16;

    #[test]
    fn requested_lambda_scales_trellis_rd_cost() {
        let base = scaled_trellis_lambda(200.0, 220.0, TRELLIS_LAMBDA0);
        let reduced = scaled_trellis_lambda(200.0, 220.0, TRELLIS_LAMBDA0 * 0.25);
        assert_eq!(base, trellis_lambda_aom(200.0, 220.0));
        assert!((reduced - base * 0.25).abs() <= f32::EPSILON * base);
    }

    fn random_block(rng: &mut impl FnMut() -> u64) -> ([f32; 256], [i32; 256]) {
        let (mut tf, mut cf) = ([0.0f32; 256], [0i32; 256]);
        for _ in 0..1 + (rng() % 40) as usize {
            let rc = (rng() % 256) as usize;
            let mag = (rng() % 6) as f32 + rng() as f32 / u64::MAX as f32;
            let sign = if rng() & 1 == 0 { 1.0 } else { -1.0 };
            tf[rc] = sign * mag;
            cf[rc] = (sign * mag.round()) as i32;
        }
        (tf, cf)
    }

    // Trellis only zeroes past the EOB and lowers magnitudes toward |tf|; it never
    // raises a level, flips a sign, or exceeds the original magnitude.
    #[test]
    fn trellis_output_is_valid() {
        let cdfs = Cdfs::new(0);
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..1000 {
            let (tf, cf) = random_block(&mut rng);
            let mut opt = cf;
            trellis_optimize_ctx(
                &mut opt,
                &tf,
                200.0,
                220.0,
                &SCAN_16X16,
                0.05,
                16,
                16,
                &cdfs,
                2,
                0,
                &cdfs.eob_bin_256_l,
                0,
                15,
                60,
            );
            if let Some(e) = SCAN_16X16.iter().rposition(|&r| opt[r as usize] != 0) {
                for &r in &SCAN_16X16[e + 1..] {
                    assert_eq!(opt[r as usize], 0);
                }
            }
            for rc in 0..256 {
                assert!(opt[rc].unsigned_abs() <= cf[rc].unsigned_abs());
                if opt[rc] != 0 {
                    assert_eq!(opt[rc].signum(), cf[rc].signum());
                }
            }
        }
    }
}
