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
use crate::av1_coder::*;
use crate::coeffs::get_lo_ctx_2d;
use crate::cost::*;
use crate::tables::{COEFF_BASE_RANGE, LO_CTX_OFF, NUM_BASE_LEVELS, level_byte};
use crate::trellis_dist::{
    trellis_dist_current_zero_scan, trellis_dist_one, trellis_round_down_scan,
};

/// Trellis RD lambda on libaom's KF rdmult shape `dc_q^2*(3.3+0.0015*dc_q)`
/// (av1/encoder/txb_rdopt.c uses the frame DC-quant rdmult), calibrated to the
/// legacy `TRELLIS_LAMBDA0*ac_q^2*2` at q=128.
const TRELLIS_AOM_CALIB: f32 = 0.045025873597302174;
#[inline]
fn trellis_lambda_aom(dc_q: f32, _ac_q: f32) -> f32 {
    TRELLIS_AOM_CALIB * dc_q * dc_q * (3.3 + 0.0015 * dc_q)
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
    cdfs: &Cdfs,
    cls: usize,
    plane: usize,
    eob_bin_cdf: &[u16],
    dcs_ctx: usize,
) {
    if lambda0 <= 0.0 {
        return;
    }
    let n = scan.len();
    let lambda = trellis_lambda_aom(dc_q, ac_q);
    let log2w = w.trailing_zeros() as usize;
    let stride = w;
    // Hoist the per-(class, plane) CDF tables once for clarity (and to avoid
    // re-walking the nested arrays on every coefficient).
    let base_tok = &cdfs.base_tok[cls][plane];
    let br_tok = &cdfs.br_tok[cls][plane];
    let eob_hi = &cdfs.eob_hi[cls][plane];
    let eob_base = &cdfs.eob_base[cls][plane];
    let dc_sign = &cdfs.dc_sign[plane];
    let dq2_dc = dc_q * dc_q;
    let dq2_ac = ac_q * ac_q;
    let dist = |rc: usize, lev: i32| trellis_dist_one(tf, rc, lev, dq2_dc, dq2_ac);

    // Precompute the base-range (hi_tok) ladder cost for every br context and
    // every total_br in 0..=12, once per call. `hi_tok_cost` otherwise reruns a
    // 4-step `cdf_cost` ladder for every level-3+ coefficient (hot at high
    // quality). Only worth the ~0.5us setup for the larger transforms (n >= 256),
    // where it is called hundreds of times; small blocks use the direct path.
    // Accumulation order matches `hi_tok_cost` exactly, so the chosen levels are
    // identical either way.
    let use_br_table = n >= 256;
    let mut br_cum = [[0f32; 13]; 21];
    if use_br_table {
        for (row, br) in br_cum.iter_mut().zip(br_tok.iter()) {
            let c = [
                cdf_cost(br, 0),
                cdf_cost(br, 1),
                cdf_cost(br, 2),
                cdf_cost(br, 3),
            ];
            for (j, slot) in row.iter_mut().enumerate() {
                let mut coded = 0i32;
                let mut bits = 0.0f32;
                for _ in 0..(COEFF_BASE_RANGE / 3) {
                    let s = (j as i32 - coded).min(3);
                    bits += c[s as usize];
                    coded += s;
                    if s < 3 {
                        break;
                    }
                }
                *slot = bits;
            }
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
            hi_tok_cost(m, &br_tok[bc])
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
        )> = const {
            std::cell::RefCell::new((
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ))
        };
    }
    SCRATCH.with_borrow_mut(|scratch| {
        let (levels, pre, suf0, dist_cur, dist_zero) = (
            &mut scratch.0,
            &mut scratch.1,
            &mut scratch.2,
            &mut scratch.3,
            &mut scratch.4,
        );
        levels.clear();
        levels.resize(w * (w + 4), 0);
        let set_level = |levels: &mut [u8], rc: usize, m: u32| {
            levels[(rc >> log2w) * stride + (rc & (w - 1))] = level_byte(m);
        };
        for &rc32 in &scan[..eu + 1] {
            let rc = rc32 as usize;
            set_level(levels, rc, cf[rc].unsigned_abs());
        }

        // Interior base-token context + br context for a position, from `levels`.
        let interior_ctx = |levels: &[u8], rc: usize| -> (usize, usize) {
            let (x, y) = (rc >> log2w, rc & (w - 1));
            let (ctx, hi_mag) = get_lo_ctx_2d(levels, x, y, &LO_CTX_OFF, stride);
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
                return cdf_cost(&base_tok[ctx], 0);
            }
            let tok = k.min(3);
            let mut b = cdf_cost(&base_tok[ctx], tok as usize);
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
            let bt0 = cdf_cost(bt, 0);
            let bt1 = cdf_cost(bt, 1);
            let bt2 = cdf_cost(bt, 2);
            let bt3 = cdf_cost(bt, 3);
            let rate_k = |k: u32| -> f32 {
                match k {
                    0 => bt0,
                    1 => bt1 + 1.0,
                    2 => bt2 + 1.0,
                    _ => (bt3 + hi_cost(k, bc)) + 1.0,
                }
            };
            let mut best_k = l;
            let mut best_c = dist(rc, l as i32) + rate_cost(lambda, rate_k(l));

            for k in (0..l).rev() {
                let dk = dist(rc, k as i32);
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
                        return cdf_cost(&base_tok[0], 0);
                    }
                    let tok = k.min(3);
                    let mut b = cdf_cost(&base_tok[0], tok as usize);
                    if tok == 3 {
                        b += hi_cost(k, bc);
                    }
                    b + cdf_cost(&dc_sign[dcs_ctx], sgn)
                };
                let mut best_k = l;
                let mut best_c = dist(rc, l as i32) + rate_cost(lambda, dc_rate(l));
                for k in (0..l).rev() {
                    let dk = dist(rc, k as i32);
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
            let mut c = cdf_cost(eob_bin_cdf, bin);
            if bin > 1 {
                let nbits = bin - 2;
                c += cdf_cost(&eob_hi[bin], (e >> nbits) & 1);
                c += nbits as f32; // remaining eob offset bits (bypass)
            }
            c
        };
        let eob_coeff_cost = |e: usize, m: u32| -> f32 {
            let ctx_e = 1 + (e > n / 8) as usize + (e > n / 4) as usize;
            let tok = m.min(3);
            let mut c = cdf_cost(&eob_base[ctx_e], tok as usize - 1);
            if tok == 3 {
                let rc = scan[e] as usize;
                let (ex, ey) = (rc >> log2w, rc & (w - 1));
                let bc = if (ex | ey) > 1 { 14 } else { 7 };
                c += hi_cost(m, bc);
            }
            c + 1.0 // sign
        };

        // pre[e] = interior cost of positions [1, e-1]; an EOB candidate at e then
        // adds only e's own eob-coeff cost. dist precomputed in scan order.
        pre.resize(n + 1, 0.0);
        pre[0] = 0.0;
        pre[1] = 0.0;
        dist_cur.resize(n, 0.0);
        dist_zero.resize(n, 0.0);
        trellis_dist_current_zero_scan(dist_cur, dist_zero, tf, cf, scan, dq2_dc, dq2_ac);
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

        // Trailing positions eu+1..n coded as zero: distortion only.
        let dist_tail = dist_zero[eu + 1..n].iter();
        let pre_tail = pre[eu + 2..n + 1].iter_mut();
        for (&dist, out_pre) in dist_tail.zip(pre_tail) {
            acc += dist;
            *out_pre = acc;
        }

        suf0.resize(n + 1, 0.0);
        suf0[n] = 0.0;

        let mut sacc = 0.0f32;

        // Suffix over 1..n, reversed.
        for (&dist, out_suf) in dist_zero[1..n]
            .iter()
            .rev()
            .zip(suf0[1..n].iter_mut().rev())
        {
            sacc += dist;
            *out_suf = sacc;
        }
        // DC contribution (rate + distortion), constant across EOB choices ≥ 1.
        let dc_rc = scan[0] as usize;
        let dc_m = cf[dc_rc].unsigned_abs();
        let dc_cost = if dc_m == 0 {
            rate_cost(lambda, cdf_cost(&base_tok[0], 0))
        } else {
            let bc = dc_brc(levels);
            let tok = dc_m.min(3);
            let mut b = cdf_cost(&base_tok[0], tok as usize);
            if tok == 3 {
                b += hi_cost(dc_m, bc);
            }
            b += cdf_cost(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize);
            rate_cost(lambda, b)
        } + dist_cur[0];

        let mut best_e: i32 = -1;
        let mut best_m: u32 = 0; // chosen level at the EOB coefficient
        let mut best_cost = f32::INFINITY;

        // eob_coeff cost + distortion for magnitude m at position e (m >= 1).
        let eob_cand = |e: usize, m: u32| -> f32 {
            rate_cost(lambda, eob_coeff_cost(e, m)) + dist(scan[e] as usize, m as i32)
        };

        for (e, (&pre_e, &suf_next)) in pre[..n].iter().zip(&suf0[1..=n]).enumerate().skip(1) {
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
            let mut c0 = cdf_cost(eob_bin_cdf, 0) + cdf_cost(&eob_base[ctx_e], tok as usize - 1);
            if tok == 3 {
                c0 += hi_cost(dc_m, dc_brc(levels));
            }
            c0 += cdf_cost(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize);
            let total0 = rate_cost(lambda, c0) + dist_cur[0] + suf0[1];
            if total0 < best_cost {
                best_cost = total0;
                best_e = 0;
                best_m = dc_m;
            }
        }
        let skip_cost = suf0[1] + dist_zero[0] + rate_cost(lambda, 1.0f32);
        if best_e < 0 || skip_cost < best_cost {
            for &rc32 in scan.iter() {
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
            for i in (e + 1)..n {
                cf[scan[i] as usize] = 0;
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
    let lambda = trellis_lambda_aom(dc_q, ac_q);
    let (dc_q2, ac_q2) = (dc_q * dc_q, ac_q * ac_q);

    let Some(eob_idx) = scan[..n].iter().rposition(|&rc| cf[rc as usize] != 0) else {
        return; // already all-zero
    };

    // Step A: per-coefficient round-down (toward zero) by local R-D.
    trellis_round_down_scan(cf, tf, &scan[..=eob_idx], dc_q2, ac_q2, lambda);

    thread_local! {
        #[allow(clippy::type_complexity)]
        static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
    }
    SCRATCH.with_borrow_mut(|scratch| {
        let (suf0, pre, dist_cur, dist_zero) = (
            &mut scratch.0,
            &mut scratch.1,
            &mut scratch.2,
            &mut scratch.3,
        );

        dist_cur.resize(n, 0.0);
        dist_zero.resize(n, 0.0);
        trellis_dist_current_zero_scan(dist_cur, dist_zero, tf, cf, scan, dc_q2, ac_q2);

        suf0.resize(n + 1, 0.0); // distortion of zeroing coeffs from i..n
        suf0[n] = 0.0; // cumulative seed (read as suf0[n]; not written by the loop)
        assert!(suf0.len() > n, "suf0 must be indexed up to n");
        for i in (0..n).rev() {
            suf0[i] = suf0[i + 1] + dist_zero[i];
        }
        pre.resize(n + 1, 0.0); // interior cost of coeffs strictly before i
        assert!(pre.len() > n, "pre must be indexed up to n");

        pre[0] = 0.0; // empty-prefix seed
        for (i, &rc32) in scan[..n].iter().enumerate() {
            let rc = rc32 as usize;
            pre[i + 1] =
                pre[i] + dist_cur[i] + rate_cost(lambda, coef_rate_bits(cf[rc].unsigned_abs()));
        }
        let eob_sig = |e: usize| -> f32 {
            let bin = if e < 2 {
                e
            } else {
                (32 - (e as u32).leading_zeros()) as usize
            };
            let extra = if bin > 1 { bin - 2 } else { 0 };
            (bin as f32) * 0.9 + extra as f32 + 2.0 // eob_pt + extra bits + eob_base token
        };

        let mut best_e: i32 = -1;
        let mut best_cost = f32::INFINITY;
        for (e, ((&rc32, &pre), &suf0)) in scan[..n]
            .iter()
            .zip(pre.iter())
            .zip(suf0[1..].iter())
            .enumerate()
        {
            let rc = rc32 as usize;
            if cf[rc] == 0 {
                continue; // the EOB must land on a nonzero
            }
            let c = pre + dist_cur[e] + rate_cost(lambda, eob_sig(e)) + suf0;
            if c < best_cost {
                best_cost = c;
                best_e = e as i32;
            }
        }
        let skip_cost = suf0[0] + rate_cost(lambda, 1.0f32); // zero everything + the txb_skip flag
        if best_e < 0 || skip_cost < best_cost {
            for &rc32 in scan.iter() {
                cf[rc32 as usize] = 0;
            }
        } else {
            for &x32 in scan[(best_e as usize + 1)..n].iter() {
                cf[x32 as usize] = 0;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1_coder::Cdfs;
    use crate::tables::SCAN_16X16;

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
                &cdfs,
                2,
                0,
                &cdfs.eob_bin_256_l,
                0,
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
