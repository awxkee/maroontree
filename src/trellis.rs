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
use crate::av1real::*;
use crate::coeffs::get_lo_ctx_2d;
use crate::cost::*;
use crate::tables::{COEFF_BASE_RANGE, LO_CTX_OFF, NUM_BASE_LEVELS, level_byte};

/// Context-accurate RDOQ for the 2D square luma transforms (TX_8X8/16X16/32X32,
/// `cls` 1/2/3). Unlike [`trellis_optimize`], the per-coefficient rate is the
/// *real* coding cost: the base token cost depends on the neighbour-magnitude
/// context (so coefficients in sparse regions are correctly cheap), the
/// base-range/Golomb tail and the EOB signalling are priced from the live CDFs.
/// Level reduction runs in reverse scan order so forward-neighbour contexts are
/// stable; a final pass re-selects the EOB with accurate `eob_pt`/`eob_base`
/// costs. Only the chosen levels change, so the result still decodes exactly.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn trellis_optimize_ctx(
    cf: &mut [i32],
    tf: &[f64],
    dc_q: f64,
    ac_q: f64,
    scan: &[usize],
    lambda0: f64,
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
    let lambda = lambda0 * ac_q * ac_q;
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
    let dist = |rc: usize, lev: i32| {
        let dq2 = if rc == 0 { dq2_dc } else { dq2_ac };
        let e = tf[rc].abs() - (lev.abs() as f64);
        dq2 * (e * e)
    };

    // Precompute the base-range (hi_tok) ladder cost for every br context and
    // every total_br in 0..=12, once per call. `hi_tok_cost` otherwise reruns a
    // 4-step `cdf_cost` ladder for every level-3+ coefficient (hot at high
    // quality). Only worth the ~0.5us setup for the larger transforms (n >= 256),
    // where it is called hundreds of times; small blocks use the direct path.
    // Accumulation order matches `hi_tok_cost` exactly, so the chosen levels are
    // identical either way.
    let use_br_table = n >= 256;
    let mut br_cum = [[0f64; 13]; 21];
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
                let mut bits = 0.0;
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
    let hi_cost = |m: u32, bc: usize| -> f64 {
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

    // Last nonzero in scan order. `rposition` scans from the end and stops at the
    // first hit, and iterating `scan` drops its bounds check (only `cf[rc]` is a
    // random access). Same value as the forward max-index scan.
    let eob: i32 = scan
        .iter()
        .rposition(|&rc| cf[rc] != 0)
        .map_or(-1, |i| i as i32);
    if eob < 0 {
        return;
    }
    let eu = eob as usize;

    // Reuse scratch allocations across calls (this runs once per coded transform
    // block, so per-call alloc+zero+free of `levels`/`pre`/`suf0`/`irate` shows
    // up in profiles). Buffers are taken from a thread-local pool and returned at
    // the single exit below. Entries are fully overwritten before use except the
    // cumulative seed (`suf0[n]`) and `levels` (a sparse magnitude map
    // that must start zeroed), which are reset explicitly.

    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<f64>, Vec<f64>, Vec<f64>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
    }
    let (mut levels, mut pre, mut suf0, mut irate) = SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        (
            std::mem::take(&mut b.0),
            std::mem::take(&mut b.1),
            std::mem::take(&mut b.2),
            std::mem::take(&mut b.3),
        )
    });
    levels.clear();
    levels.resize(w * (w + 4), 0);
    let set_level = |levels: &mut [u8], rc: usize, m: u32| {
        levels[(rc >> log2w) * stride + (rc & (w - 1))] = level_byte(m);
    };
    for &rc in &scan[..eu + 1] {
        set_level(&mut levels, rc, cf[rc].unsigned_abs());
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
    let interior_rate = |ctx: usize, bc: usize, k: u32| -> f64 {
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
        let rc = scan[i];
        let l = cf[rc].unsigned_abs();
        if l == 0 {
            continue;
        }
        let (ctx, bc) = interior_ctx(&levels, rc);
        // Hoist the four base-token costs (tok 0..=3) out of the k-loop; only the
        // br/Golomb tail (k >= 3) and distortion vary per candidate. Float-op
        // order matches `interior_rate` exactly so the choice is unchanged.
        let bt = &base_tok[ctx];
        let bt0 = cdf_cost(bt, 0);
        let bt1 = cdf_cost(bt, 1);
        let bt2 = cdf_cost(bt, 2);
        let bt3 = cdf_cost(bt, 3);
        let rate_k = |k: u32| -> f64 {
            match k {
                0 => bt0,
                1 => bt1 + 1.0,
                2 => bt2 + 1.0,
                _ => (bt3 + hi_cost(k, bc)) + 1.0,
            }
        };
        let mut best_k = l;
        let mut best_c = dist(rc, l as i32) + lambda * rate_k(l);

        for k in (0..l).rev() {
            let dk = dist(rc, k as i32);
            // dist grows monotonically as k falls below l (l <= |tf|), and the
            // rate is non-negative, so once dist alone reaches best_c no smaller
            // level can win. Exact, just stops the scan early.
            if dk >= best_c {
                break;
            }
            let c = dk + lambda * rate_k(k);
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
            set_level(&mut levels, rc, best_k);
        }
    }
    {
        let rc = scan[0];
        let l = cf[rc].unsigned_abs();
        if l != 0 {
            let bc = dc_brc(&levels);
            let sgn = (cf[rc] < 0) as usize;
            let dc_rate = |k: u32| -> f64 {
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
            let mut best_c = dist(rc, l as i32) + lambda * dc_rate(l);
            for k in (0..l).rev() {
                let dk = dist(rc, k as i32);
                if dk >= best_c {
                    break;
                }
                let c = dk + lambda * dc_rate(k);
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
                set_level(&mut levels, rc, best_k);
            }
        }
    }

    // Step B: EOB-position selection with accurate eob_pt / eob_base costs.
    let eob_pt_cost = |e: usize| -> f64 {
        let bin = if e < 2 {
            e
        } else {
            32 - (e as u32).leading_zeros() as usize
        };
        let mut c = cdf_cost(eob_bin_cdf, bin);
        if bin > 1 {
            let nbits = bin - 2;
            c += cdf_cost(&eob_hi[bin], (e >> nbits) & 1);
            c += nbits as f64; // remaining eob offset bits (bypass)
        }
        c
    };
    let eob_coeff_cost = |e: usize, m: u32| -> f64 {
        let ctx_e = 1 + (e > n / 8) as usize + (e > n / 4) as usize;
        let tok = m.min(3);
        let mut c = cdf_cost(&eob_base[ctx_e], tok as usize - 1);
        if tok == 3 {
            let rc = scan[e];
            let (ex, ey) = (rc >> log2w, rc & (w - 1));
            let bc = if (ex | ey) > 1 { 14 } else { 7 };
            c += hi_cost(m, bc);
        }
        c + 1.0 // sign
    };

    // Interior (base_tok) rate of each position at its current level, for the
    // running prefix; positions are priced as interior even if they will end up
    // being the EOB (corrected by swapping in eob_coeff_cost at the candidate).
    // Driven by zipped slice iterators so the sequential index checks drop out;
    // accumulation order (`acc + lambda*r + d`) matches the indexed form exactly.
    pre.resize(n + 1, 0.0);
    irate.resize(n, 0.0);
    let mut acc = 0.0f64; // pre[1]: empty prefix
    // Interior positions [1, eob]: priced with neighbour context.
    for ((&rc, ir), p) in scan[1..eu + 1]
        .iter()
        .zip(irate[1..eu + 1].iter_mut())
        .zip(pre[2..eu + 2].iter_mut())
    {
        let (ctx, bc) = interior_ctx(&levels, rc);
        let r = interior_rate(ctx, bc, cf[rc].unsigned_abs());
        *ir = r;
        acc = (acc + lambda * r) + dist(rc, cf[rc]);
        *p = acc;
    }
    // Trailing positions (eob, n): coded as zeros, distortion only.
    for (&rc, p) in scan[eu + 1..n].iter().zip(pre[eu + 2..n + 1].iter_mut()) {
        acc += dist(rc, 0);
        *p = acc;
    }
    suf0.resize(n + 1, 0.0);
    suf0[n] = 0.0; // suffix seed (read as suf0[n]; not written by the loop below)
    let mut sacc = 0.0f64;
    for (&rc, s) in scan[1..n].iter().rev().zip(suf0[1..n].iter_mut().rev()) {
        sacc += dist(rc, 0);
        *s = sacc;
    }
    // DC contribution (rate + distortion), constant across EOB choices ≥ 1.
    let dc_rc = scan[0];
    let dc_m = cf[dc_rc].unsigned_abs();
    let dc_cost = if dc_m == 0 {
        lambda * cdf_cost(&base_tok[0], 0)
    } else {
        let bc = dc_brc(&levels);
        let tok = dc_m.min(3);
        let mut b = cdf_cost(&base_tok[0], tok as usize);
        if tok == 3 {
            b += hi_cost(dc_m, bc);
        }
        b += cdf_cost(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize);
        lambda * b
    } + dist(dc_rc, cf[dc_rc]);

    let mut best_e: i32 = -1;
    let mut best_cost = f64::INFINITY;

    assert!(scan.len() >= n, "scan must be indexed up to n-1");
    assert!(irate.len() >= n, "irate must be indexed up to n");
    assert!(pre.len() > n, "pre must be indexed up to n+1");
    assert!(suf0.len() > n, "suf0 must be indexed up to n+1");

    for e in 1..n {
        let rc = scan[e];
        if cf[rc] == 0 {
            continue; // EOB must land on a nonzero
        }
        // pre[e] prices position e as interior; replace with eob_coeff cost.
        // A nonzero at `e` implies `e <= eob`, so `irate[e]` was filled above
        // (identical value to interior_rate here, just cached).
        let interior_e = lambda * irate[e];
        let c = dc_cost
            + (pre[e + 1] - interior_e)
            + lambda * (eob_pt_cost(e) + eob_coeff_cost(e, cf[rc].unsigned_abs()))
            + suf0[e + 1];
        if c < best_cost {
            best_cost = c;
            best_e = e as i32;
        }
    }
    // EOB at DC (only DC nonzero) and the all-zero (txb_skip) alternative.
    if dc_m != 0 {
        let ctx_e = 1usize; // e == 0
        let tok = dc_m.min(3);
        let mut c0 = cdf_cost(eob_bin_cdf, 0) + cdf_cost(&eob_base[ctx_e], tok as usize - 1);
        if tok == 3 {
            c0 += hi_cost(dc_m, dc_brc(&levels));
        }
        c0 += cdf_cost(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize);
        let total0 = lambda * c0 + dist(dc_rc, cf[dc_rc]) + suf0[1];
        if total0 < best_cost {
            best_cost = total0;
            best_e = 0;
        }
    }
    let skip_cost = suf0[1] + dist(dc_rc, 0) + lambda * 1.0;
    if best_e < 0 || skip_cost < best_cost {
        for &rc in scan.iter() {
            cf[rc] = 0;
        }
    } else {
        for i in (best_e as usize + 1)..n {
            cf[scan[i]] = 0;
        }
    }
    SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        b.0 = std::mem::take(&mut levels);
        b.1 = std::mem::take(&mut pre);
        b.2 = std::mem::take(&mut suf0);
        b.3 = std::mem::take(&mut irate);
    });
}

/// Rate-distortion optimized quantization (trellis / RDOQ). Given baseline
/// rounded levels `cf` and the matching pre-round real targets `tf` (the value
/// `cf` was rounded from, = forward coefficient * SCALE / quant-step), this
/// lowers coefficient magnitudes and trims the end-of-block wherever doing so
/// reduces `D + lambda*R`, where `D` is the dequantized-domain squared error
/// (orthonormal transform, so proportional to pixel SSE) and `R` the estimated
/// coded bits. Only the written levels change, so the decoder — which simply
/// inverse-transforms whatever levels are coded — stays bit-exact; the caller
/// reconstructs from this same `cf`.
pub(crate) fn trellis_optimize(
    cf: &mut [i32],
    tf: &[f64],
    dc_q: f64,
    ac_q: f64,
    scan: &[usize],
    lambda0: f64,
) {
    if lambda0 <= 0.0 {
        return; // trellis disabled
    }
    let n = scan.len();
    let lambda = lambda0 * ac_q * ac_q;
    let dqf = |rc: usize| if rc == 0 { dc_q } else { ac_q };
    let d = |rc: usize, lev: i32| {
        let dq = dqf(rc);
        let t = tf[rc].abs();
        dq * dq * (t - lev.unsigned_abs() as f64).powi(2)
    };

    let mut eob_idx: i32 = -1;
    for (i, &x) in scan[..n].iter().enumerate() {
        if cf[x] != 0 {
            eob_idx = i as i32;
        }
    }
    if eob_idx < 0 {
        return; // already all-zero
    }

    // Step A: per-coefficient round-down (toward zero) by local R-D.
    for &rc in scan[..=eob_idx as usize].iter() {
        let l = cf[rc].unsigned_abs();
        if l == 0 {
            continue;
        }
        let cost_l = d(rc, l as i32) + lambda * coef_rate_bits(l);
        let cost_dn = d(rc, (l - 1) as i32) + lambda * coef_rate_bits(l - 1);
        if cost_dn < cost_l {
            let s = if cf[rc] < 0 { -1 } else { 1 };
            cf[rc] = s * (l as i32 - 1);
        }
    }

    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<f64>, Vec<f64>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    let (mut suf0, mut pre) = SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        (std::mem::take(&mut b.0), std::mem::take(&mut b.1))
    });

    suf0.resize(n + 1, 0.0); // distortion of zeroing coeffs from i..n
    suf0[n] = 0.0; // cumulative seed (read as suf0[n]; not written by the loop)
    assert!(suf0.len() > n, "suf0 must be indexed up to n");
    for (i, &s) in (0..n).rev().zip(scan[..n].iter().rev()) {
        suf0[i] = suf0[i + 1] + d(s, 0);
    }
    pre.resize(n + 1, 0.0); // interior cost of coeffs strictly before i
    assert!(pre.len() > n, "pre must be indexed up to n");

    pre[0] = 0.0; // empty-prefix seed
    for (i, &rc) in scan[..n].iter().enumerate() {
        pre[i + 1] = pre[i] + d(rc, cf[rc]) + lambda * coef_rate_bits(cf[rc].unsigned_abs());
    }
    let eob_sig = |e: usize| -> f64 {
        let bin = if e < 2 {
            e
        } else {
            (32 - (e as u32).leading_zeros()) as usize
        };
        let extra = if bin > 1 { bin - 2 } else { 0 };
        (bin as f64) * 0.9 + extra as f64 + 2.0 // eob_pt + extra bits + eob_base token
    };

    let mut best_e: i32 = -1;
    let mut best_cost = f64::INFINITY;
    for (e, ((&rc, &pre), &suf0)) in scan[..n]
        .iter()
        .zip(pre.iter())
        .zip(suf0[1..].iter())
        .enumerate()
    {
        if cf[rc] == 0 {
            continue; // the EOB must land on a nonzero
        }
        let c = pre + d(rc, cf[rc]) + lambda * eob_sig(e) + suf0;
        if c < best_cost {
            best_cost = c;
            best_e = e as i32;
        }
    }
    let skip_cost = suf0[0] + lambda * 1.0; // zero everything + the txb_skip flag
    if best_e < 0 || skip_cost < best_cost {
        for &rc in scan.iter() {
            cf[rc] = 0;
        }
    } else {
        for &x in scan[(best_e as usize + 1)..n].iter() {
            cf[x] = 0;
        }
    }

    SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        b.0 = std::mem::take(&mut suf0);
        b.1 = std::mem::take(&mut pre);
    });
}
