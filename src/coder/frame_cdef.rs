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

// Frame-level CDEF search, per-unit decisions and application.

/// CDEF damping derived from the base quantizer (spec range 3..=6); higher q ->
/// stronger ringing -> a touch more damping. Kept simple and deterministic.
fn cdef_damping(base_q_idx: u8) -> u8 {
    3 + ((base_q_idx as u32) / 64).min(3) as u8
}

/// Frame CDEF decision: header params plus, when `params.bits == 1`, the
/// per-64x64-unit on/off grid (frame unit raster order; 1 = index 1 = filtered).
/// Units whose 8x8 blocks are all skip carry no `cdef_idx` and are never
/// filtered; their grid entry stays 0.
pub(crate) struct CdefFrameDecision {
    pub(crate) params: crate::obu::CdefParams,
    pub(crate) grid: Vec<u8>,
    pub(crate) unit_cols: usize,
}

#[allow(clippy::too_many_arguments)]
fn frame_cdef(
    dsp: &crate::cdef::CdefDispatch,
    rd: &crate::rd_sse::RdDispatch,
    recon: &mut [Vec<u16>; 3],
    src: &[Vec<u16>; 3],
    skip8: &[bool],
    sb8w: usize,
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    disp_w: usize,
    disp_h: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    base_q_idx: u8,
    bd: u8,
    speed: Speed,
    pool: &Pool,
) -> Option<CdefFrameDecision> {
    use crate::cdef;
    let signaled_damping = cdef_damping(base_q_idx) as i32;
    let damping = signaled_damping + (bd as i32 - 8);

    // The decoder reconstructs and reads REAL pixels across the whole
    // mi-aligned (coded) area — CDEF taps see the coded padding columns/rows as
    // ordinary neighbors, and only samples beyond the mi extent are
    // CDEF_VERY_LARGE (handled by the stride bounds in `sample`). The padding is
    // never deblocked (see `filter_plane`'s visible clip), so the snapshot below
    // matches the decoder's pre-CDEF frame everywhere. Distortion, however, is
    // measured over the VISIBLE window only — invisible pixels must not drive
    // the decision.
    let (cw_vis, ch_vis) = (disp_w.div_ceil(1 << sub_x), disp_h.div_ceil(1 << sub_y));
    let snap_y = recon[0].clone();

    let nbx = w8.div_ceil(8);
    let nby = h8.div_ceil(8);
    let mut ldirs = vec![0usize; nbx * nby];
    let mut lvars = vec![0i32; nbx * nby];
    {
        // Directions read the decoder's frame buffer, which contains REAL
        // reconstructed pixels across the whole mi-aligned (coded) area — the
        // padding columns/rows are coded and reconstructed, only CDEF taps
        // beyond them are masked. So the direction search uses the unmarked
        // recon.
        let luma = &recon[0];
        #[allow(clippy::type_complexity)]
        let items: Vec<(usize, (&mut [usize], &mut [i32]))> = ldirs
            .chunks_mut(nbx)
            .zip(lvars.chunks_mut(nbx))
            .enumerate()
            .collect();
        pool.for_each(pool.width(), items, |(by, (drow, vrow))| {
            for bx in 0..nbx {
                if bx * 8 < w8 && by * 8 < h8 {
                    let (d, v) = dsp.direction(luma, w8, bx * 8, by * 8, bd);
                    drow[bx] = d;
                    vrow[bx] = v;
                }
            }
        });
    }

    // Per-8x8 luma skip map: CDEF is NOT applied to skip blocks (the decoder
    // leaves them untouched), so the encoder must mirror that exactly. `skip8`
    // is already a per-8x8 frame map.
    let lskip: Vec<bool> = (0..nbx * nby)
        .map(|i| {
            let (bx, by) = (i % nbx, i / nbx);
            skip8.get(by * sb8w + bx).copied().unwrap_or(true)
        })
        .collect();

    // 64x64 CDEF unit grid; a unit is "signaled" (carries a cdef_idx literal)
    // iff it contains at least one non-skip 8x8.
    let uc = w8.div_ceil(64);
    let ur = h8.div_ceil(64);
    let n_units = uc * ur;
    let mut signaled = vec![false; n_units];
    for by in 0..nby {
        for bx in 0..nbx {
            if !lskip[by * nbx + bx] {
                signaled[(by / 8) * uc + bx / 8] = true;
            }
        }
    }
    let n_sig = signaled.iter().filter(|&&s| s).count();
    if n_sig == 0 {
        return None;
    }

    // Decision knobs:
    //   gate_thresh  directional-variance gate threshold (0 disables)
    //   perceptual   plain SSE vs perceptual cdef_dist
    //   margin per mille distortion margin a unit must clear to filter
    let tun = crate::tuning::get();
    let gate_thresh = tun.cdef_gate as i64;
    // Unification V6: one metric (normalized wSSE, luma and chroma) and the
    // frame `lambda_d`, or the legacy libaom `cdef_dist_8x8` + plain chroma.
    let metric = if tun.cdef_wsse {
        CdefMetric::Wsse
    } else {
        CdefMetric::CdefDist
    };
    let margin = tun.cdef_margin as i64;
    let chroma_w = tun.cdef_chroma_w;

    // Directional-variance gate for the GLOBAL (filter-everything) option only:
    // that mode has no per-unit off-switch, so it is offered solely when most
    // units show a dominant edge direction (where ringing lives — CDEF's
    // target). Isotropic fine detail (fractals) has no dominant direction;
    // there the variance-masked metric collapses to plain SSE and rewards
    // smoothing SSIMULACRA2 punishes — measured on the corpus as a −0.2 SS2
    // regression whenever global filtering engages on such content. Per-unit
    // decisions are NOT gated: the distortion margin protects them.
    let mut dv_sum = vec![0i64; n_units];
    let mut dv_cnt = vec![0i64; n_units];
    for by in 0..nby {
        for bx in 0..nbx {
            if !lskip[by * nbx + bx] {
                let u = (by / 8) * uc + bx / 8;
                dv_sum[u] += lvars[by * nbx + bx] as i64;
                dv_cnt[u] += 1;
            }
        }
    }
    let trusted: Vec<bool> = (0..n_units)
        .map(|u| gate_thresh == 0 || dv_sum[u] / dv_cnt[u].max(1) >= gate_thresh)
        .collect();
    let n_trusted = trusted.iter().filter(|&&t| t).count();

    // ---- Candidate strengths ------------------------------------------------
    // Slow: two-stage full search — all 15 primary-only strengths, then the
    // top-K primaries crossed with every secondary (libaom pickcdef coverage).
    // Fast: the small legacy set. Index 0 is always (0,0) = unfiltered.
    // Every candidate strength pair is filtered in one pass over the frame;
    // the staged search below only reads these tables.
    let luma_all = cdef_luma_unit_dists_all(
        dsp, rd, &snap_y, &src[0], w8, h8, disp_w, disp_h, &ldirs, &lvars, &lskip, nbx, uc,
        damping, bd, metric, pool,
    );
    let luma_tab = |pri: i32, sec: i32| -> Vec<i64> {
        let c = cdef::cand_index(pri, sec);
        luma_all.iter().map(|u| u[c]).collect()
    };
    // A unit only counts as gained when filtering clears the margin: a small
    // guaranteed improvement, standing in for the SSE/SSIMULACRA2 mismatch.
    let clears = |off: i64, on: i64| -> i64 {
        let thr = off - off.saturating_mul(margin) / 1000;
        if on < thr { off - on } else { 0 }
    };

    let slow = speed.at_least_slow();
    let mut cands: Vec<(i32, i32)> = vec![(0, 0)];
    let mut ly: Vec<Vec<i64>> = vec![luma_tab(0, 0)];
    let luma_off: Vec<i64> = ly[0].clone();
    if slow {
        // Stage A: primary-only sweep. Strengths above 4 are deliberately NOT
        // searched: the corpus shows the decision metric over-rates strong
        // smoothing (SSIMULACRA2 regressions up to −0.4 when pri 5..15 entries
        // were offered), matching the legacy candidate cap.
        let pri_list: Vec<i32> = vec![1, 2, 4];
        let tabs: Vec<Vec<i64>> = pri_list.iter().map(|&p| luma_tab(p, 0)).collect();
        let mut ranked: Vec<(i64, usize)> = tabs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    (0..n_units).map(|u| clears(luma_off[u], t[u])).sum::<i64>(),
                    i,
                )
            })
            .collect();
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push((pri_list[i], 0));
            ly.push(t);
        }
        // Stage B: top-K primaries (plus pri 0) crossed with the secondaries.
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut stage_b: Vec<(i32, i32)> = Vec::new();
        for sec in [1, 2] {
            stage_b.push((0, sec));
            for &(gain, i) in ranked.iter().take(4) {
                if gain > 0 {
                    stage_b.push((pri_list[i], sec));
                }
            }
        }
        let tabs: Vec<Vec<i64>> = stage_b.iter().map(|&(p, s)| luma_tab(p, s)).collect();
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push(stage_b[i]);
            ly.push(t);
        }
    } else {
        let list: Vec<(i32, i32)> = cdef::PRI_CANDIDATES
            .iter()
            .flat_map(|&pri| cdef::SEC_CANDIDATES.iter().map(move |&sec| (pri, sec)))
            .filter(|&(pri, sec)| !(pri == 0 && sec == 0))
            .collect();
        let tabs: Vec<Vec<i64>> = list.iter().map(|&(p, s)| luma_tab(p, s)).collect();
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push(list[i]);
            ly.push(t);
        }
    }

    // Chroma reuses the LUMA direction (remapped via uv_dir) and damping-1, and
    // is filtered at chroma sub-block granularity tied to the covering luma 8x8:
    // 8x8 for 4:4:4, 4x8 for 4:2:2, 4x4 for 4:2:0 — exactly as the decoders do
    // (dav1d cdef.fb[I444 - layout]). No per-block variance scaling for chroma.
    let uv_dir: [usize; 8] = if sub_x == 1 && sub_y == 0 {
        [7, 0, 2, 4, 5, 6, 6, 6] // 4:2:2
    } else {
        [0, 1, 2, 3, 4, 5, 6, 7] // 4:2:0 / 4:4:4 (identity)
    };
    let chroma_damping = damping - 1;
    let snap_uv: [Vec<u16>; 2] = if mono {
        [Vec::new(), Vec::new()]
    } else {
        [recon[1].clone(), recon[2].clone()]
    };
    // Chroma candidate space: (0,0) plus the same list as luma (its tables are
    // cheap relative to luma). Monochrome keeps only (0,0).
    let (c_cands, lc): (Vec<(i32, i32)>, Vec<Vec<i64>>) = if mono {
        (vec![(0, 0)], vec![vec![0; n_units]])
    } else {
        let plane_all = |pl: usize| {
            cdef_chroma_unit_sse_all(
                dsp,
                rd,
                &snap_uv[pl],
                &src[1 + pl],
                cw8,
                cw_vis,
                ch_vis,
                &ldirs,
                &uv_dir,
                &lskip,
                nbx,
                nby,
                uc,
                sub_x,
                sub_y,
                chroma_damping,
                bd,
                metric,
                pool,
            )
        };
        let (u_all, v_all) = (plane_all(0), plane_all(1));
        let tabs: Vec<Vec<i64>> = cands
            .iter()
            .map(|&(pri, sec)| {
                let c = cdef::cand_index(pri, sec);
                (0..n_units)
                    .map(|i| {
                        let d = u_all[i][c] + v_all[i][c];
                        if chroma_w == 1.0 {
                            d
                        } else {
                            (d as f64 * chroma_w as f64) as i64
                        }
                    })
                    .collect()
            })
            .collect();
        (cands.clone(), tabs)
    };
    let chroma_off: Vec<i64> = lc[0].clone();

    let d_off: Vec<i64> = (0..n_units).map(|u| luma_off[u] + chroma_off[u]).collect();
    // Margin threshold each unit must clear to filter (see `clears`).
    let thr: Vec<i64> = d_off
        .iter()
        .map(|&o| o - o.saturating_mul(margin) / 1000)
        .collect();

    // Greedy set construction over all strength PAIRS, exactly libaom's
    // `joint_strength_search_dual`: each added entry minimizes the frame total
    // when every unit picks its best entry (with the off-fallback margin rule
    // once the set contains the all-zero entry). The greedy is deterministic
    // and incremental, so one run to 8 entries yields the candidate sets for
    // every cdef_bits at its prefixes of size 1, 2, 4, 8.
    let nc = c_cands.len();
    let n_pairs = cands.len() * nc;
    let d_pair = |p: usize, u: usize| ly[p / nc][u] + lc[p % nc][u];
    let want = pool.width().min(n_pairs);

    // Global (cdef_bits = 0) candidate: the raw frame-total minimizer — every
    // signaled unit filters with it, no off-fallback.
    let raw_totals: Vec<i64> = pool.map_indexed(want, n_pairs, |p| {
        (0..n_units)
            .map(|u| if signaled[u] { d_pair(p, u) } else { d_off[u] })
            .sum()
    });
    let mut global_entry = 0usize;
    let mut prefix_tot = [i64::MAX; 4]; // totals at set sizes 1, 2, 4, 8
    for (p, &t) in raw_totals.iter().enumerate() {
        if t < prefix_tot[0] {
            prefix_tot[0] = t;
            global_entry = p;
        }
    }

    // Per-unit sets are seeded with the all-zero entry: a unit must always be
    // able to signal OFF (the Phase-1 {off, strength} table generalized), and
    // the margin rule sends units that don't clearly gain back to it.
    let mut cur_min: Vec<i64> = d_off.clone();
    let mut entries: Vec<usize> = vec![0];
    while entries.len() < 8 {
        let scores: Vec<i64> = pool.map_indexed(want, n_pairs, |p| {
            let mut tot = 0i64;
            for u in 0..n_units {
                if !signaled[u] {
                    tot += d_off[u];
                    continue;
                }
                let b = cur_min[u].min(d_pair(p, u));
                tot += if b >= thr[u] { d_off[u] } else { b };
            }
            tot
        });
        let mut best_p = 0usize;
        let mut best_tot = i64::MAX;
        for (p, &t) in scores.iter().enumerate() {
            if t < best_tot {
                best_tot = t;
                best_p = p;
            }
        }
        entries.push(best_p);
        for (u, m) in cur_min.iter_mut().enumerate() {
            *m = (*m).min(d_pair(best_p, u));
        }
        let n = entries.len();
        if n.is_power_of_two() && n <= 8 {
            prefix_tot[n.trailing_zeros() as usize] = best_tot;
        }
    }

    // A unit does not take its raw metric-minimizing entry: with several
    // entries available, per-unit metric errors accumulate (measured as SS2
    // regressions on detail-rich content). Instead each unit takes the MILDEST
    // entry that still clears the margin against off — least filtering among
    // the options the metric is confident in. Units clearing nothing take the
    // all-zero entry when present (else the raw minimum). The R-D totals are
    // computed from this realized assignment, not the greedy's raw minimum.
    let strength_mag = |p: usize| -> i32 {
        let (yp, ys) = cands[p / nc];
        let (up, us) = c_cands[p % nc];
        yp + ys + up + us
    };
    let assign = |set: &[usize]| -> (Vec<u8>, i64) {
        let z_in_set = set.iter().position(|&p| p == 0);
        let mut total = 0i64;
        let grid: Vec<u8> = (0..n_units)
            .map(|u| {
                if !signaled[u] {
                    total += d_off[u];
                    return 0;
                }
                let mut pick: Option<(i32, i64, usize)> = None; // (mag, d, e)
                let mut raw_best = (i64::MAX, 0usize);
                for (e, &p) in set.iter().enumerate() {
                    let d = d_pair(p, u);
                    if d < raw_best.0 {
                        raw_best = (d, e);
                    }
                    if d < thr[u] {
                        let mag = strength_mag(p);
                        if pick.is_none_or(|(m, pd, _)| mag < m || (mag == m && d < pd)) {
                            pick = Some((mag, d, e));
                        }
                    }
                }
                match (pick, z_in_set) {
                    (Some((_, d, e)), _) => {
                        total += d;
                        e as u8
                    }
                    (None, Some(z)) => {
                        total += d_off[u];
                        z as u8
                    }
                    (None, None) => {
                        total += raw_best.0;
                        raw_best.1 as u8
                    }
                }
            })
            .collect();
        (grid, total)
    };

    // Rate deltas against the always-written noop CDEF header: each strength
    // entry past the first costs 12 header bits (6 mono), and every signaled
    // unit carries a raw `cdef_idx` literal of `cdef_bits` bits.
    let lambda = if tun.cdef_lambda_d {
        frame_lambda_d(base_q_idx, bd, sub_x, sub_y)
    } else {
        mode_lambda_q(dc_q(base_q_idx, bd) as f32)
    };
    let total_off: i64 = d_off.iter().sum();
    let cost_off = total_off as f32;
    let per_entry_bits = if mono { 6.0f32 } else { 12.0f32 };
    let mut best_bits: Option<u8> = None;
    let mut best_cost = cost_off;
    let max_bits = 3u8;
    for b in 0..=max_bits {
        let nb = 1usize << b;
        let total = if b == 0 {
            // Global: every signaled unit filters with `global_entry`.
            prefix_tot[0]
        } else {
            assign(&entries[..nb]).1
        };
        if total == i64::MAX {
            continue;
        }
        // Global filtering (cdef_bits = 0) has no per-unit off-switch, so it is
        // offered only when most units show a dominant edge direction (the
        // regime the metric is trusted in) and the whole frame clears the
        // margin. Per-unit modes are protected by the margin rule instead.
        if b == 0
            && (global_entry == 0
                || n_trusted * 4 < n_sig * 3
                || total >= total_off - total_off.saturating_mul(margin) / 1000)
        {
            continue;
        }
        let cost =
            total as f32 + lambda * (per_entry_bits * (nb - 1) as f32 + b as f32 * n_sig as f32);
        if cost < best_cost {
            best_cost = cost;
            best_bits = Some(b);
        }
    }

    let bits = best_bits?;
    let nb = 1usize << bits;
    let global_set = [global_entry];
    let set: &[usize] = if bits == 0 {
        &global_set
    } else {
        &entries[..nb]
    };
    let grid: Vec<u8> = if bits == 0 {
        vec![0; n_units]
    } else {
        assign(set).0
    };

    let strengths: Vec<(u8, u8, u8, u8)> = set
        .iter()
        .map(|&p| {
            let (yp, ys) = cands[p / nc];
            let (up, us) = c_cands[p % nc];
            (yp as u8, ys as u8, up as u8, us as u8)
        })
        .collect();
    // Per-unit effective strengths for the apply pass (constant for bits == 0;
    // all-skip units never filter regardless).
    let unit_y: Vec<(i32, i32)> = (0..n_units)
        .map(|u| {
            let (yp, ys, _, _) = strengths[grid[u] as usize];
            (yp as i32, ys as i32)
        })
        .collect();
    let unit_uv: Vec<(i32, i32)> = (0..n_units)
        .map(|u| {
            let (_, _, up, us) = strengths[grid[u] as usize];
            (up as i32, us as i32)
        })
        .collect();

    let decision = CdefFrameDecision {
        params: crate::obu::CdefParams {
            bits,
            damping: signaled_damping as u8,
            strengths,
        },
        grid: if bits > 0 { grid } else { Vec::new() },
        unit_cols: uc,
    };

    // Apply the decision to the reconstruction, reading every tap from the
    // pre-CDEF snapshot exactly as the decoder does.
    if unit_y.iter().any(|&(p, s)| p != 0 || s != 0) {
        apply_cdef_plane(
            &mut recon[0],
            &snap_y,
            w8,
            h8,
            &ldirs,
            &lvars,
            &lskip,
            nbx,
            uc,
            &unit_y,
            damping,
            bd,
            pool,
        );
    }
    if !mono && unit_uv.iter().any(|&(p, s)| p != 0 || s != 0) {
        for plane in 1..3 {
            apply_cdef_chroma(
                &mut recon[plane],
                &snap_uv[plane - 1],
                cw8,
                ch8,
                &ldirs,
                &uv_dir,
                skip8,
                sb8w,
                nbx,
                nby,
                sub_x,
                sub_y,
                uc,
                &unit_uv,
                chroma_damping,
                bd,
                pool,
            );
        }
    }

    Some(decision)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cdef_block_dist_vis(
    src: &[u16],
    dst: &[u16],
    stride: usize,
    vis_w: usize,
    vis_h: usize,
    x: usize,
    y: usize,
    coeff_shift: u32,
    perceptual: bool,
) -> i64 {
    if perceptual && x + 8 <= vis_w && y + 8 <= vis_h {
        let rows = dst.len() / stride;
        return crate::cdef::cdef_dist_8x8(src, dst, stride, rows, x, y, coeff_shift);
    }
    let mut s = 0i64;
    for yy in y..(y + 8).min(vis_h) {
        for xx in x..(x + 8).min(vis_w) {
            let d = dst[yy * stride + xx] as i64 - src[yy * stride + xx] as i64;
            s += d * d;
        }
    }
    s
}

/// Per-64x64-unit luma distortion for every [`crate::cdef::cand_index`] strength
/// pair in one pass: entry `[u][c]` equals `cdef_luma_unit_dists(.., pri,
/// sec, ..)[u]` for candidate `c`. Unit rows are independent, so they run in
/// parallel.
#[allow(clippy::too_many_arguments)]
/// Per-8x8 CDEF decision distortion (see `Tuning::cdef_wsse`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CdefMetric {
    /// Plain SSE on every plane (tests).
    #[cfg(test)]
    Sse,
    /// libaom variance-masked `cdef_dist_8x8` on luma, plain SSE on chroma.
    CdefDist,
    /// Normalized weighted SSE on every plane (partial edge blocks: plain SSE).
    Wsse,
}

#[allow(clippy::too_many_arguments)]
fn cdef_luma_unit_dists_all(
    dsp: &crate::cdef::CdefDispatch,
    rd: &crate::rd_sse::RdDispatch,
    recon: &[u16],
    src: &[u16],
    w: usize,
    h: usize,
    disp_w: usize,
    disp_h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    damping: i32,
    bd: u8,
    metric: CdefMetric,
    pool: &Pool,
) -> Vec<[i64; crate::cdef::N_CAND]> {
    let coeff_shift = (bd - 8) as u32;
    let perceptual = metric == CdefMetric::CdefDist;
    let ur = h.div_ceil(64);
    let rows = pool.map_indexed(pool.width().min(ur), ur, |uy| {
        let mut out = vec![[0i64; crate::cdef::N_CAND]; uc];
        let mut cand = [[0i32; 64]; crate::cdef::N_CAND];
        for y in (uy * 64..(uy * 64 + 64).min(h)).step_by(8) {
            if y >= disp_h {
                break;
            }
            for x in (0..w).step_by(8) {
                let bi = (y / 8) * nbx + x / 8;
                if skip.get(bi).copied().unwrap_or(true) || x >= disp_w {
                    continue;
                }
                dsp.block_candidates(
                    recon,
                    w,
                    x,
                    y,
                    8,
                    8,
                    dirs[bi],
                    Some(vars[bi]),
                    damping,
                    bd,
                    &mut cand,
                );
                let whole = x + 8 <= disp_w && y + 8 <= disp_h;
                let full = perceptual && whole;
                let (vw, vh) = ((disp_w - x).min(8), (disp_h - y).min(8));
                let acc = &mut out[x / 64];
                if metric == CdefMetric::Wsse && whole {
                    let mut err = [0i32; 64];
                    for (a, cb) in acc.iter_mut().zip(cand.iter()) {
                        for i in 0..8 {
                            let srow = &src[(y + i) * w + x..][..8];
                            for (e, (&d, &sv)) in err[i * 8..i * 8 + 8]
                                .iter_mut()
                                .zip(cb[i * 8..i * 8 + 8].iter().zip(srow))
                            {
                                *e = d - i32::from(sv);
                            }
                        }
                        *a += rd.wsse_err(&err, 8, 8);
                    }
                    continue;
                }
                let (mut ss, mut ss2) = (0i64, 0i64);
                if full {
                    for i in 0..8 {
                        for &s in &src[(y + i) * w + x..][..8] {
                            ss += s as i64;
                            ss2 += s as i64 * s as i64;
                        }
                    }
                }
                if full {
                    let mut sums = [[0i64; 3]; crate::cdef::N_CAND];
                    dsp.cand_sums(src, w, x, y, &cand, &mut sums);
                    for (a, s) in acc.iter_mut().zip(sums.iter()) {
                        *a += crate::cdef::cdef_dist_from_sums(
                            ss,
                            s[0],
                            ss2,
                            s[1],
                            s[2],
                            coeff_shift,
                        );
                    }
                    continue;
                }
                for (a, cb) in acc.iter_mut().zip(cand.iter()) {
                    *a += {
                        let mut s = 0i64;
                        for i in 0..vh {
                            let srow = &src[(y + i) * w + x..][..vw];
                            for (&sv, &d) in srow.iter().zip(&cb[i * 8..i * 8 + vw]) {
                                let e = d as i64 - sv as i64;
                                s += e * e;
                            }
                        }
                        s
                    };
                }
            }
        }
        out
    });
    rows.into_iter().flatten().collect()
}

/// Per-64x64-unit chroma SSE for every [`crate::cdef::cand_index`] strength pair of
/// one chroma plane: entry `[u][c]` equals `cdef_chroma_unit_sse(.., pri,
/// sec, ..)[u]` for candidate `c`.
#[allow(clippy::too_many_arguments)]
fn cdef_chroma_unit_sse_all(
    dsp: &crate::cdef::CdefDispatch,
    rd: &crate::rd_sse::RdDispatch,
    recon: &[u16],
    src: &[u16],
    cw: usize,
    cw_vis: usize,
    ch_vis: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    lskip: &[bool],
    nbx: usize,
    nby: usize,
    uc: usize,
    sub_x: usize,
    sub_y: usize,
    damping: i32,
    bd: u8,
    metric: CdefMetric,
    pool: &Pool,
) -> Vec<[i64; crate::cdef::N_CAND]> {
    let (cbw, cbh) = (8 >> sub_x, 8 >> sub_y);
    let ur = nby.div_ceil(8);
    let rows = pool.map_indexed(pool.width().min(ur), ur, |uy| {
        let mut out = vec![[0i64; crate::cdef::N_CAND]; uc];
        let mut cand = [[0i32; 64]; crate::cdef::N_CAND];
        for lby in uy * 8..(uy * 8 + 8).min(nby) {
            let cy = (lby * 8) >> sub_y;
            if cy >= ch_vis {
                break;
            }
            for lbx in 0..nbx {
                let bi = lby * nbx + lbx;
                let cx = (lbx * 8) >> sub_x;
                if lskip.get(bi).copied().unwrap_or(true) || cx >= cw_vis {
                    continue;
                }
                let dir = uv_dir[ldirs.get(bi).copied().unwrap_or(0)];
                dsp.block_candidates(
                    recon, cw, cx, cy, cbw, cbh, dir, None, damping, bd, &mut cand,
                );
                let (vw, vh) = ((cw_vis - cx).min(cbw), (ch_vis - cy).min(cbh));
                let acc = &mut out[lbx / 8];
                if metric == CdefMetric::Wsse && vw == cbw && vh == cbh {
                    let mut err = [0i32; 64];
                    for (a, cb) in acc.iter_mut().zip(cand.iter()) {
                        for i in 0..cbh {
                            let srow = &src[(cy + i) * cw + cx..][..cbw];
                            for (e, (&d, &sv)) in err[i * cbw..i * cbw + cbw]
                                .iter_mut()
                                .zip(cb[i * cbw..i * cbw + cbw].iter().zip(srow))
                            {
                                *e = d - i32::from(sv);
                            }
                        }
                        *a += rd.wsse_err(&err, cbw, cbh);
                    }
                    continue;
                }
                for (a, cb) in acc.iter_mut().zip(cand.iter()) {
                    let mut s = 0i64;
                    for i in 0..vh {
                        let srow = &src[(cy + i) * cw + cx..][..vw];
                        for (&sv, &d) in srow.iter().zip(&cb[i * cbw..i * cbw + vw]) {
                            let e = d as i64 - sv as i64;
                            s += e * e;
                        }
                    }
                    *a += s;
                }
            }
        }
        out
    });
    rows.into_iter().flatten().collect()
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cdef_luma_unit_dists(
    recon: &[u16],
    src: &[u16],
    w: usize,
    h: usize,
    disp_w: usize,
    disp_h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    n_units: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
    perceptual: bool,
) -> Vec<i64> {
    use crate::cdef;
    let coeff_shift = (bd - 8) as u32;
    let filtering = pri != 0 || sec != 0;
    let mut tmp = if filtering {
        // Distortion consumes each filtered block before the next row band.
        vec![0u16; w * 8]
    } else {
        Vec::new()
    };
    let mut out = vec![0i64; n_units];
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let bi = (y / 8) * nbx + x / 8;
            if skip.get(bi).copied().unwrap_or(true) {
                continue;
            }
            if x >= disp_w || y >= disp_h {
                continue; // fully outside the visible frame
            }
            let u = (y / 64) * uc + x / 64;
            let dist = if filtering {
                // adjust_pri must be applied to the bit-depth-shifted strength
                // (matches the decoders' adjust_strength, which scales the
                // already-shifted level); scaling then shifting does not commute
                // because of the `+8 >> 4` rounding.
                let apri = cdef::adjust_pri(pri << (bd - 8), vars[bi]);
                cdef::cdef_filter_block(
                    &mut tmp,
                    y,
                    recon,
                    w,
                    x,
                    y,
                    8,
                    8,
                    apri,
                    sec << (bd - 8),
                    // Decoders pass dir 0 when the SIGNALED pri strength is 0
                    // (`pri_strength ? dir : 0`) — sec-only filtering then runs
                    // on direction 0, not the block's estimated direction.
                    if pri == 0 { 0 } else { dirs[bi] },
                    damping,
                    bd,
                );
                cdef_block_dist_vis(
                    &src[y * w..],
                    &tmp,
                    w,
                    disp_w,
                    disp_h.saturating_sub(y),
                    x,
                    0,
                    coeff_shift,
                    perceptual,
                )
            } else {
                cdef_block_dist_vis(src, recon, w, disp_w, disp_h, x, y, coeff_shift, perceptual)
            };
            out[u] += dist;
        }
    }
    out
}

/// Per-64x64-unit chroma SSE for one candidate strength (`0,0` = baseline),
/// accumulated into the covering LUMA unit (the cdef_idx is shared with luma).
/// One chroma sub-block per non-skip luma 8x8: 8x8 (4:4:4), 4x8 (4:2:2),
/// 4x4 (4:2:0). Plain SSE — chroma sub-blocks are too small for the 8x8
/// variance calibration of the perceptual metric (libaom also uses MSE here).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cdef_chroma_unit_sse(
    recon: &[u16],
    src: &[u16],
    cw: usize,
    _ch: usize,
    cw_vis: usize,
    ch_vis: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    lskip: &[bool],
    nbx: usize,
    nby: usize,
    uc: usize,
    n_units: usize,
    sub_x: usize,
    sub_y: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) -> Vec<i64> {
    use crate::cdef;
    let cbw = 8 >> sub_x;
    let cbh = 8 >> sub_y;
    let filtering = pri != 0 || sec != 0;
    let mut tmp = if filtering {
        vec![0u16; cw * cbh]
    } else {
        Vec::new()
    };
    let mut out = vec![0i64; n_units];
    for lby in 0..nby {
        for lbx in 0..nbx {
            if lskip.get(lby * nbx + lbx).copied().unwrap_or(true) {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            let cy = (lby * 8) >> sub_y;
            if cx >= cw_vis || cy >= ch_vis {
                continue; // fully outside the visible chroma frame
            }
            let u = (lby / 8) * uc + lbx / 8;
            let cand: &[u16] = if filtering {
                // dir 0 when the signaled pri strength is 0 (see luma above).
                let dir = if pri == 0 {
                    0
                } else {
                    uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)]
                };
                cdef::cdef_filter_block(
                    &mut tmp,
                    cy,
                    recon,
                    cw,
                    cx,
                    cy,
                    cbw,
                    cbh,
                    pri << (bd - 8),
                    sec << (bd - 8),
                    dir,
                    damping,
                    bd,
                );
                &tmp
            } else {
                recon
            };
            let mut sse = 0i64;
            for yy in cy..(cy + cbh).min(ch_vis) {
                for xx in cx..(cx + cbw).min(cw_vis) {
                    let row = if filtering { yy - cy } else { yy };
                    let d = cand[row * cw + xx] as i64 - src[yy * cw + xx] as i64;
                    sse += d * d;
                }
            }
            out[u] += sse;
        }
    }
    out
}

/// Apply the luma CDEF strength to a whole plane in place, reading from the
/// MARKED pre-CDEF snapshot (out-of-frame samples = `CDEF_VERY_LARGE`) so every
/// 8x8 filters the same source pixels the decoder does. With a per-unit `grid`
/// (`cdef_bits = 1`), only 8x8s inside on-units are filtered.
#[allow(clippy::too_many_arguments)]
fn apply_cdef_plane(
    plane: &mut [u16],
    snapshot: &[u16],
    w: usize,
    _h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    unit_pri_sec: &[(i32, i32)],
    damping: i32,
    bd: u8,
    pool: &Pool,
) {
    use crate::cdef;
    // Every 8x8 reads the pre-CDEF snapshot and writes only its own rows, so
    // 8-row bands filter in parallel with identical output to the serial pass.
    let items: Vec<(usize, &mut [u16])> = plane.chunks_mut(8 * w).enumerate().collect();
    pool.for_each(pool.width(), items, |(bi8, band)| {
        let y = bi8 * 8;
        for x in (0..w).step_by(8) {
            let bxi = x / 8;
            let byi = y / 8;
            let bi = byi * nbx + bxi;
            // Skip blocks are left untouched, matching the decoder.
            if skip.get(bi).copied().unwrap_or(true) {
                continue;
            }
            // The unit's signaled strength; (0,0) units stay untouched.
            let (pri, sec) = unit_pri_sec[(y / 64) * uc + x / 64];
            if pri == 0 && sec == 0 {
                continue;
            }
            // Use the precomputed direction/variance (same pre-CDEF snapshot the
            // search used). Identical to the old in-loop recompute, but free.
            // Decoders force dir 0 when the signaled pri strength is 0.
            let dir = if pri == 0 { 0 } else { dirs[bi] };
            let var = vars[bi];
            // adjust_pri on the bit-depth-shifted strength (see search above).
            let apri = cdef::adjust_pri(pri << (bd - 8), var);
            cdef::cdef_filter_block(
                band,
                y,
                snapshot,
                w,
                x,
                y,
                8,
                8,
                apri,
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn apply_cdef_chroma(
    plane: &mut [u16],
    snapshot: &[u16],
    cw: usize,
    ch: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    skip8: &[bool],
    sb8w: usize,
    nbx: usize,
    nby: usize,
    sub_x: usize,
    sub_y: usize,
    uc: usize,
    unit_pri_sec: &[(i32, i32)],
    damping: i32,
    bd: u8,
    pool: &Pool,
) {
    use crate::cdef;
    let cbw = 8 >> sub_x; // chroma sub-block width per luma 8x8
    let cbh = 8 >> sub_y;
    // One band per luma block-row (cbh chroma rows): disjoint writes, shared
    // pre-CDEF snapshot reads — identical output to the serial pass.
    let items: Vec<(usize, &mut [u16])> = plane.chunks_mut(cbh * cw).enumerate().collect();
    pool.for_each(pool.width(), items, |(lby, band)| {
        if lby >= nby {
            return;
        }
        let cy = (lby * 8) >> sub_y;
        for lbx in 0..nbx {
            // Skip if the covering luma 8x8 is a skip block.
            if skip8.get(lby * sb8w + lbx).copied().unwrap_or(true) {
                continue;
            }
            // The unit's signaled chroma strength; (0,0) units stay untouched.
            let (pri, sec) = unit_pri_sec[(lby / 8) * uc + lbx / 8];
            if pri == 0 && sec == 0 {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            if cx >= cw || cy >= ch {
                continue;
            }
            // Decoders force dir 0 when the signaled pri strength is 0.
            let dir = if pri == 0 {
                0
            } else {
                uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)]
            };
            cdef::cdef_filter_block(
                band,
                cy,
                snapshot,
                cw,
                cx,
                cy,
                cbw,
                cbh,
                pri << (bd - 8),
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    });
}
