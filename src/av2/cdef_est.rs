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

//! Per-block CDEF strength estimation for AV2.
//!
//! AV2's CDEF strength used to be fed forward from the quantizer index — a
//! content-blind guess. A single global strength cannot work either: it must
//! filter every superblock, so it can't clean flat-region ringing without
//! smearing detail. This module signals CDEF **per 64x64 superblock**
//! (`nb_cdef_strengths = 2`: off, or one searched active strength), each SB
//! choosing by a two-pass reconstruction search — the shape of AVM's
//! `av2/encoder/pickcdef.c` per-SB decision, reusing the shared bit-exact
//! primitives in [`crate::cdef`]; the AV1 path is untouched.
//!
//! Decision metric: libaom's variance-weighted `cdef_dist_8x8` (perceptual,
//! down-weights error in textured blocks) rather than plain MSE, since the codec
//! targets SSIMULACRA2. That alone does not protect isotropic fine detail
//! (its weight collapses to SSE at high texture variance), so an SB is also gated
//! on its mean `cdef_direction` variance — high only where ringing lives along a
//! dominant edge. Together these keep the flat-ringing wins (synthetic art
//! +1..+2 SS2) while never turning CDEF on where it would erase detail.
//!
//! Two-pass, mirroring the 420 CCSO flow: pass 1 builds the reconstruction and
//! derives the per-SB grid; pass 2 re-emits with the grid signalled. Runs on the
//! whole-frame (single-tile) path and, per tile, on the multi-tile path (the grid
//! is re-indexed to each tile's local superblock coordinates). Validated
//! end-to-end via the avifdec oracle (real deblock -> CDEF -> CCSO). Honest
//! approximation: the search reads the *undeblocked* recon (AV2 is a signal-only
//! pipeline); for the per-SB on/off decision the perturbation is small.

use crate::cdef;

/// Candidate primary strengths (the 6-bit signaled field, `strength >> 2`).
/// Wider than AV1's `{0,1,2,4}` because AV2 ringing-heavy content wants a
/// stronger primary (ceiling sweeps favored field ~8 on synthetic images).
static PRI_CANDS: [i32; 6] = [0, 1, 2, 4, 7, 11];
/// Candidate secondary strengths (`strength & 3`; 3 would decode to 4, unused).
static SEC_CANDS: [i32; 3] = [0, 1, 2];

/// Fraction of the "off" SSE a candidate must save to be accepted. Guards the
/// SSE-vs-SSIMULACRA2 mismatch: a gentle denoise can shave SSE on a smooth photo
/// while erasing detail the perceptual metric rewards, so require a real gain.
const MARGIN_DEN: i64 = 1000;

/// CDEF damping derived from the base quantizer (spec range 3..=6). Local copy so
/// the AV1 coder is not touched.
fn damping_for(base_q_idx: u8) -> i32 {
    3 + ((base_q_idx as i32) / 64).min(3)
}

/// Round an integer-valued f32 recon/source plane to i32 for the fixed-point
/// filter. AV2 reconstruction is stored as exact integers in f32, so this is
/// lossless.
fn to_i32(p: &[f32]) -> Vec<i32> {
    p.iter().map(|&v| v as i32).collect()
}

fn block_sse(a: &[i32], b: &[i32], w: usize, h: usize, x: usize, y: usize) -> i64 {
    let mut s = 0i64;
    let yh = (y + 8).min(h);
    let xw = (x + 8).min(w);
    for yy in y..yh {
        for xx in x..xw {
            let d = (a[yy * w + xx] - b[yy * w + xx]) as i64;
            s += d * d;
        }
    }
    s
}

/// libaom's perceptual CDEF distortion for one 8x8 block (`dist_8x8_16bit`): the
/// SSE weighted by `0.5 * (svar + dvar + c1) / sqrt(c2 + svar*dvar)`, where `svar`
/// / `dvar` are the (x64) source / reconstructed variances. The weight up-weights
/// error in flat blocks (where ringing is visible) and down-weights it in
/// high-variance textured blocks (masking) — so filtering, which cuts a block's
/// variance, raises its weight and only wins where the SSE drop is real ringing.
/// `dst` is the candidate (filtered/unfiltered) recon, `src` the source. Partial
/// edge blocks fall back to plain SSE (the ×64 variance calibration needs a full
/// 8x8). This is an encoder-side decision metric, not part of the bitstream.
fn cdef_dist_8x8(
    src: &[i32],
    dst: &[i32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    coeff_shift: u32,
) -> i64 {
    if x + 8 > w || y + 8 > h {
        return block_sse(dst, src, w, h, x, y);
    }
    let (mut ss, mut sd, mut ss2, mut sd2, mut ssd) = (0i64, 0i64, 0i64, 0i64, 0i64);
    for yy in y..y + 8 {
        for xx in x..x + 8 {
            let s = src[yy * w + xx] as i64;
            let d = dst[yy * w + xx] as i64;
            ss += s;
            sd += d;
            ss2 += s * s;
            sd2 += d * d;
            ssd += s * d;
        }
    }
    let svar = ss2 - ((ss * ss + 32) >> 6);
    let dvar = sd2 - ((sd * sd + 32) >> 6);
    let sse = (sd2 + ss2 - 2 * ssd) as f64;
    let c1 = (400i64 << (2 * coeff_shift)) as f64;
    let c2 = (20000i64 << (4 * coeff_shift)) as f64;
    let w = 0.5 * (svar as f64 + dvar as f64 + c1) / (c2 + svar as f64 * dvar as f64).sqrt();
    (0.5 + sse * w).floor() as i64
}

// ---------------------------------------------------------------------------
// Per-block CDEF: one active strength, chosen per superblock (64x64 CDEF unit).
// Each SB decides off (index 0) vs the active strength (index 1) by its own SSE,
// so detail-heavy SBs keep their texture while flat/ringing SBs are cleaned. This
// is what a single global strength cannot do (it must filter every block).
// ---------------------------------------------------------------------------

/// Frame CDEF decision handed from the pass-1 search to the pass-2 emit.
#[derive(Clone)]
pub(crate) struct CdefDecision {
    pub(crate) damping: u8,
    pub(crate) y_str: u8,
    pub(crate) uv_str: u8,
    /// Per-CDEF-unit index (0 = off, 1 = active), row-major over `sb_cols`.
    pub(crate) grid: Vec<u8>,
    pub(crate) sb_cols: usize,
}

/// Fraction (per-mille) of a superblock's "off" distortion that filtering must
/// save for that SB to turn CDEF on. Small: the perceptual cdef_dist already
/// handles the texture/ringing decision; this only stands in for the per-SB index
/// rate (~1 bit) so negligible-gain SBs stay off.
const SB_MARGIN_NUM: i64 = 10;

/// 64x64 superblock index for the 8x8 luma block at grid coords `(bx, by)`.
fn sb_of(bx: usize, by: usize, sb_cols: usize) -> usize {
    (by / 8) * sb_cols + (bx / 8)
}

/// Per-SB off/filtered luma SSE. `out[sb]` accumulates the SSE of every 8x8 block
/// in that SB (filtered at `(pri, sec)` when `pri|sec != 0`, else unfiltered).
#[allow(clippy::too_many_arguments)]
fn luma_sse_per_sb(
    rec: &[i32],
    src: &[i32],
    w: usize,
    h: usize,
    dirs: &[usize],
    vars: &[i32],
    nbx: usize,
    sb_cols: usize,
    n_sb: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) -> Vec<i64> {
    let coeff_shift = (bd - 8) as u32;
    let mut out = vec![0i64; n_sb];
    let mut tmp = if pri != 0 || sec != 0 {
        rec.to_vec()
    } else {
        Vec::new()
    };
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let bx = x / 8;
            let by = y / 8;
            // Perceptual (variance-weighted) distortion drives the per-SB decision.
            let dist = if pri != 0 || sec != 0 {
                let bi = by * nbx + bx;
                let apri = cdef::adjust_pri(pri << (bd - 8), vars[bi]);
                cdef::cdef_filter_8x8(
                    &mut tmp,
                    rec,
                    w,
                    x,
                    y,
                    apri,
                    sec << (bd - 8),
                    dirs[bi],
                    damping,
                    bd,
                );
                cdef_dist_8x8(src, &tmp, w, h, x, y, coeff_shift)
            } else {
                cdef_dist_8x8(src, rec, w, h, x, y, coeff_shift)
            };
            out[sb_of(bx, by, sb_cols)] += dist;
        }
    }
    out
}

/// Per-SB off/filtered chroma SSE, accumulated into the covering *luma* SB bucket
/// (the CDEF index is shared with luma). `(sub_x, sub_y)` map chroma pixels back to
/// luma coordinates.
#[allow(clippy::too_many_arguments)]
fn chroma_sse_per_sb(
    rec: &[i32],
    src: &[i32],
    cw: usize,
    ch: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    nbx: usize,
    sb_cols: usize,
    n_sb: usize,
    sub_x: usize,
    sub_y: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) -> Vec<i64> {
    let cbw = 8 >> sub_x;
    let cbh = 8 >> sub_y;
    let mut out = vec![0i64; n_sb];
    let mut tmp = if pri != 0 || sec != 0 {
        rec.to_vec()
    } else {
        Vec::new()
    };
    let nby = ldirs.len().div_ceil(nbx);
    // One chroma CDEF block corresponds to one luma 8x8 direction block.
    for lby in 0..nby {
        for lbx in 0..nbx {
            let cx = (lbx * 8) >> sub_x;
            let cy = (lby * 8) >> sub_y;
            if cx >= cw || cy >= ch {
                continue;
            }
            let sse = if pri != 0 || sec != 0 {
                let dir = uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)];
                cdef::cdef_filter_block(
                    &mut tmp,
                    0,
                    rec,
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
                let mut s = 0i64;
                for yy in cy..(cy + cbh).min(ch) {
                    for xx in cx..(cx + cbw).min(cw) {
                        let d = (tmp[yy * cw + xx] - src[yy * cw + xx]) as i64;
                        s += d * d;
                    }
                }
                s
            } else {
                let mut s = 0i64;
                for yy in cy..(cy + cbh).min(ch) {
                    for xx in cx..(cx + cbw).min(cw) {
                        let d = (rec[yy * cw + xx] - src[yy * cw + xx]) as i64;
                        s += d * d;
                    }
                }
                s
            };
            out[sb_of(lbx, lby, sb_cols)] += sse;
        }
    }
    out
}

/// Lower bound on a superblock's mean `cdef_direction` variance for CDEF to be
/// allowed on. That variance measures how strongly the block is dominated by a
/// single edge direction: high where transform ringing lives along real edges
/// (CDEF's target — a win), low on flat gradients and on isotropic fine detail
/// (fractals) that has no dominant direction. The perceptual `cdef_dist` metric
/// alone does not protect the latter (its weight collapses to plain SSE at high
/// texture variance), so gate on the directional variance too.
const SB_DIR_VAR_THRESH: i64 = 15000;

/// Given per-SB off/filtered perceptual distortion and per-SB mean directional
/// variance, the achievable total when each SB independently picks off vs filtered
/// (subject to the rate-proxy margin and the directional-variance gate), plus the
/// on/off grid. Returns `(total, grid, n_on)`.
fn apply_grid(off: &[i64], filt: &[i64], sb_dir_var: &[i64]) -> (i64, Vec<u8>, usize) {
    let m = SB_MARGIN_NUM;
    let vthr = SB_DIR_VAR_THRESH;
    let mut total = 0i64;
    let mut grid = vec![0u8; off.len()];
    let mut n_on = 0;
    for i in 0..off.len() {
        let thr = off[i] - off[i].saturating_mul(m) / MARGIN_DEN.max(1);
        if filt[i] < thr && sb_dir_var[i] >= vthr {
            total += filt[i];
            grid[i] = 1;
            n_on += 1;
        } else {
            total += off[i];
        }
    }
    (total, grid, n_on)
}

/// Apply a signaled per-block CDEF decision to the deblocked reconstruction.
/// Every output sample reads from an immutable pre-CDEF snapshot.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_per_block(
    rec: &mut [Vec<f32>],
    pw: usize,
    ph: usize,
    cw: usize,
    ch: usize,
    sub_x: usize,
    sub_y: usize,
    has_chroma: bool,
    decision: &CdefDecision,
    bd: u8,
) {
    if rec.is_empty() || rec[0].len() < pw.saturating_mul(ph) {
        return;
    }
    let src_y = to_i32(&rec[0]);
    let mut dst_y = src_y.clone();
    let nbx = pw.div_ceil(8);
    let nby = ph.div_ceil(8);
    let blocks: Vec<(usize, i32)> = (0..nbx * nby)
        .map(|idx| {
            let (bx, by) = (idx % nbx, idx / nbx);
            cdef::cdef_direction(&src_y, pw, bx * 8, by * 8, bd)
        })
        .collect();
    let dirs: Vec<usize> = blocks.iter().map(|&(dir, _)| dir).collect();
    let vars: Vec<i32> = blocks.iter().map(|&(_, var)| var).collect();
    let coeff_shift = bd - 8;
    let decode_strength = |strength: u8| {
        let pri = (strength >> 2) as i32;
        let sec = match strength & 3 {
            3 => 4,
            value => value as i32,
        };
        (pri, sec)
    };

    let (y_pri, y_sec) = decode_strength(decision.y_str);
    for by in 0..nby {
        for bx in 0..nbx {
            let sb = sb_of(bx, by, decision.sb_cols);
            if decision.grid.get(sb).copied().unwrap_or(0) == 0 {
                continue;
            }
            let bi = by * nbx + bx;
            cdef::cdef_filter_8x8(
                &mut dst_y,
                &src_y,
                pw,
                bx * 8,
                by * 8,
                cdef::adjust_pri(y_pri << coeff_shift, vars[bi]),
                y_sec << coeff_shift,
                dirs[bi],
                decision.damping as i32,
                bd,
            );
        }
    }
    rec[0] = dst_y.into_iter().map(|value| value as f32).collect();

    if !has_chroma || rec.len() < 3 || rec[1].len() < cw * ch || rec[2].len() < cw * ch {
        return;
    }
    let (uv_pri, uv_sec) = decode_strength(decision.uv_str);
    if uv_pri == 0 && uv_sec == 0 {
        return;
    }
    let uv_dir: [usize; 8] = if sub_x == 1 && sub_y == 0 {
        [7, 0, 2, 4, 5, 6, 6, 6]
    } else {
        [0, 1, 2, 3, 4, 5, 6, 7]
    };
    let (cbw, cbh) = (8 >> sub_x, 8 >> sub_y);
    for plane in &mut rec[1..=2] {
        let src = to_i32(plane);
        let mut dst = src.clone();
        for by in 0..nby {
            for bx in 0..nbx {
                let sb = sb_of(bx, by, decision.sb_cols);
                if decision.grid.get(sb).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let (cx, cy) = ((bx * 8) >> sub_x, (by * 8) >> sub_y);
                if cx >= cw || cy >= ch {
                    continue;
                }
                cdef::cdef_filter_block(
                    &mut dst,
                    0,
                    &src,
                    cw,
                    cx,
                    cy,
                    cbw,
                    cbh,
                    uv_pri << coeff_shift,
                    uv_sec << coeff_shift,
                    uv_dir[dirs[by * nbx + bx]],
                    (decision.damping as i32 - 1).max(1),
                    bd,
                );
            }
        }
        *plane = dst.into_iter().map(|value| value as f32).collect();
    }
}

/// Per-block CDEF search. Returns a per-SB decision, or `None` when no superblock
/// benefits. `rec`/`src` are `[Y,(U,V)]` integer-valued f32 planes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_per_block(
    rec: &[Vec<f32>],
    src: &[Vec<f32>],
    pw: usize,
    ph: usize,
    cw: usize,
    ch: usize,
    sub_x: usize,
    sub_y: usize,
    has_chroma: bool,
    base_q_idx: u8,
    bd: u8,
) -> Option<CdefDecision> {
    let damping = damping_for(base_q_idx);
    let recy = to_i32(&rec[0]);
    let srcy = to_i32(&src[0]);
    let nbx = pw.div_ceil(8);
    let nby = ph.div_ceil(8);
    // Every caller runs this single-threaded-context (see the strength search
    // below), so the per-8×8-block direction/variance precompute — one
    // independent `cdef_direction` per block — parallelises cleanly and
    // byte-identically (pure per block; the scatter preserves index order).
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let blk: Vec<(usize, i32)> = crate::av2::helpers::par_map_indexed(nthreads, nbx * nby, |idx| {
        let bx = idx % nbx;
        let by = idx / nbx;
        if bx * 8 < pw && by * 8 < ph {
            cdef::cdef_direction(&recy, pw, bx * 8, by * 8, bd)
        } else {
            (0usize, 0i32)
        }
    });
    let dirs: Vec<usize> = blk.iter().map(|&(d, _)| d).collect();
    let vars: Vec<i32> = blk.iter().map(|&(_, v)| v).collect();
    let sb_cols = pw.div_ceil(64);
    let sb_rows = ph.div_ceil(64);
    let n_sb = sb_cols * sb_rows;

    // Per-SB mean directional variance (from cdef_direction) for the gate.
    let mut dv_sum = vec![0i64; n_sb];
    let mut dv_cnt = vec![0i64; n_sb];
    for by in 0..nby {
        for bx in 0..nbx {
            if bx * 8 < pw && by * 8 < ph {
                let s = sb_of(bx, by, sb_cols);
                dv_sum[s] += vars[by * nbx + bx] as i64;
                dv_cnt[s] += 1;
            }
        }
    }
    let sb_dir_var: Vec<i64> = (0..n_sb).map(|i| dv_sum[i] / dv_cnt[i].max(1)).collect();

    let luma_off = luma_sse_per_sb(
        &recy, &srcy, pw, ph, &dirs, &vars, nbx, sb_cols, n_sb, 0, 0, damping, bd,
    );

    // Search the single active luma strength: pick the (pri, sec) whose per-SB
    // optimal application minimises total perceptual distortion (two-stage).
    let eval = |pri: i32, sec: i32| -> (i64, Vec<u8>, usize) {
        let filt = luma_sse_per_sb(
            &recy, &srcy, pw, ph, &dirs, &vars, nbx, sb_cols, n_sb, pri, sec, damping, bd,
        );
        apply_grid(&luma_off, &filt, &sb_dir_var)
    };
    let off_total: i64 = luma_off.iter().sum();
    let mut best = (off_total, vec![0u8; n_sb], 0usize);
    // Each strength candidate rescans the whole frame independently, so evaluate
    // them on a work-stealing map and then reduce serially in candidate order —
    // byte-identical to the serial first-wins loop (i64 SSE, so order-invariant).
    // Every caller runs `search_per_block` in a single-threaded context (whole-
    // frame encode, or the tiled frame-level call after the tile barrier — the
    // per-tile core search is gated off by `capture_recon`), so this never nests
    // under tile parallelism.
    let mut best_pri = 0i32;
    let pri_cands: Vec<i32> = PRI_CANDS.iter().skip(1).copied().collect();
    let pri_results =
        crate::av2::helpers::par_map_indexed(nthreads, pri_cands.len(), |i| eval(pri_cands[i], 0));
    for (i, r) in pri_results.into_iter().enumerate() {
        if r.0 < best.0 {
            best = r;
            best_pri = pri_cands[i];
        }
    }
    if best_pri == 0 || best.2 == 0 {
        return None;
    }
    let mut best_sec = 0i32;
    let sec_cands: Vec<i32> = SEC_CANDS.iter().skip(1).copied().collect();
    let sec_results = crate::av2::helpers::par_map_indexed(nthreads, sec_cands.len(), |i| {
        eval(best_pri, sec_cands[i])
    });
    for (i, r) in sec_results.into_iter().enumerate() {
        if r.0 < best.0 {
            best = r;
            best_sec = sec_cands[i];
        }
    }
    let (best_total, grid, n_on) = best;
    // Frame-level gain gate: engaging per-block CDEF costs a full second encode
    // pass, so require a meaningful whole-frame perceptual-distortion reduction
    // (0.5%). Marginal frames (most photos, where only a few edge SBs flicker on)
    // skip pass 2 and leave CDEF off — saving the re-encode and any sub-noise SS2
    // wobble. Strong flat-ringing frames clear it easily.
    const FRAME_MIN_GAIN: i64 = 5; // per-mille
    let gain = off_total - best_total;
    if n_on == 0 || gain <= off_total.saturating_mul(FRAME_MIN_GAIN) / 1000 {
        return None;
    }
    let y_str = ((best_pri << 2) | best_sec) as u8;

    // Chroma: same per-SB grid (shared index). Pick the uv strength that minimises
    // chroma SSE over the on-SBs; if none beats off, uv stays 0 (chroma CDEF off).
    let uv_str = if has_chroma {
        let uv_dir: [usize; 8] = if sub_x == 1 && sub_y == 0 {
            [7, 0, 2, 4, 5, 6, 6, 6]
        } else {
            [0, 1, 2, 3, 4, 5, 6, 7]
        };
        let c_damping = (damping - 1).max(1);
        let recu = to_i32(&rec[1]);
        let srcu = to_i32(&src[1]);
        let recv = to_i32(&rec[2]);
        let srcv = to_i32(&src[2]);
        let on_sse = |pri: i32, sec: i32| -> i64 {
            let u = chroma_sse_per_sb(
                &recu, &srcu, cw, ch, &dirs, &uv_dir, nbx, sb_cols, n_sb, sub_x, sub_y, pri, sec,
                c_damping, bd,
            );
            let v = chroma_sse_per_sb(
                &recv, &srcv, cw, ch, &dirs, &uv_dir, nbx, sb_cols, n_sb, sub_x, sub_y, pri, sec,
                c_damping, bd,
            );
            (0..n_sb)
                .filter(|&i| grid[i] == 1)
                .map(|i| u[i] + v[i])
                .sum()
        };
        let off_on = on_sse(0, 0);
        let mut c_best = off_on;
        // Same work-stealing candidate eval + serial in-order reduce as luma
        // (i64 SSE ⇒ byte-identical to the serial first-wins loop).
        let mut c_pri = 0i32;
        let cpri_cands: Vec<i32> = PRI_CANDS.iter().skip(1).copied().collect();
        let cpri_res = crate::av2::helpers::par_map_indexed(nthreads, cpri_cands.len(), |i| {
            on_sse(cpri_cands[i], 0)
        });
        for (i, s) in cpri_res.into_iter().enumerate() {
            if s < c_best {
                c_best = s;
                c_pri = cpri_cands[i];
            }
        }
        let mut c_sec = 0i32;
        if c_pri != 0 {
            let csec_cands: Vec<i32> = SEC_CANDS.iter().skip(1).copied().collect();
            let csec_res = crate::av2::helpers::par_map_indexed(nthreads, csec_cands.len(), |i| {
                on_sse(c_pri, csec_cands[i])
            });
            for (i, s) in csec_res.into_iter().enumerate() {
                if s < c_best {
                    c_best = s;
                    c_sec = csec_cands[i];
                }
            }
        }
        ((c_pri << 2) | c_sec) as u8
    } else {
        0
    };

    Some(CdefDecision {
        damping: damping as u8,
        y_str,
        uv_str,
        grid,
        sb_cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_filters_only_enabled_superblocks() {
        let (w, h) = (128usize, 64usize);
        let y: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, row) = (i % w, i / w);
                ((x * 5 + row * 3 + ((x + row) % 7) * 11) & 255) as f32
            })
            .collect();
        let mut rec = vec![y.clone(), vec![128.0; 64 * 32], vec![128.0; 64 * 32]];
        apply_per_block(
            &mut rec,
            w,
            h,
            64,
            32,
            1,
            1,
            true,
            &CdefDecision {
                damping: 4,
                y_str: (7 << 2) | 2,
                uv_str: 0,
                grid: vec![1, 0],
                sb_cols: 2,
            },
            8,
        );
        assert!((0..h).any(|row| rec[0][row * w..row * w + 64] != y[row * w..row * w + 64]));
        for row in 0..h {
            assert_eq!(
                &rec[0][row * w + 64..(row + 1) * w],
                &y[row * w + 64..(row + 1) * w]
            );
        }
    }
}
