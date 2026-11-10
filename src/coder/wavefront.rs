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

// Per-tile superblock wavefront: parallel capture into semantic tokens and the raster entropy packer.

/// Completed output of one parallel superblock decision. Unlike the old
/// `DecisionRecord`, this contains no search winners, coefficient copies, or
/// reconstruction copies: only the compact syntax operations consumed by the
/// raster entropy lane.
struct CapturedSb {
    tokens: crate::odec::EntropyTokens,
}

struct WavefrontPlanes {
    recon: [Vec<u16>; 3],
    maps: BlockMaps,
}

/// Whether this WPP cell is dependency-downstream of the complete top-row
/// pilot. Above-right dependencies advance that proof one column per row.
#[inline]
fn use_warmed_decision_cdf(row: usize, col: usize, prefix: usize) -> bool {
    if row == 0 {
        col >= prefix
    } else {
        row + col >= prefix.saturating_sub(1)
    }
}

/// Parallel SB-wavefront capture pass over one tile: decide every superblock's
/// RD choices out of raster order (schedule `d = 2r + c`, so top, left AND
/// above-right neighbors are always finished), publish final reconstruction
/// and filter maps directly into disjoint shared regions, and stream compact
/// entropy operations to the raster packer. The packer restores full CDF
/// adaptivity without traversing blocks a second time.
///
/// Safety of the parallelism rests on three proven properties:
/// - decisions read reconstruction only inside the halo bands;
/// - every ctx-array read stays inside the SB's exact own column/row segment
///   (zero margin—neighboring `l_*` rows have a different last writer under
///   diagonal order than under raster order);
/// - decisions never read mutable adaptive CDFs or encoder state. They use one
///   of two immutable, dependency-selected `dec_cdfs` snapshots, so each worker
///   captures into a sink encoder without schedule-dependent rate estimates.
///
/// Shared state is written disjointly: each cell writes only its own SB block
/// of the `done` recon planes and its own segments of the ctx handoff arrays
/// (top→bottom handoff for `a_*`, left→right for `l_*`), exactly mirroring
/// what the serial raster loop would have left there for that reader.
#[allow(clippy::needless_range_loop)] // `p` indexes recon planes AND their writers in lockstep
fn wavefront_capture(
    nthreads: usize,
    job: &TileJob<'_>,
    tsrc: &[Vec<u16>; 3],
    r: &TileRect,
    ibc_index: Option<&std::sync::OnceLock<LossyIbcIndex>>,
    tx: std::sync::mpsc::Sender<(usize, CapturedSb)>,
) -> WavefrontPlanes {
    use crate::av2::helpers::{PlaneWriter, par_wavefront_pool_with};
    const HB: usize = 4; // halo band thickness (AV1 intra needs 1 px; generous)
    let sb_rows = r.th.div_ceil(64);
    let sb_cols = r.tw.div_ceil(64);
    let warm_decisions = job.warm_decisions();
    let warm_prefix = if warm_decisions { sb_cols.min(4) } else { 0 };
    // `mk_tile` runs once per worker.  Computing this inside it made every
    // worker rescan the complete source tile when AQ/variance boost was on.
    // The reference activity is immutable and identical for all workers.
    let ref_act = job.ref_activity(tsrc, r);
    let mk_tile = || {
        let mut t = job.build_tile(tsrc, r, ibc_index, ref_act);
        // Decisions run normally, but their final syntax is captured directly
        // as semantic entropy tokens. There is no winner/coeff replay record.
        let slots = t.ent.cdfs.semantic_slots();
        t.ent.enc.set_semantic_cdfs(&slots);
        // Capture records the table identity + symbol. Only the raster packer
        // consumes live probabilities, so adapting worker-private CDF copies
        // is duplicate work (decision RDO reads `dec_cdfs`, never `cdfs`).
        t.ent.enc.updating_cdf = false;
        t.ent.enc.sink = true;
        t
    };
    // Prototype: geometry, frame-initial ctx array contents, and the AQ grid
    // (identical to the serial pass's — bit-exact by `aq_grid_matches_serial`).
    let proto = mk_tile();
    let initial_dec_cdfs = proto.model.dec_cdfs.clone();
    let warm_dec_cdfs = std::sync::Mutex::new((*initial_dec_cdfs).clone());
    let published_dec_cdfs = std::sync::OnceLock::<std::sync::Arc<Cdfs>>::new();
    // Per-SB prefix sums (along each row) of emitted all-zero counts, for the
    // adaptive skip price. A cell writes its own slot once; readers only touch
    // slots of cells the wavefront has already finished.
    let skip_prefix = std::sync::Mutex::new(vec![[[[0u32; 2]; 13]; 5]; sb_rows * sb_cols]);
    let aq_grid = proto.precompute_aq_grid();
    // Unfinished regions of the shared planes must hold exactly what a serial
    // tile's uncoded regions hold (the `recon_src_init` estimate, or zeros),
    // or wavefront decisions diverge from -t1.
    let est_init: std::sync::Arc<[Vec<u16>; 3]> = std::sync::Arc::new(proto.ctx.recon.clone());
    let mut done = [
        proto.ctx.recon[0].clone(),
        proto.ctx.recon[1].clone(),
        proto.ctx.recon[2].clone(),
    ];
    let stride4 = r.tw / 4;
    let stride8 = r.tw.div_ceil(8);
    let BlockMaps {
        mut skip8,
        mut blk4,
        mut blk4h,
        mut blk4v,
        mut blk4t,
        mut pblk4,
        mut pblk4h,
        mut pblk4v,
        mut pblk4t,
    } = BlockMaps::new(r.tw, r.th);
    // Ctx handoff arrays, frame-initial (clones of a fresh tile's arrays).
    let mut h_arrs: Vec<Vec<u8>> = vec![
        proto.ctx.a_coef[0].clone(),
        proto.ctx.a_coef[1].clone(),
        proto.ctx.a_coef[2].clone(),
        proto.ctx.l_coef[0].clone(),
        proto.ctx.l_coef[1].clone(),
        proto.ctx.l_coef[2].clone(),
        proto.ctx.a_part.clone(),
        proto.ctx.l_part.clone(),
        proto.ctx.a_skip.clone(),
        proto.ctx.l_skip.clone(),
        proto.ctx.a_mode.clone(),
        proto.ctx.l_mode.clone(),
        proto.ctx.a_uv_mode.clone(),
        proto.ctx.l_uv_mode.clone(),
    ];
    let (ss422, ss420) = (proto.cfg.is_422(), proto.cfg.is_420());
    let cw = proto.cfg.cw;
    let allow_ibc = proto.cfg.allow_intrabc;
    let (w4, h4) = (r.tw / 4, r.th / 4);
    let proto_a_tx: Vec<u8> = proto.ctx.a_tx.iter().map(|&v| v as u8).collect();
    let proto_l_tx: Vec<u8> = proto.ctx.l_tx.iter().map(|&v| v as u8).collect();
    drop(proto);
    // Shared IntraBC MV plane: each cell writes its own SB region after
    // finishing (packed (dy<<16)|dx, 0 = none — a zero DV cannot exist);
    // cells preload their predictor window from it (finished neighbors) so
    // the DV predictor sees exactly the serial pass's state.
    let mut ibc_plane: Vec<i32> = vec![0; if allow_ibc { w4 * h4 } else { 0 }];
    // Companion plane: luma prediction-block dims per finished 4x4 cell
    // ((w4 << 8) | h4, 0 = not yet coded) — the IntraBC DV predictor's
    // refmvs scan steps by neighbor block sizes, intra blocks included.
    let mut ibcd_plane: Vec<i32> = vec![0; if allow_ibc { w4 * h4 } else { 0 }];
    // Shared palette-context planes: the above/left neighbor palettes
    // (a_palette / l_palette), packed 9 x i32 per 4-sample position
    // ([len, c0..c7]). Handing these across cells lets the palette DECISION
    // pricing use the real color cache + mode context (it was deliberately
    // context-free before, which underprices palette in dense regions).
    let mut apal_plane: Vec<i32> = vec![0; w4 * 9];
    let mut lpal_plane: Vec<i32> = vec![0; h4 * 9];
    // The UV palette context needs the same handoff as the luma one. Without
    // it `a_palette_uv`/`l_palette_uv` kept whatever the WORKER last left in
    // them, so the UV palette symbols were coded against a context that
    // depended on the cell->worker assignment. The decoder recomputes the
    // correct context, so the stream desynced (volcanic.png, 4:4:4 q97: the
    // recon is bit-identical run to run, only the coded bits move).
    // Allocated only when there is chroma; mono never reads these.
    let uvpal = cw > 0;
    let mut apal_uv_plane: Vec<i32> = vec![0; if uvpal { w4 * 9 } else { 0 }];
    let mut lpal_uv_plane: Vec<i32> = vec![0; if uvpal { h4 * 9 } else { 0 }];
    let mut atx_plane: Vec<u8> = proto_a_tx;
    let mut ltx_plane: Vec<u8> = proto_l_tx;
    // Disjoint-write views. Plane p stride: luma w, chroma cw.
    let dws: Vec<PlaneWriter<u16>> = {
        let mut it = done.iter_mut();
        let l = it.next().unwrap();
        let u = it.next().unwrap();
        let v = it.next().unwrap();
        vec![
            PlaneWriter::new(l, r.tw),
            PlaneWriter::new(u, cw.max(1)),
            PlaneWriter::new(v, cw.max(1)),
        ]
    };
    let ibcw = PlaneWriter::new(&mut ibc_plane, w4.max(1));
    let ibcdw = PlaneWriter::new(&mut ibcd_plane, w4.max(1));
    let apalw = PlaneWriter::new(&mut apal_plane, (w4 * 9).max(1));
    let lpalw = PlaneWriter::new(&mut lpal_plane, (h4 * 9).max(1));
    let apaluvw = PlaneWriter::new(&mut apal_uv_plane, (w4 * 9).max(1));
    let lpaluvw = PlaneWriter::new(&mut lpal_uv_plane, (h4 * 9).max(1));
    let atxw = PlaneWriter::new(&mut atx_plane, w4.max(1));
    let ltxw = PlaneWriter::new(&mut ltx_plane, h4.max(1));
    let hws: Vec<PlaneWriter<u8>> = h_arrs
        .iter_mut()
        .map(|a| {
            let stride = a.len().max(1);
            PlaneWriter::new(a, stride)
        })
        .collect();
    let skip8w = PlaneWriter::new(&mut skip8, stride8.max(1));
    let blk4w = PlaneWriter::new(&mut blk4, stride4.max(1));
    let blk4hw = PlaneWriter::new(&mut blk4h, stride4.max(1));
    let blk4vw = PlaneWriter::new(&mut blk4v, stride4.max(1));
    let blk4tw = PlaneWriter::new(&mut blk4t, stride4.max(1));
    let pblk4w = PlaneWriter::new(&mut pblk4, stride4.max(1));
    let pblk4hw = PlaneWriter::new(&mut pblk4h, stride4.max(1));
    let pblk4vw = PlaneWriter::new(&mut pblk4v, stride4.max(1));
    let pblk4tw = PlaneWriter::new(&mut pblk4t, stride4.max(1));
    // Above-row reach: the classic above-right neighbour, or on IntraBC frames
    // `ibc_wavefront_reach` so every spec-legal DV source is a finished cell
    // (`find_intrabc` applies the same cone at every thread count).
    let above_right = if allow_ibc {
        (crate::tuning::get().ibc_wavefront_reach as usize).max(1)
    } else {
        1
    };
    par_wavefront_pool_with(
        nthreads,
        sb_rows,
        sb_cols,
        above_right,
        mk_tile,
        |t: &mut LossyTile, row: usize, col: usize| {
            // With above-right WPP dependencies, (row + col) >= prefix-1
            // proves that the four top-row pilot cells are complete. This
            // triangular boundary avoids a global startup barrier while making
            // the selected snapshot independent of worker scheduling.
            let use_warm = warm_decisions && use_warmed_decision_cdf(row, col, warm_prefix);
            if use_warm != t.model.decision_cdf_warmed {
                let next = if use_warm {
                    published_dec_cdfs
                        .get()
                        .expect("decision CDF warmup was not published")
                        .clone()
                } else {
                    initial_dec_cdfs.clone()
                };
                t.switch_decision_cdfs(next, use_warm);
            }
            let (sb_x, sb_y) = (col * 64, row * 64);
            // Per-plane geometry: (subsample x shift, subsample y shift).
            let shift = |p: usize| -> (usize, usize) {
                if p == 0 {
                    (0, 0)
                } else {
                    (((ss420 || ss422) as usize), (ss420 as usize))
                }
            };
            // --- copy-in: recon halo bands from the finished-SB planes ---
            for p in 0..3 {
                if t.ctx.recon[p].is_empty() {
                    continue;
                }
                let (sx, sy) = shift(p);
                let pw = if p == 0 { t.cfg.w } else { t.cfg.cw };
                let ph = t.ctx.recon[p].len() / pw;
                let (bx, by) = (sb_x >> sx, sb_y >> sy);
                let (bw, bh) = (64usize >> sx, 64usize >> sy);
                let x0 = bx.saturating_sub(HB);
                let x1 = (bx + 2 * bw).min(pw);
                let y0 = by.saturating_sub(HB);
                // A worker may receive an unrelated next cell, so initialize
                // only samples that are unavailable to that cell. The valid
                // top and left halos are overwritten from `done` below; zeroing
                // them as part of the previous cell's teardown was redundant.
                for row2 in by..(by + bh).min(ph) {
                    t.ctx.recon[p][row2 * pw + bx..row2 * pw + x1]
                        .copy_from_slice(&est_init[p][row2 * pw + bx..row2 * pw + x1]);
                }
                // SAFETY: halo regions belong to earlier diagonals (finished,
                // not being written); regions are in-plane.
                unsafe {
                    if by > y0 {
                        dws[p].copy_region_to(&mut t.ctx.recon[p], y0, x0, by - y0, x1 - x0);
                    }
                    if bx > x0 {
                        let yh = (by + bh).min(ph) - by;
                        dws[p].copy_region_to(&mut t.ctx.recon[p], by, x0, yh, bx - x0);
                    }
                }
            }
            // --- copy-in: exact own ctx segments from the handoff arrays ---
            let cx = sb_x >> ((ss420 || ss422) as usize);
            let cy = sb_y >> (ss420 as usize);
            let segs: [(usize, usize); 14] = [
                (sb_x / 4, 16), // a_coef[0] (luma 4x4 cols, 16 per SB)
                (cx / 4, 16 >> ((ss420 || ss422) as usize)),
                (cx / 4, 16 >> ((ss420 || ss422) as usize)),
                (sb_y / 4, 16), // l_coef[0]
                (cy / 4, 16 >> (ss420 as usize)),
                (cy / 4, 16 >> (ss420 as usize)),
                (sb_x / 8, 8),  // a_part
                (sb_y / 8, 8),  // l_part
                (sb_x / 4, 16), // a_skip
                (sb_y / 4, 16), // l_skip
                (sb_x / 4, 16), // a_mode
                (sb_y / 4, 16), // l_mode
                (sb_x / 4, 16), // a_uv_mode
                (sb_y / 4, 16), // l_uv_mode
            ];
            let arr_of = |t: &mut LossyTile<'_>, i: usize| -> *mut Vec<u8> {
                match i {
                    0 => &mut t.ctx.a_coef[0],
                    1 => &mut t.ctx.a_coef[1],
                    2 => &mut t.ctx.a_coef[2],
                    3 => &mut t.ctx.l_coef[0],
                    4 => &mut t.ctx.l_coef[1],
                    5 => &mut t.ctx.l_coef[2],
                    6 => &mut t.ctx.a_part,
                    7 => &mut t.ctx.l_part,
                    8 => &mut t.ctx.a_skip,
                    9 => &mut t.ctx.l_skip,
                    10 => &mut t.ctx.a_mode,
                    11 => &mut t.ctx.l_mode,
                    12 => &mut t.ctx.a_uv_mode,
                    _ => &mut t.ctx.l_uv_mode,
                }
            };
            for (i, &(s, n)) in segs.iter().enumerate() {
                // SAFETY: arr_of returns a field pointer used immediately,
                // no aliasing (one array at a time).
                let arr = unsafe { &mut *arr_of(t, i) };
                if arr.is_empty() {
                    continue;
                }
                let e = (s + n).min(arr.len());
                if e > s {
                    // SAFETY: the segment was last written by a finished cell
                    // (above SB for a_*, left SB for l_*) or is frame-initial.
                    unsafe { hws[i].copy_region_to(&mut arr[..], 0, s, 1, e - s) };
                }
            }
            // --- IntraBC: preload the DV-predictor window from the shared
            // MV plane and expose the finished-recon read view ---
            if allow_ibc {
                let (ibp, _ibl, ibs) = ibcw.read_view();
                let (idp, _idl, ids) = ibcdw.read_view();
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let x0 = sbx4.saturating_sub(8);
                let x1 = (sbx4 + 24).min(w4);
                let y0 = sby4.saturating_sub(8);
                let y1 = (sby4 + 24).min(h4);
                for y in y0..y1 {
                    for x in x0..x1 {
                        // SAFETY: finished cells hold real values; unfinished
                        // cells hold the initial zeros, which decode to None —
                        // exactly the serial "not yet coded" state.
                        let v = unsafe { *ibp.add(y * ibs + x) };
                        t.ctx.ibc_mv[y * (t.cfg.w / 4) + x] = if v == 0 {
                            None
                        } else {
                            Some(((v >> 16) as i16, v as i16))
                        };
                        // SAFETY: as above; 0 = not yet coded, keep the
                        // local (initial) value exactly like serial.
                        let d = unsafe { *idp.add(y * ids + x) };
                        if d != 0 {
                            t.ctx.maps.pblk4[y * (t.cfg.w / 4) + x] = (d >> 8) as u8;
                            t.ctx.maps.pblk4h[y * (t.cfg.w / 4) + x] = d as u8;
                        }
                    }
                }
                t.cfg.ibc_shared = Some(IbcSharedRecon {
                    planes: [dws[0].read_view(), dws[1].read_view(), dws[2].read_view()],
                });
            }
            // --- palette-context copy-in: own segments, packed 9xi32 ---
            {
                let (ap, _al, _s1) = apalw.read_view();
                let (lp, _ll, _s2) = lpalw.read_view();
                let (apu, _aul, _s3) = apaluvw.read_view();
                let (lpu, _lul, _s4) = lpaluvw.read_view();
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                // Same loop, four arrays: the UV pair is skipped entirely when
                // there is no chroma, so mono pays nothing for this.
                let uvn = if uvpal { 4 } else { 2 };
                for (arr, base, ptr, lim) in [
                    (&mut t.ctx.a_palette, sbx4, ap, w4),
                    (&mut t.ctx.l_palette, sby4, lp, h4),
                    (&mut t.ctx.a_palette_uv, sbx4, apu, w4),
                    (&mut t.ctx.l_palette_uv, sby4, lpu, h4),
                ]
                .into_iter()
                .take(uvn)
                {
                    for i in 0..16usize {
                        let pos = base + i;
                        if pos >= lim {
                            break;
                        }
                        // SAFETY: own segment, last written by the finished
                        // above/left cell (or frame-initial zeros = empty).
                        let n = unsafe { *ptr.add(pos * 9) } as usize;
                        let v = &mut arr[pos];
                        v.clear();
                        for k in 0..n.min(8) {
                            v.push(unsafe { *ptr.add(pos * 9 + 1 + k) });
                        }
                    }
                }
            }
            // --- tx-size ctx copy-in: own segments (i8 as raw bytes) ---
            {
                let (atp, _al, _s1) = atxw.read_view();
                let (ltp, _ll, _s2) = ltxw.read_view();
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let na = 16usize.min(w4.saturating_sub(sbx4));
                for i in 0..na {
                    // SAFETY: own segment, last written by the finished above
                    // cell (or frame-initial).
                    t.ctx.a_tx[sbx4 + i] = unsafe { *atp.add(sbx4 + i) } as i8;
                }
                let nl = 16usize.min(h4.saturating_sub(sby4));
                for i in 0..nl {
                    // SAFETY: own segment (left cell / frame-initial).
                    t.ctx.l_tx[sby4 + i] = unsafe { *ltp.add(sby4 + i) } as i8;
                }
            }
            // --- per-cell resets ---
            t.ent.enc.begin_semantic_sink();
            let own = t.code_sb(
                sb_x,
                sb_y,
                aq_grid.get(row * sb_cols + col),
                &initial_dec_cdfs,
                || {
                    let pf = skip_prefix.lock().expect("skip prefix poisoned");
                    skip_stats_cone(&pf, row, col, sb_cols)
                },
            );
            skip_prefix_store(
                &mut skip_prefix.lock().expect("skip prefix poisoned"),
                row,
                col,
                sb_cols,
                &own,
            );
            // --- write-out: copy each own recon block directly into the
            // finished planes, without a transient contiguous SB scratch ---
            for p in 0..3 {
                if t.ctx.recon[p].is_empty() {
                    continue;
                }
                let (sx, sy) = shift(p);
                let pw = if p == 0 { t.cfg.w } else { t.cfg.cw };
                let ph = t.ctx.recon[p].len() / pw;
                let (bx, by) = (sb_x >> sx, sb_y >> sy);
                let (bw, bh) = (64usize >> sx, 64usize >> sy);
                let w2 = (bx + bw).min(pw) - bx;
                let h2 = (by + bh).min(ph) - by;
                // SAFETY: own SB block — no other concurrent writer.
                unsafe { dws[p].copy_region_from(&t.ctx.recon[p], pw, by, bx, h2, w2) };
            }
            if allow_ibc {
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let bw = (sbx4 + 16).min(w4) - sbx4;
                let bh = (sby4 + 16).min(h4) - sby4;
                let mut buf = [0i32; 16 * 16];
                for y in 0..bh {
                    for x in 0..bw {
                        buf[y * bw + x] = match t.ctx.ibc_mv[(sby4 + y) * (t.cfg.w / 4) + sbx4 + x]
                        {
                            Some((dy, dx)) => ((dy as u16 as i32) << 16) | (dx as u16 as i32),
                            None => 0,
                        };
                    }
                }
                // SAFETY: own SB region — no other concurrent writer.
                unsafe { ibcw.write_block(sby4, sbx4, bh, bw, &buf[..bw * bh]) };
                for y in 0..bh {
                    for x in 0..bw {
                        let i = (sby4 + y) * (t.cfg.w / 4) + sbx4 + x;
                        buf[y * bw + x] =
                            (i32::from(t.ctx.maps.pblk4[i]) << 8) | i32::from(t.ctx.maps.pblk4h[i]);
                    }
                }
                // SAFETY: own SB region — no other concurrent writer.
                unsafe { ibcdw.write_block(sby4, sbx4, bh, bw, &buf[..bw * bh]) };
            }
            {
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let uvn = if uvpal { 4 } else { 2 };
                for (arr, base, w, lim) in [
                    (&t.ctx.a_palette, sbx4, &apalw, w4),
                    (&t.ctx.l_palette, sby4, &lpalw, h4),
                    (&t.ctx.a_palette_uv, sbx4, &apaluvw, w4),
                    (&t.ctx.l_palette_uv, sby4, &lpaluvw, h4),
                ]
                .into_iter()
                .take(uvn)
                {
                    let mut buf = [0i32; 9 * 16];
                    let n = 16usize.min(lim.saturating_sub(base));
                    for i in 0..n {
                        let v = &arr[base + i];
                        buf[i * 9] = v.len() as i32;
                        for (k, &c) in v.iter().take(8).enumerate() {
                            buf[i * 9 + 1 + k] = c;
                        }
                    }
                    if n > 0 {
                        // SAFETY: own segment — no other concurrent writer.
                        unsafe { w.write_block(0, base * 9, 1, n * 9, &buf[..n * 9]) };
                    }
                }
            }
            // --- write-out: own ctx segments into the handoff arrays ---
            for (i, &(s, n)) in segs.iter().enumerate() {
                let arr = unsafe { &mut *arr_of(t, i) };
                if arr.is_empty() {
                    continue;
                }
                let e = (s + n).min(arr.len());
                if e > s {
                    // SAFETY: own exact segment — disjoint from every
                    // concurrent cell's segment.
                    unsafe { hws[i].write_block(0, s, 1, e - s, &arr[s..e]) };
                }
            }
            // --- tx-size ctx write-out: own segments ---
            {
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let na = 16usize.min(w4.saturating_sub(sbx4));
                if na > 0 {
                    let mut buf = [0u8; 16];
                    for (dst, &src) in buf.iter_mut().zip(&t.ctx.a_tx[sbx4..sbx4 + na]) {
                        *dst = src as u8;
                    }
                    // SAFETY: own exact segment — disjoint writers.
                    unsafe { atxw.write_block(0, sbx4, 1, na, &buf[..na]) };
                }
                let nl = 16usize.min(h4.saturating_sub(sby4));
                if nl > 0 {
                    let mut buf = [0u8; 16];
                    for (dst, &src) in buf.iter_mut().zip(&t.ctx.l_tx[sby4..sby4 + nl]) {
                        *dst = src as u8;
                    }
                    // SAFETY: own exact segment — disjoint writers.
                    unsafe { ltxw.write_block(0, sby4, 1, nl, &buf[..nl]) };
                }
            }
            let x4 = sb_x / 4;
            let y4 = sb_y / 4;
            let stride4 = t.cfg.w / 4;
            let bw4 = 16usize.min(stride4.saturating_sub(x4));
            let bh4 = 16usize.min((t.cfg.h / 4).saturating_sub(y4));
            let x8 = sb_x / 8;
            let y8 = sb_y / 8;
            let stride8 = t.cfg.w.div_ceil(8);
            let bw8 = 8usize.min(stride8.saturating_sub(x8));
            let bh8 = 8usize.min(t.cfg.h.div_ceil(8).saturating_sub(y8));
            // SAFETY: every writer covers only this cell's disjoint map block.
            unsafe {
                skip8w.copy_region_from(&t.ctx.maps.skip8, stride8, y8, x8, bh8, bw8);
                blk4w.copy_region_from(&t.ctx.maps.blk4, stride4, y4, x4, bh4, bw4);
                blk4hw.copy_region_from(&t.ctx.maps.blk4h, stride4, y4, x4, bh4, bw4);
                blk4vw.copy_region_from(&t.ctx.maps.blk4v, stride4, y4, x4, bh4, bw4);
                blk4tw.copy_region_from(&t.ctx.maps.blk4t, stride4, y4, x4, bh4, bw4);
                pblk4w.copy_region_from(&t.ctx.maps.pblk4, stride4, y4, x4, bh4, bw4);
                pblk4hw.copy_region_from(&t.ctx.maps.pblk4h, stride4, y4, x4, bh4, bw4);
                pblk4vw.copy_region_from(&t.ctx.maps.pblk4v, stride4, y4, x4, bh4, bw4);
                pblk4tw.copy_region_from(&t.ctx.maps.pblk4t, stride4, y4, x4, bh4, bw4);
            }
            let cell = CapturedSb {
                tokens: t.ent.enc.take_semantic(),
            };
            if warm_decisions && row == 0 && col < warm_prefix {
                let mut model = warm_dec_cdfs.lock().expect("decision CDF warmup poisoned");
                let mut sink = OdEcEncoder::new();
                sink.sink = true;
                let mut slots = model.semantic_slots();
                sink.replay_semantic(&cell.tokens, &mut slots);
                if col + 1 == warm_prefix {
                    let _ = published_dec_cdfs.set(std::sync::Arc::new(model.clone()));
                }
            }
            tx.send((row * sb_cols + col, cell))
                .expect("wavefront entropy receiver dropped");
        },
    );
    WavefrontPlanes {
        recon: done,
        maps: BlockMaps {
            skip8,
            blk4,
            blk4h,
            blk4v,
            blk4t,
            pblk4,
            pblk4h,
            pblk4v,
            pblk4t,
        },
    }
}

fn pack_wavefront_tile(
    base_q_idx: u8,
    r: &TileRect,
    record: bool,
    updating_cdf: bool,
    rx: &std::sync::mpsc::Receiver<(usize, CapturedSb)>,
) -> TileOut {
    let sb_cols = r.tw.div_ceil(64);
    let sb_rows = r.th.div_ceil(64);
    let sb_count = sb_cols * sb_rows;
    let mut pending: Vec<Option<CapturedSb>> = (0..sb_count).map(|_| None).collect();

    let mut enc = OdEcEncoder::new().with_updating_cdf(updating_cdf);
    if record {
        enc.begin_trace();
    }
    let mut cdfs = Cdfs::new(crate::coef_q::qcat(base_q_idx));
    let mut slots = cdfs.semantic_slots();

    for sb_i in 0..sb_count {
        while pending[sb_i].is_none() {
            let (i, cell) = rx.recv().expect("wavefront capture stopped early");
            assert!(i < pending.len(), "wavefront cell index out of range");
            assert!(pending[i].is_none(), "duplicate wavefront cell");
            pending[i] = Some(cell);
        }
        let cell = pending[sb_i].take().unwrap();

        enc.trace_mark();
        enc.replay_semantic(&cell.tokens, &mut slots);
    }

    let trace = enc.take_trace();
    let payload = enc.done();
    TileOut {
        payload,
        trace,
        recon: Default::default(),
        maps: BlockMaps::default(),
    }
}
