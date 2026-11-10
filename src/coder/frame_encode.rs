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

// Frame-level lossy driver: `encode_lossy_tilegroup`, tile-group and frame OBU assembly.

/// `encode_lossy_tilegroup` output: tile-group bytes, tiling plan, CDEF and
/// loop-restoration header params, `allow_intrabc`, and the searched deblock
/// levels (`None` = the q-law).
pub(crate) type LossyTilegroup = (
    Vec<u8>,
    Tiling,
    Option<crate::obu::CdefParams>,
    Option<crate::obu::LrParams>,
    bool,
    Option<(i32, i32, i32)>,
);

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_tilegroup(
    base_q_idx: u8,
    bd: u8,
    w8: usize,
    h8: usize,
    disp_w: usize,
    disp_h: usize,
    src: &[Vec<u16>; 3],
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    context: &EncodingContext<'_>,
    aq: bool,
    cdef_on: bool,
    wiener_on: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc_allowed: bool,
) -> LossyTilegroup {
    let pool = context.thread_pool;
    let speed = context.speed;
    let vb = &context.boost;
    let dct = context.dct_dispatch();
    let idct = context.idct_dispatch();
    let intrapred = context.intrapred;
    let kmeans = context.kmeans;
    let rd = context.rd;
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;

    let want = pool.width();
    // Always the spec-minimum tiling (one tile unless the frame is wider than
    // MAX_TILE_WIDTH), parallelized by the SB wavefront inside each tile.
    let plan = plan_tiling(sb_cols, sb_rows, 1);
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    let (cw8, ch8) = (w8 >> sub_x, h8 >> sub_y);

    // Tile rectangles in raster order (top-to-bottom, left-to-right).
    let mut rects: Vec<TileRect> = Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        let th = y1 - y0;
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            let tw = x1 - x0;
            rects.push(TileRect {
                x0,
                y0,
                tw,
                th,
                cx0: x0 >> sub_x,
                cy0: y0 >> sub_y,
                ctw: tw >> sub_x,
                cth: th >> sub_y,
            });
        }
    }

    // Screen-content / IntraBC frame decision: a rate model over the frame's
    // texture mass, see `screen_frame_gate`. Measured whenever either consumer
    // is on: it also gates the text palette tools (`LossyTile::screen_frame`),
    // so that half must not depend on `--no-intrabc`.
    let gate = if !mono && (screen_content || intrabc_allowed) {
        screen_frame_gate(src, w8, h8, sub_x, sub_y, bd, base_q_idx, pool)
    } else {
        ScreenGate::default()
    };
    let allow_intrabc = intrabc_allowed && gate.ibc_pays;
    let screen_frame = screen_content && (gate.ibc_pays || gate.few_colour);

    let n = rects.len();
    let nthreads = want.clamp(1, n.max(1));

    // SB-wavefront: parallelize WITHIN each tile — parallel decision capture
    // (d = 2r+c wavefront) plus a lightweight raster entropy lane,
    // byte-identical per tile.
    // Tiles are then encoded SEQUENTIALLY with the full thread budget inside
    // each (wavefront capture spawns its own scoped workers; nesting it under
    // the tile pool would oversubscribe — same no-nesting rule as the AV2
    // wavefront). Wide frames get mandatory column tiles (MAX_TILE_WIDTH), so
    // per-tile is the only shape that covers them.
    let wf_threads = if want > 1 { want } else { 0 };

    // Recording the symbol trace lets a winning Wiener unit or a per-unit CDEF
    // grid be signaled by a cheap replay instead of a second full encode of
    // every tile.
    let record = !allow_intrabc && (wiener_on || cdef_on) && base_q_idx != 0;
    let job = TileJob {
        base_q_idx,
        bd,
        full_w: w8,
        full_h: h8,
        disp: (disp_w, disp_h),
        cw8,
        chroma: Chroma::from_sub(mono, sub_x, sub_y),
        speed,
        k: Kernels {
            dct,
            idct,
            intrapred,
            kmeans,
            rd,
        },
        aq,
        vb,
        record,
        allow_intrabc,
        updating_cdf,
        screen_content,
        screen_frame,
    };
    let mut outs: Vec<TileOut> = if wf_threads > 1 && n < want {
        // Split the budget: `outer` tiles in flight × `inner`-wide wavefront
        // inside each (outer*inner == want, no oversubscription — the inner
        // scoped workers replace their outer worker's idle time while other
        // tiles are packed). Single-tile frames get the full
        // budget inside the one tile.
        // Capture/packing pipelining needs at least three lanes per tile: one
        // packing lane and two decision lanes. If there are already at least
        // as many geometry-stable tiles as workers, the branch below simply
        // runs those tiles directly without an unnecessary inner wavefront.
        let outer = n.min((wf_threads / 3).max(1));
        let inner = (wf_threads / outer).max(2);
        pool.map_indexed(outer, n, |i| encode_one_tile(&job, src, &rects[i], inner))
    } else {
        pool.map_indexed(nthreads, n, |i| encode_one_tile(&job, src, &rects[i], 0))
    };

    let mut payloads: Vec<Vec<u8>> = outs
        .iter_mut()
        .map(|o| std::mem::take(&mut o.payload))
        .collect();
    let traces: Vec<_> = outs.iter_mut().map(|o| o.trace.take()).collect();

    let sb8w = w8.div_ceil(8);
    let sb8h = h8.div_ceil(8);
    let nc4f = w8 / 4;
    let nr4f = h8 / 4;
    let single_tile = outs.len() == 1;
    let WavefrontPlanes {
        mut recon,
        maps:
            BlockMaps {
                mut skip8,
                blk4: mut blk4f,
                blk4h: mut blk4hf,
                blk4v: mut blk4vf,
                blk4t: mut blk4tf,
                pblk4: mut pblk4f,
                pblk4h: mut pblk4hf,
                pblk4v: mut pblk4vf,
                pblk4t: mut pblk4tf,
            },
    } = if single_tile {
        // The tile already owns a full-frame reconstruction and filter maps.
        // Transfer them directly instead of allocating and stitching a copy.
        let out = outs.pop().unwrap();
        WavefrontPlanes {
            recon: out.recon,
            maps: out.maps,
        }
    } else {
        // Assemble all tiles before frame filters run across tile boundaries.
        WavefrontPlanes {
            recon: if mono {
                [vec![0u16; w8 * h8], Vec::new(), Vec::new()]
            } else {
                [
                    vec![0u16; w8 * h8],
                    vec![0u16; cw8 * ch8],
                    vec![0u16; cw8 * ch8],
                ]
            },
            maps: BlockMaps::new(w8, h8),
        }
    };
    for (r, out) in rects.iter().zip(outs.iter()) {
        let tsb8w = r.tw.div_ceil(8);
        let (ox8, oy8) = (r.x0 / 8, r.y0 / 8);
        for ty in 0..r.th.div_ceil(8) {
            for tx in 0..tsb8w {
                let (fx, fy) = (ox8 + tx, oy8 + ty);
                if fx < sb8w && fy < sb8h {
                    skip8[fy * sb8w + fx] = out.maps.skip8[ty * tsb8w + tx];
                }
            }
        }
        let tnc4 = r.tw / 4;
        let (ox4, oy4) = (r.x0 / 4, r.y0 / 4);
        for ty in 0..(r.th / 4) {
            for tx in 0..tnc4 {
                let (fx, fy) = (ox4 + tx, oy4 + ty);
                if fx < nc4f && fy < nr4f {
                    blk4f[fy * nc4f + fx] = out.maps.blk4[ty * tnc4 + tx];
                    blk4hf[fy * nc4f + fx] = out.maps.blk4h[ty * tnc4 + tx];
                    blk4vf[fy * nc4f + fx] = out.maps.blk4v[ty * tnc4 + tx];
                    blk4tf[fy * nc4f + fx] = out.maps.blk4t[ty * tnc4 + tx];
                    pblk4f[fy * nc4f + fx] = out.maps.pblk4[ty * tnc4 + tx];
                    pblk4hf[fy * nc4f + fx] = out.maps.pblk4h[ty * tnc4 + tx];
                    pblk4vf[fy * nc4f + fx] = out.maps.pblk4v[ty * tnc4 + tx];
                    pblk4tf[fy * nc4f + fx] = out.maps.pblk4t[ty * tnc4 + tx];
                }
            }
        }
    }

    // Pixel planes: every tile row owns a disjoint horizontal band of each
    // plane, so (plane, tile row) pairs stitch in parallel.
    let ncols = col_starts.len();
    if !single_tile {
        let mut items: Vec<(usize, usize, &mut [u16])> = Vec::new();
        for (pl, plane) in recon.iter_mut().enumerate() {
            if plane.is_empty() {
                continue;
            }
            let pw = if pl == 0 { w8 } else { cw8 };
            let mut rest = &mut plane[..];
            let mut consumed = 0usize;
            for ti in 0..row_starts.len() {
                let r0 = &rects[ti * ncols];
                let (py0, pth) = if pl == 0 {
                    (r0.y0, r0.th)
                } else {
                    (r0.cy0, r0.cth)
                };
                debug_assert_eq!(consumed, py0 * pw);
                let (band, r2) = std::mem::take(&mut rest).split_at_mut(pth * pw);
                rest = r2;
                consumed += band.len();
                items.push((pl, ti, band));
            }
        }
        pool.for_each(nthreads, items, |(pl, ti, band)| {
            for (r, out) in rects[ti * ncols..(ti + 1) * ncols]
                .iter()
                .zip(&outs[ti * ncols..(ti + 1) * ncols])
            {
                let (pw, px0, ptw, pth) = if pl == 0 {
                    (w8, r.x0, r.tw, r.th)
                } else {
                    (cw8, r.cx0, r.ctw, r.cth)
                };
                stitch_plane(band, pw, px0, 0, &out.recon[pl], ptw, pth);
            }
        });
    }
    drop(outs);

    // Frame-level in-loop deblocking filter, applied once on the stitched
    // reconstruction so that inter-tile edges are filtered exactly as the
    // decoder does (deblocking is not tile-independent in AV1). `filter_plane`
    // is a no-op when the derived level is 0 (e.g. lossless).
    let mut lf_levels: Option<(i32, i32, i32)> = None;
    if !allow_intrabc {
        let (lvl_y, lvl_uv) = crate::obu::loop_filter_levels(base_q_idx, sub_x + sub_y);
        let sharp = crate::obu::loop_filter_sharpness(base_q_idx);
        let geom = DeblockGeom {
            w8,
            h8,
            cw8,
            ch8,
            disp_w,
            disp_h,
            blk4: &blk4f,
            blk4h: &blk4hf,
            blk4v: &blk4vf,
            blk4t: &blk4tf,
            pblk4: &pblk4f,
            pblk4h: &pblk4hf,
            pblk4v: &pblk4vf,
            pblk4t: &pblk4tf,
            nc4: nc4f,
            sub_x,
            sub_y,
            mono,
            sharp,
            bd,
        };
        let (lvl_y, lvl_u, lvl_v) = if crate::tuning::get().lf_search && base_q_idx != 0 {
            let l = frame_lf_search(
                &context.loopfilter,
                pool,
                &rd,
                &recon,
                src,
                &geom,
                lvl_y,
                lvl_uv,
            );
            lf_levels = Some(l);
            l
        } else {
            (lvl_y, lvl_uv, lvl_uv)
        };
        frame_deblock(
            &context.loopfilter,
            pool,
            &mut recon,
            &geom,
            lvl_y,
            lvl_u,
            lvl_v,
        );
    }

    // Frame-level CDEF (R-D searched; may pick per-64x64-unit signaling).
    let cdef_decision = if !allow_intrabc && cdef_on && base_q_idx != 0 {
        frame_cdef(
            &context.cdef,
            &frame_decision_rd(&rd, sub_x, sub_y, mono),
            &mut recon,
            src,
            &skip8,
            sb8w,
            w8,
            h8,
            cw8,
            ch8,
            disp_w,
            disp_h,
            sub_x,
            sub_y,
            mono,
            base_q_idx,
            bd,
            speed,
            pool,
        )
    } else {
        None
    };

    // Frame-level luma Wiener loop restoration (searched on the CDEF-filtered
    // recon, matching the decoder's filter order).
    let lr_unit = if !allow_intrabc && wiener_on && base_q_idx != 0 {
        frame_wiener_search(
            &recon[0],
            &src[0],
            w8,
            h8,
            bd,
            pool,
            &frame_decision_rd(&rd, sub_x, sub_y, mono),
            frame_lambda_d(base_q_idx, bd, sub_x, sub_y),
        )
    } else {
        None
    };

    // Signal the winning filter syntax by replaying each tile's recorded
    // symbols with the read_lr symbols / per-unit cdef_idx literals
    // interleaved (byte-identical to a full re-encode; recon is unchanged
    // because both only add their own symbols).
    let cdef_replay = cdef_decision
        .as_ref()
        .filter(|d| d.params.bits > 0)
        .map(|d| CdefReplay {
            grid: &d.grid,
            unit_cols: d.unit_cols,
            bits: d.params.bits as u32,
        });
    if lr_unit.is_some() || cdef_replay.is_some() {
        payloads = pool.map_indexed(nthreads, n, |i| {
            let trace = traces[i]
                .as_deref()
                .expect("trace recorded for filter replay");
            replay_tile_with_filters(
                &rects[i],
                trace,
                lr_unit.as_ref(),
                cdef_replay.as_ref(),
                w8,
                h8,
                updating_cdf,
            )
        });
    }
    let lr = lr_unit.map(|_| crate::obu::LrParams { luma_wiener: true });
    let tilegroup = assemble_tilegroup(payloads);
    (
        tilegroup,
        plan,
        cdef_decision.map(|d| d.params),
        lr,
        allow_intrabc,
        lf_levels,
    )
}

/// Concatenate per-tile payloads into a tile-group. A single tile is returned
/// verbatim (no header byte, no size prefix). For `NumTiles > 1` the spec
/// `tile_group_obu` prepends `tile_start_and_end_present_flag = 0` followed by
/// `byte_alignment()` (one `0x00` byte), then every tile except the last is
/// prefixed with `tile_size_minus_1` as `TileSizeBytes = 4` little-endian bytes.
fn assemble_tilegroup(payloads: Vec<Vec<u8>>) -> Vec<u8> {
    if payloads.len() == 1 {
        return payloads.into_iter().next().unwrap();
    }
    let mut out = Vec::new();
    out.push(0u8);
    let last = payloads.len() - 1;
    for (i, p) in payloads.iter().enumerate() {
        if i != last {
            let sz_minus_1 = (p.len() - 1) as u32; // TileSizeBytes = 4
            out.extend_from_slice(&sz_minus_1.to_le_bytes());
        }
        out.extend_from_slice(p);
    }
    out
}

/// Build the frame OBU(s) that follow the sequence header. A single tile is
/// emitted as one combined `OBU_FRAME` (type 6) — byte-identical to the previous
/// output. Multi-tile frames are emitted as a separate `OBU_FRAME_HEADER`
/// (type 3) + `OBU_TILE_GROUP` (type 4), which strict parsers (ffmpeg's
/// `av1_frame_merge` BSF) handle reliably where a multi-tile combined
/// `OBU_FRAME` does not.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_frame_obus(
    base_q_idx: u8,
    qm: QmLevels,
    uac_delta: i32,
    udc_delta: i32,
    plan: &Tiling,
    tilegroup: &[u8],
    mono: bool,
    sub: usize,
    aq: bool,
    allow_intrabc: bool,
    cdef: Option<&crate::obu::CdefParams>,
    lr: Option<&crate::obu::LrParams>,
    updating_cdf: bool,
    lf: Option<(i32, i32, i32)>,
) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = frame_header_lossy_multitile_th(
            base_q_idx,
            qm,
            uac_delta,
            udc_delta,
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            mono,
            sub,
            aq,
            allow_intrabc,
            cdef,
            lr,
            updating_cdf,
            lf,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = frame_header_lossy_multitile(
            base_q_idx,
            qm,
            uac_delta,
            udc_delta,
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            mono,
            sub,
            aq,
            allow_intrabc,
            cdef,
            lr,
            updating_cdf,
            lf,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}
