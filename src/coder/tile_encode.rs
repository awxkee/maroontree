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

// Per-tile lossy driver: loop-restoration syntax, `TileRect` / `TileOut`, `encode_one_tile` and the filter-syntax replay.

/// Default (untrained) inverse CDF for the `use_wiener` flag, shared by tile
/// entropy init and the LR replay path.
pub(crate) fn wiener_restore_icdf() -> Vec<u16> {
    icdf(&[11570])
}

/// `read_lr` symbols owed by the superblock at tile-local `(sb_x, sb_y)` (spec
/// 5.11.57); geometry rationale on `LossyTile::emit_lr_sb`.
#[allow(clippy::too_many_arguments)]
fn emit_lr_sb_syms(
    enc: &mut OdEcEncoder,
    wr_cdf: &mut [u16],
    lr_ref_v: &mut [i32; 3],
    lr_ref_h: &mut [i32; 3],
    unit: &crate::wiener::WienerUnit,
    frame_x0: usize,
    frame_y0: usize,
    frame_w: usize,
    frame_h: usize,
    sb_x: usize,
    sb_y: usize,
) {
    const UNIT: usize = 64;
    const MI: usize = 4;
    let count_units = |frame: usize| -> usize { 1.max((frame + (UNIT >> 1)) / UNIT) };
    let unit_rows = count_units(frame_h);
    let unit_cols = count_units(frame_w);
    // Frame-absolute superblock position in 4x4 MI units (luma).
    let r = (frame_y0 + sb_y) / MI;
    let c = (frame_x0 + sb_x) / MI;
    let sb_mi = UNIT / MI; // 16
    let urs = (r * MI).div_ceil(UNIT);
    let ure = unit_rows.min(((r + sb_mi) * MI).div_ceil(UNIT));
    let ucs = (c * MI).div_ceil(UNIT);
    let uce = unit_cols.min(((c + sb_mi) * MI).div_ceil(UNIT));
    for _ur in urs..ure {
        for _uc in ucs..uce {
            emit_lr_unit_syms(enc, wr_cdf, lr_ref_v, lr_ref_h, unit);
        }
    }
}

/// One `read_lr_unit` for a RESTORE_WIENER luma unit (spec 5.11.58): `use_wiener`,
/// then v/h taps signed-subexp coded against (and updating) the running refs.
fn emit_lr_unit_syms(
    enc: &mut OdEcEncoder,
    wr_cdf: &mut [u16],
    lr_ref_v: &mut [i32; 3],
    lr_ref_h: &mut [i32; 3],
    unit: &crate::wiener::WienerUnit,
) {
    use crate::wiener::{WIENER_TAPS_K, WIENER_TAPS_MAX, WIENER_TAPS_MIN};
    enc.encode_symbol(1, wr_cdf);
    for axis in 0..2 {
        let (taps, refs) = if axis == 0 {
            (unit.v, &mut *lr_ref_v)
        } else {
            (unit.h, &mut *lr_ref_h)
        };
        for j in 0..3usize {
            let lo = WIENER_TAPS_MIN[j];
            let hi = WIENER_TAPS_MAX[j] + 1; // exclusive high
            let k = WIENER_TAPS_K[j] as u32;
            enc.encode_signed_subexp_with_ref(taps[j], lo, hi, k, refs[j]);
            refs[j] = taps[j];
        }
    }
}

/// Pixel rectangle of one tile, in both luma and (subsampled) chroma coords.
#[derive(Clone, Copy)]
struct TileRect {
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
    cx0: usize,
    cy0: usize,
    ctw: usize,
    cth: usize,
}

/// Encoded output of one tile: its entropy-coded payload plus the tile-local
/// reconstruction (luma `tw*th`, chroma `ctw*cth`). Owned + `Send`, so it can be
/// produced on a worker thread and moved back to the caller.
struct TileOut {
    payload: Vec<u8>,
    trace: Option<Box<crate::odec::SymbolTrace>>,
    recon: [Vec<u16>; 3],
    /// Tile-local block maps, for the frame-level deblocking and CDEF passes.
    maps: BlockMaps,
}

/// Everything that is the same for every tile of one lossy frame encode.
struct TileJob<'v> {
    base_q_idx: u8,
    bd: u8,
    /// 8-aligned coded frame luma size.
    full_w: usize,
    full_h: usize,
    /// Displayed frame luma size.
    disp: (usize, usize),
    /// Coded chroma plane width.
    cw8: usize,
    chroma: Chroma,
    speed: Speed,
    k: Kernels,
    aq: bool,
    vb: &'v VarianceBoost,
    /// Record the symbol trace (needed to interleave frame-level filter syntax).
    record: bool,
    allow_intrabc: bool,
    updating_cdf: bool,
    screen_content: bool,
    screen_frame: bool,
}

impl TileJob<'_> {
    /// The one place a lossy tile's state is built and configured: the serial
    /// driver and every wavefront worker get an identical tile from here.
    /// `ref_act` is the tile's mean activity when AQ is on (`None` = AQ off).
    fn build_tile<'a>(
        &self,
        tsrc: &'a [Vec<u16>; 3],
        r: &TileRect,
        ibc_index: Option<&'a std::sync::OnceLock<LossyIbcIndex>>,
        ref_act: Option<f32>,
    ) -> LossyTile<'a> {
        let (disp, vb) = (self.disp, self.vb);
        let mut tile = LossyTile::new(
            self.chroma,
            self.base_q_idx,
            self.bd,
            r.tw,
            r.th,
            tsrc,
            vb.qm,
        )
        .with_dispatch(
            self.k.dct,
            self.k.idct,
            self.k.intrapred,
            self.k.kmeans,
            self.k.rd,
        )
        .with_speed(self.speed)
        .with_visible(disp.0.saturating_sub(r.x0), disp.1.saturating_sub(r.y0))
        .with_intrabc(self.allow_intrabc, ibc_index)
        .with_screen_content(self.screen_content)
        .with_updating_cdf(self.updating_cdf);
        tile.cfg.mono_large = self.chroma == Chroma::Mono && vb.mono_large;
        tile.cfg.screen_frame = self.screen_frame;
        // Loop restoration is frame-relative: record this tile's frame-absolute luma
        // origin and the full frame luma size so `read_lr` is computed in frame
        // coordinates regardless of tiling.
        tile.cfg.frame_x0 = r.x0;
        tile.cfg.frame_y0 = r.y0;
        tile.cfg.frame_w = self.full_w;
        tile.cfg.frame_h = self.full_h;
        tile.cfg.ibc_end4 = (
            (tile.cfg.w / 4).min(disp.0.div_ceil(4).saturating_sub(r.x0 / 4)),
            (tile.cfg.h / 4).min(disp.1.div_ceil(4).saturating_sub(r.y0 / 4)),
        );
        if let Some(ref_act) = ref_act {
            tile.enable_aq(self.base_q_idx, ref_act, vb);
        }
        tile
    }

    /// The tile's mean activity, the zero-delta AQ reference: centering the
    /// per-SB deltas on it keeps the average quantizer at `base_q_idx`
    /// (zero-mean deltas => ~rate-neutral). `None` when AQ is off.
    fn ref_activity(&self, tsrc: &[Vec<u16>; 3], r: &TileRect) -> Option<f32> {
        self.aq
            .then(|| tile_ref_activity(&tsrc[0], r.tw, r.tw, r.th))
    }

    /// Whether decisions switch to the pilot-trained CDF snapshot. Only 4:4:4
    /// passed the extended unseen-image holdout. 4:2:0 averaged a +0.28%
    /// regression there, while 4:2:2 regressed synthetic content; mono has not
    /// been trained. Those three keep frame-initial pricing.
    fn warm_decisions(&self) -> bool {
        self.updating_cdf && self.chroma == Chroma::I444
    }
}

impl LossyTile<'_> {
    /// Code the superblock at luma (sb_x, sb_y): the per-superblock setup, the
    /// block walk, and the memo teardown. Shared by the serial raster loop and
    /// the wavefront cells so both run exactly the same sequence.
    ///
    /// `aq_cell` is the SB's precomputed AQ state (`None` = AQ off);
    /// `skip_cone` yields the all-zero statistics of the SBs finished before
    /// this one and is only called when the adaptive skip price is on, priced
    /// against `prior`. Returns this SB's own emitted all-zero counts.
    fn code_sb(
        &mut self,
        sb_x: usize,
        sb_y: usize,
        aq_cell: Option<&AqCell>,
        prior: &Cdfs,
        skip_cone: impl FnOnce() -> SkipStats,
    ) -> SkipStats {
        self.ctx.cdef_point_marked = false;
        if let Some(cell) = aq_cell {
            self.aq_begin_sb_cell(cell);
        }
        if self.adaptive_skip_price_enabled() {
            let cone = skip_cone();
            self.set_skip_price(&cone, prior);
        }
        let skip_stats0 = self.ent.cdfs.skip_stats;
        self.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false, None);
        self.clear_sb_rd_caches();
        skip_stats_sub(&self.ent.cdfs.skip_stats, &skip_stats0)
    }
}

/// Encode a single tile as an independent sub-frame. Pure function of its inputs
/// (no shared mutable state), so it is safe to run on any thread. At 4:0:0
/// only the luma plane is coded (`src[1]`/`src[2]` ignored, chroma recon empty).
/// `wf_threads > 1` runs the tile's superblocks as a wavefront.
fn encode_one_tile(
    job: &TileJob<'_>,
    src: &[Vec<u16>; 3],
    r: &TileRect,
    wf_threads: usize,
) -> TileOut {
    // A full-frame tile already has the required contiguous source layout.
    // Borrow it rather than duplicating every plane before worker allocation.
    let cropped_src = if r.x0 == 0 && r.y0 == 0 && r.tw == job.full_w && r.th == job.full_h {
        None
    } else if job.chroma == Chroma::Mono {
        Some([
            crop_plane(&src[0], job.full_w, r.x0, r.y0, r.tw, r.th),
            Vec::new(),
            Vec::new(),
        ])
    } else {
        Some([
            crop_plane(&src[0], job.full_w, r.x0, r.y0, r.tw, r.th),
            crop_plane(&src[1], job.cw8, r.cx0, r.cy0, r.ctw, r.cth),
            crop_plane(&src[2], job.cw8, r.cx0, r.cy0, r.ctw, r.cth),
        ])
    };
    let tsrc = cropped_src.as_ref().unwrap_or(src);
    // The IntraBC exact-match index is a pure function of the tile source, so
    // ONE cell is shared by every consumer instead of each building its own:
    // the wavefront spawns a `LossyTile` per worker, and all of them used to
    // rebuild the identical table (12 threads x 34 ms on a 3450x1900
    // screenshot — most of the multithreaded IntraBC regression). Still lazy,
    // because Fast prices no IntraBC candidate at all on many frames and must
    // not pay the build.
    let ibc_cell: std::sync::OnceLock<LossyIbcIndex> = std::sync::OnceLock::new();
    let ibc_index = job.allow_intrabc.then_some(&ibc_cell);
    if wf_threads <= 1 {
        return encode_tile_serial(job, tsrc, r, ibc_index);
    }
    // Pipeline the lightweight adaptive-CDF token packer behind parallel
    // capture. The packer does no block traversal or reconstruction.
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        // The entropy lane spends most of its life recv()-blocked on
        // capture, so capture keeps the full requested worker budget.
        let capture =
            scope.spawn(move || wavefront_capture(wf_threads, job, tsrc, r, ibc_index, tx));
        let mut packed = pack_wavefront_tile(job.base_q_idx, r, job.record, job.updating_cdf, &rx);
        let planes = capture.join().expect("wavefront capture panicked");
        packed.recon = planes.recon;
        packed.maps = planes.maps;
        packed
    })
}

/// One decide+emit pass over the tile's superblocks in raster order.
fn encode_tile_serial<'a>(
    job: &TileJob<'_>,
    tsrc: &'a [Vec<u16>; 3],
    r: &TileRect,
    ibc_index: Option<&'a std::sync::OnceLock<LossyIbcIndex>>,
) -> TileOut {
    let mut tile = job.build_tile(tsrc, r, ibc_index, job.ref_activity(tsrc, r));
    if job.record {
        tile.ent.enc.begin_trace();
    }
    // The whole per-SB AQ qindex sequence is a pure function of the source, so
    // it is precomputed up front (bit-exact vs the serial accumulator — see
    // `aq_grid_matches_serial`). This is what lets the wavefront decide SBs out
    // of raster order: each SB reads its cell instead of advancing shared
    // state. Empty when AQ is off.
    let aq_grid = tile.precompute_aq_grid();
    let sb_cols = r.tw.div_ceil(64);
    let sb_count = sb_cols * r.th.div_ceil(64);
    let warm_decisions = job.warm_decisions();
    let warm_prefix = sb_cols.min(4);
    let initial_dec_cdfs = tile.model.dec_cdfs.clone();
    let mut warm_dec_cdfs: Option<std::sync::Arc<Cdfs>> = None;
    let mut skip_prefix: Vec<SkipStats> = vec![[[[0u32; 2]; 13]; 5]; sb_count];
    let mut sb_i = 0usize;
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            let (sb_row, sb_col) = (sb_y / 64, sb_x / 64);
            let use_warm = warm_decisions && use_warmed_decision_cdf(sb_row, sb_col, warm_prefix);
            if use_warm != tile.model.decision_cdf_warmed {
                let next = if use_warm {
                    warm_dec_cdfs
                        .as_ref()
                        .expect("decision CDF warmup was not captured")
                        .clone()
                } else {
                    initial_dec_cdfs.clone()
                };
                tile.switch_decision_cdfs(next, use_warm);
            }
            // The mark sits exactly where a replay would interleave the LR
            // symbols owed by this superblock (`emit_lr_sb` is a no-op here).
            tile.ent.enc.trace_mark();
            tile.emit_lr_sb(sb_x, sb_y);
            let own = tile.code_sb(sb_x, sb_y, aq_grid.get(sb_i), &initial_dec_cdfs, || {
                skip_stats_cone(&skip_prefix, sb_row, sb_col, sb_cols)
            });
            skip_prefix_store(&mut skip_prefix, sb_row, sb_col, sb_cols, &own);
            if warm_decisions && sb_i + 1 == warm_prefix {
                let band_tilt = tile.model.dec_cdfs.band_tilt;
                let mut snapshot = tile.ent.cdfs.clone();
                snapshot.band_tilt = band_tilt;
                warm_dec_cdfs = Some(std::sync::Arc::new(snapshot));
            }
            sb_i += 1;
        }
    }
    // NOTE: the in-loop deblocking filter is deliberately NOT applied here.
    // In AV1 the deblocking filter is a frame-level post-filter that operates
    // ACROSS tile boundaries (only entropy coding and prediction are
    // tile-independent). Applying it per tile leaves the inter-tile edges
    // unfiltered, diverging from the decoder at every tile boundary. The filter
    // is instead applied once on the stitched frame in `encode_lossy_tilegroup`.
    // Intra prediction already used the unfiltered recon during `decode_sb`, so
    // deferring the filter does not change any coding decision.
    let trace = tile.ent.enc.take_trace();
    let payload = tile.ent.enc.done();
    TileOut {
        payload,
        trace,
        recon: std::mem::take(&mut tile.ctx.recon),
        maps: std::mem::take(&mut tile.ctx.maps),
    }
}

/// Per-64x64-unit CDEF replay context: the frame-level on/off grid plus its
/// column count. Each SB with a recorded `read_cdef()` point gets its unit's
/// index inserted there as a raw `L(1)` literal.
struct CdefReplay<'a> {
    grid: &'a [u8],
    unit_cols: usize,
    /// cdef_bits: width of the per-unit raw index literal (1..=3).
    bits: u32,
}

/// Replay a tile's recorded symbols with the frame-level filter syntax
/// interleaved: the Wiener `read_lr` symbols at each SB start (when `lr` is
/// set) and the per-unit `cdef_idx` literal at each SB's `read_cdef()` point
/// (when `cdef` is set). LR touches only its own CDF + raw bits and `cdef_idx`
/// is an equiprobable literal, so this is byte-identical to a re-encode.
fn replay_tile_with_filters(
    r: &TileRect,
    trace: &crate::odec::SymbolTrace,
    lr: Option<&crate::wiener::WienerUnit>,
    cdef: Option<&CdefReplay>,
    frame_w: usize,
    frame_h: usize,
    updating_cdf: bool,
) -> Vec<u8> {
    let mut enc = OdEcEncoder::new().with_updating_cdf(updating_cdf);
    let mut wr_cdf = wiener_restore_icdf();
    let mut lr_ref_v = crate::wiener::WIENER_TAPS_MID;
    let mut lr_ref_h = crate::wiener::WIENER_TAPS_MID;
    let mut i = 0usize;
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            if let Some(unit) = lr {
                emit_lr_sb_syms(
                    &mut enc,
                    &mut wr_cdf,
                    &mut lr_ref_v,
                    &mut lr_ref_h,
                    unit,
                    r.x0,
                    r.y0,
                    frame_w,
                    frame_h,
                    sb_x,
                    sb_y,
                );
            }
            match cdef {
                Some(c) => {
                    let (pre, post) = trace.sb_ops_split(i);
                    enc.replay(pre);
                    if let Some(post) = post {
                        // This SB has a non-skip block: its 64x64 unit carries a
                        // cdef_idx literal at the recorded read_cdef() point.
                        let u = ((r.y0 + sb_y) / 64) * c.unit_cols + (r.x0 + sb_x) / 64;
                        let idx = c.grid.get(u).copied().unwrap_or(0);
                        enc.encode_literal(idx as u32, c.bits);
                        enc.replay(post);
                    }
                }
                None => enc.replay(trace.sb_ops(i)),
            }
            i += 1;
        }
    }
    debug_assert_eq!(i, trace.sb_count(), "trace/SB iteration mismatch");
    enc.done()
}

/// Resolve the requested thread count: `0` => all available cores (fallback 1),
/// otherwise the value as-is. The caller still caps this at the tile count.
pub(crate) fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}
