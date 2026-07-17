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

use super::super::*;
use crate::av2::cfl::{cfl_partition_prediction, cfl_prediction};

#[derive(Clone, Copy)]
pub(super) struct Core422Planes<'a> {
    pub(super) y: &'a [f32],
    pub(super) u: &'a [f32],
    pub(super) v: &'a [f32],
}

/// One superblock's parallel-decide output for 4:2:2: the captured record. Merged
/// in raster order after the wavefront so the serial Replay consumes the exact
/// sequence the serial Capture would have produced.
struct WfSlot422 {
    record: replay::DecisionRecord,
}

/// Per-worker reusable scratch for the 4:2:2 wavefront decide: a private luma recon
/// (`ry`, full-res `pw×ph`) + two half-width chroma recons (`ru`/`rv`, `pcw×pch`)
/// plus the emit-side neighbour context arrays. Each cell sees the frame-initial
/// context values, reset per cell so the result never depends on which unrelated
/// cell a worker handled previously.
#[derive(Default)]
struct WfScratch422 {
    ry: Vec<f32>,
    ru: Vec<f32>,
    rv: Vec<f32>,
    above: Vec<u8>,
    left: Vec<u8>,
    apctx: Vec<u8>,
    midx: Vec<u8>,
    ua: Vec<i32>,
    va: Vec<i32>,
    ul: Vec<i32>,
    vl: Vec<i32>,
    cfa: Vec<i32>,
    cfl: Vec<i32>,
    dbq: Vec<u16>,
}

#[derive(Clone, Copy)]
struct WfScratch422Shape {
    pw: usize,
    ph: usize,
    pcw: usize,
    pch: usize,
    tmc: i64,
    tmr: i64,
    sb_cols: usize,
    sb_rows: usize,
}

struct Sb422Decision<'a, 'record> {
    enc: &'a mut RangeEncoder,
    aqs: &'a mut aq::AqState,
    decide_mode: &'a mut crate::av2::replay::DecideMode<'record>,
    yp: &'a [f32],
    up: &'a [f32],
    vp: &'a [f32],
    recy: &'a mut [f32],
    recu: &'a mut [f32],
    recv: &'a mut [f32],
    above: &'a mut [u8],
    left: &'a mut [u8],
    above_pctx: &'a mut [u8],
    left_pctx: &'a mut [u8],
    midx_grid: &'a mut [u8],
    u_above: &'a mut [i32],
    v_above: &'a mut [i32],
    u_left: &'a mut [i32],
    v_left: &'a mut [i32],
    cfl_above: &'a mut [i32],
    cfl_left: &'a mut [i32],
    db_sb_qidx: &'a mut [u16],
    mhccp_bounds: cfl::MhccpBounds,
    pw: usize,
    pcw: usize,
    width: usize,
    height: usize,
    sb_cols: usize,
    tmc: i64,
    tmr: i64,
    qstep_i: i32,
    needs_partition: bool,
    aq_grid: &'a [aq::AqCell],
    use_grid: bool,
    row: usize,
    col: usize,
}

impl WfScratch422 {
    fn ensure(&mut self, shape: WfScratch422Shape) {
        let WfScratch422Shape {
            pw,
            ph,
            pcw,
            pch,
            tmc,
            tmr,
            sb_cols,
            sb_rows,
        } = shape;
        let ny = pw * ph;
        if self.ry.len() != ny {
            self.ry = vec![0f32; ny];
        }
        let nc = pcw * pch + 1;
        if self.ru.len() != nc {
            self.ru = vec![0f32; nc];
            self.rv = vec![0f32; nc];
        }
        let (mc, mr) = (tmc as usize + 16, tmr as usize + 16);
        self.above.resize(pw / 4 + 16, 0x40);
        self.left.resize(ph / 4 + 16, 0x40);
        self.apctx.resize(mc, 0);
        self.midx.resize(sb_cols * sb_rows, 0xff);
        self.ua.resize(mc, 0);
        self.va.resize(mc, 0);
        self.ul.resize(mr, 0);
        self.vl.resize(mr, 0);
        self.cfa.resize(mc, 0);
        self.cfl.resize(mr, 0);
        self.dbq.resize(sb_cols * sb_rows, 0);
    }

    /// Restore the exact frame-initial context state before every independently
    /// decided SB (thread-pool assignment is unstable, so nothing may carry over).
    fn reset_contexts(&mut self) {
        self.above.fill(0x40);
        self.left.fill(0x40);
        self.apctx.fill(0);
        self.midx.fill(0xff);
        self.ua.fill(0);
        self.va.fill(0);
        self.ul.fill(0);
        self.vl.fill(0);
        self.cfa.fill(0);
        self.cfl.fill(0);
        self.dbq.fill(0);
    }
}

thread_local! {
    static WF_SCRATCH_422: std::cell::RefCell<WfScratch422> =
        std::cell::RefCell::new(WfScratch422::default());
}

impl Av2Encoder {
    /// SB-loop core for one 4:2:2 region (whole frame or one tile). Pads the region
    /// planes, runs the per-SB encode, and returns the entropy coder; header/finish
    /// (or multi-tile assembly) happens in the caller.
    pub(super) fn encode_422_core(
        &self,
        planes: Core422Planes<'_>,
        width: usize,
        height: usize,
        cdef_pre: Option<&cdef_est::CdefDecision>,
        // Staged replay: `Capture` logs the whole-64 luma mode-search winner;
        // `Replay` reuses it (skips `encode_luma_sb`). `Off` = today's behavior.
        mut decide_mode: replay::DecideMode<'_>,
    ) -> RangeEncoder {
        let Core422Planes {
            y: yf,
            u: cbf,
            v: crf,
        } = planes;
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph); // chroma: half width, full height
        let mhccp_bounds = cfl::MhccpBounds::from_luma(width, height, true, false);
        let yp = pad_plane(yf, width, height, pw, ph);
        let up = pad_plane(cbf, width.div_ceil(2), height, pcw, pch);
        let vp = pad_plane(crf, width.div_ceil(2), height, pcw, pch);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = true;
        enc.mhccp_ssy = false;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
        let qstep_i = quant::qstep(self.base_q_idx as u32) as i32;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        if let Some(pre) = cdef_pre {
            enc.cdef_nb = 2;
            enc.cdef_cols = pre.sb_cols;
            enc.cdef_grid = pre.grid.clone();
            enc.cdef_decided = Some(pre.clone());
        }
        // Per-SB directional midx (0xff = non-directional) for y-mode neighbor context.
        let mut midx_grid = vec![0xff_u8; sb_cols * sb_rows];
        // Native (no-pad) 4:2:2 luma extents for the 32-family; else pad to whole SBs.
        let native_mi = native_422_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        // Per-mi chroma neighbor presence, indexed in luma-mi space (the shared tree
        // drives both planes, so neighbor relations match the luma footprint).
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        // Per-mi CfL-usage neighbors for get_cfl_ctx.
        let mut cfl_above = vec![0i32; tmc as usize + 16];
        let mut cfl_left = vec![0i32; tmr as usize + 16];
        let needs_partition = native_mi.is_some() && lossy_needs_partition(width, height);
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = [0u8; 16];
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present && !needs_partition {
                aq::tile_ref_activity(&yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
            0, /* uv delta not yet wired for 4:2:2 */
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        )
        .with_dark_aq(self.tune.dark_aq);

        // Per-SB qindex for the deblock filter (captured after each SB's per_sb commit).
        let mut db_sb_qidx = vec![self.base_q_idx as u16; sb_rows * sb_cols];
        // Serial AQ pre-pass grid: the per-SB values a wavefront (out-of-raster-order)
        // decide reads instead of the serial `aqs` accumulator. Bit-exact with the
        // `per_sb`/`current` sequence for wavefront-eligible frames.
        let aq_grid = aqs.precompute_grid(&yp, pw, width, height, needs_partition);
        // The serial path retains its accumulated contexts. Wavefront workers reset
        // their private contexts per SB and read AQ from the precomputed grid.
        let fresh_ctx = false;
        // Halo geometry: intra reads a single reference line + above-right
        // extension of the top row; MHCCP/CfL read the same 1px border. Copy a generous
        // superset — a 32px perpendicular band + 64px above-right margin. Luma is
        // full-res (`pw`); chroma is HALF-WIDTH (`pcw`), so its x-origin is `sb_x/2` and
        // its own block is 32 wide (32×64 TX). This halo is the complete read-set
        // required to isolate a wavefront worker.
        const HALO_BAND: usize = 32;
        const HALO_AR: usize = 64;
        const POISON: f32 = 0.0;
        let halo_mode = false;
        let (mut recy_p, mut recu_p, mut recv_p) = if halo_mode {
            (
                vec![0f32; pw * ph],
                vec![0f32; pcw * pch + 1],
                vec![0f32; pcw * pch + 1],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let copy_halo_luma =
            |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize, copy_rd_anchor: bool| {
                let bx = sb_x.saturating_sub(HALO_BAND);
                let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                let by = sb_y.saturating_sub(HALO_BAND);
                for r in by..sb_y {
                    dst[r * pw + bx..r * pw + tx_end]
                        .copy_from_slice(&src[r * pw + bx..r * pw + tx_end]);
                }
                let ly_end = (sb_y + 64).min(ph);
                for r in sb_y..ly_end {
                    dst[r * pw + bx..r * pw + sb_x]
                        .copy_from_slice(&src[r * pw + bx..r * pw + sb_x]);
                }
                // Mirror the whole-64 RD path's full-plane `pixel_sse_rounded_block`
                // read at (0,0) (a shipping decision dependency; see 4:4:4).
                if copy_rd_anchor && (sb_y != 0 || sb_x != 0) {
                    for r in 0..64.min(ph) {
                        dst[r * pw..r * pw + 64].copy_from_slice(&src[r * pw..r * pw + 64]);
                    }
                }
            };
        let copy_halo_chroma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let cx = sb_x / 2;
            let bx = cx.saturating_sub(HALO_BAND);
            let tx_end = (cx + 32 + HALO_AR).min(pcw);
            let by = sb_y.saturating_sub(HALO_BAND);
            for r in by..sb_y {
                dst[r * pcw + bx..r * pcw + tx_end]
                    .copy_from_slice(&src[r * pcw + bx..r * pcw + tx_end]);
            }
            let ly_end = (sb_y + 64).min(pch);
            for r in sb_y..ly_end {
                dst[r * pcw + bx..r * pcw + cx].copy_from_slice(&src[r * pcw + bx..r * pcw + cx]);
            }
        };
        let copy_own_luma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let ly_end = (sb_y + 64).min(ph);
            for r in sb_y..ly_end {
                dst[r * pw + sb_x..r * pw + sb_x + 64]
                    .copy_from_slice(&src[r * pw + sb_x..r * pw + sb_x + 64]);
            }
        };
        let copy_own_chroma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let cx = sb_x / 2;
            let ly_end = (sb_y + 64).min(pch);
            for r in sb_y..ly_end {
                dst[r * pcw + cx..r * pcw + cx + 32]
                    .copy_from_slice(&src[r * pcw + cx..r * pcw + cx + 32]);
            }
        };
        // Wavefront gate: stills only (all 4:2:2), and only where the AQ grid is
        // bit-exact with the serial `aqs` for the active paths — i.e. non-padded frames
        // (all whole-64/split accumulate) OR padded frames above the split band
        // (base_q>38, walk uses base qstep, no accumulation). Padded base_q<=38 mixes
        // accumulating split leaves with non-accumulating walk leaves, which the grid
        // model does not reproduce, so it stays serial.
        // Default parallelism for stills: fires on any multithreaded single-tile
        // Capture pass. Suppressed inside per-tile sub-encodes, which already
        // parallelize.
        let wavefront = Self::resolve_threads(self.threads) > 1
            && !replay::in_tile_subencode()
            && !self.video_mode.load(std::sync::atomic::Ordering::Relaxed)
            && matches!(decide_mode, replay::DecideMode::Capture(_))
            && (!needs_partition || self.base_q_idx > 38);
        if wavefront {
            let enc_qc = enc.qc;
            let enc_cfl = enc.cfl;
            let enc_mhccp = enc.mhccp;
            let enc_dqp = enc.delta_q_present;
            let adaptive = self.tune.updating_cdf && self.base_q_idx != 0;
            let base_q = self.base_q_idx as i32;
            let nthreads = Self::resolve_threads(self.threads);
            let wy = helpers::PlaneWriter::new(&mut recy, pw);
            let wu = helpers::PlaneWriter::new(&mut recu, pcw);
            let wv = helpers::PlaneWriter::new(&mut recv, pcw);
            // Persistent-pool WPP wavefront: workers spawn once and loop over the
            // diagonals with a barrier between each, so each worker's thread_local
            // recon scratch is allocated once (not re-allocated per diagonal).
            let slots: Vec<Option<WfSlot422>> =
                helpers::par_wavefront_pool(nthreads, sb_rows, sb_cols, true, |r, c| {
                    let sb_y = r * 64;
                    let sb_x = c * 64;
                    let cx = sb_x / 2;

                    WF_SCRATCH_422.with(|sc| {
                        let s = &mut *sc.borrow_mut();
                        s.ensure(WfScratch422Shape {
                            pw,
                            ph,
                            pcw,
                            pch,
                            tmc,
                            tmr,
                            sb_cols,
                            sb_rows,
                        });
                        s.reset_contexts();
                        let ly_end = (sb_y + 64).min(ph);
                        let lyc_end = (sb_y + 64).min(pch);
                        // Luma halo geometry.
                        let bx = sb_x.saturating_sub(HALO_BAND);
                        let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                        let by = sb_y.saturating_sub(HALO_BAND);
                        // Chroma halo geometry (half-width plane).
                        let cbx = cx.saturating_sub(HALO_BAND);
                        let ctx_end = (cx + 32 + HALO_AR).min(pcw);
                        // The thread-local buffer is REUSED across cells; it must present
                        // decide the SAME all-zero-except-halo view the serial fresh plane
                        // does. It starts zero (in `ensure`) and each cell RE-ZEROES exactly
                        // what it dirtied (below), so on entry it is all-zero. Copy the halo
                        // (finished top/left neighbours) from the shared recon `out`.
                        // SAFETY: under WPP these regions are earlier-diagonal (finished),
                        // not being written concurrently; own-block writes are disjoint.
                        unsafe {
                            if sb_y != 0 || sb_x != 0 {
                                wy.copy_region_to(&mut s.ry, 0, 0, 64.min(ph), 64);
                            }
                            if sb_y > by {
                                wy.copy_region_to(&mut s.ry, by, bx, sb_y - by, tx_end - bx);
                            }
                            if ly_end > sb_y && sb_x > bx {
                                wy.copy_region_to(&mut s.ry, sb_y, bx, ly_end - sb_y, sb_x - bx);
                            }
                            if sb_y > by {
                                wu.copy_region_to(&mut s.ru, by, cbx, sb_y - by, ctx_end - cbx);
                                wv.copy_region_to(&mut s.rv, by, cbx, sb_y - by, ctx_end - cbx);
                            }
                            if lyc_end > sb_y && cx > cbx {
                                wu.copy_region_to(&mut s.ru, sb_y, cbx, lyc_end - sb_y, cx - cbx);
                                wv.copy_region_to(&mut s.rv, sb_y, cbx, lyc_end - sb_y, cx - cbx);
                            }
                        }
                        let mut e = RangeEncoder::new();
                        e.qc = enc_qc;
                        if adaptive {
                            e.enable_adaptive_cdf(enc_qc);
                        }
                        e.cfl = enc_cfl;
                        e.mhccp = enc_mhccp;
                        e.mhccp_ssx = true;
                        e.mhccp_ssy = false;
                        e.delta_q_present = enc_dqp;
                        let mut a = aq::AqState::new(enc_dqp, base_q, qstep_i, 0.0, 0);
                        let mut rec = replay::DecisionRecord::new();
                        let mut dm = replay::DecideMode::Capture(&mut rec);
                        let mut lpctx = [0u8; 16];
                        self.decide_sb_422(Sb422Decision {
                            enc: &mut e,
                            aqs: &mut a,
                            decide_mode: &mut dm,
                            yp: &yp,
                            up: &up,
                            vp: &vp,
                            recy: &mut s.ry,
                            recu: &mut s.ru,
                            recv: &mut s.rv,
                            above: &mut s.above,
                            left: &mut s.left,
                            above_pctx: &mut s.apctx,
                            left_pctx: &mut lpctx,
                            midx_grid: &mut s.midx,
                            u_above: &mut s.ua,
                            v_above: &mut s.va,
                            u_left: &mut s.ul,
                            v_left: &mut s.vl,
                            cfl_above: &mut s.cfa,
                            cfl_left: &mut s.cfl,
                            db_sb_qidx: &mut s.dbq,
                            mhccp_bounds,
                            pw,
                            pcw,
                            width,
                            height,
                            sb_cols,
                            tmc,
                            tmr,
                            qstep_i,
                            needs_partition,
                            aq_grid: &aq_grid,
                            use_grid: true,
                            row: r,
                            col: c,
                        });
                        // Write own blocks back to the shared recon. Luma 64×64 at
                        // (sb_y, sb_x); chroma 32×64 at (sb_y, cx). SAFETY: this SB's own
                        // block is disjoint from every other worker's.
                        let hl = ly_end - sb_y;
                        let hc = lyc_end - sb_y;
                        let mut blk = [0f32; 64 * 64];
                        let gather =
                            |plane: &[f32],
                             stride: usize,
                             x: usize,
                             w: usize,
                             h: usize,
                             blk: &mut [f32; 64 * 64]| {
                                for rr in 0..h {
                                    blk[rr * w..rr * w + w].copy_from_slice(
                                        &plane[(sb_y + rr) * stride + x
                                            ..(sb_y + rr) * stride + x + w],
                                    );
                                }
                            };
                        gather(&s.ry, pw, sb_x, 64, hl, &mut blk);
                        unsafe { wy.write_block(sb_y, sb_x, hl, 64, &blk[..hl * 64]) };
                        gather(&s.ru, pcw, cx, 32, hc, &mut blk);
                        unsafe { wu.write_block(sb_y, cx, hc, 32, &blk[..hc * 32]) };
                        gather(&s.rv, pcw, cx, 32, hc, &mut blk);
                        unsafe { wv.write_block(sb_y, cx, hc, 32, &blk[..hc * 32]) };
                        // Restore the buffer to all-zero: re-zero exactly what was dirtied
                        // (halo top+left bands + own block) so the next cell on this worker
                        // sees the all-zero-except-halo invariant.
                        for rr in by..ly_end {
                            s.ry[rr * pw + bx..rr * pw + tx_end].fill(0.0);
                        }
                        for rr in by..lyc_end {
                            s.ru[rr * pcw + cbx..rr * pcw + ctx_end].fill(0.0);
                            s.rv[rr * pcw + cbx..rr * pcw + ctx_end].fill(0.0);
                        }
                        if sb_y != 0 || sb_x != 0 {
                            for rr in 0..64.min(ph) {
                                s.ry[rr * pw..rr * pw + 64].fill(0.0);
                            }
                        }
                        WfSlot422 { record: rec }
                    })
                });
            // Merge per-SB records in raster order → the Capture record (the serial
            // Replay consumes the exact sequence a serial Capture would have logged).
            if let replay::DecideMode::Capture(rec) = &mut decide_mode {
                for slot in slots.into_iter() {
                    let slot = slot.expect("every SB decided");
                    rec.append(slot.record);
                }
            }
            for (i, cell) in aq_grid.iter().enumerate() {
                db_sb_qidx[i] = cell.qidx as u16;
            }
        } else {
            for row in 0..sb_rows {
                left_pctx.iter_mut().for_each(|p| *p = 0);
                for col in 0..sb_cols {
                    if fresh_ctx {
                        above.iter_mut().for_each(|v| *v = 0x40);
                        left.iter_mut().for_each(|v| *v = 0x40);
                        above_pctx.iter_mut().for_each(|v| *v = 0);
                        left_pctx.iter_mut().for_each(|v| *v = 0);
                        midx_grid.iter_mut().for_each(|v| *v = 0xff);
                        u_above.iter_mut().for_each(|v| *v = 0);
                        v_above.iter_mut().for_each(|v| *v = 0);
                        u_left.iter_mut().for_each(|v| *v = 0);
                        v_left.iter_mut().for_each(|v| *v = 0);
                        cfl_above.iter_mut().for_each(|v| *v = 0);
                        cfl_left.iter_mut().for_each(|v| *v = 0);
                    }
                    let sb_y = row * 64;
                    let sb_x = col * 64;
                    if halo_mode {
                        recy_p.iter_mut().for_each(|v| *v = POISON);
                        recu_p.iter_mut().for_each(|v| *v = POISON);
                        recv_p.iter_mut().for_each(|v| *v = POISON);
                        copy_halo_luma(&mut recy_p, &recy, sb_y, sb_x, true);
                        copy_halo_chroma(&mut recu_p, &recu, sb_y, sb_x);
                        copy_halo_chroma(&mut recv_p, &recv, sb_y, sb_x);
                    }
                    let (ry, ru, rv): (&mut [f32], &mut [f32], &mut [f32]) = if halo_mode {
                        (
                            recy_p.as_mut_slice(),
                            recu_p.as_mut_slice(),
                            recv_p.as_mut_slice(),
                        )
                    } else {
                        (
                            recy.as_mut_slice(),
                            recu.as_mut_slice(),
                            recv.as_mut_slice(),
                        )
                    };
                    self.decide_sb_422(Sb422Decision {
                        enc: &mut enc,
                        aqs: &mut aqs,
                        decide_mode: &mut decide_mode,
                        yp: &yp,
                        up: &up,
                        vp: &vp,
                        recy: ry,
                        recu: ru,
                        recv: rv,
                        above: &mut above,
                        left: &mut left,
                        above_pctx: &mut above_pctx,
                        left_pctx: &mut left_pctx,
                        midx_grid: &mut midx_grid,
                        u_above: &mut u_above,
                        v_above: &mut v_above,
                        u_left: &mut u_left,
                        v_left: &mut v_left,
                        cfl_above: &mut cfl_above,
                        cfl_left: &mut cfl_left,
                        db_sb_qidx: &mut db_sb_qidx,
                        mhccp_bounds,
                        pw,
                        pcw,
                        width,
                        height,
                        sb_cols,
                        tmc,
                        tmr,
                        qstep_i,
                        needs_partition,
                        aq_grid: &aq_grid,
                        use_grid: fresh_ctx,
                        row,
                        col,
                    });
                    if halo_mode {
                        copy_own_luma(&mut recy, &recy_p, sb_y, sb_x);
                        copy_own_chroma(&mut recu, &recu_p, sb_y, sb_x);
                        copy_own_chroma(&mut recv, &recv_p, sb_y, sb_x);
                    }
                }
            }
        }
        // Real deblocking (4:2:2). At deblock-active quality the RD split is off
        // (base_q>38 gate), so every full-interior SB is whole-64 -> one 64x64 luma
        // leaf per SB. Chroma is half-width/full-height (ssx=1, ssy=0) with 32x64 TX
        // (cap (8,16) chroma MI). Skips the edge-residue partition path for now.
        let df_quant = {
            let qs = crate::av2::quant::qstep(self.base_q_idx as u32) as i32;
            ((qs + 4) >> 3) >> 6
        };
        if self.tune.deblock && df_quant >= 1 && !needs_partition {
            let eff_dy = if self.tune.db_delta_y == i32::MIN {
                if df_quant >= 5 { 0 } else { 1 }
            } else {
                self.tune.db_delta_y
            };
            let mut leaves = Vec::with_capacity(sb_rows * sb_cols);
            for r in 0..sb_rows {
                for c in 0..sb_cols {
                    leaves.push((r * 16, c * 16, 16usize, 16usize));
                }
            }
            deblock::deblock_frame(deblock::FrameDeblock {
                recy: &mut recy,
                recu: &mut recu,
                recv: &mut recv,
                luma_stride: pw,
                chroma_stride: pcw,
                width,
                height,
                ssx: 1,
                ssy: 0,
                has_chroma: true,
                chroma_deblock: self.tune.chroma_deblock,
                bit_depth: self.bit_depth as u32,
                delta_y: eff_dy,
                delta_uv: self.tune.db_delta_uv,
                base_q: self.base_q_idx as u16,
                sb_cols,
                sb_qidx: &db_sb_qidx,
                leaves: &leaves,
                chroma_leaves: &leaves,
                skip_leaves: &[],
                luma_tx_cap: (8, 8),

                chroma_tx_cap: (8, 16),
            });
        }
        // Multi-tile CDEF stitches per-tile recon; capture it when asked.
        if self
            .capture_recon
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.recon = vec![recy.clone(), recu.clone(), recv.clone()];
        }
        // Pass 1 (single-tile whole-frame): derive the per-block CDEF decision. The
        // multi-tile path forces `capture_recon` and searches the stitched recon, so
        // skip here when recon is captured (avoids a redundant per-tile search).
        if self.tune.cdef
            && cdef_pre.is_none()
            && !self
                .capture_recon
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.cdef_decided = cdef_est::search_per_block(
                &[recy, recu, recv],
                &[yp, up, vp],
                pw,
                ph,
                pcw,
                pch,
                1,
                0,
                true,
                self.base_q_idx,
                self.bit_depth,
            );
        }
        enc
    }

    /// Decide + emit one 4:2:2 superblock (the extracted body of the `for col`
    /// loop). Behaviour-preserving; byte-identical. Split out so the SB-wavefront
    /// can drive it per cell with private scratch buffers.
    fn decide_sb_422(&self, decision: Sb422Decision<'_, '_>) {
        let Sb422Decision {
            enc,
            aqs,
            mut decide_mode,
            yp,
            up,
            vp,
            recy,
            recu,
            recv,
            above,
            left,
            above_pctx,
            left_pctx,
            midx_grid,
            u_above,
            v_above,
            u_left,
            v_left,
            cfl_above,
            cfl_left,
            db_sb_qidx,
            mhccp_bounds,
            pw,
            pcw,
            width,
            height,
            sb_cols,
            tmc,
            tmr,
            qstep_i,
            needs_partition,
            aq_grid,
            use_grid,
            row,
            col,
        } = decision;
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let cell = aq_grid[row * sb_cols + col];
        let sb_y = row * 64;
        let sb_x = col * 64;
        if enc.cdef_nb >= 2 {
            enc.cdef_pending = true;
            enc.cdef_sb_rc = (row, col);
        }
        let full_interior = sb_x + 64 <= width && sb_y + 64 <= height;
        let (probe_qstep, probe_scale) = if use_grid {
            (cell.qs_before, cell.resid_scale_before)
        } else {
            let (q, r, _, _) = aqs.current();
            (q, r)
        };
        let luma_partition = if full_interior && self.tune.chroma_split && self.base_q_idx <= 38 {
            choose_luma_64x64_partition(
                recy,
                &LumaSource {
                    plane: yp,
                    stride: pw,
                },
                &LumaFrameBlock {
                    frame_width: width,
                    frame_height: height,
                    y: sb_y,
                    x: sb_x,
                },
                &LumaGridBlock {
                    mi_cols: tmc,
                    mi_rows: tmr,
                    y: sb_y,
                    x: sb_x,
                },
                &LumaPartitionSearch {
                    quant: LumaQuantSpec {
                        basis: &bases.luma,
                        qstep: probe_qstep,
                        scan: &tables::SCAN,
                        neutral,
                        quant_context: qc,
                        rdoq_lambda: self.tune.rdoq_lambda,
                        speed: self.speed,
                        bit_depth: self.bit_depth as i32,
                    },
                    sb: LumaSbSearch {
                        residual_scale: probe_scale,
                        allow_directional: self.speed.try_directional(),
                    },
                    basis16: &bases.luma16x16,
                    basis8: &bases.c8x8,
                    allow_16x16: false,
                    allow_8x8: false,
                },
            )
        } else {
            LumaPartitionDecision::default()
        };
        let sb_rd_split = full_interior && luma_partition.split64;
        if !needs_partition && !sb_rd_split {
            // Whole 64×64 luma SB → one 32×64 (TX_32X64) chroma TU per plane.
            let (fmr, fmc) = (row * 16, col * 16);
            let (sb_qstep, sb_resid_scale, _, _) = if use_grid {
                enc.delta_q_signaled = cell.sig;
                (cell.qs, cell.resid_scale, cell.qs_c, cell.resid_scale_c)
            } else {
                aqs.per_sb(enc, yp, pw, sb_y, sb_x, width, height)
            };
            db_sb_qidx[row * sb_cols + col] = if use_grid {
                cell.qidx as u16
            } else {
                aqs.current_qidx() as u16
            };
            // Match avm get_entropy_context_1d: ANY nonzero along the block's
            // above/left edge (whole 64-wide SB span), not just the first unit.
            let any = |g: &[i32], s: usize, n: usize| {
                g[s..(s + n).min(g.len())].iter().any(|&x| x != 0) as i32
            };
            let ua = if fmr > 0 { any(u_above, fmc, 16) } else { 0 };
            let ul = if fmc > 0 { any(u_left, fmr, 16) } else { 0 };
            let va = if fmr > 0 { any(v_above, fmc, 16) } else { 0 };
            let vl = if fmc > 0 { any(v_left, fmr, 16) } else { 0 };
            // Staged replay: capture/replay the whole-64 luma mode-search winner.
            let replay_l = if let crate::av2::replay::DecideMode::Replay(cur) = &mut decide_mode {
                cur.next_luma420()
            } else {
                None
            };
            let (tus, mode_idx, adelta) = if let Some(l) = replay_l {
                for r in 0..64 {
                    let off = (sb_y + r) * pw + sb_x;
                    recy[off..off + 64].copy_from_slice(&l.recon_y[r * 64..r * 64 + 64]);
                }
                (l.tus, l.mode_idx, l.adelta)
            } else {
                let (tus, mode_idx, adelta, _) = encode_luma_sb(
                    recy,
                    &LumaSource {
                        plane: yp,
                        stride: pw,
                    },
                    &LumaFrameBlock {
                        frame_width: width,
                        frame_height: height,
                        y: sb_y,
                        x: sb_x,
                    },
                    &LumaQuantSpec {
                        basis: &bases.luma,
                        qstep: sb_qstep,
                        scan: &tables::SCAN,
                        neutral,
                        quant_context: qc,
                        rdoq_lambda: self.tune.rdoq_lambda,
                        speed: self.speed,
                        bit_depth: self.bit_depth as i32,
                    },
                    &LumaSbSearch {
                        residual_scale: sb_resid_scale,
                        allow_directional: {
                            // Directional intra (Slow tier), whole 64x64 SBs only; partial
                            // edge SBs use the non-directional path. predict_luma reference
                            // availability now uses the true MI extent (decoder-matching).
                            self.speed.try_directional()
                                && sb_x + 64 <= width
                                && sb_y + 64 <= height
                        },
                    },
                );
                if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
                    let mut recon_y = Vec::with_capacity(64 * 64);
                    for r in 0..64 {
                        let off = (sb_y + r) * pw + sb_x;
                        recon_y.extend_from_slice(&recy[off..off + 64]);
                    }
                    rec.push_luma420(crate::av2::replay::Luma420 {
                        tus: tus.clone(),
                        mode_idx,
                        adelta,
                        recon_y,
                    });
                }
                (tus, mode_idx, adelta)
            };
            let (skip_cdfs, dc_sign_ctxs) =
                sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
            let (cy, cx) = (sb_y, sb_x / 2);
            let bd = self.bit_depth as i32;
            // CfL decision (4:2:2 whole-64 fast path). 32x64 chroma; luma subsampled
            // horizontally (<<2); neighbor DC via cfl_avg_l(ssx=true, ssy=false).
            let cfl_a = if fmr > 0 { cfl_above[fmc] } else { 0 };
            let cfl_l = if fmc > 0 { cfl_left[fmr] } else { 0 };
            enc.cfl_ctx = (cfl_a + cfl_l) as usize;
            // Staged replay: capture/replay the whole-64 chroma CfL decision so
            // the emit-only pass skips the `cfl_decide` RD search (pure emit).
            let replay_cfl = if let crate::av2::replay::DecideMode::Replay(cur) = &mut decide_mode {
                cur.next_chroma422_whole()
            } else {
                None
            };
            let cfl_choice = match replay_cfl {
                Some(c) => c,
                None => {
                    let c = if enc.cfl {
                        let avg_l = cfl::cfl_avg_l(recy, pw, sb_y, sb_x, 32, 64, true, false, bd);
                        let mut suf = [0f32; 32 * 64];
                        let mut svf = [0f32; 32 * 64];
                        for (((suf_row, svf_row), up_row), vp_row) in suf
                            .as_chunks_mut::<32>()
                            .0
                            .iter_mut()
                            .zip(svf.as_chunks_mut::<32>().0.iter_mut())
                            .zip(rect_rows(up, pcw, cy, cx, 32, 64))
                            .zip(rect_rows(vp, pcw, cy, cx, 32, 64))
                        {
                            suf_row.copy_from_slice(up_row);
                            svf_row.copy_from_slice(vp_row);
                        }
                        let dc_u_f =
                            dc_pred_rect_subsampled(recu, pcw, cy, cx, 32, 64, neutral, bd);
                        let dc_v_f =
                            dc_pred_rect_subsampled(recv, pcw, cy, cx, 32, 64, neutral, bd);
                        cfl::cfl_decide(
                            &cfl::CflDecisionInput {
                                reconstructed_luma: recy,
                                luma_stride: pw,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                source_u: &suf,
                                source_v: &svf,
                                dc_u: dc_u_f,
                                dc_v: dc_v_f,
                                width: 32,
                                height: 64,
                                subsample_x: true,
                                subsample_y: false,
                                luma_average_q3: avg_l,
                            },
                            &cfl::ChromaRdSpec {
                                basis: &bases.chroma422,
                                qstep: qstep_i,
                                lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                bit_depth: bd,
                            },
                        )
                    } else {
                        None
                    };
                    if let replay::DecideMode::Capture(rec) = &mut decide_mode {
                        rec.push_chroma422_whole(c.clone());
                    }
                    c
                }
            };
            if let Some(ref ch) = cfl_choice {
                enc.cfl_use = true;
                enc.cfl_js = ch.js;
                enc.cfl_mag_u = ch.mag_u;
                enc.cfl_mag_v = ch.mag_v;
                enc.cfl_ctx_u = ch.ctx_u;
                enc.cfl_ctx_v = ch.ctx_v;
            } else {
                enc.cfl_use = false;
                enc.cfl_signaled = false;
            }
            enc.delta_q_pending = enc.delta_q_present;
            let lmidx = if col > 0 {
                midx_grid[row * sb_cols + col - 1]
            } else {
                0xff
            };
            let amidx = if row > 0 {
                midx_grid[(row - 1) * sb_cols + col]
            } else {
                0xff
            };
            let (_cul, sb_midx) = encode_luma_block_split_dir(
                enc,
                &LumaSplitDirSpec {
                    tus: &tus,
                    skip_cdfs: &skip_cdfs,
                    dc_sign_ctxs: &dc_sign_ctxs,
                    mode_idx,
                    angle_delta: adelta,
                    has_chroma: true,
                    part_cdf: partition::sb_none_do_split_cdf(row, col, above_pctx, left_pctx),
                    left_midx: lmidx,
                    above_midx: amidx,
                },
            );
            midx_grid[row * sb_cols + col] = sb_midx;
            let scan = &tables::SCAN;
            let (levu, levv) = if let Some(ref ch) = cfl_choice {
                let mut ru = [0f32; 32 * 64];
                let mut rv = [0f32; 32 * 64];
                cfl_prediction::<32>(pcw, up, vp, cy, cx, &ch, &mut ru, &mut rv);
                let levu =
                    bases
                        .chroma422
                        .project_scan(&aq::scale_resid(&ru, sb_resid_scale), 0.0, scan);
                let levv =
                    bases
                        .chroma422
                        .project_scan(&aq::scale_resid(&rv, sb_resid_scale), 0.0, scan);
                put_block_rect(
                    recu,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &itx422::reconstruct_chroma_cfl(&ch.pred_u, &levu, sb_qstep, scan, 32, 64, bd),
                );
                put_block_rect(
                    recv,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &itx422::reconstruct_chroma_cfl(&ch.pred_v, &levv, sb_qstep, scan, 32, 64, bd),
                );
                (levu, levv)
            } else {
                let predu = dc_pred_rect(recu, pcw, cy, cx, 32, 64, neutral, bd);
                let levu = bases.chroma422.project_scan(
                    &aq::scale_resid(
                        &get_residual_rect(up, pcw, cy, cx, 32, 64, predu),
                        sb_resid_scale,
                    ),
                    0.0,
                    scan,
                );
                put_block_rect(
                    recu,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &recon_422_chroma(predu, &levu, sb_qstep, scan, 32, 64, &bases.chroma422, bd),
                );
                let predv = dc_pred_rect(recv, pcw, cy, cx, 32, 64, neutral, bd);
                let levv = bases.chroma422.project_scan(
                    &aq::scale_resid(
                        &get_residual_rect(vp, pcw, cy, cx, 32, 64, predv),
                        sb_resid_scale,
                    ),
                    0.0,
                    scan,
                );
                put_block_rect(
                    recv,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &recon_422_chroma(predv, &levv, sb_qstep, scan, 32, 64, &bases.chroma422, bd),
                );
                (levu, levv)
            };
            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
            let u_skip = CHROMA_SKIP_TX64_QC[qc][(6 + ua + ul) as usize] as u32;
            encode_chroma_block_rect(
                enc,
                &uc,
                u_skip,
                true,
                &tables::SCAN,
                EobCdf::ChrEobBin,
                CHROMA_EOB_HI_BIT_QC[qc],
                1024,
            );
            let up_ = uc.iter().any(|&(_, l)| l != 0);
            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
            encode_chroma_block_rect(
                enc,
                &vc,
                v_skip,
                false,
                &tables::SCAN,
                EobCdf::ChrEobBin,
                CHROMA_EOB_HI_BIT_QC[qc],
                1024,
            );
            let v_present = vc.iter().any(|&(_, l)| l != 0);
            let cfl_used = cfl_choice.is_some() as i32;
            for c in fmc..fmc + 16 {
                u_above[c] = up_ as i32;
                v_above[c] = v_present as i32;
                cfl_above[c] = cfl_used;
            }
            for r in fmr..fmr + 16 {
                u_left[r] = up_ as i32;
                v_left[r] = v_present as i32;
                cfl_left[r] = cfl_used;
            }
            partition::sb_none_pctx(row, col, above_pctx, left_pctx);
            return;
        }

        let qstep_i = if sb_rd_split {
            if use_grid {
                enc.delta_q_signaled = cell.sig;
                cell.qs
            } else {
                aqs.per_sb(enc, yp, pw, sb_y, sb_x, width, height).0
            }
        } else {
            qstep_i
        };
        let ops = if sb_rd_split {
            partition::sb_rd_split_ops(
                row,
                col,
                luma_partition.split32,
                luma_partition.split16,
                above_pctx,
                left_pctx,
            )
        } else {
            partition::sb_partition_ops(row, col, tmr as usize, tmc as usize, above_pctx, left_pctx)
        };
        // A full-interior RD-split SB quantizes its leaves at the AQ per-SB
        // qstep (set via `aqs.per_sb` above), so it must EMIT that same
        // per-SB delta_q — otherwise the decoder dequantizes at the base
        // qindex and every split SB reconstructs at the wrong scale (uniform
        // quality collapse). Only the neutral edge / non-split partition path
        // (base qstep, `aqs.per_sb` not called) forces delta_q = 0.
        if !sb_rd_split {
            enc.delta_q_signaled = 0;
        }
        enc.delta_q_pending = enc.delta_q_present;
        enc.y_ctx = 0;
        // Reset the per-SB coded-mi mask consumed by MHCCP (see y420).
        enc.sb_coded = [0u8; 256];
        for op in &ops {
            let (bw_mi, bh_mi, pc, lmr, lmc) = match op {
                partition::Op::RectType { cdf, val, ctx } => {
                    enc.bool_rect_type(*cdf, *val, *ctx);
                    continue;
                }
                partition::Op::Split {
                    do_split_cdf,
                    square_cdf,
                } => {
                    enc.bool_do_split(*do_split_cdf, 1);
                    if *square_cdf != 0 {
                        enc.bool_do_square_split(*square_cdf, 1);
                    }
                    continue;
                }
                partition::Op::Leaf {
                    bw_mi,
                    bh_mi,
                    part_cdf,
                    mi_row,
                    mi_col,
                } => (*bw_mi, *bh_mi, part_cdf.unwrap_or(12276), *mi_row, *mi_col),
            };
            let sb_y = lmr * 4;
            let sb_x = lmc * 4;
            // Decoder y-mode index context (get_y_mode_idx_ctx): a walk leaf
            // only sees a directional neighbor when it touches its SB's
            // top/left edge and that SB was coded whole with directional luma.
            enc.y_ctx = ((lmr % 16 == 0 && row > 0 && midx_grid[(row - 1) * sb_cols + col] != 0xff)
                as usize)
                + ((lmc % 16 == 0 && col > 0 && midx_grid[row * sb_cols + col - 1] != 0xff)
                    as usize);
            // ANY nonzero across the block's above/left span (avm
            // get_entropy_context_1d); single-entry reads desync the chroma
            // txb-skip context next to partitioned neighbors.
            let any = |g: &[i32], s: usize, n: usize| {
                g[s..(s + n).min(g.len())].iter().any(|&x| x != 0) as i32
            };
            let ua = if lmr > 0 { any(u_above, lmc, bw_mi) } else { 0 };
            let ul = if lmc > 0 { any(u_left, lmr, bh_mi) } else { 0 };
            let va = if lmr > 0 { any(v_above, lmc, bw_mi) } else { 0 };
            let vl = if lmc > 0 { any(v_left, lmr, bh_mi) } else { 0 };
            let (cy, cx) = (sb_y, sb_x / 2);
            let cfl_a = if lmr > 0 { cfl_above[lmc] } else { 0 };
            let cfl_l = if lmc > 0 { cfl_left[lmr] } else { 0 };
            enc.cfl_ctx = (cfl_a + cfl_l) as usize;
            enc.cfl_use = false;
            enc.cfl_signaled = false;
            enc.mhccp_use = false;
            let (u_present, v_present) = match (bw_mi, bh_mi) {
                (16, 16) => {
                    let (tus, mode_idx, _, _) = encode_luma_sb(
                        recy,
                        &LumaSource {
                            plane: yp,
                            stride: pw,
                        },
                        &LumaFrameBlock {
                            frame_width: width,
                            frame_height: height,
                            y: sb_y,
                            x: sb_x,
                        },
                        &LumaQuantSpec {
                            basis: &bases.luma,
                            qstep: qstep_i,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                        &LumaSbSearch {
                            residual_scale: 1.0,
                            allow_directional: false,
                        },
                    );
                    let (skip_cdfs, dc_sign_ctxs) =
                        sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
                    const PARTITION_CFL: bool = true;
                    let bd = self.bit_depth as i32;
                    let cfl_choice = if enc.cfl && PARTITION_CFL {
                        let avg_l = cfl::cfl_avg_l(recy, pw, sb_y, sb_x, 32, 64, true, false, bd);
                        let mut suf = [0f32; 32 * 64];
                        let mut svf = [0f32; 32 * 64];
                        cfl_partition_prediction::<32>(pcw, up, vp, cy, cx, &mut suf, &mut svf);
                        let dc_u_f =
                            dc_pred_rect_subsampled(recu, pcw, cy, cx, 32, 64, neutral, bd);
                        let dc_v_f =
                            dc_pred_rect_subsampled(recv, pcw, cy, cx, 32, 64, neutral, bd);
                        cfl::cfl_decide(
                            &cfl::CflDecisionInput {
                                reconstructed_luma: recy,
                                luma_stride: pw,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                source_u: &suf,
                                source_v: &svf,
                                dc_u: dc_u_f,
                                dc_v: dc_v_f,
                                width: 32,
                                height: 64,
                                subsample_x: true,
                                subsample_y: false,
                                luma_average_q3: avg_l,
                            },
                            &cfl::ChromaRdSpec {
                                basis: &bases.chroma422,
                                qstep: qstep_i,
                                lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                bit_depth: bd,
                            },
                        )
                    } else {
                        None
                    };
                    if let Some(ref ch) = cfl_choice {
                        enc.cfl_use = true;
                        enc.cfl_js = ch.js;
                        enc.cfl_mag_u = ch.mag_u;
                        enc.cfl_mag_v = ch.mag_v;
                        enc.cfl_ctx_u = ch.ctx_u;
                        enc.cfl_ctx_v = ch.ctx_v;
                    }
                    encode_luma_block_split(
                        enc,
                        &tus,
                        &skip_cdfs,
                        &dc_sign_ctxs,
                        mode_idx,
                        true,
                        pc,
                    );
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 32,
                            ch: 64,
                            basis: &bases.chroma422,
                            scan: &tables::SCAN,
                            eob_cdf: EobCdf::ChrEobBin,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 1024,
                            u_skip_row: &CHROMA_SKIP_TX64_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: cfl_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (16, 8) => {
                    let (tus2, mode_idx) = encode_luma_leaf32(
                        recy,
                        &LumaSource {
                            plane: yp,
                            stride: pw,
                        },
                        &LumaGridBlock {
                            mi_cols: tmc,
                            mi_rows: tmr,
                            y: sb_y,
                            x: sb_x,
                        },
                        &LumaQuantSpec {
                            basis: &bases.luma,
                            qstep: qstep_i,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let (skip2, dcs2) =
                        sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 32,
                            height: 32,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.chroma420,
                            scan: &tables::SCAN,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let uv_pred = if mh_choice.is_none() && self.tune.chroma_mode_search {
                        chroma422::chroma_mode_decide_leaf(
                            enc,
                            &chroma422::ChromaModePlanes {
                                reconstructed_u: recu,
                                reconstructed_v: recv,
                                source_u: up,
                                source_v: vp,
                                stride: pcw,
                            },
                            &chroma422::ChromaModeBlock {
                                y: cy,
                                x: cx,
                                size: 32,
                                coded_width: width / 2,
                                coded_height: height,
                            },
                            &chroma422::ChromaModeSearch {
                                neutral,
                                basis: &bases.chroma420,
                                scan: &tables::SCAN,
                                quant_context: qc,
                                qstep: qstep_i,
                                sb_qstep: qstep_i,
                                residual_scale: 1.0,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                reduced: self.speed.reduced_modes(),
                                angle_directional: self.speed.chroma_angle_directional(),
                                bit_depth: self.bit_depth as i32,
                            },
                        )
                    } else {
                        None
                    };
                    encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 32,
                            ch: 32,
                            basis: &bases.chroma420,
                            scan: &tables::SCAN,
                            eob_cdf: EobCdf::ChrEobBin,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 1024,
                            u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: uv_pred
                                .as_ref()
                                .map(|(u, v)| (u.as_slice(), v.as_slice())),
                        },
                    )
                }
                (8, 16) => {
                    let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                        recy,
                        &LumaSource {
                            plane: yp,
                            stride: pw,
                        },
                        &LumaGridBlock {
                            mi_cols: tmc,
                            mi_rows: tmr,
                            y: sb_y,
                            x: sb_x,
                        },
                        &LumaQuantSpec {
                            basis: &bases.luma,
                            qstep: qstep_i,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let (skip2, dcs2) = sb_tu_contexts_pos(
                        &[(0, 0), (32, 0)],
                        &tus2,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: false,
                        },
                    );
                    let s2 = [skip2[0], skip2[1]];
                    let d2 = [dcs2[0], dcs2[1]];
                    encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 16,
                            ch: 64,
                            basis: &bases.luma16x64,
                            scan: &SCAN16X32,
                            eob_cdf: EobCdf::ChrEob512,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 512,
                            u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: None,
                            mode_predictors: None,
                        },
                    )
                }
                (8, 8) => {
                    let (tu, mode_idx) = encode_luma_leaf_s32x32(
                        recy,
                        &LumaSource {
                            plane: yp,
                            stride: pw,
                        },
                        &LumaGridBlock {
                            mi_cols: tmc,
                            mi_rows: tmr,
                            y: sb_y,
                            x: sb_x,
                        },
                        &LumaQuantSpec {
                            basis: &bases.luma,
                            qstep: qstep_i,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let (skip2, dcs2) = sb_tu_contexts_pos(
                        &[(0, 0)],
                        std::slice::from_ref(&tu),
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 16,
                            height: 32,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.c16x32,
                            scan: &SCAN16X32,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_32x32(enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 16,
                            ch: 32,
                            basis: &bases.c16x32,
                            scan: &SCAN16X32,
                            eob_cdf: EobCdf::ChrEob512,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 512,
                            u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (4, 16) => {
                    // Right-edge 16×64 luma leaf → 4:2:2 chroma 8×64 (TX_8X64,
                    // coeff 8×32, SCAN8X32, eob 256, skip class 3).
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 16, 64, neutral, self.bit_depth as i32);
                    let lev = bases.luma16x64.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 16, 64, pred),
                        0.0,
                        &SCAN16X32,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        16,
                        64,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &SCAN16X32,
                            16,
                            64,
                            self.bit_depth as i32,
                        ),
                    );
                    let tu = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        4,
                        16,
                    );
                    encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 8,
                            ch: 64,
                            basis: &bases.c8x64,
                            scan: &SCAN8X32,
                            eob_cdf: EobCdf::ChrEob256,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 256,
                            u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: None,
                            mode_predictors: None,
                        },
                    )
                }
                (16, 4) => {
                    // Bottom-edge 64×16 luma leaf → 4:2:2 chroma 32×16 (TX_32X16,
                    // coeff 32×16, SCAN32X16, eob 512, skip class 3).
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 64, 16, neutral, self.bit_depth as i32);
                    let lev = bases.luma64x16.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 64, 16, pred),
                        0.0,
                        &SCAN32X16,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        64,
                        16,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &SCAN32X16,
                            64,
                            16,
                            self.bit_depth as i32,
                        ),
                    );
                    let tu = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        16,
                        4,
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 32,
                            height: 16,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.c32x16,
                            scan: &SCAN32X16,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 32,
                            ch: 16,
                            basis: &bases.c32x16,
                            scan: &SCAN32X16,
                            eob_cdf: EobCdf::ChrEob512,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 512,
                            u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (4, 4) => {
                    // Bottom-right 16×16 corner leaf (residue 4 in both dims).
                    // Full-AC native TX_16X16 luma (entropy class 2, eob class 256),
                    // tx_type RD-chosen among DCT_DCT / ADST_ADST / ADST_DCT /
                    // DCT_ADST (EXT_NEW_TX_SET, DC mode); 4:2:2 chroma 8×16 stays
                    // DCT (tx_type is luma-only).
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 16, 16, neutral, self.bit_depth as i32);
                    let resid = get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred);
                    let pred_flat = [pred; 256];
                    let mut src16 = [0f32; 256];
                    for (dst_row, src_row) in src16
                        .as_chunks_mut::<16>()
                        .0
                        .iter_mut()
                        .zip(rect_rows(yp, pw, sb_y, sb_x, 16, 16))
                    {
                        dst_row.copy_from_slice(src_row);
                    }
                    let rate = coeff_rate_f32;
                    let sse = |rec: &[f32]| -> f32 { tx16_distortion(&src16, rec) };
                    let lambda = leaf::part_lambda(qstep_i, self.tune.part_lambda_c);
                    // DCT_DCT candidate (idx 0).
                    let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                    let rec_dct = itx422::reconstruct_luma16(
                        &pred_flat,
                        &lev_dct,
                        qstep_i,
                        &SCAN16,
                        self.bit_depth as i32,
                    );
                    let dist_dct = sse(&rec_dct);
                    let cost_dct = dist_dct + lambda * rate(&lev_dct);
                    // ADST_ADST candidate (idx 1, DST-VII both axes).
                    let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
                    let rec_adst = itx422::reconstruct_luma16_adst(
                        &pred_flat,
                        &lev_adst,
                        qstep_i,
                        &SCAN16,
                        true,
                        true,
                        self.bit_depth as i32,
                    );
                    let dist_adst = sse(&rec_adst);
                    let cost_adst =
                        dist_adst + lambda * (rate(&lev_adst) + TX16_TYPE_RATE_DELTA[1]);
                    // ADST_DCT candidate (idx 2: ADST vertical, DCT horizontal →
                    // inverse row_adst=false, col_adst=true; ~3.1 extra bits).
                    let lev_ad = bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
                    let rec_ad = itx422::reconstruct_luma16_adst(
                        &pred_flat,
                        &lev_ad,
                        qstep_i,
                        &SCAN16,
                        false,
                        true,
                        self.bit_depth as i32,
                    );
                    let dist_ad = sse(&rec_ad);
                    let cost_ad = dist_ad + lambda * (rate(&lev_ad) + TX16_TYPE_RATE_DELTA[2]);
                    // DCT_ADST candidate (idx 3: DCT vertical, ADST horizontal →
                    // inverse row_adst=true, col_adst=false; ~2.7 extra bits).
                    let lev_da = bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
                    let rec_da = itx422::reconstruct_luma16_adst(
                        &pred_flat,
                        &lev_da,
                        qstep_i,
                        &SCAN16,
                        true,
                        false,
                        self.bit_depth as i32,
                    );
                    let dist_da = sse(&rec_da);
                    let cost_da = dist_da + lambda * (rate(&lev_da) + TX16_TYPE_RATE_DELTA[3]);
                    let choice = choose_tx16_type(
                        [cost_dct, cost_adst, cost_ad, cost_da],
                        [dist_dct, dist_adst, dist_ad, dist_da],
                        [
                            false,
                            tx16_dc_only(&lev_adst),
                            tx16_dc_only(&lev_ad),
                            tx16_dc_only(&lev_da),
                        ],
                    );
                    let (lev, rec, tx_idx): (&[f32], &[f32; 256], usize) = match choice {
                        1 => (&lev_adst, &rec_adst, 1),
                        2 => (&lev_ad, &rec_ad, 2),
                        3 => (&lev_da, &rec_da, 3),
                        _ => (&lev_dct, &rec_dct, 0),
                    };
                    put_block_rect(recy, pw, sb_y, sb_x, 16, 16, rec);
                    let tu: Vec<Coeff> = levels_to_coeffs(lev);
                    let (_s, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        4,
                        4,
                    );
                    let skip = SKIP_TX16_QC[qc][0] as u32;
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 8,
                            height: 16,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.c8x16,
                            scan: &tables::SCAN8X16,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_16x16_full(enc, &tu, skip, dcs, 0, true, pc, 11074, tx_idx);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 8,
                            ch: 16,
                            basis: &bases.c8x16,
                            scan: &tables::SCAN8X16,
                            eob_cdf: EobCdf::ChrEob128,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 128,
                            u_skip_row: &SKIP_TX16_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (2, 8) => {
                    // Right-edge 8×32 DC-only luma leaf → 4:2:2 chroma 4×32
                    // (TX_4X32, coeff 4×32, SCAN4X32, eob 128 NO-ESCAPE, class 2).
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 8, 32, neutral, self.bit_depth as i32);
                    let lev = bases.luma8x32.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 8, 32, pred),
                        0.0,
                        &SCAN8X32,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        8,
                        32,
                        &crate::av2::itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &SCAN8X32,
                            8,
                            32,
                            self.bit_depth as i32,
                        ),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        2,
                        8,
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 4,
                            height: 32,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.c4x32,
                            scan: &tables::SCAN4X32,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_8x32(enc, &tu, skip, dcs, 0, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 4,
                            ch: 32,
                            basis: &bases.c4x32,
                            scan: &tables::SCAN4X32,
                            eob_cdf: EobCdf::ChrEob128,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 128,
                            u_skip_row: &SKIP_TX16_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (8, 2) => {
                    // Bottom-edge 32×8 DC-only luma leaf → 4:2:2 chroma 16×8
                    // (TX_16X8, coeff 16×8, SCAN16X8, eob 128 NO-ESCAPE, class 2).
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 32, 8, neutral, self.bit_depth as i32);
                    let lev = bases.luma32x8.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 32, 8, pred),
                        0.0,
                        &SCAN32X8,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        32,
                        8,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &SCAN32X8,
                            32,
                            8,
                            self.bit_depth as i32,
                        ),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        8,
                        2,
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 16,
                            height: 8,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.c16x8,
                            scan: &tables::SCAN16X8,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_32x8(enc, &tu, skip, dcs, 0, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 16,
                            ch: 8,
                            basis: &bases.c16x8,
                            scan: &tables::SCAN16X8,
                            eob_cdf: EobCdf::ChrEob128,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 128,
                            u_skip_row: &SKIP_TX16_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (2, 2) => {
                    // Both-axis residue-2 corner: 8×8 luma (TX_8X8) + 4×8 chroma per
                    // plane (4:2:2: half-width, full-height → TX_4X8).
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 8, neutral, bd);
                    let lev = bases.c8x8.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 8, 8, pred),
                        0.0,
                        &tables::SCAN8X8,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        8,
                        8,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &tables::SCAN8X8,
                            8,
                            8,
                            bd,
                        ),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        2,
                        2,
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 4,
                            height: 8,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.c4x8,
                            scan: &tables::SCAN4X8,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_8x8(
                        enc,
                        &tu,
                        skip,
                        dcs,
                        0,
                        true,
                        pc,
                        3148,
                        Some((&coder::TXTP_EXT8, 0, 6)),
                    );
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 4,
                            ch: 8,
                            basis: &bases.c4x8,
                            scan: &tables::SCAN4X8,
                            eob_cdf: EobCdf::ChrEob32,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 32,
                            u_skip_row: &SKIP_TX8_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (2, 4) => {
                    // residue-2 W × residue-4 H: 8×16 luma (TX_8X16) + 4×16 chroma
                    // per plane (4:2:2 half-width/full-height → TX_4X16, eob64 ctx1).
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
                    let lev = bases.c8x16.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 8, 16, pred),
                        0.0,
                        &tables::SCAN8X16,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        8,
                        16,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &tables::SCAN8X16,
                            8,
                            16,
                            bd,
                        ),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        2,
                        4,
                    );
                    coder::encode_luma_leaf_rect128(
                        enc,
                        &tu,
                        &LumaLeafRect128Spec {
                            skip_cdf: skip,
                            dc_sign_ctx: dcs,
                            mode_idx: 0,
                            has_chroma: true,
                            width_mi: 2,
                            height_mi: 4,
                            part_cdf: pc,
                            tx_part_cdf: 12348,
                            scan: &tables::SCAN8X16,
                            tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
                        },
                    );
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 4,
                            ch: 16,
                            basis: &bases.c4x16,
                            scan: &tables::SCAN4X16,
                            eob_cdf: EobCdf::ChrEob64,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 64,
                            u_skip_row: &SKIP_TX8_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: None,
                            mode_predictors: None,
                        },
                    )
                }
                (4, 2) => {
                    // residue-4 W × residue-2 H: 16×8 luma (TX_16X8) + 8×8 chroma
                    // per plane (4:2:2 half-width/full-height → TX_8X8, eob64 ctx1).
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
                    let lev = bases.c16x8.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 16, 8, pred),
                        0.0,
                        &tables::SCAN16X8,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        16,
                        8,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            qstep_i,
                            &tables::SCAN16X8,
                            16,
                            8,
                            bd,
                        ),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        4,
                        2,
                    );
                    let uv_pred = if self.tune.chroma_mode_search {
                        chroma422::chroma_mode_decide_leaf(
                            enc,
                            &chroma422::ChromaModePlanes {
                                reconstructed_u: recu,
                                reconstructed_v: recv,
                                source_u: up,
                                source_v: vp,
                                stride: pcw,
                            },
                            &chroma422::ChromaModeBlock {
                                y: cy,
                                x: cx,
                                size: 8,
                                coded_width: width / 2,
                                coded_height: height,
                            },
                            &chroma422::ChromaModeSearch {
                                neutral,
                                basis: &bases.c8x8,
                                scan: &SCAN8X8,
                                quant_context: qc,
                                qstep: qstep_i,
                                sb_qstep: qstep_i,
                                residual_scale: 1.0,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                reduced: self.speed.reduced_modes(),
                                angle_directional: self.speed.chroma_angle_directional(),
                                bit_depth: self.bit_depth as i32,
                            },
                        )
                    } else {
                        None
                    };
                    crate::av2::coder::encode_luma_leaf_rect128(
                        enc,
                        &tu,
                        &LumaLeafRect128Spec {
                            skip_cdf: skip,
                            dc_sign_ctx: dcs,
                            mode_idx: 0,
                            has_chroma: true,
                            width_mi: 4,
                            height_mi: 2,
                            part_cdf: pc,
                            tx_part_cdf: 12348,
                            scan: &tables::SCAN16X8,
                            tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
                        },
                    );
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 8,
                            ch: 8,
                            basis: &bases.c8x8,
                            scan: &SCAN8X8,
                            eob_cdf: EobCdf::ChrEob64,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 64,
                            u_skip_row: &SKIP_TX8_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: None,
                            mode_predictors: uv_pred
                                .as_ref()
                                .map(|(u, v)| (u.as_slice(), v.as_slice())),
                        },
                    )
                }
                (4, 8) => {
                    // residue-4 W × residue-{6,8} H: 16×32 luma (TX_16X32 long-side-32)
                    // + 8×32 chroma per plane (4:2:2 half-width → TX_8X32, eob256 ctx2).
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 32, neutral, bd);
                    let lev = bases.luma16x32.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 16, 32, pred),
                        0.0,
                        &SCAN16X32,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        16,
                        32,
                        &itx422::reconstruct_chroma(pred, &lev, qstep_i, &SCAN16X32, 16, 32, bd),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        4,
                        8,
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 8,
                            height: 32,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.luma8x32,
                            scan: &SCAN8X32,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_16x32(enc, &tu, skip, dcs, 0, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 8,
                            ch: 32,
                            basis: &bases.luma8x32,
                            scan: &SCAN8X32,
                            eob_cdf: EobCdf::ChrEob256,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 256,
                            u_skip_row: &SKIP_TX16_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: None,
                        },
                    )
                }
                (8, 4) => {
                    // residue-{6,8} W × residue-4 H: 32×16 luma (TX_32X16 long-side-32)
                    // + 16×16 chroma per plane (4:2:2 half-width → TX_16X16, eob256 ctx2).
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 32, 16, neutral, bd);
                    let lev = bases.luma32x16.project_scan(
                        &get_residual_rect(yp, pw, sb_y, sb_x, 32, 16, pred),
                        0.0,
                        &SCAN32X16,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        32,
                        16,
                        &itx422::reconstruct_chroma(pred, &lev, qstep_i, &SCAN32X16, 32, 16, bd),
                    );
                    let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                    let (skip, dcs) = sb_tu_contexts_rect(
                        &tu,
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: true,
                        },
                        8,
                        4,
                    );
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
                        &chroma422::ChromaLeafPlanes {
                            reconstructed_luma: recy,
                            reconstructed_u: recu,
                            reconstructed_v: recv,
                            source_u: up,
                            source_v: vp,
                            luma_stride: pw,
                            chroma_stride: pcw,
                        },
                        &chroma422::ChromaLeafGeometry {
                            bounds: mhccp_bounds,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            chroma_y: cy,
                            chroma_x: cx,
                            width: 16,
                            height: 16,
                            subsample_x: true,
                            subsample_y: false,
                            have_top: lmr > 0,
                            have_left: lmc > 0,
                        },
                        &chroma422::ChromaLeafRd {
                            neutral,
                            basis: &bases.luma16x16,
                            scan: &SCAN16,
                            qstep: qstep_i,
                            lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let uv_pred = if mh_choice.is_none() && self.tune.chroma_mode_search {
                        chroma422::chroma_mode_decide_leaf(
                            enc,
                            &chroma422::ChromaModePlanes {
                                reconstructed_u: recu,
                                reconstructed_v: recv,
                                source_u: up,
                                source_v: vp,
                                stride: pcw,
                            },
                            &chroma422::ChromaModeBlock {
                                y: cy,
                                x: cx,
                                size: 16,
                                coded_width: width / 2,
                                coded_height: height,
                            },
                            &chroma422::ChromaModeSearch {
                                neutral,
                                basis: &bases.luma16x16,
                                scan: &tables::SCAN16,
                                quant_context: qc,
                                qstep: qstep_i,
                                sb_qstep: qstep_i,
                                residual_scale: 1.0,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                lambda: leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                reduced: self.speed.reduced_modes(),
                                angle_directional: self.speed.chroma_angle_directional(),
                                bit_depth: self.bit_depth as i32,
                            },
                        )
                    } else {
                        None
                    };
                    encode_luma_leaf_32x16(enc, &tu, skip, dcs, 0, true, pc);
                    code_422_chroma_tu(
                        enc,
                        ChromaPlanes {
                            rec_u: recu,
                            rec_v: recv,
                            src_u: up,
                            src_v: vp,
                            stride: pcw,
                            coded_width: mhccp_bounds.chroma_width,
                            coded_height: mhccp_bounds.chroma_height,
                        },
                        &ChromaTxSpec {
                            cw: 16,
                            ch: 16,
                            basis: &bases.luma16x16,
                            scan: &tables::SCAN16,
                            eob_cdf: EobCdf::ChrEob256,
                            eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                            area: 256,
                            u_skip_row: &SKIP_TX16_QC[qc],
                        },
                        QuantCtx {
                            qc,
                            neutral,
                            qstep: qstep_i,
                            rdoq_lambda: self.tune.chroma_rdoq_lambda,
                        },
                        ChromaNeighbors { ua, ul, va, vl },
                        &ChromaTuInput {
                            y: cy,
                            x: cx,
                            bit_depth: self.bit_depth as i32,
                            cfl: mh_choice.as_ref(),
                            mode_predictors: uv_pred
                                .as_ref()
                                .map(|(u, v)| (u.as_slice(), v.as_slice())),
                        },
                    )
                }
                other => unreachable!("unsupported native 4:2:2 leaf {:?}", other),
            };
            let cfl_used = enc.cfl_signaled as i32;
            for c in lmc..lmc + bw_mi {
                u_above[c] = u_present as i32;
                v_above[c] = v_present as i32;
                cfl_above[c] = cfl_used;
            }
            for r in lmr..lmr + bh_mi {
                u_left[r] = u_present as i32;
                v_left[r] = v_present as i32;
                cfl_left[r] = cfl_used;
            }
            // Mark this leaf's luma mi as coded for the next leaf's MHCCP.
            for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                for c in (lmc & 15)..((lmc & 15) + bw_mi).min(16) {
                    enc.sb_coded[r * 16 + c] = 1;
                }
            }
        }
    }
}
