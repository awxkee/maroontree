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
use crate::av2::video::mc;

// Max per-TU nonzeros for NEWMV residual; denser TUs fall back to intra
// (dense inter-TX32 coefficient coding is not yet bit-exact).
const NEWMV_MAX_TU_NZ: usize = 1024;

/// Phase 3 precomputed CCSO state: per-plane edge-filter result and the per-SB
/// on/off decision grid, derived in a first pass over the completed recon.
pub(super) struct CcsoPrecomp {
    pub(super) u: Option<(ccso::CcsoEdgeResult, Vec<u8>)>,
    pub(super) v: Option<(ccso::CcsoEdgeResult, Vec<u8>)>,
    pub(super) sb_cols: usize,
}

/// Convert a CcsoEdgeResult into the header PlaneResult (always edge mode here).
fn edge_to_plane(r: &ccso::CcsoEdgeResult) -> ccso::PlaneResult {
    ccso::PlaneResult::Edge {
        scale_idx: r.scale_idx,
        quant_idx: r.quant_idx,
        ext_filter_support: r.ext_filter_support,
        edge_clf: r.edge_clf,
        max_band_log2: r.max_band_log2,
        offsets: r.offsets.clone(),
    }
}

impl Av2Encoder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_420_core(
        &self,
        yf: &[f32],
        cbf: &[f32],
        crf: &[f32],
        width: usize,
        height: usize,
        // Phase 3: precomputed CCSO state from a prior decision pass. When `Some`,
        // the per-SB flags are emitted from the decision grids and the filter is
        // applied gated by them (no search). When `None`, the pass searches all-on
        // and records the result+grids on the returned encoder for a second pass.
        ccso_pre: Option<&CcsoPrecomp>,
        // Per-tile inter reference. `None` on the whole-frame path (region == frame,
        // so region-local coordinates already equal frame coordinates). When
        // `Some`, it carries the FULL reference frame plus this tile's frame origin;
        // inter reads add that origin so motion vectors that cross tile boundaries
        // resolve against real neighbor content, matching the decoder.
        tile_ref: Option<&crate::av2::tiling::TileRefCtx>,
        // Per-block CDEF pass 2: the frame decision (strengths + per-SB grid).
        cdef_pre: Option<&crate::av2::cdef_est::CdefDecision>,
        // Staged replay: `Off` = search+emit (today); `Capture` logs the whole-64
        // luma mode-search winner; `Replay` reuses it (skips `encode_luma_sb`).
        mut decide_mode: crate::av2::replay::DecideMode<'_>,
    ) -> RangeEncoder {
        let bases = &self.bases;
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph / 2);
        let yp = pad_plane(yf, width, height, pw, ph);
        let up = pad_plane(cbf, width.div_ceil(2), height.div_ceil(2), pcw, pch);
        let vp = pad_plane(crf, width.div_ceil(2), height.div_ceil(2), pcw, pch);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.inter_tile = self.inter_tile.load(std::sync::atomic::Ordering::Relaxed);
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = true;
        enc.mhccp_ssy = true;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
        // CCSO band classification reads luma. The AV2 decoder classifies bands from the
        // POST-deblock, pre-CDEF luma. `recy` matches that byte-exactly when either the
        // luma deblock is inert (df_quant == 0, not signalled) OR the DF has been applied
        // to recy before the CCSO block below. The whole-SB DF runs when
        // `filter_active && !delta_q_present`, so on this path recy is correct post-deblock
        // whenever (df_quant == 0) OR (!AQ). Gate CCSO to that exact-correctness condition
        // (previously restricted to df_quant == 0 only, before real deblock existed). The
        // CCSO offset depends only on the luma-derived band, so luma-match is the sole
        // correctness requirement.
        let df_quant = {
            let qs = quant::qstep(self.base_q_idx as u32) as i32;
            ((qs + 4) >> 3) >> 6
        };
        let luma_deblock_active = self.tune.deblock && df_quant >= 1;
        // recy is byte-exact post-deblock luma when the DF is applied (AQ off) or inert.
        let recy_is_post_deblock = !luma_deblock_active || !enc.delta_q_present;
        let ccso_ok = self.tune.ccso && self.base_q_idx != 0 && recy_is_post_deblock;
        enc.ccso_u_enable = ccso_ok;
        enc.ccso_v_enable = ccso_ok;
        enc.ccso_cols = pw / 64;
        // Phase 3: the decision pass (ccso_pre == None) is a throwaway recon pass —
        // it searches the filter and decides per-SB on/off but must NOT emit any
        // ccso flags into its (discarded) bitstream. Only the emit pass, which has
        // the decision grids, emits flags and applies the gated filter. So flag
        // emission is gated on `ccso_pre.is_some()`; the search itself still needs
        // the enable booleans, tracked separately below.
        let ccso_search_u = enc.ccso_u_enable;
        let ccso_search_v = enc.ccso_v_enable;
        if ccso_pre.is_none() {
            enc.ccso_u_enable = false;
            enc.ccso_v_enable = false;
        }
        // Phase 3: in the emit pass, install the per-SB decision grids so the flag
        // emission and filter application are gated by the RD decisions. A plane that
        // the decision pass turned off entirely is disabled here (no flags, no header).
        if let Some(pre) = ccso_pre {
            enc.ccso_cols = pre.sb_cols;
            match &pre.u {
                Some((_, grid)) => enc.ccso_grid = grid.clone(),
                None => enc.ccso_u_enable = false,
            }
            match &pre.v {
                Some((_, grid)) => enc.ccso_grid_v = grid.clone(),
                None => enc.ccso_v_enable = false,
            }
        }
        let qc = enc.qc;
        let neutral = self.dc_neutral();
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
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];
        // Per-SB CfL-usage grid for get_cfl_ctx (whole-64 fast path).
        let mut cfl_has = vec![0i32; sb_cols * sb_rows];

        // Native edge partitioning (no padding) when the geometry is supported and the
        // image has residue edges. Otherwise, the whole-SB path codes one 32x32 chroma TU
        // per SB (padded to SB on non-aligned dims).
        let native_mi = native_420_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        // 4:2:0 edge handling. The fast whole-SB path codes one 32×32 chroma TU per SB
        // against PADDED extents, but `native_420_mi` hands the context functions the
        // NATIVE (un-padded) mi extents for residues ≥6/==4. For residues {10,12,14}
        // `lossy_needs_partition` is false, so the old code took the fast path with that
        // native/padded extent mismatch — desyncing the decoder on the partial edge SB
        // (e.g. any side ≡ 40 mod 64, like 1000). Routing every non-64-aligned dimension
        // through the edge-aware partition walk fixes it; 64-aligned sizes are unchanged.
        let mc_edge = (((width + 7) & !7) / 4) as i64 % 16;
        let mr_edge = (((height + 7) & !7) / 4) as i64 % 16;
        let needs_partition = native_mi.is_some()
            && (lossy_needs_partition(width, height)
                || mc_edge != 0
                || mr_edge != 0
                || self.tune.chroma_split);
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = [0u8; 16];

        if needs_partition {
            let mut tx_leaves: Vec<(usize, usize, usize, usize)> = Vec::new();
            let mut skip_leaves: Vec<(usize, usize, usize, usize)> = Vec::new();
            let mut sb_qidx = vec![self.base_q_idx as u16; sb_rows * sb_cols];
            self.encode_yuv420_partition(
                &mut enc,
                LumaPlanes {
                    rec: &mut recy,
                    src: &yp,
                },
                ChromaPlaneRefs {
                    rec_u: &mut recu,
                    rec_v: &mut recv,
                    src_u: &up,
                    src_v: &vp,
                },
                &PartitionPass {
                    luma_stride: pw,
                    chroma_stride: pcw,
                    width,
                    height,
                    sb_rows,
                    sb_cols,
                    tmc,
                    tmr,
                    quant: QuantCtx {
                        qc,
                        neutral,
                        qstep: qstep_i,
                        rdoq_lambda: self.tune.chroma_rdoq_lambda,
                    },
                },
                PartitionNeighbors {
                    above: &mut above,
                    left: &mut left,
                    above_pctx: &mut above_pctx,
                    left_pctx: &mut left_pctx,
                },
                ChromaNeighborBufs {
                    u_above: &mut u_above,
                    v_above: &mut v_above,
                    u_left: &mut u_left,
                    v_left: &mut v_left,
                },
                tile_ref,
                &mut tx_leaves,
                &mut skip_leaves,
                &mut sb_qidx,
                decide_mode,
            );
            // Real deblocking on the partition path (the default 4:2:0 route). Build
            // the luma transform-boundary grid from the recorded leaves (each leaf
            // tiles into min(dim,32)-sized TUs) plus the per-SB qindex, and apply the
            // loop filter. Works with AQ on (per-SB qindex feeds the DF thresholds).
            let filter_active = self.tune.deblock && df_quant >= 1;
            if filter_active {
                let eff_dy = if self.tune.db_delta_y == i32::MIN {
                    if df_quant >= 5 { 0 } else { 1 }
                } else {
                    self.tune.db_delta_y
                };
                crate::av2::deblock::deblock_frame(crate::av2::deblock::FrameDeblock {
                    recy: &mut recy,
                    recu: &mut recu,
                    recv: &mut recv,
                    luma_stride: pw,
                    chroma_stride: pcw,
                    width,
                    height,
                    ssx: 1,
                    ssy: 1,
                    has_chroma: true,
                    chroma_deblock: self.tune.chroma_deblock,
                    bit_depth: self.bit_depth as u32,
                    delta_y: eff_dy,
                    delta_uv: self.tune.db_delta_uv,
                    base_q: self.base_q_idx as u16,
                    sb_cols,
                    sb_qidx: &sb_qidx,
                    leaves: &tx_leaves,
                    chroma_leaves: &tx_leaves,
                    skip_leaves: &skip_leaves,
                    luma_tx_cap: (8, 8),

                    chroma_tx_cap: (8, 8),
                });
            }
            let mut filtered_recon = vec![recy, recu, recv];
            if let Some(pre) = cdef_pre {
                crate::av2::cdef_est::apply_per_block(
                    &mut filtered_recon,
                    pw,
                    ph,
                    pcw,
                    pch,
                    1,
                    1,
                    true,
                    pre,
                    self.bit_depth,
                );
            } else if self.tune.cdef && !replay::in_tile_subencode() {
                enc.cdef_decided = crate::av2::cdef_est::search_per_block(
                    &filtered_recon,
                    &[yp.clone(), up.clone(), vp.clone()],
                    pw,
                    ph,
                    pcw,
                    pch,
                    1,
                    1,
                    true,
                    self.base_q_idx,
                    self.bit_depth,
                );
            }
            // Video DPB capture also on the native-partition (edge-tile) path, so a
            // stitched full-frame reference has no holes where edge tiles sit.
            if self
                .capture_recon
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                enc.recon = filtered_recon;
            }
            return enc;
        }

        let mut midx_grid = vec![0xff_u8; sb_cols * sb_rows];
        // Adaptive-quantization reference: center the per-SB qindex delta on this
        // tile's mean log-activity so the deltas are zero-mean (flat SBs get finer
        // q, busy SBs coarser) without a net quantizer bias that would just trade
        // size for PSNR. Computed in a cheap pre-pass over the tile's SBs.
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(&yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
            0, /* uv delta not yet wired for 4:2:0 */
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        )
        .with_dark_aq(self.tune.dark_aq);
        // Inter reference: either the whole frame (region == frame; local coords are
        // frame coords, so `ref_x0/ref_y0` are 0 and the reference stride is `pw`) or,
        // for a tile, the FULL reference frame at frame stride with this tile's origin.
        // Reference reads below add `(ref_x0, ref_y0)` to region-local block positions
        // and use `ref_ls`/`ref_cs`, keeping motion compensation frame-global.
        let (core_last_ref, ref_x0, ref_y0, ref_ls, ref_cs): (
            std::sync::Arc<Vec<Vec<f32>>>,
            usize,
            usize,
            usize,
            usize,
        ) = if enc.inter_tile {
            match tile_ref {
                Some(r) => (
                    std::sync::Arc::clone(&r.planes),
                    r.x0,
                    r.y0,
                    r.luma_stride,
                    r.chroma_stride,
                ),
                None => (
                    std::sync::Arc::clone(&self.last_ref.lock().unwrap()),
                    0,
                    0,
                    pw,
                    pcw,
                ),
            }
        } else {
            (std::sync::Arc::new(Vec::new()), 0, 0, pw, pcw)
        };
        let core_has_last = core_last_ref.len() >= 3 && !core_last_ref[0].is_empty();
        let frame_mv_seed = *self.video_mv_seed.lock().unwrap();
        // Frame-scoped inter preparation in the same f32 sample representation as
        // reconstruction, filtering and the DPB.
        const BRD: usize = 72;
        let inter_luma: Option<(Vec<f32>, usize, usize)> = if core_has_last {
            let refh = core_last_ref[0].len() / ref_ls;
            let (bref, bstride) = mc::bordered(&core_last_ref[0], ref_ls, refh, ref_ls, BRD);
            Some((bref, bstride, refh))
        } else {
            None
        };
        // Frame-scoped bordered chroma references (U, V) built once. The NEWMV
        // residual path predicted chroma per superblock, re-converting and
        // re-bordering the whole U/V reference plane each time (twice per SB).
        let inter_chroma: Option<[(Vec<f32>, usize); 2]> = if core_has_last {
            let cbrd = BRD / 2;
            let build = |plane: &[f32]| {
                let ch = plane.len() / ref_cs;
                mc::bordered(plane, ref_cs, ch, ref_cs, cbrd)
            };
            Some([build(&core_last_ref[1]), build(&core_last_ref[2])])
        } else {
            None
        };
        let mut core_skip_above = vec![0u8; sb_cols.max(1)];
        // Per-SB inter-ness for intra_inter context (0=intra,1=inter).
        let mut core_inter_above = vec![0u8; sb_cols.max(1)];
        let mut core_newmv_above = vec![0u8; sb_cols.max(1)];
        // Per-SB decoded MVs (eighth-pel) for the DRL[0] predictor model.
        let mut core_mv_above: Vec<Option<crate::av2::video::mv::Mv>> = vec![None; sb_cols.max(1)];
        let mut core_me_scratch = crate::av2::video::me::MeScratch::default();
        let mut core_mc_tmp = Vec::new();
        for row in 0..sb_rows {
            let mut core_skip_left = 0u8;
            let mut core_inter_left = 0u8;
            let mut core_newmv_left = 0u8;
            let mut core_mv_left: Option<crate::av2::video::mv::Mv> = None;
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                if enc.cdef_nb >= 2 {
                    enc.cdef_pending = true;
                    enc.cdef_sb_rc = (row, col);
                }
                // GLOBALMV zero-motion skip: static full 64x64 block copies LAST.
                if core_has_last && sb_x + 64 <= width && sb_y + 64 <= height {
                    let ly = &core_last_ref[0];
                    let sse = rect_sse_f32(
                        &PlaneRect {
                            plane: &yp,
                            stride: pw,
                            y: sb_y,
                            x: sb_x,
                        },
                        &PlaneRect {
                            plane: ly,
                            stride: ref_ls,
                            y: ref_y0 + sb_y,
                            x: ref_x0 + sb_x,
                        },
                        64,
                        64,
                    );
                    let up = row > 0;
                    let lf = col > 0;
                    let ia = core_inter_above[col] == 1;
                    let il = core_inter_left == 1;
                    enc.intra_inter_ctx = if up && lf {
                        let n_intra = (!il as u8) + (!ia as u8);
                        if n_intra == 2 { 3 } else { n_intra as usize }
                    } else if up {
                        if ia { 0 } else { 3 }
                    } else if lf {
                        if il { 0 } else { 3 }
                    } else {
                        0
                    };
                    let sa = core_skip_above[col];
                    let sl = core_skip_left;
                    let skip_ctx = if up && lf {
                        (sl + sa) as usize
                    } else if up {
                        (2 * sa) as usize
                    } else if lf {
                        (2 * sl) as usize
                    } else {
                        0
                    };
                    let mode_ctx = (core_inter_above[col] + core_inter_left) as usize
                        + if core_newmv_above[col] != 0 || core_newmv_left != 0 {
                            2
                        } else {
                            0
                        };
                    // Zero-motion skip vs intra: skip wins when its full block
                    // syntax plus distortion beats the intra rate proxy. AVM's rdmult is
                    // ~q^2, so the accept bound scales with q^2, not linear q. The
                    // reference intra rate proxy (~2 bits/px) is folded into the
                    // bound; SS2 weight retunes the operating point.
                    use crate::av2::video::rd;
                    let skip_dist = sse * rd::SS2_INTER_DIST_W;
                    let skip_rate = rd::inter_syntax_bits(&enc, skip_ctx, mode_ctx, true, 1, None);
                    let skip_cost = rd::rd_cost(skip_dist, skip_rate, qstep_i as u32);
                    let intra_bound = rd::rd_cost(0.0, 2.0 * 64.0 * 64.0, qstep_i as u32);
                    if skip_cost < intra_bound {
                        // intra_inter ctx per AVM: up fills above+above_right, left fills
                        // left+bottom_left; ctx from neighbor inter-ness (both intra=3,
                        // one=1, both inter=0; single neighbor intra=2/inter=0; none=0).
                        let up = row > 0;
                        let lf = col > 0;
                        let ia = core_inter_above[col] == 1; // above SB inter?
                        let il = core_inter_left == 1; // left SB inter?
                        enc.intra_inter_ctx = if up && lf {
                            let n_intra = (!il as u8) + (!ia as u8);
                            if n_intra == 2 { 3 } else { n_intra as usize }
                        } else if up {
                            if ia { 0 } else { 3 }
                        } else if lf {
                            if il { 0 } else { 3 }
                        } else {
                            0
                        };
                        // skip_txfm ctx: neighbors_line_buffer fills 2 slots; up avail
                        // -> above+above_right, left avail -> left+bottom_left. So a
                        // present neighbor's skip counts twice when it's the only side.
                        let up = row > 0;
                        let lf = col > 0;
                        let sa = core_skip_above[col];
                        let sl = core_skip_left;
                        let skip_ctx = if up && lf {
                            (sl + sa) as usize // [bottom_left, above_right]
                        } else if up {
                            (2 * sa) as usize // [above_right, above]
                        } else if lf {
                            (2 * sl) as usize // [bottom_left, left]
                        } else {
                            0
                        };
                        // mode_ctx = nearest_match + (newmv_count>0)*2 (AVM mvref_common.c:2796).
                        let mode_ctx = (core_inter_above[col] + core_inter_left) as usize
                            + if core_newmv_above[col] != 0 || core_newmv_left != 0 {
                                2
                            } else {
                                0
                            };
                        coder::emit_inter_skip_block(&mut enc, 12276, skip_ctx, mode_ctx);
                        for (dst_row, src_row) in rect_rows_mut(&mut recy, pw, sb_y, sb_x, 64, 64)
                            .zip(rect_rows(
                                &core_last_ref[0],
                                ref_ls,
                                ref_y0 + sb_y,
                                ref_x0 + sb_x,
                                64,
                                64,
                            ))
                        {
                            dst_row.copy_from_slice(src_row);
                        }
                        let (cy, cx) = (sb_y / 2, sb_x / 2);
                        let (rcy, rcx) = (ref_y0 / 2 + cy, ref_x0 / 2 + cx);
                        let dst_rows = rect_rows_mut(&mut recu, pcw, cy, cx, 32, 32)
                            .zip(rect_rows_mut(&mut recv, pcw, cy, cx, 32, 32));
                        let src_rows = rect_rows(&core_last_ref[1], ref_cs, rcy, rcx, 32, 32)
                            .zip(rect_rows(&core_last_ref[2], ref_cs, rcy, rcx, 32, 32));
                        for ((dst_u, dst_v), (src_u, src_v)) in dst_rows.zip(src_rows) {
                            dst_u.copy_from_slice(src_u);
                            dst_v.copy_from_slice(src_v);
                        }
                        core_skip_above[col] = 1;
                        core_skip_left = 1;
                        core_inter_above[col] = 1;
                        core_inter_left = 1;
                        core_newmv_above[col] = 0;
                        core_newmv_left = 0;
                        // GLOBALMV stores mv=0; it feeds the DRL[0] scan as a candidate.
                        core_mv_above[col] = Some(video::mv::Mv::ZERO);
                        core_mv_left = Some(video::mv::Mv::ZERO);
                        u_has[row * sb_cols + col] = 0;
                        v_has[row * sb_cols + col] = 0;
                        cfl_has[row * sb_cols + col] = 0;
                        // AVM av2_reset_entropy_context: skip_txfm=1 clears this SB's
                        // luma coeff entropy context (memset 0) so the next block's
                        // txb_skip ctx sees empty neighbors. 0x40 is our empty marker.
                        {
                            let cx = sb_x / 4;
                            let cy = sb_y / 4;
                            let ae = (cx + 16).min(above.len());
                            let le = (cy + 16).min(left.len());
                            for v in above[cx..ae].iter_mut() {
                                *v = 0x40;
                            }
                            for v in left[cy..le].iter_mut() {
                                *v = 0x40;
                            }
                        }
                        continue;
                    }
                }
                // NEWMV residual (CORE 64x64): search LAST; MC residual if a nonzero
                // MV beats the intra estimate.
                if core_has_last && sb_x + 64 <= width && sb_y + 64 <= height {
                    use crate::av2::video::{me, mv::Mv};
                    let bd = self.bit_depth as i32;
                    let (_ly, lu, lv) = (&core_last_ref[0], &core_last_ref[1], &core_last_ref[2]);
                    let (nmv_qstep, nmv_scale, _, _) =
                        aqs.per_sb_probe(&yp, pw, sb_y, sb_x, width, height);
                    // Frame-scoped buffers built once before the loop; borrow them.
                    let (bref, bstride, refh) = {
                        let (b, s, h) = inter_luma.as_ref().unwrap();
                        (b.as_slice(), *s, *h)
                    };
                    // Spatial MV predictors: zero, left, above, above-right. These
                    // seed the diamond so real camera/object motion is found without
                    // relying on the exhaustive grid alone (AVM DRL-style candidates).
                    let mut preds = me::MeCandidates::new();
                    if frame_mv_seed != Mv::ZERO {
                        preds.push_unique(frame_mv_seed);
                    }
                    for cand in [
                        core_mv_left,
                        if row > 0 { core_mv_above[col] } else { None },
                        if row > 0 && col + 1 < sb_cols {
                            core_mv_above[col + 1]
                        } else {
                            None
                        },
                    ]
                    .into_iter()
                    .flatten()
                    {
                        preds.push_unique(cand);
                    }
                    let ref_mv = core_mv_left
                        .or(if row > 0 { core_mv_above[col] } else { None })
                        .or(if row > 0 && col + 1 < sb_cols {
                            core_mv_above[col + 1]
                        } else {
                            None
                        })
                        .unwrap_or(Mv::ZERO);
                    let (mv, _) = me::search(
                        &me::MePlanes {
                            current: &yp[sb_y * pw + sb_x..],
                            current_stride: pw,
                            reference: bref,
                            reference_stride: bstride,
                        },
                        preds.as_slice(),
                        &me::MeSearchSpec {
                            origin_x: (ref_x0 + sb_x + BRD) as isize,
                            origin_y: (ref_y0 + sb_y + BRD) as isize,
                            width: 64,
                            height: 64,
                            reference_mv: ref_mv,
                            lambda_mv: (nmv_qstep as u32).max(1),
                            max_dx: self.video_search_range,
                            max_dy: self.video_search_range,
                            predictor_gate_sad_per_pixel: self.video_predictor_gate,
                            integer_satd_radius: self.video_integer_satd_radius,
                            bit_depth: self.bit_depth,
                            frame_width: bstride,
                            frame_height: refh + 2 * BRD,
                        },
                        &mut core_me_scratch,
                    );
                    let mut mv = mc::clamp_umv(
                        mv,
                        (ref_x0 + sb_x) as i32,
                        (ref_y0 + sb_y) as i32,
                        64,
                        64,
                        ref_ls as i32,
                        refh as i32,
                    );
                    // Drop the fixed magnitude clamp that discarded moderate camera
                    // motion the search had already found; the NEWMV attempt still
                    // requires nonzero MV (zero-motion is handled by the skip branch).
                    if mv.row != 0 || mv.col != 0 {
                        let mut pred_u = [0f32; 64 * 64];
                        mc::predict_with_tmp(
                            &mut pred_u,
                            64,
                            bref,
                            bstride,
                            &mc::MotionBlock {
                                origin_x: (ref_x0 + sb_x + BRD) as isize,
                                origin_y: (ref_y0 + sb_y + BRD) as isize,
                                mv,
                                width: 64,
                                height: 64,
                                bit_depth: self.bit_depth,
                            },
                            &mut core_mc_tmp,
                        );
                        let mut sse = rect_sse_f32(
                            &PlaneRect {
                                plane: &yp,
                                stride: pw,
                                y: sb_y,
                                x: sb_x,
                            },
                            &PlaneRect {
                                plane: &pred_u,
                                stride: 64,
                                y: 0,
                                x: 0,
                            },
                            64,
                            64,
                        );
                        // NEWMV residual vs intra: include the live mode/skip/DRL
                        // syntax and the searched MV's signaling cost.
                        use crate::av2::video::rd;
                        let upn = row > 0;
                        let lfn = col > 0;
                        let ia = core_inter_above[col] == 1;
                        let il = core_inter_left == 1;
                        enc.intra_inter_ctx = if upn && lfn {
                            let n_intra = (!il as u8) + (!ia as u8);
                            if n_intra == 2 { 3 } else { n_intra as usize }
                        } else if upn {
                            if ia { 0 } else { 3 }
                        } else if lfn {
                            if il { 0 } else { 3 }
                        } else {
                            0
                        };
                        let sa = core_skip_above[col];
                        let sl = core_skip_left;
                        let skip_ctx = if upn && lfn {
                            (sl + sa) as usize
                        } else if upn {
                            (2 * sa) as usize
                        } else if lfn {
                            (2 * sl) as usize
                        } else {
                            0
                        };
                        let mode_ctx = (core_inter_above[col] + core_inter_left) as usize
                            + if core_newmv_above[col] != 0 || core_newmv_left != 0 {
                                2
                            } else {
                                0
                            };
                        let pred_mv = ref_mv;
                        let bounded_pred_mv = mc::clamp_umv(
                            pred_mv,
                            (ref_x0 + sb_x) as i32,
                            (ref_y0 + sb_y) as i32,
                            64,
                            64,
                            ref_ls as i32,
                            refh as i32,
                        );
                        if pred_mv != Mv::ZERO && pred_mv != mv && bounded_pred_mv == pred_mv {
                            let searched_mv = mv;
                            let searched_sse = sse;
                            mc::predict_with_tmp(
                                &mut pred_u,
                                64,
                                bref,
                                bstride,
                                &mc::MotionBlock {
                                    origin_x: (ref_x0 + sb_x + BRD) as isize,
                                    origin_y: (ref_y0 + sb_y + BRD) as isize,
                                    mv: pred_mv,
                                    width: 64,
                                    height: 64,
                                    bit_depth: self.bit_depth,
                                },
                                &mut core_mc_tmp,
                            );
                            let near_sse = rect_sse_f32(
                                &PlaneRect {
                                    plane: &yp,
                                    stride: pw,
                                    y: sb_y,
                                    x: sb_x,
                                },
                                &PlaneRect {
                                    plane: &pred_u,
                                    stride: 64,
                                    y: 0,
                                    x: 0,
                                },
                                64,
                                64,
                            );
                            let searched_delta = searched_mv.diff(pred_mv);
                            debug_assert_ne!(searched_delta, Mv::ZERO);
                            if rd::prefer_nearmv(
                                &enc,
                                rd::NearMvRdSpec {
                                    skip_ctx,
                                    mode_ctx,
                                    skip_txfm: false,
                                    near_distortion: near_sse * rd::SS2_INTER_DIST_W,
                                    new_distortion: searched_sse * rd::SS2_INTER_DIST_W,
                                    new_mvd: Mv {
                                        row: searched_delta.row / 2,
                                        col: searched_delta.col / 2,
                                    },
                                    qstep: nmv_qstep as u32,
                                },
                            ) {
                                mv = pred_mv;
                                sse = near_sse;
                            } else {
                                mc::predict_with_tmp(
                                    &mut pred_u,
                                    64,
                                    bref,
                                    bstride,
                                    &mc::MotionBlock {
                                        origin_x: (ref_x0 + sb_x + BRD) as isize,
                                        origin_y: (ref_y0 + sb_y + BRD) as isize,
                                        mv: searched_mv,
                                        width: 64,
                                        height: 64,
                                        bit_depth: self.bit_depth,
                                    },
                                    &mut core_mc_tmp,
                                );
                            }
                        }
                        let inter_dist = sse * rd::SS2_INTER_DIST_W;
                        let delta = mv.diff(pred_mv);
                        let candidate_mode = if delta == Mv::ZERO { 0 } else { 2 };
                        let scaled_mvd = (candidate_mode == 2).then_some(Mv {
                            row: delta.row / 2,
                            col: delta.col / 2,
                        });
                        let syntax_rate = rd::inter_syntax_bits(
                            &enc,
                            skip_ctx,
                            mode_ctx,
                            false,
                            candidate_mode,
                            scaled_mvd,
                        );
                        let inter_cost = rd::rd_cost(inter_dist, syntax_rate, nmv_qstep as u32);
                        let intra_bound = rd::rd_cost(0.0, 16.0 * 64.0 * 64.0, nmv_qstep as u32);
                        if inter_cost < intra_bound {
                            static POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
                            let mut tus: [Vec<Coeff>; 4] =
                                [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                            for (i, &(ty, tx)) in POS.iter().enumerate() {
                                let (y0, x0) = (sb_y + ty, sb_x + tx);
                                let mut pblk = [0f32; 1024];
                                let mut resid = [0f32; 1024];
                                crate::av2::metrics::scaled_residual_f32(
                                    &mut resid,
                                    &yp[y0 * pw + x0..],
                                    &pred_u[ty * 64 + tx..],
                                    crate::av2::metrics::ResidualSpec {
                                        src_stride: pw,
                                        pred_stride: 64,
                                        width: 32,
                                        height: 32,
                                        scale: nmv_scale,
                                    },
                                );
                                for (dst, prediction) in pblk
                                    .as_chunks_mut::<32>()
                                    .0
                                    .iter_mut()
                                    .zip(rect_rows(&pred_u, 64, ty, tx, 32, 32))
                                {
                                    dst.copy_from_slice(prediction);
                                }
                                let lev = bases.luma.project(&resid[..], 0.0);
                                let rb =
                                    reconstruct_luma(&pblk, &lev, nmv_qstep, &tables::SCAN, bd);
                                put_block(&mut recy, pw, y0, x0, 32, &rb);
                                tus[i] = levels_to_coeffs(&lev);
                            }
                            let (cy, cx) = (sb_y / 2, sb_x / 2);
                            let mut uv_coeffs: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
                            for (pi, (src_c, ref_c, rec_c)) in
                                [(&up, lu, &mut recu), (&vp, lv, &mut recv)]
                                    .into_iter()
                                    .enumerate()
                            {
                                let _ = ref_c; // reference now borrowed frame-scoped
                                let cbrd = BRD / 2;
                                let (bcref, bcstride) = {
                                    let (b, s) = &inter_chroma.as_ref().unwrap()[pi];
                                    (b.as_slice(), *s)
                                };
                                let mut cpredu = [0f32; 32 * 32];
                                // 4:2:0 chroma is half-res: AVM scales the luma MV by
                                // (1<<(1-ss)) so chroma sees half the luma displacement.
                                let cmv = Mv {
                                    row: mv.row / 2,
                                    col: mv.col / 2,
                                };
                                mc::predict_with_tmp(
                                    &mut cpredu,
                                    32,
                                    bcref,
                                    bcstride,
                                    &mc::MotionBlock {
                                        origin_x: (ref_x0 / 2 + cx + cbrd) as isize,
                                        origin_y: (ref_y0 / 2 + cy + cbrd) as isize,
                                        mv: cmv,
                                        width: 32,
                                        height: 32,
                                        bit_depth: self.bit_depth,
                                    },
                                    &mut core_mc_tmp,
                                );
                                let mut cres = [0f32; 1024];
                                let mut cpred = [0i32; 1024];
                                crate::av2::metrics::scaled_residual_f32(
                                    &mut cres,
                                    &src_c[cy * pcw + cx..],
                                    &cpredu,
                                    crate::av2::metrics::ResidualSpec {
                                        src_stride: pcw,
                                        pred_stride: 32,
                                        width: 32,
                                        height: 32,
                                        scale: nmv_scale,
                                    },
                                );
                                for (pred, &reference) in cpred.iter_mut().zip(&cpredu) {
                                    *pred = reference as i32;
                                }
                                let lev = bases.chroma420.project(&cres[..], 0.0);
                                let rb = itx422::reconstruct_chroma_cfl(
                                    &cpred,
                                    &lev,
                                    nmv_qstep,
                                    &tables::SCAN,
                                    32,
                                    32,
                                    bd,
                                );
                                for (dst_row, src_row) in rect_rows_mut(rec_c, pcw, cy, cx, 32, 32)
                                    .zip(rb.as_chunks::<32>().0.iter())
                                {
                                    dst_row.copy_from_slice(src_row);
                                }
                                uv_coeffs[pi] = levels_to_coeffs(&lev);
                            }
                            let upn = row > 0;
                            let lfn = col > 0;
                            let ia = core_inter_above[col] == 1;
                            let il = core_inter_left == 1;
                            enc.intra_inter_ctx = if upn && lfn {
                                let n_intra = (!il as u8) + (!ia as u8);
                                if n_intra == 2 { 3 } else { n_intra as usize }
                            } else if upn {
                                if ia { 0 } else { 3 }
                            } else if lfn {
                                if il { 0 } else { 3 }
                            } else {
                                0
                            };
                            let sa = core_skip_above[col];
                            let sl = core_skip_left;
                            let skip_ctx = if upn && lfn {
                                (sl + sa) as usize
                            } else if upn {
                                (2 * sa) as usize
                            } else if lfn {
                                (2 * sl) as usize
                            } else {
                                0
                            };
                            let mode_ctx = (core_inter_above[col] + core_inter_left) as usize
                                + if core_newmv_above[col] != 0 || core_newmv_left != 0 {
                                    2
                                } else {
                                    0
                                };
                            // DRL[0] predictor (AVM scan, reorder disabled, single LAST,
                            // no TMVP): left SB's MV, else above, else above-right, else 0.
                            let pred_mv = core_mv_left
                                .or(if row > 0 { core_mv_above[col] } else { None })
                                .or(if row > 0 && col + 1 < sb_cols {
                                    core_mv_above[col + 1]
                                } else {
                                    None
                                })
                                .unwrap_or(Mv::ZERO);
                            let mvd_row = mv.row - pred_mv.row;
                            let mvd_col = mv.col - pred_mv.col;
                            // mv == DRL[0] -> NEARMV (no MVD); else NEWMV.
                            let inter_mode = if mvd_row == 0 && mvd_col == 0 { 0 } else { 2 };
                            enc.delta_q_pending = enc.delta_q_present;
                            // Dense inter-TX32 residual coeff coding isn't bit-exact yet;
                            // reject NEWMV for dense TUs and let the block fall to intra.
                            let max_tu_nz = tus.iter().map(|t| t.len()).max().unwrap_or(0);
                            if max_tu_nz <= NEWMV_MAX_TU_NZ {
                                let any_resid = tus.iter().any(|t| !t.is_empty())
                                    || !uv_coeffs[0].is_empty()
                                    || !uv_coeffs[1].is_empty();
                                if any_resid {
                                    // Commit AQ only for residual blocks (skip SBs emit
                                    // no delta_q; decoder qindex unchanged).
                                    let _ =
                                        aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height);
                                    // Chroma skip ctxs from the neighbor presence grids
                                    // (matches the decoder's get_txb_skip_ctx / V offset).
                                    let at = |g: &[i32], dr: usize, dc: usize| {
                                        g[(row - dr) * sb_cols + (col - dc)]
                                    };
                                    let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                                    let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                                    let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                                    let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                                    let u_present = uv_coeffs[0].iter().any(|&(_, l)| l != 0);
                                    let v_present = uv_coeffs[1].iter().any(|&(_, l)| l != 0);
                                    let u_skip = (6 + ua + ul) as u32;
                                    let v_skip = (6 * (u_present as i32) + va + vl) as u32;
                                    let (tu_skip_cdfs, tu_dc_sign) = sb_tu_contexts(
                                        &tus,
                                        sb_y,
                                        sb_x,
                                        &mut above,
                                        &mut left,
                                        enc.qc,
                                        (pw / 4) as i64,
                                        (ph / 4) as i64,
                                    );
                                    coder::emit_inter_newmv_residual_block_multi(
                                        &mut enc,
                                        &coder::InterResidualSpec {
                                            part_cdf: 12276,
                                            skip_ctx,
                                            mode_ctx,
                                            drl_ctx: mode_ctx,
                                            mode: inter_mode,
                                            scaled_row: mvd_row / 2,
                                            scaled_col: mvd_col / 2,
                                            luma_tus: &tus,
                                            luma_skip_cdfs: &tu_skip_cdfs,
                                            luma_dc_sign_ctxs: &tu_dc_sign,
                                        },
                                        &coder::InterChromaTus {
                                            u_tus: std::slice::from_ref(&uv_coeffs[0]),
                                            v_tus: std::slice::from_ref(&uv_coeffs[1]),
                                            u_skip_cdfs: &[u_skip],
                                            v_skip_cdfs: &[v_skip],
                                            u_tx64: false,
                                        },
                                    );
                                    u_has[row * sb_cols + col] = u_present as i32;
                                    v_has[row * sb_cols + col] = v_present as i32;
                                    cfl_has[row * sb_cols + col] = 0;
                                } else {
                                    crate::av2::coder::emit_inter_mode_block(
                                        &mut enc,
                                        12276,
                                        skip_ctx,
                                        mode_ctx,
                                        mode_ctx,
                                        inter_mode,
                                        mvd_row / 2,
                                        mvd_col / 2,
                                    );
                                    // skip_txfm=1: clear this SB's luma coeff entropy
                                    // context (AVM av2_reset_entropy_context).
                                    let cx = sb_x / 4;
                                    let cy = sb_y / 4;
                                    let ae = (cx + 16).min(above.len());
                                    let le = (cy + 16).min(left.len());
                                    for v in above[cx..ae].iter_mut() {
                                        *v = 0x40;
                                    }
                                    for v in left[cy..le].iter_mut() {
                                        *v = 0x40;
                                    }
                                    u_has[row * sb_cols + col] = 0;
                                    v_has[row * sb_cols + col] = 0;
                                    cfl_has[row * sb_cols + col] = 0;
                                }
                                core_skip_above[col] = if any_resid { 0 } else { 1 };
                                core_skip_left = if any_resid { 0 } else { 1 };
                                core_inter_above[col] = 1;
                                core_inter_left = 1;
                                core_newmv_above[col] = (inter_mode == 2) as u8;
                                core_newmv_left = (inter_mode == 2) as u8;
                                core_mv_above[col] = Some(mv);
                                core_mv_left = Some(mv);
                                continue;
                            }
                        }
                    }
                }
                // Intra block in inter frame: intra_inter ctx from neighbor inter-ness
                // (mirrors GLOBALMV branch; AVM neighbor doubling).
                {
                    let up = row > 0;
                    let lf = col > 0;
                    let ia = core_inter_above[col] == 1;
                    let il = core_inter_left == 1;
                    enc.intra_inter_ctx = if up && lf {
                        let n_intra = (!il as u8) + (!ia as u8);
                        if n_intra == 2 { 3 } else { n_intra as usize }
                    } else if up {
                        if ia { 0 } else { 3 }
                    } else if lf {
                        if il { 0 } else { 3 }
                    } else {
                        0
                    };
                }
                core_skip_above[col] = 0;
                core_skip_left = 0;
                core_inter_above[col] = 0;
                core_inter_left = 0;
                core_newmv_above[col] = 0;
                core_newmv_left = 0;
                core_mv_above[col] = None;
                core_mv_left = None;
                // Per-SB adaptive quantization: pick this SB's qstep from source
                // activity (signaling the delta), then quantize+reconstruct luma and
                // chroma at it. (qstep_i, 1.0) when AQ is off.
                let (sb_qstep, sb_resid_scale, _, _) =
                    aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height);
                // Staged replay: capture/replay the whole-64 luma mode-search winner
                // (skips `encode_luma_sb` in pass 2). One push/pop per SB reaching this
                // fast path — deterministic, so the `luma420` cursor stays aligned.
                let replay_l = if let crate::av2::replay::DecideMode::Replay(cur) = &mut decide_mode
                {
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
                        &mut recy,
                        &LumaSource {
                            plane: &yp,
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
                let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                    &tus,
                    sb_y,
                    sb_x,
                    &mut above,
                    &mut left,
                    qc,
                    (pw / 4) as i64,
                    (ph / 4) as i64,
                );
                let (cy, cx) = (sb_y / 2, sb_x / 2);
                // CfL decision (4:2:0 whole-64 fast path). recy is final after
                // encode_luma_sb; sets the per-block CfL state read by encode_intra_modes
                // during the luma encode below. Context from the per-SB CfL-usage grid.
                let cfl_a = if row > 0 {
                    cfl_has[(row - 1) * sb_cols + col]
                } else {
                    0
                };
                let cfl_l = if col > 0 {
                    cfl_has[row * sb_cols + col - 1]
                } else {
                    0
                };
                enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                let cfl_choice = if enc.cfl {
                    let bd = self.bit_depth as i32;
                    let avg_l = cfl::cfl_avg_l(&recy, pw, sb_y, sb_x, 32, 32, true, true, bd);
                    let mut suf = [0f32; 32 * 32];
                    let mut svf = [0f32; 32 * 32];
                    cfl_partition_prediction::<32>(pcw, &up, &vp, cy, cx, &mut suf, &mut svf);
                    let dc_u_f = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                    let dc_v_f = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                    cfl::cfl_decide(
                        &cfl::CflDecisionInput {
                            reconstructed_luma: &recy,
                            luma_stride: pw,
                            luma_y: sb_y,
                            luma_x: sb_x,
                            source_u: suf.as_slice(),
                            source_v: svf.as_slice(),
                            dc_u: dc_u_f,
                            dc_v: dc_v_f,
                            width: 32,
                            height: 32,
                            subsample_x: true,
                            subsample_y: true,
                            luma_average_q3: avg_l,
                        },
                        &cfl::ChromaRdSpec {
                            basis: &bases.chroma420,
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
                    enc.uv_mode = 0;
                } else {
                    enc.cfl_use = false;
                    enc.cfl_signaled = false;
                    enc.uv_mode = 0;
                }
                // MHCCP: multi-parameter cross-component predictor competing with
                // the incumbent (DC or classic CfL) on J = SSE + lambda*bits.
                let mhccp_choice = if enc.mhccp {
                    let bd = self.bit_depth as i32;
                    let lam = leaf::part_lambda(qstep_i, self.tune.part_lambda_c);
                    let dc_u_f = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                    let dc_v_f = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                    // Chroma source blocks (same extraction cfl_decide uses).
                    let mut suf_m = [0f32; 32 * 32];
                    let mut svf_m = [0f32; 32 * 32];
                    cfl_partition_prediction::<32>(pcw, &up, &vp, cy, cx, &mut suf_m, &mut svf_m);
                    let baseline_j = if let Some(ref ch) = cfl_choice {
                        let alpha_bits = 2.0
                            + 3.0
                            + if ch.sign_u != 0 { 3.0 } else { 0.0 }
                            + if ch.sign_v != 0 { 3.0 } else { 0.0 };
                        cfl::score_predictor(
                            &cfl::PredictorScoreInput {
                                source_u: suf_m.as_slice(),
                                source_v: svf_m.as_slice(),
                                predictor_u: &ch.pred_u,
                                predictor_v: &ch.pred_v,
                                width: 32,
                                height: 32,
                                mode_bits: alpha_bits,
                            },
                            &cfl::ChromaRdSpec {
                                basis: &bases.chroma420,
                                qstep: qstep_i,
                                lambda: lam,
                                bit_depth: bd,
                            },
                        )
                    } else {
                        let dc_u_pred = [dc_u_f.round() as i32; 32 * 32];
                        let dc_v_pred = [dc_v_f.round() as i32; 32 * 32];
                        cfl::score_predictor(
                            &cfl::PredictorScoreInput {
                                source_u: suf_m.as_slice(),
                                source_v: svf_m.as_slice(),
                                predictor_u: &dc_u_pred,
                                predictor_v: &dc_v_pred,
                                width: 32,
                                height: 32,
                                mode_bits: 0.0,
                            },
                            &cfl::ChromaRdSpec {
                                basis: &bases.chroma420,
                                qstep: qstep_i,
                                lambda: lam,
                                bit_depth: bd,
                            },
                        )
                    };
                    let mctx = cfl::MhccpCtx {
                        recy: &recy,
                        pw,
                        recu: &recu,
                        recv: &recv,
                        pcw,
                        bounds: cfl::MhccpBounds::from_luma(width, height, true, true),
                        ly: sb_y,
                        lx: sb_x,
                        cy,
                        cx,
                        ssx: true,
                        ssy: true,
                        have_top: row > 0,
                        have_left: col > 0,
                        // Whole-superblock chroma block: its top edge is the SB
                        // boundary, so one padding line above (avm is_top_sb_boundary).
                        is_top_sb_boundary: true,
                        size_group: cfl::mhccp_size_group_32(),
                        // Whole-SB block: top-right / bottom-left resolve via the
                        // prev-SB-row/col special cases, so no coded-mi mask needed.
                        coded: &[],
                    };
                    cfl::mhccp_decide(
                        &mctx,
                        &cfl::MhccpDecisionInput {
                            source_u: suf_m.as_slice(),
                            source_v: svf_m.as_slice(),
                            width: 32,
                            height: 32,
                            rd: cfl::ChromaRdSpec {
                                basis: &bases.chroma420,
                                qstep: qstep_i,
                                lambda: lam,
                                bit_depth: bd,
                            },
                            scan: &tables::SCAN,
                            baseline_j,
                        },
                    )
                } else {
                    None
                };
                // This 32x32-partition leaf is MHCCP-eligible (block size in
                // [8x8,64x64], not 4x4). Mark it so the switch symbol is emitted
                // on this CfL block, matching the decoder's is_mhccp_allowed gate.
                enc.mhccp_allowed = enc.mhccp;
                if let Some(mh) = mhccp_choice.as_ref().and_then(|c| c.mhccp) {
                    enc.cfl_use = true;
                    enc.mhccp_use = true;
                    enc.mhccp_dir = mh.mh_dir;
                    enc.mhccp_size_group = mh.size_group;
                    enc.uv_mode = 0;
                } else {
                    enc.mhccp_use = false;
                }
                // When MHCCP wins, its CflChoice (carrying the MHCCP predictor in
                // pred_u/pred_v) replaces the incumbent so the shared cfl_choice
                // reconstruction path below uses the MHCCP prediction.
                let cfl_choice = if enc.mhccp_use {
                    mhccp_choice
                } else {
                    cfl_choice
                };
                // Chroma intra-mode search (non-CfL). Runs BEFORE the luma/uv mode
                // emitter (encode_luma_block_split_dir) so `enc.uv_mode` is set when
                // the uv-mode symbol is written. AV2 shares one uv-mode across U+V;
                // at TX_32X32 the ext-tx set is DCT/IDTX so every candidate still
                // reconstructs with DCT_DCT — the search improves prediction only.
                // The chosen levels + reconstruction are cached and reused below.
                let bd = self.bit_depth as i32;
                #[allow(clippy::type_complexity)]
                let chroma_search: Option<(
                    Vec<f32>,
                    Vec<f32>,
                    Vec<f32>,
                    Vec<f32>,
                )> = if cfl_choice.is_none() {
                    let dcu = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                    let dcv = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                    let cand_modes: &[usize] = if !self.tune.chroma_mode_search {
                        &[0] // search disabled: DC only (byte-identical to baseline)
                    } else if self.speed.reduced_modes() {
                        &[0, 1, 4]
                    } else {
                        &[0, 1, 2, 3, 4]
                    };
                    let lambda = leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
                    let mut best_mode = 0usize;
                    let mut best_cost = f32::INFINITY;
                    let mut best = None;
                    let predict_uv = |m: usize| -> (Vec<f32>, Vec<f32>) {
                        if m == 0 {
                            (vec![dcu; 32 * 32], vec![dcv; 32 * 32])
                        } else {
                            (
                                chroma422::predict_chroma_mode_dims(
                                    &recu,
                                    pcw,
                                    cy,
                                    cx,
                                    32,
                                    m,
                                    neutral,
                                    width / 2,
                                    height / 2,
                                ),
                                chroma422::predict_chroma_mode_dims(
                                    &recv,
                                    pcw,
                                    cy,
                                    cx,
                                    32,
                                    m,
                                    neutral,
                                    width / 2,
                                    height / 2,
                                ),
                            )
                        }
                    };
                    // SATD prune: each chroma mode is one 32x32 prediction from
                    // neighbours (independent) — rank by SATD over U+V, full-encode top-K.
                    let keep_uv = if self.speed.reduced_modes() { 2 } else { 3 };
                    let cand_modes: Vec<usize> = if cand_modes.len() > keep_uv {
                        let mut r: Vec<(u64, usize)> = cand_modes
                            .iter()
                            .map(|&m| {
                                let (pu, pv) = predict_uv(m);
                                (
                                    crate::av2::metrics::satd_f32(
                                        &up[cy * pcw + cx..],
                                        pcw,
                                        &pu,
                                        32,
                                        32,
                                        32,
                                    ) + crate::av2::metrics::satd_f32(
                                        &vp[cy * pcw + cx..],
                                        pcw,
                                        &pv,
                                        32,
                                        32,
                                        32,
                                    ),
                                    m,
                                )
                            })
                            .collect();
                        r.sort_by_key(|&(s, _)| s);
                        r.truncate(keep_uv);
                        r.into_iter().map(|(_, m)| m).collect()
                    } else {
                        cand_modes.to_vec()
                    };
                    for &m in &cand_modes {
                        let (pu, pv) = predict_uv(m);
                        let pu_i: Vec<i32> = pu.iter().map(|&p| (p + 0.5) as i32).collect();
                        let pv_i: Vec<i32> = pv.iter().map(|&p| (p + 0.5) as i32).collect();
                        let mut ru = [0f32; 32 * 32];
                        let mut rv = [0f32; 32 * 32];
                        let dst_rows = ru
                            .as_chunks_mut::<32>()
                            .0
                            .iter_mut()
                            .zip(rv.as_chunks_mut::<32>().0.iter_mut());
                        let src_rows = rect_rows(&up, pcw, cy, cx, 32, 32)
                            .zip(rect_rows(&vp, pcw, cy, cx, 32, 32));
                        let pred_rows = pu
                            .as_chunks::<32>()
                            .0
                            .iter()
                            .zip(pv.as_chunks::<32>().0.iter());
                        for (((ru_row, rv_row), (u_row, v_row)), (pu_row, pv_row)) in
                            dst_rows.zip(src_rows).zip(pred_rows)
                        {
                            for (((ru, rv), (&u, &v)), (&pred_u, &pred_v)) in ru_row
                                .iter_mut()
                                .zip(rv_row)
                                .zip(u_row.iter().zip(v_row))
                                .zip(pu_row.iter().zip(pv_row))
                            {
                                *ru = u - pred_u;
                                *rv = v - pred_v;
                            }
                        }
                        let lu = chroma422::project_chroma_rdoq(
                            &bases.chroma420,
                            &aq::scale_resid(&ru, sb_resid_scale),
                            &tables::SCAN,
                            qc,
                            1024,
                            0,
                            self.tune.chroma_rdoq_lambda,
                        );
                        let lv = chroma422::project_chroma_rdoq(
                            &bases.chroma420,
                            &aq::scale_resid(&rv, sb_resid_scale),
                            &tables::SCAN,
                            qc,
                            1024,
                            4,
                            self.tune.chroma_rdoq_lambda,
                        );
                        let recu_b = itx422::reconstruct_chroma_cfl(
                            &pu_i,
                            &lu,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        );
                        let recv_b = itx422::reconstruct_chroma_cfl(
                            &pv_i,
                            &lv,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        );
                        let sse = pixel_sse_rounded_block(&up, pcw, cy, cx, &recu_b, 32, 32, 32)
                            + pixel_sse_rounded_block(&vp, pcw, cy, cx, &recv_b, 32, 32, 32);
                        let rate = coeff_abs_rate_f32(&lu) + coeff_abs_rate_f32(&lv);
                        // Small bias toward DC to avoid spending uv-mode bits for a
                        // marginal SSE gain.
                        let mode_bits = if m == 0 { 0.0 } else { 2.0 };
                        let cost = sse + lambda * (rate + mode_bits);
                        if cost < best_cost {
                            best_cost = cost;
                            best_mode = m;
                            best = Some((lu, lv, recu_b, recv_b));
                        }
                    }
                    enc.uv_mode = best_mode;
                    best
                } else {
                    None
                };
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
                // Arm this SB's delta-Q (value already set above); the mode emitter
                // consumes it after the partition bit, exactly once per SB.
                enc.delta_q_pending = enc.delta_q_present;
                // Arm this SB's CCSO flag (consumed before delta-Q). (row, col) feed
                // the neighbor-based context.
                enc.ccso_pending = enc.ccso_u_enable || enc.ccso_v_enable;
                enc.ccso_sb_rc = (row, col);
                let (_cul, sb_midx) = encode_luma_block_split_dir(
                    &mut enc,
                    &LumaSplitDirSpec {
                        tus: &tus,
                        skip_cdfs: &skip_cdfs,
                        dc_sign_ctxs: &dc_sign_ctxs,
                        mode_idx,
                        angle_delta: adelta,
                        has_chroma: true,
                        part_cdf: 12276,
                        left_midx: lmidx,
                        above_midx: amidx,
                    },
                );
                midx_grid[row * sb_cols + col] = sb_midx;

                let (levu, levv) = if let Some(ref ch) = cfl_choice {
                    let mut ru = [0f32; 32 * 32];
                    let mut rv = [0f32; 32 * 32];
                    cfl_prediction::<32>(pcw, &up, &vp, cy, cx, &ch, &mut ru, &mut rv);
                    let levu = chroma422::project_chroma_rdoq(
                        &bases.chroma420,
                        &aq::scale_resid(&ru, sb_resid_scale),
                        &tables::SCAN,
                        qc,
                        1024,
                        0,
                        self.tune.chroma_rdoq_lambda,
                    );
                    let levv = chroma422::project_chroma_rdoq(
                        &bases.chroma420,
                        &aq::scale_resid(&rv, sb_resid_scale),
                        &tables::SCAN,
                        qc,
                        1024,
                        4,
                        self.tune.chroma_rdoq_lambda,
                    );
                    put_block(
                        &mut recu,
                        pcw,
                        cy,
                        cx,
                        32,
                        &itx422::reconstruct_chroma_cfl(
                            &ch.pred_u,
                            &levu,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        ),
                    );
                    put_block(
                        &mut recv,
                        pcw,
                        cy,
                        cx,
                        32,
                        &itx422::reconstruct_chroma_cfl(
                            &ch.pred_v,
                            &levv,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        ),
                    );
                    (levu, levv)
                } else {
                    // Reuse the chroma intra-mode search already performed (and
                    // signaled via enc.uv_mode) before the luma/uv-mode emitter.
                    let (levu, levv, recu_b, recv_b) =
                        chroma_search.expect("non-CfL chroma path must have a cached mode search");
                    put_block(&mut recu, pcw, cy, cx, 32, &recu_b);
                    put_block(&mut recv, pcw, cy, cx, 32, &recv_b);
                    (levu, levv)
                };
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = (6 + ua + ul) as u32;
                encode_chroma_block_ex(&mut enc, &ucoeffs, u_skip, true, false);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip = (6 * (u_present as i32) + va + vl) as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
                cfl_has[row * sb_cols + col] = cfl_choice.is_some() as i32;
            }
        }
        // Real deblocking: apply the AV2 loop filter to the reconstruction so the
        // encoder recon matches the decoder (post-deblock), CDEF searches the filtered
        // input, and CCSO reads the decoder's post-deblock luma. Decoder order is
        // deblock -> (snapshot ext_rec_y) -> CDEF -> CCSO, so this runs before them.
        // Whole-SB path: synthesize one 64x64 luma leaf per SB (SPLIT tx -> 4x 32x32).
        // Gated to AQ off here (uniform qindex); the partition path handles AQ.
        let filter_active = self.tune.deblock && df_quant >= 1;
        if filter_active && !enc.delta_q_present {
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
            let sb_qidx = vec![self.base_q_idx as u16; sb_rows * sb_cols];
            crate::av2::deblock::deblock_frame(crate::av2::deblock::FrameDeblock {
                recy: &mut recy,
                recu: &mut recu,
                recv: &mut recv,
                luma_stride: pw,
                chroma_stride: pcw,
                width,
                height,
                ssx: 1,
                ssy: 1,
                has_chroma: true,
                chroma_deblock: self.tune.chroma_deblock,
                bit_depth: self.bit_depth as u32,
                delta_y: eff_dy,
                delta_uv: self.tune.db_delta_uv,
                base_q: self.base_q_idx as u16,
                sb_cols,
                sb_qidx: &sb_qidx,
                leaves: &leaves,
                chroma_leaves: &leaves,
                skip_leaves: &[],
                luma_tx_cap: (8, 8),

                chroma_tx_cap: (8, 8),
            });
        }
        // Per-block CDEF pass 1 (single-tile whole-frame): decide the per-SB grid
        // from this reconstruction. Runs BEFORE the CCSO block below, which mutates
        // recon to the post-CCSO result — the decoder applies CDEF before CCSO, so
        // the CDEF decision must see the pre-CCSO recon. The multi-tile path forces
        // `capture_recon` and searches the stitched recon instead.
        if self.tune.cdef && cdef_pre.is_none() && !replay::in_tile_subencode() {
            enc.cdef_decided = crate::av2::cdef_est::search_per_block(
                &[recy.clone(), recu.clone(), recv.clone()],
                &[yp.clone(), up.clone(), vp.clone()],
                pw,
                ph,
                pcw,
                pch,
                1,
                1,
                true,
                self.base_q_idx,
                self.bit_depth,
            );
        }
        if let Some(pre) = cdef_pre {
            let mut filtered_recon = vec![recy, recu, recv];
            crate::av2::cdef_est::apply_per_block(
                &mut filtered_recon,
                pw,
                ph,
                pcw,
                pch,
                1,
                1,
                true,
                pre,
                self.bit_depth,
            );
            let mut planes = filtered_recon.into_iter();
            recy = planes.next().unwrap();
            recu = planes.next().unwrap();
            recv = planes.next().unwrap();
        }
        // CCSO (U and V planes, edge-classified). The per-SB flags were already
        // emitted (all on). Build the border-extended luma once, search the best
        // filter LUT per plane against source, apply it to recon so the encoder
        // output matches the decoder's post-filter result. 4:2:0 => hscale=vscale=1.
        if ccso_search_u || ccso_search_v || ccso_pre.is_some() {
            let (ext, estride) = crate::av2::ccso::extend_luma(&recy, pw, ph);
            let bd = self.bit_depth as u32;
            let sb_cols = pw / 64;
            let sb_rows = ph / 64;
            let ccso_geometry = ccso::CcsoGeometry {
                extended_luma: &ext,
                extended_stride: estride,
                width: pcw,
                height: pch,
                horizontal_subsampling: 1,
                vertical_subsampling: 1,
                bit_depth: bd,
            };
            // Approximate RD multiplier from the frame qstep: rate (bits) weighs
            // ~rd_mult against SSE. Tuned so the per-SB flag cost is comparable to a
            // meaningful SSE change. rd_scale lets the threshold be retuned.
            let qstep = quant::qstep(self.base_q_idx as u32) as f32;
            let rd_scale: f32 = self.tune.ccso_rd_scale;
            // RD multiplier matching AVM's RDCOST scaling. AVM compares
            // `(ssd << 7) + (rate*rdmult >> 9)`; dividing through by 128 gives
            // `ssd + rate * rdmult/65536`, so our per-SB rd_mult (weight on the
            // flag's bit cost vs SSE) is `rdmult / 65536`, with `rdmult ~ c*qstep^2`.
            // rd_scale folds the AVM constant (~0.85) and a tuning factor together.
            let rd_mult = rd_scale * qstep * qstep / 1048576.0;
            if let Some(pre) = ccso_pre {
                // Emit pass: apply the precomputed result gated by the grid.
                if let Some((r, grid)) = &pre.u {
                    ccso::apply_edge_gated(
                        &ccso_geometry,
                        &mut recu,
                        r,
                        &ccso::CcsoGrid {
                            flags: grid,
                            sb_cols: pre.sb_cols,
                        },
                    );
                    enc.ccso_u_result = Some(edge_to_plane(r));
                }
                if let Some((r, grid)) = &pre.v {
                    ccso::apply_edge_gated(
                        &ccso_geometry,
                        &mut recv,
                        r,
                        &ccso::CcsoGrid {
                            flags: grid,
                            sb_cols: pre.sb_cols,
                        },
                    );
                    enc.ccso_v_result = Some(edge_to_plane(r));
                }
            } else {
                let dec_rd = rd_mult;
                if ccso_search_u
                    && let Some(r) = ccso::search_edge(
                        &ccso::CcsoPlane {
                            geometry: ccso_geometry,
                            source: &up,
                            reconstruction: &recu,
                        },
                        None,
                    )
                {
                    let (grid, any) = ccso::decide_blk_md(
                        &ccso::CcsoPlane {
                            geometry: ccso_geometry,
                            source: &up,
                            reconstruction: &recu,
                        },
                        &r,
                        &ccso::CcsoDecisionSpec {
                            sb_cols,
                            sb_rows,
                            rd_mult: dec_rd,
                            plane: 1,
                        },
                    );
                    if any {
                        enc.ccso_decided_u = Some((r, grid));
                    }
                }
                if ccso_search_v
                    && let Some(r) = ccso::search_edge(
                        &ccso::CcsoPlane {
                            geometry: ccso_geometry,
                            source: &vp,
                            reconstruction: &recv,
                        },
                        None,
                    )
                {
                    let (grid, any) = ccso::decide_blk_md(
                        &ccso::CcsoPlane {
                            geometry: ccso_geometry,
                            source: &vp,
                            reconstruction: &recv,
                        },
                        &r,
                        &ccso::CcsoDecisionSpec {
                            sb_cols,
                            sb_rows,
                            rd_mult: dec_rd,
                            plane: 2,
                        },
                    );
                    if any {
                        enc.ccso_decided_v = Some((r, grid));
                    }
                }
                enc.ccso_sb_cols_out = sb_cols;
            }
        }
        // Video: stash final recon (Y,U,V) for the DPB; skipped for still tiles.
        if self
            .capture_recon
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.recon = vec![recy.clone(), recu.clone(), recv.clone()];
        }
        enc
    }
}
