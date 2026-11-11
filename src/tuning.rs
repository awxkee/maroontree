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

#[derive(Clone, Copy)]
pub(crate) struct Tuning {
    /// Rectangle-vs-square balance at the 16x16 node, non-4:2:2.
    pub(crate) rect16_bias: f32,
    /// Same, 4:2:2 (measured strongly positive there, hence untaxed).
    pub(crate) rect16_bias_422: f32,
    /// PARTITION_HORZ search at 4:4:4. Shipped OFF — the leg was disabled when
    /// the leaves were under-tooled; this reopens it for the study.
    pub(crate) rect16_horz_444: bool,
    /// PARTITION_VERT search at 4:4:4. Shipped OFF.
    pub(crate) rect16_vert_444: bool,
    /// Allow the 4:2:0 HORZ/VERT legs at Medium, not just Slow.
    pub(crate) rect16_420_medium: bool,
    /// NONE-vs-split bias at the top quality band, per format.
    pub(crate) none16_top_bias_420: f32,
    pub(crate) none16_top_bias_422: f32,
    pub(crate) none16_top_bias_444: f32,
    /// Tax on the under-tooled four-strip (HORZ_4/VERT_4) proxies.
    pub(crate) quad4_bias: f32,
    /// Top-K size for the non-square 16-level partition staging, per tier.
    pub(crate) part_budget_slow: u32,
    /// Minimum frame `base_q_idx` for the non-square staging to run at all.
    pub(crate) part_budget_qmin: u32,
    /// Restrict the non-square partition budget to 4:4:4.
    pub(crate) part_budget_444_only: bool,
    /// Rect16 leaf (every chroma format): allow RTX_16X8/8X16 -> two TX_8X8
    /// (tx_depth 1) in the luma transform trial (estimator and emitter).
    pub(crate) rect_tx_split: bool,
    /// 4:2:0 rect16 leaf: full chroma uv-mode search on the 8x4/4x8 chroma
    /// (was DC/CfL only).
    pub(crate) rect_uv_search: bool,
    /// 4:2:0 rect16 leaf: joint (mode, delta) SATD shortlist for the 8x4/4x8
    /// chroma (tr/bl-free set), full RD per kept pair
    pub(crate) rect_uv_joint: bool,
    /// Shortlist size of `rect_uv_joint`.
    pub(crate) rect_uv_joint_k: u32,
    /// Rect16 leaf luma: decoder-exact edge flags (dav1d `init_edges` for the
    /// halves.
    pub(crate) rect_full_modes: bool,
    /// H4/V4 strip leaves (16x4 / 4x16)
    pub(crate) strip_modes: bool,
    /// H4/V4 strip pairs at 4:2:0.
    pub(crate) strip_uv_joint: bool,
    /// Rect16 leaf: re-decide whole-TX vs two TX_8X8 at the coding site with
    /// the decoder-faithful split (half 1 predicted from the reconstructed
    /// half 0) instead of the whole-prediction estimate alone.
    pub(crate) rect_split_seq: bool,
    /// Pre-fill the tile reconstruction with the source so decision-stage
    /// predictions of not-yet-coded neighbors (later leaves of a candidate
    /// partition) see a proxy of the eventual reconstruction, not zeros.
    pub(crate) recon_src_init: bool,
    /// Multiplier on the 16-level SPLIT leg (all bands; `none16_top_bias_*`
    /// is the top-band NONE-side counterpart).
    pub(crate) split16_bias: f32,
    /// Multiplier on the 8x8 SPLIT4 (four BLOCK_4X4) leg inside the 16-level
    /// decision.
    pub(crate) split4_bias: f32,
    pub(crate) part_budget_medium: u32,
    pub(crate) part_budget_fast: u32,

    /// Master strength of the top-quality ease ramp (`quant::TOP_EASE`).
    pub(crate) top_ease: f32,
    /// `top_ease_t` ramp: full strength at qindex 0, zero at `knee`, over
    /// `width` qindex steps. Shipped (20, 12) => active only above ~quality 89.
    pub(crate) top_ease_knee: f32,
    pub(crate) top_ease_width: f32,
    /// `qm_high_quality_t` ramp, same shape. Shipped (12, 5) => ~quality 93+.
    pub(crate) qm_hq_knee: f32,
    pub(crate) qm_hq_width: f32,
    /// `top_band()` qindex ceiling: which frames count as "top band" at all.
    pub(crate) top_band_q: u32,
    /// NONE-vs-SPLIT thumbs at 32 and 64 (the SATD proxy under-values the
    /// detail a merged large block loses). Re-fit jointly with the Phase 3
    /// bridge (study `p3bridge`, 2026-11-06): 1.054 / 1.249; resetting them
    /// to 1.0 under the bridge costs +1.2..+1.6.
    pub(crate) none32_split_bias: f32,
    pub(crate) none64_split_bias: f32,
    /// 4:2:0 top-band NONE bias, split at qindex 20.
    pub(crate) top_none_bias_420_hi: f32,
    pub(crate) top_none_bias_420_lo: f32,
    /// 4:4:4 top-band chroma AC qindex delta (`quant::TOP_444_UAC`).
    pub(crate) top_444_uac: f32,
    /// Top-band tilt of the SSIMULACRA2 mode-lambda weight
    pub(crate) mode_lambda_lo_floor: f32,
    pub(crate) mode_lambda_lo_full: f32,
    /// Weight at/below `mode_lambda_lo_floor` (1.0 = SVT ramp-down; >1.56 = ramp-up).
    pub(crate) mode_lambda_lo_w: f32,
    /// `loop_filter_sharpness` (0..=7), see `obu::loop_filter_sharpness`. SHIPPED 7.
    pub(crate) lf_sharpness: u32,
    /// Scale on the q-derived deblock levels (luma / chroma / 4:2:0 chroma).
    pub(crate) lf_level_scale: f32,
    pub(crate) lf_uv_level_scale: f32,
    pub(crate) lf_uv_level_scale_420: f32,
    /// Frame-level deblock level search (aom `LPF_PICK_FROM_FULL_IMAGE_NON_DUAL`
    /// style bisection on the stitched recon, SSE vs source) instead of the q-law.
    pub(crate) lf_search: bool,
    /// Scale on aom's bias against raising the level (1.0 = aom).
    pub(crate) lf_search_bias: f32,
    /// 0 = search starts at the q-law levels, 1 = starts at 0 (aom intra).
    pub(crate) lf_search_start0: bool,
    /// 0 = SSE, 1 = libaom perceptual `cdef_dist_8x8` on luma (chroma stays SSE),
    /// 2 = normalized wSSE on every plane (unification V7).
    pub(crate) lf_search_metric: u32,
    /// CDEF per-unit distortion margin (per mille) a unit must clear to filter.
    /// SHIPPED 40 (2026-10-08, re-fit after the chroma SSE wrap fix: synth 420
    /// -0.95 / 444 -1.37 vs off, 4 photos -0.09; 22 was -0.82 / -1.33 / +0.05).
    pub(crate) cdef_margin: f32,
    /// CDEF global-mode directional-variance trust gate (0 disables).
    pub(crate) cdef_gate: f32,
    /// Weight on the chroma SSE in the joint (luma perceptual + chroma) CDEF metric.
    pub(crate) cdef_chroma_w: f32,
    /// Decision-metric unification Phase 2 (V6): CDEF per-unit distortion is
    /// the normalized wSSE of luma AND chroma (one metric) instead of libaom
    /// `cdef_dist_8x8` luma + plain chroma SSE. SHIPPED true (2026-11-06):
    /// with `cdef_lambda_d` and the 40 per-mille margin kept, tuning32 -0.21 /
    /// -0.18 / -0.22 (420 / 422 / 444), holdout -0.31 / -0.43 / -0.42, no
    /// image worse than +0.06; margin 0 (pure RD) was +0.11 / +0.11 / -0.14
    /// (x_volcanic +1.4) and stays out. +0.75% instr at Slow.
    pub(crate) cdef_wsse: bool,
    /// (V6) CDEF header / `cdef_idx` bits priced with the frame `lambda_d`
    /// (SS2 weight, `mlam_scale`, / `K_PART_LAMBDA`) instead of the bare
    /// `mode_lambda_q`. SHIPPED true (with `cdef_wsse`; alone it is inert to
    /// 2 decimals).
    pub(crate) cdef_lambda_d: bool,
    /// (V7) Wiener search on normalized wSSE + `lambda_d` x signalling bits.
    /// SHIPPED true for consistency; Wiener is off by default (`--wiener`),
    /// so this is not measured on the corpora.
    pub(crate) wiener_wsse: bool,
    /// (V4) `uv_palette_gate` compares the normalized wSSE of the palette
    /// prediction error with the wSSE best total, no unit lift. SHIPPED true
    /// (444 tuning -0.00, holdout +0.01: a unit fix, BD-neutral).
    pub(crate) uv_pal_gate_wsse: bool,
    /// (V5) `trial_struct_penalty` lifted to the trial axis unit (x `K_PART_LAMBDA`).
    pub(crate) trial_struct_wsse: bool,
    /// (V9) 4:4:4 16x16 uv search prices on `emit_mlam_c`, not the luma lambda.
    /// INERT by value: at 4:4:4 the chroma quantiser equals luma, so
    /// `emit_mlam_c == emit_mlam` (32 / 32 images 0.00). Kept off.
    pub(crate) uv16_444_mlam_c: bool,
    /// (V8) TX-type guard tolerance at 4:2:0 / 4:2:2 / 4:0:0: a candidate's
    /// wSSE may exceed the DCT's by `1 / 2^shift` and still win on RD. SHIPPED
    /// 4 (1/16; 2026-11-06 re-sweep after the kernel normalization: tuning
    /// 420 -0.04 / 422 -0.05, holdout -0.15 / -0.15; 1/64 and strict were
    /// neutral or worse). 63 = strict.
    pub(crate) tx_guard_shift: u32,
    /// (V8) The same at 4:4:4: 1/16 measured tuning -0.01 but holdout +0.08
    /// (h_abstract +0.61), so 4:4:4 keeps 1/32.
    pub(crate) tx_guard_shift_444: u32,
    /// Phase 3: the SATD partition price is bridged into the proxy unit
    /// (`K_PART_LAMBDA` x normalized wSSE) through the measured
    /// `partition_rd::k_satd(q, format)`: luma = SATD price x K_PART_LAMBDA /
    /// K_satd x `proxy_luma_scale`, so luma, chroma, IntraBC and trial legs
    /// share one unit and the proxy lambda equals the decision lambda.
    pub(crate) proxy_luma_bridge: bool,
    /// Phase 3: scale on the bridged luma price (1.0 = the unified law).
    /// SHIPPED 7.13 (Optuna `p3bridge`, 2026-11-06, q30 + screen crops): the
    /// unit-exact law (1.0) loses +1.9 / +1.6 on kodak05 + x_screen and every
    /// good trial sits at 6.4-7.3, i.e. the proxy wants the old effective
    /// SATD weight (K_satd / K_PART_LAMBDA) back; what the bridge buys is the
    /// q-flat unit and the joint re-fit of the biases below: tuning32
    /// -0.06 / -0.40 / -0.13, holdout -0.93 / -0.48 / -0.12, new444 -0.63 /
    /// -0.82 / -0.12 (420 / 422 / 444), ExtraSlow -0.22 / -0.30 / -0.07.
    pub(crate) proxy_luma_scale: f32,
    /// Phase 3 (V1): multiplier on the normalized-wSSE chroma leg of the
    /// partition proxies, in proxy units. 2.775 (= K_PART_LAMBDA) is the
    /// Phase 1/2 state (what the x6 landed), 1.0 the pre-wSSE calibration,
    /// 6.0 the raw-kernel x6.
    pub(crate) proxy_chroma_factor: f32,
    /// Phase 3 (V3): `rd_pick_luma64` picks mode / delta / TX on the decision
    /// wSSE like every other emitter instead of on the SATD price. SHIPPED
    /// true (420 tuning -0.02 / holdout -0.00; inert at 4:4:4).
    pub(crate) luma64_wsse: bool,
    /// Chroma weight of the partition proxies (`coder::chroma_part_rd_weight`)
    /// at 4:2:0 / 4:2:2 and at 4:4:4. Were 1/16 and 1/32; re-fit with the
    /// bridge (study `p3bridge`) to 0.082 / 0.054. Keeping the old weights with
    /// the rest of the re-fit trims the worst images but loses the 4:4:4
    /// holdout sign (+0.05) and new444 gain (0.00 vs -0.12).
    pub(crate) chroma_part_w_sub: f32,
    pub(crate) chroma_part_w_444: f32,
    /// `emit_mlam_c` law: the 4:2:0 chroma decision lambda is
    /// `min(lambda_d, lambda_d_c) x this` (0.5 shipped, see `emit_mlam_c`).
    pub(crate) emit_mlam_c_420: f32,
    /// 4:2:0 chroma qindex delta held at qindex >= 136 (see `quant::uv420_mid_delta`).
    /// SHIPPED 28 = the mid-band -28 never decays.
    pub(crate) uv420_tail_delta: f32,
    /// 4:2:0 chroma qindex delta in the mid band (qindex 64..=96), probe knob for
    /// the shipped `quant::UV420_MID_D` = 28 (aom uses -16 at every qindex).
    pub(crate) uv420_mid_d: f32,
    /// Slow-tier luma mode shortlist budget for subsampled formats (shipped 3; 444 = 5).
    pub(crate) mode_budget_slow_sub: u32,
    /// Per-superblock realistic-edge fill for the decision proxies, gated on the
    /// source: a hard edge (8x8 std >= `edge_fill_hi`) inside a mostly flat
    /// superblock (>= `edge_fill_frac` of the 8x8s with std <= `edge_fill_lo`).
    /// Without it the SPLIT children predict from zero-filled recon and the
    /// edge is swallowed by one big DC block.
    pub(crate) edge_fill: bool,
    pub(crate) edge_fill_hi: f32,
    pub(crate) edge_fill_lo: f32,
    pub(crate) edge_fill_frac: f32,
    /// Realism of the recon estimate the proxies see (`fill_recon_estimate_region`):
    /// `recon_fill_k` scales the quantized 4x4 residual added to the 4x4 mean
    /// (1.0 = shipped estimate, 0.0 = 4x4 means only); `recon_fill_dim` 8 fills
    /// flat 8x8 means instead (k ignored). Lower realism = less 4x4 over-split.
    pub(crate) recon_fill_k: f32,
    pub(crate) recon_fill_dim: u32,
    /// 8x8 leaf angle deltas: `ad8_full` prices all six deltas (not the SATD
    /// top-2); `ad8_runner` also refines the runner-up directional mode.
    /// SHIPPED true/true with `split4_ad` (2026-10-16): illustrations 420
    /// tori -4.4 / top -3.3 / jake -1.7, 444 -1.9, 422 -3.6; tuning -0.28,
    /// synth -0.56, holdout -0.08; ~+10-15% Slow time.
    pub(crate) ad8_full: bool,
    pub(crate) ad8_runner: bool,
    /// Add two globally ranked (mode, nonzero angle) luma finalists at 8/16/32.
    /// ExtraSlow always enables this; other presets keep their current budget.
    pub(crate) joint_angle_search: bool,
    /// 8x8-vs-split4 decision prices the 8x8 side with angle deltas + real
    /// tr/bl. Without it edge 8x8s went to four 4x4s, which cannot signal
    /// angle_delta (spec: bsize >= 8x8) and predict outlines 3-9x worse.
    pub(crate) split4_ad: bool,
    /// Real-R-D trial-encode partition decision (`coder/trial.rs`): bitmask
    /// of square nodes whose NONE-vs-SPLIT choice codes BOTH legs for real
    /// (children with their own decisions) and keeps the cheaper one.
    /// 8 = the 64x64 node, 4 = the 32x32 node, 2 = the 16x16 node.
    /// 0 = the proxy decision everywhere.
    pub(crate) trial_part: u32,
    /// Same bitmask, additionally applied at ExtraSlow. SHIPPED 6.
    pub(crate) trial_part_xslow: u32,
    /// Trial comparison axis: false = both legs priced on the node's own
    /// perceptual scale; true = each 16x16 quadrant's distortion normalized by
    /// that quadrant's own lambda (what the bottom-up proxy sum does).
    pub(crate) trial_child_prdo: bool,
    /// Multiplier on the SPLIT leg's trial cost (1.0 = pure R-D).
    pub(crate) trial_split_bias: f32,
    /// Which proxy verdicts the trial may overturn: 0 = always trial (the
    /// proxy is skipped where it is not needed), 1 = trial only where the
    /// proxy chose NONE (the swallowed-edge defect; a proxy SPLIT stands),
    /// 2 = trial only where the proxy chose SPLIT, 3 = as 1, plus at a 16x16
    /// node where the proxy chose SPLIT the rect legs (`trial_rect`) are
    /// trialled against SPLIT (NONE stands rejected).
    pub(crate) trial_gate: u32,
    /// Source gate on the trial (0 = none): run it only where some 8x8 of the
    /// node has std >= `trial_edge_hi` (there is structure a big block could
    /// swallow) and, when `trial_edge_lo` > 0, some 8x8 has std <= it (the
    /// structure sits in otherwise flat content: an outline, not texture).
    pub(crate) trial_edge_hi: f32,
    pub(crate) trial_edge_lo: f32,
    /// 16x16 trial also codes PARTITION_HORZ / PARTITION_VERT for real
    /// (where the chroma format has the emitter: VERT is spec-forbidden at
    /// 4:2:2) and keeps the cheapest of the four legs.
    pub(crate) trial_rect: bool,
    /// Multiplier on the HORZ/VERT legs' trial cost (1.0 = pure R-D). SS2
    /// penalises marginal rect wins that trade luma SSE for a few bits.
    pub(crate) trial_rect_bias: f32,
    /// A HORZ/VERT leg may win only if its luma SSE is not above a square
    /// leg's: 0 = off, 1 = the best-cost square leg's, 2 = the lower of the
    /// two square legs' (rect wins that trade luma SSE for bits lose SS2).
    pub(crate) trial_rect_sse_gate: u32,
    /// Weight on the HORZ/VERT legs' luma distortion in the trial cost.
    pub(crate) trial_rect_dy_w: f32,
    /// Structural penalty on every trial leg: k * sum over 8x8 cells of
    /// (std_src - std_recon)^2 * 64, in luma-SSE units (0 = off). Charges
    /// lost or invented fine detail that plain SSE prices too cheaply.
    pub(crate) trial_struct_k: f32,
    /// ExtraSlow package (applied on top of the three knobs above at that
    /// tier only, like `trial_part_xslow`): rect legs with luma-SSE gate 1 and
    /// structural penalty 1.0. Measured at real ExtraSlow vs the gated trial
    /// alone: 420 15 images -0.25 (11/15), 444 -0.40, 422 -0.29, +5..10% time.
    pub(crate) trial_rect_xslow: bool,
    pub(crate) trial_rect_sse_gate_xslow: u32,
    pub(crate) trial_struct_k_xslow: f32,
    /// Proxy-margin gate on the trial (0 = off): a square node is trialed
    /// only where the proxy's relative NONE-vs-SPLIT margin
    /// |none - split| / min(none, split) is below this (a pruned leg counts
    /// as an infinite margin: the proxy verdict stands). Bounds the real-R-D
    /// work to the nodes the proxy is unsure about, the way jixel's layout DP
    /// scores a piece in full only inside its bound slack.
    pub(crate) trial_margin: f32,
    /// What the margin gate does at a node whose margin is UNKNOWN (the proxy
    /// never priced the other leg: 32-level flat-block SPLIT skip, 16-level
    /// moment breakout): 0 = trial it, 1 = the proxy verdict stands, 2 = trial
    /// it only where some 8x8 has source std >= `trial_margin_edge`. A leg
    /// pruned by the exact branch-and-bound keeps its bound as a conservative
    /// (lower-bound) margin and is not "unknown".
    pub(crate) trial_margin_pruned: u32,
    pub(crate) trial_margin_edge: f32,
    /// Partition proxies (`rd_cost_square` 8/16) price the winner's six angle
    /// deltas. Measured inert on top of the three above (-1.4 alone, +8% time).
    pub(crate) proxy_ad: bool,
    /// 16x16 leaf (winner + `ad_runner`) prices all six angle deltas.
    pub(crate) ad16_full: bool,
    /// 4:2:0 8x8 leaf: directional chroma candidates compete against CfL
    pub(crate) uv8_vs_cfl: bool,
    /// 4:2:0 8x8 leaf: joint (uv mode, angle delta) SATD shortlist over the
    /// full directional set incl. zone-1/3 (D45/D67/D203), predicted with the
    /// leaf's real top-right / bottom-left availability (decoder-exact), in
    /// place of the delta-0 top-3 + winner-only delta refinement (`uv8_ad`).
    pub(crate) uv8_joint: bool,
    /// Shortlist size of `uv8_joint` (full RD per kept candidate).
    pub(crate) uv8_joint_k: u32,
    /// 4:2:0 16x16 leaf: the same joint (uv mode, angle delta) shortlist.
    pub(crate) uv16_joint: bool,
    /// Masking (perceptual RD) strength K for subsampled / 4:4:4 formats
    /// (`perceptual_rd_scale`); 0 = masking off.
    pub(crate) prdo_k: f32,
    pub(crate) prdo_k_444: f32,
    /// Contextual trellis Step A dead-zone: an interior level-1 coefficient
    /// whose quantization target is below this many steps is zeroed before the
    /// local R-D choice (0 = off). Synthetic-gap equilibrium study.
    pub(crate) trellis_dz: f32,
    /// Rate-aware CfL alpha refinement at the square leaves (8/16/32, every
    /// chroma format): from the energy-optimal alpha, step at most this many
    /// alphas toward zero while the realized per-plane RD cost (recon SSE +
    /// lambda x coef bits + the plane's alpha-signal bits) keeps falling, else
    /// the same away from zero (aom `cfl_pick_plane_rd`). 0 = energy pick only.
    pub(crate) cfl_alpha_rd_steps: u32,
    /// What the alpha walk scores: 0 = recon SSE + lambda x (coef + alpha
    /// bits), 1 = SSE + lambda x coef bits only, 2 = realized SSE only.
    pub(crate) cfl_alpha_rd_mode: u32,
    /// Filter-intra finalists given full RD at the square 8/16/32 leaves
    pub(crate) filter_intra_budget: u32,
    /// Same at 4:4:4, where the budget is BD-neutral (new444 14 images
    /// -0.00) and only costs time: stays 0.
    pub(crate) filter_intra_budget_444: u32,
    /// Admit a filter-intra candidate on RD cost alone (libaom), instead of
    /// requiring it to also beat the best plain candidate's SSE.
    pub(crate) filter_intra_rd_only: bool,
    /// Filter-intra candidates on the 16x8 / 8x16 leaves.
    pub(crate) filter_intra_rect: bool,
    /// 4:2:0 8x8 leaf: CfL-vs-DC decided on the chroma lambda (emit_mlam_c).
    pub(crate) uv8_cfl_mlam_c: bool,
    /// 4:2:0 8x8 leaf: zone-2 chroma angle deltas for the directional winner
    pub(crate) uv8_ad: bool,
    /// 32x32 luma leaf: price `tx_depth = 2` (sixteen TX_8X8, per-TX txtp)
    /// against the whole / depth-1 transform by RD. Illustrations: a 32x32 DC
    /// block with one outline edge in a corner coded its residual with one
    /// TX_32X32 (~45 B per non-skipped 32x32 on jake_cat vs aom's 0.9 B).
    pub(crate) tx32_depth2: bool,
    /// 32x32 luma: try the depth-1 split (four TX_16X16) on plain RD on every
    /// block, not only under `banding_risk` (synthetic-gap study P2).
    pub(crate) tx32_split_rd: bool,
    pub(crate) uv422_mid_scale: f32,
    /// Source-domain SPLIT breakout ratio at Slow. 1.5 -> 2.77 in the Phase 3
    /// re-fit (fewer breakouts; most of the Phase 3 +2% instr, x_volcanic 444
    /// +7.6%).
    pub(crate) split_breakout_slow: f32,
    pub(crate) fixed_size_fast: u32,
    pub(crate) fixed_size_medium: u32,
    pub(crate) fixed_size_slow: u32,

    /// Partition-signaling entry fee, in bits, charged to every non-NONE candidate.
    /// 24 -> 33.9 in the Phase 3 re-fit (importance 0.007: a weak knob).
    pub(crate) part_signal_bits: f32,
    /// How much of it the 4:2:0/4:2:2 seam ramp removes at full strength.
    pub(crate) part_signal_seam_drop: f32,
    /// Multiplier on the entry fee for the bottom-up 32-level SPLIT leg.
    pub(crate) split32_signal_mult: f32,
    /// Whole-64 must be within this factor of the SPLIT bound to be refined.
    pub(crate) b64_refinement_window: f32,
    /// Fraction of the 64-level SPLIT bound refined before deciding.
    pub(crate) b64_split_refinement: f32,
    /// Source variance / ac_q^2 below which the 32-level SPLIT search is skipped (4:2:0).
    pub(crate) vbp_thresh_420: f32,
    /// Seam-band width for the 4:2:0 partition-signal ramp (0 disables the ramp).
    pub(crate) seam_w_420: f32,
    /// Same for 4:2:2.
    pub(crate) seam_w_422: f32,
    /// Joint luma/UV large-block gain factor.
    pub(crate) joint_large_gain: f32,
    /// luma_satd_scale base: SATD-to-SSE bridge for partition pricing.
    pub(crate) satd_base: f32,
    /// luma_satd_scale at the top quality band (4:4:4 only). 2.5 -> 2.92 in
    /// the Phase 3 re-fit (the bridge divides the price by K_satd, which
    /// carries the old ramp, so this ramp is re-fit on top of a flat unit).
    pub(crate) satd_top: f32,
    /// qindex at/below which satd_top applies.
    pub(crate) satd_knee_lo: f32,
    /// qindex at/above which satd_base applies.
    pub(crate) satd_knee_hi: f32,
    /// Adaptive-quant strength.
    pub(crate) aq_slope: f32,
    /// Adaptive-quant qindex clamp.
    pub(crate) aq_max_delta: f32,
    /// Adaptive-quant boost/cut ratio clamp.
    pub(crate) aq_r: f32,
    /// Trellis frequency-tilt magnitude cap.
    pub(crate) trellis_tilt_mag_cap: f32,
    /// Cost of signaling a non-DC uv_mode for the 4:2:0 4x4 SMOOTH_V trial.
    pub(crate) smooth_v_uv_signal_bits: f32,
    /// Masking reference blend toward the superblock activity.
    pub(crate) local_ref_blend: f32,
    /// Price 16x16 partition symbols from the frozen CDFs instead of the flat
    /// 24-bit non-NONE fee. See `LossyTile::partition16_bits`.
    pub(crate) exact_part_bits: bool,
    /// Charge the per-leaf uv_mode symbol in the chroma partition proxy.
    pub(crate) chroma_uv_mode_bits: bool,
    /// Stage 2 of the 16-level chroma partition proxy: how many still-in-
    /// contention candidates get the CfL trial (0 = DC-only everywhere).
    /// 4:4:4 only -- 4:2:2 measured NEGATIVE (see `chroma_refine_topk`).
    pub(crate) chroma_refine_topk: usize,
    /// Ranked-path palette finalists the proxy fully prices
    pub(crate) palette_proxy_finalists: usize,
    pub(crate) palette_proxy_ranked: bool,
    /// Evaluate rectangles at decision time with the SAME tools the emitter uses.
    pub(crate) rect_decision_refine: bool,
    /// Exact per-symbol partition pricing at the 32x32 and 64x64 nodes.
    pub(crate) exact_part_bits_3264: bool,
    /// As above but 4:4:4 only. SHIPPED true. Slow, 210 cases:
    pub(crate) exact_part_bits_3264_444: bool,
    /// Charge the 8x8 `part_bl8` symbol in the NONE-vs-SPLIT4 comparison.
    pub(crate) exact_part_bits_8: bool,
    /// Enable the 8x4/4x8 partition search (4:2:0, Slow). Was a hardcoded false.
    pub(crate) rect8_enabled: bool,
    /// Rectangle-vs-square balance at the 8x8 node (analogue of `rect16_bias`).
    pub(crate) rect8_bias: f32,
    /// Frame-edge 16x16 nodes (exactly one dimension runs past the coded
    /// frame): decide PARTITION_HORZ / VERT (one fully in-frame 16x8 / 8x16
    /// leaf; the other half is never coded) against the SPLIT into the two
    /// in-frame 8x8s by real R-D, instead of always coding SPLIT there.
    pub(crate) edge_rect: bool,
    /// Rect-vs-split balance at those edge nodes (1.0 = unbiased R-D).
    pub(crate) edge_rect_bias: f32,
    /// Guided partitioning at the 16 node: commit to SPLIT without pricing NONE
    /// when `m_split * k < m_none` in the source model. 0 = off (full R-D).
    pub(crate) vbp_medium_420: bool,
    pub(crate) vbp_444: bool,
    pub(crate) vbp_422: bool,
    /// Threshold for the non-4:2:0 formats (4:2:0 uses `vbp_thresh_420`).
    pub(crate) vbp_thresh_hi: f32,
    // --- Medium leaf-work gates (mode budgets / transform + chroma trials) ---
    pub(crate) proxy_rate_fast: bool,
    pub(crate) proxy_rate_medium: bool,
    pub(crate) proxy_rate_slow: bool,
    /// Multiplier on the exact-rate abort bound; < 1.0 makes it lossy.
    pub(crate) rate_bound_slack_fast: f32,
    pub(crate) rate_bound_slack_medium: f32,
    pub(crate) rate_bound_slack_slow: f32,
    pub(crate) mode_budget_medium: u32,
    /// Whether Medium runs the full partition R-D (chroma partition costs and
    /// the non-square admission). The ledger suggests dropping this instead of
    /// cutting the mode budget.
    pub(crate) full_part_rdo_medium: bool,
    pub(crate) palette_medium: bool,
    pub(crate) palette_budget_medium: u32,
    pub(crate) angle_deltas_medium: bool,
    /// Gradient-orientation gate on the global angular finalists: the block's
    /// dominant source edge direction must carry at least this share of the
    /// gradient energy (0 = no gate), and finalists must lie within
    /// `angle_hog_win` degrees of it. SHIPPED 0.3 / 15: same BD as ungated
    /// (gate 0.3 / 9 deg lost a third of the gain) at +4-6% instructions
    /// instead of +17-19%.
    pub(crate) angle_hog_gate: f32,
    pub(crate) angle_hog_win: f32,
    /// Number of global (mode, nonzero delta) finalists (study; shipped 2).
    pub(crate) joint_angle_count: u32,
    /// Oracle: admit every nonzero (mode, delta) pair to full RD.
    pub(crate) joint_angle_all: bool,
    pub(crate) full_chroma_rdo_medium: bool,
    pub(crate) min_size_16_fast: bool,
    pub(crate) min_size_16_medium: bool,
    /// PARTITION_HORZ/VERT at the 64x64 superblock (BLOCK_64X32 / BLOCK_32X64).
    pub(crate) rect64: bool,
    /// R-D multiplier on the rect64 legs (1.0 = priced like whole-64 NONE).
    pub(crate) rect64_bias: f32,
    /// Also open the rect64 legs on `screen_frame`s (Slow+), where they carry
    /// the 64x32 / 32x64 palette leaves (`palette64`); photos unaffected.
    /// OFF: measured BD-inert next to the 64x64 palette leaf (n_bbscreen
    /// +-5 B) for ~8% screen-frame time.
    pub(crate) rect64_screen: bool,
    /// TX_64X64 as a whole-64 luma alternative to four TX_32X32.
    pub(crate) tx64: bool,
    /// R-D multiplier on the TX_64X64 candidate (1.0 = priced like 4xTX_32X32).
    pub(crate) tx64_bias: f32,
    /// Also offer the DC TX_64X64 leg to the whole-64 NONE-vs-SPLIT estimator.
    pub(crate) tx64_none: bool,
    /// Whole-64 luma also tries tx_depth 2 (sixteen TX_16X16) on the winning
    /// mode, against the four TX_32X32 (and TX_64X64) legs.
    pub(crate) tx64_depth2: bool,
    /// Square-leaf angle-delta refinement also on V_PRED / H_PRED
    pub(crate) ad_vh: bool,
    /// Rect16 leaves also search V_PRED / H_PRED angle deltas
    pub(crate) rect_ad: bool,
    /// 16x16 leaves also refine the runner-up directional mode's deltas
    pub(crate) ad_runner: bool,
    /// Zone-2 chroma angle-delta refinement of a directional uv winner at the
    pub(crate) uv_ad: bool,
    /// Rect16 leaves also search the zone-2 diagonals D113/D135/D157 x 7 deltas.
    pub(crate) rect_diag: bool,
    /// `uv_ad` also at the 32x32 leaf (420 16x16 chroma).
    pub(crate) uv_ad32: bool,
    /// 4:2:2 16x16 leaf: full directional/smooth chroma mode search + zone-2
    /// deltas (was DC/CfL only).
    pub(crate) uv422_modes: bool,
    /// Per-format multiplier on the mode lambda (decisions: modes, partitions,
    /// palettes; not the trellis)
    pub(crate) mlam_scale_420: f32,
    pub(crate) mlam_scale_422: f32,
    pub(crate) mlam_scale_444: f32,
    pub(crate) min_size_16_slow: bool,
    pub(crate) guided16_k_fast: f32,
    pub(crate) guided16_k_medium: f32,
    pub(crate) guided16_k_slow: f32,
    /// Source-domain SPLIT4 breakout ratio at Slow (was hardcoded 1.5).
    pub(crate) split4_breakout_slow: f32,
    /// Price the emitted 4x4 luma mode symbols in the 8x8 NONE-vs-SPLIT4
    /// decision and in the SPLIT4 leaf search (4:2:0/4:4:4; 4:2:2 is gated at
    /// the call sites because it measured negative).
    pub(crate) exact_8x8_mode_rate: bool,
    pub(crate) split4_decision_txtypes: bool,
    /// Lossy luma palette index-map smoothing: a pixel takes its LEFT/ABOVE
    /// neighbor's palette index when that center is within
    /// `ac_q * palette_smooth / 256` (pixel units) of the nearest one. The
    /// residual absorbs the extra error; the smoother map codes far cheaper
    /// under the neighbor-context index coder. 0 = off (nearest only).
    pub(crate) palette_smooth: u32,
    /// With `palette_smooth`: only pixels the nearest center does NOT hit
    /// exactly may move (protects exact palettes / exact-hit pixels).
    pub(crate) palette_smooth_guard: bool,
    /// Row-wise Viterbi over the lossy luma palette index map (replaces the
    /// `palette_smooth` pass when > 0): with the rows above final, each row is
    /// an exact chain (state = the left pixel's index) minimizing squared
    /// center error + lambda * the real `palette_y_color` context bits.
    /// The value multiplies the tile's mode lambda; 0 = off. Per format
    /// (`_444` = 4:4:4, `_420` = 4:2:0 / 4:2:2 / mono). SHIPPED 1.0 / 0.5
    /// behind the `screen_frame` gate (2026-10-24): x_screen -1.94 / -0.45,
    /// n_bbscreen -1.10 / -3.40, every non-screen frame byte-identical;
    /// lam >= 2.0 over-merges and loses on every image.
    /// 1.0 -> 0.35 (unification Phase 4, 2026-11-06, 2-point check + 0.25):
    /// the only images that exercise it (444 screen frames) x_screen -2.47,
    /// n_bbscreen -1.26; 0.5 -1.67 / 0.25 -1.81 on the pair; holdout flat.
    pub(crate) palette_viterbi_444: f32,
    pub(crate) palette_viterbi_420: f32,
    /// Also build the partition proxy's palette estimates (`rd_cost_square`
    /// bracket) with the Viterbi; false keeps nearest + smoothing there and
    /// runs the Viterbi only for the emitter's candidates.
    pub(crate) palette_viterbi_proxy: bool,
    /// Where the Viterbi runs: 0 = every frame, 1 = `screen_frame` only
    /// (screen content WITH exact 16x16 repeats), 2 = `screen_content` only.
    /// Ungated, photos measured BD-neutral (+-0.8 per image, ~0 avg) for ~10%
    /// time and synthetic edges lose (x_fractal +1.0..1.3).
    pub(crate) palette_viterbi_screen_only: u32,
    /// Lossy UV palette index map by the same row Viterbi (4:4:4, the only
    /// format with UV palettes): cost = (dU^2 + dV^2) + lambda * the real
    /// `palette_uv_color` context bits; the value multiplies
    /// `min(mlam, mlam_c)`. Same `palette_viterbi_screen_only` gate. 0 = off
    /// (nearest-pair map).
    pub(crate) palette_viterbi_uv: f32,
    /// Luma palette centers by the exact 1-D DP (globally SSE-optimal integer
    /// centers, `dp_palette_center_sets`) instead of quantile-seeded Lloyd,
    /// which can stop in a local minimum or reject on seed collapse. Same
    /// selector slots, so replay and the dominant-color family are unchanged.
    /// SHIPPED ON 2026-10-25, ungated: on text the background holds > 75% of
    /// a block, the quantile seeds collapse onto it and Lloyd rejected, so
    /// x_screen -9.0 (444) / -11.5 (420), n_bbscreen -4.1 / -4.9, n_scmap 420
    /// -2.4; the 48 non-screen corpus images are BD-neutral (-0.01 / +0.03).
    pub(crate) palette_dp_centers: bool,
    /// Luma palette candidates at BLOCK_64X64 and the rect64 halves
    /// (64x32 / 32x64), with the matching partition-proxy bound. Screen
    /// frames only (`screen_frame`), Slow+.
    pub(crate) palette64: bool,
    /// `screen_frame_gate`: IntraBC pays when the copied texture mass exceeds
    /// this fraction of the photo-like mass that loses the loop filters.
    pub(crate) ibc_gate_beta: f32,
    /// `screen_frame_gate`: per-copy DV overhead in SAD units, as a multiple of
    /// the frame's AC dequant step.
    pub(crate) ibc_gate_dv: f32,
    /// `screen_frame_gate`: build the copy index only when at least this share
    /// of 16x16 blocks has <= 16 luma values (photos never do).
    pub(crate) ibc_gate_prescreen: f32,
    /// Above-row reach of the capture wavefront on IntraBC frames (cell (r, c)
    /// waits for (r-1, c+reach)), and the matching DV legality in
    /// `find_intrabc`: finished set `{r' < r: c' <= c + reach*(r-r')}`.
    /// 1 = the classic d = 2r + c schedule, which forbade most spec-legal
    /// copy sources (s_manga: 1% of exact 8x8 sources admitted vs 56% legal);
    /// 5 = libaom's `av1_is_dv_valid` gradient, every spec-legal source.
    /// 2 keeps 98% of the manga gain (39218 vs 39148 B) at 1.15x the -t8
    /// time of reach 1 instead of 1.85x; it covers the row above completely
    /// and two rows above to c + 4 (spec: c + 5).
    pub(crate) ibc_wavefront_reach: u32,
    /// Odd luma DVs for IntraBC at 4:2:0 / 4:2:2: the subsampled planes are
    /// predicted at the half-sample position with AV1's BILINEAR filter
    /// (`ibc_chroma_pred`), as the decoders do, instead of requiring a
    /// chroma-parity-even DV (integer copy). Exact-copy semantics are kept:
    /// the filtered chroma must reproduce the source. Off = the old parity
    /// restriction (and the parity-grid hash index).
    pub(crate) ibc_odd_dv: bool,
    /// IntraBC with a coded residual at 16x16 / 32x32 (screen frames, 4:2:0 and
    /// 4:4:4): an approximate copy (any legal DV from the 8x8 anchor hits, the
    /// DV predictor and the neighbours' DVs) plus the prediction error coded as
    /// an AV1 INTER block (skip = 0, one luma TX of block size, DCT_DCT in the
    /// inter tx-type set, decoder-derived chroma type). OFF: measured BD
    /// x_screen -0.2 (420) / -0.6 (444) but n_bbscreen +0.3 / +0.4, manga and
    /// line-art ladders unchanged (the SATD proxy rightly keeps b/w flips
    /// out), for 1.03-1.28x screen-frame time.
    pub(crate) ibc_residual: bool,
    /// Approximate-copy candidates that get the full residual RD (ranked by
    /// source luma SAD).
    pub(crate) ibc_residual_cands: u32,
    /// `screen_frame_gate`: palette-exact 16x16 share that turns the text
    /// palette tools on (libaom's screen-content test uses 10%).
    pub(crate) screen_few_share: f32,
    /// 4:4:4-only top-band NONE-vs-SPLIT bias at the 32/64 levels: the bias
    /// ramps from `none32/64_split_bias` at qindex >= knee to these values at
    /// qindex <= knee - width.
    pub(crate) none32_split_bias_444_top: f32,
    pub(crate) none64_split_bias_444_top: f32,
    pub(crate) top_bias_knee_444: f32,
    pub(crate) top_bias_width_444: f32,
    /// UV palette candidate pre-filter: skip the residual transform/trellis/
    /// rate work when the ZERO-residual palette cost (raw prediction SSE +
    /// header/map bits) already exceeds `k` x the incumbent chroma cost.
    /// 0 = off.
    pub(crate) uv_pal_gate_k: f32,
    /// Contextual trellis: price the DC-only (eob == 0) terminal with the
    /// emitter's contexts (`eob_base[0]`, `br_tok[0]`) instead of
    /// `eob_base[1]` + an AC-derived br context.
    pub(crate) trellis_dc_ctx0: bool,
    /// Bits charged to the all-zero (txb_skip = 1) alternative in both
    /// trellises, RELATIVE to the nonzero candidates, which omit the common
    /// skip=0 + txtp overhead. The historical constant is 1.0; the exact
    /// relative value is cost(skip=1) - cost(skip=0) - cost(txtp), usually
    /// negative.
    pub(crate) trellis_zero_bits: f32,
    /// Contextual trellis: also scan lower DC levels for the DC-only
    /// terminal (the Step-A DC level was optimized assuming the AC stays).
    pub(crate) trellis_dc_only_scan: bool,
    /// Contextual trellis: run the libaom `av1_optimize_txb` search
    /// structure (terminal-first EOB walk with max 2 kept nonzeros,
    /// level-1-only interior lowering) on our rate model and lambda.
    pub(crate) trellis_aom: bool,
    /// Contextual trellis: price every EOB candidate in ITS OWN contexts.
    /// The prefix costs were built once from the post-Step-A level map and
    /// reused for every shorter EOB, although zeroing the suffix lowers the
    /// base/br contexts of up to five earlier coefficients per removed
    /// position. On, Step B walks the EOB downward, zeroes one position per
    /// step and re-prices only its (<= 5) earlier stencil dependants.
    pub(crate) trellis_eob_reprice: bool,
    /// With `trellis_eob_reprice`: also offer positions Step A zeroed as the
    /// EOB coefficient (terminals are priced off `eob_base`, not the
    /// neighbour context that made the interior level-0 win).
    pub(crate) trellis_eob_zero_term: bool,
    /// Contextual trellis: pad the level-map stride to `h + 2`. The emitter
    /// fills its map progressively in reverse scan, so its stencil reads past
    /// the last row (`y + 1`, `y + 2`) land on not-yet-written zeros, as in
    /// the decoder. The trellis fills the whole map up front at stride `h`,
    /// so for the last two rows those reads WRAP onto earlier coefficients'
    /// real levels and inflate their contexts. The buffer was already sized
    /// `(w + 2) * (h + 2)`.
    pub(crate) trellis_pad_stride: bool,
    /// Luma mode beam: give the mode with the smallest CENTRED residual energy
    /// (cheapest DC-only correction) the last shortlist slot when the SATD
    /// ranking pruned it
    pub(crate) beam_sparse_slot: bool,
    /// Multiplier on the trellis lambda calibration (TRELLIS_AOM_CALIB)
    pub(crate) trellis_calib: f32,
    /// High-bit-depth (bd > 8) top-band trellis law: the trellis lambda is
    /// scaled by a factor that ramps linearly from 1 at frame qindex
    /// `hbd_trellis_hi` down to `hbd_trellis_top` at `hbd_trellis_lo` and
    /// below. 8-bit encodes are untouched. These three are 4:2:0; 4:2:2 has
    /// its own triplet; 4:4:4 and 4:0:0 keep the full lambda.
    ///
    /// SHIPPED 2026-11-11 at 4:2:0: 0.174 (= trellis_calib 0.2 / 1.15) over
    /// qindex 36 -> 20. At the top the shipped trellis discards ~1.7x aom's
    /// distortion per step^2 at 10 bits and the encoder buys it back with a
    /// finer qindex. Fit (q65-96, SS2 BD): 10-bit tuning -0.31, hi-bd SDR
    /// -0.26, hi-bd PQ -0.55; q65-80 +0.00..+0.03. Held out: holdout@10
    /// -0.81, HLG -0.55, PQ (jyrki31) -0.98, downscaled HDR -0.70/-0.69;
    /// Fast -0.60, Medium -0.20, ExtraSlow -0.47. 4:2:2 measured neutral
    /// (-0.11..+0.22 for 0.174/0.3 ramps at 24->10 / 20->8) and 4:4:4
    /// neutral (+0.1), so both stay at 1.0.
    pub(crate) hbd_trellis_top: f32,
    pub(crate) hbd_trellis_hi: f32,
    pub(crate) hbd_trellis_lo: f32,
    pub(crate) hbd_trellis_top_422: f32,
    pub(crate) hbd_trellis_hi_422: f32,
    pub(crate) hbd_trellis_lo_422: f32,
    /// Band-tilt `ramped_tilt(base, q, extra)` per format (444 / 420)
    pub(crate) tilt_base_444: f32,
    pub(crate) tilt_extra_444: f32,
    pub(crate) tilt_base_420: f32,
    pub(crate) tilt_extra_420: f32,
    /// Decision distortion = 4x4-DCT perceptually weighted SSE
    /// (`rd_sse::weighted_sse_scalar`, the hpvca weights) instead of plain
    /// SSE, per chroma format
    pub(crate) wsse_420: bool,
    pub(crate) wsse_422: bool,
    pub(crate) wsse_444: bool,
    pub(crate) wsse_400: bool,
    /// Multiplier on the (mean-1 normalized) weighted distortion. 1.0 = the
    /// normalized unit; the historical 6.0 = 2.162 (unit) x 2.775
    /// (`cost::K_PART_LAMBDA`, now an explicit lambda ratio).
    pub(crate) wsse_scale: f32,
}

impl Tuning {
    /// The shipped constants. Changing a value here changes the encoder.
    pub(crate) const SHIPPED: Tuning = Tuning {
        rect16_bias: 1.0708447,
        rect16_bias_422: 1.0,
        rect16_horz_444: true,
        rect16_vert_444: true,
        rect16_420_medium: false,
        none16_top_bias_420: 1.17077,
        none16_top_bias_422: 1.25,
        none16_top_bias_444: 1.103276,
        quad4_bias: 1.1,
        part_budget_slow: 3,
        part_budget_qmin: 0,
        part_budget_444_only: true,
        rect_tx_split: true,
        rect_uv_search: false,
        rect_uv_joint: true,
        rect_uv_joint_k: 4,
        rect_full_modes: true,
        strip_modes: true,
        strip_uv_joint: true,
        rect_split_seq: true,
        recon_src_init: false,
        split16_bias: 1.0,
        split4_bias: 1.0,
        part_budget_medium: 0,
        part_budget_fast: 0,
        top_ease: 1.0,
        top_ease_knee: 20.0,
        top_ease_width: 12.0,
        qm_hq_knee: 12.0,
        qm_hq_width: 5.0,
        top_band_q: 55,
        none32_split_bias: 1.05446,
        none64_split_bias: 1.248794,
        top_none_bias_420_hi: 1.45,
        top_none_bias_420_lo: 1.15,
        top_444_uac: 6.0,
        mode_lambda_lo_floor: 0.0,
        mode_lambda_lo_full: 72.0,
        mode_lambda_lo_w: 3.0,
        lf_sharpness: 7,
        lf_level_scale: 1.0,
        lf_uv_level_scale: 1.0,
        lf_uv_level_scale_420: 0.5,
        lf_search: false,
        lf_search_bias: 1.0,
        lf_search_start0: false,
        lf_search_metric: 0,
        cdef_margin: 40.0,
        cdef_gate: 15000.0,
        cdef_chroma_w: 1.0,
        cdef_wsse: true,
        cdef_lambda_d: true,
        wiener_wsse: true,
        uv_pal_gate_wsse: true,
        trial_struct_wsse: false,
        uv16_444_mlam_c: false,
        tx_guard_shift: 4,
        tx_guard_shift_444: 5,
        proxy_luma_bridge: true,
        proxy_luma_scale: 7.130643,
        proxy_chroma_factor: 2.775,
        luma64_wsse: true,
        chroma_part_w_sub: 0.08221585,
        emit_mlam_c_420: 0.5,
        chroma_part_w_444: 0.05426356,
        uv420_tail_delta: 28.0,
        uv420_mid_d: 28.0,
        mode_budget_slow_sub: 3,
        edge_fill: false,
        edge_fill_hi: 12.0,
        edge_fill_lo: 4.0,
        edge_fill_frac: 0.5,
        recon_fill_k: 1.0,
        recon_fill_dim: 4,
        ad8_full: true,
        ad8_runner: true,
        joint_angle_search: false,
        split4_ad: true,
        trial_part: 0,
        trial_part_xslow: 6,
        trial_child_prdo: false,
        trial_split_bias: 1.0,
        trial_gate: 1,
        trial_edge_hi: 16.0,
        trial_edge_lo: 0.0,
        trial_rect: false,
        trial_rect_bias: 1.0,
        trial_rect_sse_gate: 0,
        trial_rect_dy_w: 1.0,
        trial_struct_k: 0.0,
        trial_rect_xslow: true,
        trial_rect_sse_gate_xslow: 1,
        trial_struct_k_xslow: 1.0,
        trial_margin: 0.0,
        trial_margin_pruned: 0,
        trial_margin_edge: 16.0,
        proxy_ad: false,
        ad16_full: false,
        uv8_vs_cfl: true,
        uv8_joint: true,
        uv8_joint_k: 4,
        uv16_joint: true,
        prdo_k: 0.4,
        prdo_k_444: 0.5,
        trellis_dz: 0.0,
        cfl_alpha_rd_steps: 0,
        cfl_alpha_rd_mode: 0,
        filter_intra_budget: 3,
        filter_intra_budget_444: 0,
        filter_intra_rd_only: false,
        filter_intra_rect: true,
        uv8_cfl_mlam_c: true,
        uv8_ad: true,
        tx32_depth2: true,
        tx32_split_rd: false,
        uv422_mid_scale: 0.5,
        split_breakout_slow: 2.771115,
        fixed_size_fast: 16,
        fixed_size_medium: 0,
        fixed_size_slow: 0,
        part_signal_bits: 33.8797,
        part_signal_seam_drop: 22.0,
        split32_signal_mult: 1.0,
        b64_refinement_window: 0.95,
        b64_split_refinement: 0.5,
        vbp_thresh_420: 0.002,
        seam_w_420: 0.0,
        seam_w_422: 60.0,
        joint_large_gain: 0.995,
        satd_base: 1.25,
        satd_top: 2.921556,
        satd_knee_lo: 20.0,
        satd_knee_hi: 48.0,
        aq_slope: 5.0,
        aq_max_delta: 28.0,
        aq_r: 1.35,
        trellis_tilt_mag_cap: 2.0,
        smooth_v_uv_signal_bits: 4.0,
        local_ref_blend: 0.5,
        exact_part_bits: true,
        chroma_uv_mode_bits: false,
        chroma_refine_topk: 2,
        palette_proxy_ranked: true,
        palette_proxy_finalists: 2,
        rect_decision_refine: false,
        exact_part_bits_3264: false,
        exact_part_bits_3264_444: true,
        exact_part_bits_8: false,
        rect8_enabled: false,
        rect8_bias: 1.0,
        edge_rect: true,
        edge_rect_bias: 1.0,
        vbp_medium_420: false,
        vbp_444: false,
        vbp_422: false,
        vbp_thresh_hi: 0.0005,
        proxy_rate_fast: false,
        proxy_rate_medium: false,
        proxy_rate_slow: false,
        rate_bound_slack_fast: 1.0,
        rate_bound_slack_medium: 1.0,
        rate_bound_slack_slow: 1.0,
        mode_budget_medium: 2,
        full_part_rdo_medium: true,
        palette_medium: true,
        palette_budget_medium: 1,
        angle_deltas_medium: true,
        angle_hog_gate: 0.3,
        angle_hog_win: 15.0,
        joint_angle_count: 2,
        joint_angle_all: false,
        full_chroma_rdo_medium: true,
        min_size_16_fast: false,
        min_size_16_medium: true,
        rect64: false,
        rect64_bias: 1.0,
        rect64_screen: false,
        tx64: false,
        tx64_bias: 1.0,
        tx64_none: false,
        tx64_depth2: true,
        ad_vh: true,
        rect_ad: true,
        ad_runner: true,
        uv_ad: true,
        rect_diag: true,
        uv_ad32: true,
        uv422_modes: true,
        mlam_scale_420: 0.8,
        mlam_scale_422: 0.65,
        mlam_scale_444: 0.6,
        min_size_16_slow: false,
        guided16_k_fast: 0.0,
        guided16_k_medium: 0.0,
        guided16_k_slow: 0.0,
        split4_breakout_slow: 1.5,
        exact_8x8_mode_rate: true,
        split4_decision_txtypes: false,
        palette_smooth: 24,
        palette_smooth_guard: true,
        palette_viterbi_444: 0.35,
        palette_viterbi_420: 0.5,
        palette_viterbi_proxy: true,
        palette_viterbi_screen_only: 1,
        palette_viterbi_uv: 0.0,
        palette_dp_centers: true,
        palette64: true,
        ibc_gate_beta: 32.0,
        ibc_gate_dv: 1.0,
        ibc_gate_prescreen: 0.02,
        ibc_wavefront_reach: 2,
        ibc_odd_dv: true,
        ibc_residual: false,
        ibc_residual_cands: 2,
        screen_few_share: 0.10,
        none32_split_bias_444_top: 1.03,
        none64_split_bias_444_top: 1.03,
        top_bias_knee_444: 40.0,
        top_bias_width_444: 20.0,
        uv_pal_gate_k: 1.0,
        trellis_dc_ctx0: true,
        trellis_zero_bits: 1.0,
        trellis_dc_only_scan: false,
        trellis_aom: false,
        trellis_eob_reprice: false,
        trellis_eob_zero_term: false,
        trellis_pad_stride: false,
        beam_sparse_slot: true,
        trellis_calib: 1.15,
        hbd_trellis_top: 0.174,
        hbd_trellis_hi: 36.0,
        hbd_trellis_lo: 20.0,
        hbd_trellis_top_422: 1.0,
        hbd_trellis_hi_422: 24.0,
        hbd_trellis_lo_422: 10.0,
        tilt_base_444: 1.9,
        tilt_extra_444: 1.4,
        tilt_base_420: 3.5,
        tilt_extra_420: 0.7,
        wsse_420: true,
        wsse_422: true,
        wsse_444: true,
        wsse_400: true,
        wsse_scale: 1.0,
    };
}

/// Fixed, calibrated encoder constants. Without the `tuning` feature there is
/// no runtime state; with it, `MT_TUNING_JSON` (offline parameter study only,
/// see parameters_fit/) overrides fields from a flat JSON object.
#[cfg(not(feature = "tuning"))]
mod imp {
    use super::Tuning;

    #[inline(always)]
    pub(crate) const fn get() -> &'static Tuning {
        &Tuning::SHIPPED
    }
}

#[cfg(feature = "tuning")]
mod imp {
    use super::Tuning;
    use std::sync::OnceLock;

    static TUNING: OnceLock<Tuning> = OnceLock::new();

    pub(crate) fn get() -> &'static Tuning {
        TUNING.get_or_init(|| {
            let Ok(path) = std::env::var("MT_TUNING_JSON") else {
                return Tuning::SHIPPED;
            };
            let Ok(text) = std::fs::read_to_string(&path) else {
                panic!("MT_TUNING_JSON={path}: unreadable");
            };
            parse(&text)
        })
    }

    /// Minimal reader for a *flat* JSON object of numbers and booleans. Any
    /// unknown key is a hard error so a typo can never become a silent no-op
    /// trial. The key list is every field of `Tuning`, in declaration order.
    fn parse(text: &str) -> Tuning {
        let mut t = Tuning::SHIPPED;
        let body = text
            .trim()
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .unwrap_or_else(|| panic!("MT_TUNING_JSON: expected a flat JSON object"));
        for entry in body.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let (key, value) = entry
                .split_once(':')
                .unwrap_or_else(|| panic!("MT_TUNING_JSON: malformed entry {entry:?}"));
            let key = key.trim().trim_matches('"');
            let value = value.trim();
            let num = |v: &str| -> f32 {
                v.parse()
                    .unwrap_or_else(|_| panic!("MT_TUNING_JSON: {key}: not a number: {v:?}"))
            };
            let flag = |v: &str| -> bool {
                match v {
                    "true" => true,
                    "false" => false,
                    _ => panic!("MT_TUNING_JSON: {key}: not a bool: {v:?}"),
                }
            };
            match key {
                "rect16_bias" => t.rect16_bias = num(value),
                "rect16_bias_422" => t.rect16_bias_422 = num(value),
                "rect16_horz_444" => t.rect16_horz_444 = flag(value),
                "rect16_vert_444" => t.rect16_vert_444 = flag(value),
                "rect16_420_medium" => t.rect16_420_medium = flag(value),
                "none16_top_bias_420" => t.none16_top_bias_420 = num(value),
                "none16_top_bias_422" => t.none16_top_bias_422 = num(value),
                "none16_top_bias_444" => t.none16_top_bias_444 = num(value),
                "quad4_bias" => t.quad4_bias = num(value),
                "part_budget_slow" => t.part_budget_slow = num(value) as u32,
                "part_budget_qmin" => t.part_budget_qmin = num(value) as u32,
                "part_budget_444_only" => t.part_budget_444_only = flag(value),
                "rect_tx_split" => t.rect_tx_split = flag(value),
                "rect_uv_search" => t.rect_uv_search = flag(value),
                "rect_uv_joint" => t.rect_uv_joint = flag(value),
                "rect_uv_joint_k" => t.rect_uv_joint_k = num(value) as u32,
                "rect_full_modes" => t.rect_full_modes = flag(value),
                "strip_modes" => t.strip_modes = flag(value),
                "strip_uv_joint" => t.strip_uv_joint = flag(value),
                "rect_split_seq" => t.rect_split_seq = flag(value),
                "recon_src_init" => t.recon_src_init = flag(value),
                "split16_bias" => t.split16_bias = num(value),
                "split4_bias" => t.split4_bias = num(value),
                "part_budget_medium" => t.part_budget_medium = num(value) as u32,
                "part_budget_fast" => t.part_budget_fast = num(value) as u32,
                "top_ease" => t.top_ease = num(value),
                "top_ease_knee" => t.top_ease_knee = num(value),
                "top_ease_width" => t.top_ease_width = num(value),
                "qm_hq_knee" => t.qm_hq_knee = num(value),
                "qm_hq_width" => t.qm_hq_width = num(value),
                "top_band_q" => t.top_band_q = num(value) as u32,
                "none32_split_bias" => t.none32_split_bias = num(value),
                "none64_split_bias" => t.none64_split_bias = num(value),
                "top_none_bias_420_hi" => t.top_none_bias_420_hi = num(value),
                "top_none_bias_420_lo" => t.top_none_bias_420_lo = num(value),
                "top_444_uac" => t.top_444_uac = num(value),
                "mode_lambda_lo_floor" => t.mode_lambda_lo_floor = num(value),
                "mode_lambda_lo_full" => t.mode_lambda_lo_full = num(value),
                "mode_lambda_lo_w" => t.mode_lambda_lo_w = num(value),
                "lf_sharpness" => t.lf_sharpness = num(value) as u32,
                "lf_level_scale" => t.lf_level_scale = num(value),
                "lf_uv_level_scale" => t.lf_uv_level_scale = num(value),
                "lf_uv_level_scale_420" => t.lf_uv_level_scale_420 = num(value),
                "lf_search" => t.lf_search = flag(value),
                "lf_search_bias" => t.lf_search_bias = num(value),
                "lf_search_start0" => t.lf_search_start0 = flag(value),
                "lf_search_metric" => t.lf_search_metric = num(value) as u32,
                "cdef_margin" => t.cdef_margin = num(value),
                "cdef_gate" => t.cdef_gate = num(value),
                "cdef_chroma_w" => t.cdef_chroma_w = num(value),
                "cdef_wsse" => t.cdef_wsse = flag(value),
                "cdef_lambda_d" => t.cdef_lambda_d = flag(value),
                "wiener_wsse" => t.wiener_wsse = flag(value),
                "uv_pal_gate_wsse" => t.uv_pal_gate_wsse = flag(value),
                "trial_struct_wsse" => t.trial_struct_wsse = flag(value),
                "uv16_444_mlam_c" => t.uv16_444_mlam_c = flag(value),
                "tx_guard_shift" => t.tx_guard_shift = num(value) as u32,
                "tx_guard_shift_444" => t.tx_guard_shift_444 = num(value) as u32,
                "proxy_luma_bridge" => t.proxy_luma_bridge = flag(value),
                "proxy_luma_scale" => t.proxy_luma_scale = num(value),
                "proxy_chroma_factor" => t.proxy_chroma_factor = num(value),
                "luma64_wsse" => t.luma64_wsse = flag(value),
                "chroma_part_w_sub" => t.chroma_part_w_sub = num(value),
                "chroma_part_w_444" => t.chroma_part_w_444 = num(value),
                "emit_mlam_c_420" => t.emit_mlam_c_420 = num(value),
                "uv420_tail_delta" => t.uv420_tail_delta = num(value),
                "uv420_mid_d" => t.uv420_mid_d = num(value),
                "mode_budget_slow_sub" => t.mode_budget_slow_sub = num(value) as u32,
                "edge_fill" => t.edge_fill = flag(value),
                "edge_fill_hi" => t.edge_fill_hi = num(value),
                "edge_fill_lo" => t.edge_fill_lo = num(value),
                "edge_fill_frac" => t.edge_fill_frac = num(value),
                "recon_fill_k" => t.recon_fill_k = num(value),
                "recon_fill_dim" => t.recon_fill_dim = num(value) as u32,
                "ad8_full" => t.ad8_full = flag(value),
                "ad8_runner" => t.ad8_runner = flag(value),
                "joint_angle_search" => t.joint_angle_search = flag(value),
                "split4_ad" => t.split4_ad = flag(value),
                "trial_part" => t.trial_part = num(value) as u32,
                "trial_part_xslow" => t.trial_part_xslow = num(value) as u32,
                "trial_child_prdo" => t.trial_child_prdo = flag(value),
                "trial_split_bias" => t.trial_split_bias = num(value),
                "trial_gate" => t.trial_gate = num(value) as u32,
                "trial_edge_hi" => t.trial_edge_hi = num(value),
                "trial_edge_lo" => t.trial_edge_lo = num(value),
                "trial_rect" => t.trial_rect = flag(value),
                "trial_rect_bias" => t.trial_rect_bias = num(value),
                "trial_rect_sse_gate" => t.trial_rect_sse_gate = num(value) as u32,
                "trial_rect_dy_w" => t.trial_rect_dy_w = num(value),
                "trial_struct_k" => t.trial_struct_k = num(value),
                "trial_rect_xslow" => t.trial_rect_xslow = flag(value),
                "trial_rect_sse_gate_xslow" => t.trial_rect_sse_gate_xslow = num(value) as u32,
                "trial_struct_k_xslow" => t.trial_struct_k_xslow = num(value),
                "trial_margin" => t.trial_margin = num(value),
                "trial_margin_pruned" => t.trial_margin_pruned = num(value) as u32,
                "trial_margin_edge" => t.trial_margin_edge = num(value),
                "proxy_ad" => t.proxy_ad = flag(value),
                "ad16_full" => t.ad16_full = flag(value),
                "uv8_vs_cfl" => t.uv8_vs_cfl = flag(value),
                "uv8_joint" => t.uv8_joint = flag(value),
                "uv8_joint_k" => t.uv8_joint_k = num(value) as u32,
                "uv16_joint" => t.uv16_joint = flag(value),
                "prdo_k" => t.prdo_k = num(value),
                "prdo_k_444" => t.prdo_k_444 = num(value),
                "trellis_dz" => t.trellis_dz = num(value),
                "cfl_alpha_rd_steps" => t.cfl_alpha_rd_steps = num(value) as u32,
                "cfl_alpha_rd_mode" => t.cfl_alpha_rd_mode = num(value) as u32,
                "filter_intra_budget" => t.filter_intra_budget = num(value) as u32,
                "filter_intra_budget_444" => t.filter_intra_budget_444 = num(value) as u32,
                "filter_intra_rd_only" => t.filter_intra_rd_only = flag(value),
                "filter_intra_rect" => t.filter_intra_rect = flag(value),
                "uv8_cfl_mlam_c" => t.uv8_cfl_mlam_c = flag(value),
                "uv8_ad" => t.uv8_ad = flag(value),
                "tx32_depth2" => t.tx32_depth2 = flag(value),
                "tx32_split_rd" => t.tx32_split_rd = flag(value),
                "uv422_mid_scale" => t.uv422_mid_scale = num(value),
                "split_breakout_slow" => t.split_breakout_slow = num(value),
                "fixed_size_fast" => t.fixed_size_fast = num(value) as u32,
                "fixed_size_medium" => t.fixed_size_medium = num(value) as u32,
                "fixed_size_slow" => t.fixed_size_slow = num(value) as u32,
                "part_signal_bits" => t.part_signal_bits = num(value),
                "part_signal_seam_drop" => t.part_signal_seam_drop = num(value),
                "split32_signal_mult" => t.split32_signal_mult = num(value),
                "b64_refinement_window" => t.b64_refinement_window = num(value),
                "b64_split_refinement" => t.b64_split_refinement = num(value),
                "vbp_thresh_420" => t.vbp_thresh_420 = num(value),
                "seam_w_420" => t.seam_w_420 = num(value),
                "seam_w_422" => t.seam_w_422 = num(value),
                "joint_large_gain" => t.joint_large_gain = num(value),
                "satd_base" => t.satd_base = num(value),
                "satd_top" => t.satd_top = num(value),
                "satd_knee_lo" => t.satd_knee_lo = num(value),
                "satd_knee_hi" => t.satd_knee_hi = num(value),
                "aq_slope" => t.aq_slope = num(value),
                "aq_max_delta" => t.aq_max_delta = num(value),
                "aq_r" => t.aq_r = num(value),
                "trellis_tilt_mag_cap" => t.trellis_tilt_mag_cap = num(value),
                "smooth_v_uv_signal_bits" => t.smooth_v_uv_signal_bits = num(value),
                "local_ref_blend" => t.local_ref_blend = num(value),
                "exact_part_bits" => t.exact_part_bits = flag(value),
                "chroma_uv_mode_bits" => t.chroma_uv_mode_bits = flag(value),
                "chroma_refine_topk" => t.chroma_refine_topk = num(value) as usize,
                "palette_proxy_finalists" => t.palette_proxy_finalists = num(value) as usize,
                "palette_proxy_ranked" => t.palette_proxy_ranked = flag(value),
                "rect_decision_refine" => t.rect_decision_refine = flag(value),
                "exact_part_bits_3264" => t.exact_part_bits_3264 = flag(value),
                "exact_part_bits_3264_444" => t.exact_part_bits_3264_444 = flag(value),
                "exact_part_bits_8" => t.exact_part_bits_8 = flag(value),
                "rect8_enabled" => t.rect8_enabled = flag(value),
                "rect8_bias" => t.rect8_bias = num(value),
                "edge_rect" => t.edge_rect = flag(value),
                "edge_rect_bias" => t.edge_rect_bias = num(value),
                "vbp_medium_420" => t.vbp_medium_420 = flag(value),
                "vbp_444" => t.vbp_444 = flag(value),
                "vbp_422" => t.vbp_422 = flag(value),
                "vbp_thresh_hi" => t.vbp_thresh_hi = num(value),
                "proxy_rate_fast" => t.proxy_rate_fast = flag(value),
                "proxy_rate_medium" => t.proxy_rate_medium = flag(value),
                "proxy_rate_slow" => t.proxy_rate_slow = flag(value),
                "rate_bound_slack_fast" => t.rate_bound_slack_fast = num(value),
                "rate_bound_slack_medium" => t.rate_bound_slack_medium = num(value),
                "rate_bound_slack_slow" => t.rate_bound_slack_slow = num(value),
                "mode_budget_medium" => t.mode_budget_medium = num(value) as u32,
                "full_part_rdo_medium" => t.full_part_rdo_medium = flag(value),
                "palette_medium" => t.palette_medium = flag(value),
                "palette_budget_medium" => t.palette_budget_medium = num(value) as u32,
                "angle_deltas_medium" => t.angle_deltas_medium = flag(value),
                "angle_hog_gate" => t.angle_hog_gate = num(value),
                "angle_hog_win" => t.angle_hog_win = num(value),
                "joint_angle_count" => t.joint_angle_count = num(value) as u32,
                "joint_angle_all" => t.joint_angle_all = flag(value),
                "full_chroma_rdo_medium" => t.full_chroma_rdo_medium = flag(value),
                "min_size_16_fast" => t.min_size_16_fast = flag(value),
                "min_size_16_medium" => t.min_size_16_medium = flag(value),
                "rect64" => t.rect64 = flag(value),
                "rect64_bias" => t.rect64_bias = num(value),
                "rect64_screen" => t.rect64_screen = flag(value),
                "tx64" => t.tx64 = flag(value),
                "tx64_bias" => t.tx64_bias = num(value),
                "tx64_none" => t.tx64_none = flag(value),
                "tx64_depth2" => t.tx64_depth2 = flag(value),
                "ad_vh" => t.ad_vh = flag(value),
                "rect_ad" => t.rect_ad = flag(value),
                "ad_runner" => t.ad_runner = flag(value),
                "uv_ad" => t.uv_ad = flag(value),
                "rect_diag" => t.rect_diag = flag(value),
                "uv_ad32" => t.uv_ad32 = flag(value),
                "uv422_modes" => t.uv422_modes = flag(value),
                "mlam_scale_420" => t.mlam_scale_420 = num(value),
                "mlam_scale_422" => t.mlam_scale_422 = num(value),
                "mlam_scale_444" => t.mlam_scale_444 = num(value),
                "min_size_16_slow" => t.min_size_16_slow = flag(value),
                "guided16_k_fast" => t.guided16_k_fast = num(value),
                "guided16_k_medium" => t.guided16_k_medium = num(value),
                "guided16_k_slow" => t.guided16_k_slow = num(value),
                "split4_breakout_slow" => t.split4_breakout_slow = num(value),
                "exact_8x8_mode_rate" => t.exact_8x8_mode_rate = flag(value),
                "split4_decision_txtypes" => t.split4_decision_txtypes = flag(value),
                "palette_smooth" => t.palette_smooth = num(value) as u32,
                "palette_smooth_guard" => t.palette_smooth_guard = flag(value),
                "palette_viterbi_444" => t.palette_viterbi_444 = num(value),
                "palette_viterbi_420" => t.palette_viterbi_420 = num(value),
                "palette_viterbi_proxy" => t.palette_viterbi_proxy = flag(value),
                "palette_viterbi_screen_only" => t.palette_viterbi_screen_only = num(value) as u32,
                "palette_viterbi_uv" => t.palette_viterbi_uv = num(value),
                "palette_dp_centers" => t.palette_dp_centers = flag(value),
                "palette64" => t.palette64 = flag(value),
                "ibc_gate_beta" => t.ibc_gate_beta = num(value),
                "ibc_gate_dv" => t.ibc_gate_dv = num(value),
                "ibc_gate_prescreen" => t.ibc_gate_prescreen = num(value),
                "ibc_wavefront_reach" => t.ibc_wavefront_reach = num(value) as u32,
                "ibc_odd_dv" => t.ibc_odd_dv = flag(value),
                "ibc_residual" => t.ibc_residual = flag(value),
                "ibc_residual_cands" => t.ibc_residual_cands = num(value) as u32,
                "screen_few_share" => t.screen_few_share = num(value),
                "none32_split_bias_444_top" => t.none32_split_bias_444_top = num(value),
                "none64_split_bias_444_top" => t.none64_split_bias_444_top = num(value),
                "top_bias_knee_444" => t.top_bias_knee_444 = num(value),
                "top_bias_width_444" => t.top_bias_width_444 = num(value),
                "uv_pal_gate_k" => t.uv_pal_gate_k = num(value),
                "trellis_dc_ctx0" => t.trellis_dc_ctx0 = flag(value),
                "trellis_zero_bits" => t.trellis_zero_bits = num(value),
                "trellis_dc_only_scan" => t.trellis_dc_only_scan = flag(value),
                "trellis_aom" => t.trellis_aom = flag(value),
                "trellis_eob_reprice" => t.trellis_eob_reprice = flag(value),
                "trellis_eob_zero_term" => t.trellis_eob_zero_term = flag(value),
                "trellis_pad_stride" => t.trellis_pad_stride = flag(value),
                "beam_sparse_slot" => t.beam_sparse_slot = flag(value),
                "trellis_calib" => t.trellis_calib = num(value),
                "hbd_trellis_top" => t.hbd_trellis_top = num(value),
                "hbd_trellis_hi" => t.hbd_trellis_hi = num(value),
                "hbd_trellis_lo" => t.hbd_trellis_lo = num(value),
                "hbd_trellis_top_422" => t.hbd_trellis_top_422 = num(value),
                "hbd_trellis_hi_422" => t.hbd_trellis_hi_422 = num(value),
                "hbd_trellis_lo_422" => t.hbd_trellis_lo_422 = num(value),
                "tilt_base_444" => t.tilt_base_444 = num(value),
                "tilt_extra_444" => t.tilt_extra_444 = num(value),
                "tilt_base_420" => t.tilt_base_420 = num(value),
                "tilt_extra_420" => t.tilt_extra_420 = num(value),
                "wsse_420" => t.wsse_420 = flag(value),
                "wsse_422" => t.wsse_422 = flag(value),
                "wsse_444" => t.wsse_444 = flag(value),
                "wsse_400" => t.wsse_400 = flag(value),
                "wsse_scale" => t.wsse_scale = num(value),
                other => panic!("MT_TUNING_JSON: unknown key {other:?}"),
            }
        }
        t
    }
}

pub(crate) use imp::get;

/// Abort-bound multiplier for the exact rate estimator (1.0 = exact).
#[inline]
pub(crate) fn rate_bound_slack(speed: crate::avif::Speed) -> f32 {
    let t = get();
    match speed {
        crate::avif::Speed::Fast => t.rate_bound_slack_fast,
        crate::avif::Speed::Medium => t.rate_bound_slack_medium,
        crate::avif::Speed::ExtraSlow | crate::avif::Speed::Slow => t.rate_bound_slack_slow,
    }
}

/// Whether `speed` forbids the 16 -> 8 descent.
#[inline]
pub(crate) fn min_size_16(speed: crate::avif::Speed) -> bool {
    let t = get();
    match speed {
        crate::avif::Speed::Fast => t.min_size_16_fast,
        crate::avif::Speed::Medium => t.min_size_16_medium,
        crate::avif::Speed::ExtraSlow | crate::avif::Speed::Slow => t.min_size_16_slow,
    }
}

/// Guided-partition threshold at the 16 node for `speed` (0 = off).
#[inline]
pub(crate) fn guided16_k(speed: crate::avif::Speed) -> f32 {
    let t = get();
    match speed {
        crate::avif::Speed::Fast => t.guided16_k_fast,
        crate::avif::Speed::Medium => t.guided16_k_medium,
        crate::avif::Speed::ExtraSlow | crate::avif::Speed::Slow => t.guided16_k_slow,
    }
}

/// The forced uniform luma block size for `speed`, or 0 for full RDO.
///
/// See `fixed_size_fast` for what this does and why it is not just another
/// search-gating heuristic.
#[inline]
pub(crate) fn fixed_size(speed: crate::avif::Speed) -> u32 {
    let t = get();
    match speed {
        crate::avif::Speed::Fast => t.fixed_size_fast,
        crate::avif::Speed::Medium => t.fixed_size_medium,
        crate::avif::Speed::ExtraSlow | crate::avif::Speed::Slow => t.fixed_size_slow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encoder paths always read the fixed shipped calibration.
    #[test]
    fn defaults_are_the_shipped_constants() {
        let t = get();
        assert_eq!(t.rect16_bias, Tuning::SHIPPED.rect16_bias);
        assert_eq!(t.rect16_horz_444, Tuning::SHIPPED.rect16_horz_444);
        assert_eq!(t.quad4_bias, Tuning::SHIPPED.quad4_bias);
    }
}
