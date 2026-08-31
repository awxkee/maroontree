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
    /// detail a merged large block loses).
    pub(crate) none32_split_bias: f32,
    pub(crate) none64_split_bias: f32,
    /// 4:2:0 top-band NONE bias, split at qindex 20.
    pub(crate) top_none_bias_420_hi: f32,
    pub(crate) top_none_bias_420_lo: f32,
    /// 4:4:4 top-band chroma AC qindex delta (`quant::TOP_444_UAC`).
    pub(crate) top_444_uac: f32,
    /// Scale on the 4:2:0 mid-band chroma qindex delta when applied at 4:2:2
    /// (peak -14 at 0.5). SHIPPED 0.5 on observable quality, not metric: at
    /// scale 0 mid-band 4:2:2 chroma ran at full luma q and turned smooth skin
    /// Cr gradients into blotchy DC patches (ClassE_set70 face, q50-65); 0.5
    /// is the visual knee (+2-3% bytes on the corpus, SS2-neutral BD).
    pub(crate) uv422_mid_scale: f32,
    /// Source-domain SPLIT breakout ratio at Slow.
    pub(crate) split_breakout_slow: f32,
    pub(crate) fixed_size_fast: u32,
    pub(crate) fixed_size_medium: u32,
    pub(crate) fixed_size_slow: u32,

    /// Partition-signaling entry fee, in bits, charged to every non-NONE candidate.
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
    /// luma_satd_scale at the top quality band (4:4:4 only).
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
    /// Guided partitioning at the 16 node: commit to SPLIT without pricing NONE
    /// when `m_split * k < m_none` in the source model. 0 = off (full R-D).
    pub(crate) vbp_medium_420: bool,
    pub(crate) vbp_444: bool,
    pub(crate) vbp_422: bool,
    /// Threshold for the non-4:2:0 formats (4:2:0 uses `vbp_thresh_420`).
    pub(crate) vbp_thresh_hi: f32,
    // --- Medium leaf-work gates (mode budgets / transform + chroma trials) ---
    /// `Speed::Medium` luma mode-search budget (shipped 2; Slow is 3, or 5 at 4:4:4).
    /// Cheap magnitude-based rate proxy instead of the exact CDF walk.
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
    pub(crate) full_chroma_rdo_medium: bool,
    pub(crate) min_size_16_fast: bool,
    pub(crate) min_size_16_medium: bool,
    pub(crate) min_size_16_slow: bool,
    pub(crate) guided16_k_fast: f32,
    pub(crate) guided16_k_medium: f32,
    pub(crate) guided16_k_slow: f32,
    /// Source-domain SPLIT4 breakout ratio at Slow (was hardcoded 1.5).
    pub(crate) split4_breakout_slow: f32,
    pub(crate) split4_legacy_record: bool,
    pub(crate) exact_8x8_mode_rate: bool,
    pub(crate) split4_decision_txtypes: bool,
    /// Bounding probe: 0 = price skip=false (shipped), 1 = skip=true, 2 = free.
    pub(crate) block_skip_price: u32,
}

impl Tuning {
    /// The shipped constants. Changing a value here changes the encoder.
    pub(crate) const SHIPPED: Tuning = Tuning {
        rect16_bias: 1.0708447,
        rect16_bias_422: 1.0,
        rect16_horz_444: true,
        rect16_vert_444: true,
        rect16_420_medium: false,
        none16_top_bias_420: 1.15,
        none16_top_bias_422: 1.25,
        none16_top_bias_444: 1.0441984,
        quad4_bias: 1.1,
        part_budget_slow: 3,
        part_budget_qmin: 0,
        part_budget_444_only: true,
        part_budget_medium: 0,
        part_budget_fast: 0,
        top_ease: 1.0,
        top_ease_knee: 20.0,
        top_ease_width: 12.0,
        qm_hq_knee: 12.0,
        qm_hq_width: 5.0,
        top_band_q: 55,
        none32_split_bias: 1.03,
        none64_split_bias: 1.03,
        top_none_bias_420_hi: 1.45,
        top_none_bias_420_lo: 1.15,
        top_444_uac: 6.0,
        uv422_mid_scale: 0.5,
        split_breakout_slow: 1.5,
        fixed_size_fast: 16,
        fixed_size_medium: 0,
        fixed_size_slow: 0,
        part_signal_bits: 24.0,
        part_signal_seam_drop: 22.0,
        split32_signal_mult: 1.0,
        b64_refinement_window: 0.95,
        b64_split_refinement: 0.5,
        vbp_thresh_420: 0.002,
        seam_w_420: 0.0,
        seam_w_422: 60.0,
        joint_large_gain: 0.995,
        satd_base: 1.25,
        satd_top: 2.5,
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
        full_chroma_rdo_medium: true,
        min_size_16_fast: false,
        min_size_16_medium: true,
        min_size_16_slow: false,
        guided16_k_fast: 0.0,
        guided16_k_medium: 0.0,
        guided16_k_slow: 0.0,
        split4_breakout_slow: 1.5,
        split4_legacy_record: false,
        exact_8x8_mode_rate: false,
        split4_decision_txtypes: false,
        block_skip_price: 0,
    };
}

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

    /// Minimal reader for a *flat* JSON object of numbers and booleans, e.g.
    /// `{"rect16_bias": 1.04, "rect16_horz_444": true}`.
    ///
    /// Deliberately not a general JSON parser and deliberately not a
    /// dependency: the lib ships with one crate dependency and this file must
    /// not add a second. Anything it cannot understand is a hard error, so a
    /// typo'd key can never be silently ignored into a bogus "no effect" trial
    /// — the failure mode that wastes a whole study.
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
                "palette_proxy_ranked" => t.palette_proxy_ranked = flag(value),
                "palette_proxy_finalists" => {
                    t.palette_proxy_finalists = value.parse().unwrap_or(t.palette_proxy_finalists)
                }
                "chroma_refine_topk" => {
                    t.chroma_refine_topk = value.parse().unwrap_or(t.chroma_refine_topk)
                }
                "rect_decision_refine" => t.rect_decision_refine = flag(value),
                "exact_part_bits_3264" => t.exact_part_bits_3264 = flag(value),
                "exact_part_bits_3264_444" => t.exact_part_bits_3264_444 = flag(value),
                "exact_part_bits_8" => t.exact_part_bits_8 = flag(value),
                "rect8_enabled" => t.rect8_enabled = flag(value),
                "rect8_bias" => t.rect8_bias = num(value),
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
                "full_chroma_rdo_medium" => t.full_chroma_rdo_medium = flag(value),
                "min_size_16_fast" => t.min_size_16_fast = flag(value),
                "min_size_16_medium" => t.min_size_16_medium = flag(value),
                "min_size_16_slow" => t.min_size_16_slow = flag(value),
                "guided16_k_fast" => t.guided16_k_fast = num(value),
                "guided16_k_medium" => t.guided16_k_medium = num(value),
                "guided16_k_slow" => t.guided16_k_slow = num(value),
                "split4_breakout_slow" => t.split4_breakout_slow = num(value),
                "split4_legacy_record" => t.split4_legacy_record = flag(value),
                "exact_8x8_mode_rate" => t.exact_8x8_mode_rate = flag(value),
                "split4_decision_txtypes" => t.split4_decision_txtypes = flag(value),
                "block_skip_price" => t.block_skip_price = num(value) as u32,
                "part_budget_medium" => t.part_budget_medium = num(value) as u32,
                "part_budget_fast" => t.part_budget_fast = num(value) as u32,
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
        crate::avif::Speed::Slow => t.rate_bound_slack_slow,
    }
}

/// Whether `speed` forbids the 16 -> 8 descent.
#[inline]
pub(crate) fn min_size_16(speed: crate::avif::Speed) -> bool {
    let t = get();
    match speed {
        crate::avif::Speed::Fast => t.min_size_16_fast,
        crate::avif::Speed::Medium => t.min_size_16_medium,
        crate::avif::Speed::Slow => t.min_size_16_slow,
    }
}

/// Guided-partition threshold at the 16 node for `speed` (0 = off).
#[inline]
pub(crate) fn guided16_k(speed: crate::avif::Speed) -> f32 {
    let t = get();
    match speed {
        crate::avif::Speed::Fast => t.guided16_k_fast,
        crate::avif::Speed::Medium => t.guided16_k_medium,
        crate::avif::Speed::Slow => t.guided16_k_slow,
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
        crate::avif::Speed::Slow => t.fixed_size_slow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the whole design: with the feature off there is no runtime
    /// state, and with it on but unconfigured the values are still the shipped
    /// ones. Either way a "defaults" trial must be a true no-op.
    #[test]
    fn defaults_are_the_shipped_constants() {
        let t = get();
        assert_eq!(t.rect16_bias, Tuning::SHIPPED.rect16_bias);
        assert_eq!(t.rect16_horz_444, Tuning::SHIPPED.rect16_horz_444);
        assert_eq!(t.quad4_bias, Tuning::SHIPPED.quad4_bias);
    }
}
