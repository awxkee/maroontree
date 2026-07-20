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

#[allow(dead_code)]
mod dpb;
mod frame;
mod gop;
mod headers;
pub mod ivf;
pub(crate) mod mc;
pub(crate) mod me;
pub(crate) mod mv;
pub(crate) mod mvd;
mod pipeline;
pub(crate) mod ratectrl;
pub(crate) mod rd;

use crate::av2::Av2Encoder;
use crate::av2::layout::Layout;
use crate::av2::video::gop::{FrameType, Gop};
use crate::av2::video::ivf::IvfWriter;
use crate::av2::video::pipeline::{
    FrameAnalysis, FrameDecision, FrameEmit, ScratchArenas, VideoFrameState,
};
use crate::err::EncodeError;
use crate::{BitDepth, ChromaFormat, Cicp, Pixel, PlanarImage, Speed};

/// One coded frame (raw AV2 OBUs, container-independent).
#[derive(Debug)]
pub struct Packet {
    pub data: Vec<u8>,
    pub key: bool,
    pub pts: u64,
}

/// Product-level video speed/quality presets. These are intentionally separate
/// from the still-image [`Speed`] tiers so video policy can grow without changing
/// still-image behavior or preset compatibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VideoPreset {
    Realtime,
    Fast,
    Balanced,
    Quality,
    #[default]
    Slow,
    Reference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoTransformSearch {
    DctOnly,
    WinnerOnly,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoRdoqEffort {
    Off,
    FinalMode,
    PerCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoThreadingPolicy {
    LowDelay,
    Tiles,
    TilesAndWavefront,
    Serial,
}

/// Declared policy behind a [`VideoPreset`]. `reference_count` reports the active
/// implementation limit, not a future target. Partition fields become execution controls
/// as the corresponding to inter block sto earches land.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoPresetConfig {
    pub search_range: u16,
    /// Normalized 8-bit SAD per pixel below which integer search is skipped;
    /// zero keeps exhaustive coarse+dense search.
    pub predictor_gate_sad_per_pixel: u16,
    /// Full-pel radius around the SAD winner reranked with transform-domain SATD.
    pub integer_satd_radius: u8,
    pub minimum_block_size: u8,
    pub maximum_partition_depth: u8,
    pub rectangular_partitions: bool,
    pub reference_count: u8,
    pub transform_search: VideoTransformSearch,
    pub rdoq_effort: VideoRdoqEffort,
    pub lookahead_frames: u8,
    pub threading: VideoThreadingPolicy,
}

impl VideoPreset {
    pub const fn config(self) -> VideoPresetConfig {
        use VideoPreset::*;
        match self {
            Realtime => VideoPresetConfig {
                search_range: 24,
                predictor_gate_sad_per_pixel: 4,
                integer_satd_radius: 0,
                minimum_block_size: 32,
                maximum_partition_depth: 1,
                rectangular_partitions: false,
                reference_count: 1,
                transform_search: VideoTransformSearch::DctOnly,
                rdoq_effort: VideoRdoqEffort::Off,
                lookahead_frames: 1,
                threading: VideoThreadingPolicy::LowDelay,
            },
            Fast => VideoPresetConfig {
                search_range: 48,
                predictor_gate_sad_per_pixel: 2,
                integer_satd_radius: 0,
                minimum_block_size: 16,
                maximum_partition_depth: 2,
                rectangular_partitions: false,
                reference_count: 1,
                transform_search: VideoTransformSearch::WinnerOnly,
                rdoq_effort: VideoRdoqEffort::FinalMode,
                lookahead_frames: 4,
                threading: VideoThreadingPolicy::Tiles,
            },
            Balanced => VideoPresetConfig {
                search_range: 64,
                predictor_gate_sad_per_pixel: 1,
                integer_satd_radius: 1,
                minimum_block_size: 16,
                maximum_partition_depth: 2,
                rectangular_partitions: true,
                reference_count: 2,
                transform_search: VideoTransformSearch::WinnerOnly,
                rdoq_effort: VideoRdoqEffort::FinalMode,
                lookahead_frames: 8,
                threading: VideoThreadingPolicy::TilesAndWavefront,
            },
            Quality => VideoPresetConfig {
                search_range: 96,
                predictor_gate_sad_per_pixel: 0,
                integer_satd_radius: 1,
                minimum_block_size: 8,
                maximum_partition_depth: 3,
                rectangular_partitions: true,
                reference_count: 2,
                transform_search: VideoTransformSearch::Full,
                rdoq_effort: VideoRdoqEffort::FinalMode,
                lookahead_frames: 16,
                threading: VideoThreadingPolicy::TilesAndWavefront,
            },
            Slow => VideoPresetConfig {
                search_range: 128,
                predictor_gate_sad_per_pixel: 0,
                integer_satd_radius: 2,
                minimum_block_size: 8,
                maximum_partition_depth: 4,
                rectangular_partitions: true,
                reference_count: 3,
                transform_search: VideoTransformSearch::Full,
                rdoq_effort: VideoRdoqEffort::PerCandidate,
                lookahead_frames: 32,
                threading: VideoThreadingPolicy::TilesAndWavefront,
            },
            Reference => VideoPresetConfig {
                search_range: 128,
                predictor_gate_sad_per_pixel: 0,
                integer_satd_radius: 2,
                minimum_block_size: 4,
                maximum_partition_depth: 4,
                rectangular_partitions: true,
                reference_count: 3,
                transform_search: VideoTransformSearch::Full,
                rdoq_effort: VideoRdoqEffort::PerCandidate,
                lookahead_frames: 32,
                threading: VideoThreadingPolicy::Serial,
            },
        }
    }
}

/// Stateful low-delay AV2 video encoder with a reusable thread budget and DPB.
pub struct Av2VideoEncoder {
    cfg: Av2Encoder,
    chroma: ChromaFormat,
    thread_budget: usize,
    threads: usize,
    gop: Gop,
    seq_emitted: bool,
    state: VideoFrameState,
    /// Previous frame's normalized 8-bit-scale luma.
    prev_luma: Vec<f32>,
    /// Frames coded since the last keyframe (gates scene-cut min-gap).
    since_key: u64,
    /// Sequence properties locked by the first frame: (width, height, bit_depth).
    seq_lock: Option<(usize, usize, u8)>,
    scratch: ScratchArenas,
    preset: VideoPreset,
    nominal_q: u8,
    cq_max_delta: u8,
    /// Caps how many DPB references the per-frame search considers, overriding the
    /// preset's `reference_count`. `1` forces single-reference prediction (no
    /// second reference is listed, no per-block ref bit). `None` uses the preset.
    reference_limit: Option<usize>,
    /// Minimum fraction of superblocks that must clearly prefer the second
    /// reference before it is listed (per-block `single_ref`-bit overhead only
    /// pays off on occlusion/reveal content). `0.0` always lists it when a second
    /// candidate exists; a large value disables it. Default 0.06.
    second_reference_threshold: f32,
}

impl Av2VideoEncoder {
    /// Build a video encoder from a still-image [`Av2Encoder`] config. `threads`
    /// follows the same convention (`0` = all cores, `1` = serial).
    pub fn new(cfg: Av2Encoder, chroma: ChromaFormat, threads: usize) -> Self {
        // Deblocking and single-tile CDEF are applied before reconstruction capture
        // in the supported 4:2:0 path. Multi-tile CDEF remains gated until tile-edge
        // tap masking is verified. CCSO is not yet DPB-safe.
        let mut cfg = cfg;
        let nominal_q = cfg.base_q_idx();
        if cfg.tune.tile_cols > 1 || cfg.tune.tile_rows > 1 {
            cfg.tune.cdef = false;
        }
        cfg.tune.ccso = false;
        // Mark every frame this encoder drives as video: the per-frame still encode
        // must keep its historical tiled/serial path, not adopt the still-image
        // multithreaded single-tile SB-wavefront default.
        cfg.video_mode
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let n = if threads == 0 {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        } else {
            threads
        };
        cfg.threads = n.max(1);
        Self {
            cfg,
            chroma,
            thread_budget: n.max(1),
            threads: n.max(1),
            gop: Gop::all_intra(),
            seq_emitted: false,
            state: VideoFrameState::default(),
            prev_luma: Vec::new(),
            since_key: 0,
            seq_lock: None,
            scratch: ScratchArenas::new(n.max(1)),
            preset: VideoPreset::Slow,
            nominal_q,
            cq_max_delta: 0,
            reference_limit: None,
            second_reference_threshold: 0.06,
        }
    }

    /// Cap the number of DPB references the search considers (`>= 1`), overriding
    /// the preset. `1` disables per-block multi-reference prediction; higher values
    /// are clamped to the preset's `reference_count`.
    pub fn set_reference_limit(&mut self, n: usize) {
        self.reference_limit = Some(n.max(1));
    }

    /// Fraction of superblocks that must prefer the second reference before it is
    /// listed. `0.0` always lists it when available (max multi-ref); higher values
    /// reserve it for stronger occlusion/reveal content. See the field docs.
    pub fn set_second_reference_threshold(&mut self, threshold: f32) {
        self.second_reference_threshold = threshold;
    }

    /// Worker budget used by per-frame tile and wavefront work.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Apply the currently available execution controls from a video preset.
    /// Changing presets resets accumulated lookahead history; callers should
    /// normally select one before frame 0.
    pub fn set_preset(&mut self, preset: VideoPreset) {
        let policy = preset.config();
        self.preset = preset;
        self.cfg.video_search_range = i32::from(policy.search_range);
        self.cfg.video_predictor_gate = u32::from(policy.predictor_gate_sad_per_pixel);
        self.cfg.video_integer_satd_radius = policy.integer_satd_radius;
        self.cfg.video_min_block_size = policy.minimum_block_size;
        self.cfg.video_max_partition_depth = policy.maximum_partition_depth;
        self.cfg.speed = match policy.transform_search {
            VideoTransformSearch::DctOnly => Speed::Fast,
            VideoTransformSearch::WinnerOnly => Speed::Medium,
            VideoTransformSearch::Full => Speed::Slow,
        };
        let lambda = match policy.rdoq_effort {
            VideoRdoqEffort::Off => 0.0,
            VideoRdoqEffort::FinalMode => 0.05,
            VideoRdoqEffort::PerCandidate => 0.09,
        };
        self.cfg.tune.rdoq_lambda = lambda;
        self.cfg.tune.chroma_rdoq_lambda = lambda;
        self.state
            .rate_control
            .reset_lookahead(policy.lookahead_frames as usize);
        if matches!(policy.threading, VideoThreadingPolicy::Serial) {
            self.threads = 1;
            self.cfg.threads = 1;
            self.scratch = ScratchArenas::new(1);
        } else if self.threads != self.thread_budget {
            self.threads = self.thread_budget;
            self.cfg.threads = self.thread_budget;
            self.scratch = ScratchArenas::new(self.thread_budget);
        }
    }

    pub fn preset(&self) -> VideoPreset {
        self.preset
    }

    pub fn preset_config(&self) -> VideoPresetConfig {
        self.preset.config()
    }

    /// Enable bounded frame-level CQ adaptation around the encoder's configured
    /// base qindex. `max_delta=0` disables adaptation. A range of 4–8 is a useful
    /// starting point for low-delay video.
    pub fn set_frame_cq(&mut self, max_delta: u8) {
        self.cq_max_delta = max_delta.min(32);
        if self.cq_max_delta == 0 {
            self.cfg.set_video_base_q(self.nominal_q);
        }
    }

    /// Qindex selected for the most recently prepared/encoded frame.
    pub fn current_qindex(&self) -> u8 {
        self.cfg.base_q_idx()
    }

    /// Keyframe interval (0 = all-intra). Frame 0 is always Key.
    pub fn set_key_interval(&mut self, interval: u64) {
        self.gop.key_interval = interval;
    }

    /// Enable SAD scene-cut keyframes. `sad_threshold` is the mean per-pixel 8-bit
    /// luma SAD above which an Inter frame is promoted to Key (0 disables). `min_gap`
    /// is the minimum Inter frames since the last keyframe before a cut may fire.
    pub fn set_scene_cut(&mut self, sad_threshold: u32, min_gap: u64) {
        self.gop.scene_cut_sad = sad_threshold;
        self.gop.scene_cut_min_gap = min_gap;
    }

    fn layout(&self) -> Layout {
        match self.chroma {
            ChromaFormat::Monochrome => Layout::Monochrome,
            ChromaFormat::Yuv420 => Layout::I420,
            ChromaFormat::Yuv422 => Layout::I422,
            ChromaFormat::Yuv444 => Layout::I444,
        }
    }

    pub fn push_frame<T: Pixel>(
        &mut self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Packet, EncodeError> {
        let bd = img.bit_depth.bits();
        if self.gop.key_interval != 0
            && (self.chroma != ChromaFormat::Yuv420 || !matches!(bd, 8 | 10))
        {
            return Err(EncodeError::UnsupportedVideoMode {
                chroma: self.chroma,
                bit_depth: BitDepth::from_u8(bd)?,
            });
        }
        if bd != self.cfg.config_for(self.layout()).bit_depth {
            return Err(EncodeError::SequenceMismatch("encoder bit depth"));
        }
        match self.chroma {
            ChromaFormat::Yuv420 => img.validate_420()?,
            ChromaFormat::Yuv422 => img.validate_422()?,
            ChromaFormat::Yuv444 => img.validate_444()?,
            ChromaFormat::Monochrome => img.validate_400()?,
        }
        let this = (img.width, img.height, bd);
        match self.seq_lock {
            Some((w, h, _)) if w != this.0 || h != this.1 => {
                return Err(EncodeError::SequenceMismatch("frame dimensions"));
            }
            Some((_, _, locked_bd)) if locked_bd != this.2 => {
                return Err(EncodeError::SequenceMismatch("bit depth"));
            }
            None => {}
            _ => {}
        }

        self.scratch.prepare(img.width.saturating_mul(img.height));
        let analysis =
            FrameAnalysis::analyze(img, &self.prev_luma, &self.state.rate_control.lookahead);
        let frame_q = ratectrl::select_cq_qindex(
            self.nominal_q,
            self.cq_max_delta,
            analysis.complexity,
            analysis.lookahead_mean,
        );
        self.cfg.set_video_base_q(frame_q);
        let mut ftype = self.gop.frame_type(self.state.order_hint);
        if matches!(ftype, FrameType::Inter)
            && self.gop.scene_cut_sad > 0
            && self.since_key >= self.gop.scene_cut_min_gap
            && !self.prev_luma.is_empty()
        {
            let sad = analysis.scene_score.round() as u32;
            if sad >= self.gop.scene_cut_sad {
                ftype = FrameType::Key;
            }
        }
        let next_since_key = if matches!(ftype, FrameType::Key) {
            0
        } else {
            self.since_key + 1
        };
        let preset_refs = self.preset.config().reference_count as usize;
        let reference_limit = self
            .reference_limit
            .map_or(preset_refs, |n| n.min(preset_refs));
        let latest_slot = self.state.references.latest_slot();
        let mut reference_candidates: Vec<((u64, usize), usize, crate::av2::video::mv::Mv)> =
            Vec::new();
        if matches!(ftype, FrameType::Inter) {
            for slot in self.state.references.populated_slots(reference_limit) {
                let reference = self
                    .state
                    .references
                    .slot(slot)
                    .expect("populated DPB slot");
                if reference.width < img.width || reference.height < img.height {
                    continue;
                }
                if let Some((score, seed)) = analysis.motion_compensated_reference_score(
                    &reference.planes[0],
                    reference.strides[0],
                    img.width,
                    img.height,
                    bd,
                    self.cfg.video_search_range,
                ) {
                    let Some(chroma_score) =
                        analysis.motion_compensated_chroma_score(img, reference, seed)
                    else {
                        continue;
                    };
                    // 4:2:0 contains four luma samples for each U and V sample.
                    let score = score.saturating_mul(4).saturating_add(chroma_score);
                    reference_candidates.push((
                        (score, usize::from(slot != latest_slot)),
                        slot,
                        seed,
                    ));
                }
            }
        }
        reference_candidates.sort_by_key(|&(rank, _, _)| rank);
        let reference_choice = reference_candidates.first().copied();
        let reference_slot = reference_choice.map(|(_, slot, _)| slot);
        // Rank-1 reference: the next-best scored slot. Listing it taxes every inter
        // block one `single_ref` bit, so it only pays off on content where a second
        // reference is genuinely the better predictor for a chunk of the frame
        // (occlusion / reveal / periodic motion). On smooth content the recent frame
        // wins everywhere and the tax is pure overhead — so gate on a cheap per-SB
        // estimate of how much of the frame prefers the second reference. Lossless
        // (`nominal_q == 0`) never lists it.
        let second_reference_slot = reference_candidates
            .get(1)
            .filter(|_| self.nominal_q != 0)
            .filter(|&&(_, second_slot, _)| {
                let (Some(r0), Some(r1)) = (
                    reference_slot.and_then(|s| self.state.references.slot(s)),
                    self.state.references.slot(second_slot),
                ) else {
                    return false;
                };
                analysis.second_reference_preferred_fraction(
                    &r0.planes[0],
                    r0.strides[0],
                    &r1.planes[0],
                    r1.strides[0],
                    img.width,
                    img.height,
                ) >= self.second_reference_threshold
            })
            .map(|&(_, slot, _)| slot);
        let motion_seed = reference_choice
            .map(|(_, _, seed)| seed)
            .unwrap_or(crate::av2::video::mv::Mv::ZERO);
        let refresh_slot = self.state.references.next_refresh_slot();
        let decision = FrameDecision::low_delay(
            ftype,
            reference_slot,
            self.cfg.base_q_idx(),
            &analysis,
            motion_seed,
        );
        debug_assert_eq!(decision.frame_type, ftype);
        debug_assert_eq!(decision.base_q_idx, self.cfg.base_q_idx());
        debug_assert!(decision.analysis_levels > 0);
        debug_assert!(decision.pyramid_samples >= img.width * img.height);
        debug_assert!(decision.coarsest_size.0 > 0 && decision.coarsest_size.1 > 0);
        debug_assert_eq!(decision.scene_score, analysis.scene_score);
        debug_assert!(decision.lookahead_mean.temporal_sad >= 0.0);
        debug_assert_eq!(decision.motion_seed, motion_seed);
        debug_assert!(decision.blocks.is_empty());
        debug_assert_eq!(
            decision.reference_slots.is_empty(),
            matches!(ftype, FrameType::Key)
        );
        self.cfg.inter_tile.store(
            matches!(decision.frame_type, FrameType::Inter),
            std::sync::atomic::Ordering::Relaxed,
        );
        *self.cfg.video_mv_seed.lock().unwrap() = decision.motion_seed;
        if matches!(decision.frame_type, FrameType::Inter) {
            if let Some(r) = decision
                .reference_slots
                .first()
                .and_then(|&slot| self.state.references.slot(slot))
            {
                *self.cfg.last_ref.lock().unwrap() = std::sync::Arc::clone(&r.planes);
            }
            *self.cfg.second_ref.lock().unwrap() = second_reference_slot
                .and_then(|slot| self.state.references.slot(slot))
                .map(|r| std::sync::Arc::clone(&r.planes))
                .unwrap_or_default();
            // Order-hint distances scale the cross-rank derived MV predictor.
            let dist_of = |slot: Option<usize>| -> i32 {
                slot.and_then(|slot| self.state.references.slot(slot))
                    .map(|r| (self.state.order_hint.saturating_sub(r.order_hint)).max(1) as i32)
                    .unwrap_or(1)
            };
            *self.cfg.ref_dists.lock().unwrap() =
                [dist_of(reference_slot), dist_of(second_reference_slot)];
        } else {
            *self.cfg.last_ref.lock().unwrap() = std::sync::Arc::new(Vec::new());
            *self.cfg.second_ref.lock().unwrap() = std::sync::Arc::new(Vec::new());
        }
        self.cfg
            .capture_recon
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let still = match self.chroma {
            ChromaFormat::Yuv420 => self.cfg.encode_yuv420(img, color)?,
            ChromaFormat::Yuv444 => self.cfg.encode_yuv444(img, color)?,
            ChromaFormat::Yuv422 => self.cfg.encode_yuv422(img, color)?,
            ChromaFormat::Monochrome => self.cfg.encode_yuv400(img, color)?,
        };
        let (cw, _ch) = still.coded_dims();
        let mut cfg = still.video_config().clone();
        cfg.allow_intrabc = still.allow_intrabc();
        let chroma_strides = match self.chroma {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => [cw, cw.div_ceil(2), cw.div_ceil(2)],
            ChromaFormat::Yuv444 => [cw, cw, cw],
            ChromaFormat::Monochrome => [cw, 0, 0],
        };
        let emitted = FrameEmit::emit(
            &still,
            &cfg,
            decision.frame_type,
            self.state.order_hint,
            !self.seq_emitted,
            chroma_strides,
            reference_slot.unwrap_or(0),
            second_reference_slot,
            refresh_slot,
        )?;
        // Commit temporal state only after coding and serialization succeed.
        self.seq_emitted = true;
        self.seq_lock = Some(this);
        self.since_key = next_since_key;
        self.state.rate_control.lookahead.push(analysis.complexity);
        self.state.frame_type = decision.frame_type;
        self.state.selected_reference_slot = reference_slot;
        self.state.cdf.begin_frame(decision.frame_type);
        if let Some(reference) = emitted.reference {
            if matches!(decision.frame_type, FrameType::Key) {
                self.state.references.reset_with_key(reference);
            } else {
                self.state.references.refresh_slot(refresh_slot, reference);
            }
        }

        let key = matches!(decision.frame_type, FrameType::Key);
        let pts = self.state.order_hint;
        self.state.order_hint += 1;
        self.prev_luma = analysis.luma;
        Ok(Packet {
            data: emitted.data,
            key,
            pts,
        })
    }

    pub fn flush(&mut self) -> Vec<Packet> {
        Vec::new()
    }

    /// Mean temporal SAD over the recent lookahead window (rate-control signal;
    /// 0 before any frames). Exposed so a rate controller or caller can inspect
    /// the analysis this encoder now collects.
    pub fn lookahead_mean_temporal_sad(&self) -> f32 {
        self.state.rate_control.lookahead.mean_temporal_sad()
    }

    /// Mean spatial activity over the recent lookahead window.
    pub fn lookahead_mean_spatial_activity(&self) -> f32 {
        self.state.rate_control.lookahead.mean_spatial_activity()
    }
}

/// Convenience: encode a whole frame sequence to an in-memory IVF stream.
#[allow(clippy::too_many_arguments)]
pub fn encode_ivf<T: Pixel>(
    cfg: Av2Encoder,
    chroma: ChromaFormat,
    threads: usize,
    frames: &[PlanarImage<T>],
    color: &Cicp,
    fps_num: u32,
    fps_den: u32,
    key_interval: u64,
) -> Result<Vec<u8>, EncodeError> {
    let mut enc = Av2VideoEncoder::new(cfg, chroma, threads);
    enc.set_key_interval(key_interval);
    let (w, h) = frames
        .first()
        .map(|f| (f.width as u16, f.height as u16))
        .unwrap_or((0, 0));
    let mut ivf = IvfWriter::new(w, h, fps_num, fps_den);
    for img in frames {
        let pkt = enc.push_frame(img, color)?;
        ivf.write_frame(&pkt.data, pkt.pts);
    }
    Ok(ivf.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_frame(bit_depth: u8) -> PlanarImage<u16> {
        PlanarImage {
            width: 64,
            height: 64,
            bit_depth: BitDepth::from_u8(bit_depth).unwrap(),
            planes: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }

    fn flat_420(value: u8) -> PlanarImage<u8> {
        PlanarImage {
            width: 64,
            height: 64,
            bit_depth: BitDepth::Eight,
            planes: [
                vec![value; 64 * 64],
                vec![128; 32 * 32],
                vec![128; 32 * 32],
                Vec::new(),
            ],
        }
    }

    fn checker_420(invert: bool) -> PlanarImage<u8> {
        let mut image = flat_420(0);
        for y in 0..64 {
            for x in 0..64 {
                image.planes[0][y * 64 + x] = if ((x + y) & 1 == 0) ^ invert { 16 } else { 240 };
            }
        }
        image
    }

    fn translated_pattern_420(dx: usize, dy: usize) -> PlanarImage<u8> {
        let mut reference = flat_420(0);
        let mut state = 0x1234_5678u32;
        for sample in &mut reference.planes[0] {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u8;
        }
        if dx == 0 && dy == 0 {
            return reference;
        }
        let mut shifted = flat_420(0);
        for y in 0..64 - dy {
            for x in 0..64 - dx {
                shifted.planes[0][y * 64 + x] = reference.planes[0][(y + dy) * 64 + x + dx];
            }
        }
        shifted
    }

    fn diagonal_edge_420() -> PlanarImage<u8> {
        let (w, h) = (128usize, 128usize);
        PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [
                (0..w * h)
                    .map(|i| {
                        let (x, row) = (i % w, i / w);
                        if x + row / 2 > 72 { 240 } else { 16 }
                    })
                    .collect(),
                vec![96; (w / 2) * (h / 2)],
                vec![160; (w / 2) * (h / 2)],
                Vec::new(),
            ],
        }
    }

    fn translated_diagonal_420(dx: usize, dy: usize) -> PlanarImage<u8> {
        let source = diagonal_edge_420();
        let mut shifted = PlanarImage {
            width: source.width,
            height: source.height,
            bit_depth: source.bit_depth,
            planes: [
                vec![16; source.width * source.height],
                source.planes[1].clone(),
                source.planes[2].clone(),
                Vec::new(),
            ],
        };
        for y in 0..source.height - dy {
            for x in 0..source.width - dx {
                shifted.planes[0][(y + dy) * source.width + x + dx] =
                    source.planes[0][y * source.width + x];
            }
        }
        shifted
    }

    fn rectangular_420() -> PlanarImage<u8> {
        let (width, height) = (96usize, 64usize);
        let mut state = 0x6d2b_79f5u32;
        let luma = (0..width * height)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        PlanarImage {
            width,
            height,
            bit_depth: BitDepth::Eight,
            planes: [
                luma,
                vec![96; width * height / 4],
                vec![160; width * height / 4],
                Vec::new(),
            ],
        }
    }

    fn translated_rectangular_420(
        source: &PlanarImage<u8>,
        dx: usize,
        dy: usize,
    ) -> PlanarImage<u8> {
        let mut shifted = PlanarImage {
            width: source.width,
            height: source.height,
            bit_depth: source.bit_depth,
            planes: [
                vec![16; source.width * source.height],
                source.planes[1].clone(),
                source.planes[2].clone(),
                Vec::new(),
            ],
        };
        for y in 0..source.height - dy {
            for x in 0..source.width - dx {
                shifted.planes[0][(y + dy) * source.width + x + dx] =
                    source.planes[0][y * source.width + x];
            }
        }
        shifted
    }

    #[test]
    fn inter_product_rejects_444_before_encoding() {
        let mut enc = Av2VideoEncoder::new(Av2Encoder::new(120), ChromaFormat::Yuv444, 1);
        enc.set_key_interval(30);
        let err = enc
            .push_frame(&empty_frame(8), &Cicp::default())
            .unwrap_err();
        assert!(matches!(
            err,
            EncodeError::UnsupportedVideoMode {
                chroma: ChromaFormat::Yuv444,
                bit_depth: BitDepth::Eight,
            }
        ));
    }

    #[test]
    fn inter_product_rejects_420_12_bit_before_encoding() {
        let mut enc =
            Av2VideoEncoder::new(Av2Encoder::with_bit_depth(120, 12), ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        let err = enc
            .push_frame(&empty_frame(12), &Cicp::default())
            .unwrap_err();
        assert!(matches!(
            err,
            EncodeError::UnsupportedVideoMode {
                chroma: ChromaFormat::Yuv420,
                bit_depth: BitDepth::Twelve,
            }
        ));
    }

    #[test]
    fn video_enables_only_the_verified_single_tile_filter_set() {
        let cfg = Av2Encoder::new(120).with_deblock(true).with_cdef(true);
        let enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        assert!(enc.cfg.tuning().deblock);
        assert!(enc.cfg.tuning().cdef);
        assert!(!enc.cfg.tuning().ccso);
    }

    #[test]
    fn video_preserves_legacy_deblock_when_cdef_is_off() {
        let cfg = Av2Encoder::new(120).with_deblock(true).with_cdef(false);
        let enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        assert!(enc.cfg.tuning().deblock);
        assert!(!enc.cfg.tuning().cdef);
    }

    #[test]
    fn video_gates_multitile_cdef_until_tile_edges_are_verified() {
        let cfg = Av2Encoder::new(120).with_tiles(2, 1).with_cdef(true);
        let enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 2);
        assert!(!enc.cfg.tuning().cdef);
    }

    #[test]
    fn video_presets_have_monotonic_effort_and_honest_reference_count() {
        let presets = [
            VideoPreset::Realtime,
            VideoPreset::Fast,
            VideoPreset::Balanced,
            VideoPreset::Quality,
            VideoPreset::Slow,
            VideoPreset::Reference,
        ];
        let configs: Vec<_> = presets.into_iter().map(VideoPreset::config).collect();
        for pair in configs.array_windows::<2>() {
            assert!(pair[0].search_range <= pair[1].search_range);
            assert!(pair[0].lookahead_frames <= pair[1].lookahead_frames);
            assert!(pair[0].minimum_block_size >= pair[1].minimum_block_size);
            assert!(pair[0].predictor_gate_sad_per_pixel >= pair[1].predictor_gate_sad_per_pixel);
            assert!(pair[0].integer_satd_radius <= pair[1].integer_satd_radius);
        }
        assert_eq!(
            configs
                .iter()
                .map(|config| config.reference_count)
                .collect::<Vec<_>>(),
            [1, 1, 2, 2, 3, 3]
        );
    }

    #[test]
    fn setting_video_preset_applies_live_search_and_lookahead_controls() {
        let mut enc = Av2VideoEncoder::new(Av2Encoder::new(120), ChromaFormat::Yuv420, 8);
        enc.set_preset(VideoPreset::Realtime);
        let config = enc.preset_config();
        assert_eq!(enc.preset(), VideoPreset::Realtime);
        assert_eq!(enc.cfg.video_search_range, i32::from(config.search_range));
        assert_eq!(
            enc.cfg.video_predictor_gate,
            u32::from(config.predictor_gate_sad_per_pixel)
        );
        assert_eq!(enc.cfg.video_min_block_size, config.minimum_block_size);
        assert_eq!(
            enc.cfg.video_integer_satd_radius,
            config.integer_satd_radius
        );
        assert_eq!(
            enc.cfg.video_max_partition_depth,
            config.maximum_partition_depth
        );
        assert!(!enc.cfg.video_allows_16x16_partitions());
        assert_eq!(
            enc.state.rate_control.lookahead.capacity(),
            config.lookahead_frames as usize
        );
        assert_eq!(enc.threads(), 8);

        enc.set_preset(VideoPreset::Reference);
        assert_eq!(enc.threads(), 1);
        assert!(enc.cfg.video_allows_16x16_partitions());
        enc.set_preset(VideoPreset::Balanced);
        assert_eq!(enc.threads(), 8);
    }

    #[test]
    fn frame_cq_consumes_lookahead_and_stays_bounded() {
        let mut enc = Av2VideoEncoder::new(Av2Encoder::new(120), ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        enc.set_frame_cq(6);
        enc.push_frame(&checker_420(false), &Cicp::default())
            .unwrap();
        assert_eq!(enc.current_qindex(), 120);
        enc.push_frame(&checker_420(true), &Cicp::default())
            .unwrap();
        assert!((114..=126).contains(&enc.current_qindex()));
        assert_ne!(enc.current_qindex(), 120);
        enc.set_frame_cq(0);
        assert_eq!(enc.current_qindex(), 120);
    }

    #[test]
    fn frame_analysis_installs_hierarchical_mv_seed_for_block_search() {
        use std::sync::atomic::Ordering;

        crate::av2::y420::INTER_RESIDUAL_64_COUNT.store(0, Ordering::Relaxed);
        let mut enc = Av2VideoEncoder::new(Av2Encoder::new(120), ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        let first = enc
            .push_frame(&translated_pattern_420(0, 0), &Cicp::default())
            .unwrap();
        let second = enc
            .push_frame(&translated_pattern_420(4, 2), &Cicp::default())
            .unwrap();
        assert_eq!(
            *enc.cfg.video_mv_seed.lock().unwrap(),
            crate::av2::video::mv::Mv { row: 16, col: 32 }
        );
        assert!(
            crate::av2::y420::INTER_RESIDUAL_64_COUNT.load(Ordering::Relaxed) > 0,
            "translated frame did not exercise the 64x64 inter residual path"
        );

        let reference = enc.state.references.last().unwrap();
        let mut expected = Vec::new();
        for (plane, stride, width, height) in [
            (&reference.planes[0], reference.strides[0], 64usize, 64usize),
            (&reference.planes[1], reference.strides[1], 32usize, 32usize),
            (&reference.planes[2], reference.strides[2], 32usize, 32usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                expected.extend(row[..width].iter().map(|&value| value as u8));
            }
        }
        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return;
        }
        let mut stream = first.data;
        stream.extend(second.data);
        let tag = format!("mt-video-inter-residual-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, stream).unwrap();
        let result = std::process::Command::new(decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        let decoded = std::fs::read(&output).unwrap_or_default();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        assert!(
            result.status.success(),
            "avmdec rejected 64x64 inter residual: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            &decoded[expected.len()..2 * expected.len()],
            expected,
            "64x64 inter residual reconstruction differs from AVM"
        );
    }

    #[test]
    fn low_delay_installs_post_emit_f32_reference() {
        let mut enc = Av2VideoEncoder::new(Av2Encoder::new(120), ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        let key = enc.push_frame(&flat_420(80), &Cicp::default()).unwrap();
        let key_planes = std::sync::Arc::clone(&enc.state.references.last().unwrap().planes);
        let inter = enc.push_frame(&flat_420(82), &Cicp::default()).unwrap();
        assert!(key.key);
        assert!(!inter.key);
        assert_eq!((key.pts, inter.pts), (0, 1));
        let reference = enc.state.references.last().unwrap();
        assert_eq!(reference.order_hint, 1);
        assert_eq!(reference.strides, [64, 32, 32]);
        assert!(reference.planes[0].iter().all(|v: &f32| v.is_finite()));
        assert!(std::sync::Arc::ptr_eq(
            &key_planes,
            &enc.cfg.last_ref.lock().unwrap()
        ));
    }

    #[test]
    fn video_cdef_reference_is_byte_exact_vs_avmdec() {
        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return;
        }
        let image = diagonal_edge_420();
        let cfg = Av2Encoder::new(220)
            .with_deblock(true)
            .with_cdef(true)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        let first = enc.push_frame(&image, &Cicp::srgb_ycbcr()).unwrap();
        let first_reference = enc.state.references.last().unwrap().planes.clone();
        let second = enc.push_frame(&image, &Cicp::srgb_ycbcr()).unwrap();
        let mut stream = first.data;
        stream.extend(second.data);

        let reference = enc.state.references.last().unwrap();
        let mut expected = Vec::new();
        for (plane, stride, height) in [
            (&reference.planes[0], 128usize, 128usize),
            (&reference.planes[1], 64usize, 64usize),
            (&reference.planes[2], 64usize, 64usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                expected.extend(row.iter().map(|&value| value as u8));
            }
        }
        let tag = format!("mt-video-cdef-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, &stream).unwrap();
        let result = std::process::Command::new(&decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "avmdec failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let decoded = std::fs::read(&output).unwrap();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let mut first_expected = Vec::new();
        for (plane, stride, height) in [
            (&first_reference[0], 128usize, 128usize),
            (&first_reference[1], 64usize, 64usize),
            (&first_reference[2], 64usize, 64usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                first_expected.extend(row.iter().map(|&value| value as u8));
            }
        }
        assert_eq!(&decoded[..expected.len()], first_expected);
        assert_eq!(&decoded[expected.len()..2 * expected.len()], expected);
    }

    #[test]
    fn split_32_inter_skip_decodes_with_avmdec() {
        use std::sync::atomic::Ordering;

        crate::av2::y420::INTER_SKIP_32_COUNT.store(0, Ordering::Relaxed);
        let image = diagonal_edge_420();
        let cfg = Av2Encoder::new(30)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        let first = enc.push_frame(&image, &Cicp::srgb_ycbcr()).unwrap();
        let second = enc.push_frame(&image, &Cicp::srgb_ycbcr()).unwrap();
        let reference = enc.state.references.last().unwrap();
        let mut expected = Vec::new();
        for (plane, stride, height) in [
            (&reference.planes[0], 128usize, 128usize),
            (&reference.planes[1], 64usize, 64usize),
            (&reference.planes[2], 64usize, 64usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                expected.extend(row.iter().map(|&value| value as u8));
            }
        }
        assert!(
            crate::av2::y420::INTER_SKIP_32_COUNT.load(Ordering::Relaxed) > 0,
            "the split frame did not exercise 32x32 inter competition"
        );
        crate::av2::y420::INTER_NEWMV_SKIP_32_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_NEARMV_SKIP_32_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_NEWMV_SKIP_16_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_NEARMV_SKIP_16_COUNT.store(0, Ordering::Relaxed);
        let third = enc
            .push_frame(&translated_diagonal_420(4, 2), &Cicp::srgb_ycbcr())
            .unwrap();
        assert!(
            crate::av2::y420::INTER_NEWMV_SKIP_32_COUNT.load(Ordering::Relaxed) > 0,
            "the translated split frame did not exercise 32x32 NEWMV"
        );
        assert!(
            crate::av2::y420::INTER_NEARMV_SKIP_16_COUNT.load(Ordering::Relaxed) > 0,
            "adjacent 16x16 leaves did not reuse the spatial MV"
        );
        let motion_reference = enc.state.references.last().unwrap();
        let mut motion_expected = Vec::new();
        for (plane, stride, height) in [
            (&motion_reference.planes[0], 128usize, 128usize),
            (&motion_reference.planes[1], 64usize, 64usize),
            (&motion_reference.planes[2], 64usize, 64usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                motion_expected.extend(row.iter().map(|&value| value as u8));
            }
        }
        crate::av2::y420::INTER_RESIDUAL_32_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_RESIDUAL_32_HIGH_EOB_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_RESIDUAL_16_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_RESIDUAL_16_HIGH_EOB_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_RESIDUAL_16_CHROMA_COUNT.store(0, Ordering::Relaxed);
        let mut residual_image = translated_diagonal_420(8, 4);
        for sample in &mut residual_image.planes[0] {
            *sample = sample.saturating_add(6);
        }
        const CHROMA16_MI: [(usize, usize); 16] = [
            (8, 12),
            (12, 12),
            (0, 16),
            (4, 16),
            (0, 20),
            (4, 20),
            (8, 16),
            (12, 16),
            (8, 20),
            (12, 20),
            (24, 4),
            (28, 4),
            (24, 8),
            (28, 8),
            (24, 12),
            (28, 12),
        ];
        for plane in &mut residual_image.planes[1..3] {
            for &(mi_row, mi_col) in &CHROMA16_MI {
                let (y0, x0) = (mi_row * 2, mi_col * 2);
                for y in y0..y0 + 8 {
                    for sample in &mut plane[y * 64 + x0..][..8] {
                        *sample = sample.saturating_add(1);
                    }
                }
            }
        }
        let fourth = enc
            .push_frame(&residual_image, &Cicp::srgb_ycbcr())
            .unwrap();
        assert!(
            crate::av2::y420::INTER_RESIDUAL_32_COUNT.load(Ordering::Relaxed) > 0,
            "motion plus a level change did not exercise dense 32x32 inter residuals"
        );
        assert!(
            crate::av2::y420::INTER_RESIDUAL_32_HIGH_EOB_COUNT.load(Ordering::Relaxed) > 0,
            "dense 32x32 inter residuals did not exercise high-EOB AC coding"
        );
        assert!(
            crate::av2::y420::INTER_RESIDUAL_16_COUNT.load(Ordering::Relaxed) > 0,
            "motion plus a level change did not exercise dense 16x16 inter residuals"
        );
        assert!(
            crate::av2::y420::INTER_RESIDUAL_16_HIGH_EOB_COUNT.load(Ordering::Relaxed) > 0,
            "dense 16x16 inter residuals did not exercise high-EOB AC coding"
        );
        assert!(
            crate::av2::y420::INTER_RESIDUAL_16_CHROMA_COUNT.load(Ordering::Relaxed) > 0,
            "dense 16x16 inter residuals did not exercise chroma TX8 coding"
        );
        let residual_reference = enc.state.references.last().unwrap();
        let mut residual_expected = Vec::new();
        for (plane, stride, height) in [
            (&residual_reference.planes[0], 128usize, 128usize),
            (&residual_reference.planes[1], 64usize, 64usize),
            (&residual_reference.planes[2], 64usize, 64usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                residual_expected.extend(row.iter().map(|&value| value as u8));
            }
        }
        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return;
        }
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        stream.extend(fourth.data);
        let tag = format!("mt-video-inter32-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, stream).unwrap();
        let result = std::process::Command::new(decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        let decoded = std::fs::read(&output).unwrap_or_default();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        assert!(
            result.status.success(),
            "avmdec rejected 32x32 inter skip: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            &decoded[expected.len()..2 * expected.len()],
            expected,
            "encoder 32x32 inter reconstruction differs from AVM"
        );
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            motion_expected,
            "encoder 32x32 NEWMV reconstruction differs from AVM"
        );
        assert_eq!(
            &decoded[3 * expected.len()..4 * expected.len()],
            residual_expected,
            "encoder dense 32x32 inter reconstruction differs from AVM"
        );
    }

    #[test]
    fn rectangular_edge_inter_skip_is_byte_exact() {
        use std::sync::atomic::Ordering;

        crate::av2::y420::INTER_SKIP_RECT_COUNT.store(0, Ordering::Relaxed);
        crate::av2::y420::INTER_MOTION_SKIP_RECT_COUNT.store(0, Ordering::Relaxed);
        let image = rectangular_420();
        let translated = translated_rectangular_420(&image, 4, 0);
        let cfg = Av2Encoder::new(130)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_key_interval(30);
        let first = enc.push_frame(&image, &Cicp::srgb_ycbcr()).unwrap();
        let second = enc.push_frame(&image, &Cicp::srgb_ycbcr()).unwrap();
        assert!(
            crate::av2::y420::INTER_SKIP_RECT_COUNT.load(Ordering::Relaxed) > 0,
            "96x64 frame did not exercise a rectangular inter leaf"
        );

        let reference = enc.state.references.last().unwrap();
        let mut expected = Vec::new();
        for (plane, stride, coded_width, height) in [
            (&reference.planes[0], reference.strides[0], 96usize, 64usize),
            (&reference.planes[1], reference.strides[1], 48usize, 32usize),
            (&reference.planes[2], reference.strides[2], 48usize, 32usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                expected.extend(row[..coded_width].iter().map(|&value| value as u8));
            }
        }
        let third = enc.push_frame(&translated, &Cicp::srgb_ycbcr()).unwrap();
        assert!(
            crate::av2::y420::INTER_MOTION_SKIP_RECT_COUNT.load(Ordering::Relaxed) > 0,
            "translated 96x64 frame did not exercise rectangular motion competition"
        );
        let motion_reference = enc.state.references.last().unwrap();
        let mut motion_expected = Vec::new();
        for (plane, stride, coded_width, height) in [
            (
                &motion_reference.planes[0],
                motion_reference.strides[0],
                96usize,
                64usize,
            ),
            (
                &motion_reference.planes[1],
                motion_reference.strides[1],
                48usize,
                32usize,
            ),
            (
                &motion_reference.planes[2],
                motion_reference.strides[2],
                48usize,
                32usize,
            ),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                motion_expected.extend(row[..coded_width].iter().map(|&value| value as u8));
            }
        }
        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return;
        }
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let tag = format!("mt-video-inter-rect-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, stream).unwrap();
        let result = std::process::Command::new(decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        let decoded = std::fs::read(&output).unwrap_or_default();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        assert!(
            result.status.success(),
            "avmdec rejected rectangular inter leaf: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            &decoded[expected.len()..2 * expected.len()],
            expected,
            "rectangular inter reconstruction differs from AVM"
        );
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            motion_expected,
            "rectangular motion reconstruction differs from AVM"
        );
    }

    #[test]
    fn chroma_aware_last2_selection_is_byte_exact() {
        let first_image = diagonal_edge_420();
        let mut intervening = PlanarImage {
            width: first_image.width,
            height: first_image.height,
            bit_depth: first_image.bit_depth,
            planes: [
                first_image.planes[0].clone(),
                first_image.planes[1].clone(),
                first_image.planes[2].clone(),
                Vec::new(),
            ],
        };
        for plane in &mut intervening.planes[1..=2] {
            for sample in plane {
                *sample = 255 - *sample;
            }
        }

        let cfg = Av2Encoder::new(120)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Slow);
        enc.set_key_interval(30);
        let first = enc.push_frame(&first_image, &Cicp::srgb_ycbcr()).unwrap();
        let second = enc.push_frame(&intervening, &Cicp::srgb_ycbcr()).unwrap();
        let third = enc.push_frame(&first_image, &Cicp::srgb_ycbcr()).unwrap();
        assert_eq!(
            enc.state.selected_reference_slot,
            Some(0),
            "chroma-only returning content should select the retained LAST2 slot"
        );

        let reference = enc.state.references.last().unwrap();
        let mut expected = Vec::new();
        for (plane, stride, coded_width, height) in [
            (
                &reference.planes[0],
                reference.strides[0],
                first_image.width,
                first_image.height,
            ),
            (
                &reference.planes[1],
                reference.strides[1],
                first_image.width / 2,
                first_image.height / 2,
            ),
            (
                &reference.planes[2],
                reference.strides[2],
                first_image.width / 2,
                first_image.height / 2,
            ),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                expected.extend(row[..coded_width].iter().map(|&value| value as u8));
            }
        }

        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return;
        }
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let tag = format!("mt-video-last2-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, stream).unwrap();
        let result = std::process::Command::new(decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        let decoded = std::fs::read(&output).unwrap_or_default();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        assert!(
            result.status.success(),
            "avmdec rejected LAST2 sequence: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "LAST2 reconstruction differs from AVM"
        );
    }

    #[test]
    fn frame_level_last3_selection_is_byte_exact() {
        let mut first_image = flat_420(96);
        first_image.planes[1].fill(24);
        first_image.planes[2].fill(232);
        let mut second_image = flat_420(96);
        second_image.planes[1].fill(96);
        second_image.planes[2].fill(160);
        let mut third_image = flat_420(96);
        third_image.planes[1].fill(224);
        third_image.planes[2].fill(32);

        let cfg = Av2Encoder::new(100)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);
        let first = enc.push_frame(&first_image, &Cicp::srgb_ycbcr()).unwrap();
        let second = enc.push_frame(&second_image, &Cicp::srgb_ycbcr()).unwrap();
        let third = enc.push_frame(&third_image, &Cicp::srgb_ycbcr()).unwrap();
        let fourth = enc.push_frame(&first_image, &Cicp::srgb_ycbcr()).unwrap();
        assert_eq!(
            enc.state.selected_reference_slot,
            Some(0),
            "returning content should select the retained LAST3 slot"
        );

        let reference = enc.state.references.last().unwrap();
        let mut expected = Vec::new();
        for (plane, stride, coded_width, height) in [
            (&reference.planes[0], reference.strides[0], 64usize, 64usize),
            (&reference.planes[1], reference.strides[1], 32usize, 32usize),
            (&reference.planes[2], reference.strides[2], 32usize, 32usize),
        ] {
            for row in plane.chunks_exact(stride).take(height) {
                expected.extend(row[..coded_width].iter().map(|&value| value as u8));
            }
        }

        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return;
        }
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        stream.extend(fourth.data);
        let tag = format!("mt-video-last3-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, stream).unwrap();
        let result = std::process::Command::new(decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        let decoded = std::fs::read(&output).unwrap_or_default();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        assert!(
            result.status.success(),
            "avmdec rejected LAST3 sequence: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            &decoded[3 * expected.len()..4 * expected.len()],
            expected,
            "LAST3 reconstruction differs from AVM"
        );
    }

    fn noise_64(seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..64 * 64)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    /// 64x64 block translated down-right by (dx, dy); revealed edges fill 128.
    fn shifted_64(block: &[u8], dx: usize, dy: usize) -> Vec<u8> {
        let mut out = vec![128u8; 64 * 64];
        for y in 0..64 - dy {
            for x in 0..64 - dx {
                out[(y + dy) * 64 + (x + dx)] = block[y * 64 + x];
            }
        }
        out
    }

    /// 128x64 4:2:0 image from two 64x64 luma blocks; flat chroma.
    fn two_sb_420(left: &[u8], right: &[u8]) -> PlanarImage<u8> {
        let (w, h) = (128usize, 64usize);
        let mut luma = vec![0u8; w * h];
        for row in 0..h {
            luma[row * w..row * w + 64].copy_from_slice(&left[row * 64..row * 64 + 64]);
            luma[row * w + 64..row * w + 128].copy_from_slice(&right[row * 64..row * 64 + 64]);
        }
        PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [
                luma,
                vec![96; (w / 2) * (h / 2)],
                vec![160; (w / 2) * (h / 2)],
                Vec::new(),
            ],
        }
    }

    fn decode_with_avmdec(stream: Vec<u8>, tag: &str) -> Option<Vec<u8>> {
        let decoder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("avmdec");
        if !decoder.is_file() {
            return None;
        }
        let tag = format!("{tag}-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{tag}.obu"));
        let output = std::env::temp_dir().join(format!("{tag}.yuv"));
        std::fs::write(&input, stream).unwrap();
        let result = std::process::Command::new(decoder)
            .arg("--codec=av2")
            .arg("--rawvideo")
            .arg("--output-bit-depth=8")
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        let decoded = std::fs::read(&output).unwrap_or_default();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        assert!(
            result.status.success(),
            "avmdec rejected stream: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        Some(decoded)
    }

    fn last_recon_bytes(enc: &Av2VideoEncoder) -> Vec<u8> {
        let reference = enc.state.references.last().unwrap();
        let mut bytes = Vec::new();
        for (plane, stride, plane_width, plane_height) in [
            (
                &reference.planes[0],
                reference.strides[0],
                128usize,
                64usize,
            ),
            (&reference.planes[1], reference.strides[1], 64usize, 32usize),
            (&reference.planes[2], reference.strides[2], 64usize, 32usize),
        ] {
            for row in plane.chunks_exact(stride).take(plane_height) {
                bytes.extend(row[..plane_width].iter().map(|&value| value as u8));
            }
        }
        bytes
    }

    /// Two static regions that each match a DIFFERENT reference frame: the
    /// left SB tracks the previous frame, the right SB returns to the key
    /// frame's content. Both zero-motion, so per-SB GLOBALMV skip must pick a
    /// different reference rank per SB and the stream must still match AVM.
    #[test]
    fn per_sb_skip_reference_rank_selection_is_byte_exact() {
        let left_key = noise_64(0x11);
        let left_new = noise_64(0x22);
        let right_key = noise_64(0x33);
        let right_mid = noise_64(0x44);

        let cfg = Av2Encoder::new(100)
            .with_deblock(true)
            .with_cdef(false)
            .with_chroma_split(false) // whole-64 fast path (rank selection lives there)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);

        let first = enc
            .push_frame(&two_sb_420(&left_key, &right_key), &Cicp::srgb_ycbcr())
            .unwrap();
        let second = enc
            .push_frame(&two_sb_420(&left_new, &right_mid), &Cicp::srgb_ycbcr())
            .unwrap();
        let rank1_before = crate::av2::y420::core_skip_rank1_count();
        let third = enc
            .push_frame(&two_sb_420(&left_new, &right_key), &Cicp::srgb_ycbcr())
            .unwrap();
        assert!(
            crate::av2::y420::core_skip_rank1_count() > rank1_before,
            "third frame should skip at least one SB from the rank-1 reference"
        );

        let expected = last_recon_bytes(&enc);
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-per-sb-ref-rank") else {
            return;
        };
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "per-SB rank-selected reconstruction differs from AVM"
        );
    }

    /// Two moving regions that each track a DIFFERENT reference: the left SB
    /// continues the previous frame's pattern, the right SB continues the key
    /// frame's. The right SB's DRL[0] must then come from the cross-rank
    /// derived (distance-scaled) candidate — the decoder derives it
    /// independently, so any mismatch breaks byte-exactness.
    #[test]
    fn cross_rank_derived_mv_prediction_is_byte_exact() {
        let left_key = noise_64(0x51);
        let left_mid = noise_64(0x61); // new content after the key frame
        let right_key = noise_64(0x77);
        let right_mid = noise_64(0x99);

        let cfg = Av2Encoder::new(100)
            .with_deblock(true)
            .with_cdef(false)
            .with_chroma_split(false) // whole-64 fast path (rank selection lives there)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);
        // NEWMV multi-ref win is motion-based; the (zero-motion) production gate does
        // not detect it, so force the second reference to exercise the derived-MV path.
        enc.set_second_reference_threshold(0.0);

        let first = enc
            .push_frame(&two_sb_420(&left_key, &right_key), &Cicp::srgb_ycbcr())
            .unwrap();
        let second = enc
            .push_frame(&two_sb_420(&left_mid, &right_mid), &Cicp::srgb_ycbcr())
            .unwrap();
        // Left continues frame 2's content (translated); right returns to the
        // key frame's content (translated) — different reference per SB, both
        // with nonzero motion.
        let rank1_before = crate::av2::y420::core_newmv_rank1_count();
        let third = enc
            .push_frame(
                &two_sb_420(&shifted_64(&left_mid, 4, 0), &shifted_64(&right_key, 2, 2)),
                &Cicp::srgb_ycbcr(),
            )
            .unwrap();
        assert!(
            crate::av2::y420::core_newmv_rank1_count() > rank1_before,
            "third frame should code at least one NEWMV SB on the rank-1 reference"
        );

        let expected = last_recon_bytes(&enc);
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-derived-mv") else {
            return;
        };
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "cross-rank derived-MV reconstruction differs from AVM"
        );
    }

    /// Same divergent-reference occlusion scenario as the core-path test, but
    /// through the PARTITION walk (chroma_split, the default). Flat per-SB blocks
    /// keep each superblock a whole-64 leaf so the rank-selecting GLOBALMV-skip
    /// branch fires; the right SB reverts to the key frame (rank 1), the left SB
    /// tracks the previous frame (rank 0). Proves the partition-walk rank bit and
    /// the same-rank mode contexts stay byte-exact with AVM.
    #[test]
    fn partition_walk_per_sb_skip_rank_is_byte_exact() {
        let solid = |v: u8| vec![v; 64 * 64];
        let cfg = Av2Encoder::new(100)
            .with_chroma_split(true) // route through the partition walk (default)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);

        let first = enc
            .push_frame(&two_sb_420(&solid(90), &solid(150)), &Cicp::srgb_ycbcr())
            .unwrap();
        let second = enc
            .push_frame(&two_sb_420(&solid(200), &solid(60)), &Cicp::srgb_ycbcr())
            .unwrap();
        let rank1_before = crate::av2::y420::partition_skip_rank1_count();
        // Left SB stays at frame 2's value (rank 0); right SB returns to the key
        // frame's value (rank 1) — different reference per SB, both zero motion.
        let third = enc
            .push_frame(&two_sb_420(&solid(200), &solid(150)), &Cicp::srgb_ycbcr())
            .unwrap();
        assert!(
            crate::av2::y420::partition_skip_rank1_count() > rank1_before,
            "the partition walk did not commit a rank-1 GLOBALMV skip"
        );

        let expected = last_recon_bytes(&enc);
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-partition-rank") else {
            return;
        };
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "partition-walk rank-selected reconstruction differs from AVM"
        );
    }

    /// Build a `n`-superblock-wide (n*64 x 64) 4:2:0 image from per-SB solid luma
    /// values; flat chroma. Each SB stays a whole-64 leaf.
    fn n_sb_420(vals: &[u8]) -> PlanarImage<u8> {
        let n = vals.len();
        let (w, h) = (n * 64, 64usize);
        let mut luma = vec![0u8; w * h];
        for (i, &v) in vals.iter().enumerate() {
            for row in 0..h {
                luma[row * w + i * 64..row * w + i * 64 + 64].fill(v);
            }
        }
        PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [
                luma,
                vec![96; (w / 2) * (h / 2)],
                vec![160; (w / 2) * (h / 2)],
                Vec::new(),
            ],
        }
    }

    /// Build a `cols`x`rows`-superblock grid ((cols*64)x(rows*64)) 4:2:0 image
    /// from a row-major grid of per-SB solid luma values; flat chroma.
    fn grid_sb_420(cols: usize, rows: usize, vals: &[u8]) -> PlanarImage<u8> {
        let (w, h) = (cols * 64, rows * 64);
        let mut luma = vec![0u8; w * h];
        for r in 0..rows {
            for c in 0..cols {
                let v = vals[r * cols + c];
                for y in 0..64 {
                    let base = (r * 64 + y) * w + c * 64;
                    luma[base..base + 64].fill(v);
                }
            }
        }
        PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [
                luma,
                vec![96; (w / 2) * (h / 2)],
                vec![160; (w / 2) * (h / 2)],
                Vec::new(),
            ],
        }
    }

    /// `full`*64 + `edge`-px wide, 64 tall, per-region solid luma. The final
    /// region is a partial (non-64) edge column, so the frame is not 64-aligned.
    fn edge_row_420(full_vals: &[u8], edge_val: u8, edge: usize) -> PlanarImage<u8> {
        let full = full_vals.len();
        let w = full * 64 + edge;
        let h = 64usize;
        let mut luma = vec![0u8; w * h];
        for y in 0..h {
            for (c, &v) in full_vals.iter().enumerate() {
                luma[y * w + c * 64..y * w + c * 64 + 64].fill(v);
            }
            luma[y * w + full * 64..y * w + w].fill(edge_val);
        }
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [luma, vec![96; cw * ch], vec![160; cw * ch], Vec::new()],
        }
    }

    /// NON-64-aligned reproducer (480x64 = 7 full SBs + 32px right edge). Interior
    /// SBs 4-6 revert to the key (rank 1) beside the rank-0 edge column. Isolates
    /// the multi-ref edge desync (aligned crops of the same shape are byte-exact).
    #[test]
    fn partition_edge_rank1_is_byte_exact() {
        let cfg = Av2Encoder::new(100)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);
        // key ramp of solids + edge; frame2 all-different; frame3 keeps cols 0-3 +
        // edge at frame2 (rank 0), cols 4-6 back to key (rank 1).
        let key = [40u8, 90, 150, 210, 60, 120, 180];
        let f2 = [200u8, 30, 220, 80, 240, 50, 160];
        let f3 = [200u8, 30, 220, 80, 60, 120, 180];
        let first = enc
            .push_frame(&edge_row_420(&key, 100, 32), &Cicp::srgb_ycbcr())
            .unwrap();
        let second = enc
            .push_frame(&edge_row_420(&f2, 130, 32), &Cicp::srgb_ycbcr())
            .unwrap();
        let rank1_before = crate::av2::y420::partition_skip_rank1_count();
        let third = enc
            .push_frame(&edge_row_420(&f3, 130, 32), &Cicp::srgb_ycbcr())
            .unwrap();
        let saw_rank1 = crate::av2::y420::partition_skip_rank1_count() > rank1_before;
        let (w, h) = (480usize, 64usize);
        let expected = {
            let reference = enc.state.references.last().unwrap();
            let mut bytes = Vec::new();
            for (plane, stride, pw, ph) in [
                (&reference.planes[0], reference.strides[0], w, h),
                (
                    &reference.planes[1],
                    reference.strides[1],
                    w.div_ceil(2),
                    h / 2,
                ),
                (
                    &reference.planes[2],
                    reference.strides[2],
                    w.div_ceil(2),
                    h / 2,
                ),
            ] {
                for row in plane.chunks_exact(stride).take(ph) {
                    bytes.extend(row[..pw].iter().map(|&v| v as u8));
                }
            }
            bytes
        };
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-edge-rank1") else {
            return;
        };
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "edge rank-1 reconstruction differs from AVM (saw_rank1={saw_rank1})"
        );
    }

    /// Rank-1 skip with a rank-1 ABOVE neighbor (multi-row), which the real clip
    /// hits and the single-row test does not: a 2-row-tall column reverts to the
    /// key (rank 1) while the rest tracks frame 2 (rank 0). Must stay byte-exact.
    #[test]
    fn partition_rank1_above_neighbor_is_byte_exact() {
        let cfg = Av2Encoder::new(100)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);
        // 3 cols x 2 rows. Frame 3: column 2 (both rows) returns to the key
        // (rank 1); columns 0-1 track frame 2 (rank 0). So SB (1,2) sits below the
        // rank-1 SB (0,2).
        let key = [40u8, 90, 150, 100, 170, 210];
        let f2 = [200u8, 30, 120, 60, 220, 80];
        let f3 = [200u8, 30, 150, 60, 220, 210]; // col 2 = key, else frame 2
        let first = enc
            .push_frame(&grid_sb_420(3, 2, &key), &Cicp::srgb_ycbcr())
            .unwrap();
        let second = enc
            .push_frame(&grid_sb_420(3, 2, &f2), &Cicp::srgb_ycbcr())
            .unwrap();
        let rank1_before = crate::av2::y420::partition_skip_rank1_count();
        let third = enc
            .push_frame(&grid_sb_420(3, 2, &f3), &Cicp::srgb_ycbcr())
            .unwrap();
        assert!(
            crate::av2::y420::partition_skip_rank1_count() >= rank1_before + 2,
            "expected rank-1 skips stacked across two rows"
        );
        let (w, h) = (192usize, 128usize);
        let expected = {
            let reference = enc.state.references.last().unwrap();
            let mut bytes = Vec::new();
            for (plane, stride, pw, ph) in [
                (&reference.planes[0], reference.strides[0], w, h),
                (&reference.planes[1], reference.strides[1], w / 2, h / 2),
                (&reference.planes[2], reference.strides[2], w / 2, h / 2),
            ] {
                for row in plane.chunks_exact(stride).take(ph) {
                    bytes.extend(row[..pw].iter().map(|&v| v as u8));
                }
            }
            bytes
        };
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-rank1-above") else {
            return;
        };
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "rank-1 above-neighbor reconstruction differs from AVM"
        );
    }

    /// Reproducer for dense contiguous rank-1 skips (what real video hits and the
    /// isolated single-SB test does not): SBs 1..=3 all revert to the key frame
    /// (rank 1) while SB 0 stays at frame 2 (rank 0), so a whole row of rank-1
    /// skips sits beside a rank-0 block. Must stay byte-exact with AVM.
    #[test]
    fn partition_contiguous_rank1_skips_are_byte_exact() {
        let cfg = Av2Encoder::new(100)
            .with_chroma_split(true)
            .with_deblock(true)
            .with_cdef(false)
            .with_aq(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(30);
        // Frame 2 changes every SB; frame 3 keeps SBs 0..=3 at frame-2 values
        // (rank 0, the better overall reference) and returns SBs 4..=5 to the key
        // (rank 1) — a contiguous pair of rank-1 skips beside rank-0 blocks.
        let first = enc
            .push_frame(
                &n_sb_420(&[40, 90, 150, 210, 100, 170]),
                &Cicp::srgb_ycbcr(),
            )
            .unwrap();
        let second = enc
            .push_frame(&n_sb_420(&[200, 30, 120, 60, 220, 80]), &Cicp::srgb_ycbcr())
            .unwrap();
        let rank1_before = crate::av2::y420::partition_skip_rank1_count();
        let third = enc
            .push_frame(
                &n_sb_420(&[200, 30, 120, 60, 100, 170]),
                &Cicp::srgb_ycbcr(),
            )
            .unwrap();
        assert!(
            crate::av2::y420::partition_skip_rank1_count() >= rank1_before + 2,
            "expected at least 2 contiguous rank-1 skips"
        );
        let expected = {
            let reference = enc.state.references.last().unwrap();
            let mut bytes = Vec::new();
            for (plane, stride, pw, ph) in [
                (
                    &reference.planes[0],
                    reference.strides[0],
                    384usize,
                    64usize,
                ),
                (
                    &reference.planes[1],
                    reference.strides[1],
                    192usize,
                    32usize,
                ),
                (
                    &reference.planes[2],
                    reference.strides[2],
                    192usize,
                    32usize,
                ),
            ] {
                for r in plane.chunks_exact(stride).take(ph) {
                    bytes.extend(r[..pw].iter().map(|&v| v as u8));
                }
            }
            bytes
        };
        let mut stream = first.data;
        stream.extend(second.data);
        stream.extend(third.data);
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-contig-rank1") else {
            return;
        };
        assert_eq!(
            &decoded[2 * expected.len()..3 * expected.len()],
            expected,
            "contiguous rank-1 reconstruction differs from AVM"
        );
    }

    /// Minimal Y4M (8-bit 4:2:0) parser: returns (width, height, first `max`
    /// frames). None if the fixture is absent.
    fn read_y4m_frames(
        path: &std::path::Path,
        max: usize,
    ) -> Option<(usize, usize, Vec<PlanarImage<u8>>)> {
        let bytes = std::fs::read(path).ok()?;
        let nl = bytes.iter().position(|&b| b == b'\n')?;
        let header = std::str::from_utf8(&bytes[..nl]).ok()?;
        let (mut w, mut h) = (0usize, 0usize);
        for tok in header.split(' ') {
            match tok.as_bytes().first() {
                Some(b'W') => w = tok[1..].parse().ok()?,
                Some(b'H') => h = tok[1..].parse().ok()?,
                _ => {}
            }
        }
        if w == 0 || h == 0 {
            return None;
        }
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (ysz, csz) = (w * h, cw * ch);
        let mut pos = nl + 1;
        let mut frames = Vec::new();
        while frames.len() < max && pos < bytes.len() {
            // Each frame is "FRAME\n" then raw planes.
            let fnl = bytes[pos..].iter().position(|&b| b == b'\n')? + pos;
            if !bytes[pos..fnl].starts_with(b"FRAME") {
                break;
            }
            pos = fnl + 1;
            if pos + ysz + 2 * csz > bytes.len() {
                break;
            }
            let y = bytes[pos..pos + ysz].to_vec();
            let u = bytes[pos + ysz..pos + ysz + csz].to_vec();
            let v = bytes[pos + ysz + csz..pos + ysz + 2 * csz].to_vec();
            pos += ysz + 2 * csz;
            frames.push(PlanarImage {
                width: w,
                height: h,
                bit_depth: BitDepth::Eight,
                planes: [y, u, v, Vec::new()],
            });
        }
        Some((w, h, frames))
    }

    // Diagnostic: inter-vs-intra leaf mode mix on a high-quality inter frame.
    #[test]
    #[ignore]
    fn diagnose_inter_mode_mix() {
        use std::sync::atomic::Ordering::Relaxed;
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/file_example_MP4_480_1_5MG.y4m");
        let Some((_w, _h, frames)) = read_y4m_frames(&fixture, 8) else {
            return;
        };
        // Crop to 64-aligned 256x256 to test whether the constant intra blocks are
        // the partial edge SBs (which the whole-64-only inter residual can't code).
        let aligned: Vec<PlanarImage<u8>> = frames
            .iter()
            .map(|f| {
                let crop = |p: &[u8], stride: usize, w: usize, h: usize| {
                    let mut o = vec![0u8; w * h];
                    for y in 0..h {
                        o[y * w..y * w + w].copy_from_slice(&p[y * stride..y * stride + w]);
                    }
                    o
                };
                PlanarImage {
                    width: 256,
                    height: 256,
                    bit_depth: BitDepth::Eight,
                    planes: [
                        crop(&f.planes[0], 480, 256, 256),
                        crop(&f.planes[1], 240, 128, 128),
                        crop(&f.planes[2], 240, 128, 128),
                        Vec::new(),
                    ],
                }
            })
            .collect();
        for (label, fset) in [("full480x270", &frames), ("crop256x256", &aligned)] {
            let q = 60u8;
            let cfg = Av2Encoder::new(q).with_deblock(true).with_cdef(false);
            let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
            enc.set_preset(VideoPreset::Reference);
            enc.set_key_interval(30);
            for (i, frame) in fset.iter().enumerate().take(5) {
                crate::av2::y420::TOTAL_LEAF_COUNT.store(0, Relaxed);
                crate::av2::y420::INTRA_LEAF_COUNT.store(0, Relaxed);
                let pkt = enc.push_frame(frame, &Cicp::srgb_ycbcr()).unwrap();
                if i == 0 {
                    continue;
                }
                eprintln!(
                    "EDGE {label} frame{i} bytes={} leaves={} intra={}",
                    pkt.data.len(),
                    crate::av2::y420::TOTAL_LEAF_COUNT.load(Relaxed),
                    crate::av2::y420::INTRA_LEAF_COUNT.load(Relaxed),
                );
            }
        }
        for q in [60u8, 120] {
            let cfg = Av2Encoder::new(q).with_deblock(true).with_cdef(false);
            let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
            enc.set_preset(VideoPreset::Reference);
            enc.set_key_interval(30);
            for (i, frame) in frames.iter().enumerate() {
                crate::av2::y420::TOTAL_LEAF_COUNT.store(0, Relaxed);
                crate::av2::y420::INTRA_LEAF_COUNT.store(0, Relaxed);
                crate::av2::y420::INTER_RESIDUAL_64_COUNT.store(0, Relaxed);
                let pkt = enc.push_frame(frame, &Cicp::srgb_ycbcr()).unwrap();
                if i == 0 {
                    continue;
                }
                let total = crate::av2::y420::TOTAL_LEAF_COUNT.load(Relaxed);
                let intra = crate::av2::y420::INTRA_LEAF_COUNT.load(Relaxed);
                eprintln!(
                    "MIX q={q} frame{i} bytes={} leaves={total} intra={intra} res64={}",
                    pkt.data.len(),
                    crate::av2::y420::INTER_RESIDUAL_64_COUNT.load(Relaxed),
                );
            }
        }
    }

    // Benchmark harness hook: encode the real clip to an IVF file at MT_ENC_Q,
    // MT_ENC_FRAMES frames, MT_ENC_KI key interval, written to MT_ENC_OUT.
    // Drives external comparisons (x264/x265/VMAF). `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn encode_ivf_for_benchmark() {
        let (Ok(q), Ok(out)) = (std::env::var("MT_ENC_Q"), std::env::var("MT_ENC_OUT")) else {
            return;
        };
        let q: u8 = q.parse().unwrap();
        let n: usize = std::env::var("MT_ENC_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(192);
        let ki: u64 = std::env::var("MT_ENC_KI")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/file_example_MP4_480_1_5MG.y4m");
        let Some((_w, _h, frames)) = read_y4m_frames(&fixture, n) else {
            return;
        };
        let cfg = Av2Encoder::new(q).with_deblock(true).with_cdef(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(ki);
        let mut stream = Vec::new();
        for img in &frames {
            stream.extend(enc.push_frame(img, &Cicp::srgb_ycbcr()).unwrap().data);
        }
        eprintln!(
            "BENCH q={q} ki={ki} frames={} bytes={}",
            frames.len(),
            stream.len()
        );
        std::fs::write(out, stream).unwrap();
    }

    // Ad-hoc multi-ref compression-win measurement on the real clip; prints sizes
    // Diagnostic: per-frame byte breakdown on the real clip. `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn diagnose_frame_sizes() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/file_example_MP4_480_1_5MG.y4m");
        let Some((_w, _h, frames)) = read_y4m_frames(&fixture, 48) else {
            return;
        };
        // Compare all-intra (what the app does with key_interval=0) vs inter frames.
        let encode_total = |ki: u64| -> usize {
            let cfg = Av2Encoder::new(120).with_deblock(true).with_cdef(false);
            let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
            enc.set_preset(VideoPreset::Reference);
            enc.set_key_interval(ki);
            frames
                .iter()
                .map(|f| enc.push_frame(f, &Cicp::srgb_ycbcr()).unwrap().data.len())
                .sum()
        };
        let all_intra = encode_total(0);
        let with_inter = encode_total(48);
        eprintln!(
            "DIAG all-intra(ki=0)={}B  with-inter(ki=48)={}B  → {:.1}x smaller with P-frames",
            all_intra,
            with_inter,
            all_intra as f64 / with_inter.max(1) as f64,
        );

        let cfg = Av2Encoder::new(120).with_deblock(true).with_cdef(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(48); // one key, rest inter
        let mut key = 0usize;
        let mut inter: Vec<usize> = Vec::new();
        for (i, img) in frames.iter().enumerate() {
            let pkt = enc.push_frame(img, &Cicp::srgb_ycbcr()).unwrap();
            if i == 0 {
                key = pkt.data.len();
            } else {
                inter.push(pkt.data.len());
            }
        }
        let total: usize = key + inter.iter().sum::<usize>();
        let mean_inter = inter.iter().sum::<usize>() as f64 / inter.len().max(1) as f64;
        let min_inter = *inter.iter().min().unwrap_or(&0);
        let max_inter = *inter.iter().max().unwrap_or(&0);
        eprintln!(
            "DIAG {} frames: KEY={}B  inter mean={:.0}B min={}B max={}B  total={}B ({:.0}B/frame)",
            frames.len(),
            key,
            mean_inter,
            min_inter,
            max_inter,
            total,
            total as f64 / frames.len() as f64,
        );
        // Source temporal change: mean per-pixel luma SAD frame N vs N-1, and N vs 0.
        let luma = |f: &PlanarImage<u8>| f.planes[0].clone();
        let sad = |a: &[u8], b: &[u8]| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
                .sum::<u32>() as f64
                / a.len() as f64
        };
        let consec: Vec<f64> = (1..frames.len())
            .map(|i| sad(&luma(&frames[i]), &luma(&frames[i - 1])))
            .collect();
        let vs0: Vec<f64> = (1..frames.len().min(6))
            .map(|i| sad(&luma(&frames[i]), &luma(&frames[0])))
            .collect();
        let mean_consec = consec.iter().sum::<f64>() / consec.len().max(1) as f64;
        eprintln!(
            "DIAG source motion: mean consecutive luma SAD/px={:.2} (max={:.2})  vs-frame0 SAD/px (frames 1..5)={:?}",
            mean_consec,
            consec.iter().cloned().fold(0.0, f64::max),
            vs0.iter()
                .map(|v| (v * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
        );
    }

    // (ignored so it doesn't run in the normal suite — `cargo test -- --ignored`).
    #[test]
    #[ignore]
    fn measure_multiref_compression_win() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/file_example_MP4_480_1_5MG.y4m");
        let Some((_w, _h, frames)) = read_y4m_frames(&fixture, 16) else {
            return;
        };
        let encode = |limit: usize| -> usize {
            let cfg = Av2Encoder::new(120).with_deblock(true).with_cdef(false);
            let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
            enc.set_preset(VideoPreset::Reference);
            enc.set_key_interval(4);
            enc.set_reference_limit(limit);
            frames
                .iter()
                .map(|img| enc.push_frame(img, &Cicp::srgb_ycbcr()).unwrap().data.len())
                .sum()
        };
        let single = encode(1);
        let multi = encode(3);
        eprintln!(
            "multi-ref win: single={single}B multi={multi}B delta={:.2}%",
            100.0 * (single as f64 - multi as f64) / single as f64
        );
    }

    /// End-to-end real-video validation: encode the first frames of a real 4:2:0
    /// clip UNTILED (so multi-reference is active) with a short key interval, then
    /// decode the whole stream with avmdec. Exercises multi-ref ref-bit emission
    /// across real content — edges (480x270 is non-64-aligned), every partition
    /// size, deblock, and genuine motion — which the synthetic 2-3 frame tests
    /// cannot. Skips cleanly when the fixture or decoder is absent.
    #[test]
    fn real_video_multiref_stream_decodes_with_avmdec() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/file_example_MP4_480_1_5MG.y4m");
        let Some((w, h, frames)) = read_y4m_frames(&fixture, 16) else {
            return;
        };
        if frames.len() < 12 {
            return;
        }
        assert!(
            w % 64 != 0 || h % 64 != 0,
            "fixture must exercise native partial-edge superblocks"
        );
        let cfg = Av2Encoder::new(120).with_deblock(true).with_cdef(false);
        let mut enc = Av2VideoEncoder::new(cfg, ChromaFormat::Yuv420, 1);
        enc.set_preset(VideoPreset::Reference);
        enc.set_key_interval(4); // frequent keys → DPB fills → num_refs=2 on later inters
        // This clip is smooth, so the production gate would (correctly) not list a
        // second reference. Force it on to validate the non-aligned edge multi-ref
        // paths stay byte-exact on real content.
        enc.set_second_reference_threshold(0.0);
        let mut stream = Vec::new();
        let mut saw_two_refs = false;
        let rank1_before = crate::av2::y420::partition_skip_rank1_count();
        for img in &frames {
            let pkt = enc.push_frame(img, &Cicp::srgb_ycbcr()).unwrap();
            stream.extend(pkt.data);
            if enc.state.references.slot(1).is_some() {
                saw_two_refs = true;
            }
        }
        assert!(
            saw_two_refs,
            "the clip never populated a second DPB slot; multi-ref path not exercised"
        );
        assert!(
            crate::av2::y420::partition_skip_rank1_count() > rank1_before,
            "the non-aligned clip never committed a rank-1 edge-path skip"
        );
        let Some(decoded) = decode_with_avmdec(stream, "mt-video-real-multiref") else {
            return;
        };
        // avmdec asserts success (no corrupt frame) inside decode_with_avmdec;
        // require the full frame set decoded.
        let frame_bytes = w * h + 2 * (w.div_ceil(2) * h.div_ceil(2));
        assert!(
            decoded.len() >= frame_bytes * frames.len(),
            "avmdec produced fewer frames than encoded: {} < {}",
            decoded.len(),
            frame_bytes * frames.len()
        );
    }
}
