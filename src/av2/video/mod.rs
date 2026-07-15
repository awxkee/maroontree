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
        }
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
        let reference_limit = self.preset.config().reference_count as usize;
        let latest_slot = self.state.references.latest_slot();
        let mut reference_choice = None;
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
                    let rank = (score, usize::from(slot != latest_slot));
                    if reference_choice
                        .as_ref()
                        .is_none_or(|(best_rank, _, _)| rank < *best_rank)
                    {
                        reference_choice = Some((rank, slot, seed));
                    }
                }
            }
        }
        let reference_slot = reference_choice.map(|(_, slot, _)| slot);
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
        } else {
            *self.cfg.last_ref.lock().unwrap() = std::sync::Arc::new(Vec::new());
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
        for pair in configs.windows(2) {
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
}
