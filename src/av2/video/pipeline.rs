/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
 */

//! Video-specific frame control.  Still-image coding remains an implementation
//! adapter for prediction/transform/quantization/entropy primitives, but it no
//! longer owns the temporal state or reference-installation order.

use super::dpb::{Dpb, RefFrame};
use super::frame;
use super::gop::{FrameType, mean_luma_sad};
use super::mv::Mv;
use super::ratectrl::{FrameComplexity, Lookahead, spatial_activity};
use crate::av2::Av2Frame;
use crate::av2::headers::{Config, obu};
use crate::av2::video::headers::sequence_header_video;
use crate::{Pixel, PlanarImage};

/// Cross-frame state owned exclusively by the video controller.
pub(crate) struct VideoFrameState {
    pub(crate) references: Dpb<f32>,
    pub(crate) frame_type: FrameType,
    pub(crate) order_hint: u64,
    pub(crate) selected_reference_slot: Option<usize>,
    pub(crate) cdf: VideoCdfState,
    pub(crate) rate_control: VideoRateControlState,
}

impl Default for VideoFrameState {
    fn default() -> Self {
        Self {
            references: Dpb::default(),
            frame_type: FrameType::Key,
            order_hint: 0,
            selected_reference_slot: None,
            cdf: VideoCdfState::default(),
            rate_control: VideoRateControlState::new(32),
        }
    }
}

/// The cross-frame CDF ownership contract.  The current low-delay syntax resets
/// at every frame; keeping that decision here prevents entropy state from being
/// hidden in the still-image adapter when primary-reference continuity lands.
#[derive(Default)]
pub(crate) struct VideoCdfState {
    pub(crate) epoch: u64,
    pub(crate) primary_reference: Option<usize>,
}

impl VideoCdfState {
    pub(crate) fn begin_frame(&mut self, frame_type: FrameType) {
        if matches!(frame_type, FrameType::Key) {
            self.epoch = self.epoch.wrapping_add(1);
        }
        // The emitted low-delay header currently signals PRIMARY_REF_NONE.
        self.primary_reference = None;
    }
}

pub(crate) struct VideoRateControlState {
    pub(crate) lookahead: Lookahead,
}

impl VideoRateControlState {
    fn new(capacity: usize) -> Self {
        Self {
            lookahead: Lookahead::new(capacity),
        }
    }

    pub(crate) fn reset_lookahead(&mut self, capacity: usize) {
        self.lookahead = Lookahead::new(capacity.max(1));
    }
}

#[derive(Clone)]
pub(crate) struct PyramidLevel {
    pub(crate) samples: Vec<f32>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Read-only measurements used by frame/GOP/rate decisions.
pub(crate) struct FrameAnalysis {
    pub(crate) luma: Vec<f32>,
    pub(crate) complexity: FrameComplexity,
    pub(crate) scene_score: f32,
    pub(crate) lookahead_mean: FrameComplexity,
    pub(crate) motion_pyramid: Vec<PyramidLevel>,
}

impl FrameAnalysis {
    pub(crate) fn analyze<T: Pixel>(
        img: &PlanarImage<T>,
        previous_luma: &[f32],
        lookahead: &Lookahead,
    ) -> Self {
        let bd = img.bit_depth.bits();
        let scale = 1.0 / (1u32 << bd.saturating_sub(8)) as f32;
        let luma: Vec<f32> = img.planes[0].iter().map(|&p| p.to_f32() * scale).collect();
        let luma8: Vec<u8> = luma.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
        let previous_luma8: Vec<u8> = previous_luma
            .iter()
            .map(|&v| v.clamp(0.0, 255.0) as u8)
            .collect();
        let temporal_sad = mean_luma_sad(&luma8, &previous_luma8) as f32;
        let complexity = FrameComplexity {
            spatial_activity: spatial_activity(&luma8, img.width, img.height),
            temporal_sad,
        };
        let motion_pyramid = build_pyramid(&luma, img.width, img.height, 3);
        Self {
            luma,
            complexity,
            scene_score: temporal_sad,
            lookahead_mean: FrameComplexity {
                spatial_activity: lookahead.mean_spatial_activity(),
                temporal_sad: lookahead.mean_temporal_sad(),
            },
            motion_pyramid,
        }
    }

    /// Build the global seed against the reconstructed DPB plane selected for
    /// this frame. References may have coded padding, so copy only the visible
    /// rectangle and normalize high bit depth to the analysis' 8-bit scale.
    /// Score a reconstructed reference after compensating its dominant frame
    /// translation. The score is sampled/normalized by `pyramid_level_cost`,
    /// with a small MV penalty to prefer simpler references on close matches.
    pub(crate) fn motion_compensated_reference_score(
        &self,
        reference: &[f32],
        stride: usize,
        width: usize,
        height: usize,
        bit_depth: u8,
        max_range: i32,
    ) -> Option<(u64, Mv)> {
        if width == 0
            || height == 0
            || stride < width
            || reference.len() < stride.saturating_mul(height)
        {
            return None;
        }
        let scale = 1.0 / (1u32 << bit_depth.saturating_sub(8)) as f32;
        let mut visible = Vec::with_capacity(width * height);
        for row in reference.chunks_exact(stride).take(height) {
            visible.extend(row[..width].iter().map(|&sample| sample * scale));
        }
        let reference_pyramid = build_pyramid(&visible, width, height, self.motion_pyramid.len());
        let seed =
            hierarchical_translation(&self.motion_pyramid, &reference_pyramid, max_range.max(0));
        let distortion = pyramid_level_cost(
            &self.motion_pyramid[0],
            &reference_pyramid[0],
            seed.col / 8,
            seed.row / 8,
        );
        let motion_penalty =
            (seed.row.unsigned_abs() as u64 + seed.col.unsigned_abs() as u64).saturating_mul(128);
        Some((distortion.saturating_add(motion_penalty), seed))
    }

    /// Fraction of 64x64 luma superblocks for which a candidate second reference
    /// is clearly a better zero-motion predictor than the primary reference — i.e.
    /// occlusion / reveal / periodic-motion regions where the older frame matches
    /// and the recent one does not. Listing a second reference taxes every inter
    /// block one `single_ref` bit, so the encoder only lists it when this fraction
    /// is high enough to offset that overhead. Both planes are the internal f32
    /// recon scale; `stride0`/`stride1` skip coded padding.
    pub(crate) fn second_reference_preferred_fraction(
        &self,
        primary: &[f32],
        stride0: usize,
        second: &[f32],
        stride1: usize,
        width: usize,
        height: usize,
    ) -> f32 {
        if width == 0 || height == 0 {
            return 0.0;
        }
        let sb_sad = |plane: &[f32], stride: usize, x0: usize, y0: usize, w: usize, h: usize| {
            let mut sad = 0.0f32;
            for dy in 0..h {
                let cur = &self.luma[(y0 + dy) * width + x0..];
                let refr = &plane[(y0 + dy) * stride + x0..];
                for dx in 0..w {
                    sad += (cur[dx] - refr[dx]).abs();
                }
            }
            sad
        };
        let mut total = 0usize;
        let mut preferred = 0usize;
        let mut y = 0;
        while y < height {
            let mut x = 0;
            let h = 64.min(height - y);
            while x < width {
                let w = 64.min(width - x);
                let s0 = sb_sad(primary, stride0, x, y, w, h);
                let s1 = sb_sad(second, stride1, x, y, w, h);
                total += 1;
                // Second ref genuinely better: at least ~20% lower SAD AND the
                // primary is not already a near-perfect match (so the win would
                // actually change the block's coded cost, not just its reference).
                let px = (w * h) as f32;
                if s1 * 1.2 < s0 && s0 > 2.0 * px {
                    preferred += 1;
                }
                x += 64;
            }
            y += 64;
        }
        if total == 0 {
            0.0
        } else {
            preferred as f32 / total as f32
        }
    }

    /// Chroma contribution for 4:2:0 reference ranking. Chroma inherits the
    /// luma motion (halved in sample space); block-level coding may still make
    /// its own residual decisions after the frame reference is selected.
    pub(crate) fn motion_compensated_chroma_score<T: Pixel>(
        &self,
        image: &PlanarImage<T>,
        reference: &RefFrame<f32>,
        luma_seed: Mv,
    ) -> Option<u64> {
        if image.planes[1].is_empty()
            || image.planes[2].is_empty()
            || reference.planes.len() < 3
            || reference.strides.len() < 3
        {
            return Some(0);
        }
        let (width, height) = (image.width.div_ceil(2), image.height.div_ceil(2));
        let (dx, dy) = (luma_seed.col / 16, luma_seed.row / 16);
        let scale = 1.0 / (1u32 << image.bit_depth.bits().saturating_sub(8)) as f32;
        let mut total = 0u64;
        for plane in 1..=2 {
            total = total.saturating_add(plane_translation_score(
                &image.planes[plane],
                width,
                &reference.planes[plane],
                reference.strides[plane],
                width,
                height,
                dx,
                dy,
                scale,
            )?);
        }
        Some(total)
    }
}

#[allow(clippy::too_many_arguments)]
fn plane_translation_score<T: Pixel>(
    current: &[T],
    current_stride: usize,
    reference: &[f32],
    reference_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
    scale: f32,
) -> Option<u64> {
    if current_stride < width
        || reference_stride < width
        || current.len() < current_stride.saturating_mul(height)
        || reference.len() < reference_stride.saturating_mul(height)
    {
        return None;
    }
    let (w, h) = (width as i32, height as i32);
    let (x0, x1) = ((-dx).max(0), (w - dx).min(w));
    let (y0, y1) = ((-dy).max(0), (h - dy).min(h));
    if x1 - x0 < w / 2 || y1 - y0 < h / 2 {
        return None;
    }
    let sample_step = width.max(height).div_ceil(64).max(1);
    let mut sad = 0.0f64;
    let mut count = 0u64;
    for y in (y0 as usize..y1 as usize).step_by(sample_step) {
        let cy = y * current_stride;
        let ry = (y as i32 + dy) as usize * reference_stride;
        for x in (x0 as usize..x1 as usize).step_by(sample_step) {
            let source = current[cy + x].to_f32() * scale;
            let prediction = reference[ry + (x as i32 + dx) as usize] * scale;
            sad += (source - prediction).abs() as f64;
            count += 1;
        }
    }
    (count != 0).then_some((sad * 1_000_000.0 / count as f64) as u64)
}

fn build_pyramid(src: &[f32], width: usize, height: usize, levels: usize) -> Vec<PyramidLevel> {
    let mut out = Vec::with_capacity(levels);
    let (mut samples, mut w, mut h) = (src.to_vec(), width, height);
    for _ in 0..levels {
        out.push(PyramidLevel {
            samples: samples.clone(),
            width: w,
            height: h,
        });
        if w <= 1 || h <= 1 {
            break;
        }
        let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
        let mut next = vec![0.0; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = 0.0;
                let mut n = 0.0;
                for yy in 2 * y..(2 * y + 2).min(h) {
                    for xx in 2 * x..(2 * x + 2).min(w) {
                        sum += samples[yy * w + xx];
                        n += 1.0;
                    }
                }
                next[y * nw + x] = sum / n;
            }
        }
        (samples, w, h) = (next, nw, nh);
    }
    out
}

fn pyramid_level_cost(cur: &PyramidLevel, reference: &PyramidLevel, dx: i32, dy: i32) -> u64 {
    if cur.width != reference.width || cur.height != reference.height {
        return u64::MAX;
    }
    let (w, h) = (cur.width as i32, cur.height as i32);
    let (x0, x1) = ((-dx).max(0), (w - dx).min(w));
    let (y0, y1) = ((-dy).max(0), (h - dy).min(h));
    if x1 - x0 < w / 2 || y1 - y0 < h / 2 {
        return u64::MAX;
    }
    let mut sad = 0.0f64;
    let mut count = 0u64;
    // Cap a global-seed evaluation to roughly 64x64 samples at large
    // resolutions. Per-block ME performs the dense final decision later.
    let sample_step = cur.width.max(cur.height).div_ceil(64).max(1);
    for y in (y0 as usize..y1 as usize).step_by(sample_step) {
        let cy = y * cur.width;
        let ry = (y as i32 + dy) as usize * reference.width;
        for x in (x0 as usize..x1 as usize).step_by(sample_step) {
            sad += (cur.samples[cy + x] - reference.samples[ry + (x as i32 + dx) as usize]).abs()
                as f64;
            count += 1;
        }
    }
    if count == 0 {
        u64::MAX
    } else {
        (sad * 1_000_000.0 / count as f64) as u64
    }
}

fn hierarchical_translation(
    current: &[PyramidLevel],
    reference: &[PyramidLevel],
    max_range: i32,
) -> Mv {
    if current.is_empty() || current.len() != reference.len() || max_range == 0 {
        return Mv::ZERO;
    }
    let mut best = (0i32, 0i32);
    for level in (0..current.len()).rev() {
        if level + 1 < current.len() {
            best.0 *= 2;
            best.1 *= 2;
        }
        let range = (max_range >> level).max(1);
        best.0 = best.0.clamp(-range, range);
        best.1 = best.1.clamp(-range, range);
        let mut best_cost = pyramid_level_cost(&current[level], &reference[level], best.0, best.1);
        let mut step = if level + 1 == current.len() {
            let mut s: i32 = 1;
            while s.saturating_mul(2) <= range {
                s *= 2;
            }
            s
        } else {
            2.min(range)
        };
        while step >= 1 {
            let mut improved = false;
            for (ox, oy) in [
                (-step, 0),
                (step, 0),
                (0, -step),
                (0, step),
                (-step, -step),
                (step, -step),
                (-step, step),
                (step, step),
            ] {
                let candidate = (
                    (best.0 + ox).clamp(-range, range),
                    (best.1 + oy).clamp(-range, range),
                );
                let cost = pyramid_level_cost(
                    &current[level],
                    &reference[level],
                    candidate.0,
                    candidate.1,
                );
                if cost < best_cost {
                    best = candidate;
                    best_cost = cost;
                    improved = true;
                }
            }
            if !improved {
                step /= 2;
            }
        }
        if level == 0 {
            // Downsampling can leave the propagated seed a few pixels from the
            // true full-resolution minimum. Finish with a small dense window.
            for center in [best, (0, 0)] {
                for oy in -8..=8 {
                    for ox in -8..=8 {
                        let candidate = (
                            (center.0 + ox).clamp(-range, range),
                            (center.1 + oy).clamp(-range, range),
                        );
                        let cost = pyramid_level_cost(
                            &current[level],
                            &reference[level],
                            candidate.0,
                            candidate.1,
                        );
                        if cost < best_cost {
                            best = candidate;
                            best_cost = cost;
                        }
                    }
                }
            }
        }
    }
    Mv {
        row: best.1 * 8,
        col: best.0 * 8,
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum PartitionDecision {
    None,
    Split,
    Horizontal,
    Vertical,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum PredictionDecision {
    Intra { mode: u8 },
    Skip,
    NearMv,
    NewMv,
}

/// One fully resolved block decision. The still coding adapter currently owns
/// these choices internally; as each inter path is migrated it appends the exact
/// partition/mode/reference/MV/transform/Q tuple here for deterministic replay.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockDecision {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) partition: PartitionDecision,
    pub(crate) prediction: PredictionDecision,
    pub(crate) reference_slot: Option<usize>,
    pub(crate) mv: Mv,
    pub(crate) transform_size: (u8, u8),
    pub(crate) qindex: u8,
}

/// The immutable decisions consumed by emission.
pub(crate) struct FrameDecision {
    pub(crate) frame_type: FrameType,
    pub(crate) reference_slots: Vec<usize>,
    pub(crate) base_q_idx: u8,
    pub(crate) analysis_levels: usize,
    pub(crate) pyramid_samples: usize,
    pub(crate) coarsest_size: (usize, usize),
    pub(crate) scene_score: f32,
    pub(crate) lookahead_mean: FrameComplexity,
    pub(crate) motion_seed: Mv,
    pub(crate) blocks: Vec<BlockDecision>,
}

impl FrameDecision {
    pub(crate) fn low_delay(
        frame_type: FrameType,
        reference_slot: Option<usize>,
        base_q_idx: u8,
        analysis: &FrameAnalysis,
        motion_seed: Mv,
    ) -> Self {
        Self {
            frame_type,
            reference_slots: reference_slot.into_iter().collect(),
            base_q_idx,
            analysis_levels: analysis.motion_pyramid.len(),
            pyramid_samples: analysis
                .motion_pyramid
                .iter()
                .map(|level| level.samples.len())
                .sum(),
            coarsest_size: analysis
                .motion_pyramid
                .last()
                .map(|level| (level.width, level.height))
                .unwrap_or((0, 0)),
            scene_score: analysis.scene_score,
            lookahead_mean: analysis.lookahead_mean,
            motion_seed,
            blocks: Vec::new(),
        }
    }
}

/// Reusable worker-owned storage. It is intentionally representation-specific:
/// prediction/reconstruction/filter samples are `f32`; coefficient work is i32.
#[derive(Default)]
pub(crate) struct ScratchArena {
    samples: Vec<f32>,
    coefficients: Vec<i32>,
    entropy: Vec<u8>,
}

impl ScratchArena {
    fn reserve_for_frame(&mut self, pixels: usize) {
        self.samples
            .reserve(pixels.saturating_sub(self.samples.capacity()));
        self.coefficients
            .reserve((pixels / 4).saturating_sub(self.coefficients.capacity()));
        self.entropy
            .reserve((pixels / 16).saturating_sub(self.entropy.capacity()));
    }
}

pub(crate) struct ScratchArenas {
    workers: Vec<ScratchArena>,
}

impl ScratchArenas {
    pub(crate) fn new(threads: usize) -> Self {
        Self {
            workers: (0..threads.max(1))
                .map(|_| ScratchArena::default())
                .collect(),
        }
    }

    pub(crate) fn prepare(&mut self, pixels: usize) {
        let per_worker = pixels.div_ceil(self.workers.len());
        for arena in &mut self.workers {
            arena.reserve_for_frame(per_worker);
        }
    }
}

pub(crate) struct EmittedFrame {
    pub(crate) data: Vec<u8>,
    pub(crate) reference: Option<RefFrame<f32>>,
}

/// Deterministic frame serialization followed by extraction of the final,
/// post-signaled-filter reconstruction. The caller may install `reference` only
/// after this function returns successfully.
pub(crate) struct FrameEmit;

impl FrameEmit {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit(
        still: &Av2Frame,
        config: &Config,
        frame_type: FrameType,
        order_hint: u64,
        emit_sequence_header: bool,
        chroma_strides: [usize; 3],
        reference_slot: usize,
        second_reference_slot: Option<usize>,
        refresh_slot: usize,
    ) -> Result<EmittedFrame, crate::EncodeError> {
        let (cw, ch) = still.coded_dims();
        let mut data = Vec::new();
        if emit_sequence_header {
            data.extend(obu(1, &sequence_header_video(config, cw as u32, ch as u32)));
        }
        let (_td, _seq, still_frame) = still.split_obus();
        let grid = still.tile_grid();
        let tile = frame::tile_payload_of(still_frame, config, cw as u32, ch as u32, grid);
        let frame_obu = match frame_type {
            FrameType::Key => {
                frame::video_key_frame(config, cw as u32, ch as u32, tile, order_hint, grid)?
            }
            FrameType::Inter => frame::video_inter_frame(
                config,
                frame::VideoInterFrameSpec {
                    sw: cw as u32,
                    sh: ch as u32,
                    tile,
                    order_hint,
                    tiles: grid,
                    reference_slot,
                    second_reference_slot,
                    refresh_slot,
                },
            )?,
        };
        data.extend(obu(2, &[]));
        data.extend(frame_obu);

        let chroma_height = match config.layout {
            crate::av2::layout::Layout::I420 => ch.div_ceil(2),
            crate::av2::layout::Layout::I422 | crate::av2::layout::Layout::I444 => ch,
            crate::av2::layout::Layout::Monochrome => 0,
        };
        let plane_heights = [ch, chroma_height, chroma_height];
        let actual_strides: [usize; 3] = std::array::from_fn(|plane| {
            let len = still.recon_planes().get(plane).map_or(0, Vec::len);
            let height = plane_heights[plane];
            if len == 0 || height == 0 {
                return 0;
            }
            if len.is_multiple_of(height) {
                len / height
            } else if len.saturating_sub(1).is_multiple_of(height) {
                (len - 1) / height
            } else {
                chroma_strides[plane]
            }
        });
        let reference = (!still.recon_planes().is_empty()).then(|| RefFrame {
            planes: std::sync::Arc::new(still.recon_planes().to_vec()),
            width: cw,
            height: ch,
            strides: actual_strides.to_vec(),
            order_hint,
        });
        Ok(EmittedFrame { data, reference })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_reduces_geometry() {
        let p = build_pyramid(&[1.0; 8 * 4], 8, 4, 3);
        assert_eq!((p[0].width, p[0].height), (8, 4));
        assert_eq!((p[1].width, p[1].height), (4, 2));
        assert_eq!((p[2].width, p[2].height), (2, 1));
        assert_eq!(p[2].samples.len(), 2);
    }

    #[test]
    fn hierarchical_pyramid_recovers_integer_translation() {
        let (w, h) = (64usize, 64usize);
        let reference: Vec<f32> = (0..w * h)
            .map(|i| ((i * 37 + (i / w) * 19 + (i % w) * (i / w)) & 255) as f32)
            .collect();
        let (dx, dy) = (5usize, 3usize);
        let mut current = vec![0.0; w * h];
        for y in 0..h - dy {
            for x in 0..w - dx {
                current[y * w + x] = reference[(y + dy) * w + x + dx];
            }
        }
        let current_pyramid = build_pyramid(&current, w, h, 3);
        let reference_pyramid = build_pyramid(&reference, w, h, 3);
        assert_eq!(
            hierarchical_translation(&current_pyramid, &reference_pyramid, 16),
            Mv {
                row: dy as i32 * 8,
                col: dx as i32 * 8,
            }
        );
    }

    #[test]
    fn selected_dpb_reference_replaces_previous_frame_seed() {
        let (w, h, stride) = (64usize, 64usize, 72usize);
        let current: Vec<u16> = (0..w * h)
            .map(|i| (((i * 37 + (i / w) * 19 + (i % w) * (i / w)) & 255) * 4) as u16)
            .collect();
        let mut previous = vec![0.0; w * h];
        for y in 0..h - 3 {
            for x in 0..w - 5 {
                previous[y * w + x] = current[(y + 3) * w + x + 5] as f32 / 4.0;
            }
        }
        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: crate::BitDepth::Ten,
            planes: [
                current.clone(),
                vec![512; w * h / 4],
                vec![512; w * h / 4],
                Vec::new(),
            ],
        };
        let analysis = FrameAnalysis::analyze(&image, &previous, &Lookahead::new(1));
        assert_ne!(
            hierarchical_translation(
                &analysis.motion_pyramid,
                &build_pyramid(&previous, w, h, analysis.motion_pyramid.len()),
                16,
            ),
            Mv::ZERO
        );

        let mut selected = vec![0.0; stride * h];
        for y in 0..h {
            for x in 0..w {
                selected[y * stride + x] = current[y * w + x] as f32;
            }
        }
        assert_eq!(
            analysis
                .motion_compensated_reference_score(&selected, stride, w, h, 10, 16)
                .unwrap()
                .1,
            Mv::ZERO
        );
    }
}
