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

//! Frame complexity analysis for rate control. Pure encoder-side measurement:
//! it never touches the bitstream. Produces the per-frame spatial (intra) and
//! temporal (inter) complexity signals a Q-selection / bit-allocation step needs.
//! This is the analysis half of the "no complexity allocation" gap; the Q
//! decision that consumes it is a separate, bitstream-affecting step.

use crate::util::FastRound;

/// Per-frame complexity metrics (8-bit luma scale).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FrameComplexity {
    /// Mean per-pixel spatial activity: average abs gradient over the luma plane.
    /// High on detailed/textured frames, low on flat ones. Proxy for intra cost.
    pub(crate) spatial_activity: f32,
    /// Mean per-pixel temporal SAD vs the previous frame (0 for the first frame).
    /// High on motion/scene change, low on static content. Proxy for inter cost.
    pub(crate) temporal_sad: f32,
}

/// Select a bounded frame CQ around `nominal_q`. Motion spikes are allowed to
/// spend fewer bits, while unusually detailed frames receive protection. Ratios
/// are measured against already-buffered lookahead statistics, keeping the
/// decision deterministic and independent of encode results.
pub(crate) fn select_cq_qindex(
    nominal_q: u8,
    max_delta: u8,
    current: FrameComplexity,
    mean: FrameComplexity,
) -> u8 {
    if nominal_q == 0 || max_delta == 0 {
        return nominal_q;
    }
    // No history yet: do not perturb the first frame.
    if mean.spatial_activity <= 0.0 && mean.temporal_sad <= 0.0 {
        return nominal_q;
    }
    let spatial_ratio = (current.spatial_activity + 1.0) / (mean.spatial_activity + 1.0);
    let temporal_ratio = (current.temporal_sad + 1.0) / (mean.temporal_sad + 1.0);
    let score = 3.0 * temporal_ratio.log2() - 2.0 * spatial_ratio.log2();
    let delta = score
        .fast_round()
        .clamp(-(max_delta as f32), max_delta as f32) as i32;
    (i32::from(nominal_q) + delta).clamp(1, 254) as u8
}

/// Mean absolute spatial gradient of an 8-bit luma plane: average of horizontal
/// and vertical neighbor differences. O(pixels), no allocation.
pub(crate) fn spatial_activity(luma: &[u8], width: usize, height: usize) -> f32 {
    if width < 2 || height < 2 || luma.len() < width * height {
        return 0.0;
    }
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for y in 0..height {
        let row = &luma[y * width..y * width + width];
        for x in 0..width - 1 {
            sum += (row[x] as i32 - row[x + 1] as i32).unsigned_abs() as u64;
            count += 1;
        }
        if y + 1 < height {
            let below = &luma[(y + 1) * width..(y + 1) * width + width];
            for x in 0..width {
                sum += (row[x] as i32 - below[x] as i32).unsigned_abs() as u64;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

/// Rolling lookahead buffer of recent frame complexities. A future rate
/// controller reads a window of these to decide each frame's quantizer (e.g.
/// coarsen high-temporal-SAD frames, protect low-complexity ones). Bounded so
/// memory stays flat regardless of stream length.
pub(crate) struct Lookahead {
    window: std::collections::VecDeque<FrameComplexity>,
    cap: usize,
}

impl Lookahead {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
        }
    }

    /// Record one frame's complexity, evicting the oldest past capacity.
    pub(crate) fn push(&mut self, c: FrameComplexity) {
        if self.window.len() == self.cap {
            self.window.pop_front();
        }
        self.window.push_back(c);
    }

    /// Mean temporal SAD over the buffered window (0 if empty). A rate
    /// controller can normalize a frame's SAD against this to spend bits where
    /// motion spikes relative to the local average.
    pub(crate) fn mean_temporal_sad(&self) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        let s: f32 = self.window.iter().map(|c| c.temporal_sad).sum();
        s / self.window.len() as f32
    }

    /// Mean spatial activity over the window (0 if empty).
    pub(crate) fn mean_spatial_activity(&self) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        let s: f32 = self.window.iter().map(|c| c.spatial_activity).sum();
        s / self.window.len() as f32
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.window.len()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_plane_has_zero_activity() {
        let flat = vec![128u8; 64 * 64];
        assert_eq!(spatial_activity(&flat, 64, 64), 0.0);
    }

    #[test]
    fn textured_plane_has_activity() {
        let mut t = vec![0u8; 8 * 8];
        for (i, p) in t.iter_mut().enumerate() {
            *p = if i % 2 == 0 { 0 } else { 255 };
        }
        assert!(spatial_activity(&t, 8, 8) > 0.0);
    }

    #[test]
    fn lookahead_evicts_and_averages() {
        let mut la = Lookahead::new(2);
        la.push(FrameComplexity {
            spatial_activity: 0.0,
            temporal_sad: 10.0,
        });
        la.push(FrameComplexity {
            spatial_activity: 0.0,
            temporal_sad: 20.0,
        });
        la.push(FrameComplexity {
            spatial_activity: 0.0,
            temporal_sad: 30.0,
        });
        assert_eq!(la.len(), 2); // capacity bound holds
        assert_eq!(la.mean_temporal_sad(), 25.0); // (20+30)/2, oldest evicted
    }

    #[test]
    fn cq_selection_is_bounded_and_uses_both_complexity_axes() {
        let mean = FrameComplexity {
            spatial_activity: 8.0,
            temporal_sad: 4.0,
        };
        let motion = select_cq_qindex(
            120,
            6,
            FrameComplexity {
                spatial_activity: 8.0,
                temporal_sad: 40.0,
            },
            mean,
        );
        let detail = select_cq_qindex(
            120,
            6,
            FrameComplexity {
                spatial_activity: 64.0,
                temporal_sad: 4.0,
            },
            mean,
        );
        assert_eq!(motion, 126);
        assert_eq!(detail, 114);
        assert_eq!(select_cq_qindex(0, 6, mean, mean), 0);
    }
}
