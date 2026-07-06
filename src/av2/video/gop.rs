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

//! GOP control: keyframe interval and scene-cut detection.
/// Frame coding type for the low-delay pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameType {
    Key,
    Inter,
}

/// Decide the coding type for frame `idx`. Phase A forces Key; phase B uses a
/// fixed keyframe interval; phase C adds SAD-spike scene-cut.
pub struct Gop {
    pub key_interval: u64, // 0 = all-intra (phase A)
    /// Force a keyframe when mean per-pixel luma SAD vs the previous frame exceeds
    /// this (8-bit scale). 0 disables scene-cut detection. Typical: ~12–20.
    pub scene_cut_sad: u32,
    /// Minimum inter frames between forced scene-cut keyframes (avoids thrashing
    /// on noisy/flashing content). Counts frames since the last keyframe.
    pub scene_cut_min_gap: u64,
}

impl Gop {
    pub fn all_intra() -> Self {
        Self {
            key_interval: 0,
            scene_cut_sad: 0,
            scene_cut_min_gap: 4,
        }
    }

    pub fn frame_type(&self, idx: u64) -> FrameType {
        if self.key_interval == 0 || idx.is_multiple_of(self.key_interval.max(1)) {
            FrameType::Key
        } else {
            FrameType::Inter
        }
    }
}

/// Mean per-pixel luma SAD between two equal-length 8-bit planes (scene-cut metric).
/// Returns 0 if lengths differ or are empty.
pub(crate) fn mean_luma_sad(cur: &[u8], prev: &[u8]) -> u32 {
    if cur.is_empty() || cur.len() != prev.len() {
        return 0;
    }
    let sum = crate::av2::helpers::sad_u8(cur, prev);
    sum / cur.len() as u32
}
