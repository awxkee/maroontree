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

mod api;
mod core;
mod lossless;
mod partition;
#[cfg(test)]
pub(crate) use partition::{
    INTER_MOTION_SKIP_RECT_COUNT, INTER_NEARMV_SKIP_16_COUNT, INTER_NEARMV_SKIP_32_COUNT,
    INTER_NEWMV_SKIP_16_COUNT, INTER_NEWMV_SKIP_32_COUNT, INTER_RESIDUAL_16_CHROMA_COUNT,
    INTER_RESIDUAL_16_COUNT, INTER_RESIDUAL_16_HIGH_EOB_COUNT, INTER_RESIDUAL_32_COUNT,
    INTER_RESIDUAL_32_HIGH_EOB_COUNT, INTER_RESIDUAL_64_COUNT, INTER_SKIP_32_COUNT,
    INTER_SKIP_RECT_COUNT, INTRA_LEAF_COUNT, TOTAL_LEAF_COUNT,
};
mod tiles;

/// Whole-64 blocks committed on reference rank 1 (two-reference frames).
#[cfg(test)]
pub(crate) fn core_skip_rank1_count() -> usize {
    core::CORE_SKIP_RANK1_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}
#[cfg(test)]
pub(crate) fn core_newmv_rank1_count() -> usize {
    core::CORE_NEWMV_RANK1_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}
/// Whole-64 GLOBALMV-skip blocks committed on rank 1 in the partition walk.
#[cfg(test)]
pub(crate) fn partition_skip_rank1_count() -> usize {
    partition::PARTITION_SKIP_RANK1_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}
