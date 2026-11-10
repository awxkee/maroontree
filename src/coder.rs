/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

use crate::Speed;
use crate::chroma_rect::*;
use crate::dct::DctDispatch;
use crate::encoding_context::EncodingContext;
use crate::idct::IdctDispatch;
use crate::kmeans::KmeansDispatch;
use crate::obu::{
    frame_header_lossy_multitile, frame_header_lossy_multitile_th, wrap_obu_frame,
    wrap_obu_frame_split,
};
use crate::odec::OdEcEncoder;
use crate::par::Pool;
use crate::quant::QmLevels;
use hashbrown::{HashMap, HashSet};
#[cfg(test)]
pub(crate) static LOSSY_PALETTE_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static LOSSY_PALETTE_RESIDUAL_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static LOSSY_INTRABC_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static LOSSY_INTRABC_ODD_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static LOSSY_INTRABC_RESID_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Serializes the tests that reset and then read the `LOSSY_*_EMITTED`
/// counters: the counters are process-global, so a reset in one test would
/// otherwise wipe the count another test is about to assert on.
#[cfg(test)]
pub(crate) static EMIT_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// Partition decision for a 16x16 luma region.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Part16 {
    Intrabc,
    None,
    Horz,
    Vert,
    Split,
    HorzA,
    HorzB,
    VertA,
    VertB,
    /// PARTITION_HORZ_4: four stacked 16x4 strips (symbol 8).
    Horz4,
    /// PARTITION_VERT_4: four side-by-side 4x16 strips (symbol 9).
    Vert4,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FixedList<T: Copy, const N: usize> {
    items: [T; N],
    len: usize,
}

impl<T: Copy, const N: usize> FixedList<T, N> {
    #[inline]
    fn new(fill: T) -> Self {
        Self {
            items: [fill; N],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, value: T) {
        debug_assert!(self.len < N);
        self.items[self.len] = value;
        self.len += 1;
    }

    #[inline]
    fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len);
        let value = self.items[index];
        self.items.copy_within(index + 1..self.len, index);
        self.len -= 1;
        value
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.items[..self.len]
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items[..self.len]
    }
}

impl<T: Copy, const N: usize> std::ops::Deref for FixedList<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy, const N: usize> std::ops::DerefMut for FixedList<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Copy + PartialEq, const N: usize> FixedList<T, N> {
    fn dedup(&mut self) {
        if self.len < 2 {
            return;
        }
        let mut out = 1usize;
        for i in 1..self.len {
            if self.items[i] != self.items[out - 1] {
                self.items[out] = self.items[i];
                out += 1;
            }
        }
        self.len = out;
    }
}

impl<'a, T: Copy, const N: usize> IntoIterator for &'a FixedList<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

use crate::aq_common::{DarkAq, dirty_log1pf};
use crate::trellis::{trellis_optimize, trellis_optimize_ctx};

use crate::coeffs::encode_tx16_coeffs_adapt;
use crate::coeffs::*;
use crate::cost::*;
use crate::intrapred::*;
use crate::quant::*;
use crate::tables::*;
use crate::util::FastRound;

/// Raw read view of the wavefront capture's finished-recon planes (see
/// `LossyTile::ibc_shared`). (ptr, len, stride) per plane, luma first.
#[derive(Clone, Copy)]
pub(crate) struct IbcSharedRecon {
    pub(crate) planes: [(*const u16, usize, usize); 3],
}
// SAFETY: reads are restricted to wavefront-finished cells (disjoint from all
// concurrent writers) by the IntraBC legality rule; the planes outlive every
// worker (allocated before the pool, dropped after it joins).
unsafe impl Send for IbcSharedRecon {}
unsafe impl Sync for IbcSharedRecon {}

// Keep the state type and shared imports in this module while splitting its
// implementation by coding responsibility. `include!` preserves private field
// access without widening the encoder's internal API.

include!("coder/state.rs");
include!("coder/scratch.rs");
include!("coder/lossy_state.rs");
include!("coder/palette.rs");
include!("coder/partition_search.rs");
include!("coder/block16.rs");
include!("coder/block8.rs");
include!("coder/block32.rs");
include!("coder/block64.rs");
include!("coder/block64_rect.rs");
include!("coder/superblock.rs");
include!("coder/trial.rs");
include!("coder/chroma_tx.rs");
include!("coder/cdfs.rs");
include!("coder/aq.rs");
include!("coder/tiling.rs");
include!("coder/tile_encode.rs");
include!("coder/wavefront.rs");
include!("coder/screen_gate.rs");
include!("coder/frame_encode.rs");
include!("coder/frame_wiener.rs");
include!("coder/frame_cdef.rs");
include!("coder/frame_deblock.rs");
include!("coder/lossless_frame.rs");

#[allow(clippy::too_many_arguments)]
#[inline]
fn sse_recon<const N: usize, const D: usize>(
    rd: &crate::rd_sse::RdDispatch,
    pred: &[i32; N],
    resid: &[i32; N],
    src: crate::rd_sse::SrcBlock<'_>,
    bd: u8,
) -> i64 {
    debug_assert_eq!(N, D * D);
    debug_assert!(src.w() == D && src.h() == D);
    rd.sse_recon(pred, resid, src, bd)
}

/// The tile decision dispatch's weighted-SSE configuration (per-format
/// `Tuning::wsse_*`, `wsse_scale`) for a frame-level filter decision.
fn frame_decision_rd(
    rd: &crate::rd_sse::RdDispatch,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
) -> crate::rd_sse::RdDispatch {
    let t = crate::tuning::get();
    let on = if mono {
        t.wsse_400
    } else if sub_x == 1 && sub_y == 1 {
        t.wsse_420
    } else if sub_x == 1 {
        t.wsse_422
    } else {
        t.wsse_444
    };
    rd.with_weighted_sse(on.then_some(t.wsse_scale))
}

/// The frame-level `lambda_d` (normalized-wSSE units per bit): the tile
/// `LossyTile::lambda_d` law without a perceptual anchor.
fn frame_lambda_d(base_q_idx: u8, bd: u8, sub_x: usize, sub_y: usize) -> f32 {
    let sub = if sub_x == 1 && sub_y == 1 {
        2
    } else if sub_x == 1 {
        1
    } else {
        0
    };
    mode_lambda_q(dc_q(base_q_idx, bd) as f32) * mode_lambda_weight(base_q_idx, sub) / K_PART_LAMBDA
}

/// TX-type guard tolerance (`Tuning::tx_guard_shift` / `_444`): the wSSE a
/// candidate may exceed the DCT's by and still be admitted on RD.
#[inline]
fn tx_guard_shift(d: i64, s: u32) -> i64 {
    if s >= 63 { 0 } else { d >> s }
}

#[inline]
fn av1_block_size_index(width: usize, height: usize) -> usize {
    match (width, height) {
        (4, 4) => 0,
        (4, 8) => 1,
        (8, 4) => 2,
        (8, 8) => 3,
        (8, 16) => 4,
        (16, 8) => 5,
        (16, 16) => 6,
        (16, 32) => 7,
        (32, 16) => 8,
        (32, 32) => 9,
        (32, 64) => 10,
        (64, 32) => 11,
        (64, 64) => 12,
        (64, 128) => 13,
        (128, 64) => 14,
        (128, 128) => 15,
        (4, 16) => 16,
        (16, 4) => 17,
        (8, 32) => 18,
        (32, 8) => 19,
        (16, 64) => 20,
        (64, 16) => 21,
        _ => panic!("unsupported AV1 block size {width}x{height}"),
    }
}

#[inline]
fn filter_intra_allowed(y_mode: usize, width: usize, height: usize) -> bool {
    y_mode == DC_PRED && width.max(height) <= 32
}

#[inline]
fn filter_intra_tx_mode(choice: Option<FilterIntraMode>, y_mode: usize) -> usize {
    match choice {
        Some(FilterIntraMode::Vertical) => V_PRED,
        Some(FilterIntraMode::Horizontal) => H_PRED,
        Some(FilterIntraMode::D157) => D157_PRED,
        Some(FilterIntraMode::Dc | FilterIntraMode::Paeth) => DC_PRED,
        None => y_mode,
    }
}

const DIRECTIONAL_RDO_TOP_K: usize = 3;

#[derive(Clone, Copy)]
struct DirectionalTopK {
    modes: [usize; DIRECTIONAL_RDO_TOP_K],
    costs: [u64; DIRECTIONAL_RDO_TOP_K],
    len: usize,
}

impl DirectionalTopK {
    #[inline]
    fn new() -> Self {
        Self {
            modes: [usize::MAX; DIRECTIONAL_RDO_TOP_K],
            costs: [u64::MAX; DIRECTIONAL_RDO_TOP_K],
            len: 0,
        }
    }

    #[inline]
    fn insert(&mut self, mode: usize, cost: u64) {
        let mut pos = self.len.min(DIRECTIONAL_RDO_TOP_K - 1);
        if self.len == DIRECTIONAL_RDO_TOP_K && cost >= self.costs[pos] {
            return;
        }
        if self.len < DIRECTIONAL_RDO_TOP_K {
            self.len += 1;
            pos = self.len - 1;
        }
        while pos > 0 && cost < self.costs[pos - 1] {
            self.costs[pos] = self.costs[pos - 1];
            self.modes[pos] = self.modes[pos - 1];
            pos -= 1;
        }
        self.costs[pos] = cost;
        self.modes[pos] = mode;
    }

    #[inline]
    fn contains(&self, mode: usize) -> bool {
        self.modes[..self.len].contains(&mode)
    }
}

#[inline]
fn is_directional_mode(mode: usize) -> bool {
    (V_PRED..=VERT_LEFT_PRED).contains(&mode)
}

fn prdo_k() -> f32 {
    crate::tuning::get().prdo_k
}

fn prdo_k_444() -> f32 {
    crate::tuning::get().prdo_k_444
}

/// Clamp `C` for the perceptual RD scale: the per-block scale is limited to
/// `[1/C, C]` so no block is starved or flooded.
fn prdo_clamp() -> f32 {
    2.0
}

fn prdo_upper_clamp() -> f32 {
    1.5
}

fn vbp_thresh_420() -> f32 {
    crate::tuning::get().vbp_thresh_420
}

fn top_none_bias_420(base_q: u8) -> f32 {
    let t = crate::tuning::get();
    if base_q <= 20 {
        t.top_none_bias_420_hi
    } else {
        t.top_none_bias_420_lo
    }
}
/// Cost of signaling a non-DC uv_mode for the 4:2:0 4x4 SMOOTH_V chroma trial.
fn smooth_v_uv_signal_bits() -> f32 {
    crate::tuning::get().smooth_v_uv_signal_bits
}
/// Required SSE improvement (in 1/1024) for the 32x32 TX-split to be accepted
/// on a banding-risk block. See `code_block32`.
const SPLIT32_SSE_MARGIN: i64 = 64;

#[inline]
fn chroma_part_rd_weight(ss420: bool, ss422: bool, _src: &[Vec<u16>; 3], _bd: u8) -> f32 {
    let t = crate::tuning::get();
    if ss420 || ss422 {
        t.chroma_part_w_sub
    } else {
        t.chroma_part_w_444
    }
}

#[cfg(test)]
mod aq_tests;
