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
mod aq;
mod avif;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod ccso;
mod cdef_est;
mod deblock;
// Staged-threading Phase 1 scaffolding: per-SB decision record + replay for the
// 4:4:4 core. Not yet wired into the encode path (converted path-by-path).
mod cdf_para;
mod cdf_state;
mod cdfs_qctx;
pub(crate) mod cdfs_uv_qcx;
mod cdfx_4tx;
#[allow(dead_code)]
mod cfl;
mod chroma422;
mod coder;
mod csc;
#[allow(dead_code)]
mod directional;
mod entropy;
mod fdct;
mod headers;
mod helpers;
mod intrapred;
pub(crate) mod itx;
pub mod itx422;
mod layout;
mod leaf;
mod lossless;
mod lossless_rd;
mod metrics;
mod mhccp;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod partition;
mod proj;
mod quant;
#[allow(dead_code)]
mod replay;
pub mod simple;
pub(crate) mod tables;
mod tables_tx32;
mod tiling;
pub mod video;
mod wht;
mod y400;
mod y420;
mod y422;
mod y444;

use crate::av2::avif::{Av2Color, Av2Format};
use crate::av2::cdfs_qctx::{
    CHROMA_EOB_HI_BIT_QC, CHROMA_SKIP_TX32_QC, CHROMA_SKIP_TX64_QC, INTER_SKIP_TX16_QC,
    SKIP_TX8_QC, SKIP_TX16_QC,
};
use crate::av2::cdfx_4tx::{TXB_SKIP_TX4_Q0, V_TXB_SKIP_TX4_Q0};
use crate::av2::chroma422::{
    ChromaNeighbors, ChromaPlanes, ChromaTuInput, ChromaTxSpec, code_422_chroma_tu,
    recon_422_chroma,
};
use crate::av2::coder::{
    Coeff, EobCdf, InterResidualSpec, LosslessDpcm, LosslessIntrabc, LosslessLumaBlock,
    LumaLeafRect128Spec, LumaSplitDirSpec, encode_chroma_block, encode_chroma_block_ex,
    encode_chroma_block_rect, encode_chroma_block_rect_w, encode_chroma_tu4,
    encode_lossless_luma_sb, encode_luma_block_horz4, encode_luma_block_split,
    encode_luma_block_split_dir, encode_luma_block_vert4, encode_luma_leaf_8x8,
    encode_luma_leaf_8x32, encode_luma_leaf_16x16_full, encode_luma_leaf_16x32,
    encode_luma_leaf_16x64, encode_luma_leaf_32x8, encode_luma_leaf_32x16, encode_luma_leaf_32x32,
    encode_luma_leaf_32x64, encode_luma_leaf_64x16, encode_luma_leaf_64x32,
};
use crate::av2::csc::{
    CB_B, CB_G, CB_R, CR_B, CR_G, CR_R, HALF, Q, Y_B, Y_G, Y_R, get_q_ctx, validate_dims,
};
use crate::av2::entropy::RangeEncoder;
use crate::av2::headers::{Config, frame_header, obu, sequence_header};
use crate::av2::helpers::{
    BoundedIntraRect, PlaneRect, TX16_TYPE_RATE_DELTA, TxbContextSpec, choose_tx16_type,
    coeff_abs_rate_f32, coeff_rate_f32, coeff_tu_rate_proxy_f32, coeff_tus_rate_proxy_f32, dc_pred,
    dc_pred_rect, dc_pred_rect_bounded, dc_pred_rect_subsampled, get_residual, get_residual_rect,
    levels_to_coeffs, lossless_sb_tus, pad_plane, par_map_indexed, pixel_sse_f32_u16_block,
    pixel_sse_rounded_block, pixel_sse_rounded_block_const, put_block, put_block_rect, rect_rows,
    rect_rows_mut, rect_sse_f32, sb_align, sb_tu_contexts, sb_tu_contexts_64x32,
    sb_tu_contexts_pos, sb_tu_contexts_rect, sb_tu4_chroma_skip, sb_tu4_contexts, tx16_dc_only,
    tx16_distortion,
};
use crate::av2::itx422::reconstruct_luma;
use crate::av2::layout::Layout;
use crate::av2::leaf::{
    LumaFrameBlock, LumaGridBlock, LumaPartitionDecision, LumaPartitionSearch, LumaQuantSpec,
    LumaSbSearch, LumaSource, choose_luma_64x64_partition, encode_luma_leaf_s32x32,
    encode_luma_leaf_v32x64, encode_luma_leaf32, encode_luma_sb,
};
use crate::av2::proj::Basis;
use crate::av2::tables::{SCAN8X8, SCAN8X32, SCAN16, SCAN16X32, SCAN32X8, SCAN32X16};
use crate::err::EncodeError;
use crate::metadata::{ContentLightLevel, Orientation};
use crate::{ChromaFormat, Cicp, Pixel, PlanarImage, Speed};
pub use aq::DarkAq;

// Free luma-leaf prediction/coding helpers live in `leaf`.

// Q0.13 RGB→YCbCr coefficients, dimension validation, and `get_q_ctx` live in `csc`.

/// Split the leading OBU off a stream. Our `obu()` writes leb128(size) then the
/// header byte then payload, so an OBU spans `leb_len + size` bytes.
fn split_one_obu(data: &[u8]) -> (&[u8], &[u8]) {
    let mut size = 0u64;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let b = data[i];
        size |= ((b & 0x7f) as u64) << shift;
        i += 1;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    let total = i + size as usize;
    (&data[..total], &data[total..])
}

/// Result of an encode: the AV2 bitstream plus the metadata needed to interpret it.
pub struct Av2Frame {
    data: Vec<u8>,
    width: usize,
    height: usize,
    /// Coded (decoder-output) dimensions = the size signaled in the OBU. Equal to
    /// width/height for lossless and SB-aligned lossy; for padded lossy they are the
    /// 64-aligned size, and the AVIF muxer adds a `clap` box cropping to width/height.
    coded_width: usize,
    coded_height: usize,
    bit_depth: u8,
    color: Cicp,
    chroma_format: ChromaFormat,
    /// Tile grid actually used (log2_cols, log2_rows, tile_size_bytes); (0,0,1) = single tile.
    pub(crate) tile_grid: (usize, usize, usize),
    /// Final recon planes (Y,U,V f32) for the video DPB; empty for still.
    pub(crate) recon: Vec<Vec<f32>>,
    /// Frame-level IBC decision used when a still tile payload is rewrapped as
    /// a video key frame. It must match the still frame header stripped by video.
    pub(crate) allow_intrabc: bool,
    /// Exact frame config used to write the stripped still header. Video reuses
    /// this so derived filters (CDEF/CCSO) cannot diverge while rewrapping tiles.
    pub(crate) video_config: Config,
}

impl Av2Frame {
    pub fn view(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Retained post-filter reconstruction planes, when requested on the encoder.
    pub fn reconstruction(&self) -> &[Vec<f32>] {
        &self.recon
    }

    /// Coded (OBU-signaled) dimensions.
    pub(crate) fn coded_dims(&self) -> (usize, usize) {
        (self.coded_width, self.coded_height)
    }

    /// Tile grid (log2_cols, log2_rows, tile_size_bytes) used by to encode.
    pub(crate) fn tile_grid(&self) -> (usize, usize, usize) {
        self.tile_grid
    }

    pub(crate) fn recon_planes(&self) -> &[Vec<f32>] {
        &self.recon
    }

    pub(crate) fn allow_intrabc(&self) -> bool {
        self.allow_intrabc
    }

    pub(crate) fn video_config(&self) -> &Config {
        &self.video_config
    }

    pub(crate) fn split_obus(&self) -> (&[u8], &[u8], &[u8]) {
        let (td, rest) = split_one_obu(&self.data);
        let (seq, frame) = split_one_obu(rest);
        (td, seq, frame)
    }
}

/// Luma transform-partition strategy for a 64x64 superblock (replaces the old
/// `AV2_TXPART` env). `ThreeWay` RD-chooses among SPLIT/VERT4/HORZ4; `Rd2` restricts
/// to {SPLIT,VERT4}; the rest force a single partition (mainly for testing).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TxPart {
    /// RD between SPLIT (4xTX_32X32), VERT4 (4xTX_16X64) and HORZ4 (4xTX_64X16).
    #[default]
    ThreeWay,
    /// RD between SPLIT and VERT4 only.
    Rd2,
    /// Force SPLIT (4xTX_32X32).
    Split,
    /// Force VERT4 (4xTX_16X64).
    Vert4,
    /// Force HORZ4 (4xTX_64X16).
    Horz4,
}

/// Encoder tuning knobs. Previously set through `AV2_*` environment variables; now
/// a plain config value carried by [`Av2Encoder`]. [`Tuning::default`] reproduces the
/// shipping defaults exactly.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// Requested tile-grid columns (>=1). The encoder rounds up to a power of two and
    /// clamps to the available superblock count; 1 means a single column of tiles.
    pub tile_cols: usize,
    /// Requested tile-grid rows (>=1).
    pub tile_rows: usize,
    /// Luma transform-partition strategy.
    pub txpart: TxPart,
    /// Trellis-RDOQ strength (level^2 per bit). `0.0` disables RDOQ (round-to-nearest
    /// + EOB truncation baseline).
    pub rdoq_lambda: f32,
    /// Trellis-RDOQ strength for chroma planes (level^2 per bit). `0.0` disables the
    /// chroma trellis (round-to-nearest + EOB truncation baseline), which is the
    /// default — chroma RDOQ is bitstream-affecting and opt-in.
    pub chroma_rdoq_lambda: f32,
    /// Multiplier `c` in the tx-partition RD lambda `lambda = c * qstep^2`.
    pub part_lambda_c: f32,
    /// Enable the in-loop deblocking filter
    pub deblock: bool,
    /// Enable deblocking of the chroma (U/V) planes.
    pub chroma_deblock: bool,
    /// Enable CCSO (cross-component sample offset) on the U plane. Phase 1:
    /// band-offset-only, every superblock filtered, offsets derived by a per-band
    /// SSE search. Off by default while under development.
    pub ccso: bool,
    /// CCSO per-superblock RD on/off threshold scale (higher = more conservative,
    /// fewer superblocks filtered). Default 16.0.
    pub ccso_rd_scale: f32,
    /// Deblock threshold quantizer-index offset for luma / chroma (range -2..=1
    /// with df_par_bits=2). 0 = thresholds derived purely from the frame qindex.
    pub db_delta_y: i32,
    pub db_delta_uv: i32,
    /// Enable the in-loop CDEF (directional de-ring)
    pub cdef: bool,
    /// AV2 chroma-from-luma (CfL) intra prediction
    pub cfl: bool,
    /// AV2 multi-hypothesis cross-component prediction (MHCCP)
    pub mhccp: bool,
    /// Enable per-superblock adaptive quantization (variance-driven delta-Q).
    /// Bitstream-affecting; spends fewer bits on busy SBs and more on flat ones.
    pub aq: bool,
    /// Variance Boost selectivity octile (1..=8). Controls how low-variance a 64x64 SB
    /// must be (across its 8x8 subblocks) before its quantizer is boosted. 1 = least
    /// selective (boost readily, more bits), 8 = most selective. Default 6
    /// (SVT-AV1-PSY default). Only active when `aq` is true.
    pub vb_octile: u8,
    /// Variance Boost strength multiplier (1.0 = nominal). Scales the whole qindex
    /// modulation. Only active when `aq` is true.
    pub vb_strength: f32,
    /// When true, Variance Boost only *boosts* low-variance SBs (net-negative, spends
    /// extra bits for quality). When false (default), it also coarsens high-variance
    /// SBs so the average quantizer tracks the frame base (rate-balanced).
    pub vb_boost_only: bool,
    /// Dark-structured-detail AQ protection (see [`aq::DarkAq`]). An independent boost,
    /// combined with the variance boost by `max`, that preserves dark textured/edge SBs
    /// at low quality. Only active when `aq` is true and base_q >= `dark_aq.min_q`.
    pub dark_aq: aq::DarkAq,
    /// Enable the non-CfL chroma intra-mode search
    pub chroma_mode_search: bool,
    /// Enable RD-driven 64x64->4x32x32 chroma-motivated square split (4:4:4/4:2:2).
    /// A 64x64 chroma transform zeros the high-frequency 3/4 of coefficients; splitting
    /// into 32x32 transforms codes all frequencies, a large win on detailed chroma.
    pub chroma_split: bool,
    pub updating_cdf: bool,
    /// Frame-level chroma AC/DC quantizer delta (AVM base_uv_ac_delta_q). Positive
    /// coarsens chroma to shift bits toward luma; 0 = neutral.
    pub uv_ac_delta_q: i32,
    /// SS2-calibration penalty on the rect (VERT4/HORZ4) luma tx-partition distortion.
    /// Rect strips (16x64 / 64x16) win on raw SSE while smearing detail along their long
    /// axis, which SSIMULACRA2 penalizes heavily (observed -3 to -4 SS2 with no rate
    /// benefit). A multiplier >1 makes rect win only when clearly better; 1.0 = off.
    pub rect_part_penalty: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            tile_cols: 1,
            tile_rows: 1,
            txpart: TxPart::ThreeWay,
            rdoq_lambda: proj::DEFAULT_RDOQ_LAMBDA,
            chroma_rdoq_lambda: proj::DEFAULT_RDOQ_LAMBDA,
            part_lambda_c: 0.0001,
            deblock: true,
            chroma_deblock: true,
            ccso: false,
            ccso_rd_scale: 1.0,
            db_delta_y: i32::MIN,
            db_delta_uv: 0,
            cdef: false,
            cfl: true,
            mhccp: true,
            aq: true,
            vb_octile: 6,
            vb_strength: 0.6,
            vb_boost_only: true,
            dark_aq: aq::DarkAq {
                enabled: true,
                ..aq::DarkAq::default()
            },
            chroma_mode_search: true,
            chroma_split: true,
            uv_ac_delta_q: 0,
            updating_cdf: true,
            rect_part_penalty: 1.05,
        }
    }
}

/// A reusable still-image encoder configured for one quality.
pub struct Av2Encoder {
    bases: proj::Bases,
    base_q_idx: u8,
    bit_depth: u8,
    tune: Tuning,
    speed: Speed,
    /// Full-pel motion-search radius selected by the video preset.
    pub(crate) video_search_range: i32,
    /// Normalized 8-bit SAD-per-pixel gate for skipping integer motion search.
    /// Zero disables approximate predictor gating.
    pub(crate) video_predictor_gate: u32,
    /// Full-pel neighborhood radius reranked with SATD after integer SAD search.
    pub(crate) video_integer_satd_radius: u8,
    /// Smallest luma partition enabled by the active video preset.
    pub(crate) video_min_block_size: u8,
    /// Maximum recursive luma partition depth enabled by the video preset.
    pub(crate) video_max_partition_depth: u8,
    /// Hierarchical frame-analysis MV seed (eighth-pel units), used only as a
    /// block-search predictor; it is never signaled as global motion.
    pub(crate) video_mv_seed: std::sync::Mutex<video::mv::Mv>,
    /// Worker-thread budget for tile/superblock parallelism (`0`/`1` = serial).
    /// Sourced from `EncodeConfig::threads` via [`Av2Encoder::with_threads`].
    threads: usize,
    /// When set, tiles emit intra_inter=0 per block (inter frame all-intra tile).
    pub(crate) inter_tile: std::sync::atomic::AtomicBool,
    /// Set by the video encoder for every frame it drives (key and inter alike).
    /// Video frames must keep their historical tiled/serial path: the still-image
    /// default (multi-threaded → single-tile SB-wavefront) is suppressed when this
    /// is set, so a video cfg's `with_tiles`/`with_threads` behaves exactly as before.
    pub(crate) video_mode: std::sync::atomic::AtomicBool,
    pub(crate) capture_recon: std::sync::atomic::AtomicBool,
    /// LAST reference recon planes (Y,U,V f32) for inter prediction; empty for intra.
    pub(crate) last_ref: std::sync::Mutex<std::sync::Arc<Vec<Vec<f32>>>>,
    /// Rank-1 reference recon planes when the frame header lists two references;
    /// empty for single-reference frames. Every inter block then codes one
    /// single_ref rank bit, so the tile coders and the header must agree on this.
    pub(crate) second_ref: std::sync::Mutex<std::sync::Arc<Vec<Vec<f32>>>>,
    /// Current-frame → reference order-hint distance per rank; scales the
    /// cross-rank derived MV predictor on two-reference frames.
    pub(crate) ref_dists: std::sync::Mutex<[i32; 2]>,
}

/// Returns the AV2 mi-unit frame extents `(mc, mr)` for a native (no-pad) lossy 4:4:4
/// encode, iff both dimensions are "boundary-safe". A dimension is boundary-safe when
/// the last superblock has >8 mi in-frame: mc%16==0 || mc%16>8, where mc =
/// ALIGN_POWER_OF_TWO(W,3)>>2 (avm's mi_cols). Returns None if either dimension is
/// not boundary-safe; the encoder then falls back to padding.
fn lossy_native_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    // The mi grid is 8-px aligned, so mc/mr are always even; the right/bottom SB has
    // (m mod 16) mi in frame. Supported partial-edge residues:
    //   0,10,12,14  → whole 64X64 leaves (m%16==0, or >8 so the implied split never
    //                 triggers; ≥9 mi in frame, coded with edge-clamped TUs);
    //   6,8         → 32-family force-split leaves (32X64 / 64X32 / 32X32 corner);
    //   4           → 16-tap family: 16X64 (right) / 64X16 (bottom) single edges, and
    //                 the 16X16 corner when BOTH dims are residue 4 (DC-only luma).
    // A residue-4 edge combined with a residue-{6,8} edge would need a 16X32 / 32X16
    // corner that is not built yet, so those fall back to padding+clap. Residue 2
    // (8px edge) also still falls back.
    let ok = |m: i64| m % 16 == 0 || m % 16 >= 6 || m % 16 == 4 || m % 16 == 2;
    if !(ok(mc) && ok(mr)) {
        return None;
    }
    // Every even-residue combination is now natively codable in all three chroma formats:
    // residue-2 (8-family), residue-4 (16-family), residue-{6,8} (32-family force-split),
    // and residue-{10,12,14} (64-whole, edge-clamped). Every pairing — including the
    // 16×32 / 32×16 (residue-4 × residue-{6,8}) corner — has a leaf arm.
    Some((mc, mr))
}

/// True when the size needs a force-split partition walk (any edge residue in
/// {6,8}); residues {0,10,12,14} tile into whole 64X64 leaves and use the fast path.
fn lossy_needs_partition(width: usize, height: usize) -> bool {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    let part = |m: i64| m % 16 == 6 || m % 16 == 8 || m % 16 == 4 || m % 16 == 2;
    part(mc) || part(mr)
}

fn native_420_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    let ok = |m: i64| m % 16 == 0 || m % 16 >= 6 || m % 16 == 4 || m % 16 == 2;
    if !(ok(mc) && ok(mr)) {
        return None;
    }
    // Every even-residue combination is now natively codable: residue-2/4 (8/16 leaves),
    // residue-{6,8} (32-family force-split), residue-{10,12,14} (64-whole, edge-clamped),
    // and all their pairings. The corner/edge leaf arms cover every (wu,hu) tuple.
    Some((mc, mr))
}

fn native_422_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    lossy_native_mi(width, height)
}

/// Quantizer context threaded through a partition pass: the q-context index, the
/// DC-neutral reconstruction offset, and the integer quant step. Bundling these three
/// (which always travel together) trims three positional args off every encode call.
#[derive(Clone, Copy)]
struct QuantCtx {
    qc: usize,
    neutral: f32,
    qstep: i32,
    /// Chroma trellis-RDOQ strength; 0.0 = round-to-nearest (no trellis).
    rdoq_lambda: f32,
}

/// Immutable per-pass geometry + quant parameters for a native edge-partition walk.
/// `chroma_stride` is unused by the luma-only (4:0:0) pass.
#[derive(Clone, Copy)]
struct PartitionPass {
    luma_stride: usize,
    chroma_stride: usize,
    width: usize,
    height: usize,
    sb_rows: usize,
    sb_cols: usize,
    tmc: i64,
    tmr: i64,
    quant: QuantCtx,
}

/// Luma reconstruction (`rec`, written) and source (`src`, read) plane refs.
struct LumaPlanes<'a> {
    rec: &'a mut [f32],
    src: &'a [f32],
}

/// Chroma reconstruction + source plane refs for a 4:2:0 partition pass.
struct ChromaPlaneRefs<'a> {
    rec_u: &'a mut [f32],
    rec_v: &'a mut [f32],
    src_u: &'a [f32],
    src_v: &'a [f32],
}

/// The luma above/left neighbor context buffers a partition pass mutates as it walks
/// superblocks (DC-sign / txb-skip contexts plus the partition-context rows).
struct PartitionNeighbors<'a> {
    above: &'a mut [u8],
    left: &'a mut [u8],
    above_pctx: &'a mut [u8],
    left_pctx: &'a mut [u8],
}

/// The per-plane chroma presence-flag buffers a 4:2:0 partition pass mutates per leaf.
struct ChromaNeighborBufs<'a> {
    u_above: &'a mut [i32],
    v_above: &'a mut [i32],
    u_left: &'a mut [i32],
    v_left: &'a mut [i32],
}

impl Av2Encoder {
    /// Retain the final reconstruction planes in the returned [`Av2Frame`].
    pub fn with_reconstruction(self, retain: bool) -> Self {
        self.capture_recon
            .store(retain, std::sync::atomic::Ordering::Relaxed);
        self
    }
    /// Build an 8-bit encoder for `base_q_idx` with default [`Tuning`].
    pub fn new(base_q_idx: u8) -> Self {
        Self::with_bit_depth(base_q_idx, 8)
    }

    /// Build an encoder for `base_q_idx` at a given coded bit depth (8, 10 or 12).
    /// The avm quantizer step is bit-depth-independent, so only the sample range,
    /// reconstruction clamp, DC-prediction neutral and the sequence-header signalling
    /// differ; the bases are unchanged.
    pub fn with_bit_depth(base_q_idx: u8, bit_depth: u8) -> Self {
        assert!(
            matches!(bit_depth, 8 | 10 | 12),
            "bit_depth must be 8, 10 or 12, got {bit_depth}"
        );
        let mut bases = proj::default_bases().rescaled_to_q(base_q_idx as u32);
        bases.set_bit_depth(bit_depth);
        let mut enc = Av2Encoder {
            bases,
            base_q_idx,
            bit_depth,
            tune: Tuning::default(),
            speed: Speed::Slow,
            video_search_range: 128,
            video_predictor_gate: 0,
            video_integer_satd_radius: 0,
            video_min_block_size: 8,
            video_max_partition_depth: 4,
            video_mv_seed: std::sync::Mutex::new(video::mv::Mv::ZERO),
            threads: 1,
            inter_tile: std::sync::atomic::AtomicBool::new(false),
            video_mode: std::sync::atomic::AtomicBool::new(false),
            capture_recon: std::sync::atomic::AtomicBool::new(false),
            last_ref: std::sync::Mutex::new(std::sync::Arc::new(Vec::new())),
            second_ref: std::sync::Mutex::new(std::sync::Arc::new(Vec::new())),
            ref_dists: std::sync::Mutex::new([1, 1]),
        };
        enc.apply_hq_lambda_schedule();
        enc
    }

    /// Set the worker-thread budget (builder style). `0` or `1` = serial; `N`
    /// allows up to `N` threads for tile/superblock parallelism. Typically wired
    /// from `EncodeConfig::threads`.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Replace the full [`Tuning`] (builder style).
    pub fn with_tuning(mut self, tune: Tuning) -> Self {
        self.tune = tune;
        self.apply_hq_lambda_schedule();
        self
    }

    /// Apply the high-quality RDOQ-lambda schedule to any lambda still at the shipping
    /// default, keyed on `base_q_idx`. Explicit lambda overrides are left untouched.
    pub(crate) fn apply_hq_lambda_schedule(&mut self) {
        let sched = proj::hq_rdoq_lambda(self.base_q_idx as u32);
        if (self.tune.rdoq_lambda - proj::DEFAULT_RDOQ_LAMBDA).abs() < 1e-12 {
            self.tune.rdoq_lambda = sched;
        }
        if (self.tune.chroma_rdoq_lambda - proj::DEFAULT_RDOQ_LAMBDA).abs() < 1e-12 {
            self.tune.chroma_rdoq_lambda = sched;
        }
    }

    #[inline]
    pub(crate) fn video_allows_16x16_partitions(&self) -> bool {
        self.video_min_block_size <= 16 && self.video_max_partition_depth >= 2
    }

    /// Request a tile grid for parallel encoding. `cols`/`rows` are rounded up to a
    /// power of two and clamped to the available superblock count; `(1, 1)` (the
    /// default) encodes a single tile. Tiles are encoded in parallel when the encode
    /// call is given more than one thread.
    pub fn with_tiles(mut self, cols: usize, rows: usize) -> Self {
        self.tune.tile_cols = cols.max(1);
        self.tune.tile_rows = rows.max(1);
        self
    }

    /// Select the luma transform-partition strategy.
    pub fn with_txpart(mut self, txpart: TxPart) -> Self {
        self.tune.txpart = txpart;
        self
    }

    /// Set the trellis-RDOQ strength (`0.0` disables RDOQ).
    pub fn with_rdoq_lambda(mut self, lambda: f32) -> Self {
        self.tune.rdoq_lambda = lambda;
        self
    }

    /// Set the chroma trellis-RDOQ strength (0.0 disables it; default 0.0).
    pub fn with_chroma_rdoq_lambda(mut self, lambda: f32) -> Self {
        self.tune.chroma_rdoq_lambda = lambda.max(0.0);
        self
    }

    /// Set the RDO effort level (see [`Speed`]). [`Speed::Slow`]
    /// (the default) does per-candidate RDOQ; faster tiers run RDOQ once on the
    /// winning luma mode, and [`Speed::Fast`] also reduces the intra set.
    pub fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    /// Enable the in-loop deblocking filter (q-derived strength)
    pub fn with_deblock(mut self, on: bool) -> Self {
        self.tune.deblock = on;
        self
    }

    /// Enable the in-loop CDEF filter at a q-derived global strength
    pub fn with_cdef(mut self, on: bool) -> Self {
        self.tune.cdef = on;
        self
    }

    /// Enable AV2 chroma-from-luma (CfL) intra prediction (experimental). Off by default.
    pub fn with_cfl(mut self, on: bool) -> Self {
        self.tune.cfl = on;
        self
    }

    /// Enable per-superblock adaptive quantization (variance-driven delta-Q).
    pub fn with_aq(mut self, on: bool) -> Self {
        self.tune.aq = on;
        self
    }

    /// Configure Variance Boost (variance-adaptive delta-Q), the perceptual quantizer
    /// used when AQ is enabled. `octile` (1..=8) sets selectivity (6 = default),
    /// `strength` scales the effect (1.0 = nominal), and `boost_only` selects pure
    /// quality-boosting (net-negative rate) vs. the default rate-balanced mode.
    pub fn with_variance_boost(mut self, octile: u8, strength: f32, boost_only: bool) -> Self {
        self.tune.vb_octile = octile.clamp(1, 8);
        self.tune.vb_strength = strength.max(0.0);
        self.tune.vb_boost_only = boost_only;
        self
    }

    /// Enable the non-CfL chroma intra-mode search
    pub fn with_chroma_mode_search(mut self, on: bool) -> Self {
        self.tune.chroma_mode_search = on;
        self
    }

    pub fn with_ccso(mut self, on: bool) -> Self {
        self.tune.ccso = on;
        self
    }

    /// Force chroma partitioning on/off. Off routes 4:2:0 through the whole-SB fast
    /// path (used to exercise the regular-tx deblock/CCSO path in tests).
    pub fn with_chroma_split(mut self, on: bool) -> Self {
        self.tune.chroma_split = on;
        self
    }

    pub fn with_mhccp(mut self, on: bool) -> Self {
        self.tune.mhccp = on;
        self
    }

    /// Enable adaptive CDF updating during tile decode
    pub fn with_updating_cdf(mut self, on: bool) -> Self {
        self.tune.updating_cdf = on;
        self
    }

    /// Current tuning.
    pub fn tuning(&self) -> Tuning {
        self.tune
    }

    /// The quality this encoder is configured for.
    pub fn base_q_idx(&self) -> u8 {
        self.base_q_idx
    }

    /// Retune the per-frame quantizer while preserving the encoder's pixel
    /// representation and tool configuration. Used by video CQ before decision.
    pub(crate) fn set_video_base_q(&mut self, base_q_idx: u8) {
        if self.base_q_idx == base_q_idx {
            return;
        }
        self.base_q_idx = base_q_idx;
        self.bases = proj::default_bases().rescaled_to_q(base_q_idx as u32);
        self.bases.set_bit_depth(self.bit_depth);
    }

    fn config(&self, layout: Layout) -> Config {
        // Deblock strength estimation from the filter's own quantizer threshold.
        // The AV2 loop filter derives its edge threshold from the plane qindex via
        // `df_quant_from_qindex(q) = ROUND_POWER_OF_TWO(ac_quant(q), QUANT_TABLE_BITS) >> 6`
        // (== `((qstep + 4) >> 3) >> 6` since our `qstep` is `av2_ac_quant_QTX`). That
        // threshold IS the filter's activity gate, so we estimate directly from it
        // instead of signalling a blind fixed delta:
        //   * df_quant == 0  -> the threshold is below any block-edge step, so the
        //     filter is provably inert (verified: at q>=70 deblock on == off to the
        //     SS2 digit). Don't apply it at all — signalling it is pure waste.
        //   * df_quant >= 5  -> very low quality; the threshold is already large and a
        //     `+1` offset over-smooths (measured: volcanic q40 delta +1 = -0.23 SS2 vs
        //     delta 0 = +0.13). Use offset 0.
        //   * 1 <= df_quant <= 4 -> low/mid quality; `+1` strengthens the filter for a
        //     net gain (volcanic/DSC/manhattan q50-q65 all prefer +1). Use offset +1.
        // Chroma deblock is gated separately by `chroma_deblock`. `db_delta_y` at its
        // `i32::MIN` "auto" default uses this estimate; any explicit value overrides.
        let df_quant = {
            let qs = crate::av2::quant::qstep(self.base_q_idx as u32) as i32;
            ((qs + 4) >> 3) >> 6
        };
        let filter_active = self.tune.deblock && df_quant >= 1;
        let auto_dy = if df_quant >= 5 { 0 } else { 1 };
        let eff_dy = if self.tune.db_delta_y == i32::MIN {
            auto_dy
        } else {
            self.tune.db_delta_y
        };
        Config {
            layout,
            base_q: self.base_q_idx as u32,
            // Chroma delta-q is wired through the 4:4:4 coding paths only so far; other
            // layouts must keep the header at delta 0 or the coding would desync.
            uv_ac_delta_q: if layout == Layout::I444 {
                self.tune.uv_ac_delta_q
            } else {
                0
            },
            deblock: filter_active,
            db_apply: (
                filter_active,
                filter_active,
                filter_active && self.tune.chroma_deblock,
                filter_active && self.tune.chroma_deblock,
            ),
            db_delta: (eff_dy, eff_dy, self.tune.db_delta_uv, self.tune.db_delta_uv),
            tx_switchable: true,
            guided_deblock: None,
            // The real strength is chosen by the two-pass per-block CDEF search and
            // folded into the config in the tiled/finish path; off until then.
            cdef: None,
            // Per-block CDEF is decided by the two-pass search (pass 1 builds recon +
            // grid, pass 2 emits); the header carries the 2-entry table only when the
            // search produced a grid, set alongside `cdef` in the tiled/finish path.
            cdef_per_block: false,
            // CCSO is computed as a post-reconstruction search pass (it needs the
            // final recon to derive offsets), so the Config carries None here and the
            // encoder fills it in just before writing the frame header.
            ccso: None,
            bit_depth: self.bit_depth,
            lossless: self.base_q_idx == 0,
            allow_intrabc: false,
            cfl: self.tune.cfl && self.base_q_idx != 0,
            // MHCCP only makes sense with chroma and lossy coding; gated further
            // per-block by is_mhccp_allowed (not 4x4, block <= 64x64).
            mhccp: self.tune.mhccp && self.base_q_idx != 0 && layout.has_chroma(),
            // Adaptive quantization: on for lossy frames when tuned on. res_log2=2
            // (qindex step 4) keeps |signaled| <= 6 covering a +/-24 qindex span.
            aq: self.tune.aq && self.base_q_idx != 0,
            aq_res_log2: 2,
            // Lossless frames always use static CDFs (AVM forces disable_cdf_update=1
            // for coded-lossless); otherwise follow the tuning flag.
            updating_cdf: self.tune.updating_cdf && self.base_q_idx != 0,
        }
    }

    /// DC-prediction neutral value for the first block (1 << (bit_depth-1)).
    fn dc_neutral(&self) -> f32 {
        (1u32 << (self.bit_depth - 1)) as f32
    }

    pub(crate) fn config_for(&self, layout: Layout) -> Config {
        self.config(layout)
    }

    /// Resolve a caller-supplied thread budget: `0` = use all available cores,
    /// `1` = serial, `N` = up to N threads. Replaces the old `SLIMAV_THREADS` env.
    fn resolve_threads(threads: usize) -> usize {
        if threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            threads
        }
    }

    // Per-format encode entry points (encode_yuv444/420/422/400, their lossless
    // variants, and encode_image_*) live in `y400`, `y420`, `y422`, and `y444`,
    // each adding methods to this same `impl Av2Encoder`.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        mut enc: RangeEncoder,
        config: &Config,
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        color: &Cicp,
    ) -> Av2Frame {
        let ccso_u_result = enc.ccso_u_result.clone();
        let ccso_v_result = enc.ccso_v_result.clone();
        let cdef_result = enc.cdef_result;
        let cdef_decided = enc.cdef_decided.clone();

        let recon = std::mem::take(&mut enc.recon);
        let tile = enc.finish();
        // AV2 derives its mode-info grid by rounding the frame to 4px
        // (ALIGN_POWER_OF_TWO(dim, MI_SIZE_LOG2)); superblocks are 64px (16 mi).
        // A square superblock at the right/bottom edge is force-split (no bits read)
        // only when *less than half* of it (<=32px, i.e. <=8 mi) is in-frame — see
        // is_partition_implied_at_boundary. When >32px is in-frame, every SB stays
        // PARTITION_NONE exactly as in the padded encode, so we can signal the real
        // size and let the decoder crop: the coded tile is byte-identical.
        // mi grid is 8px-aligned (avm dec_set_mb_mi); superblocks are 64px (16 mi).
        let mi_cols = ((width + 7) & !7) / 4;
        let mi_rows = ((height + 7) & !7) / 4;
        const MIB: usize = 16; // 64px superblock in 4px mode-info units
        // Lossless now codes every boundary geometry via the recursive forced-split
        // partition coder, so it always signals the real size (decoder crops to W x H).
        // Lossy doesn't clip its tx blocks at boundaries, so it pads unless SB-aligned.
        let aligned = mi_cols.is_multiple_of(MIB) && mi_rows.is_multiple_of(MIB);
        // Boundary-safe lossy 4:4:4 can also signal real W×H natively (the partial-edge
        // superblock decodes correctly with the edge-clamped entropy contexts).
        let lossy_native = !config.lossless
            && ((config.layout == Layout::I444 && lossy_native_mi(width, height).is_some())
                || (config.layout == Layout::I422 && native_422_mi(width, height).is_some())
                || (config.layout == Layout::I420 && native_420_mi(width, height).is_some())
                || (config.layout == Layout::Monochrome
                    && lossy_native_mi(width, height).is_some()));
        let exact = config.lossless || aligned || lossy_native;
        // Signaled dimensions: real size when boundary-safe, else the padded size.
        let (sw, sh) = if exact { (width, height) } else { (pw, ph) };
        // Fold the derived CCSO U/V results into the config for the frame header.
        let mut config = (*config).clone();
        // Fold the CDEF decision into the header. Per-block (pass 2) sets a 2-entry
        // strength table + cdef_per_block; a legacy global result (tiled path) sets
        // the single strength.
        if self.tune.cdef {
            if let Some(d) = &cdef_decided {
                config.cdef = Some((d.y_str, d.uv_str, d.damping));
                config.cdef_per_block = true;
            } else {
                config.cdef = cdef_result;
            }
        }
        let to_plane = |r: &ccso::PlaneResult| -> headers::CcsoPlane {
            use crate::av2::ccso::PlaneResult::*;
            match r {
                Edge {
                    scale_idx,
                    quant_idx,
                    ext_filter_support,
                    edge_clf,
                    max_band_log2,
                    offsets,
                } => headers::CcsoPlane {
                    bo_only: false,
                    scale_idx: *scale_idx,
                    quant_idx: *quant_idx,
                    ext_filter_support: *ext_filter_support,
                    edge_clf: *edge_clf,
                    max_band_log2: *max_band_log2,
                    offsets: offsets.clone(),
                },
            }
        };
        if ccso_u_result.is_some() || ccso_v_result.is_some() {
            config.ccso = Some(headers::CcsoConfig {
                enable: [false, ccso_u_result.is_some(), ccso_v_result.is_some()],
                planes: [
                    None,
                    ccso_u_result.as_ref().map(&to_plane),
                    ccso_v_result.as_ref().map(&to_plane),
                ],
            });
        }
        let config = &config;
        let mut frame = frame_header(config, sw as u32, sh as u32, (0, 0, 1));
        frame.extend(&tile);
        let mut data = vec![];
        data.extend(obu(2, &[]));
        data.extend(obu(1, &sequence_header(config, sw as u32, sh as u32)));
        data.extend(obu(4, &frame));
        Av2Frame {
            data,
            width,
            height,
            // Coded size = the OBU-signaled size (decoder output). The muxer crops to
            // width/height via `clap` when this is larger (padded lossy).
            coded_width: sw,
            coded_height: sh,
            // Coded bit depth signaled in the sequence header (8/10/12). av2C/pixi in
            // the AVIF muxer must use this.
            bit_depth: self.bit_depth,
            color: *color,

            recon,
            allow_intrabc: config.allow_intrabc,
            video_config: config.clone(),
            tile_grid: (0, 0, 1),
            chroma_format: match config.layout {
                Layout::Monochrome => ChromaFormat::Monochrome,
                Layout::I420 => ChromaFormat::Yuv420,
                Layout::I422 => ChromaFormat::Yuv422,
                Layout::I444 => ChromaFormat::Yuv444,
            },
        }
    }

    /// Finish wrapping a color AV1 OBU stream in an AVIF container.
    pub fn wrap_avif(
        frame: &Av2Frame,
        icc_profile: Option<&[u8]>,
        exif: Option<&[u8]>,
        orientation: Orientation,
        clli: Option<ContentLightLevel>,
    ) -> Result<Vec<u8>, EncodeError> {
        let format = Av2Format {
            bit_depth: frame.bit_depth,
            monochrome: frame.chroma_format == ChromaFormat::Monochrome,
            chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                || frame.chroma_format == ChromaFormat::Yuv420,
            chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
        };
        let color = match icc_profile {
            Some(icc) => Av2Color::Both {
                cicp: frame.color,
                icc: icc.to_vec(),
            },
            None => Av2Color::Cicp(frame.color),
        };
        Ok(avif::to_avif_color(
            frame,
            &format,
            &color,
            exif,
            orientation,
            clli,
        ))
    }

    /// Wrap a color frame together with a monochrome alpha auxiliary item into an
    /// AVIF (alpha = an `encode_yuv400` result, typically of the alpha plane). The
    /// alpha item is linked via `auxl` and tagged with the standard alpha `auxC` URN.
    pub fn wrap_avif_alpha(
        frame: &Av2Frame,
        alpha: &Av2Frame,
        icc_profile: Option<&[u8]>,
        exif: Option<&[u8]>,
        orientation: Orientation,
        clli: Option<ContentLightLevel>,
    ) -> Result<Vec<u8>, EncodeError> {
        let format = Av2Format {
            bit_depth: frame.bit_depth,
            monochrome: frame.chroma_format == ChromaFormat::Monochrome,
            chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                || frame.chroma_format == ChromaFormat::Yuv420,
            chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
        };
        let color = match icc_profile {
            Some(icc) => Av2Color::Both {
                cicp: frame.color,
                icc: icc.to_vec(),
            },
            None => Av2Color::Cicp(frame.color),
        };
        Ok(avif::to_avif_color_alpha(
            frame,
            alpha,
            &format,
            &color,
            exif,
            orientation,
            clli,
        ))
    }
}

/// Maps CLI quality 1–100 to AV2 `base_q_idx` 1–254.
/// quality 100 → q≈3 (near-lossless), quality 60 → q≈100, quality 1 → q=254.
pub fn av2_map_quality(quality: u8) -> u8 {
    debug_assert!((1..=100).contains(&quality));
    ((100 - quality as u32) * 254 / 99).clamp(1, 254) as u8
}
