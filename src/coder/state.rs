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

// `LossyTile` and the state groups it is made of.

/// Chroma format of a lossy tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Chroma {
    /// 4:0:0: luma only (`NumPlanes = 1`, no chroma syntax).
    Mono,
    /// 4:2:0: chroma subsampled horizontally and vertically.
    I420,
    /// 4:2:2: chroma subsampled horizontally.
    I422,
    /// 4:4:4: full-resolution chroma.
    I444,
}

impl Chroma {
    /// Format for the frame header's `(mono_chrome, subsampling_x, subsampling_y)`.
    fn from_sub(mono: bool, sub_x: usize, sub_y: usize) -> Self {
        if mono {
            Chroma::Mono
        } else {
            match (sub_x, sub_y) {
                (0, 0) => Chroma::I444,
                (1, 0) => Chroma::I422,
                _ => Chroma::I420,
            }
        }
    }

    /// Horizontal chroma shift (0 for 4:0:0, which has no chroma planes).
    #[inline]
    fn sub_x(self) -> usize {
        usize::from(matches!(self, Chroma::I420 | Chroma::I422))
    }

    /// Vertical chroma shift.
    #[inline]
    fn sub_y(self) -> usize {
        usize::from(self == Chroma::I420)
    }
}

/// Immutable per-tile configuration: geometry, chroma format, source planes and
/// frame-level switches. Set while the tile is built, read-only while it codes.
struct TileCfg<'a> {
    bd: u8,
    w: usize,
    h: usize,
    cw: usize, // chroma plane width (= w for 4:4:4, w/2 for 4:2:2 and 4:2:0)
    chroma: Chroma,
    /// 4:0:0 with 16x16/32x32 leaves (see `VarianceBoost::mono_large`).
    mono_large: bool,
    /// Frame-level U/V weight used by the shared-tree partition proxy.
    chroma_part_weight: f32,
    allow_intrabc: bool,
    /// Search the screen-content tools (palette). Frame-level; see
    /// [`crate::EncodeConfig::with_screen_content`].
    screen_content: bool,
    /// Frame passed the repeated-content (IntraBC coverage) test: enables the
    /// text palette tools — palette-aware 8x8-vs-split4, 16x8/8x16 palette
    /// leaves, a wider palette refine budget. Photos never pass it, so they
    /// keep the photo-tuned palette Behavior bit-for-bit.
    screen_frame: bool,
    /// Tile-local (column, row) ends in 4x4 units of dav1d's refmvs scan
    /// window: the tile, clipped to the DISPLAY frame (`iw4 = (w + 3) >> 2`,
    /// not the 8-aligned coded width).
    ibc_end4: (usize, usize),
    /// Per-plane tile-local DISPLAYED extent `(width, height)`: the source is
    /// padded to the 8-aligned coded grid and every RD distortion is clipped
    /// to this region (see [`crate::rd_sse::SrcBlock`]). Equal to the plane
    /// size for interior tiles and 8-aligned frames.
    vis: [(usize, usize); 3],
    src: &'a [Vec<u16>; 3],
    /// Tile-wide IntraBC exact-match index (see `LossyIbcIndex`), owned by
    /// `encode_one_tile`.
    ibc_index: Option<&'a std::sync::OnceLock<LossyIbcIndex>>,
    /// Wavefront-capture read view of the shared finished-recon planes.
    ibc_shared: Option<IbcSharedRecon>,
    updating_cdf: bool,
    /// RDO effort: [`Speed::Slow`] (default) or [`Speed::Fast`] (winner-only
    /// RDOQ, DCT-only transform choice, reduced intra mode set).
    speed: Speed,
    /// Global luma Wiener filter to signal per superblock (`read_lr`), or `None`
    /// for RESTORE_NONE (no per-SB LR syntax emitted). When set, every 64x64
    /// restoration unit codes `use_wiener = 1` with these taps, delta-coded
    /// against the running reference `lr_ref_*` (spec 5.11.58).
    wiener: Option<crate::wiener::WienerUnit>,
    /// Frame-absolute luma pixel origin of this tile and the full frame luma
    /// dimensions. Loop-restoration units are frame-relative, so `read_lr` is
    /// computed in frame coordinates even though the tile encodes locally.
    /// (`0,0` and the tile's own size for a single-tile frame.)
    frame_x0: usize,
    frame_y0: usize,
    frame_w: usize,
    frame_h: usize,
    /// Base quant index this tile was built with
    base_q_idx: u8,
}

impl TileCfg<'_> {
    #[inline]
    fn is_420(&self) -> bool {
        self.chroma == Chroma::I420
    }

    #[inline]
    fn is_422(&self) -> bool {
        self.chroma == Chroma::I422
    }

    #[inline]
    fn is_444(&self) -> bool {
        self.chroma == Chroma::I444
    }

    #[inline]
    fn is_mono(&self) -> bool {
        self.chroma == Chroma::Mono
    }
}

/// Pre-resolved per-encode SIMD kernel dispatch tables.
struct Kernels {
    /// Pre-resolved per-encode forward-DCT implementations.
    dct: DctDispatch,
    /// Pre-resolved per-encode inverse-transform implementations.
    idct: IdctDispatch,
    intrapred: IntraPredDispatch,
    kmeans: KmeansDispatch,
    rd: crate::rd_sse::RdDispatch,
}

/// Per-block geometry and skip maps of a tile or a stitched frame: what the
/// frame-level deblocking and CDEF passes read after the tiles are coded.
#[derive(Clone, Default, PartialEq)]
struct BlockMaps {
    blk4: Vec<u8>, // luma block WIDTH (in 4-sample units) per 4x4 luma unit; for the deblock filter (vertical edges)
    blk4h: Vec<u8>, // luma block HEIGHT (in 4-sample units) per 4x4 luma unit; for the deblock filter (horizontal edges)
    blk4v: Vec<bool>, // true where a luma block starts at this 4x4 column
    blk4t: Vec<bool>, // true where a luma block starts at this 4x4 row
    /// Per-4x4 luma PREDICTION-BLOCK geometry (width/height in 4-units and the
    /// block-start flags), as distinct from `blk4`/`blk4h` which are refined to
    /// TRANSFORM granularity by the TX-split paths.
    pblk4: Vec<u8>,
    pblk4h: Vec<u8>,
    pblk4v: Vec<bool>,
    pblk4t: Vec<bool>,
    skip8: Vec<bool>, // per-8x8-luma-unit block skip flag (true = no coded coeffs); for CDEF
}

impl BlockMaps {
    /// Maps for a `w` x `h` luma area with nothing coded yet (all skip).
    fn new(w: usize, h: usize) -> Self {
        let n4 = (w / 4) * (h / 4);
        BlockMaps {
            blk4: vec![0; n4],
            blk4h: vec![0; n4],
            blk4v: vec![false; n4],
            blk4t: vec![false; n4],
            pblk4: vec![0; n4],
            pblk4h: vec![0; n4],
            pblk4v: vec![false; n4],
            pblk4t: vec![false; n4],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
        }
    }
}

/// Live coding context: everything the decoder also tracks while parsing a tile
/// (reconstruction, above/left context arrays, block geometry maps, the
/// current quantizers). This is the state a trial snapshot must cover.
#[derive(Clone, PartialEq)]
struct CodingCtx {
    quant: Quant,
    cquant: Quant,
    ibc_mv: Vec<Option<(i16, i16)>>,
    recon: [Vec<u16>; 3],
    a_coef: [Vec<u8>; 3], // len w/4, absolute bx4
    l_coef: [Vec<u8>; 3], // len h/4, absolute by4
    /// Per-4x4-column coded-TX log2-width (dav1d `a->tx_intra`), -1 at tile
    /// start; feeds the `tx_depth` symbol context (`get_tx_ctx`).
    a_tx: Vec<i8>,
    /// Per-4x4-row coded-TX log2-height (dav1d `l->tx_intra`).
    l_tx: Vec<i8>,
    a_part: Vec<u8>,    // len w/8, absolute x8
    l_part: Vec<u8>,    // len h/8, absolute y8
    a_skip: Vec<u8>,    // block skip flag per 4x4 col, absolute bx4
    l_skip: Vec<u8>,    // block skip flag per 4x4 row, absolute by4
    a_mode: Vec<u8>,    // luma intra mode per 4x4 col (for kf y-mode context)
    l_mode: Vec<u8>,    // luma intra mode per 4x4 row
    a_uv_mode: Vec<u8>, // chroma intra mode above each luma 4x4 column
    l_uv_mode: Vec<u8>, // chroma intra mode left of each luma 4x4 row
    a_palette: Vec<Vec<i32>>,
    /// Above/left neighbor chroma-palette U colors (the UV color cache is
    /// U-plane only, dav1d al_pal[..][1]); empty = no UV palette there.
    a_palette_uv: Vec<Vec<i32>>, // luma palette colors above each 4x4 column
    l_palette: Vec<Vec<i32>>, // luma palette colors left of each 4x4 row
    l_palette_uv: Vec<Vec<i32>>,
    /// Block geometry and skip maps (handed to the frame-level filters).
    maps: BlockMaps,
    /// Whether the current superblock already recorded its `read_cdef()` trace
    /// point (the first non-skip block carries the per-unit `cdef_idx`).
    cdef_point_marked: bool,
    /// Adaptive-quantization state; `AqCtx::off()` unless enabled per tile.
    aq: AqCtx,
    /// Running Wiener tap reference for delta coding (horizontal, vertical), one
    /// per coded tap. Reset to the spec midpoints at the start of the tile.
    lr_ref_h: [i32; 3],
    lr_ref_v: [i32; 3],
}

/// The live entropy coder and the adaptive CDFs it codes against.
struct Entropy {
    enc: OdEcEncoder,
    cdfs: Cdfs,
}

/// Decision-side rate model: the immutable CDF snapshot R-D estimates price
/// against and the cost tables lazily derived from it.
struct RateModel {
    /// Lazily built flat f32 cost tables for the decision `palette_y_color`
    /// CDFs ([size-2][5 map ctx][8 symbols]): the index-map rate walk touches
    /// one entry per map cell, and the triple-Vec `cdf_cost` pointer chase was
    /// the walk's hottest load. Values are exactly `cdf_cost(..)`, so sums are
    /// bit-identical. Decision CDFs are immutable per tile (the same
    /// assumption `pal_est_cache` already relies on).
    #[allow(clippy::type_complexity)]
    pal_y_cost: std::cell::RefCell<Option<Box<[[[f32; 8]; 5]; 7]>>>,
    /// `palette_uv_color` counterpart of `pal_y_cost` (UV map Viterbi).
    #[allow(clippy::type_complexity)]
    pal_uv_cost: std::cell::RefCell<Option<Box<[[[f32; 8]; 5]; 7]>>>,
    /// Snapshot `pal_y_cost` is built from: the FRAME-INITIAL decision CDFs,
    /// captured on the first `switch_decision_cdfs` (None until then, when
    /// `dec_cdfs` still is that snapshot). Pinned so that every worker and the
    /// serial path price palette maps identically regardless of which cell
    /// they see first — and identically to the pre-warm-up encoder.
    pal_cdfs: Option<std::sync::Arc<Cdfs>>,
    /// Coefficient token cost tables of `dec_cdfs` (see
    /// `crate::rate::CoefCostTables`), built lazily and dropped on every
    /// decision-snapshot switch.
    coef_cost: std::cell::RefCell<Option<Box<crate::rate::CoefCostTables>>>,
    /// Immutable CDF snapshot used by decision-side rate estimates (RDOQ
    /// trellis, filter-intra / angle-delta mode costs). Starts at the frame
    /// prior and may switch once to the deterministic pilot-trained snapshot.
    dec_cdfs: std::sync::Arc<Cdfs>,
    /// Which of the two immutable decision snapshots this worker currently has.
    decision_cdf_warmed: bool,
    /// Adaptive all-zero price table `[cls][ctx][symbol]` for the SB being decided.
    pub(crate) skip_price: Option<Box<[[[f32; 2]; 13]; 5]>>,
    /// Per-slot copies of the decision CDF tables that price trial tokens;
    /// dropped on every decision-snapshot switch.
    trial_cost: std::cell::RefCell<Option<Vec<Vec<u16>>>>,
}

/// Decision memos. Interior-mutable so `&self` decision paths can fill them;
/// the epoch-stamped ones are invalidated by `emit_epoch`.
struct Memo {
    /// One-entry cache of the current superblock's activity, for the
    /// LOCAL masking reference (see `perceptual_rd_scale`). Blocks are coded
    /// in SB order so a single slot hits almost always. Key is (sb_x, sb_y).
    sb_act_cache: std::cell::Cell<(u32, u32, f32)>,
    /// Per-block memo of the IntraBC match search (`find_intrabc`).
    ibc_match_cache: std::cell::RefCell<HashMap<u64, IbcMatches>>,
    /// Memo for the partition proxy's palette candidate (rd_cost_square):
    /// keyed by packed (px, py, dim). The bottom-up partition search re-prices
    /// the same square up to ~3x; the palette candidate (histogram + weighted
    /// k-means + map + distortion) is the costliest per-call piece, and its
    /// inputs (source, quant, decision CDFs) are immutable per tile.
    pal_est_cache: std::cell::RefCell<HashMap<u64, [(f32, f32); 3]>>,
    /// Screen estimates additionally depend on neighbor palettes/modes and
    /// lambda. Keep their context separate so photo cache entries stay small.
    #[allow(clippy::type_complexity)]
    screen_pal_est_cache:
        std::cell::RefCell<HashMap<u64, (PaletteEstimateContext, [(f32, f32); 3])>>,
    /// Exact, epoch-stamped memo for derived chroma block RD costs shared by
    /// multiple partition families under the same perceptual multiplier.
    chroma_rd_cache: std::cell::RefCell<HashMap<u128, (u64, f32)>>,
    /// Memo for the full 16-level partition decision (`rd_choice_16`): the
    /// bottom-up pricing evaluates each 16-block up to ~3x (64-level refine,
    /// 32-level SPLIT leg, then decode_sb's own descend). Epoch stamping keeps
    /// reuse bit-exact after any emitted block changes prediction context.
    rd16_cache: std::cell::RefCell<HashMap<u32, (u64, Part16, f32, f32)>>,
    /// Epoch-stamped memo for the decision-stage rect16 leaf proxy
    #[allow(clippy::type_complexity)]
    rect_leaf_cache: std::cell::RefCell<HashMap<(u64, u32), (u64, f32, f32)>>,
    /// Exact 8x8-as-four-4x4 luma proxy, reused by the 16-level decision and
    /// the eventual 8-level decision within one reconstruction epoch.
    split4_rd_cache: std::cell::RefCell<HashMap<u128, (u64, f32)>>,
    /// Bumped on every emitted (or sink-captured) leaf block.
    emit_epoch: std::cell::Cell<u64>,
}

/// Trial-encode scratch entropy state (see `coder/trial.rs`), reused across nodes.
struct SbRun {
    /// Scratch semantic-token sink the trial legs encode into (lazily
    /// created, reused across nodes).
    trial_enc: Option<OdEcEncoder>,
    /// Slot-tagged CDF copy the trial legs key their tokens by.
    trial_cdfs: Option<Box<Cdfs>>,
}

/// Per-tile lossy encoder state, grouped by lifetime and mutability.
struct LossyTile<'a> {
    cfg: TileCfg<'a>,
    k: Kernels,
    ctx: CodingCtx,
    ent: Entropy,
    model: RateModel,
    memo: Memo,
    run: SbRun,
    /// Reusable trial buffers (see [`CoderScratch`]); `Rc<RefCell>` so the
    /// [`SBuf`] guards and the `&self` decision paths can reach the pool
    /// without borrowing the tile.
    scratch: std::rc::Rc<std::cell::RefCell<CoderScratch>>,
}

/// Outcome of a proxy partition decision at one node.
#[derive(Clone, Copy)]
struct PartVerdict {
    part: Part16,
    /// R-D cost of `part` (a placeholder in fixed-partition mode).
    cost: f32,
    /// Relative NONE-vs-SPLIT margin behind the choice: how close the two
    /// legs were. `INFINITY` when it was not measured (a leg was pruned,
    /// IntraBC won, or the level short-circuited); read by the trial gate.
    margin: f32,
}

impl PartVerdict {
    /// A verdict with no NONE-vs-SPLIT margin behind it.
    #[inline]
    fn unmeasured(part: Part16, cost: f32) -> Self {
        PartVerdict {
            part,
            cost,
            margin: f32::INFINITY,
        }
    }
}

/// Which luma distortion a mode pick measures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LumaDist {
    /// The SATD price of `luma_partition_distortion`.
    Satd,
    /// The decision weighted SSE, in proxy units.
    Wsse,
}

/// Transform type of a 4x4 luma transform in the split-4x4 leaf.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum TxSel {
    #[default]
    Dct,
    Adst,
    Idtx,
    /// 1-D hybrids (V_DCT / H_DCT).
    VDct,
    HDct,
}

/// The luma winner of a 16x16 leaf, as the search hands it to the chroma
/// search and the emitter.
#[derive(Clone, Copy)]
struct Luma16<'b> {
    /// Quantized luma coefficients (packing depends on `txtp16`).
    cf: &'b [i32; 256],
    /// Whole-block luma prediction.
    pred: &'b [i32; 256],
    y_mode: usize,
    /// All luma coefficients are zero.
    luma_zero: bool,
    /// Luma transform choice: 0 DCT_DCT, 1 ADST_ADST, 2 ADST_DCT, 3 DCT_ADST,
    /// 4 four TX_8X8 (`s8_txtps`), 5 IDTX, 6 sixteen TX_4X4 (`s16_txtps`).
    txtp16: u8,
    s8_txtps: [u8; 4],
    s16_txtps: [u8; 16],
    angle_delta: i32,
    filter_intra: Option<FilterIntraMode>,
    palette: Option<&'b LossyLumaPalette>,
    /// Decoder-exact luma reconstruction of a TX-split winner (`txtp16` 4 or
    /// 6), where `pred + inverse(cf)` does not apply.
    split_rec: Option<&'b [u16; 256]>,
    /// Top-right / bottom-left edge availability of the block.
    have_tr: bool,
    have_bl: bool,
}

/// The chroma winner of a square leaf: what the chroma search returns and the
/// emitter consumes. `C` holds one plane's coefficients, `P` one plane's
/// samples; both depend on the chroma format.
struct UvPick<C, P> {
    coeffs: [C; 2],
    /// Per-pixel prediction per plane (unused for plain DC where `dc` applies).
    pred: [P; 2],
    /// Inverse residual per plane, when the search already has it.
    resid: Option<[P; 2]>,
    /// Scalar DC predictor per plane.
    dc: [i32; 2],
    /// Chroma prediction mode (`DC_PRED` under CfL; see `cfl`).
    mode: usize,
    /// CfL alphas when chroma-from-luma won.
    cfl: Option<[i32; 2]>,
    /// Angle delta of a directional `mode`.
    delta: i32,
    palette: Option<LossyUvPalette>,
}

/// The owned result of the 16x16 luma search (see [`Luma16`] for the fields).
struct Luma16Pick {
    cf: SBuf<[i32; 256]>,
    pred: SBuf<[i32; 256]>,
    y_mode: usize,
    txtp16: u8,
    s8_txtps: [u8; 4],
    s16_txtps: [u8; 16],
    angle_delta: i32,
    filter_intra: Option<FilterIntraMode>,
    palette: Option<LossyLumaPalette>,
    split_rec: Option<[u16; 256]>,
}

/// The luma winner of a 32x32 leaf, as the search hands it to the chroma
/// search and the emitter.
#[derive(Clone, Copy)]
struct Luma32<'b> {
    /// Quantized luma coefficients (packing depends on the transform split).
    cf: &'b [i32; 1024],
    /// Whole-block luma prediction.
    pred: &'b [i32; 1024],
    y_mode: usize,
    /// All luma coefficients are zero.
    luma_zero: bool,
    angle_delta: i32,
    filter_intra: Option<FilterIntraMode>,
    palette: Option<&'b LossyLumaPalette>,
    /// Coded as four TX_16X16 (`tx_depth = 1`).
    tx_split: bool,
    /// Coded as sixteen TX_8X8 (`tx_depth = 2`) with these transform types.
    tx_d2: Option<[u8; 16]>,
    /// Decoder-exact luma reconstruction of a TX-split winner, where
    /// `pred + inverse(cf)` does not apply.
    split_rec: Option<&'b [u16; 1024]>,
    /// Top-right / bottom-left edge availability of the block.
    have_tr: bool,
    have_bl: bool,
}

/// The owned result of the 32x32 luma search (see [`Luma32`] for the fields).
struct Luma32Pick {
    cf: SBuf<[i32; 1024]>,
    pred: SBuf<[i32; 1024]>,
    y_mode: usize,
    angle_delta: i32,
    filter_intra: Option<FilterIntraMode>,
    palette: Option<LossyLumaPalette>,
    tx_split: bool,
    tx_d2: Option<[u8; 16]>,
    split_rec: Option<[u16; 1024]>,
}

/// Everything the 8x8 leaf search hands to its emitter. One struct for all
/// chroma formats; only the buffers of the tile's format are meaningful.
struct Block8Pick {
    /// Luma winner.
    best_mode: usize,
    best_delta: i32,
    best_filter_intra: Option<FilterIntraMode>,
    best_palette: Option<LossyLumaPalette>,
    lcf: [i32; 64],
    lpred_arr: [i32; 64],
    /// Luma transform: one of the type flags, or four TX_4X4 (`s4_txtps`) with
    /// their decoder-exact reconstruction `s4_rec`.
    best_is_adst: bool,
    best_is_idtx: bool,
    best_is_adstdct: bool,
    best_is_dctadst: bool,
    best_is_vdct: bool,
    best_is_hdct: bool,
    best_is_txsplit4: bool,
    s4_txtps: [u8; 4],
    s4_rec: [u16; 64],
    /// Chroma winner: CfL, or the per-format mode (`chosen_uv_block` is the
    /// fallback the 4:2:0 / 4:2:2 choices override).
    use_cfl: bool,
    cfl_alpha_uv: [i32; 2],
    chosen_uv_block: usize,
    chosen_uv_444: usize,
    chosen_uv_420: usize,
    chosen_uv_422: usize,
    uv_delta_8: i32,
    uv_pal8: Option<LossyUvPalette>,
    cpred: [i32; 2],
    /// 4:4:4 chroma (8x8): coefficients and per-pixel predictions.
    ccf8: [[i32; 64]; 2],
    cpred444: [[i32; 64]; 2],
    paeth_pred444: [[i32; 64]; 2],
    /// 4:2:0 chroma (4x4).
    ccf44: [[i32; 16]; 2],
    cpred420: [[i32; 16]; 2],
    paeth_pred420: [[i32; 16]; 2],
    sv_preds_420: [[i32; 16]; 2],
    /// 4:2:2 chroma (4x8).
    ccf48: [[i32; 32]; 2],
    cpred422: [[i32; 32]; 2],
    paeth_pred422: [[i32; 32]; 2],
    /// Skip flag, from the coefficients as they stood before the chroma mode
    /// searches (the block header's `skip` precedes them in coding order).
    block_skip: bool,
}

/// A 64x64 leaf after its decisions and reconstruction: what remains to be
/// written to the bitstream.
struct Block64Coded {
    block_skip: bool,
    y_mode: usize,
    y_delta: i32,
    uv_mode: usize,
    palette: Option<LossyLumaPalette>,
    tx64: bool,
    grid16: bool,
    lcf: [SBuf<[i32; 1024]>; 4],
    uflat: SBuf<[i32; 4096]>,
    vflat: SBuf<[i32; 4096]>,
}

/// One 64x32 / 32x64 leaf after its decisions and reconstruction: what
/// remains to be written to the bitstream.
struct Rect64Coded {
    block_skip: bool,
    y_mode: usize,
    y_delta: i32,
    uv_mode: usize,
    palette: Option<LossyLumaPalette>,
    lcf: [[i32; 1024]; 2],
    uv: [SBuf<[i32; 4096]>; 2],
}

/// Search result of a 16x8 / 8x16 leaf at 4:2:0, for its emitter.
struct Rect16Pick420 {
    block_skip: bool,
    split: bool,
    y_mode: usize,
    y_delta: i32,
    use_cfl: bool,
    uv_delta_rect: i32,
    chosen_uv: usize,
    cfl_alpha: [i32; 2],
    rpal: Option<LossyLumaPalette>,
    rfi: Option<FilterIntraMode>,
    split_cf: [[i32; 64]; 2],
    ltxtp: usize,
    lcf: [i32; 128],
    tx_mode: usize,
    lpred_arr: [i32; 128],
    ccf: [[i32; 32]; 2],
    uv_tx: ChromaTx,
    cfl_px: [[i32; 32]; 2],
    uv_pred: [[i32; 32]; 2],
    cpred: [i32; 2],
}

/// Search result of one 16x4 / 4x16 strip, for its emitter.
struct Strip16Pick {
    block_skip: bool,
    y_mode: usize,
    y_delta: i32,
    chroma: Option<(usize, usize, usize, usize)>,
    chosen_uv: usize,
    uv_delta_strip: i32,
    lcf: [i32; 64],
    lpred_arr: [i32; 64],
    ccf: [[i32; 64]; 2],
    uv_tx: ChromaTx,
    uv_pred: [[i32; 32]; 2],
    cpred: [i32; 2],
}

/// Search result of one 8x16 half of a 16x16 VERT partition (4:4:4), for its emitter.
struct Rect16PickV444 {
    block_skip: bool,
    split: bool,
    y_mode: usize,
    yctx: usize,
    y_delta: i32,
    chosen_uv: usize,
    cfl_opt: Option<[i32; 2]>,
    rpal: Option<LossyLumaPalette>,
    rfi: Option<FilterIntraMode>,
    split_cf: [[i32; 64]; 2],
    ltxtp: usize,
    lcf: [i32; 128],
    tx_mode: usize,
    lpred_arr: [i32; 128],
    ccf: [[i32; 128]; 2],
    uv_pred: Option<[[i32; 128]; 2]>,
    use_cfl: bool,
    cfl_pred: [[i32; 128]; 2],
    cpred: [i32; 2],
}

/// Search result of one 16x8 half of a 16x16 HORZ partition (4:2:2), for its emitter.
struct Rect16PickH422 {
    block_skip: bool,
    split: bool,
    y_mode: usize,
    yctx: usize,
    y_delta: i32,
    use_cfl: bool,
    cfl_alpha: [i32; 2],
    rpal: Option<LossyLumaPalette>,
    rfi: Option<FilterIntraMode>,
    split_cf: [[i32; 64]; 2],
    ltxtp: usize,
    lcf: [i32; 128],
    tx_mode: usize,
    lpred_arr: [i32; 128],
    cbx4: usize,
    cby4: usize,
    ccf: [[i32; 64]; 2],
    cy: usize,
    cx: usize,
    cfl_px: [[i32; 64]; 2],
    cpred: [i32; 2],
}

/// Search result of one 16x8 half of a 16x16 HORZ partition (4:4:4), for its emitter.
struct Rect16PickH444 {
    block_skip: bool,
    split: bool,
    y_mode: usize,
    yctx: usize,
    y_delta: i32,
    chosen_uv: usize,
    cfl_opt: Option<[i32; 2]>,
    rpal: Option<LossyLumaPalette>,
    rfi: Option<FilterIntraMode>,
    split_cf: [[i32; 64]; 2],
    ltxtp: usize,
    lcf: [i32; 128],
    tx_mode: usize,
    lpred_arr: [i32; 128],
    ccf: [[i32; 128]; 2],
    uv_pred: Option<[[i32; 128]; 2]>,
    use_cfl: bool,
    cfl_pred: [[i32; 128]; 2],
    cpred: [i32; 2],
}

/// Search result of one 8x4 / 4x8 leaf, for its emitter.
struct Rect8Pick {
    bx4: usize,
    by4: usize,
    block_skip: bool,
    best_mode: usize,
    has_chroma: bool,
    lcf: [i32; 32],
    lpred_arr: [i32; 32],
    cw: usize,
    ch: usize,
    cbx4: usize,
    cby4: usize,
    ccf: [[i32; 64]; 2],
    cy: usize,
    cx: usize,
    cpred: [i32; 2],
}

/// Search result of one 4x4 block of an 8x8 SPLIT, for its emitter.
struct Sub4Pick {
    block_skip: bool,
    best_mode: usize,
    has_chroma: bool,
    use_cfl: bool,
    cfl_alpha_uv: [i32; 2],
    tx4: TxSel,
    lcf: [i32; 16],
    lpred: [i32; 16],
    chx: usize,
    chy: usize,
    ccf: [[i32; 16]; 2],
    cstride: usize,
    cpred_px: [[i32; 16]; 2],
}
