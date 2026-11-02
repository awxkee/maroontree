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

use super::super::*;
use crate::av2::cfl::cfl_partition_prediction;
use crate::av2::video::{mc, me, mv::Mv};

const INTER_BORDER_420: usize = 72;
use crate::av2::leaf::{MIDX_SMOOTH, midx_class};
const ENABLE_DENSE_INTER_32: bool = true;
const ENABLE_DENSE_INTER_16: bool = true;

/// Compile each expensive leaf geometry as its own optimization unit. The
/// boundary is once per coded leaf, outside transform and prediction kernels.
#[inline(never)]
fn outline_leaf_420<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(test)]
pub(crate) static INTER_SKIP_32_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Total coded leaves and intra-coded leaves on the current frame (mode-mix diag).
#[cfg(test)]
pub(crate) static TOTAL_LEAF_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTRA_LEAF_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Whole-64 GLOBALMV-skip blocks committed on reference rank 1 (partition walk).
#[cfg(test)]
pub(crate) static PARTITION_SKIP_RANK1_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_SKIP_RECT_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_MOTION_SKIP_RECT_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_NEWMV_SKIP_32_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_NEARMV_SKIP_32_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_NEWMV_SKIP_16_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_NEARMV_SKIP_16_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_RESIDUAL_64_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_RESIDUAL_32_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_RESIDUAL_32_HIGH_EOB_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_RESIDUAL_16_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_RESIDUAL_16_HIGH_EOB_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static INTER_RESIDUAL_16_CHROMA_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// DRL[0] spatial predictor used by the current single-LAST inter path.
///
/// The decoder scans left, above, then above-right for the first usable MV.
/// Keeping this in one helper makes the motion-search reference, emitted MVD,
/// and stored encoder reconstruction use the same predictor.
#[inline]
fn drl0_mv(left: Option<Mv>, above: Option<Mv>, above_right: Option<Mv>) -> Mv {
    left.or(above).or(above_right).unwrap_or(Mv::ZERO)
}

/// Return the inter mode and quarter-pel MVD for an eighth-pel absolute MV.
/// `NEARMV` carries no delta when the searched MV equals DRL[0]; otherwise
/// `NEWMV` codes `mv - predictor`, never the absolute MV.
#[inline]
fn inter_mode_qtr_mvd(mv: Mv, predictor: Mv) -> (usize, i32, i32) {
    let delta = mv.diff(predictor);
    debug_assert_eq!(
        delta.row & 1,
        0,
        "QTR-pel MVD row must be even in eighth-pel units"
    );
    debug_assert_eq!(
        delta.col & 1,
        0,
        "QTR-pel MVD col must be even in eighth-pel units"
    );
    let mode = if delta == Mv::ZERO { 0 } else { 2 };
    (mode, delta.row / 2, delta.col / 2)
}

/// Copy a `w × h` reconstruction rectangle at `(y, x)` out of a plane (row-major).
/// Used to snapshot a walk leaf's luma reconstruction for the decouple record.
fn gather_rect(recy: &[f32], pw: usize, y: usize, x: usize, w: usize, h: usize) -> Vec<f32> {
    let mut v = vec![0f32; w * h];
    for (dst, src) in v.chunks_exact_mut(w).zip(rect_rows(recy, pw, y, x, w, h)) {
        dst.copy_from_slice(src);
    }
    v
}

#[inline]
fn refill_coeffs(dst: &mut Vec<Coeff>, levels: &[f32]) {
    dst.clear();
    dst.extend(
        levels
            .iter()
            .enumerate()
            .filter(|&(_, level)| *level != 0.0)
            .map(|(scan_pos, &level)| (scan_pos, level as i32)),
    );
}

/// One superblock's parallel-decide output for 4:2:0: the captured record plus the
/// deblock TU rectangles it recorded. Merged in raster order after the wavefront so
/// the serial Replay consumes the exact sequence a serial Capture would produce.
struct WfSlot420 {
    record: replay::DecisionRecord,
    tx: Vec<(usize, usize, usize, usize)>,
    ch: Vec<(usize, usize, usize, usize)>,
}

/// Per-worker reusable scratch for the 4:2:0 wavefront decide: a private full-res
/// luma recon (`ry`, `pw×ph`) + two QUARTERED chroma recons (`ru`/`rv`,
/// `pcw×pch`, half-width AND half-height) plus the emit-side neighbor context
/// arrays (entropy + chroma-presence + CfL + inter-state). Each cell sees the
/// frame-initial context values, reset per cell so the result never depends on
/// which unrelated cell a worker handled previously (the 4:4:4 race fix).
#[derive(Default)]
struct WfScratch420 {
    ry: Vec<f32>,
    ru: Vec<f32>,
    rv: Vec<f32>,
    above: Vec<u8>,
    left: Vec<u8>,
    apctx: Vec<u8>,
    ua: Vec<i32>,
    va: Vec<i32>,
    ul: Vec<i32>,
    vl: Vec<i32>,
    cfa: Vec<i32>,
    cfl: Vec<i32>,
    mia: Vec<i32>,
    mil: Vec<i32>,
    imia: Vec<u8>,
    imil: Vec<u8>,
    skmia: Vec<u8>,
    skmil: Vec<u8>,
    nmvia: Vec<u8>,
    nmvil: Vec<u8>,
    mvia: Vec<Option<Mv>>,
    mvil: Vec<Option<Mv>>,
    nmva: Vec<u8>,
    mva: Vec<Option<Mv>>,
    ska: Vec<u8>,
    ina: Vec<u8>,
    rfa: Vec<u8>,   // SB-granular reference rank (above)
    rfmia: Vec<u8>, // mi-granular reference rank (above)
    rfmil: Vec<u8>, // mi-granular reference rank (left)
    dbq: Vec<u16>,
    me: me::MeScratch<f32>,
    inter_pred: InterPredScratch420,
}

/// Per-worker MC output storage for the largest Phase-2 luma/chroma leaf.
/// Resizing retains capacity, so repeated block candidates do not allocate or
/// clear a full 64x64 prediction on the stack.
#[derive(Default)]
struct InterPredScratch420 {
    y: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    tx_pred: Vec<f32>,
    residual: Vec<f32>,
    chroma_pred: Vec<i32>,
    convolve_tmp: Vec<i32>,
    luma_coeffs: [Vec<Coeff>; 4],
    chroma_coeffs: [Vec<Coeff>; 2],
}

struct WholeSbInterScratch<'a> {
    y: &'a mut [f32],
    u: &'a mut [f32],
    v: &'a mut [f32],
    tx_pred: &'a mut [f32],
    residual: &'a mut [f32],
    chroma_pred: &'a mut [i32],
    convolve_tmp: &'a mut Vec<i32>,
    luma_coeffs: &'a mut [Vec<Coeff>; 4],
    chroma_coeffs: &'a mut [Vec<Coeff>; 2],
}

struct InterLeafSearch420<'a> {
    enc: &'a mut RangeEncoder,
    source_y: &'a [f32],
    source_u: &'a [f32],
    source_v: &'a [f32],
    luma_reference: &'a (Vec<f32>, usize, usize),
    chroma_references: &'a [(Vec<f32>, usize); 2],
    me_scratch: &'a mut me::MeScratch<f32>,
    prediction_scratch: &'a mut InterPredScratch420,
    source_stride: usize,
    chroma_stride: usize,
    reference_x: usize,
    reference_y: usize,
    reference_luma_stride: usize,
    block_x: usize,
    block_y: usize,
    block_width: usize,
    block_height: usize,
    predictor_mv: Mv,
    frame_mv_seed: Mv,
    skip_ctx: usize,
    mode_ctx: usize,
    qstep: i32,
}

#[derive(Clone, Copy)]
struct InterLeafCandidate420 {
    mv: Mv,
    mode: usize,
    mvd: Option<Mv>,
    rd_cost: f32,
}

struct DenseInterResidualInput420<'a> {
    enc: &'a RangeEncoder,
    source_y: &'a [f32],
    source_u: &'a [f32],
    source_v: &'a [f32],
    prediction_y: &'a [f32],
    prediction_u: &'a [f32],
    prediction_v: &'a [f32],
    source_stride: usize,
    chroma_stride: usize,
    block_x: usize,
    block_y: usize,
    block_size: usize,
    residual_scale: f32,
    qstep: i32,
    skip_ctx: usize,
    mode_ctx: usize,
    inter_mode: usize,
    mvd: Option<Mv>,
}

struct DenseInterResidual420 {
    rd_cost: f32,
    y_coeffs: Vec<Coeff>,
    u_coeffs: Vec<Coeff>,
    v_coeffs: Vec<Coeff>,
    y_recon: [f32; 32 * 32],
    u_recon: Vec<f32>,
    v_recon: Vec<f32>,
}

impl InterPredScratch420 {
    fn planes(
        &mut self,
        luma_len: usize,
        chroma_len: usize,
    ) -> (&mut [f32], &mut [f32], &mut [f32], &mut Vec<i32>) {
        self.y.resize(luma_len, 0.0);
        self.u.resize(chroma_len, 0.0);
        self.v.resize(chroma_len, 0.0);
        (
            &mut self.y,
            &mut self.u,
            &mut self.v,
            &mut self.convolve_tmp,
        )
    }

    fn whole_sb(&mut self) -> WholeSbInterScratch<'_> {
        self.y.resize(64 * 64, 0.0);
        self.u.resize(32 * 32, 0.0);
        self.v.resize(32 * 32, 0.0);
        self.tx_pred.resize(32 * 32, 0.0);
        self.residual.resize(32 * 32, 0.0);
        self.chroma_pred.resize(32 * 32, 0);
        for coeffs in self
            .luma_coeffs
            .iter_mut()
            .chain(self.chroma_coeffs.iter_mut())
        {
            coeffs.clear();
            if coeffs.capacity() < 32 * 32 {
                coeffs.reserve(32 * 32 - coeffs.capacity());
            }
        }
        WholeSbInterScratch {
            y: &mut self.y,
            u: &mut self.u,
            v: &mut self.v,
            tx_pred: &mut self.tx_pred,
            residual: &mut self.residual,
            chroma_pred: &mut self.chroma_pred,
            convolve_tmp: &mut self.convolve_tmp,
            luma_coeffs: &mut self.luma_coeffs,
            chroma_coeffs: &mut self.chroma_coeffs,
        }
    }
}

#[derive(Clone, Copy)]
struct WfScratch420Shape {
    pw: usize,
    ph: usize,
    pcw: usize,
    pch: usize,
    tmc: i64,
    tmr: i64,
    sb_cols: usize,
    sb_rows: usize,
}

struct Sb420Decision<'a, 'record> {
    enc: &'a mut RangeEncoder,
    aqs: &'a mut aq::AqState,
    decide_mode: &'a mut replay::DecideMode<'record>,
    recy: &'a mut [f32],
    recu: &'a mut [f32],
    recv: &'a mut [f32],
    yp: &'a [f32],
    up: &'a [f32],
    vp: &'a [f32],
    above: &'a mut [u8],
    left: &'a mut [u8],
    above_pctx: &'a mut [u8],
    left_pctx: &'a mut [u8],
    u_above: &'a mut [i32],
    v_above: &'a mut [i32],
    u_left: &'a mut [i32],
    v_left: &'a mut [i32],
    cfl_above: &'a mut [i32],
    cfl_left: &'a mut [i32],
    /// Luma directional `midx` of the coded neighbours per mi (-1 = none):
    /// avm's above_right / bottom_left mode-set neighbours.
    midx_above: &'a mut [i32],
    midx_left: &'a mut [i32],
    inter_mi_above: &'a mut [u8],
    inter_mi_left: &'a mut [u8],
    skip_mi_above: &'a mut [u8],
    skip_mi_left: &'a mut [u8],
    newmv_mi_above: &'a mut [u8],
    newmv_mi_left: &'a mut [u8],
    mv_mi_above: &'a mut [Option<Mv>],
    mv_mi_left: &'a mut [Option<Mv>],
    newmv_above: &'a mut [u8],
    mv_above: &'a mut [Option<Mv>],
    skip_above: &'a mut [u8],
    inter_above: &'a mut [u8],
    // Reference rank grids for two-reference frames (meaningful where the inter
    // grid is 1). SB-granular for the whole-64 skip mode ctx; mi-granular for
    // the per-block single_ref bit context read by every leaf.
    ref_above: &'a mut [u8],
    ref_mi_above: &'a mut [u8],
    ref_mi_left: &'a mut [u8],
    tx_leaves: &'a mut Vec<(usize, usize, usize, usize)>,
    skip_leaves: &'a mut Vec<(usize, usize, usize, usize)>,
    /// SDP chroma-tree leaves (luma-mi rects) for the chroma deblock grid.
    chroma_leaves: &'a mut Vec<(usize, usize, usize, usize)>,
    sb_qidx: &'a mut [u16],
    last_ref: &'a [Vec<f32>],
    ref_x0: usize,
    ref_y0: usize,
    ref_ls: usize,
    ref_cs: usize,
    has_last: bool,
    // Rank-1 reference planes (raw f32 Y,U,V) for the zero-motion skip mode on
    // two-reference frames; empty when the frame lists a single reference.
    second_ref: &'a [Vec<f32>],
    has_second: bool,
    inter_luma: Option<&'a (Vec<f32>, usize, usize)>,
    inter_chroma: Option<&'a [(Vec<f32>, usize); 2]>,
    me_scratch: &'a mut me::MeScratch<f32>,
    inter_pred_scratch: &'a mut InterPredScratch420,
    frame_mv_seed: Mv,
    mhccp_bounds: cfl::MhccpBounds,
    pw: usize,
    pcw: usize,
    width: usize,
    height: usize,
    sb_cols: usize,
    tmc: i64,
    tmr: i64,
    qc: usize,
    neutral: f32,
    rdoq_lambda: f32,
    aq_grid: &'a [aq::AqCell],
    use_grid: bool,
    skip_left: u8,
    inter_left: u8,
    newmv_left: u8,
    mv_left: Option<Mv>,
    ref_left: u8,
    row: usize,
    col: usize,
    /// Encode with this partition instead of probing (an SB partition trial).
    forced_part: Option<LumaPartitionDecision>,
}

impl WfScratch420 {
    fn ensure(&mut self, shape: WfScratch420Shape) {
        let WfScratch420Shape {
            pw,
            ph,
            pcw,
            pch,
            tmc,
            tmr,
            sb_cols,
            sb_rows,
        } = shape;
        let ny = pw * ph;
        if self.ry.len() != ny {
            self.ry = vec![0f32; ny];
        }
        let nc = pcw * pch + 1;
        if self.ru.len() != nc {
            self.ru = vec![0f32; nc];
            self.rv = vec![0f32; nc];
        }
        let (mc, mr) = (tmc as usize + 16, tmr as usize + 16);
        self.above.resize(pw / 4 + 16, 0x40);
        self.left.resize(ph / 4 + 16, 0x40);
        self.apctx.resize(mc, 0);
        self.ua.resize(mc, 0);
        self.va.resize(mc, 0);
        self.ul.resize(mr, 0);
        self.vl.resize(mr, 0);
        self.cfa.resize(mc, 0);
        self.cfl.resize(mr, 0);
        self.mia.resize(mc, -1);
        self.mil.resize(mr, -1);
        self.imia.resize(mc, 0);
        self.imil.resize(mr, 0);
        self.skmia.resize(mc, 0);
        self.skmil.resize(mr, 0);
        self.nmvia.resize(mc, 0);
        self.nmvil.resize(mr, 0);
        self.mvia.resize(mc, None);
        self.mvil.resize(mr, None);
        self.nmva.resize(sb_cols.max(1), 0);
        self.mva.resize(sb_cols.max(1), None);
        self.ska.resize(sb_cols.max(1), 0);
        self.ina.resize(sb_cols.max(1), 0);
        self.rfa.resize(sb_cols.max(1), 0);
        self.rfmia.resize(mc, 0);
        self.rfmil.resize(mr, 0);
        self.dbq.resize(sb_cols * sb_rows, 0);
    }

    /// Restore the exact frame-initial context state before every independently
    /// decided SB (thread-pool assignment is unstable, so nothing may carry over).
    fn reset_contexts(&mut self) {
        self.above.fill(0x40);
        self.left.fill(0x40);
        self.apctx.fill(0);
        self.ua.fill(0);
        self.va.fill(0);
        self.ul.fill(0);
        self.vl.fill(0);
        self.cfa.fill(0);
        self.cfl.fill(0);
        self.mia.fill(-1);
        self.mil.fill(-1);
        self.imia.fill(0);
        self.imil.fill(0);
        self.skmia.fill(0);
        self.skmil.fill(0);
        self.nmvia.fill(0);
        self.nmvil.fill(0);
        self.mvia.fill(None);
        self.mvil.fill(None);
        self.nmva.fill(0);
        self.mva.fill(None);
        self.ska.fill(0);
        self.ina.fill(0);
        self.rfa.fill(0);
        self.rfmia.fill(0);
        self.rfmil.fill(0);
        self.dbq.fill(0);
    }
}

thread_local! {
    static WF_SCRATCH_420: std::cell::RefCell<WfScratch420> =
        std::cell::RefCell::new(WfScratch420::default());
}

/// A rect luma leaf coded as square TUs along its long side (see
/// [`Av2Encoder::multi_tu_search`]).
struct RectSplit {
    tus: Vec<Vec<Coeff>>,
    tx: Vec<usize>,
    mode: (usize, i8),
    /// Pixel SSE + lambda * (coefficient + mode + tx-type bits).
    cost: f32,
    recon: Vec<Vec<f32>>,
}

/// Geometry of a [`Av2Encoder::multi_tu_search`] leaf.
struct RectSplitBlock<'a> {
    yp: &'a [f32],
    pw: usize,
    y: usize,
    x: usize,
    w: usize,
    h: usize,
    frame: (usize, usize),
    coded: &'a [u8; 256],
    smooth_neighbour: bool,
}

impl RectSplitBlock<'_> {
    /// TU side (the block's short side, at most 32) and the TU origins in coding
    /// (raster) order.
    fn tus(&self) -> (usize, Vec<(usize, usize)>) {
        let s = self.w.min(self.h).min(32);
        let mut pos = Vec::new();
        for ty in (0..self.h).step_by(s) {
            for tx in (0..self.w).step_by(s) {
                pos.push((self.y + ty, self.x + tx));
            }
        }
        (s, pos)
    }
}

impl Av2Encoder {
    /// RD search of a 2:1 / 4:1 rect leaf as square TUs along its long side (TX
    /// partition VERT / HORZ / VERT4 / HORZ4): one luma mode for the block, each TU
    /// predicted as a square from the coded-mi mask. 16x16 TUs search every mode
    /// (DC, SMOOTH*, PAETH, SATD-best directional pairs) and their four TX_16X16
    /// types; 8x8 TUs the non-directional modes with DCT. Leaves `recy` as found.
    fn multi_tu_search(
        &self,
        recy: &mut [f32],
        b: &RectSplitBlock<'_>,
        qstep: i32,
        lambda: f32,
    ) -> Option<RectSplit> {
        self.leaf_eval(recy, b, qstep, lambda, false)
    }

    /// [`Self::multi_tu_search`] over any leaf shape (squares too: one TU, or four
    /// TX_32X32 for 64x64; TX_32X32 TUs DCT only), optionally with the zero-residual
    /// palette candidate. The partition tree search prices every shape with it.
    fn leaf_eval(
        &self,
        recy: &mut [f32],
        b: &RectSplitBlock<'_>,
        qstep: i32,
        lambda: f32,
        with_palette: bool,
    ) -> Option<RectSplit> {
        let bases = &self.bases;
        let bd = self.bit_depth as i32;
        let neutral = self.dc_neutral();
        let pw = b.pw;
        let (sz, pos) = b.tus();
        let n = pos.len();
        let area = sz * sz;
        let src: Vec<Vec<f32>> = pos
            .iter()
            .map(|&(y, x)| {
                rect_rows(b.yp, pw, y, x, sz, sz)
                    .flatten()
                    .copied()
                    .collect()
            })
            .collect();
        let saved: Vec<Vec<f32>> = pos
            .iter()
            .map(|&(y, x)| {
                rect_rows(recy, pw, y, x, sz, sz)
                    .flatten()
                    .copied()
                    .collect()
            })
            .collect();
        let predict = |recy: &[f32], k: usize, (m, d): (usize, i32)| -> Vec<f32> {
            let (y, x) = pos[k];
            if m == 0 {
                vec![dc_pred_rect(recy, pw, y, x, sz, sz, neutral, bd); area]
            } else {
                leaf::predict_luma_square(
                    recy,
                    &leaf::SquareLumaBlock {
                        stride: pw,
                        frame_width: b.frame.0,
                        frame_height: b.frame.1,
                        y,
                        x,
                        size: sz,
                    },
                    b.coded,
                    leaf::SquareLumaMode {
                        mode: m,
                        angle_delta: d,
                        smooth_neighbour: b.smooth_neighbour,
                    },
                    neutral,
                    bd,
                )
            }
        };
        let mode_bits = |m: usize| -> f32 {
            if m >= 5 {
                crate::av2::rd::knob_or(if sz == 8 { "MBDIR8" } else { "MBDIR" }, 9.0)
            } else if m == 0 {
                0.0
            } else {
                crate::av2::rd::knob_or("MB16", 6.0)
            }
        };
        let mut cands: Vec<(usize, i32)> = (0..=4usize).map(|m| (m, 0)).collect();
        let dir_size = sz >= 16 || crate::av2::rd::knob_or("GDIR8", 1.0) > 0.0;
        if dir_size && self.speed.try_directional() {
            cands.extend(leaf::screen_directional(
                crate::av2::rd::knob_or("DIR16", 3.0) as usize,
                self.speed.try_angle_deltas(),
                crate::av2::rd::knob_or("GDREF", 8.0) as usize,
                |m, d| {
                    (0..n)
                        .map(|k| {
                            crate::av2::metrics::satd_f32(
                                &src[k],
                                sz,
                                &predict(recy, k, (m, d)),
                                sz,
                                sz,
                                sz,
                            )
                        })
                        .sum()
                },
            ));
        }
        let sse = |s: &[f32], r: &[f32]| -> f32 {
            s.iter()
                .zip(r)
                .map(|(&a, &r)| (a - r.round()).powi(2))
                .sum()
        };
        // Two-stage: rank every mode by a cheap DCT-only, no-trellis RD cost (TUs
        // chained through their DCT recon), then only the best few get the full
        // trellis + tx-type evaluation below.
        let topk = crate::av2::rd::knob_or("GTOPK", 0.0) as usize;
        if topk > 0 && cands.len() > topk {
            let (mut fast, scan): (crate::av2::proj::Basis, &[u16]) = match sz {
                32 => (bases.luma.clone(), &tables::SCAN),
                16 => (bases.luma16x16.clone(), &SCAN16),
                _ => (bases.c8x8.clone(), &SCAN8X8),
            };
            fast.rdoq = None;
            let scale = fast.qstep as f32 / qstep as f32;
            let satd_stage = crate::av2::rd::knob_or("GST1", 1.0) > 0.0;
            let lam_satd = lambda.sqrt() * crate::av2::rd::knob_or("GSL", 1.0);
            let mut scored: Vec<(f32, (usize, i32))> = cands
                .iter()
                .map(|&md| {
                    if satd_stage {
                        let satd: u64 = (0..n)
                            .map(|k| {
                                crate::av2::metrics::satd_f32(
                                    &src[k],
                                    sz,
                                    &predict(recy, k, md),
                                    sz,
                                    sz,
                                    sz,
                                )
                            })
                            .sum();
                        return (satd as f32 + lam_satd * mode_bits(md.0), md);
                    }
                    let mut cost = lambda * mode_bits(md.0);
                    for k in 0..n {
                        let pred = predict(recy, k, md);
                        let resid: Vec<f32> = src[k]
                            .iter()
                            .zip(&pred)
                            .map(|(&s, &p)| (s - p) * scale)
                            .collect();
                        let lev = fast.project_scan(&resid, 0.0, scan);
                        let p_i: Vec<i32> = pred.iter().map(|&v| v as i32).collect();
                        let rec =
                            itx422::reconstruct_chroma_pred(&p_i, &lev, qstep, scan, sz, sz, bd);
                        cost += sse(&src[k], &rec) + lambda * coeff_rate_f32(&lev);
                        if k + 1 < n {
                            let (y, x) = pos[k];
                            put_block_rect(recy, pw, y, x, sz, sz, &rec);
                        }
                    }
                    for (k, &(y, x)) in pos.iter().enumerate().take(n.saturating_sub(1)) {
                        put_block_rect(recy, pw, y, x, sz, sz, &saved[k]);
                    }
                    (cost, md)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            cands = scored.into_iter().take(topk).map(|(_, md)| md).collect();
        }
        let mut best: Option<RectSplit> = None;
        for md in cands {
            let mut cost = lambda * mode_bits(md.0);
            let mut tus = Vec::with_capacity(n);
            let mut tx = Vec::with_capacity(n);
            let mut recon = Vec::with_capacity(n);
            for k in 0..n {
                let pred = predict(recy, k, md);
                let (j, lev, rec, t) = if sz == 32 {
                    let scale = bases.luma.qstep as f32 / qstep as f32;
                    let resid: Vec<f32> = src[k]
                        .iter()
                        .zip(&pred)
                        .map(|(&s, &p)| (s - p) * scale)
                        .collect();
                    let lev = bases.luma.project_scan(&resid, 0.0, &tables::SCAN);
                    let rec = itx422::reconstruct_luma(&pred, &lev, qstep, &tables::SCAN, bd);
                    (
                        sse(&src[k], &rec) + lambda * coeff_rate_f32(&lev),
                        lev,
                        rec.to_vec(),
                        0,
                    )
                } else if sz == 16 {
                    let scale = bases.luma16x16.qstep as f32 / qstep as f32;
                    let resid: Vec<f32> = src[k]
                        .iter()
                        .zip(&pred)
                        .map(|(&s, &p)| (s - p) * scale)
                        .collect();
                    let mut src16 = [0f32; 256];
                    src16.copy_from_slice(&src[k]);
                    // (type-selection cost, SSE-based cost, levels, recon, tx type)
                    type Tx16Pick = (f32, f32, Vec<f32>, Vec<f32>, usize);
                    let mut tb: Option<Tx16Pick> = None;
                    for (t, &delta) in TX16_TYPE_RATE_DELTA.iter().enumerate() {
                        let (basis, flips) = match t {
                            0 => (&bases.luma16x16, (false, false)),
                            1 => (&bases.luma16x16_adst, (true, true)),
                            2 => (&bases.luma16x16_adst_dct, (false, true)),
                            _ => (&bases.luma16x16_dct_adst, (true, false)),
                        };
                        let lev = basis.project_scan(&resid, 0.0, &SCAN16);
                        if t > 0 && tx16_dc_only(&lev) {
                            continue;
                        }
                        let rec = if t == 0 {
                            itx422::reconstruct_luma16(&pred, &lev, qstep, &SCAN16, bd)
                        } else {
                            itx422::reconstruct_luma16_adst(
                                &pred, &lev, qstep, &SCAN16, flips.0, flips.1, bd,
                            )
                        };
                        let rate = coeff_rate_f32(&lev) + delta;
                        let jt = tx16_distortion(&src16, &rec) + lambda * rate;
                        if tb.as_ref().is_none_or(|t0| jt < t0.0) {
                            tb = Some((
                                jt,
                                sse(&src[k], &rec) + lambda * rate,
                                lev,
                                rec.to_vec(),
                                t,
                            ));
                        }
                    }
                    let (_, j, lev, rec, t) = tb.expect("DCT_DCT is always a candidate");
                    (j, lev, rec, t)
                } else {
                    let scale = bases.c8x8.qstep as f32 / qstep as f32;
                    let resid: Vec<f32> = src[k]
                        .iter()
                        .zip(&pred)
                        .map(|(&s, &p)| (s - p) * scale)
                        .collect();
                    let lev = bases.c8x8.project_scan(&resid, 0.0, &SCAN8X8);
                    let p_i: Vec<i32> = pred.iter().map(|&v| v as i32).collect();
                    let rec =
                        itx422::reconstruct_chroma_pred(&p_i, &lev, qstep, &SCAN8X8, 8, 8, bd);
                    (
                        sse(&src[k], &rec) + lambda * coeff_rate_f32(&lev),
                        lev,
                        rec,
                        0,
                    )
                };
                cost += j;
                let (y, x) = pos[k];
                put_block_rect(recy, pw, y, x, sz, sz, &rec);
                tus.push(levels_to_coeffs(&lev));
                tx.push(t);
                recon.push(rec);
            }
            for (k, &(y, x)) in pos.iter().enumerate() {
                put_block_rect(recy, pw, y, x, sz, sz, &saved[k]);
            }
            if best.as_ref().is_none_or(|bst| cost < bst.cost) {
                best = Some(RectSplit {
                    tus,
                    tx,
                    mode: (md.0, md.1 as i8),
                    cost,
                    recon,
                });
            }
        }
        // Screen content: the zero-residual palette of a square >= 16x16.
        if with_palette && b.w == b.h && b.w >= 16 && crate::av2::palette::enabled() {
            let blk: Vec<f32> = rect_rows(b.yp, pw, b.y, b.x, b.w, b.h)
                .flatten()
                .copied()
                .collect();
            if let Some(pal) = crate::av2::palette::search(&blk, b.w, b.h, bd as u32, lambda) {
                let pred = pal.pred();
                let cost = sse(&blk, &pred) + lambda * pal.bits;
                if best.as_ref().is_none_or(|bst| cost < bst.cost) {
                    let recon = pos
                        .iter()
                        .map(|&(y, x)| {
                            rect_rows(&pred, b.w, y - b.y, x - b.x, sz, sz)
                                .flatten()
                                .copied()
                                .collect()
                        })
                        .collect();
                    best = Some(RectSplit {
                        tus: Vec::new(),
                        tx: Vec::new(),
                        mode: (crate::av2::palette::PAL_MODE, 0),
                        cost,
                        recon,
                    });
                }
            }
        }
        best
    }
}

/// Inputs of [`Av2Encoder::gtree_search`].
struct GSearch<'a> {
    yp: &'a [f32],
    pw: usize,
    frame: (usize, usize),
    qstep: i32,
    lambda: f32,
    /// SB origin (px).
    sb: (usize, usize),
    /// Probe decision to refine: its split nodes stay quad splits, its leaves may
    /// become VERT/HORZ pairs (None = free search).
    guide: Option<LumaPartitionDecision>,
}

/// A searched subtree: RD cost, pre-order codes, reconstruction (row-major w x h).
struct GNode {
    cost: f32,
    codes: Vec<u8>,
    recon: Vec<f32>,
}

impl Av2Encoder {
    fn g_save(recy: &[f32], pw: usize, y: usize, x: usize, w: usize, h: usize) -> Vec<f32> {
        rect_rows(recy, pw, y, x, w, h).flatten().copied().collect()
    }

    fn g_mark(coded: &mut [u8; 256], sb: (usize, usize), y: usize, x: usize, w: usize, h: usize) {
        let (r0, c0) = ((y - sb.0) / 4, (x - sb.1) / 4);
        for r in r0..r0 + h / 4 {
            coded[r * 16 + c0..r * 16 + c0 + w / 4].fill(1);
        }
    }

    /// RD of one leaf (best mode over its square TUs) in the current context;
    /// returns (cost incl. the leaf's partition bit, w x h recon).
    #[allow(clippy::too_many_arguments)]
    fn g_leaf(
        &self,
        recy: &mut [f32],
        g: &GSearch<'_>,
        coded: &[u8; 256],
        y: usize,
        x: usize,
        w: usize,
        h: usize,
    ) -> (f32, Vec<f32>) {
        let b = RectSplitBlock {
            yp: g.yp,
            pw: g.pw,
            y,
            x,
            w,
            h,
            frame: g.frame,
            coded,
            smooth_neighbour: false,
        };
        let r = self
            .leaf_eval(recy, &b, g.qstep, g.lambda, w == h)
            .expect("DC is always a candidate");
        let (sz, pos) = b.tus();
        let mut out = vec![0f32; w * h];
        for (k, &(ty, tx)) in pos.iter().enumerate() {
            for rr in 0..sz {
                let o = (ty - y + rr) * w + (tx - x);
                out[o..o + sz].copy_from_slice(&r.recon[k][rr * sz..rr * sz + sz]);
            }
        }
        let mut cost = r.cost;
        // A 64x64 leaf may also code one TX_64X64 (the leaf coder picks it by RD).
        if w == 64 && h == 64 && crate::av2::rd::knob_or("TX64", 1.0) > 0.0 {
            let (c64, _, _, rec64) =
                self.tx64_eval(recy, g.yp, g.pw, y, x, g.frame, coded, g.qstep, g.lambda);
            if c64 < cost {
                cost = c64;
                out = rec64;
            }
        }
        (cost + g.lambda * G_LEAF_BITS, out)
    }

    /// Best subtree of the `s` x `s` square at (y, x). On return the region holds
    /// its reconstruction and `coded` marks it. The quad (VERT, then HORZ halves) is
    /// searched first in coding order; its squares are reused by the rect options.
    fn g_square(
        &self,
        recy: &mut [f32],
        g: &GSearch<'_>,
        coded: &mut [u8; 256],
        y: usize,
        x: usize,
        s: usize,
    ) -> GNode {
        let pw = g.pw;
        let saved = Self::g_save(recy, pw, y, x, s, s);
        let coded0 = *coded;
        if s == 8 {
            let (cost, recon) = self.g_leaf(recy, g, coded, y, x, 8, 8);
            put_block_rect(recy, pw, y, x, 8, 8, &recon);
            Self::g_mark(coded, g.sb, y, x, 8, 8);
            return GNode {
                cost,
                codes: vec![0],
                recon,
            };
        }
        let h = s / 2;
        let split = g.lambda * G_SPLIT_BITS;
        // Guided: is this square split in the probe's quad tree?
        let probe_split = g.guide.as_ref().map(|p| {
            let (dy, dx) = (y - g.sb.0, x - g.sb.1);
            let s32 = |dy: usize, dx: usize| p.split32[(dy / 32) * 2 + dx / 32];
            match s {
                64 => p.split64,
                32 => p.split64 && s32(dy, dx),
                _ => p.split64 && s32(dy, dx) && p.split16[(dy / 16) * 4 + dx / 16],
            }
        });
        // Quad, VERT-first coding order: TL, BL, TR, BR.
        let quad = (probe_split != Some(false)).then(|| {
            let tl = self.g_square(recy, g, coded, y, x, h);
            let bl = self.g_square(recy, g, coded, y + h, x, h);
            let tr = self.g_square(recy, g, coded, y, x + h, h);
            let br = self.g_square(recy, g, coded, y + h, x + h, h);
            let mut codes = vec![2, 1];
            codes.extend(&tl.codes);
            codes.extend(&bl.codes);
            codes.push(1);
            codes.extend(&tr.codes);
            codes.extend(&br.codes);
            let node = GNode {
                cost: tl.cost + bl.cost + tr.cost + br.cost + 3.0 * split,
                codes,
                recon: Self::g_save(recy, pw, y, x, s, s),
            };
            (node, [tl, bl, tr, br])
        });
        // A guided split whose four children are all split again: rect pairs would
        // span that detail, so keep the quad.
        let deep = probe_split == Some(true)
            && s >= 32
            && crate::av2::rd::knob_or("GPDEEP", 1.0) > 0.0
            && g.guide.as_ref().is_some_and(|p| {
                let (dy, dx) = (y - g.sb.0, x - g.sb.1);
                let (h2, cs) = (s / 2, s / 2);
                [(0, 0), (0, h2), (h2, 0), (h2, h2)]
                    .iter()
                    .all(|&(oy, ox)| {
                        let (cy, cx) = (dy + oy, dx + ox);
                        if cs == 32 {
                            p.split32[(cy / 32) * 2 + cx / 32]
                        } else {
                            p.split32[(cy / 32) * 2 + cx / 32] && p.split16[(cy / 16) * 4 + cx / 16]
                        }
                    })
            });
        let pair_at_split = crate::av2::rd::knob_or("GPMODE", 1.0) > 0.0 && !deep;
        if probe_split == Some(true) && !pair_at_split {
            return quad.expect("guided split computes the quad").0;
        }
        let quad_recon = quad.as_ref().map(|q| q.0.recon.clone());
        let mut best = match &quad {
            Some(q) => GNode {
                cost: q.0.cost,
                codes: q.0.codes.clone(),
                recon: q.0.recon.clone(),
            },
            None => GNode {
                cost: f32::INFINITY,
                codes: Vec::new(),
                recon: Vec::new(),
            },
        };
        let mut best_coded = *coded;
        let restore = |recy: &mut [f32], coded: &mut [u8; 256]| {
            put_block_rect(recy, pw, y, x, s, s, &saved);
            *coded = coded0;
        };
        // Whole leaf (not where the guiding probe split).
        restore(recy, coded);
        let (lc, lrec) = if probe_split == Some(true) {
            (f32::INFINITY, Vec::new())
        } else {
            self.g_leaf(recy, g, coded, y, x, s, s)
        };
        if lc < best.cost {
            let mut c = coded0;
            Self::g_mark(&mut c, g.sb, y, x, s, s);
            best = GNode {
                cost: lc,
                codes: vec![0],
                recon: lrec,
            };
            best_coded = c;
        }
        // VERT / HORZ pairs; each half: leaf, its 4:1 pair, or the quad's squares.
        let dirs: &[u8] = if crate::av2::rd::knob_or("GNORECT", 0.0) > 0.0 {
            &[]
        } else {
            &[2, 1]
        };
        let rect_extra = g.lambda * crate::av2::rd::knob_or("GRB", 0.0);
        for &dir in dirs {
            restore(recy, coded);
            let mut cost = split + rect_extra;
            let mut codes = vec![dir];
            let mut longs = 0;
            for half in 0..2 {
                let (ry, rx, rw, rh) = if dir == 2 {
                    (y, x + half * h, h, s)
                } else {
                    (y + half * h, x, s, h)
                };
                let rsaved = Self::g_save(recy, pw, ry, rx, rw, rh);
                let rcoded = *coded;
                // Leaf.
                let (lc, lrec) = self.g_leaf(recy, g, coded, ry, rx, rw, rh);
                let mut lcoded = rcoded;
                Self::g_mark(&mut lcoded, g.sb, ry, rx, rw, rh);
                let mut opt: (f32, Vec<u8>, Vec<f32>, [u8; 256]) = (lc, vec![0], lrec, lcoded);
                let mut is_long = false;
                // 4:1 pair (split parallel to the long side).
                let (sw, sh) = if rw > rh { (rw, rh / 2) } else { (rw / 2, rh) };
                if sw.min(sh) >= 8 && crate::av2::rd::knob_or("GSHORT", 0.0) > 0.0 {
                    let sdir = if rw > rh { 1u8 } else { 2u8 };
                    let mut c2 = rcoded;
                    let (c_a, rec_a) = self.g_leaf(recy, g, &c2, ry, rx, sw, sh);
                    put_block_rect(recy, pw, ry, rx, sw, sh, &rec_a);
                    Self::g_mark(&mut c2, g.sb, ry, rx, sw, sh);
                    let (by, bx) = if rw > rh {
                        (ry + sh, rx)
                    } else {
                        (ry, rx + sw)
                    };
                    let (c_b, rec_b) = self.g_leaf(recy, g, &c2, by, bx, sw, sh);
                    put_block_rect(recy, pw, by, bx, sw, sh, &rec_b);
                    Self::g_mark(&mut c2, g.sb, by, bx, sw, sh);
                    let c = c_a + c_b + split;
                    if c < opt.0 {
                        opt = (
                            c,
                            vec![sdir, 0, 0],
                            Self::g_save(recy, pw, ry, rx, rw, rh),
                            c2,
                        );
                    }
                    put_block_rect(recy, pw, ry, rx, rw, rh, &rsaved);
                }
                // The quad's two squares (split across the long side).
                if let (Some((_, [tl, bl, tr, br])), Some(quad_recon)) = (&quad, &quad_recon) {
                    let (a, b2, ldir) = match (dir, half) {
                        (2, 0) => (tl, bl, 1u8),
                        (2, _) => (tr, br, 1u8),
                        (_, 0) => (tl, tr, 2u8),
                        _ => (bl, br, 2u8),
                    };
                    let c = a.cost + b2.cost + split;
                    if c < opt.0 {
                        let mut lc2 = rcoded;
                        Self::g_mark(&mut lc2, g.sb, ry, rx, rw, rh);
                        let mut codes2 = vec![ldir];
                        codes2.extend(&a.codes);
                        codes2.extend(&b2.codes);
                        let rec: Vec<f32> = rect_rows(quad_recon, s, ry - y, rx - x, rw, rh)
                            .flatten()
                            .copied()
                            .collect();
                        opt = (c, codes2, rec, lc2);
                        is_long = true;
                    }
                }
                put_block_rect(recy, pw, ry, rx, rw, rh, &opt.2);
                *coded = opt.3;
                cost += opt.0;
                codes.extend(&opt.1);
                longs += is_long as usize;
            }
            // HORZ with both halves split across is the quad again.
            if dir == 1 && longs == 2 {
                continue;
            }
            if cost < best.cost {
                best = GNode {
                    cost,
                    codes,
                    recon: Self::g_save(recy, pw, y, x, s, s),
                };
                best_coded = *coded;
            }
        }
        put_block_rect(recy, pw, y, x, s, s, &best.recon);
        *coded = best_coded;
        best
    }

    /// The 64x64 luma leaf at (y, x) as one TX_64X64 (DCT; only its 32x32 low
    /// frequencies are coded) with each non-directional mode predicted as one 64x64
    /// block. Returns the RD-best (cost, levels, mode, recon) — cost = pixel SSE +
    /// lambda * (coefficient proxy + mode bits). Leaves `recy` as found.
    #[allow(clippy::too_many_arguments)]
    fn tx64_eval(
        &self,
        recy: &[f32],
        yp: &[f32],
        pw: usize,
        y: usize,
        x: usize,
        frame: (usize, usize),
        coded: &[u8; 256],
        qstep: i32,
        lambda: f32,
    ) -> (f32, Vec<f32>, usize, Vec<f32>) {
        let bases = &self.bases;
        let bd = self.bit_depth as i32;
        let neutral = self.dc_neutral();
        let src: Vec<f32> = rect_rows(yp, pw, y, x, 64, 64).flatten().copied().collect();
        let scale = bases.chroma444.qstep as f32 / qstep as f32;
        let mut best: Option<(f32, Vec<f32>, usize, Vec<f32>)> = None;
        for m in 0..=4usize {
            let pred: Vec<f32> = if m == 0 {
                vec![dc_pred_rect(recy, pw, y, x, 64, 64, neutral, bd); 4096]
            } else {
                leaf::predict_luma_square(
                    recy,
                    &leaf::SquareLumaBlock {
                        stride: pw,
                        frame_width: frame.0,
                        frame_height: frame.1,
                        y,
                        x,
                        size: 64,
                    },
                    coded,
                    leaf::SquareLumaMode::nondir(m),
                    neutral,
                    bd,
                )
            };
            let resid: Vec<f32> = src
                .iter()
                .zip(&pred)
                .map(|(&s, &p)| (s - p) * scale)
                .collect();
            let lev = bases.chroma444.project_scan(&resid, 0.0, &tables::SCAN);
            let p_i: Vec<i32> = pred.iter().map(|&v| v as i32).collect();
            let rec =
                itx422::reconstruct_chroma_pred_tx(&p_i, &lev, qstep, &tables::SCAN, 64, 64, bd, 0);
            let sse: f32 = src
                .iter()
                .zip(&rec)
                .map(|(&a, &r)| (a - r.round()).powi(2))
                .sum();
            let mb = if m == 0 {
                0.0
            } else {
                crate::av2::rd::knob_or("MB16", 6.0)
            };
            let cost = sse + lambda * (coeff_rate_f32(&lev) + mb);
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, lev, m, rec));
            }
        }
        best.expect("DC is always a candidate")
    }

    /// Free binary-tree luma partition search of a full interior SB (squares, 2:1
    /// and 4:1 leaves from 64 down to 8), every leaf priced by [`Self::leaf_eval`].
    /// Leaves `recy` as found.
    fn gtree_search(&self, recy: &mut [f32], g: &GSearch<'_>) -> crate::av2::leaf::GTree {
        let saved = Self::g_save(recy, g.pw, g.sb.0, g.sb.1, 64, 64);
        let mut coded = [0u8; 256];
        let n = self.g_square(recy, g, &mut coded, g.sb.0, g.sb.1, 64);
        put_block_rect(recy, g.pw, g.sb.0, g.sb.1, 64, 64, &saved);
        crate::av2::leaf::GTree::from_codes(&n.codes)
    }
}

/// Partition signalling in the tree search: a leaf's do_split = 0, a split's
/// do_split + rect_type.
const G_LEAF_BITS: f32 = 1.0;
const G_SPLIT_BITS: f32 = 2.0;

impl Av2Encoder {
    /// Search the live DRL/NEWMV candidates and materialize the selected luma
    /// and chroma predictors in reusable scratch. Keeping motion estimation,
    /// mode comparison, MC and combined-plane distortion out of the partition
    /// state machine gives them an independent optimization unit.
    #[inline(never)]
    fn search_inter_leaf_420(
        &self,
        search: InterLeafSearch420<'_>,
    ) -> Option<InterLeafCandidate420> {
        let InterLeafSearch420 {
            enc,
            source_y,
            source_u,
            source_v,
            luma_reference,
            chroma_references,
            me_scratch,
            prediction_scratch,
            source_stride,
            chroma_stride,
            reference_x,
            reference_y,
            reference_luma_stride,
            block_x,
            block_y,
            block_width,
            block_height,
            predictor_mv,
            frame_mv_seed,
            skip_ctx,
            mode_ctx,
            qstep,
        } = search;
        let (reference, reference_stride, reference_height) = luma_reference;
        let mut predictors = me::MeCandidates::new();
        for candidate in [predictor_mv, Mv::ZERO, frame_mv_seed] {
            predictors.push_unique(candidate);
        }
        let (mv, _) = me::search(
            &me::MePlanes {
                current: &source_y[block_y * source_stride + block_x..],
                current_stride: source_stride,
                reference,
                reference_stride: *reference_stride,
            },
            predictors.as_slice(),
            &me::MeSearchSpec {
                origin_x: (reference_x + block_x + INTER_BORDER_420) as isize,
                origin_y: (reference_y + block_y + INTER_BORDER_420) as isize,
                width: block_width,
                height: block_height,
                reference_mv: predictor_mv,
                lambda_mv: (qstep as u32).max(1),
                max_dx: self.video_search_range,
                max_dy: self.video_search_range,
                predictor_gate_sad_per_pixel: self.video_predictor_gate,
                integer_satd_radius: self.video_integer_satd_radius,
                bit_depth: self.bit_depth,
                frame_width: *reference_stride,
                frame_height: *reference_height + 2 * INTER_BORDER_420,
            },
            me_scratch,
        );
        let mut mv = mc::clamp_umv(
            mv,
            (reference_x + block_x) as i32,
            (reference_y + block_y) as i32,
            block_width as i32,
            block_height as i32,
            reference_luma_stride as i32,
            *reference_height as i32,
        );
        if mv == Mv::ZERO {
            return None;
        }

        let chroma_width = block_width / 2;
        let chroma_height = block_height / 2;
        let (pred_y, pred_u, pred_v, convolve_tmp) =
            prediction_scratch.planes(block_width * block_height, chroma_width * chroma_height);
        let motion_block = |mv| mc::MotionBlock {
            origin_x: (reference_x + block_x + INTER_BORDER_420) as isize,
            origin_y: (reference_y + block_y + INTER_BORDER_420) as isize,
            mv,
            width: block_width,
            height: block_height,
            bit_depth: self.bit_depth,
        };
        mc::predict_with_tmp(
            pred_y,
            block_width,
            reference,
            *reference_stride,
            &motion_block(mv),
            convolve_tmp,
        );

        let searched_mv = mv;
        let searched_luma_sse = rect_sse_f32(
            &PlaneRect {
                plane: source_y,
                stride: source_stride,
                y: block_y,
                x: block_x,
            },
            &PlaneRect {
                plane: pred_y,
                stride: block_width,
                y: 0,
                x: 0,
            },
            block_width,
            block_height,
        );
        let (searched_mode, searched_mvd_row, searched_mvd_col) =
            inter_mode_qtr_mvd(searched_mv, predictor_mv);
        let searched_mvd = Mv {
            row: searched_mvd_row,
            col: searched_mvd_col,
        };
        let bounded_predictor = mc::clamp_umv(
            predictor_mv,
            (reference_x + block_x) as i32,
            (reference_y + block_y) as i32,
            block_width as i32,
            block_height as i32,
            reference_luma_stride as i32,
            *reference_height as i32,
        );
        if predictor_mv != Mv::ZERO
            && predictor_mv != searched_mv
            && bounded_predictor == predictor_mv
        {
            mc::predict_with_tmp(
                pred_y,
                block_width,
                reference,
                *reference_stride,
                &motion_block(predictor_mv),
                convolve_tmp,
            );
            let near_sse = rect_sse_f32(
                &PlaneRect {
                    plane: source_y,
                    stride: source_stride,
                    y: block_y,
                    x: block_x,
                },
                &PlaneRect {
                    plane: pred_y,
                    stride: block_width,
                    y: 0,
                    x: 0,
                },
                block_width,
                block_height,
            );
            debug_assert_eq!(searched_mode, 2);
            if crate::av2::video::rd::prefer_nearmv(
                enc,
                crate::av2::video::rd::NearMvRdSpec {
                    skip_ctx,
                    mode_ctx,
                    skip_txfm: true,
                    near_distortion: near_sse * crate::av2::video::rd::SS2_INTER_DIST_W,
                    new_distortion: searched_luma_sse * crate::av2::video::rd::SS2_INTER_DIST_W,
                    new_mvd: searched_mvd,
                    qstep: qstep as u32,
                },
            ) {
                mv = predictor_mv;
            } else {
                mc::predict_with_tmp(
                    pred_y,
                    block_width,
                    reference,
                    *reference_stride,
                    &motion_block(searched_mv),
                    convolve_tmp,
                );
            }
        }

        let chroma_x = block_x / 2;
        let chroma_y = block_y / 2;
        let chroma_mv = Mv {
            row: mv.row / 2,
            col: mv.col / 2,
        };
        for (prediction, (reference, stride)) in [(&mut *pred_u), (&mut *pred_v)]
            .into_iter()
            .zip(chroma_references)
        {
            mc::predict_with_tmp(
                prediction,
                chroma_width,
                reference,
                *stride,
                &mc::MotionBlock {
                    origin_x: (reference_x / 2 + chroma_x + INTER_BORDER_420 / 2) as isize,
                    origin_y: (reference_y / 2 + chroma_y + INTER_BORDER_420 / 2) as isize,
                    mv: chroma_mv,
                    width: chroma_width,
                    height: chroma_height,
                    bit_depth: self.bit_depth,
                },
                convolve_tmp,
            );
        }
        let mut distortion = rect_sse_f32(
            &PlaneRect {
                plane: source_y,
                stride: source_stride,
                y: block_y,
                x: block_x,
            },
            &PlaneRect {
                plane: pred_y,
                stride: block_width,
                y: 0,
                x: 0,
            },
            block_width,
            block_height,
        );
        for (source, prediction) in [(source_u, &*pred_u), (source_v, &*pred_v)] {
            distortion += rect_sse_f32(
                &PlaneRect {
                    plane: source,
                    stride: chroma_stride,
                    y: chroma_y,
                    x: chroma_x,
                },
                &PlaneRect {
                    plane: prediction,
                    stride: chroma_width,
                    y: 0,
                    x: 0,
                },
                chroma_width,
                chroma_height,
            );
        }
        let (mode, mvd_row, mvd_col) = inter_mode_qtr_mvd(mv, predictor_mv);
        let mvd = (mode == 2).then_some(Mv {
            row: mvd_row,
            col: mvd_col,
        });
        let rate =
            crate::av2::video::rd::inter_syntax_bits(enc, skip_ctx, mode_ctx, true, mode, mvd);
        Some(InterLeafCandidate420 {
            mv,
            mode,
            mvd,
            rd_cost: crate::av2::video::rd::rd_cost(
                distortion * crate::av2::video::rd::SS2_INTER_DIST_W,
                rate,
                qstep as u32,
            ),
        })
    }

    /// Quantize and reconstruct a square dense-inter candidate. The partition
    /// walker only decides whether to commit the returned candidate and updates
    /// entropy-neighbor state; transform and predictor mechanics stay here.
    #[inline(never)]
    fn evaluate_dense_inter_residual_420(
        &self,
        input: DenseInterResidualInput420<'_>,
    ) -> DenseInterResidual420 {
        let DenseInterResidualInput420 {
            enc,
            source_y,
            source_u,
            source_v,
            prediction_y,
            prediction_u,
            prediction_v,
            source_stride,
            chroma_stride,
            block_x,
            block_y,
            block_size,
            residual_scale,
            qstep,
            skip_ctx,
            mode_ctx,
            inter_mode,
            mvd,
        } = input;
        debug_assert!(matches!(block_size, 16 | 32));
        let chroma_size = block_size / 2;
        let chroma_x = block_x / 2;
        let chroma_y = block_y / 2;
        let bd = self.bit_depth as i32;

        let mut y_residual = [0.0f32; 32 * 32];
        let y_len = block_size * block_size;
        crate::av2::metrics::scaled_residual_f32(
            &mut y_residual[..y_len],
            &source_y[block_y * source_stride + block_x..],
            prediction_y,
            crate::av2::metrics::ResidualSpec {
                src_stride: source_stride,
                pred_stride: block_size,
                width: block_size,
                height: block_size,
                scale: residual_scale,
            },
        );
        let y_levels = if block_size == 32 {
            self.bases.luma.project(&y_residual[..y_len], 0.0)
        } else {
            self.bases
                .luma16x16
                .project_scan(&y_residual[..y_len], 0.0, &SCAN16)
        };
        let mut y_recon = [0.0f32; 32 * 32];
        if block_size == 32 {
            y_recon = reconstruct_luma(prediction_y, &y_levels, qstep, &tables::SCAN, bd);
        } else {
            y_recon[..y_len].copy_from_slice(&itx422::reconstruct_luma16(
                prediction_y,
                &y_levels,
                qstep,
                &SCAN16,
                bd,
            ));
        }

        let mut chroma_levels = [Vec::new(), Vec::new()];
        let mut chroma_recon = [Vec::new(), Vec::new()];
        for (plane, (source, prediction)) in [(source_u, prediction_u), (source_v, prediction_v)]
            .into_iter()
            .enumerate()
        {
            let mut residual = [0.0f32; 16 * 16];
            let mut prediction_i = [0i32; 16 * 16];
            let chroma_len = chroma_size * chroma_size;
            crate::av2::metrics::f32_prediction_and_scaled_residual_i32(
                &mut prediction_i[..chroma_len],
                &mut residual[..chroma_len],
                &source[chroma_y * chroma_stride + chroma_x..],
                prediction,
                crate::av2::metrics::ResidualSpec {
                    src_stride: chroma_stride,
                    pred_stride: chroma_size,
                    width: chroma_size,
                    height: chroma_size,
                    scale: residual_scale,
                },
            );
            let levels = if chroma_size == 16 {
                self.bases
                    .luma16x16
                    .project_scan(&residual[..chroma_len], 0.0, &SCAN16)
            } else {
                self.bases
                    .c8x8
                    .project_scan(&residual[..chroma_len], 0.0, &SCAN8X8)
            };
            let scan = if chroma_size == 16 {
                &SCAN16[..]
            } else {
                &SCAN8X8[..]
            };
            chroma_recon[plane] = itx422::reconstruct_chroma_cfl(
                &prediction_i[..chroma_len],
                &levels,
                qstep,
                scan,
                chroma_size,
                chroma_size,
                bd,
            );
            chroma_levels[plane] = levels;
        }

        let mut distortion = rect_sse_f32(
            &PlaneRect {
                plane: source_y,
                stride: source_stride,
                y: block_y,
                x: block_x,
            },
            &PlaneRect {
                plane: &y_recon,
                stride: block_size,
                y: 0,
                x: 0,
            },
            block_size,
            block_size,
        );
        for (source, reconstruction) in [(source_u, &chroma_recon[0]), (source_v, &chroma_recon[1])]
        {
            distortion += rect_sse_f32(
                &PlaneRect {
                    plane: source,
                    stride: chroma_stride,
                    y: chroma_y,
                    x: chroma_x,
                },
                &PlaneRect {
                    plane: reconstruction,
                    stride: chroma_size,
                    y: 0,
                    x: 0,
                },
                chroma_size,
                chroma_size,
            );
        }
        let rate = crate::av2::video::rd::inter_syntax_bits(
            enc, skip_ctx, mode_ctx, false, inter_mode, mvd,
        ) + coeff_rate_f32(&y_levels)
            + coeff_rate_f32(&chroma_levels[0])
            + coeff_rate_f32(&chroma_levels[1]);
        DenseInterResidual420 {
            rd_cost: crate::av2::video::rd::rd_cost(
                distortion * crate::av2::video::rd::SS2_INTER_DIST_W,
                rate,
                qstep as u32,
            ),
            y_coeffs: levels_to_coeffs(&y_levels),
            u_coeffs: levels_to_coeffs(&chroma_levels[0]),
            v_coeffs: levels_to_coeffs(&chroma_levels[1]),
            y_recon,
            u_recon: std::mem::take(&mut chroma_recon[0]),
            v_recon: std::mem::take(&mut chroma_recon[1]),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_yuv420_partition(
        &self,
        enc: &mut RangeEncoder,
        luma: LumaPlanes,
        chroma: ChromaPlaneRefs,
        ctx: &PartitionPass,
        nb: PartitionNeighbors,
        cnb: ChromaNeighborBufs,
        // Inter reference already resolved by the core; `None` on the whole-frame
        // path. `Some` carries the FULL reference frame + this tile's origin so
        // motion compensation stays frame-global (see `TileRefCtx`).
        tile_ref: Option<&tiling::TileRefCtx>,
        // Deblock support: each luma leaf's `(mi_row, mi_col, bw_mi, bh_mi)` is
        // recorded here so the caller can build the transform-boundary grid the DF
        // needs. Empty vec when deblock is inactive.
        tx_leaves: &mut Vec<(usize, usize, usize, usize)>,
        // Prediction blocks that signal skip_txfm. AVM derives their deblock
        // geometry from the prediction block rather than residual TX tiling.
        skip_leaves: &mut Vec<(usize, usize, usize, usize)>,
        // SDP chroma-tree leaves (empty without SDP: chroma follows `tx_leaves`).
        chroma_leaves: &mut Vec<(usize, usize, usize, usize)>,
        // Per-SB settled AC qindex (`final_qindex_ac`, decoder `ts.last_qidx`),
        // indexed `row * sb_cols + col`. The DF thresholds are per-qindex, so with
        // AQ on this varies per SB. Pre-sized `sb_rows * sb_cols` by the caller.
        sb_qidx: &mut [u16],
        // Staged replay: `Capture` logs each walk SB's per-leaf luma + chroma
        // winners; `Replay` reuses them (skips the leaf mode/tx + CfL/MHCCP
        // searches); `Off` = today's inline behavior. Only the whole-64 `(16,16)`
        // intra leaf is captured so far — every other shape (and any inter-coded
        // SB) records `Fallback` and is re-searched byte-identically on replay.
        mut decide_mode: replay::DecideMode<'_>,
    ) {
        let LumaPlanes { rec: recy, src: yp } = luma;
        let ChromaPlaneRefs {
            rec_u: recu,
            rec_v: recv,
            src_u: up,
            src_v: vp,
        } = chroma;
        let &PartitionPass {
            luma_stride: pw,
            chroma_stride: pcw,
            width,
            height,
            sb_rows,
            sb_cols,
            tmc,
            tmr,
            quant:
                QuantCtx {
                    qc,
                    neutral,
                    qstep: qstep_i,
                    rdoq_lambda,
                },
        } = ctx;
        let PartitionNeighbors {
            above,
            left,
            above_pctx,
            left_pctx,
        } = nb;
        let ChromaNeighborBufs {
            u_above,
            v_above,
            u_left,
            v_left,
        } = cnb;
        // Per-mi CfL-usage neighbors for get_cfl_ctx (one bit per chroma block).
        let mut cfl_above = vec![0i32; tmc as usize + 16];
        let mut cfl_left = vec![0i32; tmr as usize + 16];
        let mut midx_above = vec![-1i32; tmc as usize + 16];
        let mut midx_left = vec![-1i32; tmr as usize + 16];
        // Per-mi interstate grids (1 = neighbor is an inter block). Needed so intra
        // edge leaves on an inter frame derive the correct intra_inter context (AVM
        // av2_get_intra_inter_context) from actual neighbor state, not all-intra.
        let mut inter_mi_above = vec![0u8; tmc as usize + 16];
        let mut inter_mi_left = vec![0u8; tmr as usize + 16];
        let mut skip_mi_above = vec![0u8; tmc as usize + 16];
        let mut skip_mi_left = vec![0u8; tmr as usize + 16];
        let mut newmv_mi_above = vec![0u8; tmc as usize + 16];
        let mut newmv_mi_left = vec![0u8; tmr as usize + 16];
        let mut mv_mi_above = vec![None; tmc as usize + 16];
        let mut mv_mi_left = vec![None; tmr as usize + 16];
        // Per-mi reference rank neighbors (two-reference frames): the single_ref
        // bit context for every leaf reads the mi-granular neighbor rank.
        let mut ref_mi_above = vec![0u8; tmc as usize + 16];
        let mut ref_mi_left = vec![0u8; tmr as usize + 16];
        // SB-granular NEWMV neighbor tracking. The inter mode context adds +2 when a
        // neighbor was coded NEWMV (AVM mvref newmv_count term); matches core.rs.
        let mut newmv_above = vec![0u8; sb_cols.max(1)];
        // Absolute decoded MVs (eighth-pel) for the DRL[0] predictor. The old
        // partition path tracked only the NEWMV mode bit and emitted the absolute
        // searched MV as an MVD. Once a left/above inter block existed, the decoder
        // added that spatial predictor again, making motion accumulate across SBs.
        let mut mv_above: Vec<Option<Mv>> = vec![None; sb_cols.max(1)];
        // SB-granular reference rank for the whole-64 skip mode context.
        let mut ref_above = vec![0u8; sb_cols.max(1)];
        let mhccp_bounds = cfl::MhccpBounds::from_luma(width, height, true, true);
        // Variance Boost on the partition path: one AqState per tile, queried per 64x64 SB.
        // When delta-Q is off (or base_q==0) `per_sb` returns `(qstep_i, 1.0)` and signals 0,
        // so this path stays byte-identical to the pre-VB encoder.
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
            0, /* uv delta not yet wired for 4:2:0 */
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength * crate::av2::rd::knob("VB"),
            self.tune.vb_boost_only,
        )
        .with_dark_aq({
            let mut d = self.tune.dark_aq;
            d.enabled &= crate::av2::rd::knob("DARK") > 0.5;
            d
        });
        // LAST reference (f32 Y,U,V) for zero-MV GLOBALMV prediction; empty if intra.
        let (last_ref, ref_x0, ref_y0, ref_ls, ref_cs): (
            std::sync::Arc<Vec<Vec<f32>>>,
            usize,
            usize,
            usize,
            usize,
        ) = if enc.inter_tile {
            match tile_ref {
                Some(r) => (
                    std::sync::Arc::clone(&r.planes),
                    r.x0,
                    r.y0,
                    r.luma_stride,
                    r.chroma_stride,
                ),
                None => (
                    std::sync::Arc::clone(&self.last_ref.lock().unwrap()),
                    0,
                    0,
                    pw,
                    pcw,
                ),
            }
        } else {
            (std::sync::Arc::new(Vec::new()), 0, 0, pw, pcw)
        };
        let has_last = last_ref.len() >= 3 && !last_ref[0].is_empty();
        // Rank-1 reference for the zero-motion skip mode (whole-frame, untiled
        // two-reference frames only; tiled frames carry a single ref per tile).
        let second_ref: std::sync::Arc<Vec<Vec<f32>>> =
            if enc.inter_tile && tile_ref.is_none() && enc.num_refs >= 2 {
                std::sync::Arc::clone(&self.second_ref.lock().unwrap())
            } else {
                std::sync::Arc::new(Vec::new())
            };
        let has_second = second_ref.len() >= 3 && !second_ref[0].is_empty();
        let frame_mv_seed = *self.video_mv_seed.lock().unwrap();
        // Prepare edge-extended references once for the entire partition pass.
        // NEWMV used to clone and border the full luma plane, then both chroma
        // planes, inside every 64x64 decision.
        let inter_luma: Option<(Vec<f32>, usize, usize)> = if has_last {
            let refh = last_ref[0].len() / ref_ls;
            let (plane, stride) = crate::av2::video::mc::bordered(
                &last_ref[0],
                ref_ls,
                refh,
                ref_ls,
                INTER_BORDER_420,
            );
            Some((plane, stride, refh))
        } else {
            None
        };
        let inter_chroma: Option<[(Vec<f32>, usize); 2]> = if has_last {
            let border = INTER_BORDER_420 / 2;
            let prepare = |plane: &[f32]| {
                crate::av2::video::mc::bordered(plane, ref_cs, plane.len() / ref_cs, ref_cs, border)
            };
            Some([prepare(&last_ref[1]), prepare(&last_ref[2])])
        } else {
            None
        };
        let mut skip_above = vec![0u8; sb_cols.max(1)];
        let mut inter_above = vec![0u8; sb_cols.max(1)];
        let mut me_scratch = me::MeScratch::default();
        let mut inter_pred_scratch = InterPredScratch420::default();
        // Serial AQ pre-pass grid: the per-SB values a wavefront (out-of-raster-order)
        // decide reads instead of the serial `aqs` probe/commit accumulator. The 4:2:0
        // partition path commits `per_sb` at EVERY SB (edge included), so the grid is
        // built with `needs_partition = false` (accumulate everywhere) to stay bit-exact
        // with that sequence. Only read when `use_grid` (fresh-ctx check + wavefront).
        let aq_grid = aqs.precompute_grid(yp, pw, width, height, false);
        // The serial path retains its accumulated contexts. Wavefront workers reset
        // their private contexts per SB and read AQ from the precomputed grid.
        let fresh_ctx = false;
        // Halo geometry (audited): intra reads a single reference line + above-right
        // extension of the top row; CfL/MHCCP read the same 1px border; the whole-64 RD
        // path reads a full-plane SSE anchor at (0,0). Copy a generous superset — a 32px
        // perpendicular band + 64px above-right margin — plus that anchor. Luma is
        // full-res (`pw`); 4:2:0 chroma is QUARTERED (half-width AND half-height, stride
        // `pcw`), so its origin is `(sb_y/2, sb_x/2)` and its own block is 32×32.
        // This halo is the complete read-set used to isolate a wavefront worker.
        let ph = sb_rows * 64;
        let pch = ph / 2;
        const HALO_BAND: usize = 32;
        const HALO_AR: usize = 64;
        const POISON: f32 = 0.0;
        let halo_mode = false;
        let (mut recy_p, mut recu_p, mut recv_p) = if halo_mode {
            (
                vec![0f32; pw * ph],
                vec![0f32; pcw * pch + 1],
                vec![0f32; pcw * pch + 1],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let copy_halo_luma =
            |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize, copy_rd_anchor: bool| {
                let bx = sb_x.saturating_sub(HALO_BAND);
                let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                let by = sb_y.saturating_sub(HALO_BAND);
                for r in by..sb_y {
                    dst[r * pw + bx..r * pw + tx_end]
                        .copy_from_slice(&src[r * pw + bx..r * pw + tx_end]);
                }
                let ly_end = (sb_y + 64).min(ph);
                for r in sb_y..ly_end {
                    dst[r * pw + bx..r * pw + sb_x]
                        .copy_from_slice(&src[r * pw + bx..r * pw + sb_x]);
                }
                if copy_rd_anchor && (sb_y != 0 || sb_x != 0) {
                    for r in 0..64.min(ph) {
                        dst[r * pw..r * pw + 64].copy_from_slice(&src[r * pw..r * pw + 64]);
                    }
                }
            };
        // 4:2:0 chroma halo: QUARTERED (both axes subsampled). Origin `(cy, cx) =
        // (sb_y/2, sb_x/2)`, own block 32×32.
        let copy_halo_chroma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let cx = sb_x / 2;
            let cy = sb_y / 2;
            let bx = cx.saturating_sub(HALO_BAND);
            let tx_end = (cx + 32 + HALO_AR).min(pcw);
            let by = cy.saturating_sub(HALO_BAND);
            for r in by..cy {
                dst[r * pcw + bx..r * pcw + tx_end]
                    .copy_from_slice(&src[r * pcw + bx..r * pcw + tx_end]);
            }
            let cy_end = (cy + 32).min(pch);
            for r in cy..cy_end {
                dst[r * pcw + bx..r * pcw + cx].copy_from_slice(&src[r * pcw + bx..r * pcw + cx]);
            }
        };
        let copy_own_luma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let ly_end = (sb_y + 64).min(ph);
            for r in sb_y..ly_end {
                dst[r * pw + sb_x..r * pw + sb_x + 64]
                    .copy_from_slice(&src[r * pw + sb_x..r * pw + sb_x + 64]);
            }
        };
        let copy_own_chroma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let cx = sb_x / 2;
            let cy = sb_y / 2;
            let cy_end = (cy + 32).min(pch);
            for r in cy..cy_end {
                dst[r * pcw + cx..r * pcw + cx + 32]
                    .copy_from_slice(&src[r * pcw + cx..r * pcw + cx + 32]);
            }
        };
        // SB-wavefront parallel decide (stills only; the inter paths carry cross-SB
        // motion state that a wavefront cannot honour, so `has_last` frames stay
        // serial). Each diagonal's cells decide independently into per-cell Capture
        // records over private (halo-seeded) recon buffers with frame-initial contexts
        // and grid AQ; the records + deblock rectangles merge in raster order into the
        // Capture record a serial Replay then consumes byte-identically.
        // Fires when the caller selected a multithreaded single-tile Capture pass;
        // the 4:2:0 API otherwise respects its configured tile grid. Suppressed
        // inside per-tile sub-encodes, which already parallelise.
        let wavefront = Self::resolve_threads(self.threads) > 1
            && !replay::in_tile_subencode()
            && !self.video_mode.load(std::sync::atomic::Ordering::Relaxed)
            && matches!(decide_mode, replay::DecideMode::Capture(_))
            && !has_last;
        if wavefront {
            let enc_qc = enc.qc;
            let enc_cfl = enc.cfl;
            let enc_sct = enc.sct;
            let enc_mhccp = enc.mhccp;
            let enc_dqp = enc.delta_q_present;
            let adaptive = self.tune.updating_cdf && self.base_q_idx != 0;
            let base_q = self.base_q_idx as i32;
            let nthreads = Self::resolve_threads(self.threads);
            let wy = helpers::PlaneWriter::new(recy, pw);
            let wu = helpers::PlaneWriter::new(recu, pcw);
            let wv = helpers::PlaneWriter::new(recv, pcw);
            // Per-mi luma mode class (`midx_above` / `midx_left` values) of every
            // SB's bottom row and right column, so a cell sees its neighbours'
            // directional / SMOOTH state exactly as the serial pass does: the
            // directional edge filter of a border leaf depends on it.
            let map_stride = sb_cols * 16;
            let mut mode_map = vec![-1i32; sb_rows * 16 * map_stride];
            let wm = helpers::PlaneWriter::new(&mut mode_map, map_stride);
            // Persistent-pool WPP wavefront: workers spawn once and loop over the
            // diagonals with a barrier between each, so each worker's thread_local
            // recon scratch is allocated once (not re-allocated per diagonal).
            let slots: Vec<Option<WfSlot420>> = crate::av2::helpers::par_wavefront_pool(
                nthreads,
                sb_rows,
                sb_cols,
                true,
                |r, c| {
                    let sb_y = r * 64;
                    let sb_x = c * 64;
                    let cx = sb_x / 2;
                    let cy = sb_y / 2;

                    WF_SCRATCH_420.with(|sc| {
                        let s = &mut *sc.borrow_mut();
                        s.ensure(WfScratch420Shape {
                            pw,
                            ph,
                            pcw,
                            pch,
                            tmc,
                            tmr,
                            sb_cols,
                            sb_rows,
                        });
                        s.reset_contexts();
                        let (mr0, mc0) = (r * 16, c * 16);
                        {
                            // SAFETY: row mr0-1 / column mc0-1 were written by the
                            // above / left cells, which finished on earlier diagonals.
                            let (mp, _, ms) = wm.read_view();
                            if r > 0 {
                                for col in mc0..(mc0 + 16).min(s.mia.len()) {
                                    s.mia[col] = unsafe { *mp.add((mr0 - 1) * ms + col) };
                                }
                            }
                            if c > 0 {
                                for row in mr0..(mr0 + 16).min(s.mil.len()) {
                                    s.mil[row] = unsafe { *mp.add(row * ms + mc0 - 1) };
                                }
                            }
                        }
                        let ly_end = (sb_y + 64).min(ph);
                        let cyc_end = (cy + 32).min(pch);
                        // Luma halo geometry.
                        let bx = sb_x.saturating_sub(HALO_BAND);
                        let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                        let by = sb_y.saturating_sub(HALO_BAND);
                        // Chroma halo geometry (quartered plane).
                        let cbx = cx.saturating_sub(HALO_BAND);
                        let ctx_end = (cx + 32 + HALO_AR).min(pcw);
                        let cby = cy.saturating_sub(HALO_BAND);
                        // Seed the private buffers from finished (earlier-diagonal) recon.
                        // SAFETY: those regions are on earlier diagonals (done); own-block
                        // writes below are disjoint from every other worker's.
                        unsafe {
                            if sb_y != 0 || sb_x != 0 {
                                wy.copy_region_to(&mut s.ry, 0, 0, 64.min(ph), 64);
                            }
                            if sb_y > by {
                                wy.copy_region_to(&mut s.ry, by, bx, sb_y - by, tx_end - bx);
                            }
                            if ly_end > sb_y && sb_x > bx {
                                wy.copy_region_to(&mut s.ry, sb_y, bx, ly_end - sb_y, sb_x - bx);
                            }
                            if cy > cby {
                                wu.copy_region_to(&mut s.ru, cby, cbx, cy - cby, ctx_end - cbx);
                                wv.copy_region_to(&mut s.rv, cby, cbx, cy - cby, ctx_end - cbx);
                            }
                            if cyc_end > cy && cx > cbx {
                                wu.copy_region_to(&mut s.ru, cy, cbx, cyc_end - cy, cx - cbx);
                                wv.copy_region_to(&mut s.rv, cy, cbx, cyc_end - cy, cx - cbx);
                            }
                        }
                        let mut e = RangeEncoder::new();
                        e.qc = enc_qc;
                        if adaptive {
                            e.enable_adaptive_cdf(enc_qc);
                        }
                        e.cfl = enc_cfl;
                        e.sct = enc_sct;
                        e.mhccp = enc_mhccp;
                        e.mhccp_ssx = true;
                        e.mhccp_ssy = true;
                        e.delta_q_present = enc_dqp;
                        let mut a = aq::AqState::new(enc_dqp, base_q, qstep_i, 0.0, 0);
                        let mut rec = crate::av2::replay::DecisionRecord::new();
                        let mut dm = crate::av2::replay::DecideMode::Capture(&mut rec);
                        let mut lpctx = [0u8; 16];
                        let mut cell_tx: Vec<(usize, usize, usize, usize)> = Vec::new();
                        let mut cell_skips: Vec<(usize, usize, usize, usize)> = Vec::new();
                        let mut cell_ch: Vec<(usize, usize, usize, usize)> = Vec::new();
                        self.decide_sb_420(Sb420Decision {
                            enc: &mut e,
                            aqs: &mut a,
                            decide_mode: &mut dm,
                            recy: &mut s.ry,
                            recu: &mut s.ru,
                            recv: &mut s.rv,
                            yp,
                            up,
                            vp,
                            above: &mut s.above,
                            left: &mut s.left,
                            above_pctx: &mut s.apctx,
                            left_pctx: &mut lpctx,
                            u_above: &mut s.ua,
                            v_above: &mut s.va,
                            u_left: &mut s.ul,
                            v_left: &mut s.vl,
                            cfl_above: &mut s.cfa,
                            cfl_left: &mut s.cfl,
                            midx_above: &mut s.mia,
                            midx_left: &mut s.mil,
                            inter_mi_above: &mut s.imia,
                            inter_mi_left: &mut s.imil,
                            skip_mi_above: &mut s.skmia,
                            skip_mi_left: &mut s.skmil,
                            newmv_mi_above: &mut s.nmvia,
                            newmv_mi_left: &mut s.nmvil,
                            mv_mi_above: &mut s.mvia,
                            mv_mi_left: &mut s.mvil,
                            newmv_above: &mut s.nmva,
                            mv_above: &mut s.mva,
                            skip_above: &mut s.ska,
                            inter_above: &mut s.ina,
                            ref_above: &mut s.rfa,
                            ref_mi_above: &mut s.rfmia,
                            ref_mi_left: &mut s.rfmil,
                            tx_leaves: &mut cell_tx,
                            skip_leaves: &mut cell_skips,
                            chroma_leaves: &mut cell_ch,
                            sb_qidx: &mut s.dbq,
                            last_ref: last_ref.as_slice(),
                            ref_x0,
                            ref_y0,
                            ref_ls,
                            ref_cs,
                            has_last,
                            second_ref: second_ref.as_slice(),
                            has_second,
                            inter_luma: inter_luma.as_ref(),
                            inter_chroma: inter_chroma.as_ref(),
                            me_scratch: &mut s.me,
                            inter_pred_scratch: &mut s.inter_pred,
                            frame_mv_seed,
                            mhccp_bounds,
                            pw,
                            pcw,
                            width,
                            height,
                            sb_cols,
                            tmc,
                            tmr,
                            qc,
                            neutral,
                            rdoq_lambda,
                            aq_grid: &aq_grid,
                            use_grid: true,
                            skip_left: 0,
                            inter_left: 0,
                            newmv_left: 0,
                            mv_left: None,
                            ref_left: 0,
                            row: r,
                            col: c,
                            forced_part: None,
                        });
                        // Write own blocks back: luma 64×64 at (sb_y,sb_x); chroma 32×32
                        // at (cy,cx). SAFETY: disjoint from every other worker's own block.
                        let hl = ly_end - sb_y;
                        let hc = cyc_end - cy;
                        let mut blk = [0f32; 64 * 64];
                        let gather =
                            |plane: &[f32],
                             stride: usize,
                             y0: usize,
                             x: usize,
                             w: usize,
                             h: usize,
                             blk: &mut [f32; 64 * 64]| {
                                for rr in 0..h {
                                    blk[rr * w..rr * w + w].copy_from_slice(
                                        &plane[(y0 + rr) * stride + x..(y0 + rr) * stride + x + w],
                                    );
                                }
                            };
                        gather(&s.ry, pw, sb_y, sb_x, 64, hl, &mut blk);
                        unsafe { wy.write_block(sb_y, sb_x, hl, 64, &blk[..hl * 64]) };
                        gather(&s.ru, pcw, cy, cx, 32, hc, &mut blk);
                        unsafe { wu.write_block(cy, cx, hc, 32, &blk[..hc * 32]) };
                        gather(&s.rv, pcw, cy, cx, 32, hc, &mut blk);
                        unsafe { wv.write_block(cy, cx, hc, 32, &blk[..hc * 32]) };
                        let edge = |v: &[i32], i: usize| v.get(i).copied().unwrap_or(-1);
                        let bottom: [i32; 16] = std::array::from_fn(|i| edge(&s.mia, mc0 + i));
                        let right: [i32; 16] = std::array::from_fn(|i| edge(&s.mil, mr0 + i));
                        // SAFETY: this SB's own mi rows/columns; no other cell writes them.
                        unsafe {
                            wm.write_block(mr0 + 15, mc0, 1, 16, &bottom);
                            wm.write_block(mr0, mc0 + 15, 16, 1, &right);
                        }
                        // Restore the buffer to all-zero: re-zero exactly what was dirtied
                        // (halo bands + own block) so the next cell sees all-zero-except-halo.
                        for rr in by..ly_end {
                            s.ry[rr * pw + bx..rr * pw + tx_end].fill(0.0);
                        }
                        for rr in cby..cyc_end {
                            s.ru[rr * pcw + cbx..rr * pcw + ctx_end].fill(0.0);
                            s.rv[rr * pcw + cbx..rr * pcw + ctx_end].fill(0.0);
                        }
                        if sb_y != 0 || sb_x != 0 {
                            for rr in 0..64.min(ph) {
                                s.ry[rr * pw..rr * pw + 64].fill(0.0);
                            }
                        }
                        WfSlot420 {
                            record: rec,
                            tx: cell_tx,
                            ch: cell_ch,
                        }
                    })
                },
            );
            // Merge per-SB records + deblock rectangles in raster order → the exact
            // sequence a serial Capture would have logged, so the serial Replay stays
            // byte-identical.
            if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
                for slot in slots.into_iter() {
                    let mut slot = slot.expect("every SB decided");
                    rec.append(slot.record);
                    tx_leaves.append(&mut slot.tx);
                    chroma_leaves.append(&mut slot.ch);
                }
            }
            for (i, cell) in aq_grid.iter().enumerate() {
                sb_qidx[i] = cell.qidx as u16;
            }
            return;
        }
        for row in 0..sb_rows {
            let mut skip_left = 0u8;
            let mut inter_left = 0u8;
            let mut newmv_left = 0u8;
            let mut mv_left: Option<Mv> = None;
            let mut ref_left = 0u8;
            left_pctx.iter_mut().for_each(|p| *p = 0);
            enc.sdp_left_c = [0; 16];
            for col in 0..sb_cols {
                if fresh_ctx {
                    // Reproduce the frame-initial (context-free) state a wavefront worker
                    // sees before each SB decide. The 4:2:0 inter-state grids stay all-zero
                    // on the stills path this check targets, so only the entropy / chroma /
                    // CfL neighbor arrays need resetting here.
                    above.iter_mut().for_each(|v| *v = 0x40);
                    left.iter_mut().for_each(|v| *v = 0x40);
                    above_pctx.iter_mut().for_each(|v| *v = 0);
                    left_pctx.iter_mut().for_each(|v| *v = 0);
                    enc.sdp_above_c.iter_mut().for_each(|v| *v = 0);
                    enc.sdp_left_c = [0; 16];
                    u_above.iter_mut().for_each(|v| *v = 0);
                    v_above.iter_mut().for_each(|v| *v = 0);
                    u_left.iter_mut().for_each(|v| *v = 0);
                    v_left.iter_mut().for_each(|v| *v = 0);
                    cfl_above.iter_mut().for_each(|v| *v = 0);
                    cfl_left.iter_mut().for_each(|v| *v = 0);
                    // midx_above / midx_left are NOT reset: wavefront cells are seeded
                    // with the neighbour SBs' mode classes (see `mode_map`).
                }
                let (sb_y, sb_x) = (row * 64, col * 64);
                if halo_mode {
                    recy_p.iter_mut().for_each(|v| *v = POISON);
                    recu_p.iter_mut().for_each(|v| *v = POISON);
                    recv_p.iter_mut().for_each(|v| *v = POISON);
                    copy_halo_luma(&mut recy_p, recy, sb_y, sb_x, true);
                    copy_halo_chroma(&mut recu_p, recu, sb_y, sb_x);
                    copy_halo_chroma(&mut recv_p, recv, sb_y, sb_x);
                }
                let (ry, ru, rv): (&mut [f32], &mut [f32], &mut [f32]) = if halo_mode {
                    (
                        recy_p.as_mut_slice(),
                        recu_p.as_mut_slice(),
                        recv_p.as_mut_slice(),
                    )
                } else {
                    (&mut *recy, &mut *recu, &mut *recv)
                };
                let (n_sl, n_il, n_nl, n_ml, n_rl) = self.decide_sb_420(Sb420Decision {
                    enc,
                    aqs: &mut aqs,
                    decide_mode: &mut decide_mode,
                    recy: ry,
                    recu: ru,
                    recv: rv,
                    yp,
                    up,
                    vp,
                    above,
                    left,
                    above_pctx,
                    left_pctx,
                    u_above,
                    v_above,
                    u_left,
                    v_left,
                    cfl_above: &mut cfl_above,
                    cfl_left: &mut cfl_left,
                    midx_above: &mut midx_above,
                    midx_left: &mut midx_left,
                    inter_mi_above: &mut inter_mi_above,
                    inter_mi_left: &mut inter_mi_left,
                    skip_mi_above: &mut skip_mi_above,
                    skip_mi_left: &mut skip_mi_left,
                    newmv_mi_above: &mut newmv_mi_above,
                    newmv_mi_left: &mut newmv_mi_left,
                    mv_mi_above: &mut mv_mi_above,
                    mv_mi_left: &mut mv_mi_left,
                    newmv_above: &mut newmv_above,
                    mv_above: &mut mv_above,
                    skip_above: &mut skip_above,
                    inter_above: &mut inter_above,
                    ref_above: &mut ref_above,
                    ref_mi_above: &mut ref_mi_above,
                    ref_mi_left: &mut ref_mi_left,
                    tx_leaves,
                    skip_leaves,
                    chroma_leaves,
                    sb_qidx,
                    last_ref: last_ref.as_slice(),
                    ref_x0,
                    ref_y0,
                    ref_ls,
                    ref_cs,
                    has_last,
                    second_ref: second_ref.as_slice(),
                    has_second,
                    inter_luma: inter_luma.as_ref(),
                    inter_chroma: inter_chroma.as_ref(),
                    me_scratch: &mut me_scratch,
                    inter_pred_scratch: &mut inter_pred_scratch,
                    frame_mv_seed,
                    mhccp_bounds,
                    pw,
                    pcw,
                    width,
                    height,
                    sb_cols,
                    tmc,
                    tmr,
                    qc,
                    neutral,
                    rdoq_lambda,
                    aq_grid: &aq_grid,
                    use_grid: fresh_ctx,
                    skip_left,
                    inter_left,
                    newmv_left,
                    mv_left,
                    ref_left,
                    row,
                    col,
                    forced_part: None,
                });
                skip_left = n_sl;
                inter_left = n_il;
                newmv_left = n_nl;
                mv_left = n_ml;
                ref_left = n_rl;
                if halo_mode {
                    copy_own_luma(recy, &recy_p, sb_y, sb_x);
                    copy_own_chroma(recu, &recu_p, sb_y, sb_x);
                    copy_own_chroma(recv, &recv_p, sb_y, sb_x);
                }
            }
        }
    }
    /// Decide + emit one 4:2:0 superblock (the extracted body of the `for col`
    /// walk loop). Behavior-preserving; byte-identical. Split out so the
    /// SB-wavefront can drive it per cell with private recon + fresh contexts.
    /// The per-row left-neighbor scalars are threaded in by value and returned.
    fn decide_sb_420(&self, decision: Sb420Decision<'_, '_>) -> (u8, u8, u8, Option<Mv>, u8) {
        let _qm = self.qm_scope(Layout::I420);
        let Sb420Decision {
            enc,
            aqs,
            mut decide_mode,
            recy,
            recu,
            recv,
            yp,
            up,
            vp,
            above,
            left,
            above_pctx,
            left_pctx,
            u_above,
            v_above,
            u_left,
            v_left,
            cfl_above,
            cfl_left,
            midx_above,
            midx_left,
            inter_mi_above,
            inter_mi_left,
            skip_mi_above,
            skip_mi_left,
            newmv_mi_above,
            newmv_mi_left,
            mv_mi_above,
            mv_mi_left,
            newmv_above,
            mv_above,
            skip_above,
            inter_above,
            ref_above,
            ref_mi_above,
            ref_mi_left,
            tx_leaves,
            skip_leaves,
            chroma_leaves,
            sb_qidx,
            last_ref,
            ref_x0,
            ref_y0,
            ref_ls,
            ref_cs,
            has_last,
            second_ref,
            has_second,
            inter_luma,
            inter_chroma,
            me_scratch,
            inter_pred_scratch,
            frame_mv_seed,
            mhccp_bounds,
            pw,
            pcw,
            width,
            height,
            sb_cols,
            tmc,
            tmr,
            qc,
            neutral,
            rdoq_lambda,
            aq_grid,
            use_grid,
            mut skip_left,
            mut inter_left,
            mut newmv_left,
            mut mv_left,
            mut ref_left,
            row,
            col,
            forced_part,
        } = decision;
        let _pal = crate::av2::palette::Scope::new(enc.sct);
        let bases = &self.bases;
        let cell = aq_grid[row * sb_cols + col];
        if enc.cdef_nb >= 2 {
            enc.cdef_pending = true;
            enc.cdef_sb_rc = (row, col);
        }
        // Probe this SB's AQ without advancing the delta-Q accumulator. A full-SB
        // inter skip carries no delta-Q syntax, so committing here would advance
        // the encoder's last_qidx while the decoder keeps the previous qindex.
        // Residual inter and the first intra leaf commit the same probe later.
        let (sb_qstep, sb_scale) = if use_grid {
            (cell.qs, cell.resid_scale)
        } else {
            let (q, r, _, _) = aqs.per_sb_probe(yp, pw, row * 64, col * 64, width, height);
            (q, r)
        };
        let (sb_y, sb_x) = (row * 64, col * 64);
        let full_interior = sb_x + 64 <= width && sb_y + 64 <= height;
        // Stills run the RD partition probe at every q (avmdec-conformant at all
        // qidx, 2026-10-31). Video keeps the old qidx <= 38 limit: above it the
        // probe would replace the geometry walker that carries the inter-skip paths.
        // Free binary-tree partition search (stills, separate chroma tree).
        let sdp_tree =
            self.still_rd() && self.sdp_still() && crate::av2::rd::knob_or("GTREE", 0.0) > 0.0;
        let luma_partition = if full_interior
            && self.tune.chroma_split
            && (self.base_q_idx <= 38
                || (self.still_rd()
                    && self.base_q_idx as f32 <= crate::av2::rd::knob_or("PARTQ", 255.0)))
        {
            // Replay: reuse the captured partition decision instead of re-running
            // the probe (the bulk of the decide cost). The probe restores the recon
            // it touches, and any split tree is a valid choice for the emitter.
            let replayed = if let crate::av2::replay::DecideMode::Replay(cur) = &mut decide_mode {
                cur.next_part()
            } else {
                None
            };
            if let Some(p) = replayed.or(forced_part) {
                p
            } else if sdp_tree {
                let lambda = leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c())
                    * crate::av2::rd::knob_or("GLAM", 1.0);
                let g = self.gtree_search(
                    recy,
                    &GSearch {
                        yp,
                        pw,
                        frame: (width, height),
                        qstep: sb_qstep,
                        lambda,
                        sb: (sb_y, sb_x),
                        guide: None,
                    },
                );
                if std::env::var("MT_GDBG").is_ok() {
                    eprintln!("GTREE sb ({row},{col}) {:?}", g.codes());
                }
                let p = LumaPartitionDecision {
                    gtree: Some(g),
                    ..Default::default()
                };
                if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
                    rec.push_part(p);
                }
                p
            } else {
                let p = choose_luma_64x64_partition(
                    recy,
                    &LumaSource {
                        plane: yp,
                        stride: pw,
                    },
                    &LumaFrameBlock {
                        frame_width: width,
                        frame_height: height,
                        y: sb_y,
                        x: sb_x,
                    },
                    &LumaGridBlock {
                        mi_cols: tmc,
                        mi_rows: tmr,
                        y: sb_y,
                        x: sb_x,
                    },
                    &LumaPartitionSearch {
                        quant: LumaQuantSpec {
                            prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                            basis: &bases.luma,
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda,
                            pixel_lambda: self.still_rd(),
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                        sb: LumaSbSearch {
                            residual_scale: sb_scale,
                            allow_directional: self.speed.try_directional(),
                        },
                        basis16: &bases.luma16x16,
                        basis8: &bases.c8x8,
                        allow_16x16: self.video_allows_16x16_partitions(),
                        // With SDP chroma has its own tree, so 8x8 luma leaves are free at
                        // every q. On the shared tree they pay off at qindex <= 128 only:
                        // below that their 4x4 chroma loses to the 16x16 leaf's MHCCP.
                        allow_8x8: self.still_rd() && (self.sdp_still() || self.base_q_idx <= 128),
                        // Shared tree: the probe prices both chroma candidates with DC, so it
                        // cannot see the 16x16 leaf's MHCCP; weighting the 4x4 chroma SSE
                        // stands in.
                        split8_chroma_weight: 1.5,
                        leaf_modes: self.still_rd(),
                        leaf_modes8: self.still_rd(),
                    },
                    // Shared-tree stills price chroma in the partition probe; with SDP the
                    // chroma tree decides chroma, and video keeps luma-only.
                    (self.still_rd() && crate::av2::rd::knob("CPART") > 0.0 && !self.sdp_still())
                        .then_some(&mut crate::av2::leaf::ChromaProbe {
                            src_u: up,
                            src_v: vp,
                            rec_u: &mut *recu,
                            rec_v: &mut *recv,
                            stride: pcw,
                            bases: [&bases.chroma420, &bases.luma16x16, &bases.c8x8, &bases.c4x4],
                            full: false,
                        }),
                    // Stills price partition symbols exactly from the live CDFs.
                    self.still_rd().then_some(&partition::PartitionPricer {
                        above_pctx: above_pctx.to_vec(),
                        left_pctx: left_pctx.to_vec(),
                        sb_row: row,
                        sb_col: col,
                        mi_rows: tmr as usize,
                        mi_cols: tmc as usize,
                    }),
                );
                // ExtraSlow SB partition trial: encode the SB for real under partitions
                // around the probe's pick (the probe's own pick survives ~half the
                // time) and keep the cheapest D + lambda * bits. -1.35 SS2 tuning /
                // -1.43 holdout at 4:2:0 for ~3.9x Slow instructions.
                let p = if self.still_rd() && matches!(self.speed, Speed::ExtraSlow) {
                    let mut cands = vec![p];
                    let mut push = |c: LumaPartitionDecision| {
                        let same = |a: &LumaPartitionDecision| {
                            a.split64 == c.split64
                                && a.rect64 == c.rect64
                                && (!c.split64
                                    || (a.split32 == c.split32
                                        && a.split16 == c.split16
                                        && a.rect32 == c.rect32
                                        && a.rect16 == c.rect16))
                        };
                        if !cands.iter().any(same) {
                            cands.push(c);
                        }
                    };
                    push(LumaPartitionDecision::default());
                    push(LumaPartitionDecision {
                        split16: [false; 16],
                        ..p
                    });
                    push(LumaPartitionDecision {
                        split64: true,
                        ..Default::default()
                    });
                    push(LumaPartitionDecision {
                        split64: true,
                        split32: [true; 4],
                        ..p
                    });
                    // Half the leaf lambda: measured -1.35 SS2 vs -0.84 at 1x (2 imgs).
                    let lambda = 0.5 * leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
                    let sse_plane = |rec: &[f32],
                                     src: &[f32],
                                     stride: usize,
                                     y: usize,
                                     x: usize,
                                     n: usize|
                     -> f32 {
                        rect_rows(rec, stride, y, x, n, n)
                            .zip(rect_rows(src, stride, y, x, n, n))
                            .map(|(r, s)| {
                                r.iter()
                                    .zip(s)
                                    .map(|(&a, &b)| (a.round() - b).powi(2))
                                    .sum::<f32>()
                            })
                            .sum()
                    };
                    let save =
                        |pl: &[f32], stride: usize, y: usize, x: usize, n: usize| -> Vec<f32> {
                            rect_rows(pl, stride, y, x, n, n)
                                .flatten()
                                .copied()
                                .collect()
                        };
                    let put =
                        |pl: &mut [f32], stride: usize, y: usize, x: usize, n: usize, v: &[f32]| {
                            for (dst, src) in
                                rect_rows_mut(pl, stride, y, x, n, n).zip(v.chunks_exact(n))
                            {
                                dst.copy_from_slice(src);
                            }
                        };
                    let (ccy, ccx) = (sb_y / 2, sb_x / 2);
                    let ry0 = save(recy, pw, sb_y, sb_x, 64);
                    let ru0 = save(recu, pcw, ccy, ccx, 32);
                    let rv0 = save(recv, pcw, ccy, ccx, 32);
                    let s_above = above.to_vec();
                    let s_left = left.to_vec();
                    let s_above_pctx = above_pctx.to_vec();
                    let s_left_pctx = left_pctx.to_vec();
                    let s_u_above = u_above.to_vec();
                    let s_v_above = v_above.to_vec();
                    let s_u_left = u_left.to_vec();
                    let s_v_left = v_left.to_vec();
                    let s_cfl_above = cfl_above.to_vec();
                    let s_cfl_left = cfl_left.to_vec();
                    let s_midx_above = midx_above.to_vec();
                    let s_midx_left = midx_left.to_vec();
                    let s_inter_mi_above = inter_mi_above.to_vec();
                    let s_inter_mi_left = inter_mi_left.to_vec();
                    let s_skip_mi_above = skip_mi_above.to_vec();
                    let s_skip_mi_left = skip_mi_left.to_vec();
                    let s_newmv_mi_above = newmv_mi_above.to_vec();
                    let s_newmv_mi_left = newmv_mi_left.to_vec();
                    let s_mv_mi_above = mv_mi_above.to_vec();
                    let s_mv_mi_left = mv_mi_left.to_vec();
                    let s_newmv_above = newmv_above.to_vec();
                    let s_mv_above = mv_above.to_vec();
                    let s_skip_above = skip_above.to_vec();
                    let s_inter_above = inter_above.to_vec();
                    let s_ref_above = ref_above.to_vec();
                    let s_ref_mi_above = ref_mi_above.to_vec();
                    let s_ref_mi_left = ref_mi_left.to_vec();
                    let s_sb_qidx = sb_qidx.to_vec();
                    let (n_tx, n_skip, n_ch) =
                        (tx_leaves.len(), skip_leaves.len(), chroma_leaves.len());
                    let bits0 = enc.tell_bits();
                    let mut eval = |c: LumaPartitionDecision| -> f32 {
                        let mut te = enc.trial_clone();
                        let mut ta = aqs.clone();
                        let mut off = crate::av2::replay::DecideMode::Off;
                        self.decide_sb_420(Sb420Decision {
                            enc: &mut te,
                            aqs: &mut ta,
                            decide_mode: &mut off,
                            recy: &mut *recy,
                            recu: &mut *recu,
                            recv: &mut *recv,
                            yp,
                            up,
                            vp,
                            above: &mut *above,
                            left: &mut *left,
                            above_pctx: &mut *above_pctx,
                            left_pctx: &mut *left_pctx,
                            u_above: &mut *u_above,
                            v_above: &mut *v_above,
                            u_left: &mut *u_left,
                            v_left: &mut *v_left,
                            cfl_above: &mut *cfl_above,
                            cfl_left: &mut *cfl_left,
                            midx_above: &mut *midx_above,
                            midx_left: &mut *midx_left,
                            inter_mi_above: &mut *inter_mi_above,
                            inter_mi_left: &mut *inter_mi_left,
                            skip_mi_above: &mut *skip_mi_above,
                            skip_mi_left: &mut *skip_mi_left,
                            newmv_mi_above: &mut *newmv_mi_above,
                            newmv_mi_left: &mut *newmv_mi_left,
                            mv_mi_above: &mut *mv_mi_above,
                            mv_mi_left: &mut *mv_mi_left,
                            newmv_above: &mut *newmv_above,
                            mv_above: &mut *mv_above,
                            skip_above: &mut *skip_above,
                            inter_above: &mut *inter_above,
                            ref_above: &mut *ref_above,
                            ref_mi_above: &mut *ref_mi_above,
                            ref_mi_left: &mut *ref_mi_left,
                            tx_leaves: &mut *tx_leaves,
                            skip_leaves: &mut *skip_leaves,
                            chroma_leaves: &mut *chroma_leaves,
                            sb_qidx: &mut *sb_qidx,
                            last_ref,
                            ref_x0,
                            ref_y0,
                            ref_ls,
                            ref_cs,
                            has_last,
                            second_ref,
                            has_second,
                            inter_luma,
                            inter_chroma,
                            me_scratch: &mut *me_scratch,
                            inter_pred_scratch: &mut *inter_pred_scratch,
                            frame_mv_seed,
                            mhccp_bounds,
                            pw,
                            pcw,
                            width,
                            height,
                            sb_cols,
                            tmc,
                            tmr,
                            qc,
                            neutral,
                            rdoq_lambda,
                            aq_grid,
                            use_grid,
                            skip_left,
                            inter_left,
                            newmv_left,
                            mv_left,
                            ref_left,
                            row,
                            col,
                            forced_part: Some(c),
                        });
                        let d = sse_plane(recy, yp, pw, sb_y, sb_x, 64)
                            + sse_plane(recu, up, pcw, ccy, ccx, 32)
                            + sse_plane(recv, vp, pcw, ccy, ccx, 32);
                        let j = d + lambda * (te.tell_bits() - bits0);
                        put(recy, pw, sb_y, sb_x, 64, &ry0);
                        put(recu, pcw, ccy, ccx, 32, &ru0);
                        put(recv, pcw, ccy, ccx, 32, &rv0);
                        above.copy_from_slice(&s_above);
                        left.copy_from_slice(&s_left);
                        above_pctx.copy_from_slice(&s_above_pctx);
                        left_pctx.copy_from_slice(&s_left_pctx);
                        u_above.copy_from_slice(&s_u_above);
                        v_above.copy_from_slice(&s_v_above);
                        u_left.copy_from_slice(&s_u_left);
                        v_left.copy_from_slice(&s_v_left);
                        cfl_above.copy_from_slice(&s_cfl_above);
                        cfl_left.copy_from_slice(&s_cfl_left);
                        midx_above.copy_from_slice(&s_midx_above);
                        midx_left.copy_from_slice(&s_midx_left);
                        inter_mi_above.copy_from_slice(&s_inter_mi_above);
                        inter_mi_left.copy_from_slice(&s_inter_mi_left);
                        skip_mi_above.copy_from_slice(&s_skip_mi_above);
                        skip_mi_left.copy_from_slice(&s_skip_mi_left);
                        newmv_mi_above.copy_from_slice(&s_newmv_mi_above);
                        newmv_mi_left.copy_from_slice(&s_newmv_mi_left);
                        mv_mi_above.copy_from_slice(&s_mv_mi_above);
                        mv_mi_left.copy_from_slice(&s_mv_mi_left);
                        newmv_above.copy_from_slice(&s_newmv_above);
                        mv_above.copy_from_slice(&s_mv_above);
                        skip_above.copy_from_slice(&s_skip_above);
                        inter_above.copy_from_slice(&s_inter_above);
                        ref_above.copy_from_slice(&s_ref_above);
                        ref_mi_above.copy_from_slice(&s_ref_mi_above);
                        ref_mi_left.copy_from_slice(&s_ref_mi_left);
                        sb_qidx.copy_from_slice(&s_sb_qidx);
                        tx_leaves.truncate(n_tx);
                        skip_leaves.truncate(n_skip);
                        chroma_leaves.truncate(n_ch);
                        j
                    };
                    let mut best = (f32::INFINITY, p);
                    for c in cands {
                        let j = eval(c);
                        if j < best.0 {
                            best = (j, c);
                        }
                    }
                    // Greedy refinement: flip each 32x32 quadrant's split, then each
                    // split quadrant's 8x8 splits, keeping every flip that lowers J.
                    if best.1.split64 {
                        let q16 = |k: usize| {
                            let (qy, qx) = (k / 2, k % 2);
                            [0usize, 1, 4, 5].map(|o| qy * 8 + qx * 2 + o)
                        };
                        for k in 0..4 {
                            let mut c = best.1;
                            c.split32[k] = !c.split32[k];
                            for i in q16(k) {
                                c.split16[i] = false;
                            }
                            let j = eval(c);
                            if j < best.0 {
                                best = (j, c);
                            }
                        }
                        if self.base_q_idx <= 128 {
                            for k in 0..4 {
                                if !best.1.split32[k] {
                                    continue;
                                }
                                let mut c = best.1;
                                let any = q16(k).iter().any(|&i| c.split16[i]);
                                for i in q16(k) {
                                    c.split16[i] = !any;
                                }
                                let j = eval(c);
                                if j < best.0 {
                                    best = (j, c);
                                }
                            }
                        }
                    }
                    // Rect leaf pairs where a node stays unsplit: 64x32 / 32x64 for a
                    // whole-64 SB, 32x16 / 16x32 per unsplit 32x32 quadrant.
                    // Measured neutral on s_mascot (rect32 picked in ~10% of SBs, almost all
                    // DC-only 32x16/16x32 leaves); interior 64x32/32x64 pairs mismatch the
                    // decoder recon at 8-12 bit (open bug, see memory) -> rect64 masked out.
                    if crate::av2::rd::knob_or("RECT", 0.0) > 0.0 {
                        let mask = crate::av2::rd::knob_or("RMASK", 12.0) as u32;
                        if !best.1.split64 {
                            for r in [1u8, 2] {
                                if mask & (r as u32) == 0 {
                                    continue;
                                }
                                let c = LumaPartitionDecision {
                                    rect64: r,
                                    ..best.1
                                };
                                let j = eval(c);
                                if j < best.0 {
                                    best = (j, c);
                                }
                            }
                        } else {
                            for k in 0..4 {
                                if best.1.split32[k] {
                                    continue;
                                }
                                for r in [1u8, 2] {
                                    if mask & ((r as u32) << 2) == 0 {
                                        continue;
                                    }
                                    let mut c = best.1;
                                    c.rect32[k] = r;
                                    let j = eval(c);
                                    if j < best.0 {
                                        best = (j, c);
                                    }
                                }
                            }
                        }
                    }
                    best.1
                } else {
                    p
                };
                // Rect refinement of the probe's (or trial's) leaves by the binary-tree search.
                let p = if self.still_rd()
                    && self.sdp_still()
                    && crate::av2::rd::knob_or(
                        "GPROBE",
                        if matches!(self.speed, Speed::ExtraSlow) {
                            1.0
                        } else {
                            0.0
                        },
                    ) > 0.0
                {
                    let lambda = leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c())
                        * crate::av2::rd::knob_or("GLAM", 1.0);
                    let g = self.gtree_search(
                        recy,
                        &GSearch {
                            yp,
                            pw,
                            frame: (width, height),
                            qstep: sb_qstep,
                            lambda,
                            sb: (sb_y, sb_x),
                            guide: Some(p),
                        },
                    );
                    LumaPartitionDecision {
                        gtree: Some(g),
                        dir8: p.dir8,
                        ..Default::default()
                    }
                } else {
                    p
                };
                if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
                    rec.push_part(p);
                }
                p
            }
        } else {
            LumaPartitionDecision::default()
        };
        let ops = if full_interior && luma_partition.is_tree() {
            match &luma_partition.gtree {
                Some(g) => partition::sb_gtree_ops(row, col, g.codes(), above_pctx, left_pctx),
                None => partition::sb_tree_ops(row, col, &luma_partition, above_pctx, left_pctx),
            }
        } else {
            partition::sb_partition_ops(row, col, tmr as usize, tmc as usize, above_pctx, left_pctx)
        };
        enc.delta_q_pending = enc.delta_q_present;
        let mut aq_committed = false;
        // Reset the per-SB coded-mi mask consumed by MHCCP so top-right /
        // bottom-left reference availability follows the true coding order.
        enc.sb_coded = [0u8; 256];
        // Staged decouple: pop this SB's captured walk (Replay), or start a
        // fresh accumulator (Capture). Only a pure whole-64 `(16,16)` intra
        // leaf is captured; any other shape (or inter-coded SB) leaves
        // `sb_walk_ok = false`, recording `Fallback` (replay re-searches it,
        // byte-safe because serial replay recon == Off recon there).
        use crate::av2::replay::{DecideMode, Leaf420, Sb420};
        let replay_walk: Option<Vec<Leaf420>> = if let DecideMode::Replay(cur) = &mut decide_mode {
            match cur.next_sb420() {
                Some(Sb420::Walk(v)) => Some(v.clone()),
                _ => None,
            }
        } else {
            None
        };
        let capturing = matches!(decide_mode, DecideMode::Capture(_));
        let mut leaf_recs: Vec<Leaf420> = Vec::new();
        let mut sb_walk_ok = false;
        let mut cap_idx = 0usize;
        // SDP: this walk is the luma tree (no chroma syntax, chroma contexts left
        // to the chroma tree, which runs after it).
        let sdp = self.sdp_still();
        enc.sdp_luma = sdp;
        let mut sdp_midx = [0xffu8; 256];
        let leaf_start = tx_leaves.len();
        // TU geometry of split rect leaves, appended to `tx_leaves` (the deblock
        // grid) after the SB: the chroma tree reads this SB's prediction blocks.
        let mut tu_splits: Vec<(usize, usize, usize, usize)> = Vec::new();
        for op in &ops {
            let (bw_mi, bh_mi, pc, lmr, lmc) = match op {
                partition::Op::RectType { cdf, val, ctx } => {
                    enc.bool_rect_type(*cdf, *val, *ctx);
                    continue;
                }
                partition::Op::Split {
                    do_split_cdf,
                    square_cdf,
                } => {
                    enc.bool_do_split(*do_split_cdf, 1);
                    if *square_cdf != 0 {
                        enc.bool_do_square_split(*square_cdf, 1);
                    }
                    continue;
                }
                partition::Op::Leaf {
                    bw_mi,
                    bh_mi,
                    part_cdf,
                    mi_row,
                    mi_col,
                } => (*bw_mi, *bh_mi, part_cdf.unwrap_or(12276), *mi_row, *mi_col),
            };
            tx_leaves.push((lmr, lmc, bw_mi, bh_mi));
            enc.pal_pos = (lmr, lmc);
            #[cfg(test)]
            TOTAL_LEAF_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let sb_y = lmr * 4;
            let sb_x = lmc * 4;
            // Snapshot the four AVM line-buffer probes before the default-rank
            // commit below overwrites this leaf's above/left spans. In particular,
            // `ref_mi_left[lmr..]` still belongs to the immediate-left block at
            // this point; clearing it early made non-aligned edge leaves treat a
            // rank-1 left neighbor as rank 0.
            let ar_mi = (lmc + bw_mi - 1).min(ref_mi_above.len() - 1);
            let bl_mi = (lmr + bh_mi - 1).min(ref_mi_left.len() - 1);
            let bl_ref_rank = ref_mi_left[bl_mi];
            let ar_ref_rank = ref_mi_above[ar_mi];
            let left_ref_rank = ref_mi_left[lmr];
            let above_ref_rank = ref_mi_above[lmc];
            // intra_inter ctx: AVM fills 2 neighbor slots when up OR left
            // available (above+above_right, or left+bottom_left), so all-intra
            // gives 3 for any edge; 0 only at the top-left corner.
            // intra_inter ctx: derive from actual neighbor inter/intra state
            // (AVM av2_get_intra_inter_context). AVM fills two line-buffer slots
            // by scanning bottom_left -> above_right -> left -> above and keeping
            // the first two available; on an inter frame an inter neighbor changes
            // the context away from the all-intra value of 3. Self-contained so it
            // needs no coder-side helper.
            {
                let up_av = lmr > 0;
                let lf_av = lmc > 0;
                let ar_c = (lmc + bw_mi - 1).min(inter_mi_above.len() - 1);
                let bl_r = (lmr + bh_mi - 1).min(inter_mi_left.len() - 1);
                // Each candidate as Option<Option<usize>> in AVM priority order:
                // None = unavailable, Some(None) = intra neighbor,
                // Some(Some(rank)) = inter neighbor predicting from that rank.
                let cand = |avail: bool, inter: u8, rank: u8| {
                    avail.then(|| (inter == 1).then_some(rank as usize))
                };
                let bottom_left = cand(lf_av, inter_mi_left[bl_r], bl_ref_rank);
                let above_right = cand(up_av, inter_mi_above[ar_c], ar_ref_rank);
                let left = cand(lf_av, inter_mi_left[lmr], left_ref_rank);
                let above = cand(up_av, inter_mi_above[lmc], above_ref_rank);
                let mut slots: [Option<Option<usize>>; 2] = [None, None];
                let mut i = 0;
                for n in [bottom_left, above_right, left, above]
                    .into_iter()
                    .flatten()
                {
                    slots[i] = Some(n);
                    i += 1;
                    if i == 2 {
                        break;
                    }
                }
                enc.intra_inter_ctx = match (slots[0], slots[1]) {
                    (Some(a), Some(b)) => {
                        let a_intra = a.is_none();
                        let b_intra = b.is_none();
                        if a_intra && b_intra {
                            3
                        } else {
                            (a_intra || b_intra) as usize
                        }
                    }
                    (Some(n), None) | (None, Some(n)) => 2 * n.is_none() as usize,
                    (None, None) => 0,
                };
                // Single-ref rank-bit ctx (n_refs=2 frames): av2_get_ref_pred_context
                // over the same two resolved line-buffer slots. Each distinct slot
                // counts once (AVM neighbors_ref_counts iterates the 2-entry buffer);
                // rank-0 count vs rank-1 count gives 0 (A<B) / 1 (A==B) / 2 (A>B).
                let mut rank_counts = [0u32; 2];
                for rank in slots.into_iter().flatten().flatten() {
                    rank_counts[rank.min(1)] += 1;
                }
                enc.ref_bit_ctx = match rank_counts[0].cmp(&rank_counts[1]) {
                    std::cmp::Ordering::Less => 0,
                    std::cmp::Ordering::Equal => 1,
                    std::cmp::Ordering::Greater => 2,
                };
                enc.ref_rank = 0;
            }
            // Capture the SB-granular above/left neighbor ranks BEFORE the reset
            // below overwrites this column/row. The whole-64 skip's same-rank mode
            // context reads these; reading the post-reset zeros would disagree with
            // the decoder (which sees the real neighbor rank) and desync.
            let sb_above_rank = ref_above[col];
            let sb_left_rank = ref_left;
            // Reset this block's reference rank to 0 across both grids. Every
            // inter commit except the whole-64 rank-1 skip predicts from rank 0,
            // so clearing here and letting only that branch overwrite keeps the
            // per-block rank correct without touching every commit site (the
            // ref-bit context only reads a rank where inter_mi is set). The reset
            // also propagates the correct rank forward: a non-skip SB leaves 0, a
            // rank-1 skip overwrites with 1, so the next block's neighbor read is
            // right.
            for c in lmc..(lmc + bw_mi).min(ref_mi_above.len()) {
                ref_mi_above[c] = 0;
            }
            for r in lmr..(lmr + bh_mi).min(ref_mi_left.len()) {
                ref_mi_left[r] = 0;
            }
            ref_above[col] = 0;
            ref_left = 0;
            // avm get_entropy_context_1d checks whether ANY 4x4 unit along
            // the block's above/left edge is nonzero (not just the first).
            // A single-entry read desyncs the chroma txb-skip context when a
            // neighboring SB is partitioned into sub-blocks with mixed
            // has-coeffs along the shared edge (the CDF then drifts and large
            // single-tile images fail to decode). Scan the full span instead.
            let any = |g: &[i32], s: usize, n: usize| {
                g[s..(s + n).min(g.len())].iter().any(|&x| x != 0) as i32
            };
            let ua = if lmr > 0 { any(u_above, lmc, bw_mi) } else { 0 };
            let ul = if lmc > 0 { any(u_left, lmr, bh_mi) } else { 0 };
            let va = if lmr > 0 { any(v_above, lmc, bw_mi) } else { 0 };
            let vl = if lmc > 0 { any(v_left, lmr, bh_mi) } else { 0 };
            // 4:2:0 chroma origin: half the luma origin in BOTH axes.
            let (cy, cx) = (sb_y / 2, sb_x / 2);
            // CfL neighbor context + per-leaf default (every eligible leaf emits an
            // is_cfl bit; only the whole-64 leaf may set it).
            let cfl_a = if lmr > 0 { cfl_above[lmc] } else { 0 };
            let cfl_l = if lmc > 0 { cfl_left[lmr] } else { 0 };
            enc.cfl_ctx = (cfl_a + cfl_l) as usize;
            enc.cfl_use = false;
            enc.cfl_signaled = false;
            // Luma mode-set neighbours (avm above_right / bottom_left mi) and the
            // y_idx context = how many of them are directional.
            let nb_midx = |v: i32| if v >= 0 { v as u8 } else { 0xff };
            let amidx = nb_midx(if lmr > 0 {
                midx_above[lmc + bw_mi - 1]
            } else {
                -1
            });
            let lmidx = nb_midx(if lmc > 0 {
                midx_left[lmr + bh_mi - 1]
            } else {
                -1
            });
            enc.y_ctx = (amidx != 0xff) as usize + (lmidx != 0xff) as usize;
            let mut leaf_midx: i32 = -1;
            // Sub-superblock GLOBALMV skip, including native rectangular edge
            // leaves. Chroma uses the same zero motion at half resolution.
            let block_w = bw_mi * 4;
            let block_h = bh_mi * 4;
            let subblock_inter = outline_leaf_420(|| {
                if has_last
                    && (bw_mi != 16 || bh_mi != 16)
                    && sb_x + block_w <= width
                    && sb_y + block_h <= height
                {
                    let chroma_w = block_w / 2;
                    let chroma_h = block_h / 2;
                    let mut sse = rect_sse_f32(
                        &PlaneRect {
                            plane: yp,
                            stride: pw,
                            y: sb_y,
                            x: sb_x,
                        },
                        &PlaneRect {
                            plane: &last_ref[0],
                            stride: ref_ls,
                            y: ref_y0 + sb_y,
                            x: ref_x0 + sb_x,
                        },
                        block_w,
                        block_h,
                    );
                    let (rcy, rcx) = (ref_y0 / 2 + cy, ref_x0 / 2 + cx);
                    for (src, reference) in [(&up, &last_ref[1]), (&vp, &last_ref[2])] {
                        sse += rect_sse_f32(
                            &PlaneRect {
                                plane: src,
                                stride: pcw,
                                y: cy,
                                x: cx,
                            },
                            &PlaneRect {
                                plane: reference,
                                stride: ref_cs,
                                y: rcy,
                                x: rcx,
                            },
                            chroma_w,
                            chroma_h,
                        );
                    }

                    let ar = (lmc + bw_mi - 1).min(skip_mi_above.len() - 1);
                    let bl = (lmr + bh_mi - 1).min(skip_mi_left.len() - 1);
                    let neighbors = [
                        (lmc > 0).then_some((
                            skip_mi_left[bl],
                            inter_mi_left[bl],
                            newmv_mi_left[bl],
                        )),
                        (lmr > 0).then_some((
                            skip_mi_above[ar],
                            inter_mi_above[ar],
                            newmv_mi_above[ar],
                        )),
                        (lmc > 0).then_some((
                            skip_mi_left[lmr],
                            inter_mi_left[lmr],
                            newmv_mi_left[lmr],
                        )),
                        (lmr > 0).then_some((
                            skip_mi_above[lmc],
                            inter_mi_above[lmc],
                            newmv_mi_above[lmc],
                        )),
                    ];
                    let mut chosen = [(0u8, 0u8, 0u8); 2];
                    let mut count = 0;
                    for neighbor in neighbors.into_iter().flatten() {
                        chosen[count] = neighbor;
                        count += 1;
                        if count == 2 {
                            break;
                        }
                    }
                    let skip_ctx = chosen[..count].iter().map(|n| n.0 as usize).sum();
                    // av2_find_mode_ctx collapses both probes on an axis to one
                    // row/column match; skip_txfm instead sums the two selected
                    // line-buffer entries above.
                    // This block predicts from rank 0, so AVM's same-reference mode ctx
                    // counts only rank-0 inter neighbors (a rank-1 neighbor, from a
                    // whole-64 skip that chose the second reference, is excluded). The
                    // rank filter is a no-op when no rank-1 block exists.
                    let left_match = lmc > 0
                        && ((inter_mi_left[bl] != 0 && bl_ref_rank == 0)
                            || (inter_mi_left[lmr] != 0 && left_ref_rank == 0));
                    let above_match = lmr > 0
                        && ((inter_mi_above[ar] != 0 && ar_ref_rank == 0)
                            || (inter_mi_above[lmc] != 0 && above_ref_rank == 0));
                    let nearest_match = usize::from(left_match) + usize::from(above_match);
                    let any_newmv = (lmc > 0
                        && ((newmv_mi_left[bl] != 0 && bl_ref_rank == 0)
                            || (newmv_mi_left[lmr] != 0 && left_ref_rank == 0)))
                        || (lmr > 0
                            && ((newmv_mi_above[ar] != 0 && ar_ref_rank == 0)
                                || (newmv_mi_above[lmc] != 0 && above_ref_rank == 0)));
                    let mode_ctx = nearest_match + 2 * usize::from(any_newmv);
                    let rate = crate::av2::video::rd::inter_syntax_bits(
                        enc, skip_ctx, mode_ctx, true, 1, None,
                    );
                    let cost = crate::av2::video::rd::rd_cost(
                        sse * crate::av2::video::rd::SS2_INTER_DIST_W,
                        rate,
                        sb_qstep as u32,
                    );
                    let intra_bound = crate::av2::video::rd::rd_cost(
                        0.0,
                        2.0 * (block_w * block_h) as f32,
                        sb_qstep as u32,
                    );
                    if cost < intra_bound {
                        #[cfg(test)]
                        if block_w == 32 && block_h == 32 {
                            INTER_SKIP_32_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        } else if block_w != block_h {
                            INTER_SKIP_RECT_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        if !aq_committed {
                            if use_grid {
                                enc.delta_q_signaled = cell.sig;
                            } else {
                                let _ = aqs.per_sb(enc, yp, pw, row * 64, col * 64, width, height);
                            }
                            aq_committed = true;
                        }
                        crate::av2::coder::emit_inter_skip_leaf(enc, pc, skip_ctx, mode_ctx, false);
                        skip_leaves.push((lmr, lmc, bw_mi, bh_mi));
                        for (dst, src) in
                            rect_rows_mut(recy, pw, sb_y, sb_x, block_w, block_h).zip(rect_rows(
                                &last_ref[0],
                                ref_ls,
                                ref_y0 + sb_y,
                                ref_x0 + sb_x,
                                block_w,
                                block_h,
                            ))
                        {
                            dst.copy_from_slice(src);
                        }
                        let dst_rows = rect_rows_mut(recu, pcw, cy, cx, chroma_w, chroma_h)
                            .zip(rect_rows_mut(recv, pcw, cy, cx, chroma_w, chroma_h));
                        let src_rows =
                            rect_rows(&last_ref[1], ref_cs, rcy, rcx, chroma_w, chroma_h).zip(
                                rect_rows(&last_ref[2], ref_cs, rcy, rcx, chroma_w, chroma_h),
                            );
                        for ((du, dv), (su, sv)) in dst_rows.zip(src_rows) {
                            du.copy_from_slice(su);
                            dv.copy_from_slice(sv);
                        }
                        for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                            inter_mi_above[c] = 1;
                            skip_mi_above[c] = 1;
                            newmv_mi_above[c] = 0;
                            mv_mi_above[c] = Some(Mv::ZERO);
                            above[c] = 0x40;
                            u_above[c] = 0;
                            v_above[c] = 0;
                        }
                        for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                            inter_mi_left[r] = 1;
                            skip_mi_left[r] = 1;
                            newmv_mi_left[r] = 0;
                            mv_mi_left[r] = Some(Mv::ZERO);
                            left[r] = 0x40;
                            u_left[r] = 0;
                            v_left[r] = 0;
                        }
                        skip_above[col] = 1;
                        skip_left = 1;
                        inter_above[col] = 1;
                        inter_left = 1;
                        newmv_above[col] = 0;
                        newmv_left = 0;
                        mv_above[col] = Some(Mv::ZERO);
                        mv_left = Some(Mv::ZERO);
                        return true;
                    }

                    // Mirror the adjacent spatial scan used for DRL[0]: bottom-left
                    // edge, above-right edge, immediate left, immediate above. A
                    // block spanning an edge appears twice but carries the same MV.
                    // Search both the square dense-inter leaves and the rectangular
                    // leaves already produced by the partition walker. Rectangles
                    // initially compete as motion-only skip blocks; residual coding
                    // remains on the bit-exact square TX paths below.
                    let motion_leaf = (block_w == block_h && matches!(block_w, 16 | 32))
                        || (block_w != block_h
                            && matches!(block_w, 16 | 32 | 64)
                            && matches!(block_h, 16 | 32 | 64));
                    if motion_leaf {
                        // DRL[0] scans same-reference (rank-0) neighbors only: AVM's
                        // setup_ref_mv_list gates spatial candidates on ref==rf, so a
                        // rank-1 neighbor (a whole-64 skip that chose the second
                        // reference, always zero motion) is skipped. Its AVM "derived"
                        // contribution would be the zero MV projected — i.e. zero — which
                        // is exactly the ZERO fallback, so no explicit derived term is
                        // needed. No-op when no rank-1 block exists.
                        let pred_mv = [
                            (lmc > 0 && bl_ref_rank == 0)
                                .then_some(mv_mi_left[bl])
                                .flatten(),
                            (lmr > 0 && ar_ref_rank == 0)
                                .then_some(mv_mi_above[ar])
                                .flatten(),
                            (lmc > 0 && left_ref_rank == 0)
                                .then_some(mv_mi_left[lmr])
                                .flatten(),
                            (lmr > 0 && above_ref_rank == 0)
                                .then_some(mv_mi_above[lmc])
                                .flatten(),
                        ]
                        .into_iter()
                        .flatten()
                        .next()
                        .unwrap_or(Mv::ZERO);
                        if let Some(candidate) = self.search_inter_leaf_420(InterLeafSearch420 {
                            enc,
                            source_y: yp,
                            source_u: up,
                            source_v: vp,
                            luma_reference: inter_luma.expect("LAST luma prepared for inter"),
                            chroma_references: inter_chroma
                                .expect("LAST chroma prepared for inter"),
                            me_scratch,
                            prediction_scratch: inter_pred_scratch,
                            source_stride: pw,
                            chroma_stride: pcw,
                            reference_x: ref_x0,
                            reference_y: ref_y0,
                            reference_luma_stride: ref_ls,
                            block_x: sb_x,
                            block_y: sb_y,
                            block_width: block_w,
                            block_height: block_h,
                            predictor_mv: pred_mv,
                            frame_mv_seed,
                            skip_ctx,
                            mode_ctx,
                            qstep: sb_qstep,
                        }) {
                            let InterLeafCandidate420 {
                                mv,
                                mode: inter_mode,
                                mvd: scaled_mvd,
                                rd_cost: motion_cost,
                            } = candidate;
                            let (mvd_row, mvd_col) =
                                scaled_mvd.map(|mvd| (mvd.row, mvd.col)).unwrap_or((0, 0));
                            let (pred_y, pred_u, pred_v, _) =
                                inter_pred_scratch.planes(block_w * block_h, chroma_w * chroma_h);
                            // between perfect prediction and intra. Quantize the
                            // dense inter residual, reconstruct it exactly as the
                            // decoder will, and let it compete against both skip
                            // and the existing intra bound.
                            if block_w == 32 && block_h == 32 && ENABLE_DENSE_INTER_32 {
                                let DenseInterResidual420 {
                                    rd_cost: residual_cost,
                                    y_coeffs,
                                    u_coeffs,
                                    v_coeffs,
                                    y_recon,
                                    u_recon,
                                    v_recon,
                                } = self.evaluate_dense_inter_residual_420(
                                    DenseInterResidualInput420 {
                                        enc,
                                        source_y: yp,
                                        source_u: up,
                                        source_v: vp,
                                        prediction_y: pred_y,
                                        prediction_u: pred_u,
                                        prediction_v: pred_v,
                                        source_stride: pw,
                                        chroma_stride: pcw,
                                        block_x: sb_x,
                                        block_y: sb_y,
                                        block_size: 32,
                                        residual_scale: sb_scale,
                                        qstep: sb_qstep,
                                        skip_ctx,
                                        mode_ctx,
                                        inter_mode,
                                        mvd: scaled_mvd,
                                    },
                                );
                                // DC-only inter residuals. Keep AC-heavy blocks on the
                                // established intra/skip paths until dense AC token coding
                                // passes the same AVM reconstruction gate.
                                if residual_cost < motion_cost && residual_cost < intra_bound {
                                    if !aq_committed {
                                        if use_grid {
                                            enc.delta_q_signaled = cell.sig;
                                        } else {
                                            let _ = aqs.per_sb(
                                                enc,
                                                yp,
                                                pw,
                                                row * 64,
                                                col * 64,
                                                width,
                                                height,
                                            );
                                        }
                                        aq_committed = true;
                                        enc.delta_q_pending = enc.delta_q_present;
                                    }

                                    let (y_skip_cdf, y_dc_sign_ctx) = sb_tu_contexts_rect(
                                        &y_coeffs,
                                        above,
                                        left,
                                        &TxbContextSpec {
                                            sb_y,
                                            sb_x,
                                            qc: enc.qc,
                                            mi_cols: tmc,
                                            mi_rows: tmr,
                                            block_eq_tx: true,
                                        },
                                        8,
                                        8,
                                    );
                                    let u_present = u_coeffs.iter().any(|&(_, level)| level != 0);
                                    let v_present = v_coeffs.iter().any(|&(_, level)| level != 0);
                                    let u_skip_cdf =
                                        INTER_SKIP_TX16_QC[enc.qc][(6 + ua + ul) as usize] as u32;
                                    let v_skip_ctx = (6 * i32::from(u_present) + va + vl) as u32;
                                    crate::av2::coder::emit_inter_residual_leaf_32(
                                        enc,
                                        pc,
                                        skip_ctx,
                                        mode_ctx,
                                        mode_ctx,
                                        inter_mode,
                                        mvd_row,
                                        mvd_col,
                                        &y_coeffs,
                                        y_skip_cdf,
                                        y_dc_sign_ctx,
                                        &u_coeffs,
                                        &v_coeffs,
                                        u_skip_cdf,
                                        v_skip_ctx,
                                    );
                                    put_block_rect(recy, pw, sb_y, sb_x, 32, 32, &y_recon);
                                    put_block_rect(recu, pcw, cy, cx, 16, 16, &u_recon);
                                    put_block_rect(recv, pcw, cy, cx, 16, 16, &v_recon);
                                    #[cfg(test)]
                                    {
                                        INTER_RESIDUAL_32_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if y_coeffs.iter().any(|&(scan, _)| scan > 900) {
                                            INTER_RESIDUAL_32_HIGH_EOB_COUNT
                                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        }
                                    }
                                    for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                                        inter_mi_above[c] = 1;
                                        skip_mi_above[c] = 0;
                                        newmv_mi_above[c] = (inter_mode == 2) as u8;
                                        mv_mi_above[c] = Some(mv);
                                        u_above[c] = i32::from(u_present);
                                        v_above[c] = i32::from(v_present);
                                    }
                                    for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                                        inter_mi_left[r] = 1;
                                        skip_mi_left[r] = 0;
                                        newmv_mi_left[r] = (inter_mode == 2) as u8;
                                        mv_mi_left[r] = Some(mv);
                                        u_left[r] = i32::from(u_present);
                                        v_left[r] = i32::from(v_present);
                                    }
                                    skip_above[col] = 0;
                                    skip_left = 0;
                                    inter_above[col] = 1;
                                    inter_left = 1;
                                    newmv_above[col] = (inter_mode == 2) as u8;
                                    newmv_left = (inter_mode == 2) as u8;
                                    mv_above[col] = Some(mv);
                                    mv_left = Some(mv);
                                    for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                                        for c in (lmc & 15)..((lmc & 15) + bw_mi).min(16) {
                                            enc.sb_coded[r * 16 + c] = 1;
                                        }
                                    }
                                    return true;
                                }
                            }
                            if block_w == 16 && block_h == 16 && ENABLE_DENSE_INTER_16 {
                                let DenseInterResidual420 {
                                    rd_cost: residual_cost,
                                    y_coeffs,
                                    u_coeffs,
                                    v_coeffs,
                                    y_recon,
                                    u_recon,
                                    v_recon,
                                } = self.evaluate_dense_inter_residual_420(
                                    DenseInterResidualInput420 {
                                        enc,
                                        source_y: yp,
                                        source_u: up,
                                        source_v: vp,
                                        prediction_y: pred_y,
                                        prediction_u: pred_u,
                                        prediction_v: pred_v,
                                        source_stride: pw,
                                        chroma_stride: pcw,
                                        block_x: sb_x,
                                        block_y: sb_y,
                                        block_size: 16,
                                        residual_scale: sb_scale,
                                        qstep: sb_qstep,
                                        skip_ctx,
                                        mode_ctx,
                                        inter_mode,
                                        mvd: scaled_mvd,
                                    },
                                );
                                if residual_cost < motion_cost && residual_cost < intra_bound {
                                    if !aq_committed {
                                        if use_grid {
                                            enc.delta_q_signaled = cell.sig;
                                        } else {
                                            let _ = aqs.per_sb(
                                                enc,
                                                yp,
                                                pw,
                                                row * 64,
                                                col * 64,
                                                width,
                                                height,
                                            );
                                        }
                                        aq_committed = true;
                                        enc.delta_q_pending = enc.delta_q_present;
                                    }

                                    let u_present = u_coeffs.iter().any(|&(_, level)| level != 0);
                                    let v_present = v_coeffs.iter().any(|&(_, level)| level != 0);
                                    let (intra_skip_cdf, y_dc_sign_ctx) = sb_tu_contexts_rect(
                                        &y_coeffs,
                                        above,
                                        left,
                                        &TxbContextSpec {
                                            sb_y,
                                            sb_x,
                                            qc: enc.qc,
                                            mi_cols: tmc,
                                            mi_rows: tmr,
                                            block_eq_tx: true,
                                        },
                                        4,
                                        4,
                                    );
                                    let y_skip_ctx = SKIP_TX16_QC[enc.qc]
                                        .iter()
                                        .position(|&cdf| u32::from(cdf) == intra_skip_cdf)
                                        .expect("TX16 skip context must resolve");
                                    let y_skip_cdf = INTER_SKIP_TX16_QC[enc.qc][y_skip_ctx] as u32;
                                    crate::av2::coder::emit_inter_residual_leaf_16(
                                        enc,
                                        pc,
                                        skip_ctx,
                                        mode_ctx,
                                        mode_ctx,
                                        inter_mode,
                                        mvd_row,
                                        mvd_col,
                                        &y_coeffs,
                                        y_skip_cdf,
                                        y_dc_sign_ctx,
                                        &u_coeffs,
                                        &v_coeffs,
                                        (6 + ua + ul) as usize,
                                        (6 * usize::from(u_present) as i32 + va + vl) as usize,
                                    );
                                    put_block_rect(recy, pw, sb_y, sb_x, 16, 16, &y_recon);
                                    put_block_rect(recu, pcw, cy, cx, 8, 8, &u_recon);
                                    put_block_rect(recv, pcw, cy, cx, 8, 8, &v_recon);
                                    #[cfg(test)]
                                    {
                                        INTER_RESIDUAL_16_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if y_coeffs.iter().any(|&(scan, _)| scan > 220) {
                                            INTER_RESIDUAL_16_HIGH_EOB_COUNT
                                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        }
                                        if u_present && v_present {
                                            INTER_RESIDUAL_16_CHROMA_COUNT
                                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        }
                                    }
                                    for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                                        inter_mi_above[c] = 1;
                                        skip_mi_above[c] = 0;
                                        newmv_mi_above[c] = (inter_mode == 2) as u8;
                                        mv_mi_above[c] = Some(mv);
                                        u_above[c] = i32::from(u_present);
                                        v_above[c] = i32::from(v_present);
                                    }
                                    for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                                        inter_mi_left[r] = 1;
                                        skip_mi_left[r] = 0;
                                        newmv_mi_left[r] = (inter_mode == 2) as u8;
                                        mv_mi_left[r] = Some(mv);
                                        u_left[r] = i32::from(u_present);
                                        v_left[r] = i32::from(v_present);
                                    }
                                    skip_above[col] = 0;
                                    skip_left = 0;
                                    inter_above[col] = 1;
                                    inter_left = 1;
                                    newmv_above[col] = (inter_mode == 2) as u8;
                                    newmv_left = (inter_mode == 2) as u8;
                                    mv_above[col] = Some(mv);
                                    mv_left = Some(mv);
                                    for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                                        for c in (lmc & 15)..((lmc & 15) + bw_mi).min(16) {
                                            enc.sb_coded[r * 16 + c] = 1;
                                        }
                                    }
                                    return true;
                                }
                            }
                            if motion_cost < intra_bound {
                                #[cfg(test)]
                                if block_w != block_h {
                                    INTER_MOTION_SKIP_RECT_COUNT
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                #[cfg(test)]
                                if block_w == 16 {
                                    if inter_mode == 2 {
                                        INTER_NEWMV_SKIP_16_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        INTER_NEARMV_SKIP_16_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }
                                #[cfg(test)]
                                if block_w == 32 {
                                    if inter_mode == 2 {
                                        INTER_NEWMV_SKIP_32_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    } else {
                                        INTER_NEARMV_SKIP_32_COUNT
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }
                                if !aq_committed {
                                    if use_grid {
                                        enc.delta_q_signaled = cell.sig;
                                    } else {
                                        let _ = aqs.per_sb(
                                            enc,
                                            yp,
                                            pw,
                                            row * 64,
                                            col * 64,
                                            width,
                                            height,
                                        );
                                    }
                                    aq_committed = true;
                                }
                                enc.cur_bw4 = bw_mi;
                                enc.cur_bh4 = bh_mi;
                                crate::av2::coder::emit_inter_mode_leaf(
                                    enc, pc, skip_ctx, mode_ctx, mode_ctx, inter_mode, mvd_row,
                                    mvd_col, false,
                                );
                                skip_leaves.push((lmr, lmc, bw_mi, bh_mi));
                                put_block_rect(recy, pw, sb_y, sb_x, block_w, block_h, pred_y);
                                for (dst, pred) in [(&mut *recu, &*pred_u), (&mut *recv, &*pred_v)]
                                {
                                    put_block_rect(dst, pcw, cy, cx, chroma_w, chroma_h, pred);
                                }
                                for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                                    inter_mi_above[c] = 1;
                                    skip_mi_above[c] = 1;
                                    newmv_mi_above[c] = (inter_mode == 2) as u8;
                                    mv_mi_above[c] = Some(mv);
                                    above[c] = 0x40;
                                    u_above[c] = 0;
                                    v_above[c] = 0;
                                }
                                for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                                    inter_mi_left[r] = 1;
                                    skip_mi_left[r] = 1;
                                    newmv_mi_left[r] = (inter_mode == 2) as u8;
                                    mv_mi_left[r] = Some(mv);
                                    left[r] = 0x40;
                                    u_left[r] = 0;
                                    v_left[r] = 0;
                                }
                                skip_above[col] = 1;
                                skip_left = 1;
                                inter_above[col] = 1;
                                inter_left = 1;
                                newmv_above[col] = (inter_mode == 2) as u8;
                                newmv_left = (inter_mode == 2) as u8;
                                mv_above[col] = Some(mv);
                                mv_left = Some(mv);
                                return true;
                            }
                        }
                    }
                }
                false
            });
            if subblock_inter {
                continue;
            }
            // GLOBALMV zero-motion skip: static 64x64 block copies LAST (no residual).
            let whole_skip = outline_leaf_420(|| {
                if has_last && bw_mi == 16 && bh_mi == 16 {
                    let bw = 64.min(width - sb_x);
                    let bh = 64.min(height - sb_y);
                    if bw == 64 && bh == 64 {
                        let (cy, cx) = (sb_y / 2, sb_x / 2);
                        let (rcy, rcx) = (ref_y0 / 2 + cy, ref_x0 / 2 + cx);
                        // Full-SB SSE (luma + chroma) against a candidate reference at
                        // zero motion, so chroma-only changes aren't wrongly skipped.
                        let full_sse = |refp: &[Vec<f32>]| -> f32 {
                            let mut sse = rect_sse_f32(
                                &PlaneRect {
                                    plane: yp,
                                    stride: pw,
                                    y: sb_y,
                                    x: sb_x,
                                },
                                &PlaneRect {
                                    plane: &refp[0],
                                    stride: ref_ls,
                                    y: ref_y0 + sb_y,
                                    x: ref_x0 + sb_x,
                                },
                                64,
                                64,
                            );
                            for (src_c, ref_c) in [(&up, &refp[1]), (&vp, &refp[2])] {
                                sse += rect_sse_f32(
                                    &PlaneRect {
                                        plane: src_c,
                                        stride: pcw,
                                        y: cy,
                                        x: cx,
                                    },
                                    &PlaneRect {
                                        plane: ref_c,
                                        stride: ref_cs,
                                        y: rcy,
                                        x: rcx,
                                    },
                                    32,
                                    32,
                                );
                            }
                            sse
                        };
                        let up_n = row > 0;
                        let lf_n = col > 0;
                        let ia = inter_above[col] == 1;
                        let il = inter_left == 1;
                        enc.intra_inter_ctx = if up_n && lf_n {
                            let n_intra = (!il as u8) + (!ia as u8);
                            if n_intra == 2 { 3 } else { n_intra as usize }
                        } else if up_n {
                            if ia { 0 } else { 3 }
                        } else if lf_n {
                            if il { 0 } else { 3 }
                        } else {
                            0
                        };
                        let sa = skip_above[col];
                        let sl = skip_left;
                        let skip_ctx = if up_n && lf_n {
                            (sl + sa) as usize
                        } else if up_n {
                            (2 * sa) as usize
                        } else if lf_n {
                            (2 * sl) as usize
                        } else {
                            0
                        };
                        // RD skip mode context: single reference keeps the legacy
                        // mi-granular formula verbatim (byte-exact). Two references use
                        // the same-rank SB-granular ctx (AVM av2_find_mode_ctx counts
                        // only neighbors predicting from this rank), which reduces to the
                        // legacy value when every neighbor is rank 0.
                        let ar = (lmc + bw_mi - 1).min(inter_mi_above.len() - 1);
                        let bl = (lmr + bh_mi - 1).min(inter_mi_left.len() - 1);
                        let legacy_mode_ctx = {
                            let left_match =
                                lmc > 0 && (inter_mi_left[bl] != 0 || inter_mi_left[lmr] != 0);
                            let above_match =
                                lmr > 0 && (inter_mi_above[ar] != 0 || inter_mi_above[lmc] != 0);
                            let any_newmv = (lmc > 0
                                && (newmv_mi_left[bl] != 0 || newmv_mi_left[lmr] != 0))
                                || (lmr > 0
                                    && (newmv_mi_above[ar] != 0 || newmv_mi_above[lmc] != 0));
                            usize::from(left_match)
                                + usize::from(above_match)
                                + 2 * usize::from(any_newmv)
                        };
                        // The context that goes into the bitstream (SB-granular, same-rank).
                        // Uses the neighbor ranks captured before the per-leaf reset.
                        let emit_mode_ctx = |rank: u8| -> usize {
                            let am = ia && sb_above_rank == rank;
                            let lm = il && sb_left_rank == rank;
                            (am as usize + lm as usize)
                                + if (am && newmv_above[col] != 0) || (lm && newmv_left != 0) {
                                    2
                                } else {
                                    0
                                }
                        };
                        let intra_bound =
                            crate::av2::video::rd::rd_cost(0.0, 2.0 * 64.0 * 64.0, sb_qstep as u32);
                        // Pick the RD-best listed reference at zero motion. Rank 0's RD
                        // uses the legacy mi-granular ctx so single-reference decisions are
                        // byte-identical; the emitted ctx is always the decoder-matching
                        // SB-granular value.
                        let mut skip_choice: Option<(f32, u8, usize)> = None;
                        for rank in 0..if has_second { 2u8 } else { 1u8 } {
                            let refp: &[Vec<f32>] = if rank == 0 { last_ref } else { second_ref };
                            let sse = full_sse(refp);
                            let rd_ctx = if rank == 0 {
                                legacy_mode_ctx
                            } else {
                                emit_mode_ctx(rank)
                            };
                            enc.ref_rank = rank as usize;
                            let rate = crate::av2::video::rd::inter_syntax_bits(
                                enc, skip_ctx, rd_ctx, true, 1, None,
                            );
                            let cost = crate::av2::video::rd::rd_cost(
                                sse * crate::av2::video::rd::SS2_INTER_DIST_W,
                                rate,
                                sb_qstep as u32,
                            );
                            if skip_choice.is_none_or(|(best, _, _)| cost < best) {
                                skip_choice = Some((cost, rank, emit_mode_ctx(rank)));
                            }
                        }
                        let (skip_cost, skip_rank, mode_ctx) =
                            skip_choice.expect("rank 0 always evaluated");
                        // The RD loop left `ref_rank` at the last evaluated rank. Reset
                        // it so the fall-through GLOBALMV-residual / NEWMV branches (which
                        // always predict rank 0) don't emit a stale rank-1 ref bit.
                        enc.ref_rank = 0;
                        if skip_cost < intra_bound {
                            let refp: &[Vec<f32>] =
                                if skip_rank == 0 { last_ref } else { second_ref };
                            enc.ref_rank = skip_rank as usize;
                            crate::av2::coder::emit_inter_skip_block(enc, pc, skip_ctx, mode_ctx);
                            skip_leaves.push((lmr, lmc, bw_mi, bh_mi));
                            for (dst_row, src_row) in rect_rows_mut(recy, pw, sb_y, sb_x, 64, 64)
                                .zip(rect_rows(
                                    &refp[0],
                                    ref_ls,
                                    ref_y0 + sb_y,
                                    ref_x0 + sb_x,
                                    64,
                                    64,
                                ))
                            {
                                dst_row.copy_from_slice(src_row);
                            }
                            let dst_rows = rect_rows_mut(recu, pcw, cy, cx, 32, 32)
                                .zip(rect_rows_mut(recv, pcw, cy, cx, 32, 32));
                            let src_rows = rect_rows(&refp[1], ref_cs, rcy, rcx, 32, 32)
                                .zip(rect_rows(&refp[2], ref_cs, rcy, rcx, 32, 32));
                            for ((dst_u, dst_v), (src_u, src_v)) in dst_rows.zip(src_rows) {
                                dst_u.copy_from_slice(src_u);
                                dst_v.copy_from_slice(src_v);
                            }
                            #[cfg(test)]
                            if skip_rank == 1 {
                                PARTITION_SKIP_RANK1_COUNT
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            skip_above[col] = 1;
                            skip_left = 1;
                            inter_above[col] = 1;
                            newmv_above[col] = 0;
                            newmv_left = 0;
                            mv_above[col] = Some(Mv::ZERO);
                            mv_left = Some(Mv::ZERO);
                            ref_above[col] = skip_rank;
                            ref_left = skip_rank;
                            for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                                inter_mi_above[c] = 1;
                                skip_mi_above[c] = 1;
                                newmv_mi_above[c] = 0;
                                mv_mi_above[c] = Some(Mv::ZERO);
                                ref_mi_above[c] = skip_rank;
                            }
                            for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                                inter_mi_left[r] = 1;
                                skip_mi_left[r] = 1;
                                newmv_mi_left[r] = 0;
                                mv_mi_left[r] = Some(Mv::ZERO);
                                ref_mi_left[r] = skip_rank;
                            }
                            // Skip block has no residual: reset coeff context (AVM av2_reset_entropy_context).
                            for c in lmc..(lmc + bw_mi).min(above.len()) {
                                above[c] = 0x40;
                            }
                            for r in lmr..(lmr + bh_mi).min(left.len()) {
                                left[r] = 0x40;
                            }
                            for c in lmc..(lmc + bw_mi).min(u_above.len()) {
                                u_above[c] = 0;
                                v_above[c] = 0;
                            }
                            for r in lmr..(lmr + bh_mi).min(u_left.len()) {
                                u_left[r] = 0;
                                v_left[r] = 0;
                            }
                            inter_left = 1;
                            return true;
                        }
                    }
                }
                false
            });
            if whole_skip {
                continue;
            }
            // GLOBALMV residual: LAST close but not exact -> code source-LAST
            // residual (DCT_DCT, 4 luma TUs), reconstruct = LAST + inv-DCT.
            let global_residual = outline_leaf_420(|| {
                if has_last
                    && bw_mi == 16
                    && bh_mi == 16
                    && 64.min(width - sb_x) == 64
                    && 64.min(height - sb_y) == 64
                {
                    let ly = &last_ref[0];
                    let mut sse = rect_sse_f32(
                        &PlaneRect {
                            plane: yp,
                            stride: pw,
                            y: sb_y,
                            x: sb_x,
                        },
                        &PlaneRect {
                            plane: ly,
                            stride: ref_ls,
                            y: ref_y0 + sb_y,
                            x: ref_x0 + sb_x,
                        },
                        64,
                        64,
                    );
                    let (cy0, cx0) = (sb_y / 2, sb_x / 2);
                    let (rcy0, rcx0) = (ref_y0 / 2 + cy0, ref_x0 / 2 + cx0);
                    for (src_c, ref_c) in [(&up, &last_ref[1]), (&vp, &last_ref[2])] {
                        sse += rect_sse_f32(
                            &PlaneRect {
                                plane: src_c,
                                stride: pcw,
                                y: cy0,
                                x: cx0,
                            },
                            &PlaneRect {
                                plane: ref_c,
                                stride: ref_cs,
                                y: rcy0,
                                x: rcx0,
                            },
                            32,
                            32,
                        );
                    }
                    let upn = row > 0;
                    let lfn = col > 0;
                    let ia = inter_above[col] == 1;
                    let il = inter_left == 1;
                    enc.intra_inter_ctx = if upn && lfn {
                        let n_intra = (!il as u8) + (!ia as u8);
                        if n_intra == 2 { 3 } else { n_intra as usize }
                    } else if upn {
                        if ia { 0 } else { 3 }
                    } else if lfn {
                        if il { 0 } else { 3 }
                    } else {
                        0
                    };
                    let sa = skip_above[col];
                    let sl = skip_left;
                    let skip_ctx = if upn && lfn {
                        (sl + sa) as usize
                    } else if upn {
                        (2 * sa) as usize
                    } else if lfn {
                        (2 * sl) as usize
                    } else {
                        0
                    };
                    let am0 = inter_above[col] == 1 && sb_above_rank == 0;
                    let lm0 = inter_left == 1 && sb_left_rank == 0;
                    let mode_ctx = (am0 as usize + lm0 as usize)
                        + if (am0 && newmv_above[col] != 0) || (lm0 && newmv_left != 0) {
                            2
                        } else {
                            0
                        };
                    let syntax_rate = crate::av2::video::rd::inter_syntax_bits(
                        enc, skip_ctx, mode_ctx, false, 1, None,
                    );
                    let _ = sse; // Gate on the CODED residual RD below, not the uncoded
                    // zero-motion SSE. The old `inter_cost = sse*W + syntax` gate rejected
                    // predictable blocks to expensive intra at high quality: the SSE term
                    // is not rdmult-scaled, so it dwarfs the fixed rdmult-scaled intra_bound
                    // as qstep shrinks. Code the residual, then compare its real RD to intra.
                    let intra_bound =
                        crate::av2::video::rd::rd_cost(0.0, 16.0 * 64.0 * 64.0, sb_qstep as u32);
                    {
                        let bd = self.bit_depth as i32;
                        let mut coeff_bits = 0.0f32;
                        static POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
                        let mut tus: [Vec<Coeff>; 4] =
                            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                        let mut resid = [0f32; 1024];
                        for (i, &(ty, tx)) in POS.iter().enumerate() {
                            let (y0, x0) = (sb_y + ty, sb_x + tx);
                            let mut pblk = [0f32; 1024];
                            crate::av2::metrics::copy_f32_prediction_and_scaled_residual(
                                &mut pblk,
                                &mut resid,
                                &yp[y0 * pw + x0..],
                                &ly[(ref_y0 + y0) * ref_ls + ref_x0 + x0..],
                                crate::av2::metrics::ResidualSpec {
                                    src_stride: pw,
                                    pred_stride: ref_ls,
                                    width: 32,
                                    height: 32,
                                    scale: sb_scale,
                                },
                            );
                            let lev = bases.luma.project(&resid[..], 0.0);
                            let rb = reconstruct_luma(&pblk, &lev, sb_qstep, &tables::SCAN, bd);
                            put_block(recy, pw, y0, x0, 32, &rb);
                            coeff_bits += coeff_rate_f32(&lev);
                            tus[i] = levels_to_coeffs(&lev);
                        }
                        // Chroma residual: source-LAST for U/V 32x32 (420).
                        let (cy, cx) = (sb_y / 2, sb_x / 2);
                        let mut uv_lev: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                        let mut uv_coeffs: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
                        for (pi, (src_c, ref_c, rec_c)) in [
                            (&up, &last_ref[1], &mut *recu),
                            (&vp, &last_ref[2], &mut *recv),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let mut cres = [0f32; 1024];
                            let mut cpred = [0i32; 1024];
                            crate::av2::metrics::f32_prediction_and_scaled_residual_i32(
                                &mut cpred,
                                &mut cres,
                                &src_c[cy * pcw + cx..],
                                &ref_c[(ref_y0 / 2 + cy) * ref_cs + ref_x0 / 2 + cx..],
                                crate::av2::metrics::ResidualSpec {
                                    src_stride: pcw,
                                    pred_stride: ref_cs,
                                    width: 32,
                                    height: 32,
                                    scale: sb_scale,
                                },
                            );
                            let lev = bases.chroma420.project(&cres[..], 0.0);
                            let rb = itx422::reconstruct_chroma_cfl(
                                &cpred,
                                &lev,
                                sb_qstep,
                                &tables::SCAN,
                                32,
                                32,
                                bd,
                            );
                            for (dst_row, src_row) in rect_rows_mut(rec_c, pcw, cy, cx, 32, 32)
                                .zip(rb.as_chunks::<32>().0.iter())
                            {
                                dst_row.copy_from_slice(src_row);
                            }
                            uv_coeffs[pi] = levels_to_coeffs(&lev);
                            coeff_bits += coeff_rate_f32(&lev);
                            uv_lev[pi] = lev;
                        }
                        // Real RD of the coded residual: reconstruction distortion (now
                        // the quantization error, not the full prediction error) plus the
                        // mode syntax and coefficient bits. Choose it over intra only when
                        // it is actually cheaper — a dense residual on hard content still
                        // loses to intra, but a cheap residual on predictable content
                        // (the common case) now wins instead of falling back to intra.
                        let mut coded_sse = rect_sse_f32(
                            &PlaneRect {
                                plane: yp,
                                stride: pw,
                                y: sb_y,
                                x: sb_x,
                            },
                            &PlaneRect {
                                plane: recy,
                                stride: pw,
                                y: sb_y,
                                x: sb_x,
                            },
                            64,
                            64,
                        );
                        for (src_c, rec_c) in [(&up, &*recu), (&vp, &*recv)] {
                            coded_sse += rect_sse_f32(
                                &PlaneRect {
                                    plane: src_c,
                                    stride: pcw,
                                    y: cy,
                                    x: cx,
                                },
                                &PlaneRect {
                                    plane: rec_c,
                                    stride: pcw,
                                    y: cy,
                                    x: cx,
                                },
                                32,
                                32,
                            );
                        }
                        let residual_cost = crate::av2::video::rd::rd_cost(
                            coded_sse * crate::av2::video::rd::SS2_INTER_DIST_W,
                            syntax_rate + coeff_bits,
                            sb_qstep as u32,
                        );
                        if residual_cost < intra_bound {
                            let up = row > 0;
                            let lf = col > 0;
                            let ia = inter_above[col] == 1;
                            let il = inter_left == 1;
                            enc.intra_inter_ctx = if up && lf {
                                let n_intra = (!il as u8) + (!ia as u8);
                                if n_intra == 2 { 3 } else { n_intra as usize }
                            } else if up {
                                if ia { 0 } else { 3 }
                            } else if lf {
                                if il { 0 } else { 3 }
                            } else {
                                0
                            };
                            let sa = skip_above[col];
                            let sl = skip_left;
                            let skip_ctx = if up && lf {
                                (sl + sa) as usize
                            } else if up {
                                (2 * sa) as usize
                            } else if lf {
                                (2 * sl) as usize
                            } else {
                                0
                            };
                            let ar = (lmc + bw_mi - 1).min(inter_mi_above.len() - 1);
                            let bl = (lmr + bh_mi - 1).min(inter_mi_left.len() - 1);
                            // Rank-0 block: count only rank-0 inter neighbors (excludes a
                            // whole-64 rank-1 skip neighbor). No-op when no rank-1 exists.
                            let left_match = lmc > 0
                                && ((inter_mi_left[bl] != 0 && bl_ref_rank == 0)
                                    || (inter_mi_left[lmr] != 0 && left_ref_rank == 0));
                            let above_match = lmr > 0
                                && ((inter_mi_above[ar] != 0 && ar_ref_rank == 0)
                                    || (inter_mi_above[lmc] != 0 && above_ref_rank == 0));
                            let any_newmv = (lmc > 0
                                && ((newmv_mi_left[bl] != 0 && bl_ref_rank == 0)
                                    || (newmv_mi_left[lmr] != 0 && left_ref_rank == 0)))
                                || (lmr > 0
                                    && ((newmv_mi_above[ar] != 0 && ar_ref_rank == 0)
                                        || (newmv_mi_above[lmc] != 0 && above_ref_rank == 0)));
                            let mode_ctx = usize::from(left_match)
                                + usize::from(above_match)
                                + 2 * usize::from(any_newmv);
                            // Residual blocks do carry delta-Q. Commit only now, after
                            // the skip decision, so encoder and decoder qindex state stay
                            // synchronized across preceding skip_txfm=1 superblocks.
                            let _ = aqs.per_sb(enc, yp, pw, row * 64, col * 64, width, height);
                            enc.delta_q_pending = enc.delta_q_present;
                            let (luma_skip, luma_dc) =
                                sb_tu_contexts(&tus, sb_y, sb_x, above, left, enc.qc, tmc, tmr);
                            let u_present = uv_coeffs[0].iter().any(|&(_, level)| level != 0);
                            let v_present = uv_coeffs[1].iter().any(|&(_, level)| level != 0);
                            coder::emit_inter_residual_block(
                                enc,
                                pc,
                                skip_ctx,
                                mode_ctx,
                                &tus,
                                &luma_skip,
                                &luma_dc,
                                &uv_coeffs[0],
                                &uv_coeffs[1],
                                (6 + ua + ul) as usize,
                                (6 * i32::from(u_present) + va + vl) as usize,
                            );
                            skip_above[col] = 0; // has coeffs
                            skip_left = 0;
                            inter_above[col] = 1;
                            newmv_above[col] = 0;
                            newmv_left = 0;
                            mv_above[col] = Some(Mv::ZERO);
                            mv_left = Some(Mv::ZERO);
                            for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                                inter_mi_above[c] = 1;
                                skip_mi_above[c] = 0;
                                newmv_mi_above[c] = 0;
                                mv_mi_above[c] = Some(Mv::ZERO);
                                u_above[c] = i32::from(u_present);
                                v_above[c] = i32::from(v_present);
                            }
                            for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                                inter_mi_left[r] = 1;
                                skip_mi_left[r] = 0;
                                newmv_mi_left[r] = 0;
                                mv_mi_left[r] = Some(Mv::ZERO);
                                u_left[r] = i32::from(u_present);
                                v_left[r] = i32::from(v_present);
                            }
                            inter_left = 1;
                            #[cfg(test)]
                            INTER_RESIDUAL_64_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return true;
                        } // end `if residual_cost < intra_bound`
                    }
                }
                false
            });
            if global_residual {
                continue;
            }
            // NEWMV residual: search LAST for motion; if a nonzero MV's MC
            // residual beats the intra estimate, code NEWMV + residual.
            let newmv_residual = outline_leaf_420(|| {
                if has_last
                    && bw_mi == 16
                    && bh_mi == 16
                    && 64.min(width - sb_x) == 64
                    && 64.min(height - sb_y) == 64
                {
                    use crate::av2::video::{mc, me};
                    let bd = self.bit_depth as i32;
                    // Keep the DPB/reconstruction f32 representation through ME/MC.
                    // The reference is the whole frame at `ref_ls`, so cross-tile
                    // motion resolves against real neighboring content.
                    let (bref, bstride, refh) = inter_luma.expect("LAST luma prepared for inter");
                    // Match the decoder's DRL[0] spatial candidate and seed ME
                    // from all available spatial MVs. The predictor used for the
                    // search-rate proxy must be the same one later subtracted from
                    // the selected absolute MV before entropy coding.
                    let above_mv = if row > 0 { mv_above[col] } else { None };
                    let above_right_mv = if row > 0 && col + 1 < sb_cols {
                        mv_above[col + 1]
                    } else {
                        None
                    };
                    // DRL[0] scans same-reference (rank-0) neighbors only. A rank-1
                    // neighbor (whole-64 skip on the second reference, zero motion) is
                    // excluded; its AVM derived contribution is the zero MV projected
                    // = zero = the drl0_mv fallback. `ref_above[col]`/`ref_left` were
                    // reset for this leaf, so use the captured neighbor ranks; the
                    // above-right column (col+1) is untouched this row.
                    let left_r0 = if sb_left_rank == 0 { mv_left } else { None };
                    let above_r0 = if sb_above_rank == 0 { above_mv } else { None };
                    let above_right_r0 = if row > 0 && col + 1 < sb_cols && ref_above[col + 1] == 0
                    {
                        above_right_mv
                    } else {
                        None
                    };
                    let pred_mv = drl0_mv(left_r0, above_r0, above_right_r0);
                    let mut preds = me::MeCandidates::new();
                    if frame_mv_seed != Mv::ZERO {
                        preds.push_unique(frame_mv_seed);
                    }
                    for candidate in [mv_left, above_mv, above_right_mv].into_iter().flatten() {
                        preds.push_unique(candidate);
                    }
                    let lambda_mv = (sb_qstep as u32).max(1);
                    let (mv, _) = me::search(
                        &me::MePlanes {
                            current: &yp[sb_y * pw + sb_x..],
                            current_stride: pw,
                            reference: bref,
                            reference_stride: *bstride,
                        },
                        preds.as_slice(),
                        &me::MeSearchSpec {
                            origin_x: (ref_x0 + sb_x + INTER_BORDER_420) as isize,
                            origin_y: (ref_y0 + sb_y + INTER_BORDER_420) as isize,
                            width: 64,
                            height: 64,
                            reference_mv: pred_mv,
                            lambda_mv,
                            max_dx: self.video_search_range,
                            max_dy: self.video_search_range,
                            predictor_gate_sad_per_pixel: self.video_predictor_gate,
                            integer_satd_radius: self.video_integer_satd_radius,
                            bit_depth: self.bit_depth,
                            frame_width: *bstride,
                            frame_height: *refh + 2 * INTER_BORDER_420,
                        },
                        me_scratch,
                    );
                    let mut mv = mc::clamp_umv(
                        mv,
                        (ref_x0 + sb_x) as i32,
                        (ref_y0 + sb_y) as i32,
                        64,
                        64,
                        ref_ls as i32,
                        *refh as i32,
                    );
                    if (mv.row != 0 || mv.col != 0) && (mv.row.abs() / 2 + mv.col.abs() / 2) <= 30 {
                        let scratch = inter_pred_scratch.whole_sb();
                        mc::predict_with_tmp(
                            scratch.y,
                            64,
                            bref,
                            *bstride,
                            &mc::MotionBlock {
                                origin_x: (ref_x0 + sb_x + INTER_BORDER_420) as isize,
                                origin_y: (ref_y0 + sb_y + INTER_BORDER_420) as isize,
                                mv,
                                width: 64,
                                height: 64,
                                bit_depth: self.bit_depth,
                            },
                            scratch.convolve_tmp,
                        );
                        // Residual SSE (luma) to gate against intra.
                        let mut sse = rect_sse_f32(
                            &PlaneRect {
                                plane: yp,
                                stride: pw,
                                y: sb_y,
                                x: sb_x,
                            },
                            &PlaneRect {
                                plane: scratch.y,
                                stride: 64,
                                y: 0,
                                x: 0,
                            },
                            64,
                            64,
                        );
                        let upn = row > 0;
                        let lfn = col > 0;
                        let ia = inter_above[col] == 1;
                        let il = inter_left == 1;
                        enc.intra_inter_ctx = if upn && lfn {
                            let n_intra = (!il as u8) + (!ia as u8);
                            if n_intra == 2 { 3 } else { n_intra as usize }
                        } else if upn {
                            if ia { 0 } else { 3 }
                        } else if lfn {
                            if il { 0 } else { 3 }
                        } else {
                            0
                        };
                        let sa = skip_above[col];
                        let sl = skip_left;
                        let skip_ctx = if upn && lfn {
                            (sl + sa) as usize
                        } else if upn {
                            (2 * sa) as usize
                        } else if lfn {
                            (2 * sl) as usize
                        } else {
                            0
                        };
                        let am0 = inter_above[col] == 1 && sb_above_rank == 0;
                        let lm0 = inter_left == 1 && sb_left_rank == 0;
                        let mode_ctx = (am0 as usize + lm0 as usize)
                            + if (am0 && newmv_above[col] != 0) || (lm0 && newmv_left != 0) {
                                2
                            } else {
                                0
                            };
                        let bounded_pred_mv = mc::clamp_umv(
                            pred_mv,
                            (ref_x0 + sb_x) as i32,
                            (ref_y0 + sb_y) as i32,
                            64,
                            64,
                            ref_ls as i32,
                            *refh as i32,
                        );
                        if pred_mv != Mv::ZERO && pred_mv != mv && bounded_pred_mv == pred_mv {
                            let searched_mv = mv;
                            let searched_sse = sse;
                            mc::predict_with_tmp(
                                scratch.y,
                                64,
                                bref,
                                *bstride,
                                &mc::MotionBlock {
                                    origin_x: (ref_x0 + sb_x + INTER_BORDER_420) as isize,
                                    origin_y: (ref_y0 + sb_y + INTER_BORDER_420) as isize,
                                    mv: pred_mv,
                                    width: 64,
                                    height: 64,
                                    bit_depth: self.bit_depth,
                                },
                                scratch.convolve_tmp,
                            );
                            let near_sse = rect_sse_f32(
                                &PlaneRect {
                                    plane: yp,
                                    stride: pw,
                                    y: sb_y,
                                    x: sb_x,
                                },
                                &PlaneRect {
                                    plane: scratch.y,
                                    stride: 64,
                                    y: 0,
                                    x: 0,
                                },
                                64,
                                64,
                            );
                            let searched_delta = searched_mv.diff(pred_mv);
                            debug_assert_ne!(searched_delta, Mv::ZERO);
                            if crate::av2::video::rd::prefer_nearmv(
                                enc,
                                crate::av2::video::rd::NearMvRdSpec {
                                    skip_ctx,
                                    mode_ctx,
                                    skip_txfm: false,
                                    near_distortion: near_sse
                                        * crate::av2::video::rd::SS2_INTER_DIST_W,
                                    new_distortion: searched_sse
                                        * crate::av2::video::rd::SS2_INTER_DIST_W,
                                    new_mvd: Mv {
                                        row: searched_delta.row / 2,
                                        col: searched_delta.col / 2,
                                    },
                                    qstep: sb_qstep as u32,
                                },
                            ) {
                                mv = pred_mv;
                                sse = near_sse;
                            } else {
                                mc::predict_with_tmp(
                                    scratch.y,
                                    64,
                                    bref,
                                    *bstride,
                                    &mc::MotionBlock {
                                        origin_x: (ref_x0 + sb_x + INTER_BORDER_420) as isize,
                                        origin_y: (ref_y0 + sb_y + INTER_BORDER_420) as isize,
                                        mv: searched_mv,
                                        width: 64,
                                        height: 64,
                                        bit_depth: self.bit_depth,
                                    },
                                    scratch.convolve_tmp,
                                );
                            }
                        }
                        let (candidate_mode, mvd_row, mvd_col) = inter_mode_qtr_mvd(mv, pred_mv);
                        let mvd = (candidate_mode == 2).then_some(Mv {
                            row: mvd_row,
                            col: mvd_col,
                        });
                        let syntax_rate = crate::av2::video::rd::inter_syntax_bits(
                            enc,
                            skip_ctx,
                            mode_ctx,
                            false,
                            candidate_mode,
                            mvd,
                        );
                        let inter_cost = crate::av2::video::rd::rd_cost(
                            sse * crate::av2::video::rd::SS2_INTER_DIST_W,
                            syntax_rate,
                            sb_qstep as u32,
                        );
                        let intra_bound = crate::av2::video::rd::rd_cost(
                            0.0,
                            16.0 * 64.0 * 64.0,
                            sb_qstep as u32,
                        );
                        if inter_cost < intra_bound {
                            #[cfg(test)]
                            INTER_RESIDUAL_64_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            static POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
                            for (i, &(ty, tx)) in POS.iter().enumerate() {
                                let (y0, x0) = (sb_y + ty, sb_x + tx);
                                crate::av2::metrics::scaled_residual_f32(
                                    scratch.residual,
                                    &yp[y0 * pw + x0..],
                                    &scratch.y[ty * 64 + tx..],
                                    crate::av2::metrics::ResidualSpec {
                                        src_stride: pw,
                                        pred_stride: 64,
                                        width: 32,
                                        height: 32,
                                        scale: sb_scale,
                                    },
                                );
                                for (dst, prediction) in scratch
                                    .tx_pred
                                    .as_chunks_mut::<32>()
                                    .0
                                    .iter_mut()
                                    .zip(rect_rows(scratch.y, 64, ty, tx, 32, 32))
                                {
                                    dst.copy_from_slice(prediction);
                                }
                                let lev = bases.luma.project(scratch.residual, 0.0);
                                let rb = reconstruct_luma(
                                    scratch.tx_pred,
                                    &lev,
                                    sb_qstep,
                                    &tables::SCAN,
                                    bd,
                                );
                                put_block(recy, pw, y0, x0, 32, &rb);
                                refill_coeffs(&mut scratch.luma_coeffs[i], &lev);
                            }
                            // Chroma MC (420: half-res MV) + residual.
                            let (cy, cx) = (sb_y / 2, sb_x / 2);
                            // 4:2:0 chroma sees half the luma displacement.
                            let cmv = Mv {
                                row: mv.row / 2,
                                col: mv.col / 2,
                            };
                            let cbrd = INTER_BORDER_420 / 2;
                            let chroma_refs = inter_chroma.expect("LAST chroma prepared for inter");
                            for (pi, (src_c, rec_c, mc_pred)) in [
                                (up, &mut *recu, &mut *scratch.u),
                                (vp, &mut *recv, &mut *scratch.v),
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                let (bcref, bcstride) = &chroma_refs[pi];
                                mc::predict_with_tmp(
                                    mc_pred,
                                    32,
                                    bcref,
                                    *bcstride,
                                    &mc::MotionBlock {
                                        origin_x: (ref_x0 / 2 + cx + cbrd) as isize,
                                        origin_y: (ref_y0 / 2 + cy + cbrd) as isize,
                                        mv: cmv,
                                        width: 32,
                                        height: 32,
                                        bit_depth: self.bit_depth,
                                    },
                                    scratch.convolve_tmp,
                                );
                                crate::av2::metrics::f32_prediction_and_scaled_residual_i32(
                                    scratch.chroma_pred,
                                    scratch.residual,
                                    &src_c[cy * pcw + cx..],
                                    mc_pred,
                                    crate::av2::metrics::ResidualSpec {
                                        src_stride: pcw,
                                        pred_stride: 32,
                                        width: 32,
                                        height: 32,
                                        scale: sb_scale,
                                    },
                                );
                                let lev = bases.chroma420.project(scratch.residual, 0.0);
                                let rb = itx422::reconstruct_chroma_cfl(
                                    scratch.chroma_pred,
                                    &lev,
                                    sb_qstep,
                                    &tables::SCAN,
                                    32,
                                    32,
                                    bd,
                                );
                                for (dst_row, src_row) in rect_rows_mut(rec_c, pcw, cy, cx, 32, 32)
                                    .zip(rb.as_chunks::<32>().0.iter())
                                {
                                    dst_row.copy_from_slice(src_row);
                                }
                                refill_coeffs(&mut scratch.chroma_coeffs[pi], &lev);
                            }
                            let upn = row > 0;
                            let lfn = col > 0;
                            let ia = inter_above[col] == 1;
                            let il = inter_left == 1;
                            enc.intra_inter_ctx = if upn && lfn {
                                let n_intra = (!il as u8) + (!ia as u8);
                                if n_intra == 2 { 3 } else { n_intra as usize }
                            } else if upn {
                                if ia { 0 } else { 3 }
                            } else if lfn {
                                if il { 0 } else { 3 }
                            } else {
                                0
                            };
                            let sa = skip_above[col];
                            let sl = skip_left;
                            let skip_ctx = if upn && lfn {
                                (sl + sa) as usize
                            } else if upn {
                                (2 * sa) as usize
                            } else if lfn {
                                (2 * sl) as usize
                            } else {
                                0
                            };
                            let am0 = inter_above[col] == 1 && sb_above_rank == 0;
                            let lm0 = inter_left == 1 && sb_left_rank == 0;
                            let mode_ctx = (am0 as usize + lm0 as usize)
                                + if (am0 && newmv_above[col] != 0) || (lm0 && newmv_left != 0) {
                                    2
                                } else {
                                    0
                                };
                            let _ = aqs.per_sb(enc, yp, pw, row * 64, col * 64, width, height);
                            enc.delta_q_pending = enc.delta_q_present;
                            let drl_ctx = mode_ctx;
                            let (inter_mode, mvd_row, mvd_col) = inter_mode_qtr_mvd(mv, pred_mv);
                            let (tu_skip_cdfs, tu_dc_sign) = crate::av2::helpers::sb_tu_contexts(
                                scratch.luma_coeffs,
                                sb_y,
                                sb_x,
                                above,
                                left,
                                enc.qc,
                                tmc,
                                tmr,
                            );
                            let u_present = scratch.chroma_coeffs[0]
                                .iter()
                                .any(|&(_, level)| level != 0);
                            let v_present = scratch.chroma_coeffs[1]
                                .iter()
                                .any(|&(_, level)| level != 0);
                            crate::av2::coder::emit_inter_newmv_residual_block(
                                enc,
                                &InterResidualSpec {
                                    part_cdf: pc,
                                    skip_ctx,
                                    mode_ctx,
                                    drl_ctx,
                                    mode: inter_mode,
                                    scaled_row: mvd_row,
                                    scaled_col: mvd_col,
                                    luma_tus: scratch.luma_coeffs,
                                    luma_skip_cdfs: &tu_skip_cdfs,
                                    luma_dc_sign_ctxs: &tu_dc_sign,
                                },
                                &scratch.chroma_coeffs[0],
                                &scratch.chroma_coeffs[1],
                                (6 + ua + ul) as usize,
                                (6 * i32::from(u_present) + va + vl) as usize,
                            );
                            skip_above[col] = 0;
                            skip_left = 0;
                            inter_above[col] = 1;
                            newmv_above[col] = (inter_mode == 2) as u8;
                            newmv_left = (inter_mode == 2) as u8;
                            mv_above[col] = Some(mv);
                            mv_left = Some(mv);
                            for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                                inter_mi_above[c] = 1;
                                skip_mi_above[c] = 0;
                                newmv_mi_above[c] = (inter_mode == 2) as u8;
                                mv_mi_above[c] = Some(mv);
                                u_above[c] = i32::from(u_present);
                                v_above[c] = i32::from(v_present);
                            }
                            for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                                inter_mi_left[r] = 1;
                                skip_mi_left[r] = 0;
                                newmv_mi_left[r] = (inter_mode == 2) as u8;
                                mv_mi_left[r] = Some(mv);
                                u_left[r] = i32::from(u_present);
                                v_left[r] = i32::from(v_present);
                            }
                            inter_left = 1;
                            return true;
                        }
                    }
                }
                false
            });
            if newmv_residual {
                continue;
            }
            #[cfg(test)]
            INTRA_LEAF_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            skip_above[col] = 0;
            skip_left = 0;
            inter_above[col] = 0;
            newmv_above[col] = 0;
            newmv_left = 0;
            mv_above[col] = None;
            mv_left = None;
            for c in lmc..(lmc + bw_mi).min(inter_mi_above.len()) {
                inter_mi_above[c] = 0;
                skip_mi_above[c] = 0;
                newmv_mi_above[c] = 0;
                mv_mi_above[c] = None;
            }
            for r in lmr..(lmr + bh_mi).min(inter_mi_left.len()) {
                inter_mi_left[r] = 0;
                skip_mi_left[r] = 0;
                newmv_mi_left[r] = 0;
                mv_mi_left[r] = None;
            }
            inter_left = 0;
            if !aq_committed {
                if use_grid {
                    // Grid path: the committed delta is `cell.sig` (the accumulate
                    // branch mirrors `per_sb`); read it directly instead of
                    // advancing the serial `aqs` accumulator.
                    enc.delta_q_signaled = cell.sig;
                } else {
                    let _ = aqs.per_sb(enc, yp, pw, row * 64, col * 64, width, height);
                }
                aq_committed = true;
            }
            // Stills: the rect shapes code as square TUs along their long side with the
            // full luma mode set (the legacy rect arms are DC-only frame-edge coders).
            let generic = sdp
                && self.still_rd()
                && crate::av2::rd::knob_or("GLEAF", 1.0) > 0.0
                && matches!(
                    (bw_mi, bh_mi),
                    (8, 4) | (4, 8) | (4, 2) | (2, 4) | (8, 2) | (2, 8) | (16, 4) | (4, 16)
                )
                && sb_y + bh_mi * 4 <= height
                && sb_x + bw_mi * 4 <= width;
            let (u_present, v_present) = if generic {
                outline_leaf_420(|| {
                    let lambda = leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
                    let smooth_neighbour = (lmr > 0 && midx_above[lmc] == MIDX_SMOOTH)
                        || (lmc > 0 && midx_left[lmr] == MIDX_SMOOTH);
                    let coded = enc.sb_coded;
                    let blk = RectSplitBlock {
                        yp,
                        pw,
                        y: sb_y,
                        x: sb_x,
                        w: bw_mi * 4,
                        h: bh_mi * 4,
                        frame: (width, height),
                        coded: &coded,
                        smooth_neighbour,
                    };
                    let sp = self
                        .multi_tu_search(recy, &blk, sb_qstep, lambda)
                        .expect("DC is always a candidate");
                    let (sz, pos) = blk.tus();
                    let mut skips = Vec::with_capacity(pos.len());
                    let mut dcss = Vec::with_capacity(pos.len());
                    for (k, &(ty, tx)) in pos.iter().enumerate() {
                        put_block_rect(recy, pw, ty, tx, sz, sz, &sp.recon[k]);
                        let (sk, dc) = sb_tu_contexts_rect(
                            &sp.tus[k],
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y: ty,
                                sb_x: tx,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: false,
                            },
                            sz / 4,
                            sz / 4,
                        );
                        skips.push(sk);
                        dcss.push(dc);
                        tu_splits.push((ty / 4, tx / 4, sz / 4, sz / 4));
                    }
                    let midx = coder::encode_luma_leaf_multi_tu(
                        enc,
                        &sp.tus,
                        &skips,
                        &dcss,
                        &sp.tx,
                        sp.mode,
                        pc,
                        (bw_mi, bh_mi),
                        (lmidx, amidx),
                    );
                    leaf_midx = if midx != 0xff {
                        midx as i32
                    } else {
                        midx_class(sp.mode.0)
                    };
                    (false, false)
                })
            } else {
                match (bw_mi, bh_mi) {
                    (16, 16) => outline_leaf_420(|| {
                        // 64x64 luma → 32x32 chroma (TX_32X32, eob 1024, skip TX32).
                        // Staged decouple: Replay restores the captured luma recon +
                        // reuses the luma winner and CfL choice (skipping the mode
                        // search + CfL RD); Capture logs them; Off searches inline.
                        // Order preserved exactly: luma → tu-contexts → CfL → emit.
                        let leaf_replay: Option<&Leaf420> =
                            replay_walk.as_ref().and_then(|v| v.get(cap_idx));
                        const PARTITION_CFL: bool = true;
                        let bd = self.bit_depth as i32;
                        let (tus, mode_idx, adelta): ([Vec<Coeff>; 4], usize, i8) =
                            if let Some(d) = leaf_replay {
                                put_block_rect(recy, pw, sb_y, sb_x, 64, 64, &d.recon_y);
                                (
                                    std::array::from_fn(|k| d.tus[k].clone()),
                                    d.mode_idx,
                                    d.angle_delta,
                                )
                            } else {
                                let (tus, mode_idx, adelta, _) = encode_luma_sb(
                                    recy,
                                    &LumaSource {
                                        plane: yp,
                                        stride: pw,
                                    },
                                    &LumaFrameBlock {
                                        frame_width: width,
                                        frame_height: height,
                                        y: sb_y,
                                        x: sb_x,
                                    },
                                    &LumaQuantSpec {
                                        prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                                        basis: &bases.luma,
                                        qstep: sb_qstep,
                                        scan: &tables::SCAN,
                                        neutral,
                                        quant_context: qc,
                                        rdoq_lambda: self.tune.rdoq_lambda,
                                        pixel_lambda: self.still_rd(),
                                        speed: self.speed,
                                        bit_depth: self.bit_depth as i32,
                                    },
                                    &LumaSbSearch {
                                        residual_scale: sb_scale,
                                        // Stills: directional modes + angle deltas (the
                                        // neighbours' midx are tracked per mi for the mode
                                        // list / y_idx context). Video keeps the old set.
                                        // Whole superblocks only: a 64x64 leaf in a partial
                                        // edge SB stays non-directional (as on the legacy
                                        // path; directional there desyncs avmdec).
                                        allow_directional: self.still_rd()
                                            && self.speed.try_directional()
                                            && sb_x + 64 <= width
                                            && sb_y + 64 <= height
                                            && crate::av2::rd::knob("DIR64") > 0.0,
                                    },
                                );
                                (tus, mode_idx, adelta)
                            };
                        // Stills: one TX_64X64 instead of the four TX_32X32, by RD.
                        let tx64: Option<(Vec<Coeff>, usize)> = if let Some(d) = leaf_replay {
                            (d.tx_idx == 64).then(|| (d.tus[0].clone(), d.mode_idx))
                        } else if sdp
                            && self.still_rd()
                            && crate::av2::rd::knob_or("TX64", 1.0) > 0.0
                            && mode_idx != crate::av2::palette::PAL_MODE
                            && sb_y + 64 <= height
                            && sb_x + 64 <= width
                        {
                            let lambda =
                                leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
                            let cur: Vec<f32> = rect_rows(recy, pw, sb_y, sb_x, 64, 64)
                                .flatten()
                                .copied()
                                .collect();
                            let sse32: f32 = rect_rows(yp, pw, sb_y, sb_x, 64, 64)
                                .flatten()
                                .zip(&cur)
                                .map(|(&a, &r)| (a - r.round()).powi(2))
                                .sum();
                            let mb = if mode_idx >= 5 {
                                crate::av2::rd::knob_or("MBDIR", 9.0)
                            } else if mode_idx != 0 {
                                crate::av2::rd::knob_or("MB16", 6.0)
                            } else {
                                0.0
                            };
                            let j32 = sse32
                                + lambda
                                    * (tus.iter().map(|t| coeff_tu_rate_proxy_f32(t)).sum::<f32>()
                                        + mb);
                            let coded = enc.sb_coded;
                            let (j64, lev, m, rec) = self.tx64_eval(
                                &*recy,
                                yp,
                                pw,
                                sb_y,
                                sb_x,
                                (width, height),
                                &coded,
                                sb_qstep,
                                lambda,
                            );
                            if j64 < j32 {
                                put_block_rect(recy, pw, sb_y, sb_x, 64, 64, &rec);
                                Some((levels_to_coeffs(&lev), m))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let (skip_cdfs, dc_sign_ctxs, dcs64) = if let Some((tu64, _)) = &tx64 {
                            let (_, d) = sb_tu_contexts_rect(
                                tu64,
                                above,
                                left,
                                &TxbContextSpec {
                                    sb_y,
                                    sb_x,
                                    qc,
                                    mi_cols: tmc,
                                    mi_rows: tmr,
                                    block_eq_tx: true,
                                },
                                16,
                                16,
                            );
                            tu_splits.push((
                                lmr,
                                lmc,
                                16 | crate::av2::deblock::TU_WHOLE,
                                16 | crate::av2::deblock::TU_WHOLE,
                            ));
                            ([0u32; 4], [0usize; 4], d)
                        } else {
                            let (a, b) =
                                sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
                            (a, b, 0)
                        };
                        let cfl_choice = if let Some(d) = leaf_replay {
                            d.chroma.clone()
                        } else if enc.cfl && PARTITION_CFL && !enc.sdp_luma {
                            let avg_l = cfl::cfl_avg_l(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                32,
                                true,
                                true,
                                bd,
                                (width, height),
                            );
                            let mut suf = [0f32; 32 * 32];
                            let mut svf = [0f32; 32 * 32];
                            cfl_partition_prediction::<32>(pcw, up, vp, cy, cx, &mut suf, &mut svf);
                            // Chroma DC neighbours stop at the 8-aligned luma extent (AVM
                            // replicates past it), not at the SB-padded plane.
                            let crect = BoundedIntraRect {
                                stride: pcw,
                                y: cy,
                                x: cx,
                                width: 32,
                                height: 32,
                                frame_width: ((width + 7) & !7) >> 1,
                                frame_height: ((height + 7) & !7) >> 1,
                            };
                            let dc_u_f = dc_pred_rect_subsampled_bounded(recu, &crect, neutral, bd);
                            let dc_v_f = dc_pred_rect_subsampled_bounded(recv, &crect, neutral, bd);
                            cfl::cfl_decide(
                                &cfl::CflDecisionInput {
                                    reconstructed_luma: recy,
                                    luma_stride: pw,
                                    luma_y: sb_y,
                                    luma_x: sb_x,
                                    source_u: &suf,
                                    source_v: &svf,
                                    dc_u: dc_u_f,
                                    dc_v: dc_v_f,
                                    width: 32,
                                    height: 32,
                                    subsample_x: true,
                                    subsample_y: true,
                                    luma_average_q3: avg_l,
                                },
                                &cfl::ChromaRdSpec {
                                    basis: &bases.chroma420,
                                    qstep: sb_qstep,
                                    lambda: leaf::part_lambda(
                                        sb_qstep,
                                        self.subsampled_part_lambda_c(),
                                    ),
                                    bit_depth: bd,
                                },
                            )
                        } else {
                            None
                        };
                        if capturing {
                            leaf_recs.push(Leaf420 {
                                bw_mi: 16,
                                bh_mi: 16,
                                tus: match &tx64 {
                                    Some((t, _)) => {
                                        vec![t.clone(), Vec::new(), Vec::new(), Vec::new()]
                                    }
                                    None => tus.to_vec(),
                                },
                                mode_idx: tx64.as_ref().map_or(mode_idx, |t| t.1),
                                angle_delta: adelta,
                                tx_idx: if tx64.is_some() { 64 } else { 0 },
                                recon_y: gather_rect(recy, pw, sb_y, sb_x, 64, 64),
                                chroma: cfl_choice.clone(),
                            });
                            sb_walk_ok = true;
                        }
                        cap_idx += 1;
                        if let Some(ref ch) = cfl_choice {
                            enc.cfl_use = true;
                            enc.cfl_js = ch.js;
                            enc.cfl_mag_u = ch.mag_u;
                            enc.cfl_mag_v = ch.mag_v;
                            enc.cfl_ctx_u = ch.ctx_u;
                            enc.cfl_ctx_v = ch.ctx_v;
                        }
                        // Partition path does not evaluate MHCCP; ensure the
                        // per-block flag is clear so no switch symbol is emitted.
                        enc.mhccp_allowed = false;
                        enc.mhccp_use = false;
                        // Partition leaves predict chroma with DC/CfL only; never emit a
                        // uv mode left over from an earlier whole-SB mode search.
                        enc.uv_mode = 0;
                        // A palette winner codes as DC_PRED plus the re-derived palette.
                        let mode_idx = if mode_idx == crate::av2::palette::PAL_MODE {
                            enc.palette_next = leaf::luma_palette(
                                &LumaSource {
                                    plane: yp,
                                    stride: pw,
                                },
                                (width, height),
                                sb_y,
                                sb_x,
                                64,
                                64,
                                &LumaQuantSpec {
                                    prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                                    basis: &bases.luma,
                                    qstep: sb_qstep,
                                    scan: &tables::SCAN,
                                    neutral,
                                    quant_context: qc,
                                    rdoq_lambda: self.tune.rdoq_lambda,
                                    pixel_lambda: self.still_rd(),
                                    speed: self.speed,
                                    bit_depth: self.bit_depth as i32,
                                },
                            )
                            .map(Box::new);
                            debug_assert!(enc.palette_next.is_some());
                            0
                        } else {
                            mode_idx
                        };
                        if let Some((tu64, m64)) = &tx64 {
                            enc.palette_next = None;
                            coder::encode_luma_block_tx64(
                                enc,
                                tu64,
                                dcs64,
                                (*m64, 0),
                                pc,
                                (lmidx, amidx),
                            );
                            leaf_midx = midx_class(*m64);
                        } else if mode_idx >= 5 {
                            let (_, midx) = encode_luma_block_split_dir(
                                enc,
                                &LumaSplitDirSpec {
                                    tus: &tus,
                                    skip_cdfs: &skip_cdfs,
                                    dc_sign_ctxs: &dc_sign_ctxs,
                                    mode_idx,
                                    angle_delta: adelta,
                                    has_chroma: true,
                                    part_cdf: pc,
                                    left_midx: lmidx,
                                    above_midx: amidx,
                                },
                            );
                            leaf_midx = midx as i32;
                        } else {
                            encode_luma_block_split(
                                enc,
                                &tus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                true,
                                pc,
                            );
                            leaf_midx = midx_class(mode_idx);
                        }
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 32,
                                ch: 32,
                                basis: &bases.chroma420,
                                scan: &tables::SCAN,
                                eob_cdf: EobCdf::ChrEobBin,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 1024,
                                u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: cfl_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (16, 8) => outline_leaf_420(|| {
                        // 64x32 luma → 32x16 chroma (TX_32X16, eob 512, skip TX32).
                        let (tus2, mode_idx, adelta2) = encode_luma_leaf32(
                            recy,
                            &LumaSource {
                                plane: yp,
                                stride: pw,
                            },
                            &LumaGridBlock {
                                mi_cols: tmc,
                                mi_rows: tmr,
                                y: sb_y,
                                x: sb_x,
                            },
                            &LumaQuantSpec {
                                prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                                basis: &bases.luma,
                                qstep: sb_qstep,
                                scan: &tables::SCAN,
                                neutral,
                                quant_context: qc,
                                rdoq_lambda: self.tune.rdoq_lambda,
                                pixel_lambda: self.still_rd(),
                                speed: self.speed,
                                bit_depth: self.bit_depth as i32,
                            },
                            self.still_rd().then_some(leaf::LeafDirSearch {
                                coded: &enc.sb_coded,
                                frame_width: width,
                                frame_height: height,
                                refine: crate::av2::rd::knob_or("DREF", 2.0) as usize,
                            }),
                        );
                        let (skip2, dcs2) =
                            sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 32,
                                height: 16,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c32x16,
                                scan: &SCAN32X16,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        if mode_idx >= 5 {
                            let midx = coder::encode_luma_leaf_rect64_dir(
                                enc,
                                &tus2,
                                &skip2,
                                &dcs2,
                                (mode_idx, adelta2),
                                pc,
                                true,
                                (lmidx, amidx),
                            );
                            leaf_midx = midx as i32;
                        } else {
                            encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, true, pc);
                            leaf_midx = midx_class(mode_idx);
                        }
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 32,
                                ch: 16,
                                basis: &bases.c32x16,
                                scan: &SCAN32X16,
                                eob_cdf: EobCdf::ChrEob512,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 512,
                                u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (8, 16) => outline_leaf_420(|| {
                        // 32x64 luma → 16x32 chroma (TX_16X32, eob 512, skip TX32).
                        let (tus2, mode_idx, adelta2) = encode_luma_leaf_v32x64(
                            recy,
                            &LumaSource {
                                plane: yp,
                                stride: pw,
                            },
                            &LumaGridBlock {
                                mi_cols: tmc,
                                mi_rows: tmr,
                                y: sb_y,
                                x: sb_x,
                            },
                            &LumaQuantSpec {
                                prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                                basis: &bases.luma,
                                qstep: sb_qstep,
                                scan: &tables::SCAN,
                                neutral,
                                quant_context: qc,
                                rdoq_lambda: self.tune.rdoq_lambda,
                                pixel_lambda: self.still_rd(),
                                speed: self.speed,
                                bit_depth: self.bit_depth as i32,
                            },
                            self.still_rd().then_some(leaf::LeafDirSearch {
                                coded: &enc.sb_coded,
                                frame_width: width,
                                frame_height: height,
                                refine: crate::av2::rd::knob_or("DREF", 2.0) as usize,
                            }),
                        );
                        let (skip2, dcs2) = sb_tu_contexts_pos(
                            &[(0, 0), (32, 0)],
                            &tus2,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: false,
                            },
                        );
                        let s2 = [skip2[0], skip2[1]];
                        let d2 = [dcs2[0], dcs2[1]];
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 16,
                                height: 32,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c16x32,
                                scan: &SCAN16X32,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        if mode_idx >= 5 {
                            let midx = coder::encode_luma_leaf_rect64_dir(
                                enc,
                                &tus2,
                                &s2,
                                &d2,
                                (mode_idx, adelta2),
                                pc,
                                false,
                                (lmidx, amidx),
                            );
                            leaf_midx = midx as i32;
                        } else {
                            encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, true, pc);
                            leaf_midx = midx_class(mode_idx);
                        }
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 16,
                                ch: 32,
                                basis: &bases.c16x32,
                                scan: &SCAN16X32,
                                eob_cdf: EobCdf::ChrEob512,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 512,
                                u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (8, 8) => outline_leaf_420(|| {
                        // 32x32 luma → 16x16 chroma (TX_16X16, eob 256, skip TX16).
                        // Replay pass: code the captured winner, skipping the search.
                        let forced32 = decide_mode.replay_leaf().map(|c| {
                            let (m, d, _) = crate::av2::replay::unpack_leaf_choice(c);
                            (m, d)
                        });
                        let (tu, mode_idx, adelta32) = encode_luma_leaf_s32x32(
                            recy,
                            &LumaSource {
                                plane: yp,
                                stride: pw,
                            },
                            &LumaGridBlock {
                                mi_cols: tmc,
                                mi_rows: tmr,
                                y: sb_y,
                                x: sb_x,
                            },
                            &LumaQuantSpec {
                                prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                                basis: &bases.luma,
                                qstep: sb_qstep,
                                scan: &tables::SCAN,
                                neutral,
                                quant_context: qc,
                                rdoq_lambda: self.tune.rdoq_lambda,
                                pixel_lambda: self.still_rd(),
                                speed: self.speed,
                                bit_depth: self.bit_depth as i32,
                            },
                            self.still_rd().then_some(leaf::LeafDirSearch {
                                coded: &enc.sb_coded,
                                frame_width: width,
                                frame_height: height,
                                refine: crate::av2::rd::knob_or("DREF", 2.0) as usize,
                            }),
                            forced32,
                        );
                        decide_mode.capture_leaf(crate::av2::replay::leaf_choice(
                            mode_idx,
                            adelta32 as i32,
                            0,
                        ));
                        let (skip2, dcs2) = sb_tu_contexts_pos(
                            &[(0, 0)],
                            std::slice::from_ref(&tu),
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 16,
                                height: 16,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.luma16x16,
                                scan: &SCAN16,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        // A palette winner codes as DC_PRED plus the re-derived palette.
                        let mode_idx = if mode_idx == crate::av2::palette::PAL_MODE {
                            enc.palette_next = leaf::luma_palette(
                                &LumaSource {
                                    plane: yp,
                                    stride: pw,
                                },
                                (tmc as usize * 4, tmr as usize * 4),
                                sb_y,
                                sb_x,
                                32,
                                32,
                                &LumaQuantSpec {
                                    prune: leaf::PRUNE_PROBE | leaf::PRUNE_LARGE,
                                    basis: &bases.luma,
                                    qstep: sb_qstep,
                                    scan: &tables::SCAN,
                                    neutral,
                                    quant_context: qc,
                                    rdoq_lambda: self.tune.rdoq_lambda,
                                    pixel_lambda: self.still_rd(),
                                    speed: self.speed,
                                    bit_depth: self.bit_depth as i32,
                                },
                            )
                            .map(Box::new);
                            debug_assert!(enc.palette_next.is_some());
                            0
                        } else {
                            mode_idx
                        };
                        if mode_idx >= 5 {
                            // Partition leaves predict chroma with DC/CfL only.
                            enc.uv_mode = 0;
                            let midx = coder::encode_luma_leaf_32x32_dir(
                                enc, &tu, skip2[0], dcs2[0], mode_idx, adelta32, true, pc, lmidx,
                                amidx,
                            );
                            leaf_midx = midx as i32;
                        } else {
                            encode_luma_leaf_32x32(enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc);
                            leaf_midx = midx_class(mode_idx);
                        }
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 16,
                                ch: 16,
                                basis: &bases.luma16x16,
                                scan: &SCAN16,
                                eob_cdf: EobCdf::ChrEob256,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 256,
                                u_skip_row: &SKIP_TX16_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (4, 16) => outline_leaf_420(|| {
                        // Right-edge 16x64 luma leaf → 4:2:0 chroma 8x32 (TX_8X32,
                        // coeff 8x32, SCAN8X32, eob 256, ctx-2 SKIP_TX16). Reuses the
                        // luma8x32 basis (identical 8x32 geometry).
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            16,
                            64,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let lev = bases.luma16x64.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 64, pred),
                                bases.luma16x64.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &SCAN16X32,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            16,
                            64,
                            &crate::av2::itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &SCAN16X32,
                                16,
                                64,
                                self.bit_depth as i32,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            4,
                            16,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 8,
                                height: 32,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.luma8x32,
                                scan: &SCAN8X32,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, true, pc);
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 8,
                                ch: 32,
                                basis: &bases.luma8x32,
                                scan: &SCAN8X32,
                                eob_cdf: EobCdf::ChrEob256,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 256,
                                u_skip_row: &SKIP_TX16_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (16, 4) => outline_leaf_420(|| {
                        // Bottom-edge 64x16 luma leaf → 4:2:0 chroma 32x8 (TX_32X8,
                        // coeff 32x8, SCAN32X8, eob 256, ctx-2 SKIP_TX16). Reuses the
                        // luma32x8 basis.
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            16,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let lev = bases.luma64x16.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 64, 16, pred),
                                bases.luma64x16.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &SCAN32X16,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            16,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &SCAN32X16,
                                64,
                                16,
                                self.bit_depth as i32,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            16,
                            4,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 32,
                                height: 8,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.luma32x8,
                                scan: &SCAN32X8,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, true, pc);
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 32,
                                ch: 8,
                                basis: &bases.luma32x8,
                                scan: &SCAN32X8,
                                eob_cdf: EobCdf::ChrEob256,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 256,
                                u_skip_row: &SKIP_TX16_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (2, 8) => outline_leaf_420(|| {
                        // Right-edge 8×32 luma leaf (residue-2 width) → 4:2:0 chroma
                        // 4×16 (TX_4X16, SCAN4X16, eob 64, ctx-1 SKIP_TX8). Luma
                        // TX_8X32 long-side-32 (min=1 short cdf).
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            8,
                            32,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let lev = bases.luma8x32.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 32, pred),
                                bases.luma8x32.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &SCAN8X32,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            8,
                            32,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &SCAN8X32,
                                8,
                                32,
                                self.bit_depth as i32,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            2,
                            8,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 4,
                                height: 16,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c4x16,
                                scan: &tables::SCAN4X16,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        encode_luma_leaf_8x32(enc, &tu, skip, dcs, 0, true, pc);
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 4,
                                ch: 16,
                                basis: &bases.c4x16,
                                scan: &tables::SCAN4X16,
                                eob_cdf: EobCdf::ChrEob64,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 64,
                                u_skip_row: &SKIP_TX8_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (8, 2) => outline_leaf_420(|| {
                        // Bottom-edge 32×8 luma leaf (residue-2 height) → 4:2:0 chroma
                        // 16×4 (TX_16X4, SCAN16X4, eob 64, ctx-1 SKIP_TX8).
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            32,
                            8,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let lev = bases.luma32x8.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 32, 8, pred),
                                bases.luma32x8.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &SCAN32X8,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            32,
                            8,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &SCAN32X8,
                                32,
                                8,
                                self.bit_depth as i32,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            8,
                            2,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 16,
                                height: 4,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c16x4,
                                scan: &tables::SCAN16X4,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        encode_luma_leaf_32x8(enc, &tu, skip, dcs, 0, true, pc);
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 16,
                                ch: 4,
                                basis: &bases.c16x4,
                                scan: &tables::SCAN16X4,
                                eob_cdf: EobCdf::ChrEob64,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 64,
                                u_skip_row: &SKIP_TX8_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (4, 8) => outline_leaf_420(|| {
                        // Bottom-right 16×32 corner leaf (residue-4 width ×
                        // residue-{6,8} height) → 4:2:0 chroma 8×16 (TX_8X16,
                        // SCAN8X16, eob class 128, ctx-2 SKIP_TX16). Luma TX_16X32
                        // is EXT_TX long-side-32 (DCT_DCT via the long32 coder).
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            16,
                            32,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let lev = bases.luma16x32.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 32, pred),
                                bases.luma16x32.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &SCAN16X32,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            16,
                            32,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &SCAN16X32,
                                16,
                                32,
                                self.bit_depth as i32,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            4,
                            8,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 8,
                                height: 16,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c8x16,
                                scan: &tables::SCAN8X16,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        encode_luma_leaf_16x32(enc, &tu, skip, dcs, 0, true, pc);
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 8,
                                ch: 16,
                                basis: &bases.c8x16,
                                scan: &tables::SCAN8X16,
                                eob_cdf: EobCdf::ChrEob128,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 128,
                                u_skip_row: &SKIP_TX16_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (8, 4) => outline_leaf_420(|| {
                        // Bottom-right 32×16 corner leaf (residue-{6,8} width ×
                        // residue-4 height) → 4:2:0 chroma 16×8 (TX_16X8, SCAN16X8,
                        // eob class 128, ctx-2 SKIP_TX16). Luma TX_32X16 long-side-32.
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            32,
                            16,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let lev = bases.luma32x16.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 32, 16, pred),
                                bases.luma32x16.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &SCAN32X16,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            32,
                            16,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &SCAN32X16,
                                32,
                                16,
                                self.bit_depth as i32,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            8,
                            4,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 16,
                                height: 8,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c16x8,
                                scan: &tables::SCAN16X8,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        encode_luma_leaf_32x16(enc, &tu, skip, dcs, 0, true, pc);
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 16,
                                ch: 8,
                                basis: &bases.c16x8,
                                scan: &tables::SCAN16X8,
                                eob_cdf: EobCdf::ChrEob128,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 128,
                                u_skip_row: &SKIP_TX16_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (4, 4) => outline_leaf_420(|| {
                        // Replay pass: code the captured winner, skipping the search.
                        let forced16 = decide_mode
                            .replay_leaf()
                            .map(crate::av2::replay::unpack_leaf_choice);
                        // Bottom-right 16×16 corner leaf (residue 4 in both dims):
                        // full-AC TX_16X16 luma with 4-way ADST RD (DCT_DCT /
                        // ADST_ADST / ADST_DCT / DCT_ADST, DC mode) → 4:2:0 chroma
                        // 8×8 (TX_8X8, SCAN8X8, eob class 64, skip txs_ctx 1).
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            16,
                            16,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let resid = aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred),
                            bases.luma16x16.qstep as f32 / sb_qstep as f32,
                        );
                        let pred_flat = [pred; 256];
                        let mut src16 = [0f32; 256];
                        for (dst_row, src_row) in src16
                            .as_chunks_mut::<16>()
                            .0
                            .iter_mut()
                            .zip(rect_rows(yp, pw, sb_y, sb_x, 16, 16))
                        {
                            dst_row.copy_from_slice(src_row);
                        }
                        let rate = coeff_rate_f32;
                        let sse = |rec: &[f32]| -> f32 { tx16_distortion(&src16, rec) };
                        let lambda = leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
                        let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                        let rec_dct = itx422::reconstruct_luma16(
                            &pred_flat,
                            &lev_dct,
                            sb_qstep,
                            &SCAN16,
                            self.bit_depth as i32,
                        );
                        let dist_dct = sse(&rec_dct);
                        let cost_dct = dist_dct + lambda * rate(&lev_dct);
                        let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
                        let rec_adst = itx422::reconstruct_luma16_adst(
                            &pred_flat,
                            &lev_adst,
                            sb_qstep,
                            &SCAN16,
                            true,
                            true,
                            self.bit_depth as i32,
                        );
                        let dist_adst = sse(&rec_adst);
                        let cost_adst =
                            dist_adst + lambda * (rate(&lev_adst) + TX16_TYPE_RATE_DELTA[1]);
                        let lev_ad = bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
                        let rec_ad = itx422::reconstruct_luma16_adst(
                            &pred_flat,
                            &lev_ad,
                            sb_qstep,
                            &SCAN16,
                            false,
                            true,
                            self.bit_depth as i32,
                        );
                        let dist_ad = sse(&rec_ad);
                        let cost_ad = dist_ad + lambda * (rate(&lev_ad) + TX16_TYPE_RATE_DELTA[2]);
                        let lev_da = bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
                        let rec_da = itx422::reconstruct_luma16_adst(
                            &pred_flat,
                            &lev_da,
                            sb_qstep,
                            &SCAN16,
                            true,
                            false,
                            self.bit_depth as i32,
                        );
                        let dist_da = sse(&rec_da);
                        let cost_da = dist_da + lambda * (rate(&lev_da) + TX16_TYPE_RATE_DELTA[3]);
                        let choice = choose_tx16_type(
                            [cost_dct, cost_adst, cost_ad, cost_da],
                            [dist_dct, dist_adst, dist_ad, dist_da],
                            [
                                false,
                                tx16_dc_only(&lev_adst),
                                tx16_dc_only(&lev_ad),
                                tx16_dc_only(&lev_da),
                            ],
                        );
                        let (lev, rec, tx_idx): (&[f32], &[f32; 256], usize) = match choice {
                            1 => (&lev_adst, &rec_adst, 1),
                            2 => (&lev_ad, &rec_ad, 2),
                            3 => (&lev_da, &rec_da, 3),
                            _ => (&lev_dct, &rec_dct, 0),
                        };
                        // Every other luma mode: SMOOTH, SMOOTH_V, SMOOTH_H, PAETH and (Slow /
                        // Medium) the directional modes, each with the four TX_16X16 types —
                        // at size class 2 av2_md_idx2type maps idx 0..=3 to the same types
                        // for every intra mode class. Directional (mode, delta) pairs are
                        // SATD-ranked and only the best few get the full RD.
                        // Mode bits: exact from the default luma-mode CDFs, or flat.
                        let exact_modes = crate::av2::rd::knob_or("EXACTMODE", 0.0) > 0.0;
                        let mb16 = |m: usize| -> f32 {
                            if m >= 5 {
                                crate::av2::rd::knob_or("MBDIR", 9.0)
                            } else if exact_modes {
                                crate::av2::entropy::RangeEncoder::y_mode_nondir_bits(enc.y_ctx, m)
                            } else if m == 0 {
                                0.0
                            } else {
                                crate::av2::rd::knob_or("MB16", 6.0)
                            }
                        };
                        let mut best_cost =
                            [cost_dct, cost_adst, cost_ad, cost_da][tx_idx] + lambda * mb16(0);
                        let mut best: Option<(Vec<f32>, [f32; 256], usize, leaf::SquareLumaMode)> =
                            None;
                        if self.still_rd() && crate::av2::rd::knob("LMODE16") > 0.0 {
                            let scale = bases.luma16x16.qstep as f32 / sb_qstep as f32;
                            let block = leaf::SquareLumaBlock {
                                stride: pw,
                                frame_width: width,
                                frame_height: height,
                                y: sb_y,
                                x: sb_x,
                                size: 16,
                            };
                            // Edge-filter type 1 when the above or left block is SMOOTH*.
                            let smooth_neighbour = (lmr > 0 && midx_above[lmc] == MIDX_SMOOTH)
                                || (lmc > 0 && midx_left[lmr] == MIDX_SMOOTH);
                            let refs = std::cell::RefCell::new(leaf::SquareRefCache::default());
                            let predict = |lm: leaf::SquareLumaMode| -> [f32; 256] {
                                let mut out = [0f32; 256];
                                leaf::predict_luma_square_cached(
                                    &*recy,
                                    &block,
                                    &enc.sb_coded,
                                    lm,
                                    neutral,
                                    self.bit_depth as i32,
                                    &mut refs.borrow_mut(),
                                    &mut out,
                                );
                                out
                            };
                            let mut cands: Vec<leaf::SquareLumaMode> =
                                (1..=4usize).map(leaf::SquareLumaMode::nondir).collect();
                            let dir_keep = crate::av2::rd::knob_or("DIR16", 3.0) as usize;
                            if let Some((m, d, _)) = forced16 {
                                cands = match m {
                                    0 | crate::av2::palette::PAL_MODE => Vec::new(),
                                    1..=4 => vec![leaf::SquareLumaMode::nondir(m)],
                                    _ => vec![leaf::SquareLumaMode {
                                        mode: m,
                                        angle_delta: d,
                                        smooth_neighbour,
                                    }],
                                };
                            } else if dir_keep > 0 && self.speed.try_directional() {
                                let dir_lm = |mode, angle_delta| leaf::SquareLumaMode {
                                    mode,
                                    angle_delta,
                                    smooth_neighbour,
                                };
                                let picked = leaf::screen_directional(
                                    dir_keep,
                                    self.speed.try_angle_deltas(),
                                    crate::av2::rd::knob_or("DREF", 2.0) as usize,
                                    |m, d| {
                                        crate::av2::metrics::satd_f32(
                                            &src16,
                                            16,
                                            &predict(dir_lm(m, d)),
                                            16,
                                            16,
                                            16,
                                        )
                                    },
                                );
                                cands.extend(picked.into_iter().map(|(m, d)| dir_lm(m, d)));
                            }
                            // Unified SATD prune (knob PRK16 = K), as the 8x8 leaf.
                            let prk16 = crate::av2::rd::knob_or("PRK16", 2.0) as usize;
                            if forced16.is_none() && prk16 > 0 && cands.len() > prk16 {
                                let mut ranked: Vec<(u64, leaf::SquareLumaMode)> = cands
                                    .iter()
                                    .map(|&lm| {
                                        (
                                            crate::av2::metrics::satd_f32(
                                                &src16,
                                                16,
                                                &predict(lm),
                                                16,
                                                16,
                                                16,
                                            ),
                                            lm,
                                        )
                                    })
                                    .collect();
                                ranked.sort_by_key(|&(sc, _)| sc);
                                cands = ranked.into_iter().take(prk16).map(|(_, lm)| lm).collect();
                            }
                            // TX16W: DCT only for every candidate, then the ADST types for
                            // the DCT-best candidate alone.
                            let tx16w =
                                crate::av2::rd::knob_or("TX16W", 1.0) > 0.0 && forced16.is_none();
                            let n_first = cands.len();
                            let mut order: Vec<(leaf::SquareLumaMode, bool)> =
                                cands.into_iter().map(|lm| (lm, !tx16w)).collect();
                            let mut dct_best: Option<(f32, leaf::SquareLumaMode)> = None;
                            let mut k = 0;
                            while k < order.len() {
                                let (lm, all_tx) = order[k];
                                let second = k >= n_first;
                                k += 1;
                                let pred_m = predict(lm);
                                let resid_m: Vec<f32> = src16
                                    .iter()
                                    .zip(&pred_m)
                                    .map(|(&s, &q)| (s - q) * scale)
                                    .collect();
                                for (tx, &tx_delta) in TX16_TYPE_RATE_DELTA.iter().enumerate() {
                                    if forced16.is_some_and(|(_, _, ft)| ft != tx) {
                                        continue;
                                    }
                                    if (!all_tx && tx > 0) || (second && tx == 0) {
                                        continue;
                                    }
                                    let (basis, flips) = match tx {
                                        0 => (&bases.luma16x16, (false, false)),
                                        1 => (&bases.luma16x16_adst, (true, true)),
                                        2 => (&bases.luma16x16_adst_dct, (false, true)),
                                        _ => (&bases.luma16x16_dct_adst, (true, false)),
                                    };
                                    let lev_m = basis.project_scan(&resid_m, 0.0, &SCAN16);
                                    if tx > 0 && tx16_dc_only(&lev_m) {
                                        continue;
                                    }
                                    let rec_m = if tx == 0 {
                                        itx422::reconstruct_luma16(
                                            &pred_m,
                                            &lev_m,
                                            sb_qstep,
                                            &SCAN16,
                                            self.bit_depth as i32,
                                        )
                                    } else {
                                        itx422::reconstruct_luma16_adst(
                                            &pred_m,
                                            &lev_m,
                                            sb_qstep,
                                            &SCAN16,
                                            flips.0,
                                            flips.1,
                                            self.bit_depth as i32,
                                        )
                                    };
                                    let cost = crate::av2::rd::rd_cost(
                                        sse(&rec_m),
                                        rate(&lev_m) + tx_delta + mb16(lm.mode),
                                        lambda,
                                    );
                                    if tx == 0 && dct_best.is_none_or(|(c, _)| cost < c) {
                                        dct_best = Some((cost, lm));
                                    }
                                    if cost < best_cost {
                                        best_cost = cost;
                                        best = Some((lev_m, rec_m, tx, lm));
                                    }
                                }
                                if tx16w
                                    && k == n_first
                                    && let Some((_, blm)) = dct_best
                                {
                                    order.push((blm, true));
                                }
                            }
                        }
                        // Screen content: the DC-predicted palette joins the RD (its
                        // syntax bits replace the mode bits). A replayed palette leaf
                        // re-derives the same palette from the source.
                        let mut best_pal: Option<crate::av2::palette::Palette> = None;
                        let pal_forced =
                            forced16.is_some_and(|(m, _, _)| m == crate::av2::palette::PAL_MODE);
                        if enc.sct
                            && sb_y + 16 <= height
                            && sb_x + 16 <= width
                            && forced16.is_none_or(|(m, _, _)| m == crate::av2::palette::PAL_MODE)
                        {
                            let lambda_p =
                                leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
                            if let Some(pal) = crate::av2::palette::search(
                                &src16,
                                16,
                                16,
                                self.bit_depth as u32,
                                lambda_p,
                            ) {
                                let mut pred_m = [0f32; 256];
                                pred_m.copy_from_slice(&pal.pred());
                                let scale = bases.luma16x16.qstep as f32 / sb_qstep as f32;
                                let resid_m: Vec<f32> = src16
                                    .iter()
                                    .zip(&pred_m)
                                    .map(|(&s, &q)| (s - q) * scale)
                                    .collect();
                                if pal_forced {
                                    best_cost = f32::INFINITY;
                                }
                                for (tx, &tx_delta) in TX16_TYPE_RATE_DELTA.iter().enumerate() {
                                    if forced16.is_some_and(|(_, _, ft)| ft != tx) {
                                        continue;
                                    }
                                    let (basis, flips) = match tx {
                                        0 => (&bases.luma16x16, (false, false)),
                                        1 => (&bases.luma16x16_adst, (true, true)),
                                        2 => (&bases.luma16x16_adst_dct, (false, true)),
                                        _ => (&bases.luma16x16_dct_adst, (true, false)),
                                    };
                                    let lev_m = basis.project_scan(&resid_m, 0.0, &SCAN16);
                                    if tx > 0 && tx16_dc_only(&lev_m) {
                                        continue;
                                    }
                                    let rec_m = if tx == 0 {
                                        itx422::reconstruct_luma16(
                                            &pred_m,
                                            &lev_m,
                                            sb_qstep,
                                            &SCAN16,
                                            self.bit_depth as i32,
                                        )
                                    } else {
                                        itx422::reconstruct_luma16_adst(
                                            &pred_m,
                                            &lev_m,
                                            sb_qstep,
                                            &SCAN16,
                                            flips.0,
                                            flips.1,
                                            self.bit_depth as i32,
                                        )
                                    };
                                    let cost = crate::av2::rd::rd_cost(
                                        sse(&rec_m),
                                        rate(&lev_m) + tx_delta + pal.bits,
                                        lambda_p,
                                    );
                                    if cost < best_cost {
                                        best_cost = cost;
                                        best = Some((
                                            lev_m,
                                            rec_m,
                                            tx,
                                            leaf::SquareLumaMode::nondir(0),
                                        ));
                                        best_pal = Some(pal.clone());
                                    }
                                }
                            }
                        }
                        let (lev, rec, tx_idx, mode16) = match &best {
                            Some((l, r, t, lm)) => (&l[..], r, *t, *lm),
                            None => (lev, rec, tx_idx, leaf::SquareLumaMode::nondir(0)),
                        };
                        decide_mode.capture_leaf(crate::av2::replay::leaf_choice(
                            if best_pal.is_some() {
                                crate::av2::palette::PAL_MODE
                            } else {
                                mode16.mode
                            },
                            mode16.angle_delta,
                            tx_idx,
                        ));
                        enc.palette_next = best_pal.map(Box::new);
                        put_block_rect(recy, pw, sb_y, sb_x, 16, 16, rec);
                        let tu: Vec<Coeff> = levels_to_coeffs(lev);
                        let (_s, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            4,
                            4,
                        );
                        let skip = SKIP_TX16_QC[qc][0] as u32;
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 8,
                                height: 8,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c8x8,
                                scan: &SCAN8X8,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        if mode16.mode >= 5 {
                            // Partition leaves predict chroma with DC/CfL only.
                            enc.uv_mode = 0;
                            let midx = coder::encode_luma_leaf_16x16_dir(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                mode16.mode,
                                mode16.angle_delta as i8,
                                true,
                                pc,
                                11074,
                                tx_idx,
                                lmidx,
                                amidx,
                            );
                            leaf_midx = midx as i32;
                        } else {
                            encode_luma_leaf_16x16_full(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                mode16.mode,
                                true,
                                pc,
                                11074,
                                tx_idx,
                            );
                            leaf_midx = midx_class(mode16.mode);
                        }
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 8,
                                ch: 8,
                                basis: &bases.c8x8,
                                scan: &SCAN8X8,
                                eob_cdf: EobCdf::ChrEob64,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 64,
                                u_skip_row: &SKIP_TX8_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (2, 2) => outline_leaf_420(|| {
                        // Bottom-right 8×8 corner leaf (residue-2 both axes), TX_8X8 ctx-1.
                        // Luma TX_8X8 (szctx=1, do_part_cdf=3148, ext-tx txtp_ext(min=1)
                        // DCT_DCT idx 0); 4:2:0 chroma is one 4×4 (TX_4X4) TU per plane.
                        let pred = dc_pred_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            8,
                            8,
                            neutral,
                            self.bit_depth as i32,
                        );
                        let scale8 = bases.c8x8.qstep as f32 / sb_qstep as f32;
                        let mut lev = bases.c8x8.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 8, pred),
                                scale8,
                            ),
                            0.0,
                            &SCAN8X8,
                        );
                        let mut rec8 = itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            sb_qstep,
                            &SCAN8X8,
                            8,
                            8,
                            self.bit_depth as i32,
                        );
                        // Non-directional luma modes (DCT only: the 8x8 leaf codes tx index 0).
                        let mut mode8 = 0usize;
                        let mut adelta8 = 0i32;
                        let mut tx8 = 0usize;
                        if self.still_rd() && crate::av2::rd::knob("LMODE8") > 0.0 {
                            let lambda =
                                leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());

                            let exact_modes = crate::av2::rd::knob_or("EXACTMODE", 0.0) > 0.0;
                            let mb8 = |m: usize| -> f32 {
                                if exact_modes {
                                    crate::av2::entropy::RangeEncoder::y_mode_nondir_bits(
                                        enc.y_ctx, m,
                                    )
                                } else if m == 0 {
                                    0.0
                                } else {
                                    crate::av2::rd::knob_or("MB8", 6.0)
                                }
                            };
                            let mut src8 = [0f32; 64];
                            for (d, &v) in src8
                                .iter_mut()
                                .zip(rect_rows(yp, pw, sb_y, sb_x, 8, 8).flatten())
                            {
                                *d = v;
                            }
                            let sse8 = |rec: &[f32]| -> f32 {
                                crate::av2::helpers::dist_rounded(&src8, 8, rec, 8, 8, 8)
                            };
                            let mut best_cost = crate::av2::rd::rd_cost(
                                sse8(&rec8),
                                coeff_rate_f32(&lev) + mb8(0),
                                lambda,
                            );
                            let block = leaf::SquareLumaBlock {
                                stride: pw,
                                frame_width: width,
                                frame_height: height,
                                y: sb_y,
                                x: sb_x,
                                size: 8,
                            };
                            let smooth_neighbour = (lmr > 0 && midx_above[lmc] == MIDX_SMOOTH)
                                || (lmc > 0 && midx_left[lmr] == MIDX_SMOOTH);
                            let refs = std::cell::RefCell::new(leaf::SquareRefCache::default());
                            let predict = |lm: leaf::SquareLumaMode| -> [f32; 64] {
                                let mut out = [0f32; 64];
                                leaf::predict_luma_square_cached(
                                    &*recy,
                                    &block,
                                    &enc.sb_coded,
                                    lm,
                                    neutral,
                                    self.bit_depth as i32,
                                    &mut refs.borrow_mut(),
                                    &mut out,
                                );
                                out
                            };
                            let mut cands: Vec<leaf::SquareLumaMode> =
                                (1..=4usize).map(leaf::SquareLumaMode::nondir).collect();
                            // Directional modes: SATD-ranked (mode, delta) pairs, best few
                            // get the full RD (as the 16x16 leaf).
                            let dir_keep = crate::av2::rd::knob_or("DIR8", 3.0) as usize;
                            if dir_keep > 0 && self.speed.try_directional() {
                                let dir_lm = |mode, angle_delta| leaf::SquareLumaMode {
                                    mode,
                                    angle_delta,
                                    smooth_neighbour,
                                };
                                let satd = |m: usize, d: i32| {
                                    crate::av2::metrics::satd_f32(
                                        &src8,
                                        8,
                                        &predict(dir_lm(m, d)),
                                        8,
                                        8,
                                        8,
                                    )
                                };
                                // Reuse: the probe's picks for this block (screened against
                                // the probe's recon) join a light re-screen of the 8 base
                                // angles, ranked on the real recon (mode 2), or replace
                                // the screen outright (mode 1).
                                let reuse = crate::av2::rd::knob_or("DIR8P", 1.0) as u32;
                                let probe = luma_partition
                                    .dir8_at(sb_y - row * 64, sb_x - col * 64)
                                    .filter(|_| reuse > 0);
                                let picked = match probe {
                                    Some(p) if reuse == 1 => p,
                                    Some(p) => {
                                        let mut ranked: Vec<(u64, (usize, i32))> =
                                            (5..=12usize).map(|m| (satd(m, 0), (m, 0))).collect();
                                        for md in p {
                                            if md.1 != 0 {
                                                ranked.push((satd(md.0, md.1), md));
                                            }
                                        }
                                        ranked.sort_by_key(|&(c, _)| c);
                                        ranked
                                            .into_iter()
                                            .take(dir_keep)
                                            .map(|(_, md)| md)
                                            .collect()
                                    }
                                    None => leaf::screen_directional(
                                        dir_keep,
                                        self.speed.try_angle_deltas(),
                                        crate::av2::rd::knob_or("DREF8", 2.0) as usize,
                                        satd,
                                    ),
                                };
                                cands.extend(picked.into_iter().map(|(m, d)| dir_lm(m, d)));
                            }
                            // Unified SATD prune (knob PRK8 = K): only the K SATD-best non-DC
                            // candidates get the full RD.
                            let prk8 = crate::av2::rd::knob_or("PRK8", 2.0) as usize;
                            if prk8 > 0 && cands.len() > prk8 {
                                let mut ranked: Vec<(u64, leaf::SquareLumaMode)> = cands
                                    .iter()
                                    .map(|&lm| {
                                        (
                                            crate::av2::metrics::satd_f32(
                                                &src8,
                                                8,
                                                &predict(lm),
                                                8,
                                                8,
                                                8,
                                            ),
                                            lm,
                                        )
                                    })
                                    .collect();
                                ranked.sort_by_key(|&(sc, _)| sc);
                                cands = ranked.into_iter().take(prk8).map(|(_, lm)| lm).collect();
                            }
                            // TX_8X8 types idx 0..=3 (DCT, ADST_ADST, ADST_DCT, DCT_ADST: the
                            // same for every intra mode class at this size).
                            use crate::av2::itx::{Tx1d, TxClass, txtp};
                            let n_tx = if crate::av2::rd::knob_or("TX8T", 0.0) > 0.0 {
                                4
                            } else {
                                1
                            };
                            let tx8_basis = |col_adst: bool, row_adst: bool| {
                                let mut b = bases.c8x8.clone();
                                b.fwd = crate::av2::fdct::FwdKind::Tx8 { col_adst, row_adst };
                                b
                            };
                            let bases8 = [
                                bases.c8x8.clone(),
                                tx8_basis(true, true),
                                tx8_basis(true, false),
                                tx8_basis(false, true),
                            ];
                            let txtp8 = [
                                0,
                                txtp(Tx1d::Adst, Tx1d::Adst, TxClass::TwoD),
                                txtp(Tx1d::Dct, Tx1d::Adst, TxClass::TwoD),
                                txtp(Tx1d::Adst, Tx1d::Dct, TxClass::TwoD),
                            ];
                            let mb_dir = crate::av2::rd::knob_or("MBDIR8", 9.0);
                            let dc_lm = leaf::SquareLumaMode::nondir(0);
                            let (mut lev_m, mut rec_m) = (Vec::new(), Vec::new());
                            for lm in std::iter::once(dc_lm).chain(cands) {
                                let p = if lm.mode == 0 {
                                    [pred; 64]
                                } else {
                                    predict(lm)
                                };
                                let mut resid = [0f32; 64];
                                let mut p_i = [0i32; 64];
                                for (((r, pi), &s), &q) in
                                    resid.iter_mut().zip(&mut p_i).zip(&src8).zip(&p)
                                {
                                    *r = (s - q) * scale8;
                                    *pi = q as i32;
                                }
                                let mbits = if lm.mode >= 5 { mb_dir } else { mb8(lm.mode) };
                                for tx in 0..n_tx {
                                    // DC + DCT_DCT is the starting point above.
                                    if lm.mode == 0 && tx == 0 {
                                        continue;
                                    }
                                    bases8[tx].project_scan_into(&resid, 0.0, &SCAN8X8, &mut lev_m);
                                    if tx > 0 && tx16_dc_only(&lev_m) {
                                        continue;
                                    }
                                    rec_m.resize(64, 0.0);
                                    itx422::reconstruct_chroma_pred_tx_into(
                                        &p_i,
                                        &lev_m,
                                        sb_qstep,
                                        &SCAN8X8,
                                        8,
                                        8,
                                        self.bit_depth as i32,
                                        txtp8[tx],
                                        &mut rec_m,
                                    );
                                    let cost = crate::av2::rd::rd_cost(
                                        sse8(&rec_m),
                                        coeff_rate_f32(&lev_m) + mbits + TX16_TYPE_RATE_DELTA[tx],
                                        lambda,
                                    );
                                    if cost < best_cost {
                                        best_cost = cost;
                                        std::mem::swap(&mut lev, &mut lev_m);
                                        std::mem::swap(&mut rec8, &mut rec_m);
                                        mode8 = lm.mode;
                                        adelta8 = lm.angle_delta;
                                        tx8 = tx;
                                    }
                                }
                            }
                        }
                        put_block_rect(recy, pw, sb_y, sb_x, 8, 8, &rec8);
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            2,
                            2,
                        );
                        use crate::av2::coder::{
                            SCAN4X4_LOSSY, SCAN4X4_LOSSY_PACKED, encode_chroma_tu4_scan,
                        };
                        // 4x4 chroma: DC or classic CfL (MHCCP is not allowed for 4x4 chroma),
                        // scored on the real 4x4 coding. Decided before the luma emit because
                        // is_cfl / the alphas are written with the mode info.
                        let bd = self.bit_depth as i32;
                        let scale4 = bases.c4x4.qstep as f32 / sb_qstep as f32;
                        let dc_u = dc_pred_rect(recu, pcw, cy, cx, 4, 4, neutral, bd);
                        let dc_v = dc_pred_rect(recv, pcw, cy, cx, 4, 4, neutral, bd);
                        let src4 = |plane: &[f32]| -> [f32; 16] {
                            let mut out = [0f32; 16];
                            for (d, &v) in out
                                .iter_mut()
                                .zip(rect_rows(plane, pcw, cy, cx, 4, 4).flatten())
                            {
                                *d = v;
                            }
                            out
                        };
                        let (su4, sv4) = (src4(up), src4(vp));
                        let rq4 = if self.still_rd() {
                            self.tune.chroma_rdoq_lambda
                        } else {
                            0.0
                        };
                        // The decoder derives the 4x4 chroma tx type from the uv mode (avm
                        // intra_mode_to_tx_type): 0 DCT_DCT, 1 ADST_ADST, 2 ADST_DCT (vertical
                        // ADST), 3 DCT_ADST (horizontal ADST).
                        use crate::av2::itx::{Tx1d, TxClass, txtp};
                        let uv_tx = |m: usize| -> usize {
                            match m {
                                1 | 4 => 1,
                                2 | 5 => 2,
                                3 | 6 => 3,
                                _ => 0,
                            }
                        };
                        let tx4_basis = |col_adst: bool, row_adst: bool| {
                            let mut b = bases.c4x4.clone();
                            b.fwd = crate::av2::fdct::FwdKind::Tx4 { col_adst, row_adst };
                            b
                        };
                        let bases4 = [
                            bases.c4x4.clone(),
                            tx4_basis(true, true),
                            tx4_basis(true, false),
                            tx4_basis(false, true),
                        ];
                        let txtp4 = [
                            0,
                            txtp(Tx1d::Adst, Tx1d::Adst, TxClass::TwoD),
                            txtp(Tx1d::Dct, Tx1d::Adst, TxClass::TwoD),
                            txtp(Tx1d::Adst, Tx1d::Dct, TxClass::TwoD),
                        ];
                        // (levels, recon) of one 4x4 chroma TU against a per-pixel predictor.
                        let code4 = |src: &[f32],
                                     pred: &[i32],
                                     voff: usize,
                                     ti: usize,
                                     lev: &mut Vec<f32>| {
                            let mut resid = [0f32; 16];
                            for ((r, &s), &q) in resid.iter_mut().zip(src).zip(pred) {
                                *r = (s - q as f32) * scale4;
                            }
                            chroma422::project_chroma_rdoq_into(
                                &bases4[ti],
                                &resid,
                                &SCAN4X4_LOSSY_PACKED,
                                qc,
                                16,
                                voff,
                                rq4,
                                lev,
                            );
                            let mut rec = [0f32; 16];
                            itx422::reconstruct_chroma_pred_tx_into(
                                pred,
                                lev,
                                sb_qstep,
                                &SCAN4X4_LOSSY_PACKED,
                                4,
                                4,
                                bd,
                                txtp4[ti],
                                &mut rec,
                            );
                            rec
                        };
                        let lambda4 = leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
                        // RD cost of coding both 4x4 planes against (U, V) predictors.
                        let lev_scratch = std::cell::RefCell::new(Vec::new());
                        let j = |preds: [&[i32]; 2], extra_bits: f32, ti: usize| -> f32 {
                            let mut j = lambda4 * extra_bits;
                            for (src, pred, voff) in [(&su4, preds[0], 0), (&sv4, preds[1], 4)] {
                                let lev = &mut *lev_scratch.borrow_mut();
                                let rec = code4(src, pred, voff, ti, lev);
                                let d =
                                    crate::av2::helpers::dist_rounded(&src[..], 4, &rec, 4, 4, 4);
                                j += crate::av2::rd::rd_cost(d, coeff_rate_f32(lev), lambda4);
                            }
                            j
                        };
                        let (dcu, dcv) = (dc_u.round() as i32, dc_v.round() as i32);
                        // UV intra mode (SMOOTH*, PAETH, V/H) against DC. The 4x4 block's
                        // top-right / bottom-left follow the co-located 8x8 luma block's
                        // availability, halved. Diagonal modes are left out: their edge filter
                        // depends on the neighbours' uv_mode, which is not tracked.
                        enc.uv_mode = 0;
                        let mut pred_uv: ([i32; 16], [i32; 16]) = ([dcu; 16], [dcv; 16]);
                        let mut tx_i = 0usize;
                        let mut best_j = f32::INFINITY;
                        if self.still_rd() && self.tune.chroma_mode_search && !enc.sdp_luma {
                            let edges4 = {
                                let (tr, bl) = leaf::square_edge_px(
                                    &leaf::SquareLumaBlock {
                                        stride: pw,
                                        frame_width: width,
                                        frame_height: height,
                                        y: sb_y,
                                        x: sb_x,
                                        size: 8,
                                    },
                                    &enc.sb_coded,
                                    true,
                                    true,
                                );
                                (tr / 2, bl / 2)
                            };
                            best_j = j([&pred_uv.0, &pred_uv.1], 0.0, 0);
                            let modes: &[usize] = if self.speed.reduced_modes() {
                                &[1, 4, 5, 6]
                            } else {
                                &[1, 2, 3, 4, 5, 6]
                            };
                            let pred4 = |m: usize, rec: &[f32]| -> [i32; 16] {
                                let mut pf = [0f32; 16];
                                chroma422::predict_chroma_sq_into(
                                    rec,
                                    pcw,
                                    cy,
                                    cx,
                                    4,
                                    m,
                                    neutral,
                                    mhccp_bounds.chroma_width,
                                    mhccp_bounds.chroma_height,
                                    Some(edges4),
                                    &mut pf,
                                );
                                pf.map(|p| (p + 0.5) as i32)
                            };
                            // CUVK4 = K: only the K uv modes with the lowest U+V prediction SAD get
                            // the full RD.
                            let cuvk4 = crate::av2::rd::knob_or("CUVK4", 0.0) as usize;
                            let mut kept4: Vec<usize> = modes.to_vec();
                            if cuvk4 > 0 && kept4.len() > cuvk4 {
                                let sad = |src: &[f32], p: &[i32; 16]| -> f32 {
                                    src.iter().zip(p).map(|(&a, &b)| (a - b as f32).abs()).sum()
                                };
                                let mut ranked: Vec<(f32, usize)> = kept4
                                    .iter()
                                    .map(|&m| {
                                        (
                                            sad(&su4, &pred4(m, &*recu))
                                                + sad(&sv4, &pred4(m, &*recv)),
                                            m,
                                        )
                                    })
                                    .collect();
                                ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
                                kept4 = ranked.into_iter().take(cuvk4).map(|(_, m)| m).collect();
                                kept4.sort_unstable();
                            }
                            for &m in &kept4 {
                                let pred = |rec: &[f32]| pred4(m, rec);
                                let (pu, pv) = (pred(&*recu), pred(&*recv));
                                let c = j([&pu, &pv], 2.0, uv_tx(m));
                                if c < best_j {
                                    best_j = c;
                                    enc.uv_mode = m;
                                    tx_i = uv_tx(m);
                                    pred_uv = (pu, pv);
                                }
                            }
                        }
                        if enc.cfl
                            && self.still_rd()
                            && !enc.sdp_luma
                            && crate::av2::rd::knob("CFL4") > 0.0
                        {
                            let luma: Vec<i32> = rect_rows(recy, pw, sb_y, sb_x, 8, 8)
                                .flatten()
                                .map(|&v| v.round() as i32)
                                .collect();
                            let luma_q3 = cfl::subsample_luma_q3(&luma, 8, 4, 4, true, true);
                            let avg_l = cfl::cfl_avg_l(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                4,
                                4,
                                true,
                                true,
                                bd,
                                (width, height),
                            );
                            let ac: Vec<i32> = luma_q3.iter().map(|&q| q - avg_l).collect();
                            let to_i = |v: &[f32]| -> Vec<i32> {
                                v.iter().map(|&x| x.round() as i32).collect()
                            };
                            if let Some(c) =
                                cfl::cfl_candidate(&to_i(&su4), &to_i(&sv4), &ac, dcu, dcv, bd)
                            {
                                let alpha_bits = 2.0
                                    + 3.0
                                    + if c.sign_u != 0 { 3.0 } else { 0.0 }
                                    + if c.sign_v != 0 { 3.0 } else { 0.0 };
                                if !best_j.is_finite() {
                                    best_j = j([&pred_uv.0, &pred_uv.1], 0.0, 0);
                                }
                                // CfL is uv_mode UV_CFL: DC_PRED's tx type (DCT_DCT).
                                if j([&c.pred_u, &c.pred_v], alpha_bits, 0) < best_j {
                                    enc.cfl_use = true;
                                    enc.cfl_js = c.js;
                                    enc.cfl_mag_u = c.mag_u;
                                    enc.cfl_mag_v = c.mag_v;
                                    enc.cfl_ctx_u = c.ctx_u;
                                    enc.cfl_ctx_v = c.ctx_v;
                                    enc.uv_mode = 0;
                                    tx_i = 0;
                                    pred_uv.0.copy_from_slice(&c.pred_u);
                                    pred_uv.1.copy_from_slice(&c.pred_v);
                                }
                            }
                        }
                        if mode8 >= 5 {
                            leaf_midx = coder::encode_luma_leaf_8x8_dir(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                (mode8, adelta8 as i8),
                                true,
                                pc,
                                3148,
                                Some((&coder::TXTP_EXT8, tx8, 6)),
                                (lmidx, amidx),
                            ) as i32;
                        } else {
                            encode_luma_leaf_8x8(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                mode8,
                                true,
                                pc,
                                3148,
                                Some((&coder::TXTP_EXT8, tx8, 6)),
                            );
                            leaf_midx = midx_class(mode8);
                        }
                        // Other 4:2:0 leaves emit `enc.uv_mode` as is and expect DC.
                        enc.uv_mode = 0;
                        let mut levu = Vec::new();
                        let recu4 = code4(&su4, &pred_uv.0, 0, tx_i, &mut levu);
                        put_block_rect(recu, pcw, cy, cx, 4, 4, &recu4);
                        let uc = levels_to_coeffs(&levu);
                        let u_ctx = (6 + ua + ul) as usize;
                        let u_skip = cdfs_qctx::SKIP_TX4_QC[enc.qc][u_ctx] as u32;
                        encode_chroma_tu4_scan(enc, &uc, u_skip, false, &SCAN4X4_LOSSY, u_ctx);
                        let u_nz = uc.iter().any(|&(_, l)| l != 0);
                        let mut levv = Vec::new();
                        let recv4 = code4(&sv4, &pred_uv.1, 4, tx_i, &mut levv);
                        put_block_rect(recv, pcw, cy, cx, 4, 4, &recv4);
                        let vc = levels_to_coeffs(&levv);
                        let v_ctx = (6 * (u_nz as i32) + va + vl) as usize;
                        let v_skip = v_ctx as u32;
                        encode_chroma_tu4_scan(enc, &vc, v_skip, true, &SCAN4X4_LOSSY, v_ctx);
                        (u_nz, vc.iter().any(|&(_, l)| l != 0))
                    }),
                    (2, 4) => outline_leaf_420(|| {
                        // residue-2 width × residue-4 height corner: 8×16 luma
                        // (TX_8X16) + 4×8 chroma per plane (4:2:0).
                        let bd = self.bit_depth as i32;
                        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
                        let lev = bases.c8x16.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 16, pred),
                                bases.c8x16.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &tables::SCAN8X16,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            8,
                            16,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &tables::SCAN8X16,
                                8,
                                16,
                                bd,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            2,
                            4,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 4,
                                height: 8,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c4x8,
                                scan: &tables::SCAN4X8,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        coder::encode_luma_leaf_rect128(
                            enc,
                            &tu,
                            &LumaLeafRect128Spec {
                                skip_cdf: skip,
                                dc_sign_ctx: dcs,
                                mode_idx: 0,
                                has_chroma: true,
                                width_mi: 2,
                                height_mi: 4,
                                part_cdf: pc,
                                tx_part_cdf: 12348,
                                scan: &tables::SCAN8X16,
                                tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
                            },
                        );
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 4,
                                ch: 8,
                                basis: &bases.c4x8,
                                scan: &tables::SCAN4X8,
                                eob_cdf: EobCdf::ChrEob32,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 32,
                                u_skip_row: &SKIP_TX8_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    (4, 2) => outline_leaf_420(|| {
                        // residue-4 width × residue-2 height corner: 16×8 luma
                        // (TX_16X8) + 8×4 chroma per plane (4:2:0).
                        let bd = self.bit_depth as i32;
                        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
                        let lev = bases.c16x8.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 8, pred),
                                bases.c16x8.qstep as f32 / sb_qstep as f32,
                            ),
                            0.0,
                            &tables::SCAN16X8,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            16,
                            8,
                            &itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                sb_qstep,
                                &tables::SCAN16X8,
                                16,
                                8,
                                bd,
                            ),
                        );
                        let tu = levels_to_coeffs(&lev);
                        let (skip, dcs) = sb_tu_contexts_rect(
                            &tu,
                            above,
                            left,
                            &TxbContextSpec {
                                sb_y,
                                sb_x,
                                qc,
                                mi_cols: tmc,
                                mi_rows: tmr,
                                block_eq_tx: true,
                            },
                            4,
                            2,
                        );
                        let mh_choice = chroma422::mhccp_decide_leaf(
                            enc,
                            decide_mode,
                            &chroma422::ChromaLeafPlanes {
                                reconstructed_luma: &*recy,
                                reconstructed_u: &*recu,
                                reconstructed_v: &*recv,
                                source_u: up,
                                source_v: vp,
                                luma_stride: pw,
                                chroma_stride: pcw,
                            },
                            &chroma422::ChromaLeafGeometry {
                                bounds: mhccp_bounds,
                                luma_y: sb_y,
                                luma_x: sb_x,
                                chroma_y: cy,
                                chroma_x: cx,
                                width: 8,
                                height: 4,
                                subsample_x: true,
                                subsample_y: true,
                                have_top: lmr > 0,
                                have_left: lmc > 0,
                            },
                            &chroma422::ChromaLeafRd {
                                neutral,
                                basis: &bases.c8x4,
                                scan: &tables::SCAN8X4,
                                qstep: sb_qstep,
                                lambda: leaf::part_lambda(
                                    sb_qstep,
                                    self.subsampled_part_lambda_c(),
                                ),
                                bit_depth: self.bit_depth as i32,
                            },
                        );
                        coder::encode_luma_leaf_rect128(
                            enc,
                            &tu,
                            &LumaLeafRect128Spec {
                                skip_cdf: skip,
                                dc_sign_ctx: dcs,
                                mode_idx: 0,
                                has_chroma: true,
                                width_mi: 4,
                                height_mi: 2,
                                part_cdf: pc,
                                tx_part_cdf: 12348,
                                scan: &tables::SCAN16X8,
                                tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
                            },
                        );
                        code_422_chroma_tu(
                            enc,
                            ChromaPlanes {
                                rec_u: &mut *recu,
                                rec_v: &mut *recv,
                                src_u: up,
                                src_v: vp,
                                stride: pcw,
                                coded_width: mhccp_bounds.chroma_width,
                                coded_height: mhccp_bounds.chroma_height,
                            },
                            &ChromaTxSpec {
                                cw: 8,
                                ch: 4,
                                basis: &bases.c8x4,
                                scan: &tables::SCAN8X4,
                                eob_cdf: EobCdf::ChrEob32,
                                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                area: 32,
                                u_skip_row: &SKIP_TX8_QC[qc],
                            },
                            QuantCtx {
                                qc,
                                neutral,
                                qstep: sb_qstep,
                                rdoq_lambda: self.tune.chroma_rdoq_lambda,
                            },
                            ChromaNeighbors { ua, ul, va, vl },
                            &ChromaTuInput {
                                y: cy,
                                x: cx,
                                bit_depth: self.bit_depth as i32,
                                cfl: mh_choice.as_ref(),
                                mode_predictors: None,
                            },
                        )
                    }),
                    other => unreachable!("unsupported native 4:2:0 leaf {:?}", other),
                }
            };
            let cfl_used = enc.cfl_signaled as i32;
            for c in lmc..lmc + bw_mi {
                if !sdp {
                    u_above[c] = u_present as i32;
                    v_above[c] = v_present as i32;
                    cfl_above[c] = cfl_used;
                }
                midx_above[c] = leaf_midx;
            }
            for r in lmr..lmr + bh_mi {
                if !sdp {
                    u_left[r] = u_present as i32;
                    v_left[r] = v_present as i32;
                    cfl_left[r] = cfl_used;
                }
                midx_left[r] = leaf_midx;
            }
            if sdp {
                let m = if leaf_midx >= 0 {
                    leaf_midx as u8
                } else {
                    0xff
                };
                for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                    sdp_midx[r * 16 + (lmc & 15)..r * 16 + ((lmc & 15) + bw_mi).min(16)].fill(m);
                }
            }
            // Mark this leaf's luma mi as coded for the next leaf's MHCCP.
            for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                for c in (lmc & 15)..((lmc & 15) + bw_mi).min(16) {
                    enc.sb_coded[r * 16 + c] = 1;
                }
            }
        }
        enc.sdp_luma = false;
        if sdp {
            let luma_leaves = tx_leaves[leaf_start..].to_vec();
            self.code_chroma_tree_420(&mut super::sdp::ChromaTree420 {
                enc: &mut *enc,
                decide_mode: &mut *decide_mode,
                recy: &*recy,
                recu: &mut *recu,
                recv: &mut *recv,
                up,
                vp,
                pw,
                pcw,
                width,
                height,
                u_above: &mut *u_above,
                v_above: &mut *v_above,
                u_left: &mut *u_left,
                v_left: &mut *v_left,
                cfl_above: &mut *cfl_above,
                cfl_left: &mut *cfl_left,
                luma_leaves: &luma_leaves,
                chroma_leaves: &mut *chroma_leaves,
                luma_midx: &sdp_midx,
                row,
                col,
                tmr,
                tmc,
                qc,
                neutral,
                sb_qstep,
                mhccp_bounds,
                ss: (1, 1),
                trial: false,
            });
        }
        tx_leaves.extend(tu_splits);
        // Staged decouple: log this SB's captured walk in raster order so the
        // serial Replay consumes the exact sequence a serial Capture produced.
        if let DecideMode::Capture(rec) = &mut decide_mode {
            rec.push_sb420(if sb_walk_ok {
                Sb420::Walk(leaf_recs)
            } else {
                Sb420::Fallback
            });
        }
        // The SB's delta-Q is settled once its leaves are emitted; record the
        // running qindex for the deblock (it carries forward across skipped SBs,
        // matching the decoder's `ts.last_qidx`).
        sb_qidx[row * sb_cols + col] = if use_grid {
            cell.qidx as u16
        } else {
            aqs.current_qidx() as u16
        };
        (skip_left, inter_left, newmv_left, mv_left, ref_left)
    }
}

#[cfg(test)]
mod tests {
    use super::{InterPredScratch420, drl0_mv, inter_mode_qtr_mvd};
    use crate::av2::video::mv::Mv;

    #[test]
    fn drl0_prefers_left_then_above_then_above_right() {
        let left = Mv { row: 8, col: -4 };
        let above = Mv { row: 16, col: 12 };
        let above_right = Mv { row: -8, col: 20 };
        assert_eq!(drl0_mv(Some(left), Some(above), Some(above_right)), left);
        assert_eq!(drl0_mv(None, Some(above), Some(above_right)), above);
        assert_eq!(drl0_mv(None, None, Some(above_right)), above_right);
        assert_eq!(drl0_mv(None, None, None), Mv::ZERO);
    }

    #[test]
    fn newmv_codes_delta_from_spatial_predictor() {
        let predictor = Mv { row: 16, col: -4 };
        let actual = Mv { row: 24, col: -8 };
        // Eighth-pel delta (8,-4) becomes quarter-pel MVD (4,-2), not
        // the old absolute-MV payload (12,-4).
        assert_eq!(inter_mode_qtr_mvd(actual, predictor), (2, 4, -2));
    }

    #[test]
    fn predictor_match_uses_nearmv_without_mvd() {
        let mv = Mv { row: -12, col: 6 };
        assert_eq!(inter_mode_qtr_mvd(mv, mv), (0, 0, 0));
    }

    #[test]
    fn inter_prediction_scratch_retains_largest_leaf_allocation() {
        let mut scratch = InterPredScratch420::default();
        let whole = scratch.whole_sb();
        assert_eq!(
            (
                whole.y.len(),
                whole.u.len(),
                whole.v.len(),
                whole.tx_pred.len(),
                whole.residual.len(),
                whole.chroma_pred.len(),
            ),
            (4096, 1024, 1024, 1024, 1024, 1024)
        );
        assert!(
            whole
                .luma_coeffs
                .iter()
                .chain(whole.chroma_coeffs.iter())
                .all(|coeffs| coeffs.capacity() >= 1024)
        );
        let capacities = (
            scratch.y.capacity(),
            scratch.u.capacity(),
            scratch.v.capacity(),
            scratch.tx_pred.capacity(),
            scratch.residual.capacity(),
            scratch.chroma_pred.capacity(),
            scratch.luma_coeffs.each_ref().map(|v| v.capacity()),
            scratch.chroma_coeffs.each_ref().map(|v| v.capacity()),
        );

        let (y, u, v, convolve_tmp) = scratch.planes(16 * 16, 8 * 8);
        assert_eq!((y.len(), u.len(), v.len()), (256, 64, 64));
        assert!(convolve_tmp.is_empty());
        assert_eq!(
            capacities,
            (
                scratch.y.capacity(),
                scratch.u.capacity(),
                scratch.v.capacity(),
                scratch.tx_pred.capacity(),
                scratch.residual.capacity(),
                scratch.chroma_pred.capacity(),
                scratch.luma_coeffs.each_ref().map(|v| v.capacity()),
                scratch.chroma_coeffs.each_ref().map(|v| v.capacity()),
            )
        );
    }
}
