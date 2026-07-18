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
}

/// Per-worker reusable scratch for the 4:2:0 wavefront decide: a private full-res
/// luma recon (`ry`, `pw×ph`) + two QUARTERED chroma recons (`ru`/`rv`,
/// `pcw×pch`, half-width AND half-height) plus the emit-side neighbour context
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
    /// entropy-neighbour state; transform and predictor mechanics stay here.
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
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        )
        .with_dark_aq(self.tune.dark_aq);
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
            let enc_mhccp = enc.mhccp;
            let enc_dqp = enc.delta_q_present;
            let adaptive = self.tune.updating_cdf && self.base_q_idx != 0;
            let base_q = self.base_q_idx as i32;
            let nthreads = Self::resolve_threads(self.threads);
            let wy = helpers::PlaneWriter::new(recy, pw);
            let wu = helpers::PlaneWriter::new(recu, pcw);
            let wv = helpers::PlaneWriter::new(recv, pcw);
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
            for col in 0..sb_cols {
                if fresh_ctx {
                    // Reproduce the frame-initial (context-free) state a wavefront worker
                    // sees before each SB decide. The 4:2:0 inter-state grids stay all-zero
                    // on the stills path this check targets, so only the entropy / chroma /
                    // CfL neighbour arrays need resetting here.
                    above.iter_mut().for_each(|v| *v = 0x40);
                    left.iter_mut().for_each(|v| *v = 0x40);
                    above_pctx.iter_mut().for_each(|v| *v = 0);
                    left_pctx.iter_mut().for_each(|v| *v = 0);
                    u_above.iter_mut().for_each(|v| *v = 0);
                    v_above.iter_mut().for_each(|v| *v = 0);
                    u_left.iter_mut().for_each(|v| *v = 0);
                    v_left.iter_mut().for_each(|v| *v = 0);
                    cfl_above.iter_mut().for_each(|v| *v = 0);
                    cfl_left.iter_mut().for_each(|v| *v = 0);
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
    /// walk loop). Behaviour-preserving; byte-identical. Split out so the
    /// SB-wavefront can drive it per cell with private recon + fresh contexts.
    /// The per-row left-neighbour scalars are threaded in by value and returned.
    fn decide_sb_420(&self, decision: Sb420Decision<'_, '_>) -> (u8, u8, u8, Option<Mv>, u8) {
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
        } = decision;
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
        // The general subsampled 32x32 leaf coefficient path is currently
        // AVM-safe only through qidx 38. Lower qualities retain the proven
        // geometry partition walker until its entropy contexts are completed.
        let luma_partition = if full_interior && self.tune.chroma_split && self.base_q_idx <= 38 {
            choose_luma_64x64_partition(
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
                        basis: &bases.luma,
                        qstep: sb_qstep,
                        scan: &tables::SCAN,
                        neutral,
                        quant_context: qc,
                        rdoq_lambda,
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
                    allow_8x8: false,
                },
            )
        } else {
            LumaPartitionDecision::default()
        };
        let ops = if full_interior && luma_partition.split64 {
            partition::sb_rd_split_ops(
                row,
                col,
                luma_partition.split32,
                luma_partition.split16,
                above_pctx,
                left_pctx,
            )
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
            let (u_present, v_present) = match (bw_mi, bh_mi) {
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
                    let (tus, mode_idx): ([Vec<Coeff>; 4], usize) = if let Some(d) = leaf_replay {
                        put_block_rect(recy, pw, sb_y, sb_x, 64, 64, &d.recon_y);
                        (std::array::from_fn(|k| d.tus[k].clone()), d.mode_idx)
                    } else {
                        let (tus, mode_idx, _, _) = encode_luma_sb(
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
                                basis: &bases.luma,
                                qstep: sb_qstep,
                                scan: &tables::SCAN,
                                neutral,
                                quant_context: qc,
                                rdoq_lambda: self.tune.rdoq_lambda,
                                speed: self.speed,
                                bit_depth: self.bit_depth as i32,
                            },
                            &LumaSbSearch {
                                residual_scale: sb_scale,
                                allow_directional: false,
                            },
                        );
                        (tus, mode_idx)
                    };
                    let (skip_cdfs, dc_sign_ctxs) =
                        sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
                    let cfl_choice = if let Some(d) = leaf_replay {
                        d.chroma.clone()
                    } else if enc.cfl && PARTITION_CFL {
                        let avg_l = cfl::cfl_avg_l(recy, pw, sb_y, sb_x, 32, 32, true, true, bd);
                        let mut suf = [0f32; 32 * 32];
                        let mut svf = [0f32; 32 * 32];
                        cfl_partition_prediction::<32>(pcw, up, vp, cy, cx, &mut suf, &mut svf);
                        let dc_u_f = dc_pred_rect(recu, pcw, cy, cx, 32, 32, neutral, bd);
                        let dc_v_f = dc_pred_rect(recv, pcw, cy, cx, 32, 32, neutral, bd);
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
                                lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                            tus: tus.to_vec(),
                            mode_idx,
                            tx_idx: 0,
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
                    encode_luma_block_split(
                        enc,
                        &tus,
                        &skip_cdfs,
                        &dc_sign_ctxs,
                        mode_idx,
                        true,
                        pc,
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
                    let (tus2, mode_idx) = encode_luma_leaf32(
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
                            basis: &bases.luma,
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let (skip2, dcs2) =
                        sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                    let mh_choice = chroma422::mhccp_decide_leaf(
                        enc,
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, true, pc);
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
                    let (tus2, mode_idx) = encode_luma_leaf_v32x64(
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
                            basis: &bases.luma,
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, true, pc);
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
                    let (tu, mode_idx) = encode_luma_leaf_s32x32(
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
                            basis: &bases.luma,
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                    );
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_32x32(enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc);
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 16, 64, neutral, self.bit_depth as i32);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 64, 16, neutral, self.bit_depth as i32);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 8, 32, neutral, self.bit_depth as i32);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 32, 8, neutral, self.bit_depth as i32);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 16, 32, neutral, self.bit_depth as i32);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 32, 16, neutral, self.bit_depth as i32);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                    // Bottom-right 16×16 corner leaf (residue 4 in both dims):
                    // full-AC TX_16X16 luma with 4-way ADST RD (DCT_DCT /
                    // ADST_ADST / ADST_DCT / DCT_ADST, DC mode) → 4:2:0 chroma
                    // 8×8 (TX_8X8, SCAN8X8, eob class 64, skip txs_ctx 1).
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 16, 16, neutral, self.bit_depth as i32);
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
                    let lambda = leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    encode_luma_leaf_16x16_full(enc, &tu, skip, dcs, 0, true, pc, 11074, tx_idx);
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
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 8, 8, neutral, self.bit_depth as i32);
                    let lev = bases.c8x8.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 8, 8, pred),
                            bases.c8x8.qstep as f32 / sb_qstep as f32,
                        ),
                        0.0,
                        &SCAN8X8,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        8,
                        8,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            sb_qstep,
                            &SCAN8X8,
                            8,
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
                        2,
                        2,
                    );
                    encode_luma_leaf_8x8(
                        enc,
                        &tu,
                        skip,
                        dcs,
                        0,
                        true,
                        pc,
                        3148,
                        Some((&coder::TXTP_EXT8, 0, 6)),
                    );
                    use crate::av2::coder::{
                        SCAN4X4_LOSSY, SCAN4X4_LOSSY_PACKED, encode_chroma_tu4_scan,
                    };
                    let bd = self.bit_depth as i32;
                    let predu = dc_pred_rect(recu, pcw, cy, cx, 4, 4, neutral, bd);
                    let levu = bases.c4x4.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(up, pcw, cy, cx, 4, 4, predu),
                            bases.c4x4.qstep as f32 / sb_qstep as f32,
                        ),
                        0.0,
                        &SCAN4X4_LOSSY_PACKED,
                    );
                    put_block_rect(
                        recu,
                        pcw,
                        cy,
                        cx,
                        4,
                        4,
                        &itx422::reconstruct_chroma(
                            predu,
                            &levu,
                            sb_qstep,
                            &SCAN4X4_LOSSY_PACKED,
                            4,
                            4,
                            bd,
                        ),
                    );
                    let uc = levels_to_coeffs(&levu);
                    let u_ctx = (6 + ua + ul) as usize;
                    let u_skip = cdfs_qctx::SKIP_TX4_QC[enc.qc][u_ctx] as u32;
                    encode_chroma_tu4_scan(enc, &uc, u_skip, false, &SCAN4X4_LOSSY, u_ctx);
                    let u_nz = uc.iter().any(|&(_, l)| l != 0);
                    let predv = dc_pred_rect(recv, pcw, cy, cx, 4, 4, neutral, bd);
                    let levv = bases.c4x4.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(vp, pcw, cy, cx, 4, 4, predv),
                            bases.c4x4.qstep as f32 / sb_qstep as f32,
                        ),
                        0.0,
                        &SCAN4X4_LOSSY_PACKED,
                    );
                    put_block_rect(
                        recv,
                        pcw,
                        cy,
                        cx,
                        4,
                        4,
                        &itx422::reconstruct_chroma(
                            predv,
                            &levv,
                            sb_qstep,
                            &SCAN4X4_LOSSY_PACKED,
                            4,
                            4,
                            bd,
                        ),
                    );
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
                            lambda: leaf::part_lambda(sb_qstep, self.tune.part_lambda_c),
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
            };
            let cfl_used = enc.cfl_signaled as i32;
            for c in lmc..lmc + bw_mi {
                u_above[c] = u_present as i32;
                v_above[c] = v_present as i32;
                cfl_above[c] = cfl_used;
            }
            for r in lmr..lmr + bh_mi {
                u_left[r] = u_present as i32;
                v_left[r] = v_present as i32;
                cfl_left[r] = cfl_used;
            }
            // Mark this leaf's luma mi as coded for the next leaf's MHCCP.
            for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                for c in (lmc & 15)..((lmc & 15) + bw_mi).min(16) {
                    enc.sb_coded[r * 16 + c] = 1;
                }
            }
        }
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
