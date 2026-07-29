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
use crate::aq_common::f_fmlaf;

/// Sum the encoded DC-sign marker bits for one coefficient-neighbor span.
/// These spans contain at most eight entries, so a compact ordered scalar
/// reduction is cheaper than routing through the wide pixel SIMD table.
#[inline]
fn sum_coef_sign(values: &[u8]) -> i32 {
    values.iter().map(|&value| (value >> 6) as i32).sum()
}

fn partition_model_sse(blocks: &[(usize, usize, usize, usize)], cells: &[(i64, i64); 16]) -> i64 {
    partition_model_sse_cells(blocks, cells, 4)
}

/// `blocks` rects are in PIXELS; `cell_px` is the side of one moment cell
/// (4 for the 16x16 grid, 8 for the 32x32 grid). Keeping the pixel count and
/// the cell indexing derived from the same rect is what makes the variance
/// term correct — deriving `n` from cell-unit dims understates it by
/// `cell_px^2` and inflates every SSE.
fn partition_model_sse_cells(
    blocks: &[(usize, usize, usize, usize)],
    cells: &[(i64, i64); 16],
    cell_px: usize,
) -> i64 {
    let mut sse = 0i64;
    for &(ox, oy, w, h) in blocks {
        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        for cell_y in oy / cell_px..(oy + h) / cell_px {
            for cell_x in ox / cell_px..(ox + w) / cell_px {
                let moments = cells[cell_y * 4 + cell_x];
                sum += moments.0;
                sum_sq += moments.1;
            }
        }
        let n = (w * h) as i64;
        sse += sum_sq - (sum * sum + n / 2) / n;
    }
    sse.max(0)
}

/// Exact sum of an integer source-plane rectangle.
#[inline]
fn block_sum_i32<P: Pel>(
    plane: &[P],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) -> i64 {
    let mut sum = 0i64;
    for y in 0..h {
        sum += plane[(py + y) * stride + px..][..w]
            .iter()
            .map(|&sample| i64::from(sample.widen()))
            .sum::<i64>();
    }
    sum
}

#[inline]
pub(crate) fn block_moments_i32<P: Pel>(
    plane: &[P],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) -> (i64, i64) {
    let mut sum = 0i64;
    let mut sum_sq = 0i64;
    for y in 0..h {
        for &sample in &plane[(py + y) * stride + px..][..w] {
            let sample = i64::from(sample.widen());
            sum += sample;
            sum_sq += sample * sample;
        }
    }
    (sum, sum_sq)
}

fn block_moment_grid_16x16<P: Pel>(
    plane: &[P],
    stride: usize,
    px: usize,
    py: usize,
) -> [(i64, i64); 16] {
    let mut cells = [(0i64, 0i64); 16];
    for y in 0..16 {
        let cell_row = (y >> 2) * 4;
        let row = &plane[(py + y) * stride + px..][..16];
        for (cell_x, samples) in row.as_chunks::<4>().0.iter().enumerate() {
            let cell = &mut cells[cell_row + cell_x];
            for &sample in samples {
                let sample = i64::from(sample.widen());
                cell.0 += sample;
                cell.1 += sample * sample;
            }
        }
    }
    cells
}

fn clamp_aq_delta_relative(base_q: i32, delta: i32, bd: u8) -> i32 {
    let r: f32 = crate::tuning::get().aq_r;
    let r = if delta > 0 {
        r
    } else {
        r.max(1.05)
    };
    if delta == 0 {
        return delta;
    }
    let base_step = ac_q(base_q.clamp(1, 255) as u8, bd) as f32;
    let (lo, hi) = (base_step / r, base_step * r);
    let mut d = delta;
    while d != 0 {
        let s = ac_q((base_q + d).clamp(1, 255) as u8, bd) as f32;
        if s >= lo && s <= hi {
            break;
        }
        d -= d.signum();
    }
    d
}

fn ramped_tilt(base: f32, qindex: u8, extra: f32) -> f32 {
    let t = ((qindex as f32 - 50.0) / 50.0).clamp(0.0, 1.0);
    base * (1.0 + extra * t) * f_fmlaf(-top_ease(), top_ease_t(qindex), 1.0)
}

fn local_ref_blend() -> f32 {
    crate::tuning::get().local_ref_blend
}

fn none16_top_bias_420() -> f32 {
    crate::tuning::get().none16_top_bias_420
}

fn none16_top_bias_422() -> f32 {
    crate::tuning::get().none16_top_bias_422
}

fn none16_top_bias_444() -> f32 {
    crate::tuning::get().none16_top_bias_444
}

const SEAM_W_420: f32 = 0.0;
const SEAM_W_422: f32 = 60.0;

fn seam_t(base_q: u8, w: f32) -> f32 {
    if w <= 0.0 {
        return if base_q <= 55 { 1.0 } else { 0.0 };
    }
    ((55.0 + w - base_q as f32) / w).clamp(0.0, 1.0)
}

fn rect16_bias() -> f32 {
    crate::tuning::get().rect16_bias
}

const RECT16_VERT_ENABLED: bool = true;

pub(crate) const fn joint_luma_uv_enabled() -> bool {
    false
}

const fn joint_luma_uv_large_enabled() -> bool {
    false
}

const fn joint_luma_uv_proxy_enabled() -> bool {
    false
}

const fn local_chroma_weight_enabled() -> bool {
    true
}

const fn exact_uv_rate_enabled() -> bool {
    true
}

/// Mild tax on under-tooled four-strip partition proxies.
fn quad4_bias() -> f32 {
    crate::tuning::get().quad4_bias
}

/// PARTITION_HORZ_4/VERT_4 search enable.
const fn quad4_enabled(ss420: bool) -> bool {
    ss420
}

impl<'a> LossyTile<'a> {
    /// Frozen decision CDFs keep RDO independent of superblock coding order.
    #[inline]
    fn dcdf(&self) -> &Cdfs {
        &self.dec_cdfs
    }

    fn new(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<u16>; 3], qm: QmLevels) -> Self {
        LossyTile {
            sb_act_cache: std::cell::Cell::new((u32::MAX, u32::MAX, 0.0)),
            dct: DctDispatch::scalar(),
            idct: IdctDispatch::scalar(),
            intrapred: IntraPredDispatch::scalar(),
            kmeans: KmeansDispatch::scalar(),
            rd: crate::rd_sse::RdDispatch::scalar(),
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma_with_delta_qm(
                q,
                chroma_dc_delta(q, 0),
                chroma_ac_delta(q, 0),
                bd,
                qm.u,
            ),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: false,
            chroma_part_weight: chroma_part_rd_weight(false, false, src, bd),
            allow_intrabc: false,
            screen_content: true,
            ibc_mv: vec![None; (w / 4) * (h / 4)],
            src,
            recon: [vec![0; w * h], vec![0; w * h], vec![0; w * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_tx: vec![-1i8; w / 4],
            l_tx: vec![-1i8; h / 4],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: vec![0; w / 4],
            l_uv_mode: vec![0; h / 4],
            a_palette: vec![Vec::new(); w / 4],
            a_palette_uv: vec![Vec::new(); w / 4],
            l_palette: vec![Vec::new(); h / 4],
            l_palette_uv: vec![Vec::new(); h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            pblk4: vec![0; (w / 4) * (h / 4)],
            pblk4h: vec![0; (w / 4) * (h / 4)],
            pblk4v: vec![false; (w / 4) * (h / 4)],
            pblk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            pal_est_cache: std::cell::RefCell::new(HashMap::new()),
            chroma_rd_cache: std::cell::RefCell::new(HashMap::new()),
            rd16_cache: std::cell::RefCell::new(HashMap::new()),
            rect_leaf_cache: std::cell::RefCell::new(HashMap::new()),
            scratch: Default::default(),
            split4_rd_cache: std::cell::RefCell::new(HashMap::new()),
            emit_epoch: std::cell::Cell::new(0),
            ibc_index: std::cell::RefCell::new(None),
            ibc_shared: None,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)).with_band_tilt(ramped_tilt(1.2, q, 1.4)),
            updating_cdf: true,
            dec_cdfs: {
                let mut c = Cdfs::decision_snapshot(crate::coef_q::qcat(q));
                c.band_tilt = ramped_tilt(1.2, q, 1.4);
                c
            },
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    fn new_mono(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<u16>; 3], qm: QmLevels) -> Self {
        LossyTile {
            sb_act_cache: std::cell::Cell::new((u32::MAX, u32::MAX, 0.0)),
            dct: DctDispatch::scalar(),
            idct: IdctDispatch::scalar(),
            intrapred: IntraPredDispatch::scalar(),
            kmeans: KmeansDispatch::scalar(),
            rd: crate::rd_sse::RdDispatch::scalar(),
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: true,
            chroma_part_weight: 0.0,
            allow_intrabc: false,
            screen_content: true,
            ibc_mv: vec![None; (w / 4) * (h / 4)],
            src,
            recon: [vec![0; w * h], Vec::new(), Vec::new()],
            a_coef: [vec![0x40; w / 4], Vec::new(), Vec::new()],
            l_coef: [vec![0x40; h / 4], Vec::new(), Vec::new()],
            a_tx: vec![-1i8; w / 4],
            l_tx: vec![-1i8; h / 4],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: Vec::new(),
            l_uv_mode: Vec::new(),
            a_palette: vec![Vec::new(); w / 4],
            a_palette_uv: vec![Vec::new(); w / 4],
            l_palette: vec![Vec::new(); h / 4],
            l_palette_uv: vec![Vec::new(); h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            pblk4: vec![0; (w / 4) * (h / 4)],
            pblk4h: vec![0; (w / 4) * (h / 4)],
            pblk4v: vec![false; (w / 4) * (h / 4)],
            pblk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            pal_est_cache: std::cell::RefCell::new(HashMap::new()),
            chroma_rd_cache: std::cell::RefCell::new(HashMap::new()),
            rd16_cache: std::cell::RefCell::new(HashMap::new()),
            rect_leaf_cache: std::cell::RefCell::new(HashMap::new()),
            scratch: Default::default(),
            split4_rd_cache: std::cell::RefCell::new(HashMap::new()),
            emit_epoch: std::cell::Cell::new(0),
            ibc_index: std::cell::RefCell::new(None),
            ibc_shared: None,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)).with_band_tilt(ramped_tilt(1.2, q, 0.4)),
            updating_cdf: true,
            dec_cdfs: {
                let mut c = Cdfs::decision_snapshot(crate::coef_q::qcat(q));
                c.band_tilt = ramped_tilt(1.2, q, 0.4);
                c
            },
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    /// 4:2:2 tile: luma is full w x h, chroma planes are subsampled to (w/2) x h.
    /// `src[1]`/`src[2]` must already be the half-width chroma planes.
    fn new_422(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<u16>; 3], qm: QmLevels) -> Self {
        let cw = w / 2;
        LossyTile {
            sb_act_cache: std::cell::Cell::new((u32::MAX, u32::MAX, 0.0)),
            dct: DctDispatch::scalar(),
            idct: IdctDispatch::scalar(),
            intrapred: IntraPredDispatch::scalar(),
            kmeans: KmeansDispatch::scalar(),
            rd: crate::rd_sse::RdDispatch::scalar(),
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma_with_delta_qm(
                q,
                chroma_dc_delta(q, 1),
                chroma_ac_delta(q, 1),
                bd,
                qm.u,
            ),
            w,
            h,
            cw,
            ss422: true,
            ss420: false,
            mono: false,
            chroma_part_weight: chroma_part_rd_weight(false, true, src, bd),
            allow_intrabc: false,
            screen_content: true,
            ibc_mv: vec![None; (w / 4) * (h / 4)],
            src,
            recon: [vec![0; w * h], vec![0; cw * h], vec![0; cw * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_tx: vec![-1i8; w / 4],
            l_tx: vec![-1i8; h / 4],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: vec![0; w / 4],
            l_uv_mode: vec![0; h / 4],
            a_palette: vec![Vec::new(); w / 4],
            a_palette_uv: vec![Vec::new(); w / 4],
            l_palette: vec![Vec::new(); h / 4],
            l_palette_uv: vec![Vec::new(); h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            pblk4: vec![0; (w / 4) * (h / 4)],
            pblk4h: vec![0; (w / 4) * (h / 4)],
            pblk4v: vec![false; (w / 4) * (h / 4)],
            pblk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            pal_est_cache: std::cell::RefCell::new(HashMap::new()),
            chroma_rd_cache: std::cell::RefCell::new(HashMap::new()),
            rd16_cache: std::cell::RefCell::new(HashMap::new()),
            rect_leaf_cache: std::cell::RefCell::new(HashMap::new()),
            scratch: Default::default(),
            split4_rd_cache: std::cell::RefCell::new(HashMap::new()),
            emit_epoch: std::cell::Cell::new(0),
            ibc_index: std::cell::RefCell::new(None),
            ibc_shared: None,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)).with_band_tilt(ramped_tilt(2.45, q, 0.4)),
            updating_cdf: true,
            dec_cdfs: {
                let mut c = Cdfs::decision_snapshot(crate::coef_q::qcat(q));
                c.band_tilt = ramped_tilt(2.45, q, 0.4);
                c
            },
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    /// 4:2:0 tile: luma is full w x h, chroma planes are subsampled to
    /// (w/2) x (h/2). `src[1]`/`src[2]` must already be the quarter-size planes.
    fn new_420(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<u16>; 3], qm: QmLevels) -> Self {
        let (cw, ch) = (w / 2, h / 2);
        LossyTile {
            sb_act_cache: std::cell::Cell::new((u32::MAX, u32::MAX, 0.0)),
            dct: DctDispatch::scalar(),
            idct: IdctDispatch::scalar(),
            intrapred: IntraPredDispatch::scalar(),
            kmeans: KmeansDispatch::scalar(),
            rd: crate::rd_sse::RdDispatch::scalar(),
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma_with_delta_qm(
                q,
                chroma_dc_delta(q, 2),
                chroma_ac_delta(q, 2),
                bd,
                qm.u,
            ),
            w,
            h,
            cw,
            ss422: false,
            ss420: true,
            mono: false,
            chroma_part_weight: chroma_part_rd_weight(true, false, src, bd),
            allow_intrabc: false,
            screen_content: true,
            ibc_mv: vec![None; (w / 4) * (h / 4)],
            src,
            recon: [vec![0; w * h], vec![0; cw * ch], vec![0; cw * ch]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; ch / 4], vec![0x40; ch / 4]],
            a_tx: vec![-1i8; w / 4],
            l_tx: vec![-1i8; h / 4],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: vec![0; w / 4],
            l_uv_mode: vec![0; h / 4],
            a_palette: vec![Vec::new(); w / 4],
            a_palette_uv: vec![Vec::new(); w / 4],
            l_palette: vec![Vec::new(); h / 4],
            l_palette_uv: vec![Vec::new(); h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            pblk4: vec![0; (w / 4) * (h / 4)],
            pblk4h: vec![0; (w / 4) * (h / 4)],
            pblk4v: vec![false; (w / 4) * (h / 4)],
            pblk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            pal_est_cache: std::cell::RefCell::new(HashMap::new()),
            chroma_rd_cache: std::cell::RefCell::new(HashMap::new()),
            rd16_cache: std::cell::RefCell::new(HashMap::new()),
            rect_leaf_cache: std::cell::RefCell::new(HashMap::new()),
            scratch: Default::default(),
            split4_rd_cache: std::cell::RefCell::new(HashMap::new()),
            emit_epoch: std::cell::Cell::new(0),
            ibc_index: std::cell::RefCell::new(None),
            ibc_shared: None,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)).with_band_tilt(ramped_tilt(3.5, q, 0.7)),
            updating_cdf: true,
            dec_cdfs: {
                let mut c = Cdfs::decision_snapshot(crate::coef_q::qcat(q));
                c.band_tilt = ramped_tilt(3.5, q, 0.7);
                c
            },
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    /// Entropy-accurate rate for a square LUMA transform of width `w` covering
    /// the whole block at (px, py) — the shape every intra R-D comparison comes
    /// in. Since the TX spans the block, `txb_skip` context is 0 (see
    /// [`Self::skip_ctx`]); TX_32X32 codes no transform-type symbol.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn luma_bits(
        &self,
        cf: &[i32],
        scan: &[u32],
        w: usize,
        px: usize,
        py: usize,
        y_mode: usize,
        txtp: usize,
    ) -> f32 {
        self.luma_bits_bounded(cf, scan, w, px, py, y_mode, txtp, f32::INFINITY)
    }

    /// [`Self::luma_bits`] with an exact abort bound (see
    /// [`crate::rate::real_block_bits_bounded`]): pass the highest bit count
    /// that could still win the caller's comparison; a return of infinity
    /// means "this candidate loses regardless".
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn luma_bits_bounded(
        &self,
        cf: &[i32],
        scan: &[u32],
        w: usize,
        px: usize,
        py: usize,
        y_mode: usize,
        txtp: usize,
        bound: f32,
    ) -> f32 {
        let (bx4, by4) = (px / 4, py / 4);
        // Whole-block contexts: txb_skip ctx 0 and the block-span DC-sign ctx.
        // Sub-transform trials must pass the PROGRESSIVE contexts instead —
        // see `luma_bits_ctx_bounded` (external review round 2, finding 3).
        let dcs = self.dc_sign_ctx_span(0, bx4, by4, w / 4, w / 4);
        self.luma_bits_ctx_bounded(cf, scan, w, px, py, y_mode, txtp, 0, dcs, bound)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn luma_bits_ctx_bounded(
        &self,
        cf: &[i32],
        scan: &[u32],
        w: usize,
        px: usize,
        py: usize,
        y_mode: usize,
        txtp: usize,
        skip_ctx: usize,
        dcs_ctx: usize,
        bound: f32,
    ) -> f32 {
        if use_proxy_rate(self.speed) {
            return block_rate_bits(cf, scan);
        }
        let c = self.dcdf();
        let _ = (px, py);
        let (cls, eob_bin, txtp_cdf): (usize, &[u16], Option<&[u16]>) = match w {
            4 => (0, &c.eob_bin_16_l, Some(&c.txtp4[y_mode])),
            8 => (1, &c.eob_bin_64_l, Some(&c.txtp[y_mode])),
            16 => (2, &c.eob_bin_256_l, Some(&c.txtp16[y_mode])),
            // TX_32X32 intra implies DCT_DCT — no txtp symbol is coded.
            _ => (3, &c.eob_bin_1024_l, None),
        };
        let ctx = crate::rate::RateCtx {
            cdfs: c,
            cls,
            plane: 0,
            w,
            h: w,
            eob_bin,
            skip_ctx,
            dcs_ctx,
            txtp: txtp_cdf.map(|cd| (cd, txtp)),
        };
        // Bound slack: the shipped abort is EXACT (bits only accumulate, so
        // aborting past `bound` cannot change a decision). Scaling below 1.0
        // makes it LOSSY.
        //
        // MEASURED INERT: slack 0.95 and 0.85 are BIT-IDENTICAL at Medium on
        // 4:4:4 and 4:2:0 (12 crops x 6 q) with no time change. Verified it is
        // a real null and not an unreached path — ~50% of calls carry a finite
        // bound (205,679 finite vs 194,321 infinite in one encode). Block bits
        // are simply either well under or well over the bound, so a 15% squeeze
        // almost never lands in the accept/reject window. Not a lever.
        let slack = crate::tuning::rate_bound_slack(self.speed);
        let bound = if slack < 1.0 { bound * slack } else { bound };
        crate::rate::real_block_bits_bounded(cf, scan, &ctx, bound)
    }

    /// The coefficient result context a coded transform block leaves in
    /// `a_coef` / `l_coef` — read-only twin of the encoders' return value
    /// (`cul.min(63) | dc_sign_bits`, or 0x40 when all-zero).
    fn coef_res_ctx(cf: &[i32], scan: &[u32]) -> u8 {
        let Some(eob) = scan.iter().rposition(|&rc| cf[rc as usize] != 0) else {
            return 0x40;
        };
        let cul: u32 = scan[..=eob]
            .iter()
            .map(|&rc| cf[rc as usize].unsigned_abs())
            .sum();
        let dc_sign_bits: u8 = if cf[0] == 0 {
            1 << 6
        } else if cf[0] < 0 {
            0
        } else {
            2 << 6
        };
        (cul.min(63) as u8) | dc_sign_bits
    }

    /// Entropy-accurate rate for a 2:1 rectangular luma transform. This is the
    /// read-only counterpart of `encode_{4x8,8x4,8x16,16x8,16x32,32x16}_luma_coeffs`:
    /// it uses the same square-up coefficient class, rectangular EOB CDF,
    /// column-major stride and shape-specific neighbor offsets. Transform-type
    /// syntax is part of this transaction whenever the emitter signals it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn luma_rect_bits(
        &self,
        cf: &[i32],
        scan: &[u32],
        w: usize,
        h: usize,
        px: usize,
        py: usize,
        y_mode: usize,
        txtp: usize,
    ) -> f32 {
        if use_proxy_rate(self.speed) {
            return block_rate_bits(cf, scan);
        }
        let c = self.dcdf();
        let (cls, eob_bin, txtp_cdf): (usize, &[u16], Option<&[u16]>) = match (w * h, w.max(h)) {
            (32, 8) => (1, &c.eob_bin_32_l, Some(&c.txtp4[y_mode])),
            // RTX_16X4 / RTX_4X16: min dim 4 -> txtp4 set, 64-coeff eob bins,
            // coefficient class 1 (matches encode_16x4/4x16_luma_coeffs).
            (64, 16) => (1, &c.eob_bin_64_l, Some(&c.txtp4[y_mode])),
            (128, 16) => (2, &c.eob_bin_128_l, Some(&c.txtp[y_mode])),
            // A rectangular transform whose square-up size is TX_32X32 is
            // DCT_DCT-only, exactly like its square counterpart.
            (512, 32) => (3, &c.eob_bin_512_l, None),
            _ => unreachable!("unsupported rectangular luma transform {w}x{h}"),
        };
        let ctx = crate::rate::RateCtx {
            cdfs: c,
            cls,
            plane: 0,
            w,
            h,
            eob_bin,
            skip_ctx: 0,
            dcs_ctx: self.dc_sign_ctx_span(0, px / 4, py / 4, w / 4, h / 4),
            txtp: txtp_cdf.map(|cd| (cd, txtp)),
        };
        crate::rate::real_block_bits(cf, scan, &ctx)
    }

    /// Exact-context RDOQ for a rectangular LUMA transform — the twin of
    /// [`Self::luma_rect_bits`], with the same size class, luma eob bins and
    /// DC-sign span. The rect leaves ran the shape-blind [`trellis_optimize`]
    /// while every square luma leaf has had exact-context RDOQ for a long
    /// time; this closes that asymmetry. The transform-type symbol is a
    /// constant across candidate LEVELS, so it does not enter the trellis.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn luma_rect_trellis(
        &self,
        cf: &mut [i32],
        tf: &[f32],
        dc_q: f32,
        ac_q: f32,
        scan: &[u32],
        lambda0: f32,
        w: usize,
        h: usize,
        px: usize,
        py: usize,
    ) {
        let c = self.dcdf();
        let (cls, eob_bin): (usize, &[u16]) = match (w * h, w.max(h)) {
            (32, 8) => (1, &c.eob_bin_32_l),
            (64, 16) => (1, &c.eob_bin_64_l),
            (128, 16) => (2, &c.eob_bin_128_l),
            (512, 32) => (3, &c.eob_bin_512_l),
            _ => unreachable!("unsupported rectangular luma transform {w}x{h}"),
        };
        trellis_optimize_ctx(
            cf,
            tf,
            dc_q,
            ac_q,
            scan,
            lambda0,
            w,
            h,
            c,
            cls,
            0,
            eob_bin,
            self.dc_sign_ctx_span(0, px / 4, py / 4, w / 4, h / 4),
            self.quant.qm_level(),
            self.quant.qidx() as i32,
        );
    }

    /// Entropy-accurate rate for a square CHROMA transform of width `w` at
    /// chroma position (cx, cy) of `plane` (1 or 2). Chroma codes no
    /// transform-type symbol, and its `txb_skip` context is the
    /// `7 + above_nz + left_nz` form the per-size helpers all share. Both chroma
    /// planes share CDF plane index 1.
    pub(crate) fn chroma_bits(
        &self,
        cf: &[i32],
        scan: &[u32],
        w: usize,
        plane: usize,
        cx: usize,
        cy: usize,
    ) -> f32 {
        self.chroma_rect_bits(cf, scan, w, w, plane, cx, cy)
    }

    /// Entropy-accurate chroma coefficient rate for square or rectangular
    /// transforms. Both chroma planes use coefficient-plane class 1, while
    /// neighbor state is read from the actual plane and full TX footprint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn chroma_rect_bits(
        &self,
        cf: &[i32],
        scan: &[u32],
        w: usize,
        h: usize,
        plane: usize,
        cx: usize,
        cy: usize,
    ) -> f32 {
        if use_proxy_rate(self.speed) {
            return block_rate_bits(cf, scan);
        }
        let c = self.dcdf();
        let (bx4, by4) = (cx / 4, cy / 4);
        let (nw, nh) = ((w / 4).max(1), (h / 4).max(1));
        let (cls, eob_bin): (usize, &[u16]) = match (w * h, w.max(h)) {
            (16, 4) => (0, &c.eob_bin_16_c),
            (32, 8) => (1, &c.eob_bin_32_c),
            (64, 8) => (1, &c.eob_bin_64_c),
            // RTX_16X4 / RTX_4X16: min dim 4 -> t_dim ctx 1, 64-coeff eob bins.
            (64, 16) => (1, &c.eob_bin_64_c),
            (128, 16) => (2, &c.eob_bin_128_c),
            (256, 16) => (2, &c.eob_bin_256_c),
            (512, 32) => (3, &c.eob_bin_512_c),
            (1024, 32) => (3, &c.eob_bin_1024_c),
            _ => unreachable!("unsupported chroma transform {w}x{h}"),
        };
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..(bx4 + nw).min(a.len())].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..(by4 + nh).min(l.len())].iter().any(|&x| x != 0x40) as usize;
        let ctx = crate::rate::RateCtx {
            cdfs: c,
            cls,
            plane: 1,
            w,
            h,
            eob_bin,
            skip_ctx: 7 + ca + cl,
            dcs_ctx: self.dc_sign_ctx_span(plane, bx4, by4, nw, nh),
            txtp: None,
        };
        crate::rate::real_block_bits(cf, scan, &ctx)
    }

    /// Exact-context RDOQ for a rectangular CHROMA transform. The shape-blind
    /// [`trellis_optimize`] prices every level against a flat model; this runs
    /// the real trellis against the SAME size class, chroma CDFs, eob bins and
    /// DC-sign context that [`Self::chroma_rect_bits`] charges and the emitter
    /// actually codes, so the levels it keeps are the ones the coefficient
    /// coder is cheap for. The size-class table is deliberately the same match
    /// as `chroma_rect_bits` — the two must never disagree.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn chroma_rect_trellis(
        &self,
        cf: &mut [i32],
        tf: &[f32],
        dc_q: f32,
        ac_q: f32,
        scan: &[u32],
        lambda0: f32,
        w: usize,
        h: usize,
        plane: usize,
        cx: usize,
        cy: usize,
    ) {
        if (!self.ss420 && !self.ss422) || (self.speed == Speed::Fast && self.ss420) {
            trellis_optimize(cf, tf, dc_q, ac_q, scan, lambda0);
            return;
        }
        let c = self.dcdf();
        let (bx4, by4) = (cx / 4, cy / 4);
        let (nw, nh) = ((w / 4).max(1), (h / 4).max(1));
        let (cls, eob_bin): (usize, &[u16]) = match (w * h, w.max(h)) {
            (16, 4) => (0, &c.eob_bin_16_c),
            (32, 8) => (1, &c.eob_bin_32_c),
            (64, 8) => (1, &c.eob_bin_64_c),
            (64, 16) => (1, &c.eob_bin_64_c),
            (128, 16) => (2, &c.eob_bin_128_c),
            (256, 16) => (2, &c.eob_bin_256_c),
            (512, 32) => (3, &c.eob_bin_512_c),
            (1024, 32) => (3, &c.eob_bin_1024_c),
            _ => unreachable!("unsupported chroma transform {w}x{h}"),
        };
        trellis_optimize_ctx(
            cf,
            tf,
            dc_q,
            ac_q,
            scan,
            lambda0,
            w,
            h,
            c,
            cls,
            1,
            eob_bin,
            self.dc_sign_ctx_span(plane, bx4, by4, nw, nh),
            self.cquant.qm_level(),
            self.cquant.qidx() as i32,
        );
    }

    /// Cost of signaling luma mode `m` at (px, py).
    /// Entropy-accurate rate for a TX_8X8 V_DCT / H_DCT block: read-only twin
    /// of `encode_tx8_coeffs_1d` (1-D class scans/contexts) against the
    /// decision CDF snapshot, including the txb_skip and tx-type symbols.
    pub(crate) fn luma_bits_1d_8x8(
        &self,
        cf: &[i32; 64],
        vertical: bool,
        px: usize,
        py: usize,
        y_mode: usize,
    ) -> f32 {
        let c = self.dcdf();
        let pos_rc = |i: usize| -> usize {
            let (x, y) = (i & 7, i >> 3);
            if vertical { (x << 3) | y } else { i }
        };
        let Some(eob) = (0..64).rev().find(|&i| cf[pos_rc(i)] != 0) else {
            return cdf_cost(&c.txb_skip[1][0], 1);
        };
        let mut bits = cdf_cost(&c.txb_skip[1][0], 0)
            + cdf_cost(&c.txtp[y_mode], if vertical { 2 } else { 3 });
        let dcs = self.dc_sign_ctx_span(0, px / 4, py / 4, 2, 2);
        let hi_bits = |m: u32, br: &[u16]| -> f32 {
            let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
            let mut b = 0f32;
            let mut coded = 0i32;
            for _ in 0..(COEFF_BASE_RANGE / 3) {
                let s = (total_br - coded).min(3);
                b += cdf_cost(br, s as usize);
                coded += s;
                if s < 3 {
                    break;
                }
            }
            if m >= 15 {
                b += golomb_cost(m - 15);
            }
            b
        };
        if eob == 0 {
            let dm = cf[0].unsigned_abs();
            bits += cdf_cost(&c.eob_bin_64_l1d, 0);
            bits += cdf_cost(&c.eob_base[1][0][0], (dm.min(3) - 1) as usize);
            if dm.min(3) == 3 {
                bits += hi_bits(dm, &c.br_tok[1][0][0]);
            } else if dm >= 15 {
                bits += golomb_cost(dm - 15);
            }
            bits += cdf_cost(&c.dc_sign[0][dcs], (cf[0] < 0) as usize);
            return bits;
        }
        let eob_bin = if eob < 2 {
            eob
        } else {
            32 - (eob as u32).leading_zeros() as usize
        };
        bits += cdf_cost(&c.eob_bin_64_l1d, eob_bin);
        if eob_bin > 1 {
            let nbits = eob_bin - 2;
            bits += cdf_cost(&c.eob_hi[1][0][eob_bin], (eob >> nbits) & 1) + nbits as f32;
        }
        let mut levels = [0u8; 16 * 10];
        let lvl = |lv: &[u8], x: usize, y: usize| -> u32 { lv[x * 16 + y] as u32 };
        let lo_ctx_1d = |lv: &[u8], x: usize, y: usize| -> (usize, u32) {
            let mut mag = lvl(lv, x, y + 1) + lvl(lv, x + 1, y) + lvl(lv, x, y + 2);
            let hi_mag = mag;
            mag += lvl(lv, x, y + 3) + lvl(lv, x, y + 4);
            let offset = 26 + if y > 1 { 10 } else { y * 5 };
            (
                offset
                    + if mag > 512 {
                        4
                    } else {
                        ((mag + 64) >> 7) as usize
                    },
                hi_mag,
            )
        };
        let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
        {
            let (x, y) = (eob & 7, eob >> 3);
            let m = cf[pos_rc(eob)].unsigned_abs();
            bits += cdf_cost(&c.eob_base[1][0][ctx_e], (m.min(3) - 1) as usize);
            if m.min(3) == 3 {
                let bc = if y != 0 { 14 } else { 7 };
                bits += hi_bits(m, &c.br_tok[1][0][bc]);
            } else if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            bits += 1.0; // sign
            levels[x * 16 + y] = level_byte(m);
        }
        for i in (1..eob).rev() {
            let (x, y) = (i & 7, i >> 3);
            let m = cf[pos_rc(i)].unsigned_abs();
            let (ctx, hi_mag) = lo_ctx_1d(&levels, x, y);
            bits += cdf_cost(&c.base_tok[1][0][ctx], m.min(3) as usize);
            if m.min(3) == 3 {
                let mag = hi_mag & 63;
                let bc = (if y != 0 { 14 } else { 7 })
                    + if mag > 12 {
                        6
                    } else {
                        ((mag + 1) >> 1) as usize
                    };
                bits += hi_bits(m, &c.br_tok[1][0][bc]);
            } else if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            if m != 0 {
                bits += 1.0; // sign
            }
            levels[x * 16 + y] = level_byte(m);
        }
        let (dc_ctx, dc_hi_mag) = lo_ctx_1d(&levels, 0, 0);
        let dm = cf[0].unsigned_abs();
        bits += cdf_cost(&c.base_tok[1][0][dc_ctx], dm.min(3) as usize);
        if dm.min(3) == 3 {
            let mag = dc_hi_mag & 63;
            let bc = if mag > 12 {
                6
            } else {
                ((mag + 1) >> 1) as usize
            };
            bits += hi_bits(dm, &c.br_tok[1][0][bc]);
        } else if dm >= 15 {
            bits += golomb_cost(dm - 15);
        }
        if dm != 0 {
            bits += cdf_cost(&c.dc_sign[0][dcs], (cf[0] < 0) as usize);
        }
        bits
    }

    /// Entropy-accurate rate for a V_DCT / H_DCT TX_4X4 block: read-only twin
    /// of `encode_tx4_coeffs_1d` (size-class 0 CDFs, `eob_bin_16_l1d`, the
    /// `txtp4` symbol, eob-coef ctx thresholds 2/4).
    pub(crate) fn luma_bits_1d_4x4(
        &self,
        cf: &[i32; 16],
        vertical: bool,
        px: usize,
        py: usize,
        y_mode: usize,
    ) -> f32 {
        let c = self.dcdf();
        let pos_rc = |i: usize| -> usize {
            let (x, y) = (i & 3, i >> 2);
            if vertical { (x << 2) | y } else { i }
        };
        let Some(eob) = (0..16).rev().find(|&i| cf[pos_rc(i)] != 0) else {
            return cdf_cost(&c.txb_skip[0][0], 1);
        };
        let mut bits = cdf_cost(&c.txb_skip[0][0], 0)
            + cdf_cost(&c.txtp4[y_mode], if vertical { 2 } else { 3 });
        let dcs = self.dc_sign_ctx_span(0, px / 4, py / 4, 1, 1);
        let hi_bits = |m: u32, br: &[u16]| -> f32 {
            let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
            let mut b = 0f32;
            let mut coded = 0i32;
            for _ in 0..(COEFF_BASE_RANGE / 3) {
                let s = (total_br - coded).min(3);
                b += cdf_cost(br, s as usize);
                coded += s;
                if s < 3 {
                    break;
                }
            }
            if m >= 15 {
                b += golomb_cost(m - 15);
            }
            b
        };
        if eob == 0 {
            let dm = cf[0].unsigned_abs();
            bits += cdf_cost(&c.eob_bin_16_l1d, 0);
            bits += cdf_cost(&c.eob_base[0][0][0], (dm.min(3) - 1) as usize);
            if dm.min(3) == 3 {
                bits += hi_bits(dm, &c.br_tok[0][0][0]);
            } else if dm >= 15 {
                bits += golomb_cost(dm - 15);
            }
            bits += cdf_cost(&c.dc_sign[0][dcs], (cf[0] < 0) as usize);
            return bits;
        }
        let eob_bin = if eob < 2 {
            eob
        } else {
            32 - (eob as u32).leading_zeros() as usize
        };
        bits += cdf_cost(&c.eob_bin_16_l1d, eob_bin);
        if eob_bin > 1 {
            let nbits = eob_bin - 2;
            bits += cdf_cost(&c.eob_hi[0][0][eob_bin], (eob >> nbits) & 1) + nbits as f32;
        }
        let mut levels = [0u8; 16 * 10];
        let lvl = |lv: &[u8], x: usize, y: usize| -> u32 { lv[x * 16 + y] as u32 };
        let lo_ctx_1d = |lv: &[u8], x: usize, y: usize| -> (usize, u32) {
            let mut mag = lvl(lv, x, y + 1) + lvl(lv, x + 1, y) + lvl(lv, x, y + 2);
            let hi_mag = mag;
            mag += lvl(lv, x, y + 3) + lvl(lv, x, y + 4);
            let offset = 26 + if y > 1 { 10 } else { y * 5 };
            (
                offset
                    + if mag > 512 {
                        4
                    } else {
                        ((mag + 64) >> 7) as usize
                    },
                hi_mag,
            )
        };
        let ctx_e = 1 + (eob > 2) as usize + (eob > 4) as usize;
        {
            let (x, y) = (eob & 3, eob >> 2);
            let m = cf[pos_rc(eob)].unsigned_abs();
            bits += cdf_cost(&c.eob_base[0][0][ctx_e], (m.min(3) - 1) as usize);
            if m.min(3) == 3 {
                let bc = if y != 0 { 14 } else { 7 };
                bits += hi_bits(m, &c.br_tok[0][0][bc]);
            } else if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            bits += 1.0; // sign
            levels[x * 16 + y] = level_byte(m);
        }
        for i in (1..eob).rev() {
            let (x, y) = (i & 3, i >> 2);
            let m = cf[pos_rc(i)].unsigned_abs();
            let (ctx, hi_mag) = lo_ctx_1d(&levels, x, y);
            bits += cdf_cost(&c.base_tok[0][0][ctx], m.min(3) as usize);
            if m.min(3) == 3 {
                let mag = hi_mag & 63;
                let bc = (if y != 0 { 14 } else { 7 })
                    + if mag > 12 {
                        6
                    } else {
                        ((mag + 1) >> 1) as usize
                    };
                bits += hi_bits(m, &c.br_tok[0][0][bc]);
            } else if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            if m != 0 {
                bits += 1.0; // sign
            }
            levels[x * 16 + y] = level_byte(m);
        }
        let (dc_ctx, dc_hi_mag) = lo_ctx_1d(&levels, 0, 0);
        let dm = cf[0].unsigned_abs();
        bits += cdf_cost(&c.base_tok[0][0][dc_ctx], dm.min(3) as usize);
        if dm.min(3) == 3 {
            let mag = dc_hi_mag & 63;
            let bc = if mag > 12 {
                6
            } else {
                ((mag + 1) >> 1) as usize
            };
            bits += hi_bits(dm, &c.br_tok[0][0][bc]);
        } else if dm >= 15 {
            bits += golomb_cost(dm - 15);
        }
        if dm != 0 {
            bits += cdf_cost(&c.dc_sign[0][dcs], (cf[0] < 0) as usize);
        }
        bits
    }

    /// Entropy-accurate rate for a V_DCT / H_DCT rect16 block (RTX_16X8 when
    /// `w == 16`, RTX_8X16 when `w == 8`): read-only twin of
    /// `encode_rect_coeffs_1d` against the decision CDF snapshot, including
    /// the txb_skip and tx-type symbols.
    pub(crate) fn luma_rect_bits_1d(
        &self,
        cf: &[i32; 128],
        w: usize,
        vertical: bool,
        px: usize,
        py: usize,
        y_mode: usize,
    ) -> f32 {
        let c = self.dcdf();
        let h = 128 / w;
        let (hsh, wsh) = (h.trailing_zeros() as usize, w.trailing_zeros() as usize);
        let pos_xy = |i: usize| -> (usize, usize) {
            if vertical {
                (i & (w - 1), i >> wsh)
            } else {
                (i & (h - 1), i >> hsh)
            }
        };
        let pos_rc = |i: usize| -> usize {
            if vertical {
                let (x, y) = (i & (w - 1), i >> wsh);
                (x << hsh) | y
            } else {
                i
            }
        };
        let Some(eob) = (0..128).rev().find(|&i| cf[pos_rc(i)] != 0) else {
            return cdf_cost(&c.txb_skip[2][0], 1);
        };
        let mut bits = cdf_cost(&c.txb_skip[2][0], 0)
            + cdf_cost(&c.txtp[y_mode], if vertical { 2 } else { 3 });
        let dcs = self.dc_sign_ctx_span(0, px / 4, py / 4, w / 4, h / 4);
        let hi_bits = |m: u32, br: &[u16]| -> f32 {
            let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
            let mut b = 0f32;
            let mut coded = 0i32;
            for _ in 0..(COEFF_BASE_RANGE / 3) {
                let s = (total_br - coded).min(3);
                b += cdf_cost(br, s as usize);
                coded += s;
                if s < 3 {
                    break;
                }
            }
            if m >= 15 {
                b += golomb_cost(m - 15);
            }
            b
        };
        if eob == 0 {
            let dm = cf[0].unsigned_abs();
            bits += cdf_cost(&c.eob_bin_128_l1d, 0);
            bits += cdf_cost(&c.eob_base[2][0][0], (dm.min(3) - 1) as usize);
            if dm.min(3) == 3 {
                bits += hi_bits(dm, &c.br_tok[2][0][0]);
            } else if dm >= 15 {
                bits += golomb_cost(dm - 15);
            }
            bits += cdf_cost(&c.dc_sign[0][dcs], (cf[0] < 0) as usize);
            return bits;
        }
        let eob_bin = if eob < 2 {
            eob
        } else {
            32 - (eob as u32).leading_zeros() as usize
        };
        bits += cdf_cost(&c.eob_bin_128_l1d, eob_bin);
        if eob_bin > 1 {
            let nbits = eob_bin - 2;
            bits += cdf_cost(&c.eob_hi[2][0][eob_bin], (eob >> nbits) & 1) + nbits as f32;
        }
        let mut levels = [0u8; 16 * 18];
        let lvl = |lv: &[u8], x: usize, y: usize| -> u32 { lv[x * 16 + y] as u32 };
        let lo_ctx_1d = |lv: &[u8], x: usize, y: usize| -> (usize, u32) {
            let mut mag = lvl(lv, x, y + 1) + lvl(lv, x + 1, y) + lvl(lv, x, y + 2);
            let hi_mag = mag;
            mag += lvl(lv, x, y + 3) + lvl(lv, x, y + 4);
            let offset = 26 + if y > 1 { 10 } else { y * 5 };
            (
                offset
                    + if mag > 512 {
                        4
                    } else {
                        ((mag + 64) >> 7) as usize
                    },
                hi_mag,
            )
        };
        let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
        {
            let (x, y) = pos_xy(eob);
            let m = cf[pos_rc(eob)].unsigned_abs();
            bits += cdf_cost(&c.eob_base[2][0][ctx_e], (m.min(3) - 1) as usize);
            if m.min(3) == 3 {
                let bc = if y != 0 { 14 } else { 7 };
                bits += hi_bits(m, &c.br_tok[2][0][bc]);
            } else if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            bits += 1.0; // sign
            levels[x * 16 + y] = level_byte(m);
        }
        for i in (1..eob).rev() {
            let (x, y) = pos_xy(i);
            let m = cf[pos_rc(i)].unsigned_abs();
            let (ctx, hi_mag) = lo_ctx_1d(&levels, x, y);
            bits += cdf_cost(&c.base_tok[2][0][ctx], m.min(3) as usize);
            if m.min(3) == 3 {
                let mag = hi_mag & 63;
                let bc = (if y != 0 { 14 } else { 7 })
                    + if mag > 12 {
                        6
                    } else {
                        ((mag + 1) >> 1) as usize
                    };
                bits += hi_bits(m, &c.br_tok[2][0][bc]);
            } else if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            if m != 0 {
                bits += 1.0; // sign
            }
            levels[x * 16 + y] = level_byte(m);
        }
        let (dc_ctx, dc_hi_mag) = lo_ctx_1d(&levels, 0, 0);
        let dm = cf[0].unsigned_abs();
        bits += cdf_cost(&c.base_tok[2][0][dc_ctx], dm.min(3) as usize);
        if dm.min(3) == 3 {
            let mag = dc_hi_mag & 63;
            let bc = if mag > 12 {
                6
            } else {
                ((mag + 1) >> 1) as usize
            };
            bits += hi_bits(dm, &c.br_tok[2][0][bc]);
        } else if dm >= 15 {
            bits += golomb_cost(dm - 15);
        }
        if dm != 0 {
            bits += cdf_cost(&c.dc_sign[0][dcs], (cf[0] < 0) as usize);
        }
        bits
    }

    pub(crate) fn mode_bits(&self, px: usize, py: usize, m: usize) -> f32 {
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];

        cdf_cost(&self.dcdf().kf_y[yctx], m)
    }

    /// Exact decision-side cost of the UV syntax emitted by `emit_uv_mode`.
    /// Keeping this keyed by the candidate luma mode is essential: AV1 has a
    /// separate UV-mode CDF for every `y_mode`, and CfL adds joint-sign and
    /// magnitude symbols whose contexts depend on both alpha signs.
    pub(crate) fn uv_mode_bits(&self, y_mode: usize, uv_mode: usize, cfl: Option<[i32; 2]>) -> f32 {
        if !exact_uv_rate_enabled() {
            return if let Some(a) = cfl {
                4.0 + 4.0 * u8::from(a[0] != 0) as f32 + 4.0 * u8::from(a[1] != 0) as f32
            } else if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
                7.0
            } else {
                4.0
            };
        }
        let c = self.dcdf();
        let Some(a) = cfl else {
            let mut bits = cdf_cost(&c.uv_mode[13 + y_mode], uv_mode);
            if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
                bits += cdf_cost(&c.angle_delta[uv_mode - V_PRED], 3);
            }
            return bits;
        };
        let su = if a[0] == 0 {
            0
        } else if a[0] < 0 {
            1
        } else {
            2
        };
        let sv = if a[1] == 0 {
            0
        } else if a[1] < 0 {
            1
        } else {
            2
        };
        let sign = su * 3 + sv;
        if sign == 0 {
            return cdf_cost(&c.uv_mode[13 + y_mode], DC_PRED);
        }
        let mut bits =
            cdf_cost(&c.uv_mode[13 + y_mode], CFL_PRED) + cdf_cost(&c.cfl_sign, sign - 1);
        if su != 0 {
            let ctx = (su == 2) as usize * 3 + sv;
            bits += cdf_cost(&c.cfl_alpha[ctx], (a[0].unsigned_abs() - 1) as usize);
        }
        if sv != 0 {
            let ctx = (sv == 2) as usize * 3 + su;
            bits += cdf_cost(&c.cfl_alpha[ctx], (a[1].unsigned_abs() - 1) as usize);
        }
        bits
    }

    fn skip_ctx(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0 // luma: TX size == block size -> ctx 0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = (a[bx4] != 0x40 || a[bx4 + 1] != 0x40) as usize;
            let cl = (l[by4] != 0x40 || l[by4 + 1] != 0x40) as usize;
            7 + ca + cl
        }
    }

    /// Luma `txb_skip` context for a transform SMALLER than its block (the
    /// dav1d `get_skip_ctx` table path — the `tx == block` case is ctx 0, see
    /// [`Self::skip_ctx`]): OR the per-4x4 coef bytes across the TX span,
    /// strip the dc-sign marker bits, and index `dav1d_skip_ctx`.
    fn skip_ctx_split(&self, bx4: usize, by4: usize, tw4: usize, th4: usize) -> usize {
        static SKIP_CTX_TBL: [[u8; 5]; 5] = [
            [1, 2, 2, 2, 3],
            [2, 4, 4, 4, 5],
            [2, 4, 4, 4, 5],
            [2, 4, 4, 4, 5],
            [3, 5, 5, 5, 6],
        ];
        let a = &self.a_coef[0];
        let l = &self.l_coef[0];
        let mut la = 0u8;
        for i in 0..tw4 {
            la |= a[bx4 + i];
        }
        let mut ll = 0u8;
        for i in 0..th4 {
            ll |= l[by4 + i];
        }
        SKIP_CTX_TBL[((la & 0x3f) as usize).min(4)][((ll & 0x3f) as usize).min(4)] as usize
    }

    fn dc_sign_ctx_span(
        &self,
        plane: usize,
        bx4: usize,
        by4: usize,
        nw: usize,
        nh: usize,
    ) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let mut s = 0i32;
        for &v in a[bx4..(bx4 + nw).min(a.len())].iter() {
            s += (v >> 6) as i32 - 1;
        }
        for &v in l[by4..(by4 + nh).min(l.len())].iter() {
            s += (v >> 6) as i32 - 1;
        }
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma = (a[bx4] >> 6) as i32 + (a[bx4 + 1] >> 6) as i32;
        let suml = (l[by4] >> 6) as i32 + (l[by4 + 1] >> 6) as i32;
        let s = suma + suml - 4;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a 16x16 transform. Luma: tx == block size -> 0.
    /// Chroma (4:4:4, chroma tx == chroma block): `7 + above_nz + left_nz` over
    /// the 4-unit (16-sample) footprint (`get_txb_skip_ctx`, ctx_offset = 7).
    fn skip_ctx_16(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
            let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
            7 + ca + cl
        }
    }

    /// dc_sign context for a 16x16 transform (4-unit footprint, baseline -8).
    fn dc_sign_ctx_16(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 4]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 4]);
        let s = suma + suml - 8;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a luma RTX_16X8 block coded as a single transform.
    fn skip_ctx_16x8_luma(&self) -> usize {
        0
    }

    fn skip_ctx_8x16_luma(&self) -> usize {
        0
    }

    fn dc_sign_ctx_8x16_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 2]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 4]);
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// dc_sign context for a luma RTX_16X8 transform: 4 above-units (16 wide) and
    /// 2 left-units (8 tall) top-bit sums, baseline -(4+2) = -6.
    fn dc_sign_ctx_16x8_luma(&self, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[0];
        let l = &self.l_coef[0];
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 4]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 2]);
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:4:4 chroma txb_skip context for an RTX_16X8 block (block == transform, so
    /// `not_one_blk` is 0): `7 + ca + cl` over 4 above-units and 2 left-units.
    fn skip_ctx_16x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:4:4 chroma dc_sign context for RTX_16X8: 4 above + 2 left, baseline -6.
    fn dc_sign_ctx_16x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 4]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 2]);
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_8x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 2].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_8x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 2]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 4]);
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx_8x4_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let sa: i32 = sum_coef_sign(&a[bx4..bx4 + 2]);
        let s = sa + (l[by4] >> 6) as i32 - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx_4x8_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let sl: i32 = sum_coef_sign(&l[by4..by4 + 2]);
        let s = (a[bx4] >> 6) as i32 + sl - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_4x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = (a[bx4] != 0x40) as usize;
        let cl = (l[by4] != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_4x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let s = (a[bx4] >> 6) as i32 + (l[by4] >> 6) as i32 - 2;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_8x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 2].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_8x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 2]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 2]);
        let s = suma + suml - 4;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_8x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 2].iter().any(|&x| x != 0x40) as usize;
        let cl = (l[by4] != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_8x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 2]);
        let suml = (l[by4] >> 6) as i32;
        let s = suma + suml - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_4x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = (a[bx4] != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_4x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma = (a[bx4] >> 6) as i32;
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 2]);
        let s = suma + suml - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    #[inline]
    /// Chroma subsampling class for the frame-level delta laws:
    /// 0 = 4:4:4 / mono, 1 = 4:2:2, 2 = 4:2:0.
    fn chroma_sub(&self) -> usize {
        if self.ss420 {
            2
        } else if self.ss422 {
            1
        } else {
            0
        }
    }

    fn luma_mode_budget_eff(&self) -> usize {
        self.speed
            .luma_mode_budget(!self.ss420 && !self.ss422 && !self.mono)
    }

    fn top_band(&self) -> bool {
        self.aq.enabled
            && !self.mono
            && (self.aq.base_q as u32) <= crate::tuning::get().top_band_q
            && self.speed == Speed::Slow
    }

    fn partition_signal_bits(&self) -> f32 {
        if (self.ss420 || self.ss422) && self.aq.enabled && !self.mono && self.speed == Speed::Slow
        {
            let tu = crate::tuning::get();
            let w = if self.ss422 {
                tu.seam_w_422
            } else {
                tu.seam_w_420
            };
            let t = seam_t(self.aq.base_q, w);
            if t > 0.0 {
                return tu.part_signal_bits - tu.part_signal_seam_drop * t;
            }
        }
        crate::tuning::get().part_signal_bits
    }

    /// Exact frozen-CDF cost of one 16x16 partition symbol
    /// (0=NONE, 1=HORZ, 2=VERT, 3=SPLIT).
    fn partition16_bits(&self, x8: usize, y8: usize, symbol: usize) -> f32 {
        self.partition_bits_bl(3, x8, y8, symbol)
    }

    /// Same, at an arbitrary partition level. `bl` follows `decode_sb`:
    /// 1 = 64x64, 2 = 32x32, 3 = 16x16, 4 = 8x8; the CDF row is `bl - 1`.
    fn partition_bits_bl(&self, bl: usize, x8: usize, y8: usize, symbol: usize) -> f32 {
        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
        cdf_cost(&self.dcdf().part_split[bl - 1][ctx], symbol)
    }

    /// Partition-symbol rate at the 32x32 / 64x64 nodes. Same defect the 16
    /// level had: NONE was free while SPLIT paid the flat fee.
    pub(crate) fn part_rate_bl(&self, bl: usize, x8: usize, y8: usize, symbol: usize) -> f32 {
        let t = crate::tuning::get();
        let ss444 = !self.ss420 && !self.ss422 && !self.mono;
        if t.exact_part_bits_3264 || (t.exact_part_bits_3264_444 && ss444) {
            self.partition_bits_bl(bl, x8, y8, symbol)
        } else if symbol == 0 {
            0.0
        } else {
            self.partition_signal_bits()
        }
    }

    /// Partition-symbol rate for a 16x16 candidate: exact when
    /// `exact_part_bits` is on, else the legacy flat fee (NONE free).
    fn part16_rate(&self, x8: usize, y8: usize, symbol: usize) -> f32 {
        if crate::tuning::get().exact_part_bits && !(self.ss422 && !self.mono) {
            self.partition16_bits(x8, y8, symbol)
        } else if symbol == 0 {
            0.0
        } else {
            self.partition_signal_bits()
        }
    }

    fn mlam(&self) -> f32 {
        mode_lambda_q(self.quant.dc_q() as f32) * self.tune_weight()
    }

    /// As [`Self::mlam`] but for chroma planes (uses `self.cquant`).
    #[inline]
    fn mlam_c(&self) -> f32 {
        mode_lambda_q(self.cquant.dc_q() as f32) * self.tune_weight()
    }

    fn chroma_partition_weight_at(&self, px: usize, py: usize, lw: usize, lh: usize) -> f32 {
        if self.mono {
            return 0.0;
        }
        if !local_chroma_weight_enabled() {
            return self.chroma_part_weight;
        }
        let (sx, sy) = (
            usize::from(self.ss420 || self.ss422),
            usize::from(self.ss420),
        );
        let (cx, cy) = (px >> sx, py >> sy);
        let (cw, ch) = ((lw >> sx).max(1), (lh >> sy).max(1));
        let variance = |plane: &[u16], stride: usize, x: usize, y: usize, w: usize, h: usize| {
            let n = (w * h) as f64;
            let (sum, sum_sq) = block_moments_i32(plane, stride, x, y, w, h);
            let sum = sum as f64;
            (sum_sq as f64 / n - (sum / n) * (sum / n)).max(0.0)
        };
        let yv = variance(&self.src[0], self.w, px, py, lw, lh);
        let cv = 0.5
            * (variance(&self.src[1], self.cw, cx, cy, cw, ch)
                + variance(&self.src[2], self.cw, cx, cy, cw, ch));
        let depth_floor = ((1u32 << self.bd.saturating_sub(8)) as f64).powi(2);
        let local_share = cv / (cv + yv + depth_floor);
        self.chroma_part_weight * f_fmlaf(0.5, local_share as f32, 1.0)
    }

    /// libaom SSIMULACRA2 rdmult weight for this frame (1.0 when tune is off).
    #[inline]
    fn tune_weight(&self) -> f32 {
        mode_lambda_weight(self.base_q_idx)
    }

    pub(crate) fn part_bl8_rate(&self, x8: usize, y8: usize, symbol: usize) -> f32 {
        if crate::tuning::get().exact_part_bits_8 {
            let ctx = get_partition_ctx(&self.a_part, &self.l_part, 4, x8, y8);
            return cdf_cost(&self.dcdf().part_bl8[ctx], symbol);
        }
        match symbol {
            0 => 0.0,
            3 => 2.0,
            _ => self.partition_signal_bits(),
        }
    }

    fn prefer_8x8_none(&self, x8: usize, y8: usize) -> bool {
        if self.mono {
            return true;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let prdo = self.perceptual_rd_scale(px, py, 8);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // Best model-shortlisted cost for one 8x8 (DCT_DCT only; cheap proxy).
        let mut eff8 = f32::INFINITY;
        let mode_shortlist = self.rank_luma_modes::<64>(modes, px, py, 8, 8, false, false, 2);
        for &m in &mode_shortlist {
            let mut pred = [0i32; 64];
            if m == DC_PRED {
                let d = self
                    .intrapred
                    .dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 64];
            } else {
                self.intrapred.predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    false,
                    false,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            self.rd
                .residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 8, 8);
            let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_8X8, lam);
            let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
            let distortion = self.luma_partition_distortion(
                px,
                py,
                8,
                8,
                self.quant.ac_q() as f32,
                &pred[..],
                0,
                &rr[..],
            );
            let eff = crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.luma_bits(&cf, &SCAN_8X8, 8, px, py, m, 1)
                    + if crate::tuning::get().exact_8x8_mode_rate {
                        self.mode_bits(px, py, m)
                    } else {
                        0.0
                    },
            );
            if eff < eff8 {
                eff8 = eff;
            }
        }
        let mut eff4_sum = self.rd_cost_split4_luma(px, py, lam, mlam);
        // In 4:4:4, splitting a 8x8 luma block also changes chroma from one
        // 8x8 block into four independently predicted 4x4 blocks. In 4:2:0 the
        // four sub-8x8 luma blocks share one 4x4 chroma reference block, so the
        // chroma geometry is unchanged and cancels from this comparison.
        if !self.ss420 && !self.ss422 {
            eff8 += self.chroma_partition_weight_at(px, py, 8, 8)
                * self.rd_cost_chroma_block(px, py, 8, 8, prdo);
            for (sx, sy) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)] {
                eff4_sum += self.chroma_partition_weight_at(px + sx, py + sy, 4, 4)
                    * self.rd_cost_chroma_block(px + sx, py + sy, 4, 4, prdo);
            }
        } else if self.ss422 {
            // Splitting changes chroma from one 4x8 into two paired 4x4s.
            eff8 += self.chroma_partition_weight_at(px, py, 8, 8)
                * self.rd_cost_chroma_block(px / 2, py, 4, 8, prdo);
            for sy in [0usize, 4] {
                eff4_sum += self.chroma_partition_weight_at(px, py + sy, 8, 4)
                    * self.rd_cost_chroma_block(px / 2, py + sy, 4, 4, prdo);
            }
        }
        let eff8 = eff8 + rate_cost(mlam, self.part_bl8_rate(x8, y8, 0));
        eff8 <= eff4_sum
    }

    /// Best luma cost of coding this 8x8 as four 4x4 (PARTITION_SPLIT at
    /// BL_8X8), extracted from [`Self::prefer_8x8_none`] so parent SPLIT legs
    /// can price the 8-level BEST (min of NONE-8 and SPLIT4) instead of the
    /// forced-NONE upper bound.
    fn rd_cost_split4_luma(&self, px: usize, py: usize, lam: f32, mlam: f32) -> f32 {
        let key = ((px as u128) << 96)
            | ((py as u128) << 64)
            | ((lam.to_bits() as u128) << 32)
            | u128::from(mlam.to_bits());
        let epoch = self.emit_epoch.get();
        if let Some(&(cached_epoch, cost)) = self.split4_rd_cache.borrow().get(&key)
            && cached_epoch == epoch
        {
            return cost;
        }
        // KNOWN MODELLING GAP: this
        // estimator trials DCT only (`dct4x4_t`), while the emitter also trials
        // ADST and IDTX per 4x4 (block16.rs). A 4x4 that needs ADST/IDTX is
        // therefore judged as its weaker DCT form and can lose the partition
        // decision it would have won.
        // Source-domain breakout, the 16-level `split_breakout_k` analogue:
        // compare one 8x8 against four 4x4s by within-block SSE (from the
        // quadrant moments) plus a flat per-block bit charge, and skip the
        // real pricing when SPLIT4 cannot pay. The caller takes
        // `none8.min(this)`, so INFINITY simply drops the candidate.
        let k4 = split4_breakout_k(self.speed);
        if k4.is_finite() {
            let (mut sum_t, mut sq_t, mut sse_split) = (0i64, 0i64, 0i64);
            for (qx, qy) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)] {
                let (s, q) = block_moments_i32(&self.src[0], self.w, px + qx, py + qy, 4, 4);
                sum_t += s;
                sq_t += q;
                sse_split += q - (s * s + 8) / 16;
            }
            let sse_none = sq_t - (sum_t * sum_t + 32) / 64;
            let m_none = rd_cost_i64(sse_none.max(0), mlam, 4.0);
            let m_split = rd_cost_i64(sse_split.max(0), mlam, 16.0);
            if m_split > m_none * k4 {
                self.split4_rd_cache
                    .borrow_mut()
                    .insert(key, (epoch, f32::INFINITY));
                return f32::INFINITY;
            }
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // Same 4x4 mode subset as the SPLIT4 emitter. Build it once for all
        // four children; this used to allocate and filter a Vec per child.
        let full_set4 = self.ss420 || self.ss422 || self.mono;
        let mut allowed_modes = FixedList::<usize, 13>::new(DC_PRED);
        for &mode in modes {
            if full_set4
                || !(mode == SMOOTH_V_PRED
                    || mode == SMOOTH_H_PRED
                    || (is_directional_mode(mode) && mode != V_PRED && mode != H_PRED))
            {
                allowed_modes.push(mode);
            }
        }
        // PARTITION_SPLIT symbol. Was a flat 2.0; now routed through
        // `part_bl8_rate` so NONE/HORZ/VERT/SPLIT4 all price from one place and
        // exact mode cannot double-charge it (see `prefer_8x8_none`).
        let mut eff4_sum = rate_cost(mlam, self.part_bl8_rate(px / 8, py / 8, 3));
        // Parent-level availability, approximated (the exact spec flags depend
        // on the z-order position within the 16 block, which this pricing
        // proxy doesn't track; the emitter computes them exactly).
        let par_tr = py > 0 && px + 8 < self.w;
        let par_bl = px > 0 && py + 8 < self.h;
        // KNOWN MODELLING BUG.
        // Nothing reconstructs between the four children here: every one
        // predicts from `self.recon`, which still holds PRE-BLOCK pixels inside
        // the 8x8. si 1's LEFT edge is si 0's right column, si 2's ABOVE edge
        // is si 0's bottom row, and si 3 depends on both — all stale, and
        // `rank_luma_modes` prunes the shortlist against them.
        //
        // The damage is in those PRIMARY above/left contexts, NOT in the tr/bl
        // flags: forcing si 2's `sub_tr` false was measured bit-identical on
        // 4:4:4, 4:2:2 and 4:2:0 (the branch is reached 32k-55k times per
        // encode, so that is a real null, not an unreached path). There is
        // therefore no cheap flag-level fix. The only correct one is to
        // reconstruct TL/TR/BL/BR sequentially into a temporary buffer, as
        // `split4tx_try` does — which needs `&mut self`, and this decision path
        // is built around `&self` + RefCell caches; converting it ripples
        // through `prefer_8x8_none` and `rd_choice_16_inner`.
        for (si, (sx, sy)) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)]
            .into_iter()
            .enumerate()
        {
            let (bx, by) = (px + sx, py + sy);
            let (sub_tr, sub_bl) = match si {
                // si 0: both edges are OUTSIDE the 8x8 -> genuinely reconstructed.
                0 => (py > 0, px > 0),
                // si 1: above-right is outside the block too (parent's flag).
                1 => (par_tr, false),
                // si 2: `sub_tr` points at si 1's area -- internal, and never
                // reconstructed in this decision. MEASURED: forcing it false
                // (reached 32k-55k times per encode) is BIT-IDENTICAL on 444,
                // 422 and 420, so this flag is NOT where the damage is.
                2 => (true, par_bl),
                _ => (false, false),
            };
            let mode_shortlist = if self.top_band() && !self.ss420 && !self.ss422 {
                // Match the equipped 4x4 emitter in the target band. The
                // historical two-mode proxy can miss the leaf's actual winner
                // and make SPLIT4 look more expensive than it emits.
                allowed_modes
            } else {
                self.rank_luma_modes::<16>(&allowed_modes, bx, by, 4, 4, sub_tr, sub_bl, 2)
            };
            let mut best = f32::INFINITY;
            for &m in &mode_shortlist {
                let mut pred = [0i32; 16];
                if m == DC_PRED {
                    let d =
                        self.intrapred
                            .dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    self.intrapred.predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        sub_tr,
                        sub_bl,
                        self.w,
                        self.h,
                        self.luma_filter_type(bx, by),
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 16];
                self.rd
                    .residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 4, 4);
                let (mut cf, tf) = self.dct.dct4x4_t(&resid, &self.quant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_4X4, lam);
                let rr = self.idct.idct_dequant_4x4(&cf, &self.quant);
                let distortion = self.luma_partition_distortion(
                    bx,
                    by,
                    4,
                    4,
                    self.quant.ac_q() as f32,
                    &pred[..],
                    0,
                    &rr[..],
                );
                // +mode/skip signaling allowance per 4x4 sub-block
                let mode_rate = if crate::tuning::get().exact_8x8_mode_rate {
                    self.mode_bits(bx, by, m)
                } else {
                    4.0f32
                };
                let mut eff = crate::partition_rd::rd_cost(
                    distortion,
                    mlam,
                    self.luma_bits(&cf, &SCAN_4X4, 4, bx, by, m, 1) + mode_rate,
                );
                if crate::tuning::get().split4_decision_txtypes {
                    for txtp in [4u8, 0] {
                        let (mut acf, atf) = match txtp {
                            4 => self.dct.adst4x4_t(&resid, &self.quant),
                            _ => self.dct.idtx4x4_t(&resid, &self.quant),
                        };
                        if txtp == 4 {
                            trellis_optimize(&mut acf, &atf, dcq, acq, &SCAN_4X4, lam);
                        }
                        let arr = match txtp {
                            4 => self.idct.iadst_dequant_4x4(&acf, &self.quant),
                            _ => self.idct.iidentity_dequant_4x4(&acf, &self.quant),
                        };
                        let adist = self.luma_partition_distortion(
                            bx,
                            by,
                            4,
                            4,
                            self.quant.ac_q() as f32,
                            &pred[..],
                            0,
                            &arr[..],
                        );
                        let aeff = crate::partition_rd::rd_cost(
                            adist,
                            mlam,
                            self.luma_bits(&acf, &SCAN_4X4, 4, bx, by, m, txtp as usize)
                                + mode_rate,
                        );
                        if aeff < eff {
                            eff = aeff;
                        }
                    }
                }
                if eff < best {
                    best = eff;
                }
            }
            eff4_sum += best;
        }
        self.split4_rd_cache
            .borrow_mut()
            .insert(key, (epoch, eff4_sum));
        eff4_sum
    }

    /// Mean and variance of a `w`x`h` luma source region at pixel origin
    /// (px, py). libaom's partition search uses exactly these per-candidate
    /// variance features (`block_var`, `horz_block_var[2]`, `sub_block_var[4]`)
    /// to steer and prune the decision before paying for full R-D.
    /// Mean of a luma source region (native depth).
    fn luma_mean(&self, px: usize, py: usize, w: usize, h: usize) -> f32 {
        let sum = block_sum_i32(&self.src[0], self.w, px, py, w, h);
        sum as f32 / (w * h) as f32
    }

    fn banding_risk(&self, px: usize, py: usize, dim: usize) -> bool {
        // High quality only: at coarse quantizers the sub-TX DCs quantize as
        // coarsely as the big block's (no banding win) and the extra rate of
        // four coded DCs is comparatively expensive.
        if self.base_q_idx >= 100 {
            return false;
        }
        let var_scale = 1.0 / (1u32 << (2 * (self.bd - 8))) as f32;
        let pix_scale = 1.0 / (1u32 << (self.bd - 8)) as f32;
        let var = self.luma_variance(px, py, dim, dim) * var_scale;
        if !(3.0..100.0).contains(&var) {
            return false;
        }
        let hd = dim / 2;
        let m = [
            self.luma_mean(px, py, hd, hd),
            self.luma_mean(px + hd, py, hd, hd),
            self.luma_mean(px, py + hd, hd, hd),
            self.luma_mean(px + hd, py + hd, hd, hd),
        ];
        let (lo, hi) = m
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        (hi - lo) * pix_scale >= 1.0
    }

    fn luma_variance(&self, px: usize, py: usize, w: usize, h: usize) -> f32 {
        let (sum, sum_sq) = block_moments_i32(&self.src[0], self.w, px, py, w, h);
        let n = (w * h) as f32;
        let mean = sum as f32 / n;
        (sum_sq as f32 / n) - mean * mean
    }

    fn prefer_split64_from_source(&self, px: usize, py: usize, prdo: f32) -> bool {
        if self.speed != Speed::Fast || self.allow_intrabc {
            return false;
        }
        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        let mut split_sse = 0i64;
        for (sx, sy) in [(0usize, 0usize), (32, 0), (0, 32), (32, 32)] {
            let (s, q) = block_moments_i32(&self.src[0], self.w, px + sx, py + sy, 32, 32);
            sum += s;
            sum_sq += q;
            split_sse += q - (s * s + 512) / 1024;
        }
        let none_sse = sum_sq - (sum * sum + 2048) / 4096;
        let mlam = self.mlam() * prdo;
        let modeled_none = rd_cost_i64(none_sse.max(0), mlam, 4.0);
        let modeled_split = rd_cost_i64(split_sse.max(0), mlam, 16.0);
        modeled_split * 1.5 < modeled_none
    }

    fn partition_model_cost16(&self, part: Part16, mlam: f32, cells: &[(i64, i64); 16]) -> f32 {
        let blocks: &[(usize, usize, usize, usize)] = match part {
            Part16::Horz => &[(0, 0, 16, 8), (0, 8, 16, 8)],
            Part16::Vert => &[(0, 0, 8, 16), (8, 0, 8, 16)],
            Part16::HorzA => &[(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 16, 8)],
            Part16::HorzB => &[(0, 0, 16, 8), (0, 8, 8, 8), (8, 8, 8, 8)],
            Part16::VertA => &[(0, 0, 8, 8), (0, 8, 8, 8), (8, 0, 8, 16)],
            Part16::VertB => &[(0, 0, 8, 16), (8, 0, 8, 8), (8, 8, 8, 8)],
            Part16::Horz4 => &[(0, 0, 16, 4), (0, 4, 16, 4), (0, 8, 16, 4), (0, 12, 16, 4)],
            Part16::Vert4 => &[(0, 0, 4, 16), (4, 0, 4, 16), (8, 0, 4, 16), (12, 0, 4, 16)],
            Part16::None => &[(0, 0, 16, 16)],
            Part16::Split => &[(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 8, 8), (8, 8, 8, 8)],
            _ => unreachable!("no partition model for {part:?}"),
        };
        rd_cost_i64(
            partition_model_sse(blocks, cells),
            mlam,
            4.0 * blocks.len() as f32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn luma_partition_distortion(
        &self,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        qstep: f32,
        pred: &[i32],
        dc: i32,
        residual: &[i32],
    ) -> f32 {
        let satd = self.rd.luma_satd(
            &self.src[0],
            self.w,
            px,
            py,
            w,
            h,
            self.bd,
            pred,
            dc,
            residual,
        );
        satd as f32
            * qstep.max(1.0)
            * 0.25
            * crate::partition_rd::luma_satd_scale(
                self.aq.base_q,
                self.ss420 || self.ss422 || self.mono,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn rd_cost_chroma_fixed<const N: usize>(
        &self,
        cx: usize,
        cy: usize,
        cw: usize,
        ch: usize,
        prdo: f32,
        scan: &[u32],
        pred: impl Fn(&[u16], usize, usize, usize, i32) -> i32,
        fwd: impl Fn(&[i32; N], &Quant) -> ([i32; N], [f32; N]),
        inv: impl Fn(&[i32; N], &Quant) -> [i32; N],
    ) -> f32 {
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        // Deliberately DC-only even though the emitters search CfL: a CfL
        // trial here (source-luma AC basis, per-plane best alpha, joint
        // two-plane decision + incremental exact uv_mode_bits signaling).
        let mut dc_total = 0.0f32;
        for plane in 1..=2 {
            let dc = pred(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            let mut resid = [0i32; N];
            self.rd
                .residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, cw, ch, dc);
            let (mut cf, tf) = fwd(&resid, &self.cquant);
            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
            let rr = inv(&cf, &self.cquant);
            let distortion = crate::partition_rd::chroma_sse(
                &self.src[plane],
                self.cw,
                cx,
                cy,
                cw,
                ch,
                self.bd,
                |i| dc + rr[i],
            );
            dc_total += crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.chroma_rect_bits(&cf, scan, cw, ch, plane, cx, cy),
            );
        }
        // Every emitted chroma leaf signals one uv_mode symbol, which this
        // proxy priced at zero. Because `rd_cost_chroma_partition` sums this
        // per DERIVED leaf, the omission scaled with leaf count -- SPLIT hid 4
        // symbols, HORZ/VERT 2, NONE 1 -- a systematic tilt toward finer
        // partitions. Priced at the DC mode, matching the DC-only proxy above.
        // MEASURED NEGATIVE (Slow, 210 cases): train +0.002 but holdout -0.0065
        // and p10 worse. `uv_mode_bits(DC, DC, None)` is a CONSTANT, so this is
        // just another flat per-leaf fee, and `exact_part_bits` already fixed
        // that tilt. Pricing the uv_mode the emitter ACTUALLY picks (incl. CfL)
        // is the real fix; a DC-proxy constant is not. Kept off.
        if crate::tuning::get().chroma_uv_mode_bits {
            dc_total += rate_cost(mlam, self.uv_mode_bits(DC_PRED, DC_PRED, None));
        }
        dc_total
    }

    fn rd_cost_chroma_block(&self, cx: usize, cy: usize, cw: usize, ch: usize, prdo: f32) -> f32 {
        let key = ((cx as u128) << 96)
            | ((cy as u128) << 64)
            | ((cw as u128) << 56)
            | ((ch as u128) << 48)
            | prdo.to_bits() as u128;
        let epoch = self.emit_epoch.get();
        if let Some(&(e, cost)) = self.chroma_rd_cache.borrow().get(&key)
            && e == epoch
        {
            return cost;
        }
        let cost = match (cw, ch) {
            (4, 4) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_4X4,
                |r, st, x, y, bd| self.intrapred.dc_pred_4x4(r, st, x, y, bd),
                |r, q| self.dct.dct4x4_t(r, q),
                |levels, q| self.idct.idct_dequant_4x4(levels, q),
            ),
            (8, 4) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_8X4,
                |r, st, x, y, bd| self.intrapred.dc_pred_8x4(r, st, x, y, bd),
                |r, q| self.dct.dct8x4_t(r, q),
                |levels, q| self.idct.idct_dequant_8x4(levels, q),
            ),
            (4, 8) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_4X8,
                |r, st, x, y, bd| self.intrapred.dc_pred_4x8(r, st, x, y, bd),
                |r, q| self.dct.dct4x8_t(r, q),
                |levels, q| self.idct.idct_dequant_4x8(levels, q),
            ),
            (16, 4) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X4,
                |r, st, x, y, bd| self.intrapred.dc_pred(r, st, x, y, 16, 4, bd),
                |r, q| self.dct.dct16x4_t(r, q),
                |levels, q| self.idct.idct_dequant_16x4(levels, q),
            ),
            (4, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_4X16,
                |r, st, x, y, bd| self.intrapred.dc_pred(r, st, x, y, 4, 16, bd),
                |r, q| self.dct.dct4x16_t(r, q),
                |levels, q| self.idct.idct_dequant_4x16(levels, q),
            ),
            (8, 8) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_8X8,
                |r, st, x, y, bd| self.intrapred.dc_pred_8x8(r, st, x, y, bd),
                |r, q| self.dct.dct8x8_t(r, q),
                |levels, q| self.idct.idct_dequant_8x8(levels, q),
            ),
            (16, 8) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X8,
                |r, st, x, y, bd| self.intrapred.dc_pred_16x8(r, st, x, y, bd),
                |r, q| self.dct.dct16x8_t(r, q),
                |levels, q| self.idct.idct_dequant_16x8(levels, q),
            ),
            (8, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_8X16,
                |r, st, x, y, bd| self.intrapred.dc_pred_8x16(r, st, x, y, bd),
                |r, q| self.dct.dct8x16_t(r, q),
                |levels, q| self.idct.idct_dequant_8x16(levels, q),
            ),
            (16, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X16,
                |r, st, x, y, bd| self.intrapred.dc_pred_16x16(r, st, x, y, bd),
                |r, q| self.dct.dct16x16_t(r, q),
                |levels, q| self.idct.idct_dequant_16x16(levels, q),
            ),
            (32, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_32X16,
                |r, st, x, y, bd| self.intrapred.dc_pred_32x16(r, st, x, y, bd),
                |r, q| self.dct.dct32x16_t(r, q),
                |levels, q| self.idct.idct_dequant_32x16(levels, q),
            ),
            (16, 32) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X32,
                |r, st, x, y, bd| self.intrapred.dc_pred_16x32(r, st, x, y, bd),
                |r, q| self.dct.dct16x32_t(r, q),
                |levels, q| self.idct.idct_dequant_16x32(levels, q),
            ),
            (32, 32) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_32X32,
                |r, st, x, y, bd| self.intrapred.dc_pred_32x32(r, st, x, y, bd),
                |r, q| self.dct.dct32x32_t(r, q),
                |levels, q| self.idct.idct_dequant_32x32(levels, q),
            ),
            _ => unreachable!("unsupported derived chroma block {cw}x{ch}"),
        };
        self.chroma_rd_cache.borrow_mut().insert(key, (epoch, cost));
        cost
    }

    /// Sum chroma R-D for the blocks derived from a luma partition candidate.
    fn rd_cost_chroma_partition(
        &self,
        px: usize,
        py: usize,
        luma_dim: usize,
        part: Part16,
        prdo: f32,
    ) -> f32 {
        let sub_x = (self.ss420 || self.ss422) as usize;
        let sub_y = self.ss420 as usize;
        let blocks: &[(usize, usize, usize, usize)] = match part {
            Part16::Intrabc => unreachable!("IntraBC has no intra chroma partition"),
            Part16::None => &[(0, 0, luma_dim, luma_dim)],
            Part16::Split => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, half),
                    (half, 0, half, half),
                    (0, half, half, half),
                    (half, half, half, half),
                ]
            }
            Part16::Horz => {
                let half = luma_dim / 2;
                &[(0, 0, luma_dim, half), (0, half, luma_dim, half)]
            }
            Part16::Vert => {
                let half = luma_dim / 2;
                &[(0, 0, half, luma_dim), (half, 0, half, luma_dim)]
            }
            Part16::HorzA => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, half),
                    (half, 0, half, half),
                    (0, half, luma_dim, half),
                ]
            }
            Part16::HorzB => {
                let half = luma_dim / 2;
                &[
                    (0, 0, luma_dim, half),
                    (0, half, half, half),
                    (half, half, half, half),
                ]
            }
            Part16::VertA => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, half),
                    (0, half, half, half),
                    (half, 0, half, luma_dim),
                ]
            }
            Part16::VertB => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, luma_dim),
                    (half, 0, half, half),
                    (half, half, half, half),
                ]
            }
            // H4/V4: at 4:2:0 the 4px strips pair up in the subsampled
            // dimension (chroma is coded once per pair), so the chroma cost is
            // that of the two pair regions; otherwise four quarter strips.
            Part16::Horz4 => {
                let (q, half) = (luma_dim / 4, luma_dim / 2);
                if sub_y == 1 {
                    &[(0, 0, luma_dim, half), (0, half, luma_dim, half)]
                } else {
                    &[
                        (0, 0, luma_dim, q),
                        (0, q, luma_dim, q),
                        (0, 2 * q, luma_dim, q),
                        (0, 3 * q, luma_dim, q),
                    ]
                }
            }
            Part16::Vert4 => {
                let (q, half) = (luma_dim / 4, luma_dim / 2);
                if sub_x == 1 {
                    &[(0, 0, half, luma_dim), (half, 0, half, luma_dim)]
                } else {
                    &[
                        (0, 0, q, luma_dim),
                        (q, 0, q, luma_dim),
                        (2 * q, 0, q, luma_dim),
                        (3 * q, 0, q, luma_dim),
                    ]
                }
            }
        };
        blocks
            .iter()
            .map(|&(ox, oy, lw, lh)| {
                self.chroma_partition_weight_at(px + ox, py + oy, lw, lh)
                    * self.rd_cost_chroma_block(
                        (px + ox) >> sub_x,
                        (py + oy) >> sub_y,
                        lw >> sub_x,
                        lh >> sub_y,
                        prdo,
                    )
            })
            .sum::<f32>()
    }

    /// R-D estimate for PARTITION_HORZ_4 / VERT_4: four DC-predicted DCT strips
    /// (RTX_16X4 / RTX_4X16) plus the shared partition-signal allowance and the
    /// chroma proxy. DC-only like the other rect estimators — the emitters code
    /// exactly this, so the estimate prices what will actually be coded.
    fn rd_cost_quad16(&self, px: usize, py: usize, vert: bool, prdo: f32) -> f32 {
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let (lam, mlam) = (trellis_lambda() * prdo, self.mlam() * prdo);
        let mut total = rate_cost(mlam, self.partition_signal_bits());
        let (lw, lh) = if vert { (4usize, 16) } else { (16, 4) };
        let scan: &[u32] = if vert { &SCAN_4X16 } else { &SCAN_16X4 };
        for i in 0..4 {
            let (sx, sy) = if vert {
                (px + 4 * i, py)
            } else {
                (px, py + 4 * i)
            };
            let dc = self
                .intrapred
                .dc_pred(&self.recon[0], self.w, sx, sy, lw, lh, self.bd as i32);
            let mut resid = [0i32; 64];
            self.rd
                .residual_dc(&mut resid, &self.src[0], self.w, sx, sy, lw, lh, dc);
            let (mut cf, tf) = if vert {
                self.dct.dct4x16_t(&resid, &self.quant)
            } else {
                self.dct.dct16x4_t(&resid, &self.quant)
            };
            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
            let rr = if vert {
                self.idct.idct_dequant_4x16(&cf, &self.quant)
            } else {
                self.idct.idct_dequant_16x4(&cf, &self.quant)
            };
            let distortion = self.luma_partition_distortion(
                sx,
                sy,
                lw,
                lh,
                self.quant.ac_q() as f32,
                &[],
                dc,
                &rr[..],
            );
            // NB deliberately the PROXY rate, unlike the competing legs'
            // entropy-accurate `luma_bits`/`luma_rect_bits`: Do not "fix".
            total += crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, scan));
        }
        total
    }

    fn block_skip_bits(&self, px: usize, py: usize, skip: bool) -> f32 {
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        cdf_cost(&self.dcdf().skip[sctx], skip as usize)
    }

    /// Rate of the `tx_depth` symbol a block at (px, py) of size `w`x`h` will
    /// code for `depth` — the read-only twin of [`Self::code_tx_depth`], same
    /// category and neighbor context. TX-split trials used flat allowances
    /// (1.0 / 7.5 / +2.5 bits) and charged the whole-transform side NOTHING,
    /// making the comparison non-absolute (external review round 2, finding 2):
    /// depth 0 codes a symbol too.
    fn tx_depth_bits(&self, px: usize, py: usize, w: usize, h: usize, depth: usize) -> f32 {
        let l2 = |d: usize| -> i8 {
            match d {
                4 => 0,
                8 => 1,
                16 => 2,
                32 => 3,
                _ => 4,
            }
        };
        let (max_lw, max_lh) = (l2(w), l2(h));
        let max_l = max_lw.max(max_lh);
        if max_l == 0 {
            return 0.0; // TX_4X4 codes no depth symbol
        }
        let cat = max_l as usize - 1;
        let (bx4, by4) = (px / 4, py / 4);
        let ctx = (self.l_tx[by4] >= max_lh) as usize + (self.a_tx[bx4] >= max_lw) as usize;
        let cdf = &self.dcdf().txsz[cat][ctx];
        if depth >= cdf.len() {
            return 0.0;
        }
        cdf_cost(cdf, depth)
    }

    /// Intra-edge availability for a child at z-order offset `(sx, sy)` inside
    /// its parent, given the parent's recursion flags. Mirrors the `decode_sb`
    /// child table exactly: TL=(1,1) TR=(thr,0) BL=(1,lhb) BR=(0,0).
    fn child_edge_flags(sx: usize, sy: usize, thr: bool, lhb: bool) -> (bool, bool) {
        match (sx == 0, sy == 0) {
            (true, true) => (true, true),
            (false, true) => (thr, false),
            (true, false) => (true, lhb),
            (false, false) => (false, false),
        }
    }

    /// The `have_tr` / `have_bl` the LEAF EMITTER at (px, py, dim) will be
    /// handed, derived from the recursion flags reaching that node — the same
    /// expression `decode_sb` uses before every `code_block*` call. The
    /// partition proxy used to hard-code `false, false` here (external review
    /// round 2, finding 8c), under-equipping directional big blocks in the
    /// decision relative to what they actually get coded with.
    fn leaf_edge_flags(
        &self,
        px: usize,
        py: usize,
        dim: usize,
        thr: bool,
        lhb: bool,
    ) -> (bool, bool) {
        (
            thr && py > 0 && (px + dim) < self.w,
            lhb && px > 0 && (py + dim) < self.h,
        )
    }

    /// Full square+rectangular partition decision for a 16x16 luma region.
    /// Evaluate legal NONE/SPLIT/HORZ/VERT and 4:2:0 A/B candidates with the
    /// same SATD + mlam*bits objective; the previous qstep/variance pre-pruning
    /// lost too many useful rectangles at medium and coarse quantization.
    fn partition_choice_16(&self, x8: usize, y8: usize, thr: bool, lhb: bool) -> Part16 {
        self.rd_choice_16(x8, y8, thr, lhb).0
    }

    /// Full 16x16 partition decision returning the winner AND its total R-D
    /// cost (luma + chroma, priced under this block's own perceptual scale).
    /// The cost is what a parent SPLIT leg should charge for this child — the
    /// child's BEST achievable total, not its forced-NONE cost.
    fn rd_choice_16(&self, x8: usize, y8: usize, thr: bool, lhb: bool) -> (Part16, f32) {
        if self.mono {
            return (Part16::Split, f32::INFINITY); // monochrome codes 8x8 luma blocks only
        }
        // Edge flags are part of the identity: the same block reached with
        // different intra-edge availability prices differently.
        let key = ((x8 as u32) << 18) | ((y8 as u32) << 2) | ((thr as u32) << 1) | lhb as u32;
        let epoch = self.emit_epoch.get();
        if let Some(&(e, p, c)) = self.rd16_cache.borrow().get(&key)
            && e == epoch
        {
            return (p, c);
        }
        let out = self.rd_choice_16_inner(x8, y8, thr, lhb);
        self.rd16_cache
            .borrow_mut()
            .insert(key, (epoch, out.0, out.1));
        out
    }

    fn rd_choice_16_inner(&self, x8: usize, y8: usize, thr: bool, lhb: bool) -> (Part16, f32) {
        let (px, py) = (x8 * 8, y8 * 8);
        // Fixed-partition mode: no R-D at this level either. See the matching
        // comment in `rd_choice_rect32` for why returning a placeholder cost is
        // sound (nothing compares it).
        match crate::tuning::fixed_size(self.speed) {
            0 => {}
            16 => return (Part16::None, 0.0),
            n if n < 16 => return (Part16::Split, 0.0),
            _ => {}
        }

        let full_part_rdo = self.speed.full_partition_rdo();
        let coupled_square =
            !self.mono && self.speed == Speed::Slow && joint_luma_uv_proxy_enabled();

        // 4:2:0 runs the rect legs at Slow; `rect16_420_medium` opens Medium
        // too. 4:4:4 is shipped OFF (the leg was disabled while the leaves were
        // under-tooled) and `rect16_horz_444` reopens it for the study.
        let tune = crate::tuning::get();
        let ss444 = !self.ss420 && !self.ss422 && !self.mono;
        let rect_tier =
            self.speed == Speed::Slow || (tune.rect16_420_medium && self.speed == Speed::Medium);
        let horz_on = full_part_rdo
            && (self.ss422
                || (self.ss420 && rect_tier)
                || (ss444 && tune.rect16_horz_444 && rect_tier))
            && HORZ_ENABLED.load(std::sync::atomic::Ordering::Relaxed);

        // Anchor the perceptual R-D scale ONCE at the parent 16x16 region and use
        // it for every candidate so all costs share one lambda axis.
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let part_lam = self.mlam() * prdo;
        let chroma_cost = |part| {
            if full_part_rdo {
                self.rd_cost_chroma_partition(px, py, 16, part, prdo)
            } else {
                0.0
            }
        };
        #[allow(clippy::if_same_then_else)]
        let part_budget = if self.ss420
            && self.aq.enabled
            && !self.mono
            && self.aq.base_q <= 20
            && self.speed == Speed::Slow
        {
            0
        } else if crate::tuning::get().part_budget_444_only
            && (self.ss420 || self.ss422 || self.mono)
        {
            0
        } else if (self.base_q_idx as u32) < crate::tuning::get().part_budget_qmin {
            // The non-square families only earn their rate at low quality; see
            // `part_budget_qmin` in src/tuning.rs for the measured band.
            0
        } else {
            self.speed.partition_refine_budget()
        };
        // The 4x4-cell moment grid drives both the non-square Top-K ranking
        // and the SPLIT breakout below, so compute it at most once.
        let cells = (part_budget != 0 || split_breakout_k(self.speed).is_finite())
            .then(|| block_moment_grid_16x16(&self.src[0], self.w, px, py));
        let guided = cells.as_ref().map(|c| {
            (
                self.partition_model_cost16(Part16::None, part_lam, c),
                self.partition_model_cost16(Part16::Split, part_lam, c),
            )
        });
        let gk = crate::tuning::guided16_k(self.speed);
        // MUST be false when the min-size floor is on: the floor forces
        // `skip_split`, so also skipping NONE would leave BOTH legs at INFINITY
        // and no valid partition at this node.
        let skip_none = !crate::tuning::min_size_16(self.speed)
            && match guided {
                Some((m_none, m_split)) if gk > 0.0 => m_split * gk < m_none,
                _ => false,
            };

        let (htr16, hbl16) = self.leaf_edge_flags(px, py, 16, thr, lhb);
        let rd_none_unbiased = if skip_none {
            f32::INFINITY
        } else {
            self.rd_cost_square(px, py, 16, htr16, hbl16, prdo)
                + if coupled_square {
                    0.0
                } else {
                    chroma_cost(Part16::None)
                }
        };
        let selection_only_bias_444 = self.top_band() && !self.ss420 && !self.ss422;
        let none_bias = if self.ss420 && self.aq.enabled && !self.mono && self.speed == Speed::Slow
        {
            1.0 + (none16_top_bias_420() - 1.0) * seam_t(self.aq.base_q, SEAM_W_420)
        } else if self.ss422 && self.aq.enabled && !self.mono && self.speed == Speed::Slow {
            1.0 + (none16_top_bias_422() - 1.0) * seam_t(self.aq.base_q, SEAM_W_422)
        } else if selection_only_bias_444 {
            none16_top_bias_444()
        } else {
            1.0
        };
        let rd_none =
            rd_none_unbiased * none_bias + rate_cost(part_lam, self.part16_rate(x8, y8, 0));
        // At the 4:2:0 top band, admitting the 16-level non-square families
        // costs substantially more rate than it recovers in distortion. This
        // leaves the SPLIT anchor (and therefore its four 8x8 children plus
        // their optional SPLIT4 search) fully enabled; transform splitting is
        // controlled independently and is unchanged.
        // Minimum-block-size floor: never descend 16 -> 8. Unlike `fixed_size_*`
        // (which pins EVERY level to one uniform size and skips the 64/32
        // decisions too) this keeps those, so block size still varies across
        // 64/32/16 — while cutting the expensive part: four 8x8 leaves plus
        // SPLIT4 plus the SPLIT chroma partition. NONE is still priced
        // normally, so the parent receives a real cost.
        let floor16 = crate::tuning::min_size_16(self.speed);
        let skip_split = floor16
            || match guided {
                Some((m_none, m_split))
                    if split_breakout_k(self.speed).is_finite() && !skip_none =>
                {
                    m_split > m_none * split_breakout_k(self.speed)
                }
                _ => false,
            };
        let split4_ok = !self.mono && SPLIT4_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let best8 = |bx: usize, by: usize| -> f32 {
            let (cthr, clhb) = Self::child_edge_flags(bx - px, by - py, thr, lhb);
            let (htr8, hbl8) = self.leaf_edge_flags(bx, by, 8, cthr, clhb);
            let none8 = self.rd_cost_square(bx, by, 8, htr8, hbl8, prdo);
            if split4_ok {
                let mut s4 =
                    self.rd_cost_split4_luma(bx, by, trellis_lambda() * prdo, self.mlam() * prdo);
                if self.ss422 && s4.is_finite() {
                    // 4:2:2 pair conversion the luma-only split4 price misses:
                    // splitting turns the child's one 4x8 chroma into two
                    // paired 4x4s.
                    s4 += self.chroma_partition_weight_at(bx, by, 8, 8)
                        * (self.rd_cost_chroma_block(bx / 2, by, 4, 4, prdo)
                            + self.rd_cost_chroma_block(bx / 2, by + 4, 4, 4, prdo)
                            - self.rd_cost_chroma_block(bx / 2, by, 4, 8, prdo));
                }
                none8.min(s4)
            } else {
                none8
            }
        };
        // On breakout none of the 8x8 children are priced at all; the A/B legs
        // consume them too, so they go with it (as in aom, which clears
        // `do_square_split` and `do_rectangular_split` together).
        let split_signal = rate_cost(part_lam, self.part16_rate(x8, y8, 3));
        let mut split_pruned = skip_split;
        let mut square8 = [0.0f32; 4];
        if !split_pruned {
            let mut luma_lower_bound = split_signal;
            for (i, (bx, by)) in [(px, py), (px + 8, py), (px, py + 8), (px + 8, py + 8)]
                .into_iter()
                .enumerate()
            {
                square8[i] = best8(bx, by);
                luma_lower_bound += square8[i];
                // With no non-square refinements, the remaining SPLIT work is
                // non-negative chroma plus non-negative child costs. Once its
                // luma lower bound cannot beat NONE, later children cannot
                // affect the decision and need not be searched.
                if part_budget == 0 && luma_lower_bound >= rd_none {
                    split_pruned = true;
                    break;
                }
            }
        }
        let rd_split = if split_pruned {
            f32::INFINITY
        } else {
            split_signal + square8.iter().sum::<f32>() + chroma_cost(Part16::Split)
        };
        let rect_bias = if self.ss422 {
            tune.rect16_bias_422
        } else {
            rect16_bias()
        };
        // VERT mirrors HORZ, but is SPEC-FORBIDDEN in 4:2:2 (and off at 444).
        // AV1 conformance prohibits ALL vertical partition types (VERT/
        // VERT_A/VERT_B/VERT_4) at 4:2:2 regardless of block size.
        // VERT is SPEC-FORBIDDEN at 4:2:2 (all vertical partition types, every
        // block size) — that gate is conformance, never a tuning knob.
        let vert_on = RECT16_VERT_ENABLED
            && full_part_rdo
            && ((self.ss420 && rect_tier) || (ss444 && tune.rect16_vert_444 && rect_tier))
            && VERT_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let quad4_on = quad4_enabled(self.ss420) && full_part_rdo;

        // Source-domain partition staging: rank every legal non-square family
        // cheaply, then run the unchanged equipped-leaf + chroma RD pipeline on
        // a speed-sized Top-K. NONE and SPLIT remain fully evaluated anchors.
        let mut modeled_parts = FixedList::<(f32, Part16), 8>::new((f32::INFINITY, Part16::None));
        if part_budget != 0 {
            let cells = cells
                .as_ref()
                .expect("cells computed when part_budget != 0");
            let mut admit = |part| {
                modeled_parts.push((self.partition_model_cost16(part, part_lam, cells), part));
            };
            if horz_on {
                admit(Part16::Horz);
            }
            if vert_on {
                admit(Part16::Vert);
            }
            if self.ss420 && full_part_rdo {
                for part in [Part16::HorzA, Part16::HorzB, Part16::VertA, Part16::VertB] {
                    admit(part);
                }
            }
            if quad4_on {
                admit(Part16::Horz4);
                if !self.ss422 {
                    admit(Part16::Vert4);
                }
            }
            modeled_parts
                .as_mut_slice()
                .sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            modeled_parts.truncate(part_budget);
        }
        let refine_part = |part| modeled_parts.iter().any(|&(_, p)| p == part);

        // Chroma distortion/rate is non-negative. Price the luma leg first and
        // avoid coding chroma when that lower bound already loses to a complete
        // square candidate. This is exact branch-and-bound, not a heuristic.
        let mut best_complete = rd_none.min(rd_split);
        let rd_horz = if !horz_on || !refine_part(Part16::Horz) {
            f32::INFINITY
        } else {
            let luma = self.rd_cost_horz(px, py, prdo);
            if luma * rect_bias >= best_complete {
                f32::INFINITY
            } else {
                (luma + chroma_cost(Part16::Horz)) * rect_bias
            }
        };
        best_complete = best_complete.min(rd_horz);

        let rd_vert = if !vert_on || !refine_part(Part16::Vert) {
            f32::INFINITY
        } else {
            let luma = self.rd_cost_vert(px, py, prdo);
            if luma * rect_bias >= best_complete {
                f32::INFINITY
            } else {
                (luma + chroma_cost(Part16::Vert)) * rect_bias
            }
        };
        best_complete = best_complete.min(rd_vert);
        let asym_signal = rate_cost(part_lam, self.partition_signal_bits());
        let mut finish_asym = |luma: f32, part: Part16| {
            if luma >= best_complete {
                f32::INFINITY
            } else {
                let cost = luma + self.rd_cost_chroma_partition(px, py, 16, part, prdo);
                best_complete = best_complete.min(cost);
                cost
            }
        };
        let (rd_horz_a, rd_horz_b, rd_vert_a, rd_vert_b) =
            if self.ss420 && full_part_rdo && !skip_split {
                let rd_horz_a = if refine_part(Part16::HorzA) {
                    let luma = asym_signal
                        + square8[0]
                        + square8[1]
                        + self.rd_cost_rect16_dependent(px, py + 8, false, true, false, prdo);
                    finish_asym(luma, Part16::HorzA)
                } else {
                    f32::INFINITY
                };
                let rd_horz_b = if refine_part(Part16::HorzB) {
                    let luma = asym_signal
                        + self.rd_cost_rect16_leaf(px, py, false, prdo)
                        + square8[2]
                        + square8[3];
                    finish_asym(luma, Part16::HorzB)
                } else {
                    f32::INFINITY
                };
                let rd_vert_a = if refine_part(Part16::VertA) {
                    let luma = asym_signal
                        + square8[0]
                        + square8[2]
                        + self.rd_cost_rect16_dependent(px + 8, py, true, false, true, prdo);
                    finish_asym(luma, Part16::VertA)
                } else {
                    f32::INFINITY
                };
                let rd_vert_b = if refine_part(Part16::VertB) {
                    let luma = asym_signal
                        + self.rd_cost_rect16_leaf(px, py, true, prdo)
                        + square8[1]
                        + square8[3];
                    finish_asym(luma, Part16::VertB)
                } else {
                    f32::INFINITY
                };
                (rd_horz_a, rd_horz_b, rd_vert_a, rd_vert_b)
            } else {
                (f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY)
            };

        let q4b = quad4_bias();
        let rd_h4 = if quad4_on && refine_part(Part16::Horz4) {
            let luma = self.rd_cost_quad16(px, py, false, prdo);
            if luma * q4b >= best_complete {
                f32::INFINITY
            } else {
                (luma + chroma_cost(Part16::Horz4)) * q4b
            }
        } else {
            f32::INFINITY
        };
        best_complete = best_complete.min(rd_h4);
        let rd_v4 = if quad4_on && !self.ss422 && refine_part(Part16::Vert4) {
            let luma = self.rd_cost_quad16(px, py, true, prdo);
            if luma * q4b >= best_complete {
                f32::INFINITY
            } else {
                (luma + chroma_cost(Part16::Vert4)) * q4b
            }
        } else {
            f32::INFINITY
        };
        // IntraBC candidate: whole-16 exact-copy (all planes priced inside
        // rd_cost_intrabc, so no chroma_cost leg is added here).
        let rd_ibc = if self.allow_intrabc {
            self.rd_cost_intrabc(px, py, 16, prdo)
                .unwrap_or(f32::INFINITY)
        } else {
            f32::INFINITY
        };
        let cands = [
            (rd_ibc, Part16::Intrabc),
            (rd_none, Part16::None),
            (rd_split, Part16::Split),
            (rd_horz, Part16::Horz),
            (rd_vert, Part16::Vert),
            (rd_horz_a, Part16::HorzA),
            (rd_horz_b, Part16::HorzB),
            (rd_vert_a, Part16::VertA),
            (rd_vert_b, Part16::VertB),
            (rd_h4, Part16::Horz4),
            (rd_v4, Part16::Vert4),
        ];
        let chosen = cands
            .into_iter()
            .fold((f32::INFINITY, Part16::Split), |b, c| {
                if c.0 < b.0 { c } else { b }
            });
        let chosen_cost = if chosen.1 == Part16::None && none_bias != 1.0 && !self.ss420 {
            rd_none_unbiased
        } else {
            chosen.0
        };
        (chosen.1, chosen_cost)
    }

    fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    fn with_dispatch(
        mut self,
        dct: DctDispatch,
        idct: IdctDispatch,
        intrapred: IntraPredDispatch,
        kmeans: KmeansDispatch,
        rd: crate::rd_sse::RdDispatch,
    ) -> Self {
        self.dct = dct;
        self.idct = idct;
        self.intrapred = intrapred;
        self.kmeans = kmeans;
        self.rd = rd;
        self
    }

    fn with_updating_cdf(mut self, updating_cdf: bool) -> Self {
        self.updating_cdf = updating_cdf;
        self.enc.updating_cdf = updating_cdf;
        self
    }

    /// Frame-level screen-content tool search (palette). See
    /// [`crate::EncodeConfig::with_screen_content`].
    fn with_screen_content(mut self, enabled: bool) -> Self {
        self.screen_content = enabled;
        self
    }

    /// Whether a palette candidate may be trialled at all: the speed tier must
    /// admit the search AND the frame must be flagged as screen content.
    #[inline]
    pub(crate) fn try_palette(&self) -> bool {
        self.screen_content && self.speed.try_palette()
    }

    fn with_intrabc(mut self, enabled: bool) -> Self {
        self.allow_intrabc = enabled;
        self
    }

    /// Turn on variance-based adaptive quantization for this tile. `ref_act` is
    /// the tile mean activity (see [`tile_ref_activity`]); the running
    /// `CurrentQIndex` starts at the frame base. Must be paired with a frame
    /// header that signals `delta_q_present = 1` and the matching `delta_q_res`.
    fn enable_aq(&mut self, base_q: u8, ref_act: f32, vb: &VarianceBoost) {
        self.aq = AqCtx {
            enabled: true,
            base_q,
            res_log2: AQ_RES_LOG2,
            cur_qidx: base_q as i32,
            prev_qidx: base_q as i32,
            ref_act,
            ref_pick: tile_ref_picked(
                &self.src[0],
                self.w,
                self.w,
                self.h,
                vb.octile.clamp(1, 8),
                self.bd,
            ),
            read_deltas: false,
            pending: 0,
            vb_enabled: vb.enabled,
            vb_octile: vb.octile.clamp(1, 8),
            vb_strength: vb.strength.max(0.0),
            vb_boost_only: vb.boost_only,
            dark: vb.dark,
            base_shift: vb.base_shift,
        };
    }

    fn emit_lr_sb(&mut self, sb_x: usize, sb_y: usize) {
        let Some(unit) = self.wiener else {
            return;
        };
        emit_lr_sb_syms(
            &mut self.enc,
            &mut self.cdfs.wiener_restore,
            &mut self.lr_ref_v,
            &mut self.lr_ref_h,
            &unit,
            self.frame_x0,
            self.frame_y0,
            self.frame_w,
            self.frame_h,
            sb_x,
            sb_y,
        );
    }

    /// The AQ target qindex for the superblock at `(sb_x, sb_y)` — a pure
    /// function of the SOURCE pixels and frame-level AQ constants (no running
    /// state), which is what makes the whole per-SB qindex sequence
    /// precomputable by [`precompute_aq_grid`] before any coding starts.
    fn aq_sb_target(&self, sb_x: usize, sb_y: usize) -> i32 {
        let base_q = self.aq.base_q as i32;
        // Dark-structured-detail protection: an extra qindex reduction for dark SBs
        // carrying real cross-scale structure. Independent of the Variance Boost, it
        // is combined with the boost by `max` (see below), matching `av2/aq.rs`. The
        // AV1 luma plane is native-depth `i32`, normalized to 8-bit range so the
        // calibrated dark thresholds apply at every bit depth.
        let dark_scale = 1.0 / (1u32 << (self.bd - 8)) as f32;
        let dark = crate::aq_common::dark_protection(
            &self.aq.dark,
            base_q,
            &self.src[0],
            self.w,
            sb_y,
            sb_x,
            self.w,
            self.h,
            dark_scale,
        );
        if self.aq.vb_enabled {
            // Variance Boost: pick the representative 8x8 variance at the configured
            // octile and map it to a qindex delta off the frame base. ref_act (the
            // tile mean whole-SB log-variance) anchors the coarse side, matching
            // `av2/aq.rs::AqState::per_sb`.
            let mut subvars = [0f32; 64];
            let filled = aq_sb_subblock_variances(
                &self.src[0],
                self.w,
                sb_y,
                sb_x,
                self.w,
                self.h,
                &mut subvars,
            );
            if filled == 0 {
                base_q
            } else {
                let picked = crate::aq_common::sb_octile_variance(&mut subvars, self.aq.vb_octile);
                let var_scale = 1.0 / (1u32 << (2 * (self.bd - 8))) as f32;
                // Field-retune ramp: legacy constants at the near-lossless
                // top, full retune from qindex 96 (see variance_boost_delta).
                let mid_t = ((base_q as f32 - 48.0) / 48.0).clamp(0.0, 1.0);
                let vb_delta = crate::aq_common::variance_boost_delta(
                    picked * var_scale,
                    self.aq.ref_act,
                    self.aq.ref_pick,
                    self.aq.vb_strength,
                    self.aq.vb_boost_only,
                    mid_t,
                    0.0,
                );
                // protection = max(flat-region boost, dark boost); keep the coarsen
                // (positive) side of the variance boost only when neither fires.
                let flat_boost = (-vb_delta).max(0);
                let protection = flat_boost.max(dark);
                let delta = if vb_delta > 0 && dark > 0 {
                    // Net-blend on CUT SBs (default since 2026-07-22; holdout
                    // 420 -0.40 / 444 -0.43 with the same-domain anchor):
                    // dark protection subtracts from the coarsen instead of
                    // replacing it — a strong texture cut on a mildly-dark SB
                    // keeps part of its cut. Boost SBs unchanged (override
                    // == min there anyway).
                    vb_delta - dark
                } else if protection > 0 {
                    -protection
                } else {
                    vb_delta
                };
                // Ceiling-program AQ cut gate: coarsening texture SBs at
                // near-lossless base q measured +0.39 SS2 of ceiling cost
                // (q100 kodak20). Ease the CUT side out toward qi=1; the
                // boost/protection side is untouched.
                let delta = if delta > 0 {
                    let g = 1.0 - top_ease() * top_ease_t(base_q as u8);
                    (delta as f32 * g).fast_round() as i32
                } else {
                    delta
                };
                let delta = if delta < 0 && self.aq.base_shift > 0 {
                    // Base-shift experiment: protection windows anchor to the
                    // PRE-shift base, so flat/dark SBs reach the effective q
                    // they had before the frame base rose. `delta` from the
                    // laws is a relative protection depth; measure it (and
                    // the clamp) from the original base, then re-express it
                    // relative to the shifted frame base.
                    let orig = base_q - self.aq.base_shift;
                    clamp_aq_delta_relative(orig, delta, self.bd) - self.aq.base_shift
                } else {
                    clamp_aq_delta_relative(base_q, delta, self.bd)
                };
                (base_q + delta).clamp(1, 255)
            }
        } else {
            let act = sb_activity(&self.src[0], self.w, sb_y, sb_x, self.w, self.h);
            let vb_delta = aq_target_qidx(base_q, act, self.aq.ref_act) - base_q;
            // Same combine as the VB path: dark protection can only deepen the
            // reduction, never override a coarsen with a smaller refine.
            let flat_boost = (-vb_delta).max(0);
            let protection = flat_boost.max(dark);
            let delta = if protection > 0 {
                -protection
            } else {
                vb_delta
            };
            let delta = clamp_aq_delta_relative(base_q, delta, self.bd);
            (base_q + delta).clamp(1, 255)
        }
    }

    /// Serial reference for the per-SB AQ state advance: compute the target,
    /// quantize it against the running `cur_qidx` accumulator, and retarget the
    /// quantizers. Kept as the bit-exactness oracle for [`precompute_aq_grid`]
    /// (see the `aq_grid_matches_serial` unit test); the SB loop itself
    /// consumes precomputed [`AqCell`]s via [`aq_begin_sb_cell`].
    #[cfg_attr(not(test), allow(dead_code))]
    fn aq_begin_sb(&mut self, sb_x: usize, sb_y: usize) {
        if !self.aq.enabled {
            return;
        }
        let target = self.aq_sb_target(sb_x, sb_y);
        let step = 1i32 << self.aq.res_log2;
        let steps = (((target - self.aq.cur_qidx) as f32) / step as f32)
            .fast_round()
            .clamp(-(AQ_MAX_STEPS as f32), AQ_MAX_STEPS as f32) as i32;
        // The decoder applies Clip3(1,255, cur + steps*step); mirror it exactly so
        // both sides agree on the new qindex even when the clamp bites.
        let newq = (self.aq.cur_qidx + steps * step).clamp(1, 255);
        self.aq_begin_sb_cell(&AqCell {
            prev: self.aq.cur_qidx as u8,
            newq: newq as u8,
            steps,
        });
    }

    /// Precompute the whole tile's per-SB AQ sequence: a cheap serial raster
    /// pass mirroring `aq_begin_sb`'s accumulator bit-exactly (the ONLY running
    /// state is `cur_qidx`). The wavefront decides SBs out of raster order, so
    /// it reads `grid[row * sb_cols + col]` instead of advancing the
    /// accumulator; the serial emit uses `.steps` for the `read_delta_qindex`
    /// token. Empty when AQ is off.
    fn precompute_aq_grid(&self) -> Vec<AqCell> {
        if !self.aq.enabled {
            return Vec::new();
        }
        let rows = self.h.div_ceil(64);
        let cols = self.w.div_ceil(64);
        let mut grid = Vec::with_capacity(rows * cols);
        let step = 1i32 << self.aq.res_log2;
        let mut cur = self.aq.cur_qidx;
        for r in 0..rows {
            for c in 0..cols {
                let target = self.aq_sb_target(c * 64, r * 64);
                let steps = (((target - cur) as f32) / step as f32)
                    .fast_round()
                    .clamp(-(AQ_MAX_STEPS as f32), AQ_MAX_STEPS as f32)
                    as i32;
                let newq = (cur + steps * step).clamp(1, 255);
                grid.push(AqCell {
                    prev: cur as u8,
                    newq: newq as u8,
                    steps,
                });
                cur = newq;
            }
        }
        grid
    }

    /// Apply one precomputed [`AqCell`] at the start of a superblock: advance
    /// the accumulator to the cell's qindex, arm the delta-q token, and
    /// retarget the quantizers.
    fn aq_begin_sb_cell(&mut self, cell: &AqCell) {
        // Install the precomputed raster-chain state ABSOLUTELY. Deriving from
        // the worker's own accumulator was worker-schedule-dependent wherever
        // the qindex clamp broke the shared step lattice (q99 corruption).
        self.aq.prev_qidx = cell.prev as i32;
        let newq = cell.newq as i32;
        self.aq.cur_qidx = newq;
        self.aq.pending = cell.steps;
        self.aq.read_deltas = true;
        self.quant = Quant::new_with_qm(newq as u8, self.bd, self.quant.qm_level());
        // The chroma-DC delta is a frame-level constant (DeltaQUDc, derived from
        // the frame base_q_idx and signaled once in the header). The decoder
        // forms the chroma-DC qindex as CurrentQIndex + DeltaQUDc, so apply the
        // frame-level delta to the AQ-adjusted qindex — not chroma_dc_delta(newq).
        let frame_dc_delta = chroma_dc_delta(self.aq.base_q, self.chroma_sub());
        let frame_ac_delta = chroma_ac_delta(self.aq.base_q, self.chroma_sub());
        self.cquant = Quant::new_chroma_with_delta_qm(
            newq as u8,
            frame_dc_delta,
            frame_ac_delta,
            self.bd,
            self.cquant.qm_level(),
        );
    }

    /// Undo the tentative AQ retarget when a whole 64x64 block is skipped.
    /// AV1 returns before `read_delta_qindex` in that case, leaving the decoder's
    /// `CurrentQIndex` unchanged; the encoder must quantize the next SB from the
    /// same state.
    fn aq_cancel_skipped_sb(&mut self) {
        if !self.aq.enabled || !self.aq.read_deltas {
            return;
        }
        let q = self.aq.prev_qidx as u8;
        self.aq.cur_qidx = self.aq.prev_qidx;
        self.aq.pending = 0;
        self.quant = Quant::new_with_qm(q, self.bd, self.quant.qm_level());
        self.cquant = Quant::new_chroma_with_delta_qm(
            q,
            chroma_dc_delta(self.aq.base_q, self.chroma_sub()),
            chroma_ac_delta(self.aq.base_q, self.chroma_sub()),
            self.bd,
            self.cquant.qm_level(),
        );
    }

    /// Emit the intra `tx_depth` symbol (spec `read_tx_size`) for a `w`x`h`-px
    /// luma block at pixel (px, py), choosing `depth` size-halvings from the
    /// block's max rect TX, then update the per-4x4 TX context rows with the
    /// CHOSEN TX dims. Mirrors dav1d decode.c (intra `b->tx` read — coded for
    /// every intra luma block > BLOCK_4X4, including skip blocks) + env.h
    /// `get_tx_ctx`: cdf `txsz[t_dim.max - 1][ctx]`, `min(max, 2) + 1` symbols.
    /// BLOCK_4X4 codes no symbol; callers use [`Self::tx_ctx_update4`].
    fn code_tx_depth(&mut self, px: usize, py: usize, w: usize, h: usize, depth: usize) {
        let l2 = |d: usize| -> i8 {
            match d {
                4 => 0,
                8 => 1,
                16 => 2,
                32 => 3,
                _ => 4,
            }
        };
        let (max_lw, max_lh) = (l2(w), l2(h));
        let cat = max_lw.max(max_lh) as usize - 1; // t_dim.max - 1 (max == max(lw, lh))
        let (bx4, by4) = (px / 4, py / 4);
        let ctx = (self.l_tx[by4] >= max_lh) as usize + (self.a_tx[bx4] >= max_lw) as usize;
        self.enc.encode_symbol(depth, &mut self.cdfs.txsz[cat][ctx]);
        // Chosen TX dims: each depth step goes to `t_dim.sub` — the square of
        // the smaller dim for rects, then square halvings (floor TX_4X4).
        let (mut lw, mut lh) = (max_lw, max_lh);
        for _ in 0..depth {
            if lw != lh {
                let m = lw.min(lh);
                lw = m;
                lh = m;
            } else {
                lw = (lw - 1).max(0);
                lh = lw;
            }
        }
        self.a_tx[bx4..bx4 + (w / 4).max(1)].fill(lw);
        self.l_tx[by4..by4 + (h / 4).max(1)].fill(lh);
    }

    /// TX context update for a BLOCK_4X4 (no `tx_depth` symbol; TX is 4x4).
    fn tx_ctx_update4(&mut self, px: usize, py: usize) {
        self.a_tx[px / 4] = 0;
        self.l_tx[py / 4] = 0;
    }

    /// Emit a block's skip flag, then the `read_cdef()` / `read_delta_qindex()`
    /// per-SB tokens in spec `intra_frame_mode_info()` order. `read_cdef()`
    /// itself emits nothing here (cdef_bits is decided frame-level after the
    /// tiles are coded); it records the trace point where a replay interleaves
    /// the per-unit `cdef_idx` literal — at the SB's FIRST non-skip block, which
    /// is exactly when the decoder's `cdef_idx[r1][c1] == -1 && !skip` fires.
    fn code_skip_and_sb_tokens(&mut self, block_skip: bool, sctx: usize) {
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        if !block_skip && !self.cdef_point_marked {
            self.cdef_point_marked = true;
            self.enc.trace_cdef_mark();
        }
        self.code_delta_q_if_armed();
        if self.allow_intrabc {
            self.enc.encode_symbol(0, &mut self.cdfs.intrabc);
        }
    }

    /// Whole-superblock counterpart of [`Self::code_skip_and_sb_tokens`]. AV1's
    /// `read_delta_qindex()` returns immediately when `MiSize == sbSize && skip`,
    /// so a skipped 64x64 block in a 64x64-superblock frame must not carry the
    /// otherwise-per-SB delta-Q symbol.
    fn code_skip_and_sb_tokens_64(&mut self, block_skip: bool, sctx: usize) {
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        if block_skip {
            self.aq_cancel_skipped_sb();
            if self.allow_intrabc {
                self.enc.encode_symbol(0, &mut self.cdfs.intrabc);
            }
            return;
        }
        if !self.cdef_point_marked {
            self.cdef_point_marked = true;
            self.enc.trace_cdef_mark();
        }
        self.code_delta_q_if_armed();
        if self.allow_intrabc {
            self.enc.encode_symbol(0, &mut self.cdfs.intrabc);
        }
    }

    fn code_delta_q_if_armed(&mut self) {
        if !self.aq.read_deltas {
            return;
        }
        self.aq.read_deltas = false;
        let m = self.aq.pending.unsigned_abs() as i32;
        const DELTA_Q_SMALL: usize = 3;
        if m < DELTA_Q_SMALL as i32 {
            self.enc.encode_symbol(m as usize, &mut self.cdfs.delta_q);
        } else {
            self.enc
                .encode_symbol(DELTA_Q_SMALL, &mut self.cdfs.delta_q);
            // m = abs_bits + (1 << rem) + 1, rem = floor(log2(m-1)) >= 1.
            let v = (m - 1) as u32;
            let rem = 31 - v.leading_zeros(); // floor(log2(v))
            let abs_bits = v - (1 << rem);
            self.enc.encode_literal(rem - 1, 3); // delta_q_rem_bits (pre-increment)
            self.enc.encode_literal(abs_bits, rem); // delta_q_abs_bits
        }
        if m != 0 {
            self.enc.encode_literal((self.aq.pending < 0) as u32, 1); // sign
        }
    }

    /// Emit-time masking weight: the same per-block activity scale as
    /// [`Self::perceptual_rd_scale`], raised to the emit strength.
    fn emit_mlam(&self, px: usize, py: usize, dim: usize) -> f32 {
        self.mlam() * self.emit_scale(px, py, dim)
    }

    fn emit_prdo(&self, px: usize, py: usize, dim: usize) -> f32 {
        self.emit_scale(px, py, dim)
    }

    fn emit_scale(&self, px: usize, py: usize, dim: usize) -> f32 {
        self.perceptual_rd_scale(px, py, dim)
    }

    pub(crate) fn var_over_qstep2(&self, px: usize, py: usize, dim: usize) -> f32 {
        let bw = dim.min(self.w.saturating_sub(px));
        let bh = dim.min(self.h.saturating_sub(py));
        if bw == 0 || bh == 0 {
            return f32::INFINITY;
        }
        let (sum, sum_sq) = block_moments_i32(&self.src[0], self.w, px, py, bw, bh);
        let n = (bw * bh) as f32;
        let mean = sum as f32 / n;
        let var = (sum_sq as f32 / n - mean * mean).max(0.0);
        let q = self.quant.ac_q() as f32;
        var / (q * q).max(1.0)
    }

    fn perceptual_rd_scale(&self, px: usize, py: usize, dim: usize) -> f32 {
        let mut k = if !self.ss420 && !self.ss422 && !self.mono {
            prdo_k_444()
        } else {
            prdo_k()
        };
        if k == 0.0 {
            return 1.0;
        }
        if self.ss422 {
            let g = f_fmlaf(-top_ease(), top_ease_t(self.aq.base_q), 1.0);
            k *= g;
            if k == 0.0 {
                return 1.0;
            }
        }
        if !self.ss422 && !self.mono {
            const Q1: f32 = 100.0;
            const Q0: f32 = 50.0;
            const FADE_SCALE: f32 = 1.0 / (Q1 - Q0);
            let mut t = ((self.aq.base_q as f32 - Q0) * FADE_SCALE).clamp(0.0, 1.0);
            let floor = if self.ss420 { 0.4 } else { 0.0 };
            t = f_fmlaf(1.0 - floor, t, floor);
            k *= t;
            if k == 0.0 {
                return 1.0;
            }
        }
        // Masking reference: blended frame/superblock activity. See
        // `local_ref_blend()`.
        let refa = {
            let a = local_ref_blend();
            let g = self.aq.ref_act;
            if a == 0.0 {
                g
            } else {
                let sbx = (px & !63) as u32;
                let sby = (py & !63) as u32;
                let c = self.sb_act_cache.get();
                let sa = if c.0 == sbx && c.1 == sby {
                    c.2
                } else {
                    let v = crate::coder::sb_activity(
                        &self.src[0],
                        self.w,
                        sby as usize,
                        sbx as usize,
                        self.w,
                        self.h,
                    );
                    self.sb_act_cache.set((sbx, sby, v));
                    v
                };
                g * (1.0 - a) + sa * a
            }
        };
        if refa <= 0.0 {
            return 1.0;
        }
        let bw = dim.min(self.w.saturating_sub(px));
        let bh = dim.min(self.h.saturating_sub(py));
        if bw == 0 || bh == 0 {
            return 1.0;
        }
        let (sum, sum_sq) = block_moments_i32(&self.src[0], self.w, px, py, bw, bh);
        let n = (bw * bh) as f32;
        let mean = sum as f32 / n;
        let var = (sum_sq as f32 / n - mean * mean).max(0.0);
        let act = dirty_log1pf(var);
        let c = prdo_clamp();
        let exponent = (k * (act - refa)).clamp(-std::f32::consts::LN_2, std::f32::consts::LN_2);
        dirty_exp2f(exponent * std::f32::consts::LOG2_E).clamp(1.0 / c, prdo_upper_clamp())
    }
}

const EXP2_TABLE_SIZE: usize = 64;

#[repr(align(64))]
struct Exp2Table([u32; EXP2_TABLE_SIZE]);

// 2^((i - 32) / 64), rounded to f32.
#[rustfmt::skip]
static EXP2F_TABLE: Exp2Table = Exp2Table([
    0x3F3504F3, 0x3F36FD92, 0x3F38FBAF, 0x3F3AFF5B, 0x3F3D08A4, 0x3F3F179A, 0x3F412C4D, 0x3F4346CD,
    0x3F45672A, 0x3F478D75, 0x3F49B9BE, 0x3F4BEC15, 0x3F4E248C, 0x3F506334, 0x3F52A81E, 0x3F54F35B,
    0x3F5744FD, 0x3F599D16, 0x3F5BFBB8, 0x3F5E60F5, 0x3F60CCDF, 0x3F633F89, 0x3F65B907, 0x3F68396A,
    0x3F6AC0C7, 0x3F6D4F30, 0x3F6FE4BA, 0x3F728177, 0x3F75257D, 0x3F77D0DF, 0x3F7A83B3, 0x3F7D3E0C,
    0x3F800000, 0x3F8164D2, 0x3F82CD87, 0x3F843A29, 0x3F85AAC3, 0x3F871F62, 0x3F88980F, 0x3F8A14D5,
    0x3F8B95C2, 0x3F8D1ADF, 0x3F8EA43A, 0x3F9031DC, 0x3F91C3D3, 0x3F935A2B, 0x3F94F4F0, 0x3F96942D,
    0x3F9837F0, 0x3F99E046, 0x3F9B8D3A, 0x3F9D3EDA, 0x3F9EF532, 0x3FA0B051, 0x3FA27043, 0x3FA43516,
    0x3FA5FED7, 0x3FA7CD94, 0x3FA9A15B, 0x3FAB7A3A, 0x3FAD583F, 0x3FAF3B79, 0x3FB123F6, 0x3FB311C4,
]);

#[inline(always)]
pub(crate) fn dirty_exp2f(d: f32) -> f32 {
    let redux = f32::from_bits(0x4b400000) / EXP2_TABLE_SIZE as f32;

    let ui = (d + redux).to_bits();
    let mut i0 = ui.wrapping_add(EXP2_TABLE_SIZE as u32 / 2);
    let k = i0 / EXP2_TABLE_SIZE as u32;
    i0 &= EXP2_TABLE_SIZE as u32 - 1;
    let uf = f32::from_bits(ui) - redux;

    let z0 = f32::from_bits(EXP2F_TABLE.0[i0 as usize]);
    let f = d - uf;

    // Sollya: fpminimax(2^x, 2, [|single...|], [-1/128; 1/128], relative);
    let mut u = 0.24022668600082397;
    u = f_fmlaf(u, f, 0.693149745464325);
    u *= f;

    let i2 = f32::from_bits(k.wrapping_add(0x7f) << 23);
    f_fmlaf(u, z0, z0) * i2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perceptual_exp2_matches_system_exp() {
        const SAMPLES: usize = 65_536;
        let lo = -std::f32::consts::LN_2;
        let hi = std::f32::consts::LN_2;
        let mut max_relative_error = 0.0f32;
        for i in 0..=SAMPLES {
            let x = lo + (hi - lo) * (i as f32 / SAMPLES as f32);
            let expected = x.exp();
            let relative_error =
                ((dirty_exp2f(x * std::f32::consts::LOG2_E) - expected) / expected).abs();
            max_relative_error = max_relative_error.max(relative_error);
        }
        assert!(
            max_relative_error <= 3.0e-7,
            "maximum relative error {max_relative_error:e} exceeded the bound"
        );
    }
}
