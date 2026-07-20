/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/// Plane pixel element: the lossy coder's planes are `u16` end to end (one
/// machinery for 8/10/12-bit); the lossless tile keeps `i32` buffers. All
/// readers widen through `i32::From` immediately.
pub(crate) trait Pel: Copy + Into<i32> {
    #[inline(always)]
    fn widen(self) -> i32 {
        self.into()
    }
    /// Store an in-range pixel value (callers clamp to [0, (1<<bd)-1] first).
    fn narrow(v: i32) -> Self;
}
impl Pel for u16 {
    #[inline(always)]
    fn narrow(v: i32) -> Self {
        v as u16
    }
}
impl Pel for i32 {
    #[inline(always)]
    fn narrow(v: i32) -> Self {
        v
    }
}

pub(crate) const DC_PRED: usize = 0;

/// Non-directional modes need no angle-delta symbol.
pub(crate) const SMOOTH_PRED: usize = 9;
pub(crate) const SMOOTH_V_PRED: usize = 10;
pub(crate) const SMOOTH_H_PRED: usize = 11;
pub(crate) const PAETH_PRED: usize = 12;
/// Directional modes added in this increment (axis-aligned only, at
/// `angle_delta = 0`): pure vertical / horizontal copy. They sit in the
/// directional range 1..=8 (`VERT_LEFT_PRED`), so they emit an `angle_delta`
/// symbol — but at delta 0 (angle 90/180) the decoder maps them straight to the
/// plain copy predictors, with no top-right/bottom-left edge extension.
pub(crate) const V_PRED: usize = 1;
pub(crate) const H_PRED: usize = 2;
/// Z2 diagonal modes (down-right directions).
pub(crate) const D135_PRED: usize = 4;
pub(crate) const D113_PRED: usize = 5;
pub(crate) const D157_PRED: usize = 6;
pub(crate) const VERT_LEFT_PRED: usize = 8;
/// Z1 diagonals (up-right): `D45` (45 deg) and `D67` (= `VERT_LEFT_PRED`, 67 deg).
/// They read the top row extended to the right (top-right samples). Z3 diagonal
/// (down-left): `D203` (203 deg), reading the left column extended downward
/// (bottom-left samples). These need the neighbor-availability derivation
/// (dav1d's intra-edge tree) and the extended reference arrays.
pub(crate) const D45_PRED: usize = 3;
pub(crate) const D203_PRED: usize = 7;
/// Chroma-from-luma. Signalled as a `uv_mode` symbol; its tx-type is **not** in
/// `txtp_from_uvmode`, so it defaults to `DCT_DCT` — i.e. CfL needs no ADST.
pub(crate) const CFL_PRED: usize = 13;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FilterIntraMode {
    Dc = 0,
    Vertical = 1,
    Horizontal = 2,
    D157 = 3,
    Paeth = 4,
}

pub(crate) static FILTER_INTRA_MODES: [FilterIntraMode; 5] = [
    FilterIntraMode::Dc,
    FilterIntraMode::Vertical,
    FilterIntraMode::Horizontal,
    FilterIntraMode::D157,
    FilterIntraMode::Paeth,
];

type PredFn = fn(usize, usize, &[i32], &[i32], &mut [i32]);
type PaethFn = fn(usize, usize, &[i32], &[i32], i32, &mut [i32]);
type DcPredFn = fn(&[u16], usize, usize, usize, usize, usize, i32) -> i32;
type DrPredictFn = fn(DrPrediction, &[i32], &[i32], &mut [i32]);
type EdgeConv5Fn = fn(&[i32], &[i32; 5], &mut [i32]);
type FilterIntraCellsFn = fn(&mut [[i32; 33]; 33], &[[i8; 7]; 8], usize, usize, i32);
type CflAc444U16Fn = fn(&[u16], usize, usize, &mut [i32]);
type CflAcSubU16Fn = fn(&[u16], usize, usize, usize, bool, bool, &mut [i32]);
type CflPredFn = fn(&mut [i32], &[i32], i32, i32, u8);
type CflBestAlphaU16Fn = fn(&[i32], &[u16], i32, usize, u8) -> i32;

pub(crate) const DR_ZONE1: u8 = 1;
pub(crate) const DR_ZONE2: u8 = 2;
pub(crate) const DR_ZONE3: u8 = 3;
pub(crate) const DR_EDGE_ORIGIN: i32 = 2;

/// Geometry shared by the scalar, NEON and AVX2 whole-block directional
/// predictors. `above` and `left` passed to [`DrPredictFn`] start at logical
/// edge index -2, so negative zone-2 references remain ordinary slice loads.
#[derive(Clone, Copy)]
pub(crate) struct DrPrediction {
    pub(crate) zone: u8,
    pub(crate) bw: usize,
    pub(crate) bh: usize,
    pub(crate) edge_len: usize,
    pub(crate) dx: i32,
    pub(crate) dy: i32,
    pub(crate) up_above: i32,
    pub(crate) up_left: i32,
}

/// Per-encode intra-prediction dispatch. CPU feature detection happens while
/// the encoding context is built, never in a block/row predictor.
#[derive(Clone, Copy)]
pub(crate) struct IntraPredDispatch {
    dc: DcPredFn,
    vertical: PredFn,
    horizontal: PredFn,
    smooth: PredFn,
    smooth_v: PredFn,
    smooth_h: PredFn,
    paeth: PaethFn,
    dr_predict: DrPredictFn,
    edge_conv5: EdgeConv5Fn,
    filter_intra_cells: FilterIntraCellsFn,
    cfl_ac_444_u16: CflAc444U16Fn,
    cfl_ac_sub_u16: CflAcSubU16Fn,
    cfl_pred: CflPredFn,
    cfl_best_alpha_u16: CflBestAlphaU16Fn,
}

macro_rules! dc_pred_method {
    ($name:ident, $width:expr, $height:expr) => {
        #[inline]
        pub(crate) fn $name(
            &self,
            recon: &[u16],
            stride: usize,
            ox: usize,
            oy: usize,
            bit_depth: i32,
        ) -> i32 {
            self.dc_pred(recon, stride, ox, oy, $width, $height, bit_depth)
        }
    };
}

impl IntraPredDispatch {
    pub(crate) const fn scalar() -> Self {
        Self {
            dc: dc_pred_u16_scalar,
            vertical: vertical_scalar,
            horizontal: horizontal_scalar,
            smooth: smooth_scalar,
            smooth_v: smooth_v_scalar,
            smooth_h: smooth_h_scalar,
            paeth: paeth_scalar,
            dr_predict: dr_predict_scalar,
            edge_conv5: edge_conv5_scalar,
            filter_intra_cells: scalar_filter_intra_cells,
            cfl_ac_444_u16: cfl_ac_444_u16_scalar,
            cfl_ac_sub_u16: cfl_ac_sub_u16_scalar,
            cfl_pred: cfl_pred_scalar,
            cfl_best_alpha_u16: cfl_best_alpha_u16_scalar,
        }
    }

    pub(crate) fn selected() -> Self {
        #[allow(unused_mut)]
        let mut dispatch = Self::scalar();
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            dispatch.dc = dc_pred_neon_dispatch;
            dispatch.vertical = vertical_neon_dispatch;
            dispatch.horizontal = horizontal_neon_dispatch;
            dispatch.smooth = smooth_neon_dispatch;
            dispatch.smooth_v = smooth_v_neon_dispatch;
            dispatch.smooth_h = smooth_h_neon_dispatch;
            dispatch.paeth = paeth_neon_dispatch;
            dispatch.dr_predict = dr_predict_neon_dispatch;
            dispatch.edge_conv5 = edge_conv5_neon_dispatch;
            dispatch.filter_intra_cells = filter_intra_cells_neon_dispatch;
            dispatch.cfl_ac_444_u16 = cfl_ac_444_u16_neon_dispatch;
            dispatch.cfl_ac_sub_u16 = cfl_ac_sub_u16_neon_dispatch;
            dispatch.cfl_pred = cfl_pred_neon_dispatch;
            dispatch.cfl_best_alpha_u16 = cfl_best_alpha_u16_neon_dispatch;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            dispatch.dc = dc_pred_avx2_dispatch;
            dispatch.vertical = vertical_avx2_dispatch;
            dispatch.horizontal = horizontal_avx2_dispatch;
            dispatch.smooth = smooth_avx2_dispatch;
            dispatch.smooth_v = smooth_v_avx2_dispatch;
            dispatch.smooth_h = smooth_h_avx2_dispatch;
            dispatch.paeth = paeth_avx2_dispatch;
            dispatch.dr_predict = dr_predict_avx2_dispatch;
            dispatch.edge_conv5 = edge_conv5_avx2_dispatch;
            dispatch.filter_intra_cells = filter_intra_cells_avx2_dispatch;
            dispatch.cfl_ac_444_u16 = cfl_ac_444_u16_avx2_dispatch;
            dispatch.cfl_ac_sub_u16 = cfl_ac_sub_u16_avx2_dispatch;
            dispatch.cfl_pred = cfl_pred_avx2_dispatch;
            dispatch.cfl_best_alpha_u16 = cfl_best_alpha_u16_avx2_dispatch;
        }
        dispatch
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn dc_pred(
        &self,
        recon: &[u16],
        stride: usize,
        ox: usize,
        oy: usize,
        width: usize,
        height: usize,
        bit_depth: i32,
    ) -> i32 {
        (self.dc)(recon, stride, ox, oy, width, height, bit_depth)
    }

    #[inline]
    pub(crate) fn cfl_pred(&self, dst: &mut [i32], ac: &[i32], dc: i32, alpha: i32, bd: u8) {
        debug_assert!(ac.len() >= dst.len());
        (self.cfl_pred)(dst, &ac[..dst.len()], dc, alpha, bd);
    }

    #[inline]
    pub(crate) fn cfl_ac_444(&self, luma_rec: &[u16], w: usize, h: usize, ac: &mut [i32]) {
        let n = w * h;
        debug_assert!(luma_rec.len() >= n);
        debug_assert!(ac.len() >= n);
        (self.cfl_ac_444_u16)(&luma_rec[..n], w, h, &mut ac[..n]);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn cfl_ac_sub(
        &self,
        luma_rec: &[u16],
        lstride: usize,
        cw: usize,
        ch: usize,
        ss_hor: bool,
        ss_ver: bool,
        ac: &mut [i32],
    ) {
        debug_assert!(ac.len() >= cw * ch);
        (self.cfl_ac_sub_u16)(luma_rec, lstride, cw, ch, ss_hor, ss_ver, ac);
    }

    #[inline]
    pub(crate) fn cfl_best_alpha(&self, ac: &[i32], src: &[u16], dc: i32, n: usize, bd: u8) -> i32 {
        debug_assert!(ac.len() >= n);
        debug_assert!(src.len() >= n);
        (self.cfl_best_alpha_u16)(&ac[..n], &src[..n], dc, n, bd)
    }

    dc_pred_method!(dc_pred_4x4, 4, 4);
    dc_pred_method!(dc_pred_8x4, 8, 4);
    dc_pred_method!(dc_pred_4x8, 4, 8);
    dc_pred_method!(dc_pred_8x8, 8, 8);
    dc_pred_method!(dc_pred_16x8, 16, 8);
    dc_pred_method!(dc_pred_8x16, 8, 16);
    dc_pred_method!(dc_pred_16x16, 16, 16);
    dc_pred_method!(dc_pred_32x16, 32, 16);
    dc_pred_method!(dc_pred_16x32, 16, 32);
    dc_pred_method!(dc_pred_32x32, 32, 32);

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_nd<P: Pel>(
        &self,
        mode: usize,
        recon: &[P],
        stride: usize,
        ox: usize,
        oy: usize,
        bw: usize,
        bh: usize,
        have_tr: bool,
        have_bl: bool,
        fw: usize,
        fh: usize,
        filter_type: bool,
        out: &mut [i32],
        bd: u8,
    ) {
        self.predict_nd_ad(
            mode,
            0,
            recon,
            stride,
            ox,
            oy,
            bw,
            bh,
            have_tr,
            have_bl,
            fw,
            fh,
            filter_type,
            out,
            bd,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_nd_ad<P: Pel>(
        &self,
        mode: usize,
        angle_delta: i32,
        recon: &[P],
        stride: usize,
        ox: usize,
        oy: usize,
        bw: usize,
        bh: usize,
        have_tr: bool,
        have_bl: bool,
        fw: usize,
        fh: usize,
        filter_type: bool,
        out: &mut [i32],
        bd: u8,
    ) {
        intra_predict_nd_ad_impl(
            self,
            mode,
            angle_delta,
            recon,
            stride,
            ox,
            oy,
            bw,
            bh,
            have_tr,
            have_bl,
            fw,
            fh,
            filter_type,
            true,
            out,
            bd,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn filter_predict<P: Pel>(
        &self,
        mode: FilterIntraMode,
        recon: &[P],
        stride: usize,
        ox: usize,
        oy: usize,
        width: usize,
        height: usize,
        out: &mut [i32],
        bit_depth: u8,
    ) {
        filter_intra_predict_impl(
            self, mode, recon, stride, ox, oy, width, height, out, bit_depth,
        );
    }
}

fn selected_intra_dispatch() -> &'static IntraPredDispatch {
    static DISPATCH: std::sync::OnceLock<IntraPredDispatch> = std::sync::OnceLock::new();
    DISPATCH.get_or_init(IntraPredDispatch::selected)
}

const EDGE_ORIGIN: usize = 2;
const EDGE_CAPACITY: usize = 132;

#[derive(Clone)]
struct IntraEdge {
    samples: [i32; EDGE_CAPACITY],
}

struct DirectionalScratch {
    above: IntraEdge,
    left_edge: IntraEdge,
    filter_source: [i32; EDGE_CAPACITY],
    upsample_input: [i32; EDGE_CAPACITY],
}

impl DirectionalScratch {
    const fn new() -> Self {
        Self {
            above: IntraEdge::new(),
            left_edge: IntraEdge::new(),
            filter_source: [0; EDGE_CAPACITY],
            upsample_input: [0; EDGE_CAPACITY],
        }
    }
}

thread_local! {
    static DIRECTIONAL_SCRATCH: std::cell::RefCell<DirectionalScratch> =
        const { std::cell::RefCell::new(DirectionalScratch::new()) };
    static FILTER_INTRA_SCRATCH: std::cell::RefCell<[[i32; 33]; 33]> =
        const { std::cell::RefCell::new([[0; 33]; 33]) };
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn filter_intra_predict<P: Pel>(
    mode: FilterIntraMode,
    recon: &[P],
    stride: usize,
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    out: &mut [i32],
    bit_depth: u8,
) {
    selected_intra_dispatch()
        .filter_predict(mode, recon, stride, ox, oy, width, height, out, bit_depth);
}

#[allow(clippy::too_many_arguments)]
fn filter_intra_predict_impl<P: Pel>(
    dispatch: &IntraPredDispatch,
    mode: FilterIntraMode,
    recon: &[P],
    stride: usize,
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    out: &mut [i32],
    bit_depth: u8,
) {
    debug_assert_eq!(width & 3, 0);
    debug_assert_eq!(height & 1, 0);
    debug_assert!(width <= 32 && height <= 32);
    debug_assert_eq!(out.len(), width * height);

    let base = 1i32 << (bit_depth - 1);
    let have_top = oy > 0;
    let have_left = ox > 0;
    let corner = match (have_top, have_left) {
        (true, true) => recon[(oy - 1) * stride + ox - 1].widen(),
        (true, false) => recon[(oy - 1) * stride + ox].widen(),
        (false, true) => recon[oy * stride + ox - 1].widen(),
        (false, false) => base,
    };
    FILTER_INTRA_SCRATCH.with_borrow_mut(|buf| {
        // The filter reads only the initialized top row, left column, and
        // preceding cells that it writes during this traversal.
        buf[0][0] = corner;
        for x in 0..width {
            buf[0][x + 1] = if have_top {
                recon[(oy - 1) * stride + ox + x].widen()
            } else if have_left {
                recon[oy * stride + ox - 1].widen()
            } else {
                base - 1
            };
        }
        for y in 0..height {
            buf[y + 1][0] = if have_left {
                recon[(oy + y) * stride + ox - 1].widen()
            } else if have_top {
                recon[(oy - 1) * stride + ox].widen()
            } else {
                base + 1
            };
        }

        let taps = &crate::tables::INTRA_FILTER_TAPS[mode as usize];
        let max_sample = (1 << bit_depth) - 1;
        (dispatch.filter_intra_cells)(buf, taps, width, height, max_sample);
        for y in 0..height {
            out[y * width..(y + 1) * width].copy_from_slice(&buf[y + 1][1..width + 1]);
        }
    });
}

/// Reference 4x2-cell filter-intra pass (the SIMD kernels are bit-exact with
/// this; it also runs on targets without one).
fn scalar_filter_intra_cells(
    buf: &mut [[i32; 33]; 33],
    taps: &[[i8; 7]; 8],
    width: usize,
    height: usize,
    max_sample: i32,
) {
    for r in (1..=height).step_by(2) {
        for c in (1..=width).step_by(4) {
            let p = [
                buf[r - 1][c - 1],
                buf[r - 1][c],
                buf[r - 1][c + 1],
                buf[r - 1][c + 2],
                buf[r - 1][c + 3],
                buf[r][c - 1],
                buf[r + 1][c - 1],
            ];
            for (k, filter) in taps.iter().enumerate() {
                let sum = filter
                    .iter()
                    .zip(p)
                    .map(|(&tap, sample)| tap as i32 * sample)
                    .sum::<i32>();
                let value =
                    ((sum + 8) >> crate::tables::INTRA_FILTER_SCALE_BITS).clamp(0, max_sample);
                buf[r + (k >> 2)][c + (k & 3)] = value;
            }
        }
    }
}

impl IntraEdge {
    const fn new() -> Self {
        Self {
            samples: [0; EDGE_CAPACITY],
        }
    }

    #[inline]
    fn get(&self, index: i32) -> i32 {
        self.samples[(index + EDGE_ORIGIN as i32) as usize]
    }

    #[inline]
    fn set(&mut self, index: i32, value: i32) {
        self.samples[(index + EDGE_ORIGIN as i32) as usize] = value;
    }

    /// Contiguous samples starting at logical `index` (>= -EDGE_ORIGIN).
    #[inline]
    fn from(&self, index: i32) -> &[i32] {
        &self.samples[(index + EDGE_ORIGIN as i32) as usize..]
    }

    /// Mutable window of `len` samples starting at logical `index`.
    #[inline]
    fn slice_mut(&mut self, index: i32, len: usize) -> &mut [i32] {
        let s = (index + EDGE_ORIGIN as i32) as usize;
        &mut self.samples[s..s + len]
    }
}

/// Widening copy (u16/i16 recon row -> i32 edge). The slice form
/// autovectorizes; the old per-element `IntraEdge::set` loop did not.
#[inline]
fn widen_into<T: Copy + Into<i32>>(dst: &mut [i32], src: &[T]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.into();
    }
}

fn vertical_scalar(bw: usize, bh: usize, top: &[i32], _left: &[i32], out: &mut [i32]) {
    for row in out.chunks_exact_mut(bw).take(bh) {
        row.copy_from_slice(&top[..bw]);
    }
}

fn horizontal_scalar(bw: usize, bh: usize, _top: &[i32], left: &[i32], out: &mut [i32]) {
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        row.fill(lv);
    }
}

fn dr_interp_scalar(e: &[i32], step: usize, shift: i32, out: &mut [i32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let b = i * step;
        *o = (e[b] * (32 - shift) + e[b + 1] * shift + 16) >> 5;
    }
}

#[inline]
fn dr_edge(edge: &[i32], index: i32) -> i32 {
    edge[(index + DR_EDGE_ORIGIN) as usize]
}

fn dr_predict_scalar(p: DrPrediction, above: &[i32], left: &[i32], out: &mut [i32]) {
    debug_assert!(out.len() >= p.bw * p.bh);
    match p.zone {
        DR_ZONE1 => {
            let max_base_x = (p.edge_len as i32 - 1) << p.up_above;
            let frac_bits = 6 - p.up_above;
            let base_inc = 1usize << p.up_above;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                let xpos = p.dx * (y as i32 + 1);
                let shift = ((xpos << p.up_above) & 0x3f) >> 1;
                let base = xpos >> frac_bits;
                let n_fit = if base >= max_base_x {
                    0
                } else {
                    ((max_base_x - base + base_inc as i32 - 1) / base_inc as i32).min(p.bw as i32)
                        as usize
                };
                dr_interp_scalar(
                    &above[(base + DR_EDGE_ORIGIN) as usize..],
                    base_inc,
                    shift,
                    &mut row[..n_fit],
                );
                row[n_fit..].fill(dr_edge(above, max_base_x));
            }
        }
        DR_ZONE3 => {
            let max_base_y = (p.edge_len as i32 - 1) << p.up_left;
            let frac_bits = 6 - p.up_left;
            let base_inc = 1i32 << p.up_left;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                for (x, dst) in row.iter_mut().enumerate() {
                    let ypos = p.dy * (x as i32 + 1);
                    let shift = ((ypos << p.up_left) & 0x3f) >> 1;
                    let base = (ypos >> frac_bits) + y as i32 * base_inc;
                    *dst = if base >= max_base_y {
                        dr_edge(left, max_base_y)
                    } else {
                        (dr_edge(left, base) * (32 - shift) + dr_edge(left, base + 1) * shift + 16)
                            >> 5
                    };
                }
            }
        }
        DR_ZONE2 => {
            let frac_bits_y = 6 - p.up_left;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                let t = (y as i32 + 1) * p.dx - 64;
                let x0 = if t <= 0 {
                    0
                } else {
                    (((t + 63) >> 6) as usize).min(p.bw)
                };
                for (x, dst) in row[..x0].iter_mut().enumerate() {
                    let ypos = ((y as i32) << 6) - (x as i32 + 1) * p.dy;
                    let base = ypos >> frac_bits_y;
                    let shift = ((ypos * (1 << p.up_left)) & 0x3f) >> 1;
                    *dst =
                        (dr_edge(left, base) * (32 - shift) + dr_edge(left, base + 1) * shift + 16)
                            >> 5;
                }
                if x0 < p.bw {
                    let xpos = ((x0 as i32) << 6) - (y as i32 + 1) * p.dx;
                    let base = xpos >> (6 - p.up_above);
                    let shift = ((xpos * (1 << p.up_above)) & 0x3f) >> 1;
                    dr_interp_scalar(
                        &above[(base + DR_EDGE_ORIGIN) as usize..],
                        1usize << p.up_above,
                        shift,
                        &mut row[x0..],
                    );
                }
            }
        }
        _ => unreachable!("invalid directional prediction zone"),
    }
}

fn edge_conv5_scalar(win: &[i32], kernel: &[i32; 5], out: &mut [i32]) {
    for (t, o) in out.iter_mut().enumerate() {
        let sum: i32 = kernel
            .iter()
            .enumerate()
            .map(|(j, &tap)| tap * win[t + j])
            .sum();
        *o = (sum + 8) >> 4;
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dc_pred_neon_dispatch(
    recon: &[u16],
    stride: usize,
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    unsafe { crate::neon::dc_pred_neon(recon, stride, ox, oy, width, height, bit_depth) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn vertical_neon_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::neon::vertical_neon(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn horizontal_neon_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::neon::horizontal_neon(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn smooth_neon_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::neon::smooth_neon(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn smooth_v_neon_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::neon::smooth_v_neon(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn smooth_h_neon_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::neon::smooth_h_neon(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn paeth_neon_dispatch(
    bw: usize,
    bh: usize,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [i32],
) {
    unsafe { crate::neon::paeth_neon(bw, bh, top, left, corner, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dr_predict_neon_dispatch(p: DrPrediction, above: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::neon::dr_predict_neon(p, above, left, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn edge_conv5_neon_dispatch(win: &[i32], kernel: &[i32; 5], out: &mut [i32]) {
    unsafe { crate::neon::edge_conv5_neon(win, kernel, out) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn filter_intra_cells_neon_dispatch(
    buf: &mut [[i32; 33]; 33],
    taps: &[[i8; 7]; 8],
    width: usize,
    height: usize,
    max_sample: i32,
) {
    unsafe { crate::neon::filter_intra_cells_neon(buf, taps, width, height, max_sample) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn cfl_pred_neon_dispatch(dst: &mut [i32], ac: &[i32], dc: i32, alpha: i32, bd: u8) {
    unsafe { crate::neon::cfl_pred_neon(dst, ac, dc, alpha, bd) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn cfl_ac_444_u16_neon_dispatch(luma_rec: &[u16], w: usize, h: usize, ac: &mut [i32]) {
    unsafe { crate::neon::cfl_ac_444_u16_neon(luma_rec, w, h, ac) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[allow(clippy::too_many_arguments)]
fn cfl_ac_sub_u16_neon_dispatch(
    luma_rec: &[u16],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    unsafe { crate::neon::cfl_ac_sub_u16_neon(luma_rec, lstride, cw, ch, ss_hor, ss_ver, ac) }
}
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn cfl_best_alpha_u16_neon_dispatch(ac: &[i32], src: &[u16], dc: i32, n: usize, bd: u8) -> i32 {
    unsafe { crate::neon::cfl_best_alpha_u16_neon(ac, src, dc, n, bd) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dc_pred_avx2_dispatch(
    recon: &[u16],
    stride: usize,
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    unsafe { crate::avx::dc_pred_avx2(recon, stride, ox, oy, width, height, bit_depth) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn vertical_avx2_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::avx::vertical_avx2(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn horizontal_avx2_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::avx::horizontal_avx2(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn smooth_avx2_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::avx::smooth_avx2(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn smooth_v_avx2_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::avx::smooth_v_avx2(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn smooth_h_avx2_dispatch(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::avx::smooth_h_avx2(bw, bh, top, left, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn paeth_avx2_dispatch(
    bw: usize,
    bh: usize,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [i32],
) {
    unsafe { crate::avx::paeth_avx2(bw, bh, top, left, corner, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dr_predict_avx2_dispatch(p: DrPrediction, above: &[i32], left: &[i32], out: &mut [i32]) {
    unsafe { crate::avx::dr_predict_avx2(p, above, left, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn edge_conv5_avx2_dispatch(win: &[i32], kernel: &[i32; 5], out: &mut [i32]) {
    unsafe { crate::avx::edge_conv5_avx2(win, kernel, out) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn filter_intra_cells_avx2_dispatch(
    buf: &mut [[i32; 33]; 33],
    taps: &[[i8; 7]; 8],
    width: usize,
    height: usize,
    max_sample: i32,
) {
    unsafe { crate::avx::filter_intra_cells_avx2(buf, taps, width, height, max_sample) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn cfl_pred_avx2_dispatch(dst: &mut [i32], ac: &[i32], dc: i32, alpha: i32, bd: u8) {
    unsafe { crate::avx::cfl_pred_avx2(dst, ac, dc, alpha, bd) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn cfl_ac_444_u16_avx2_dispatch(luma_rec: &[u16], w: usize, h: usize, ac: &mut [i32]) {
    unsafe { crate::avx::cfl_ac_444_u16_avx2(luma_rec, w, h, ac) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[allow(clippy::too_many_arguments)]
fn cfl_ac_sub_u16_avx2_dispatch(
    luma_rec: &[u16],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    unsafe { crate::avx::cfl_ac_sub_u16_avx2(luma_rec, lstride, cw, ch, ss_hor, ss_ver, ac) }
}
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn cfl_best_alpha_u16_avx2_dispatch(ac: &[i32], src: &[u16], dc: i32, n: usize, bd: u8) -> i32 {
    unsafe { crate::avx::cfl_best_alpha_u16_avx2(ac, src, dc, n, bd) }
}

#[inline]
pub(crate) fn is_smooth_mode(mode: usize) -> bool {
    matches!(mode, SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED)
}

fn intra_edge_filter_strength(w: usize, h: usize, filter_type: bool, delta: i32) -> u8 {
    let d = delta.abs();
    let wh = w + h;
    let mut strength = 0;
    if !filter_type {
        if wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if wh <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if wh <= 32 {
            if d >= 1 {
                strength = 1;
            }
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if d >= 1 {
            strength = 3;
        }
    } else if wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else if d >= 1 {
        strength = 3;
    }
    strength
}

fn filter_intra_edge(
    dispatch: &IntraPredDispatch,
    edge: &mut IntraEdge,
    size: usize,
    strength: u8,
    source: &mut [i32; EDGE_CAPACITY],
) {
    if strength == 0 {
        return;
    }
    static KERNELS: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    source[..size].copy_from_slice(&edge.from(-1)[..size]);
    let kernel = KERNELS[strength as usize - 1];
    // Output o = i - 1 reads source[o - 1 + j], j in 0..5, clamped to
    // [0, size - 1]; the clamp is inert for o in [1, size - 4] — that middle
    // run is a plain 5-tap convolution.
    let conv = |source: &[i32; EDGE_CAPACITY], o: usize| -> i32 {
        let mut sum = 0;
        for (j, &tap) in kernel.iter().enumerate() {
            let k = (o as i32 - 1 + j as i32).clamp(0, size as i32 - 1);
            sum += tap * source[k as usize];
        }
        (sum + 8) >> 4
    };
    let n = size - 1;
    let mid_end = size.saturating_sub(3).max(1).min(n);
    if mid_end > 1 {
        let win = &source[..size];
        (dispatch.edge_conv5)(win, &kernel, edge.slice_mut(1, mid_end - 1));
    }
    // Only the clamped boundaries cannot use the plain convolution kernel.
    edge.set(0, conv(source, 0));
    for o in mid_end..n {
        edge.set(o as i32, conv(source, o));
    }
}

fn filter_intra_corner(above: &mut IntraEdge, left: &mut IntraEdge) {
    let filtered = (5 * left.get(0) + 6 * above.get(-1) + 5 * above.get(0) + 8) >> 4;
    above.set(-1, filtered);
    left.set(-1, filtered);
}

#[inline]
fn use_intra_edge_upsample(w: usize, h: usize, filter_type: bool, delta: i32) -> bool {
    let d = delta.abs();
    d != 0 && d < 40 && if filter_type { w + h <= 8 } else { w + h <= 16 }
}

fn upsample_intra_edge(
    edge: &mut IntraEdge,
    num_px: usize,
    bd: u8,
    input: &mut [i32; EDGE_CAPACITY],
) {
    input[0] = edge.get(-1);
    input[1] = edge.get(-1);
    for i in 0..num_px {
        input[i + 2] = edge.get(i as i32);
    }
    input[num_px + 2] = edge.get(num_px as i32 - 1);
    edge.set(-2, input[0]);
    let max_sample = (1 << bd) - 1;
    for i in 0..num_px {
        let value = -input[i] + 9 * input[i + 1] + 9 * input[i + 2] - input[i + 3];
        edge.set(2 * i as i32 - 1, ((value + 8) >> 4).clamp(0, max_sample));
        edge.set(2 * i as i32, input[i + 2]);
    }
}

/// `default_cfl_sign_cdf` (libaom/dav1d): joint sign of the U/V alphas, 8 symbols.
pub(crate) static CFL_SIGN_CDF: [u16; 7] = [1418, 2123, 13340, 18405, 26972, 28343, 32294];
/// `default_cfl_alpha_cdf[6]`: per-plane alpha magnitude (1..=16 -> symbols 0..=15),
/// indexed by a context derived from the joint sign.
pub(crate) static CFL_ALPHA_CDF: [[u16; 15]; 6] = [
    [
        7637, 20719, 31401, 32481, 32657, 32688, 32692, 32696, 32700, 32704, 32708, 32712, 32716,
        32720, 32724,
    ],
    [
        14365, 23603, 28135, 31168, 32167, 32395, 32487, 32573, 32620, 32647, 32668, 32672, 32676,
        32680, 32684,
    ],
    [
        11532, 22380, 28445, 31360, 32349, 32523, 32584, 32649, 32673, 32677, 32681, 32685, 32689,
        32693, 32697,
    ],
    [
        26990, 31402, 32282, 32571, 32692, 32696, 32700, 32704, 32708, 32712, 32716, 32720, 32724,
        32728, 32732,
    ],
    [
        17248, 26058, 28904, 30608, 31305, 31877, 32126, 32321, 32394, 32464, 32516, 32560, 32576,
        32593, 32622,
    ],
    [
        14738, 21678, 25779, 27901, 29024, 30302, 30980, 31843, 32144, 32413, 32520, 32594, 32622,
        32656, 32660,
    ],
];

/// `dav1d_dr_intra_derivative[44]` — angle -> projection step (1/64 px). Indexed
/// `[(angle-90)>>1]` for the vertical step and `[(180-angle)>>1]` for the
/// horizontal step in the Z2 predictor.
pub(crate) static DR_INTRA_DERIVATIVE: [i32; 44] = [
    0, 1023, 0, 547, 372, 0, 0, 273, 215, 0, 178, 151, 0, 132, 116, 0, 102, 0, 90, 80, 0, 71, 64,
    0, 57, 51, 0, 45, 0, 40, 35, 0, 31, 27, 0, 23, 19, 0, 15, 0, 11, 0, 7, 3,
];

/// `dav1d_intra_mode_context` — maps an intra mode to its keyframe y-mode CDF
/// context (0..=4), used for both the above and left neighbors.
pub(crate) static INTRA_MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `dav1d_sm_weights` slice for a given block dimension (SMOOTH predictors).
pub(crate) fn sm_weights(n: usize) -> &'static [i32] {
    match n {
        4 => &[255, 149, 85, 64],
        8 => &[255, 197, 146, 105, 73, 50, 37, 32],
        16 => &[
            255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
        ],
        32 => &[
            255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59,
            52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
        ],
        _ => unreachable!("sm_weights size {}", n),
    }
}

pub(crate) fn cfl_ac_444<P: Pel>(luma_rec: &[P], w: usize, h: usize, ac: &mut [i32]) {
    let n = w * h;
    for (ac, luma) in ac[..n].iter_mut().zip(luma_rec[..n].iter()) {
        *ac = luma.widen() << 3;
    }
    let log2sz = w.trailing_zeros() + h.trailing_zeros();
    let mut sum: i64 = (1i64 << log2sz) >> 1;
    for ac in ac[..n].iter() {
        sum += *ac as i64;
    }
    let mean = (sum >> log2sz) as i32;
    for ac in ac[..n].iter_mut() {
        *ac -= mean;
    }
}

/// CfL luma-AC for subsampled chroma (dav1d `cfl_ac` with `ss_hor`/`ss_ver`).
/// `luma_rec` is the reconstructed luma block, stride `lstride`, covering the
/// chroma block of `(cw, ch)` samples. Each chroma position sums the covered
/// luma samples and is scaled by `1 << (1 + !ss_ver + !ss_hor)` (so 4:4:4 ->
/// `<< 3`, matching `cfl_ac_444`), then the block mean is removed.
pub(crate) fn cfl_ac_sub<P: Pel>(
    luma_rec: &[P],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    let shift = 1 + (!ss_ver as u32) + (!ss_hor as u32);
    let sx = ss_hor as usize;
    let sy = ss_ver as usize;
    for y in 0..ch {
        let ac = &mut ac[y * cw..y * cw + cw];
        for (x, dst) in ac[..cw].iter_mut().enumerate() {
            let ly = y << sy;
            let lx = x << sx;
            let mut s = luma_rec[ly * lstride + lx].widen();
            if ss_hor {
                s += luma_rec[ly * lstride + lx + 1].widen();
            }
            if ss_ver {
                s += luma_rec[(ly + 1) * lstride + lx].widen();
                if ss_hor {
                    s += luma_rec[(ly + 1) * lstride + lx + 1].widen();
                }
            }
            *dst = s << shift;
        }
    }
    let n = cw * ch;
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();
    let mut sum: i64 = (1i64 << log2sz) >> 1;
    for v in ac[..n].iter() {
        sum += *v as i64;
    }
    let mean = (sum >> log2sz) as i32;
    for v in ac[..n].iter_mut() {
        *v -= mean;
    }
}

fn cfl_ac_444_u16_scalar(luma_rec: &[u16], w: usize, h: usize, ac: &mut [i32]) {
    cfl_ac_444(luma_rec, w, h, ac);
}

#[allow(clippy::too_many_arguments)]
fn cfl_ac_sub_u16_scalar(
    luma_rec: &[u16],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    cfl_ac_sub(luma_rec, lstride, cw, ch, ss_hor, ss_ver, ac);
}

/// CfL prediction combine (dav1d `cfl_pred`): `dc + sign(diff)*((|diff|+32)>>6)`.
#[inline]
pub(crate) fn cfl_pred_pixel(dc: i32, ac: i32, alpha: i32, bd: u8) -> i32 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 32) >> 6;
    let s = if diff < 0 { -mag } else { mag };
    (dc + s).clamp(0, (1 << bd) - 1)
}

pub(crate) fn cfl_pred_scalar(dst: &mut [i32], ac: &[i32], dc: i32, alpha: i32, bd: u8) {
    for (dst, &ac) in dst.iter_mut().zip(ac) {
        *dst = cfl_pred_pixel(dc, ac, alpha, bd);
    }
}

/// Energy-minimising CfL alpha for one plane, in dav1d alpha units (the predictor
/// applies `alpha/64` after the <<3 AC scaling). Returns the best of the analytic
/// optimum and its +/-3 neighborhood by pre-quantization residual energy, clamped
/// to the signaled range [-16, 16] (0 means "CfL useless for this plane").
pub(crate) fn cfl_best_alpha<P: Pel>(ac: &[i32], src: &[P], dc: i32, n: usize, bd: u8) -> i32 {
    let mut num: i64 = 0;
    let mut den: i64 = 0;
    for (&src, &ac) in src[..n].iter().zip(ac[..n].iter()) {
        num += (src.widen() - dc) as i64 * ac as i64;
        den += ac as i64 * ac as i64;
    }
    if den == 0 {
        return 0;
    }
    let a0 = ((64 * num + (den >> 1) * num.signum()) / den).clamp(-16, 16) as i32;
    let mut best_a = 0i32;
    let mut best_e = i64::MAX;
    for cand in (a0 - 3)..=(a0 + 3) {
        if !(-16..=16).contains(&cand) {
            continue;
        }
        let mut e: i64 = 0;
        for (&src, &ac) in src[..n].iter().zip(ac[..n].iter()) {
            let d = (src.widen() - cfl_pred_pixel(dc, ac, cand, bd)) as i64;
            e += d * d;
        }
        if e < best_e {
            best_e = e;
            best_a = cand;
        }
    }
    best_a
}

fn cfl_best_alpha_u16_scalar(ac: &[i32], src: &[u16], dc: i32, n: usize, bd: u8) -> i32 {
    cfl_best_alpha(ac, src, dc, n, bd)
}

pub(crate) fn recon_add_pred<P: Pel>(dst: &mut [P], pred: &[i32], resid: &[i32], max: i32) {
    for ((d, &p), &r) in dst.iter_mut().zip(pred).zip(resid) {
        *d = P::narrow((p + r).clamp(0, max));
    }
}

pub(crate) fn recon_add_dc<P: Pel>(dst: &mut [P], dc: i32, resid: &[i32], max: i32) {
    for (d, &r) in dst.iter_mut().zip(resid) {
        *d = P::narrow((dc + r).clamp(0, max));
    }
}

/// Test entry point for intra prediction with an explicit AV1 `angle_delta`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn intra_predict_nd_ad<P: Pel>(
    mode: usize,
    angle_delta: i32,
    recon: &[P],
    stride: usize,
    ox: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    have_tr: bool,
    have_bl: bool,
    fw: usize,
    fh: usize,
    filter_type: bool,
    out: &mut [i32],
    bd: u8,
) {
    selected_intra_dispatch().predict_nd_ad(
        mode,
        angle_delta,
        recon,
        stride,
        ox,
        oy,
        bw,
        bh,
        have_tr,
        have_bl,
        fw,
        fh,
        filter_type,
        out,
        bd,
    )
}

/// Lossless counterpart of [`intra_predict_nd_ad`] reading reconstructed
/// integer samples directly from the encoder's `i16` source/recon plane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_predict_nd_ad_i16(
    mode: usize,
    angle_delta: i32,
    recon: &[i16],
    stride: usize,
    ox: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    have_tr: bool,
    have_bl: bool,
    fw: usize,
    fh: usize,
    filter_type: bool,
    out: &mut [i32],
    bd: u8,
) {
    intra_predict_nd_ad_impl(
        selected_intra_dispatch(),
        mode,
        angle_delta,
        recon,
        stride,
        ox,
        oy,
        bw,
        bh,
        have_tr,
        have_bl,
        fw,
        fh,
        filter_type,
        false,
        out,
        bd,
    )
}

#[allow(clippy::too_many_arguments)]
fn intra_predict_nd_ad_impl<T: Copy + Into<i32>>(
    dispatch: &IntraPredDispatch,
    mode: usize,
    angle_delta: i32,
    recon: &[T],
    stride: usize,
    ox: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    have_tr: bool,
    have_bl: bool,
    fw: usize,
    fh: usize,
    filter_type: bool,
    enable_edge_filter: bool,
    out: &mut [i32],
    bd: u8,
) {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let base = 1i32 << (bd - 1);
    DIRECTIONAL_SCRATCH.with_borrow_mut(|scratch| {
        let DirectionalScratch {
            above,
            left_edge,
            filter_source,
            upsample_input,
        } = scratch;
        let edge_len = bw + bh;
        if have_top {
            widen_into(
                above.slice_mut(0, bw),
                &recon[(oy - 1) * stride + ox..][..bw],
            );
        } else {
            let fill = if have_left {
                recon[oy * stride + ox - 1].into()
            } else {
                base - 1
            };
            above.slice_mut(0, edge_len).fill(fill);
        }
        if have_left {
            for j in 0..bh {
                left_edge.set(j as i32, recon[(oy + j) * stride + ox - 1].into());
            }
        } else {
            let fill = if have_top {
                recon[(oy - 1) * stride + ox].into()
            } else {
                base + 1
            };
            left_edge.slice_mut(0, edge_len).fill(fill);
        }
        let corner = if have_left {
            if have_top {
                recon[(oy - 1) * stride + ox - 1].into()
            } else {
                recon[oy * stride + ox - 1].into()
            }
        } else if have_top {
            recon[(oy - 1) * stride + ox].into()
        } else {
            base
        };
        above.set(-1, corner);
        left_edge.set(-1, corner);
        if have_top {
            let px_have = if have_tr {
                bw.min(bh).min(fw.saturating_sub(ox + bw))
            } else {
                0
            };
            widen_into(
                above.slice_mut(bw as i32, px_have),
                &recon[(oy - 1) * stride + ox + bw..][..px_have],
            );
            let fill = above.get((bw + px_have).saturating_sub(1) as i32);
            above
                .slice_mut((bw + px_have) as i32, edge_len - (bw + px_have))
                .fill(fill);
        }
        if have_left {
            let px_have = if have_bl {
                // Mirror of the top-right rule: min(w, h) real bottom-left pixels
                // (wrong for wide rects' zone-3: 8x4/16x8 D203).
                bw.min(bh).min(fh.saturating_sub(oy + bh))
            } else {
                0
            };
            for i in 0..px_have {
                left_edge.set(
                    (bh + i) as i32,
                    recon[(oy + bh + i) * stride + ox - 1].into(),
                );
            }
            let fill = left_edge.get((bh + px_have).saturating_sub(1) as i32);
            left_edge
                .slice_mut((bh + px_have) as i32, edge_len - (bh + px_have))
                .fill(fill);
        }

        let angle = match mode {
            V_PRED => 90 + angle_delta * 3,
            H_PRED => 180 + angle_delta * 3,
            D45_PRED => 45 + angle_delta * 3,
            VERT_LEFT_PRED => 67 + angle_delta * 3,
            D135_PRED => 135 + angle_delta * 3,
            D113_PRED => 113 + angle_delta * 3,
            D157_PRED => 157 + angle_delta * 3,
            D203_PRED => 203 + angle_delta * 3,
            _ => 0,
        };
        let directional = (V_PRED..=VERT_LEFT_PRED).contains(&mode);
        let mut upsample_above = false;
        let mut upsample_left = false;
        if directional && angle != 90 && angle != 180 {
            // Edge preparation above always materializes both references, filling a
            // missing side from the available side (or the bit-depth midpoint).
            // AV1 filters/upsamples those synthesized references too. Gating this
            // on physical neighbor availability leaves the edge unprocessed while
            // the projector still uses upsampled coordinates, which desynchronizes
            // zone-2 prediction at tile/frame boundaries.
            if enable_edge_filter {
                if angle > 90 && angle < 180 && edge_len >= 24 {
                    filter_intra_corner(above, left_edge);
                }
                let strength = intra_edge_filter_strength(bw, bh, filter_type, angle - 90);
                filter_intra_edge(
                    dispatch,
                    above,
                    bw + 1 + if angle < 90 { bh } else { 0 },
                    strength,
                    filter_source,
                );
                let strength = intra_edge_filter_strength(bh, bw, filter_type, angle - 180);
                filter_intra_edge(
                    dispatch,
                    left_edge,
                    bh + 1 + if angle > 180 { bw } else { 0 },
                    strength,
                    filter_source,
                );
            }
            upsample_above =
                enable_edge_filter && use_intra_edge_upsample(bw, bh, filter_type, angle - 90);
            upsample_left =
                enable_edge_filter && use_intra_edge_upsample(bh, bw, filter_type, angle - 180);
            if upsample_above {
                upsample_intra_edge(
                    above,
                    bw + if angle < 90 { bh } else { 0 },
                    bd,
                    upsample_input,
                );
            }
            if upsample_left {
                upsample_intra_edge(
                    left_edge,
                    bh + if angle > 180 { bw } else { 0 },
                    bd,
                    upsample_input,
                );
            }
        }
        let top = &above.from(0)[..edge_len];
        let left = &left_edge.from(0)[..edge_len];

        match mode {
            _ if directional && angle == 90 => {
                (dispatch.vertical)(bw, bh, top, left, out);
            }
            _ if directional && angle == 180 => {
                (dispatch.horizontal)(bw, bh, top, left, out);
            }
            _ if directional && angle < 90 => {
                let dx = DR_INTRA_DERIVATIVE[(angle >> 1) as usize];
                (dispatch.dr_predict)(
                    DrPrediction {
                        zone: DR_ZONE1,
                        bw,
                        bh,
                        edge_len,
                        dx,
                        dy: 0,
                        up_above: upsample_above as i32,
                        up_left: 0,
                    },
                    above.from(-DR_EDGE_ORIGIN),
                    left_edge.from(-DR_EDGE_ORIGIN),
                    out,
                );
            }
            _ if directional && angle > 180 => {
                let dy = DR_INTRA_DERIVATIVE[((270 - angle) >> 1) as usize];
                (dispatch.dr_predict)(
                    DrPrediction {
                        zone: DR_ZONE3,
                        bw,
                        bh,
                        edge_len,
                        dx: 0,
                        dy,
                        up_above: 0,
                        up_left: upsample_left as i32,
                    },
                    above.from(-DR_EDGE_ORIGIN),
                    left_edge.from(-DR_EDGE_ORIGIN),
                    out,
                );
            }
            _ if directional => {
                let dy = DR_INTRA_DERIVATIVE[((angle - 90) >> 1) as usize];
                let dx = DR_INTRA_DERIVATIVE[((180 - angle) >> 1) as usize];
                (dispatch.dr_predict)(
                    DrPrediction {
                        zone: DR_ZONE2,
                        bw,
                        bh,
                        edge_len,
                        dx,
                        dy,
                        up_above: upsample_above as i32,
                        up_left: upsample_left as i32,
                    },
                    above.from(-DR_EDGE_ORIGIN),
                    left_edge.from(-DR_EDGE_ORIGIN),
                    out,
                );
            }
            PAETH_PRED => (dispatch.paeth)(bw, bh, top, left, corner, out),
            SMOOTH_PRED => (dispatch.smooth)(bw, bh, top, left, out),
            SMOOTH_V_PRED => (dispatch.smooth_v)(bw, bh, top, left, out),
            SMOOTH_H_PRED => (dispatch.smooth_h)(bw, bh, top, left, out),
            _ => unreachable!("intra_predict_nd called with mode {}", mode),
        }
    });
}

fn paeth_scalar(bw: usize, _bh: usize, top: &[i32], left: &[i32], corner: i32, out: &mut [i32]) {
    for (y, orow) in out.chunks_exact_mut(bw).enumerate() {
        let lv = left[y];
        for (o, &tv) in orow.iter_mut().zip(top.iter()) {
            let b = lv + tv - corner;
            let (ld, td, cd) = ((lv - b).abs(), (tv - b).abs(), (corner - b).abs());
            *o = if ld <= td && ld <= cd {
                lv
            } else if td <= cd {
                tv
            } else {
                corner
            };
        }
    }
}

/// AV1 palette predictor.
///
/// `indices` contains two palette indices per byte: the low nibble selects the
/// even pixel and the high nibble selects the odd pixel.  AV1 palettes contain
/// at most eight entries, so only the low three bits of either nibble are used.
/// Index rows are tightly packed with a stride of `bw.div_ceil(2)`, while the
/// destination may have a larger, caller-provided stride.
pub(crate) fn palette_pred(
    dst: &mut [i32],
    dst_stride: usize,
    palette: &[i32],
    indices: &[u8],
    bw: usize,
    bh: usize,
) {
    assert!((2..=8).contains(&palette.len()));
    assert!(dst_stride >= bw);

    let index_stride = bw.div_ceil(2);
    assert!(indices.len() >= index_stride * bh);
    let dst_len = if bh == 0 {
        0
    } else {
        (bh - 1) * dst_stride + bw
    };
    assert!(dst.len() >= dst_len);

    for y in 0..bh {
        let dst_row = &mut dst[y * dst_stride..y * dst_stride + bw];
        let index_row = &indices[y * index_stride..(y + 1) * index_stride];
        let (pairs, remainder) = dst_row.as_chunks_mut::<2>();
        for (pair, &packed) in pairs.iter_mut().zip(index_row) {
            let lo = (packed & 7) as usize;
            let hi = ((packed >> 4) & 7) as usize;
            assert!(lo < palette.len() && hi < palette.len());
            pair[0] = palette[lo];
            pair[1] = palette[hi];
        }
        if let Some(last) = remainder.first_mut() {
            let index = (index_row[bw / 2] & 7) as usize;
            assert!(index < palette.len());
            *last = palette[index];
        }
    }
}

/// Expand one shared chroma palette index map into packed U/V predictors.
/// The palette has at most eight entries, so scalar indexed loads outperform
/// setting up a vector gather while keeping the state-machine code declarative.
pub(crate) fn palette_uv_pred(
    dst_u: &mut [i32],
    dst_v: &mut [i32],
    map: &[u8],
    colors_u: &[i32],
    colors_v: &[i32],
) {
    debug_assert!(dst_u.len() >= map.len());
    debug_assert!(dst_v.len() >= map.len());
    for ((dst_u, dst_v), &index) in dst_u.iter_mut().zip(dst_v).zip(map) {
        *dst_u = colors_u[index as usize];
        *dst_v = colors_v[index as usize];
    }
}

/// Scalar AV1 SMOOTH predictor (4-tap vertical+horizontal weighted blend).
fn smooth_scalar(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let (wv, wh) = (sm_weights(bh), sm_weights(bw));
    let (right, bottom) = (top[bw - 1], left[bh - 1]);
    for ((orow, &wvy), &lv) in out.chunks_exact_mut(bw).zip(wv.iter()).zip(left.iter()) {
        for (o, (&tv, &whx)) in orow.iter_mut().zip(top.iter().zip(wh.iter())) {
            let pred = wvy * tv + (256 - wvy) * bottom + whx * lv + (256 - whx) * right;
            *o = (pred + 256) >> 9;
        }
    }
}

/// Scalar AV1 SMOOTH_V predictor.
fn smooth_v_scalar(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let wv = sm_weights(bh);
    let bottom = left[bh - 1];
    for (orow, &wvy) in out.chunks_exact_mut(bw).zip(wv.iter()) {
        for (o, &tv) in orow.iter_mut().zip(top.iter()) {
            *o = (wvy * tv + (256 - wvy) * bottom + 128) >> 8;
        }
    }
}

/// Scalar AV1 SMOOTH_H predictor.
fn smooth_h_scalar(bw: usize, _bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let wh = sm_weights(bw);
    let right = top[bw - 1];
    for (orow, &lv) in out.chunks_exact_mut(bw).zip(left.iter()) {
        for (o, &whx) in orow.iter_mut().zip(wh.iter()) {
            *o = (whx * lv + (256 - whx) * right + 128) >> 8;
        }
    }
}

pub(crate) static ND_LUMA_MODES: [usize; 11] = [
    DC_PRED,
    V_PRED,
    H_PRED,
    D45_PRED,
    D135_PRED,
    D113_PRED,
    D157_PRED,
    D203_PRED,
    VERT_LEFT_PRED,
    SMOOTH_PRED,
    PAETH_PRED,
];

/// Candidate luma modes evaluated by the intra mode search.
pub(crate) fn nd_modes() -> &'static [usize] {
    &ND_LUMA_MODES
}

/// Reduced candidate set for the fast RDO path (`speed >= 1`): keep DC, the
/// planar-like SMOOTH, and PAETH, and drop the SMOOTH_V/SMOOTH_H variants
/// (their wins over SMOOTH are rare and small). Mirrors libaom's intra-mode
/// pruning at higher `--cpu-used`.
pub(crate) fn fast_nd_modes() -> &'static [usize] {
    static FAST: [usize; 3] = [DC_PRED, SMOOTH_PRED, PAETH_PRED];
    &FAST
}

pub(crate) fn dc_pred<P: Pel>(
    recon: &[P],
    stride: usize,
    ox: usize,
    oy: usize,
    w: usize,
    h: usize,
    bd: i32,
) -> i32 {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let above_sum = if have_top {
        recon[(oy - 1) * stride + ox..][..w]
            .iter()
            .map(|&v| v.widen())
            .sum::<i32>()
    } else {
        0
    };
    let left_sum = if have_left {
        recon[oy * stride + ox - 1..]
            .iter()
            .step_by(stride)
            .take(h)
            .map(|&v| v.widen())
            .sum::<i32>()
    } else {
        0
    };
    dc_pred_from_sum(above_sum + left_sum, w, h, have_top, have_left, bd)
}

pub(crate) fn dc_pred_from_sum(
    edge_sum: i32,
    w: usize,
    h: usize,
    have_top: bool,
    have_left: bool,
    bd: i32,
) -> i32 {
    match (have_top, have_left) {
        (true, true) => {
            let mut s = ((w + h) >> 1) as i32 + edge_sum;
            s >>= (w + h).trailing_zeros();
            if w != h {
                let mult: u32 = if w > 2 * h || h > 2 * w {
                    0x3334
                } else {
                    0x5556
                };
                s = (((s as u32) * mult) >> 16) as i32;
            }
            s
        }
        (true, false) => ((w >> 1) as i32 + edge_sum) >> w.trailing_zeros(),
        (false, true) => ((h >> 1) as i32 + edge_sum) >> h.trailing_zeros(),
        (false, false) => 1 << (bd - 1),
    }
}

fn dc_pred_u16_scalar(
    recon: &[u16],
    stride: usize,
    ox: usize,
    oy: usize,
    w: usize,
    h: usize,
    bd: i32,
) -> i32 {
    dc_pred(recon, stride, ox, oy, w, h, bd)
}

#[cfg(test)]
mod intra_edge_tests {
    use super::*;

    /// Pseudo-random edge/reference generator shared by the kernel tests.
    fn lcg(state: &mut u32) -> i32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state % 1024) as i32
    }

    #[test]
    fn cfl_ac_444_dispatch_matches_scalar_for_shapes_depths_and_tails() {
        let dispatch = IntraPredDispatch::selected();
        let mut state = 0xa409_3822u32;
        for &(w, h) in &[
            (1usize, 1usize),
            (3, 5),
            (4, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (32, 16),
            (16, 32),
            (32, 32),
        ] {
            let n = w * h;
            for &bd in &[8u8, 10, 12] {
                let max = (1 << bd) - 1;
                let src: Vec<u16> = (0..n)
                    .map(|i| match i % 5 {
                        0 => 0,
                        1 => max as u16,
                        _ => lcg(&mut state).clamp(0, max) as u16,
                    })
                    .collect();
                let mut scalar = vec![0x55aa_i32; n + 3];
                let mut selected = scalar.clone();
                cfl_ac_444(&src, w, h, &mut scalar);
                dispatch.cfl_ac_444(&src, w, h, &mut selected);
                assert_eq!(selected, scalar, "w={w} h={h} bd={bd}");
            }
        }
    }

    #[test]
    fn cfl_ac_sub_dispatch_matches_scalar_for_modes_shapes_depths_and_tails() {
        let dispatch = IntraPredDispatch::selected();
        let mut state = 0x299f_31d0u32;
        for &(cw, ch) in &[
            (1usize, 1usize),
            (3, 5),
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
        ] {
            for &(ss_hor, ss_ver) in &[(false, false), (true, false), (false, true), (true, true)] {
                let lw = cw << usize::from(ss_hor);
                let lh = ch << usize::from(ss_ver);
                let stride = lw + 3;
                for &bd in &[8u8, 10, 12] {
                    let max = (1 << bd) - 1;
                    let src: Vec<u16> = (0..stride * lh)
                        .map(|i| match i % 5 {
                            0 => 0,
                            1 => max as u16,
                            _ => lcg(&mut state).clamp(0, max) as u16,
                        })
                        .collect();
                    let n = cw * ch;
                    let mut scalar = vec![0x55aa_i32; n + 3];
                    let mut selected = scalar.clone();
                    cfl_ac_sub(&src, stride, cw, ch, ss_hor, ss_ver, &mut scalar);
                    dispatch.cfl_ac_sub(&src, stride, cw, ch, ss_hor, ss_ver, &mut selected);
                    assert_eq!(
                        selected, scalar,
                        "cw={cw} ch={ch} ss_hor={ss_hor} ss_ver={ss_ver} bd={bd}"
                    );
                }
            }
        }
    }

    #[test]
    fn cfl_pred_simd_matches_scalar_for_tails_and_bit_depths() {
        let dispatch = IntraPredDispatch::selected();
        let mut state = 0x6d2b_79f5u32;
        for &len in &[0usize, 1, 3, 4, 7, 8, 15, 16, 31, 64, 127, 1024] {
            for &bd in &[8u8, 10, 12] {
                let max = (1 << bd) - 1;
                let ac_limit = max << 3;
                let ac: Vec<i32> = (0..len)
                    .map(|i| match i % 8 {
                        0 => -ac_limit,
                        1 => ac_limit,
                        2 => -33,
                        3 => -32,
                        4 => 31,
                        5 => 32,
                        _ => lcg(&mut state).clamp(0, ac_limit) - (ac_limit >> 1),
                    })
                    .collect();
                for alpha in -16..=16 {
                    for dc in [0, max >> 1, max, lcg(&mut state).clamp(0, max)] {
                        let mut want = vec![0i32; len];
                        let mut got = vec![0i32; len];
                        cfl_pred_scalar(&mut want, &ac, dc, alpha, bd);
                        dispatch.cfl_pred(&mut got, &ac, dc, alpha, bd);
                        assert_eq!(got, want, "len={len} bd={bd} alpha={alpha} dc={dc}");
                    }
                }
            }
        }
    }

    #[test]
    fn cfl_dispatch_selects_active_arch_kernel() {
        let dispatch = IntraPredDispatch::selected();
        let selected_ac_444 = dispatch.cfl_ac_444_u16;
        let selected_ac_sub = dispatch.cfl_ac_sub_u16;
        let selected = dispatch.cfl_pred;
        let selected_best_alpha = dispatch.cfl_best_alpha_u16;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            assert!(std::ptr::fn_addr_eq(
                selected_ac_444,
                cfl_ac_444_u16_neon_dispatch as CflAc444U16Fn
            ));
            assert!(std::ptr::fn_addr_eq(
                selected_ac_sub,
                cfl_ac_sub_u16_neon_dispatch as CflAcSubU16Fn
            ));
            assert!(std::ptr::fn_addr_eq(
                selected,
                cfl_pred_neon_dispatch as CflPredFn
            ));
            assert!(std::ptr::fn_addr_eq(
                selected_best_alpha,
                cfl_best_alpha_u16_neon_dispatch as CflBestAlphaU16Fn
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            assert!(std::ptr::fn_addr_eq(
                selected_ac_444,
                if std::is_x86_feature_detected!("avx2") {
                    cfl_ac_444_u16_avx2_dispatch as CflAc444U16Fn
                } else {
                    cfl_ac_444_u16_scalar as CflAc444U16Fn
                }
            ));
            assert!(std::ptr::fn_addr_eq(
                selected_ac_sub,
                if std::is_x86_feature_detected!("avx2") {
                    cfl_ac_sub_u16_avx2_dispatch as CflAcSubU16Fn
                } else {
                    cfl_ac_sub_u16_scalar as CflAcSubU16Fn
                }
            ));
            assert!(std::ptr::fn_addr_eq(
                selected,
                if std::is_x86_feature_detected!("avx2") {
                    cfl_pred_avx2_dispatch as CflPredFn
                } else {
                    cfl_pred_scalar as CflPredFn
                }
            ));
            assert!(std::ptr::fn_addr_eq(
                selected_best_alpha,
                if std::is_x86_feature_detected!("avx2") {
                    cfl_best_alpha_u16_avx2_dispatch as CflBestAlphaU16Fn
                } else {
                    cfl_best_alpha_u16_scalar as CflBestAlphaU16Fn
                }
            ));
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            assert!(std::ptr::fn_addr_eq(
                selected_ac_444,
                cfl_ac_444_u16_scalar as CflAc444U16Fn
            ));
            assert!(std::ptr::fn_addr_eq(
                selected_ac_sub,
                cfl_ac_sub_u16_scalar as CflAcSubU16Fn
            ));
            assert!(std::ptr::fn_addr_eq(selected, cfl_pred_scalar as CflPredFn));
            assert!(std::ptr::fn_addr_eq(
                selected_best_alpha,
                cfl_best_alpha_u16_scalar as CflBestAlphaU16Fn
            ));
        }
    }

    #[test]
    fn cfl_best_alpha_dispatch_matches_scalar_for_highbd_and_tails() {
        let dispatch = IntraPredDispatch::selected();
        let mut state = 0x243f_6a88u32;
        for &len in &[0usize, 1, 7, 8, 9, 15, 16, 31, 64, 127, 256, 1024] {
            for &bd in &[8u8, 10, 12] {
                let max = (1 << bd) - 1;
                let ac_limit = max << 3;
                let ac: Vec<i32> = (0..len)
                    .map(|i| match i % 8 {
                        0 => -ac_limit,
                        1 => ac_limit,
                        2 => -33,
                        3 => -32,
                        4 => 31,
                        5 => 32,
                        _ => lcg(&mut state).clamp(0, ac_limit) - (ac_limit >> 1),
                    })
                    .collect();
                let src: Vec<u16> = (0..len)
                    .map(|i| match i % 4 {
                        0 => 0,
                        1 => max as u16,
                        _ => lcg(&mut state).clamp(0, max) as u16,
                    })
                    .collect();
                for dc in [0, max >> 1, max, lcg(&mut state).clamp(0, max)] {
                    let scalar = cfl_best_alpha(&ac, &src, dc, len, bd);
                    let selected = dispatch.cfl_best_alpha(&ac, &src, dc, len, bd);
                    assert_eq!(selected, scalar, "len={len} bd={bd} dc={dc}");
                }
            }
        }
    }

    #[test]
    fn cfl_pred_simd_exhaustive_highbd_ac_range() {
        let dispatch = IntraPredDispatch::selected();
        for bd in [10u8, 12] {
            let max = (1 << bd) - 1;
            let ac_limit = max << 3;
            let ac: Vec<i32> = (-ac_limit..=ac_limit).collect();
            let mut scalar = vec![0i32; ac.len()];
            let mut simd = vec![0i32; ac.len()];
            for alpha in -16..=16 {
                for dc in [0, max >> 1, max] {
                    cfl_pred_scalar(&mut scalar, &ac, dc, alpha, bd);
                    dispatch.cfl_pred(&mut simd, &ac, dc, alpha, bd);
                    assert_eq!(simd, scalar, "bd={bd} alpha={alpha} dc={dc}");
                }
            }
        }
    }

    /// Every predictor entry point dispatches to the active SIMD kernel;
    /// compare each against the spec formula for all legal block shapes.
    /// Validates NEON on aarch64 and AVX2 on x86_64 (whichever is built).
    #[test]
    fn simd_predictors_match_scalar_reference() {
        let simd = IntraPredDispatch::selected();
        let scalar = IntraPredDispatch::scalar();
        let mut st = 0x9e37_79b9u32;
        let mut top = [0i32; 128];
        let mut left = [0i32; 128];
        for v in top.iter_mut() {
            *v = lcg(&mut st);
        }
        for v in left.iter_mut() {
            *v = lcg(&mut st);
        }
        let corner = lcg(&mut st);
        for &bw in &[4usize, 8, 16, 32] {
            for &bh in &[4usize, 8, 16, 32] {
                let n = bw * bh;
                let mut got = vec![0i32; n];
                (simd.smooth)(bw, bh, &top, &left, &mut got);
                let mut want = vec![0i32; n];
                (scalar.smooth)(bw, bh, &top, &left, &mut want);
                assert_eq!(got, want, "smooth {bw}x{bh}");

                (simd.smooth_v)(bw, bh, &top, &left, &mut got);
                (scalar.smooth_v)(bw, bh, &top, &left, &mut want);
                assert_eq!(got, want, "smooth_v {bw}x{bh}");

                (simd.smooth_h)(bw, bh, &top, &left, &mut got);
                (scalar.smooth_h)(bw, bh, &top, &left, &mut want);
                assert_eq!(got, want, "smooth_h {bw}x{bh}");

                (simd.paeth)(bw, bh, &top, &left, corner, &mut got);
                (scalar.paeth)(bw, bh, &top, &left, corner, &mut want);
                assert_eq!(got, want, "paeth {bw}x{bh}");

                (simd.vertical)(bw, bh, &top, &left, &mut got);
                (scalar.vertical)(bw, bh, &top, &left, &mut want);
                assert_eq!(got, want, "vertical {bw}x{bh}");

                (simd.horizontal)(bw, bh, &top, &left, &mut got);
                (scalar.horizontal)(bw, bh, &top, &left, &mut want);
                assert_eq!(got, want, "horizontal {bw}x{bh}");
            }
        }

        // Paeth and the copy predictors are also valid for partial vector
        // rows. Exercise every AVX2/NEON tail width explicitly.
        for bw in 1..16 {
            let bh = 7;
            let mut got = vec![0i32; bw * bh];
            let mut want = vec![0i32; bw * bh];
            (simd.paeth)(bw, bh, &top, &left, corner, &mut got);
            (scalar.paeth)(bw, bh, &top, &left, corner, &mut want);
            assert_eq!(got, want, "paeth tail {bw}x{bh}");
            (simd.vertical)(bw, bh, &top, &left, &mut got);
            (scalar.vertical)(bw, bh, &top, &left, &mut want);
            assert_eq!(got, want, "vertical tail {bw}x{bh}");
            (simd.horizontal)(bw, bh, &top, &left, &mut got);
            (scalar.horizontal)(bw, bh, &top, &left, &mut want);
            assert_eq!(got, want, "horizontal tail {bw}x{bh}");
        }

        let stride = 80;
        let mut recon = vec![0u16; stride * 80];
        for sample in &mut recon {
            *sample = lcg(&mut st) as u16;
        }
        for &(w, h) in &[
            (4usize, 4usize),
            (8, 4),
            (4, 8),
            (16, 8),
            (8, 16),
            (32, 16),
            (16, 32),
            (32, 32),
        ] {
            for &(ox, oy) in &[(0usize, 0usize), (4, 0), (0, 4), (4, 4)] {
                let got = (simd.dc)(&recon, stride, ox, oy, w, h, 10);
                let want = (scalar.dc)(&recon, stride, ox, oy, w, h, 10);
                assert_eq!(got, want, "dc {w}x{h} at {ox},{oy}");
            }
        }
    }

    /// The SIMD filter-intra cell pass must equal the scalar tap loop for
    /// every mode and legal block shape. Vacuous (and reported) on targets
    /// with no kernel.
    #[test]
    fn filter_intra_cells_simd_matches_scalar() {
        let simd = IntraPredDispatch::selected();
        let scalar = IntraPredDispatch::scalar();
        let mut st = 0x1234_5678u32;
        for &mode in FILTER_INTRA_MODES.iter() {
            for &(w, h) in &[
                (4usize, 4usize),
                (8, 8),
                (16, 16),
                (32, 32),
                (4, 8),
                (8, 4),
                (16, 8),
            ] {
                let mut a = [[0i32; 33]; 33];
                // Only row 0 and column 0 are live references; cells
                // overwrite the rest.
                for v in a[0].iter_mut() {
                    *v = lcg(&mut st);
                }
                for row in a.iter_mut() {
                    row[0] = lcg(&mut st);
                }
                let mut b = a;
                let taps = &crate::tables::INTRA_FILTER_TAPS[mode as usize];
                (scalar.filter_intra_cells)(&mut a, taps, w, h, 255);
                (simd.filter_intra_cells)(&mut b, taps, w, h, 255);
                assert_eq!(a, b, "filter_intra cells {mode:?} {w}x{h}");
            }
        }
    }

    /// The SIMD 5-tap edge convolution must equal the scalar convolution over
    /// the clamp-free middle run, for every kernel and edge length.
    #[test]
    fn edge_conv5_simd_matches_scalar() {
        let simd = IntraPredDispatch::selected();
        let scalar = IntraPredDispatch::scalar();
        const KERNELS: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
        let mut st = 0xfeed_face_u32;
        for kernel in KERNELS {
            for len in 1..40usize {
                let win: Vec<i32> = (0..len + 4).map(|_| lcg(&mut st)).collect();
                let mut got = vec![0i32; len];
                let mut want = vec![0i32; len];
                (simd.edge_conv5)(&win, &kernel, &mut got);
                (scalar.edge_conv5)(&win, &kernel, &mut want);
                assert_eq!(got, want, "edge_conv5 kernel={kernel:?} len={len}");
            }
        }
    }

    #[test]
    fn directional_block_simd_matches_scalar_reference() {
        let simd = IntraPredDispatch::selected();
        let scalar = IntraPredDispatch::scalar();
        let mut state = 0x2545_f491u32;
        let mut above = [0i32; EDGE_CAPACITY];
        let mut left = [0i32; EDGE_CAPACITY];
        for value in above.iter_mut().chain(left.iter_mut()) {
            *value = lcg(&mut state) & 4095;
        }
        for &(bw, bh) in &[
            (4usize, 4usize),
            (8, 4),
            (4, 8),
            (8, 8),
            (16, 8),
            (8, 16),
            (16, 16),
            (32, 16),
            (16, 32),
            (32, 32),
            (64, 32),
            (32, 64),
            (64, 64),
        ] {
            let edge_len = bw + bh;
            for &(zone, angle) in &[
                (DR_ZONE1, 36i32),
                (DR_ZONE1, 45),
                (DR_ZONE1, 76),
                (DR_ZONE2, 104),
                (DR_ZONE2, 113),
                (DR_ZONE2, 135),
                (DR_ZONE2, 157),
                (DR_ZONE2, 166),
                (DR_ZONE3, 194),
                (DR_ZONE3, 203),
                (DR_ZONE3, 212),
            ] {
                let dx = match zone {
                    DR_ZONE1 => DR_INTRA_DERIVATIVE[(angle >> 1) as usize],
                    DR_ZONE2 => DR_INTRA_DERIVATIVE[((180 - angle) >> 1) as usize],
                    _ => 0,
                };
                let dy = match zone {
                    DR_ZONE2 => DR_INTRA_DERIVATIVE[((angle - 90) >> 1) as usize],
                    DR_ZONE3 => DR_INTRA_DERIVATIVE[((270 - angle) >> 1) as usize],
                    _ => 0,
                };
                for up in 0..=usize::from(edge_len <= 16) {
                    let p = DrPrediction {
                        zone,
                        bw,
                        bh,
                        edge_len,
                        dx,
                        dy,
                        up_above: up as i32,
                        up_left: up as i32,
                    };
                    let mut got = vec![0i32; bw * bh];
                    let mut want = vec![0i32; bw * bh];
                    (simd.dr_predict)(p, &above, &left, &mut got);
                    (scalar.dr_predict)(p, &above, &left, &mut want);
                    assert_eq!(
                        got, want,
                        "directional zone={zone} angle={angle} {bw}x{bh} up={up}"
                    );
                }
            }
        }
    }

    #[test]
    fn directional_predictor_dispatch_matches_scalar_for_prepared_edges() {
        let simd = IntraPredDispatch::selected();
        let scalar = IntraPredDispatch::scalar();
        let stride = 96usize;
        let mut state = 0x1357_9bdfu32;
        for bd in [8u8, 10, 12] {
            let mask = (1i32 << bd) - 1;
            let mut recon = vec![0u16; stride * stride];
            for value in &mut recon {
                *value = (lcg(&mut state) & mask) as u16;
            }
            for &(bw, bh) in &[
                (4usize, 4usize),
                (8, 4),
                (4, 8),
                (8, 8),
                (16, 8),
                (8, 16),
                (16, 16),
                (32, 16),
                (16, 32),
                (32, 32),
                (64, 32),
                (32, 64),
                (64, 64),
            ] {
                for mode in D45_PRED..=VERT_LEFT_PRED {
                    if mode == H_PRED {
                        continue;
                    }
                    for delta in -3..=3 {
                        for &(ox, oy, have_tr, have_bl) in &[
                            (0usize, 0usize, false, false),
                            (8, 0, true, false),
                            (0, 8, false, true),
                            (8, 8, true, true),
                        ] {
                            for filter_type in [false, true] {
                                let mut got = vec![0i32; bw * bh];
                                let mut want = vec![0i32; bw * bh];
                                simd.predict_nd_ad(
                                    mode,
                                    delta,
                                    &recon,
                                    stride,
                                    ox,
                                    oy,
                                    bw,
                                    bh,
                                    have_tr,
                                    have_bl,
                                    stride,
                                    stride,
                                    filter_type,
                                    &mut got,
                                    bd,
                                );
                                scalar.predict_nd_ad(
                                    mode,
                                    delta,
                                    &recon,
                                    stride,
                                    ox,
                                    oy,
                                    bw,
                                    bh,
                                    have_tr,
                                    have_bl,
                                    stride,
                                    stride,
                                    filter_type,
                                    &mut want,
                                    bd,
                                );
                                assert_eq!(
                                    got, want,
                                    "mode={mode} delta={delta} {bw}x{bh} at {ox},{oy} \
                                     tr={have_tr} bl={have_bl} filter={filter_type} bd={bd}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn palette_predicts_packed_indices_into_strided_destination() {
        let palette = [3, 17, 42, 99, 255, 511, 777, 1023];
        let indices = [0x10, 0x32, 0x04, 0x76, 0x54, 0x03];
        let mut dst = [-1; 16];

        palette_pred(&mut dst, 8, &palette, &indices, 5, 2);

        assert_eq!(&dst[..5], &[3, 17, 42, 99, 255]);
        assert_eq!(&dst[8..13], &[777, 1023, 255, 511, 99]);
        assert_eq!(&dst[5..8], &[-1; 3]);
        assert_eq!(&dst[13..], &[-1; 3]);
    }

    #[test]
    fn palette_accepts_all_normative_sizes() {
        for size in 2..=8 {
            let palette: Vec<i32> = (0..size as i32).map(|v| v * 100).collect();
            let last = (size - 1) as u8;
            let indices = [last | (last << 4)];
            let mut dst = [0; 2];
            palette_pred(&mut dst, 2, &palette, &indices, 2, 1);
            assert_eq!(dst, [palette[size - 1]; 2]);
        }
    }

    #[test]
    fn filter_intra_constant_edges_stay_constant() {
        let sizes = [
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (4, 16),
            (16, 4),
            (8, 32),
            (32, 8),
        ];
        for bd in [8u8, 10, 12] {
            let value = 37 << (bd - 8);
            let recon = vec![value; 40 * 40];
            for &(w, h) in &sizes {
                for mode in FILTER_INTRA_MODES {
                    let mut out = vec![0; w * h];
                    filter_intra_predict(mode, &recon, 40, 1, 1, w, h, &mut out, bd);
                    assert!(out.iter().all(|&v| v == value), "{mode:?} {w}x{h} bd={bd}");
                }
            }
        }
    }

    #[test]
    fn filter_intra_handles_missing_edges_and_clips() {
        for bd in [8u8, 10, 12] {
            let max = (1 << bd) - 1;
            let recon = vec![max; 32 * 32];
            for &(ox, oy) in &[(0, 0), (1, 0), (0, 1), (1, 1)] {
                for mode in FILTER_INTRA_MODES {
                    let mut out = [0; 64];
                    filter_intra_predict(mode, &recon, 32, ox, oy, 8, 8, &mut out, bd);
                    assert!(out.iter().all(|&v| (0..=max).contains(&v)));
                }
            }
        }
    }

    #[test]
    fn filter_intra_is_recursive_across_groups() {
        let mut recon = vec![128; 16 * 16];
        for (x, r) in recon.iter_mut().enumerate().take(8) {
            *r = 16 + x as i32 * 20;
        }
        for y in 0..8 {
            recon[y * 16] = 240 - y as i32 * 18;
        }
        let mut a = [0; 64];
        let mut b = [0; 64];
        filter_intra_predict(FilterIntraMode::D157, &recon, 16, 1, 1, 8, 8, &mut a, 8);
        recon[0] ^= 63;
        filter_intra_predict(FilterIntraMode::D157, &recon, 16, 1, 1, 8, 8, &mut b, 8);
        assert_ne!(a[..8], b[..8]);
        assert_ne!(a[32..], b[32..]);
    }

    #[test]
    fn strength_thresholds_match_av1() {
        assert_eq!(intra_edge_filter_strength(4, 4, false, 55), 0);
        assert_eq!(intra_edge_filter_strength(4, 4, false, 56), 1);
        assert_eq!(intra_edge_filter_strength(8, 16, false, 8), 1);
        assert_eq!(intra_edge_filter_strength(8, 16, false, 16), 2);
        assert_eq!(intra_edge_filter_strength(8, 16, false, 32), 3);
    }

    #[test]
    fn upsample_thresholds_match_av1() {
        assert!(use_intra_edge_upsample(8, 8, false, 23));
        assert!(!use_intra_edge_upsample(8, 8, false, 40));
        assert!(!use_intra_edge_upsample(8, 8, true, 23));
    }

    #[test]
    fn filtering_preserves_constant_edge_and_corner() {
        for strength in 1..=3 {
            let mut edge = IntraEdge::new();
            for i in -1..16 {
                edge.set(i, 317);
            }
            let mut source = [0; EDGE_CAPACITY];
            filter_intra_edge(
                selected_intra_dispatch(),
                &mut edge,
                17,
                strength,
                &mut source,
            );
            assert_eq!(edge.get(-1), 317);
            for i in 0..16 {
                assert_eq!(edge.get(i), 317);
            }
        }
    }

    #[test]
    fn upsampling_preserves_original_samples_and_range() {
        let mut edge = IntraEdge::new();
        edge.set(-1, 9);
        let original = [12, 80, 220, 255, 17, 91, 43, 199];
        for (i, &value) in original.iter().enumerate() {
            edge.set(i as i32, value);
        }
        let mut input = [0; EDGE_CAPACITY];
        upsample_intra_edge(&mut edge, original.len(), 8, &mut input);
        for (i, &value) in original.iter().enumerate() {
            assert_eq!(edge.get(2 * i as i32), value);
        }
        for i in -2..(2 * original.len() as i32 - 1) {
            assert!((0..=255).contains(&edge.get(i)));
        }
    }

    #[test]
    fn corner_filter_updates_both_copies() {
        let mut above = IntraEdge::new();
        let mut left = IntraEdge::new();
        above.set(-1, 100);
        left.set(-1, 100);
        above.set(0, 140);
        left.set(0, 60);
        filter_intra_corner(&mut above, &mut left);
        assert_eq!(above.get(-1), left.get(-1));
        assert_eq!(above.get(-1), 100);
    }

    #[test]
    fn z2_prediction_prepares_missing_tile_corner_edges() {
        let recon = vec![0; 32 * 32];
        let mut out = [0; 64];
        intra_predict_nd_ad(
            D157_PRED, 0, &recon, 32, 0, 0, 8, 8, false, false, 32, 32, false, &mut out, 8,
        );
        // Reference output from AV1's edge preparation + zone-2 projection.
        // In particular, no uninitialized gaps (the former 40/80/12 values)
        // may appear when both physical neighbors are unavailable.
        assert_eq!(
            out,
            [
                129, 128, 127, 127, 127, 127, 127, 127, 129, 129, 129, 129, 128, 127, 127, 127,
                129, 129, 129, 129, 129, 129, 128, 127, 129, 129, 129, 129, 129, 129, 129, 129,
                129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
                129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129,
            ]
        );
    }
}
