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
use hashbrown::HashMap;
#[cfg(test)]
pub(crate) static FORCE_SPLIT4: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub(crate) static LOSSY_PALETTE_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static LOSSY_PALETTE_RESIDUAL_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static LOSSY_INTRABC_EMITTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(not(test))]
pub(crate) static FORCE_SPLIT4: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) static SPLIT4_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

pub(crate) static FORCE_HORZ: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub static HORZ_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub static VERT_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[derive(Clone, Copy, PartialEq, Eq)]
enum RawSseGuard {
    FilterIntra,
    TxType,
    TxSplit,
}

fn raw_sse_guard_disabled(kind: RawSseGuard) -> bool {
    let _ = kind;
    false
}

/// Apply one heuristic SSE policy while retaining the pure-RD choice as a
/// shadow decision.
fn raw_sse_guard_choice(
    _tag: &'static str,
    kind: RawSseGuard,
    _baseline_sse: i64,
    _candidate_sse: i64,
    baseline_rd: f32,
    candidate_rd: f32,
    guarded_choice: bool,
) -> bool {
    let rd_choice = candidate_rd < baseline_rd;
    if raw_sse_guard_disabled(kind) {
        rd_choice
    } else {
        guarded_choice
    }
}

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

/// AV1 chroma transform type derived from the intra mode (`Mode_To_Txfm`)
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChromaTx {
    DctDct,
    AdstAdst,
    AdstDct,
    DctAdst,
}

/// `Mode_To_Txfm` for the chroma intra modes the encoder searches. SMOOTH/PAETH
/// (ADST_ADST), SMOOTH_V (ADST_DCT) and SMOOTH_H (DCT_ADST) need no angle_delta.
/// V_PRED (ADST_DCT) and H_PRED (DCT_ADST) are directional: they additionally
/// emit a chroma `angle_delta` symbol (delta 0), and are only offered where the
/// chroma block is >= 8x8 (4:4:4), matching AV1's `use_angle_delta`.
fn chroma_tx_for_mode(mode: usize) -> ChromaTx {
    // Mirrors dav1d_txtp_from_uvmode: the decoder derives the chroma transform
    // type from the uv_mode. (At tx sizes whose square is >= TX_32X32 the spec
    // forces DCT_DCT regardless; callers at those sizes ignore this and use DCT.)
    match mode {
        // ADST_ADST: SMOOTH, PAETH, D135 (DIAG_DOWN_RIGHT)
        m if m == PAETH_PRED || m == SMOOTH_PRED || m == D135_PRED => ChromaTx::AdstAdst,
        // ADST_DCT: SMOOTH_V, V, D113 (VERT_RIGHT), D67 (VERT_LEFT)
        m if m == SMOOTH_V_PRED || m == V_PRED || m == D113_PRED || m == VERT_LEFT_PRED => {
            ChromaTx::AdstDct
        }
        // DCT_ADST: SMOOTH_H, H, D157 (HOR_DOWN), D203 (HOR_UP)
        m if m == SMOOTH_H_PRED || m == H_PRED || m == D157_PRED || m == D203_PRED => {
            ChromaTx::DctAdst
        }
        // DCT_DCT: DC, D45 (DIAG_DOWN_LEFT), and anything else.
        _ => ChromaTx::DctDct,
    }
}

/// Forward transform + trellis quant for an 8x8 chroma block under the given
/// chroma tx kind. Returns levels + unrounded targets like the other `*_t`.
fn fwd_chroma_8x8(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 64],
    q: &Quant,
) -> ([i32; 64], [f32; 64]) {
    match tx {
        ChromaTx::DctDct => dct.dct8x8_t(resid, q),
        ChromaTx::AdstAdst => dct.adst8x8_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct8x8_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst8x8_t(resid, q),
    }
}

fn inv_chroma_8x8(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_8x8(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_8x8(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_8x8(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_8x8(levels, q),
    }
}

/// Forward transform + trellis quant for a 16x16 chroma block under the given
/// chroma tx kind (mirrors `fwd_chroma_8x8` at TX_16X16).
fn fwd_chroma_16x16(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 256],
    q: &Quant,
) -> ([i32; 256], [f32; 256]) {
    match tx {
        ChromaTx::DctDct => dct.dct16x16_t(resid, q),
        ChromaTx::AdstAdst => dct.adst16x16_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct16x16_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst16x16_t(resid, q),
    }
}

fn inv_chroma_16x16(
    idct: &IdctDispatch,
    tx: ChromaTx,
    levels: &[i32; 256],
    q: &Quant,
) -> [i32; 256] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_16x16(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_16x16(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_16x16(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_16x16(levels, q),
    }
}

/// Forward transform + quant for a 16x8 / 8x16 chroma block under the given
/// chroma tx kind (the decoder derives it from the uv_mode; rect chroma at
/// 4:4:4 rect16 leaves).
fn fwd_chroma_16x8(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 128],
    q: &Quant,
) -> ([i32; 128], [f32; 128]) {
    match tx {
        ChromaTx::DctDct => dct.dct16x8_t(resid, q),
        ChromaTx::AdstAdst => dct.adst16x8_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct16x8_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst16x8_t(resid, q),
    }
}

fn inv_chroma_16x8(
    idct: &IdctDispatch,
    tx: ChromaTx,
    levels: &[i32; 128],
    q: &Quant,
) -> [i32; 128] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_16x8(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_16x8(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_16x8(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_16x8(levels, q),
    }
}

fn fwd_chroma_8x16(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 128],
    q: &Quant,
) -> ([i32; 128], [f32; 128]) {
    match tx {
        ChromaTx::DctDct => dct.dct8x16_t(resid, q),
        ChromaTx::AdstAdst => dct.adst8x16_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct8x16_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst8x16_t(resid, q),
    }
}

fn inv_chroma_8x16(
    idct: &IdctDispatch,
    tx: ChromaTx,
    levels: &[i32; 128],
    q: &Quant,
) -> [i32; 128] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_8x16(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_8x16(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_8x16(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_8x16(levels, q),
    }
}

fn fwd_chroma_4x4(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 16],
    q: &Quant,
) -> ([i32; 16], [f32; 16]) {
    match tx {
        ChromaTx::DctDct => dct.dct4x4_t(resid, q),
        ChromaTx::AdstAdst => dct.adst4x4_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct4x4_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst4x4_t(resid, q),
    }
}

fn inv_chroma_4x4(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_4x4(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_4x4(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_4x4(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_4x4(levels, q),
    }
}

fn fwd_chroma_4x8(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 32],
    q: &Quant,
) -> ([i32; 32], [f32; 32]) {
    match tx {
        ChromaTx::DctDct => dct.dct4x8_t(resid, q),
        ChromaTx::AdstAdst => dct.adst4x8_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct4x8_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst4x8_t(resid, q),
    }
}

fn inv_chroma_4x8(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_4x8(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_4x8(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_4x8(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_4x8(levels, q),
    }
}

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

pub(crate) struct Cdfs {
    /// Trellis frequency-tilt strength for this tile's format — see
    /// `trellis_tilt_mag_cap()` in trellis.rs. Holdout-selected per chroma
    /// format 2026-07-21 (NB selected ON the holdout after many iterations,
    /// so treat the exact constants with a grain of salt): 420 t=5.0 =
    /// **-5.18% BD, all 9 images negative**; 422 t=3.5 = -4.28% all
    /// negative; 444 t=1.2 = -2.10% (h_abstract +0.89, the one exception —
    /// 444 is chronically tilt-sensitive on sharp synthetics). Applied to
    /// the luma plane only in practice (chroma coding does not run
    /// trellis_optimize_ctx). Zero disables (lossless, tests).
    pub(crate) band_tilt: f32,
    pub(crate) skip: Vec<Vec<u16>>,             // block skip [3 ctx]
    pub(crate) intrabc: Vec<u16>,               // use_intrabc
    pub(crate) mv_joint: Vec<u16>,              // MV_JOINT
    pub(crate) mv_sign: [Vec<u16>; 2],          // vertical, horizontal
    pub(crate) mv_classes: [Vec<u16>; 2],       // MV class 0..10
    pub(crate) mv_class0: [Vec<u16>; 2],        // class-zero integer bit
    pub(crate) mv_class_n: [[Vec<u16>; 10]; 2], // class-N integer bits
    pub(crate) part_bl8: Vec<Vec<u16>>,         // PARTITION_NONE @ 8x8 [4 ctx]
    pub(crate) part_split: Vec<Vec<Vec<u16>>>,  // SPLIT [bl-1=0..3][4 ctx]
    pub(crate) kf_y: Vec<Vec<u16>>,             // kf_y_mode[5*5], index [above_ctx*5 + left_ctx]
    pub(crate) uv_mode: Vec<Vec<u16>>,          // uv_mode[2*13], index [cfl_allowed*13 + y_mode]
    pub(crate) angle_delta: Vec<Vec<u16>>,      // angle_delta[8 directional modes]
    pub(crate) filter_intra: Vec<Vec<u16>>,     // use_filter_intra [BLOCK_SIZES_ALL]
    pub(crate) filter_intra_mode: Vec<u16>,     // five filter-intra predictors
    pub(crate) palette_y_mode: Vec<Vec<Vec<u16>>>, // [7 bsize ctx][3 neighbor ctx]
    pub(crate) palette_y_size: Vec<Vec<u16>>,   // [7 bsize ctx], sizes 2..8
    pub(crate) palette_uv_mode: [Vec<u16>; 2],  // luma palette absent/present
    pub(crate) palette_y_color: Vec<Vec<Vec<u16>>>, // [size-2][5 map ctx]
    pub(crate) palette_uv_size: Vec<Vec<u16>>,  // [7 bsize ctx], sizes 2..8
    pub(crate) palette_uv_color: Vec<Vec<Vec<u16>>>, // [size-2][5 map ctx]
    pub(crate) cfl_sign: Vec<u16>,              // cfl joint-sign (8 symbols)
    pub(crate) cfl_alpha: Vec<Vec<u16>>,        // cfl alpha magnitude [6 ctx]
    pub(crate) txsz: [Vec<Vec<u16>>; 4],        // intra tx_depth [t_dim.max-1][3 ctx]
    pub(crate) txtp: Vec<Vec<u16>>,             // intra txtp TX_8X8 luma, per intra mode [13]
    pub(crate) txtp4: Vec<Vec<u16>>,            // intra txtp TX_4X4 luma, per intra mode [13]
    pub(crate) txtp16: Vec<Vec<u16>>,           // intra txtp TX_16X16 luma, per intra mode [13]
    pub(crate) txb_skip: [Vec<Vec<u16>>; 4],    // [class][13 ctx] (class 3 = TX_32X32)
    pub(crate) base_tok: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][41/42 ctx]
    pub(crate) br_tok: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][21 ctx]
    pub(crate) eob_base: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][4 ctx]
    pub(crate) eob_hi: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][11 bins], each a 2-sym CDF
    pub(crate) dc_sign: [Vec<Vec<u16>>; 2],     // [plane][3 ctx]
    pub(crate) eob_bin_16_c: Vec<u16>,          // chroma, 4x4
    pub(crate) eob_bin_16_l: Vec<u16>,          // luma, 4x4
    pub(crate) eob_bin_32_c: Vec<u16>,
    pub(crate) eob_bin_32_l: Vec<u16>,
    pub(crate) eob_bin_64_l: Vec<u16>, // luma, 8x8
    /// 1-D tx-class (V_DCT/H_DCT) luma eob bins — dav1d eob_bin_64[0][1].
    pub(crate) eob_bin_16_l1d: Vec<u16>, // luma 4x4, 1-D tx classes
    pub(crate) eob_bin_64_l1d: Vec<u16>,
    /// Luma 128-coeff eob bins for the 1-D tx classes (V_DCT/H_DCT at
    /// RTX_16X8/RTX_8X16): dav1d `eob_bin_128[0][is_1d=1]`.
    pub(crate) eob_bin_128_l1d: Vec<u16>,
    pub(crate) eob_bin_64_c: Vec<u16>,   // chroma, 8x8
    pub(crate) eob_bin_256_l: Vec<u16>,  // luma, 16x16 (class 2)
    pub(crate) eob_bin_256_c: Vec<u16>,  // chroma, 16x16 (class 2)
    pub(crate) eob_bin_128_c: Vec<u16>,  // chroma, RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_128_l: Vec<u16>,  // luma, RTX_16X8/RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_1024_l: Vec<u16>, // luma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_1024_c: Vec<u16>, // chroma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_512_c: Vec<u16>,
    pub(crate) eob_bin_512_l: Vec<u16>,
    pub(crate) delta_q: Vec<u16>, // superblock delta-q magnitude (4 symbols)
    pub(crate) wiener_restore: Vec<u16>, // use_wiener flag (2-symbol)
}

impl Cdfs {
    /// Return every adaptive CDF leaf in a stable logical order. The raw
    /// pointers are used only while this `Cdfs` instance remains alive and no
    /// table Vec is resized. Capture workers translate their own addresses to
    /// these indices; the raster packer uses the same indices on its live
    /// adaptive state.
    fn semantic_slots(&mut self) -> Vec<(*mut u16, usize)> {
        let mut out = Vec::with_capacity(1400);
        let mut push = |v: &mut Vec<u16>| out.push((v.as_mut_ptr(), v.len()));

        for v in &mut self.skip {
            push(v);
        }
        push(&mut self.intrabc);
        push(&mut self.mv_joint);
        for v in &mut self.mv_sign {
            push(v);
        }
        for v in &mut self.mv_classes {
            push(v);
        }
        for v in &mut self.mv_class0 {
            push(v);
        }
        for plane in &mut self.mv_class_n {
            for v in plane {
                push(v);
            }
        }
        for v in &mut self.part_bl8 {
            push(v);
        }
        for level in &mut self.part_split {
            for v in level {
                push(v);
            }
        }
        for v in &mut self.kf_y {
            push(v);
        }
        for v in &mut self.uv_mode {
            push(v);
        }
        for v in &mut self.angle_delta {
            push(v);
        }
        for v in &mut self.filter_intra {
            push(v);
        }
        push(&mut self.filter_intra_mode);
        for size in &mut self.palette_y_mode {
            for v in size {
                push(v);
            }
        }
        for v in &mut self.palette_y_size {
            push(v);
        }
        for v in &mut self.palette_uv_mode {
            push(v);
        }
        for size in &mut self.palette_y_color {
            for v in size {
                push(v);
            }
        }
        for v in &mut self.palette_uv_size {
            push(v);
        }
        for size in &mut self.palette_uv_color {
            for v in size {
                push(v);
            }
        }
        push(&mut self.cfl_sign);
        for v in &mut self.cfl_alpha {
            push(v);
        }
        for cat in &mut self.txsz {
            for v in cat {
                push(v);
            }
        }
        for v in &mut self.txtp {
            push(v);
        }
        for v in &mut self.txtp4 {
            push(v);
        }
        for v in &mut self.txtp16 {
            push(v);
        }
        for class in &mut self.txb_skip {
            for v in class {
                push(v);
            }
        }
        for class in &mut self.base_tok {
            for plane in class {
                for v in plane {
                    push(v);
                }
            }
        }
        for class in &mut self.br_tok {
            for plane in class {
                for v in plane {
                    push(v);
                }
            }
        }
        for class in &mut self.eob_base {
            for plane in class {
                for v in plane {
                    push(v);
                }
            }
        }
        for class in &mut self.eob_hi {
            for plane in class {
                for v in plane {
                    push(v);
                }
            }
        }
        for plane in &mut self.dc_sign {
            for v in plane {
                push(v);
            }
        }
        for v in [
            &mut self.eob_bin_16_c,
            &mut self.eob_bin_16_l,
            &mut self.eob_bin_32_c,
            &mut self.eob_bin_32_l,
            &mut self.eob_bin_64_l,
            &mut self.eob_bin_16_l1d,
            &mut self.eob_bin_64_l1d,
            &mut self.eob_bin_128_l1d,
            &mut self.eob_bin_64_c,
            &mut self.eob_bin_256_l,
            &mut self.eob_bin_256_c,
            &mut self.eob_bin_128_c,
            &mut self.eob_bin_128_l,
            &mut self.eob_bin_1024_l,
            &mut self.eob_bin_1024_c,
            &mut self.eob_bin_512_c,
            &mut self.eob_bin_512_l,
            &mut self.delta_q,
            &mut self.wiener_restore,
        ] {
            push(v);
        }
        debug_assert!(out.len() <= u16::MAX as usize);
        out
    }

    pub(crate) fn with_band_tilt(mut self, t: f32) -> Self {
        self.band_tilt = t;
        self
    }

    /// Frame-initial lossless state. The static mode deliberately restores the
    /// legacy inverse-CDF tables used before adaptation was introduced.
    pub(crate) fn new_lossless(updating_cdf: bool) -> Self {
        let mut c = Self::new(0);
        if updating_cdf {
            return c;
        }
        use crate::cdf_tables as legacy;
        c.txb_skip[0] = legacy::C_SKIP.iter().map(|v| v.to_vec()).collect();
        c.eob_bin_16_l = legacy::EOB_BIN16[0].to_vec();
        c.eob_bin_16_c = legacy::EOB_BIN16[1].to_vec();
        for plane in 0..2 {
            c.eob_hi[0][plane] = legacy::EOB_HI[plane].iter().map(|v| v.to_vec()).collect();
            c.eob_base[0][plane] = legacy::EOB_BASE[plane].iter().map(|v| v.to_vec()).collect();
            c.base_tok[0][plane] = legacy::BASE_TOK[plane].iter().map(|v| v.to_vec()).collect();
            c.br_tok[0][plane] = legacy::BR_TOK[plane].iter().map(|v| v.to_vec()).collect();
            c.dc_sign[plane] = legacy::DC_SIGN[plane].iter().map(|v| v.to_vec()).collect();
        }
        for (level, table) in [
            &legacy::PART_SPLIT_64,
            &legacy::PART_SPLIT_32,
            &legacy::PART_SPLIT_16,
        ]
        .into_iter()
        .enumerate()
        {
            c.part_split[level] = table.iter().map(|v| v.to_vec()).collect();
        }
        c.part_bl8 = legacy::PART_8.iter().map(|v| v.to_vec()).collect();
        c
    }

    /// Frozen snapshot used for DECISION-side rate estimates (`dec_cdfs`).
    /// Mostly the frame-initial CDFs, except symbols whose default prior sits
    /// far from its adapted steady state on real content: there a frozen
    /// default systematically mis-prices the choice for the whole frame
    /// (adaptive coding self-corrects; a frozen estimate cannot).
    pub(crate) fn decision_snapshot(qctx: usize) -> Box<Self> {
        let mut c = Self::new(qctx);
        for e in c.filter_intra.iter_mut() {
            *e = icdf(&[16384]);
        }
        Box::new(c)
    }

    pub(crate) fn new(qctx: usize) -> Self {
        use crate::coef_q as Q;
        let rows = |t: &[[u16; 3]]| t.iter().map(|r| icdf(r)).collect::<Vec<_>>();
        let rows2 = |t: &[[u16; 2]]| t.iter().map(|r| icdf(r)).collect::<Vec<_>>();
        let his = |t: &[u16]| t.iter().map(|&v| icdf(&[v])).collect::<Vec<_>>();
        // skip CDFs by tx class
        let txb_skip = [
            Q::SKIP_TX4[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
            Q::SKIP_TX8[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
            Q::SKIP_TX16[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
            Q::SKIP_TX32[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
        ];
        // base/br/eob_base/eob_hi per [class][plane]
        let base_tok = [
            [
                rows(&Q::BASE_TOK_TX4_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX4_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BASE_TOK_TX8_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX8_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BASE_TOK_TX16_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX16_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BASE_TOK_TX32_LUMA_Q[qctx]),
                rows(&Q::BASE_TOK_TX32_CHROMA_Q[qctx]),
            ],
        ];
        let br_tok = [
            [
                rows(&Q::BR_TOK_TX4_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX4_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BR_TOK_TX8_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX8_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BR_TOK_TX16_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX16_CHROMA_Q[qctx]),
            ],
            [
                rows(&Q::BR_TOK_TX32_LUMA_Q[qctx]),
                rows(&Q::BR_TOK_TX32_CHROMA_Q[qctx]),
            ],
        ];
        let eob_base = [
            [
                rows2(&Q::EOB_BASE_TX4_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX4_CHROMA_Q[qctx]),
            ],
            [
                rows2(&Q::EOB_BASE_TX8_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX8_CHROMA_Q[qctx]),
            ],
            [
                rows2(&Q::EOB_BASE_TX16_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX16_CHROMA_Q[qctx]),
            ],
            [
                rows2(&Q::EOB_BASE_TX32_LUMA_Q[qctx]),
                rows2(&Q::EOB_BASE_TX32_CHROMA_Q[qctx]),
            ],
        ];
        let eob_hi = [
            [
                his(&Q::EOB_HI_TX4_LUMA[qctx]),
                his(&Q::EOB_HI_TX4_CHROMA[qctx]),
            ],
            [
                his(&Q::EOB_HI_TX8_LUMA[qctx]),
                his(&Q::EOB_HI_TX8_CHROMA[qctx]),
            ],
            [
                his(&Q::EOB_HI_TX16_LUMA[qctx]),
                his(&Q::EOB_HI_TX16_CHROMA[qctx]),
            ],
            [
                his(&Q::EOB_HI_TX32_LUMA[qctx]),
                his(&Q::EOB_HI_TX32_CHROMA[qctx]),
            ],
        ];
        Cdfs {
            band_tilt: 0.0,
            skip: SKIP_CDF.iter().map(|&v| icdf(&[v])).collect(),
            intrabc: icdf(&[30531]),
            mv_joint: icdf(&[4096, 11264, 19328]),
            mv_sign: std::array::from_fn(|_| icdf(&[16384])),
            mv_classes: std::array::from_fn(|_| {
                icdf(&[
                    28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767,
                ])
            }),
            mv_class0: std::array::from_fn(|_| icdf(&[27648])),
            mv_class_n: std::array::from_fn(|_| {
                [
                    icdf(&[17408]),
                    icdf(&[17920]),
                    icdf(&[18944]),
                    icdf(&[20480]),
                    icdf(&[22528]),
                    icdf(&[24576]),
                    icdf(&[28672]),
                    icdf(&[29952]),
                    icdf(&[29952]),
                    icdf(&[30720]),
                ]
            }),
            part_bl8: PART_BL8_CDF.iter().map(|r| icdf(r)).collect(),
            txsz: [
                TXSZ_CAT0_CDF.iter().map(|r| icdf(r)).collect(),
                TXSZ_CAT1_CDF.iter().map(|r| icdf(r)).collect(),
                TXSZ_CAT2_CDF.iter().map(|r| icdf(r)).collect(),
                TXSZ_CAT3_CDF.iter().map(|r| icdf(r)).collect(),
            ],
            part_split: PART_SPLIT_CDF
                .iter()
                .map(|lvl| lvl.iter().map(|r| icdf(r)).collect())
                .collect(),
            kf_y: {
                let mut v = Vec::with_capacity(25);
                #[allow(clippy::needless_range_loop)]
                for a in 0..5 {
                    for l in 0..5 {
                        v.push(icdf(&KF_Y_MODE_CDF[a][l]));
                    }
                }
                v
            },
            angle_delta: ANGLE_DELTA_CDF.iter().map(|r| icdf(r)).collect(),
            filter_intra: FILTER_INTRA_CDF
                .iter()
                .map(|&threshold| icdf(&[threshold]))
                .collect(),
            filter_intra_mode: icdf(&FILTER_INTRA_MODE_CDF),
            palette_y_mode: palette_y_mode_cdfs(),
            palette_y_size: palette_y_size_cdfs(),
            palette_uv_mode: [icdf(&[32461]), icdf(&[21488])],
            palette_y_color: palette_y_color_cdfs(),
            palette_uv_size: palette_uv_size_cdfs(),
            palette_uv_color: palette_uv_color_cdfs(),
            cfl_sign: icdf(&CFL_SIGN_CDF),
            cfl_alpha: CFL_ALPHA_CDF.iter().map(|r| icdf(r)).collect(),
            uv_mode: {
                let mut v = Vec::with_capacity(26);
                #[allow(clippy::needless_range_loop)]
                for m in 0..13 {
                    v.push(icdf(&UV_MODE_NOCFL_CDF[m]));
                }
                #[allow(clippy::needless_range_loop)]
                for m in 0..13 {
                    v.push(icdf(&UV_MODE_CFL_CDF[m]));
                }
                v
            },
            txtp: TXTP_INTRA1_TX8.iter().map(|r| icdf(r)).collect(),
            txtp16: TXTP_INTRA2_TX16.iter().map(|r| icdf(r)).collect(),
            txtp4: TXTP_INTRA1_TX4.iter().map(|r| icdf(r)).collect(),
            txb_skip,
            base_tok,
            br_tok,
            eob_base,
            eob_hi,
            dc_sign: [
                Q::DC_SIGN_Q[qctx][0].iter().map(|&v| icdf(&[v])).collect(),
                Q::DC_SIGN_Q[qctx][1].iter().map(|&v| icdf(&[v])).collect(),
            ],
            eob_bin_16_c: icdf(&Q::EOB_BIN_16_CHROMA[qctx]),
            eob_bin_16_l: icdf(&Q::EOB_BIN_16_LUMA[qctx]),
            eob_bin_32_c: icdf(&Q::EOB_BIN_32_CHROMA[qctx]),
            eob_bin_32_l: icdf(&Q::EOB_BIN_32_LUMA[qctx]),
            eob_bin_64_l: icdf(&Q::EOB_BIN_64_LUMA[qctx]),
            eob_bin_16_l1d: icdf(&Q::EOB_BIN_16_LUMA_1D[qctx]),
            eob_bin_64_l1d: icdf(&Q::EOB_BIN_64_LUMA_1D[qctx]),
            eob_bin_128_l1d: icdf(&Q::EOB_BIN_128_LUMA_1D[qctx]),
            eob_bin_64_c: icdf(&Q::EOB_BIN_64_CHROMA[qctx]),
            eob_bin_256_l: icdf(&Q::EOB_BIN_256_LUMA[qctx]),
            eob_bin_256_c: icdf(&Q::EOB_BIN_256_CHROMA[qctx]),
            eob_bin_128_c: icdf(&Q::EOB_BIN_128_CHROMA[qctx]),
            eob_bin_128_l: icdf(&Q::EOB_BIN_128_LUMA[qctx]),
            eob_bin_1024_l: icdf(&Q::EOB_BIN_1024_LUMA[qctx]),
            eob_bin_1024_c: icdf(&Q::EOB_BIN_1024_CHROMA[qctx]),
            eob_bin_512_c: icdf(&Q::EOB_BIN_512_CHROMA[qctx]),
            eob_bin_512_l: icdf(&Q::EOB_BIN_512_LUMA[qctx]),
            // AV1 Default_Delta_Q_Cdf = AOM_CDF4(28160, 32120, 32677); a single
            // (context-free) 4-symbol CDF for the delta-q magnitude token. Adapts
            // like every other symbol via OdEcEncoder::encode_symbol.
            delta_q: icdf(&[28160, 32120, 32677]),
            // Default LrWiener (use_wiener) CDF (AV1 Default_Wiener_Restore_Cdf).
            wiener_restore: wiener_restore_icdf(),
        }
    }
}

const AQ_RES_LOG2: u8 = 2;
pub(crate) const AQ_DELTA_Q_RES_LOG2: u8 = AQ_RES_LOG2;
const AQ_MAX_STEPS: i32 = 12;

/// Base-shift experiment scaffolding. **Unreachable as shipped.**
///
/// `BASEQ_SHIFT_K` is a plain const with no setter, and `baseq_shift` returns 0
/// unconditionally when it is 0. So `dispatch.rs`'s `if shift != 0` block never
/// runs, `VarianceBoost::base_shift` is always 0, and the
/// `if delta < 0 && self.aq.base_shift > 0` re-anchoring branch in
/// `aq_sb_target` is dead with it.
///
/// Unlike `aq_slope`/`aq_max_delta` (revivable via
/// `EncodeConfig::with_variance_boost(false)`), there is NO configuration that
/// makes this live — verified by tracing every assignment to `base_shift`.
/// Safe to delete along with `baseq_shift`, the `base_shift` field, the
/// dispatch block and the `aq_sb_target` branch.
pub(crate) const BASEQ_SHIFT_K: i32 = 0;
pub(crate) fn baseq_shift(base_q_idx: u8) -> i32 {
    let qi = base_q_idx as i32;
    let k = BASEQ_SHIFT_K;
    if k == 0 || qi <= 40 {
        0
    } else if qi < 56 {
        k * (qi - 40) / 16
    } else if qi <= 112 {
        k
    } else if qi < 144 {
        k * (144 - qi) / 32
    } else {
        0
    }
}
/// qindex per unit of log-activity (how hard flat vs busy regions are pushed).
///
/// ONLY REACHED WITH VARIANCE BOOST DISABLED. Its sole consumer is
/// `aq_target_qidx`, which lives in the `else` of `if self.aq.vb_enabled` in
/// `aq_sb_target`; Variance Boost is on by default, so the default encoder
/// never evaluates this. A liveness sweep therefore reports it as dead — it is
/// not: `EncodeConfig::variance_boost(false)` makes this path live again.
/// Do not delete it, and do not tune it against default-config measurements.
fn aq_slope() -> f32 {
    crate::tuning::get().aq_slope
}
/// per-superblock qindex delta clamp, before res-quantization.
/// Same reachability caveat as [`aq_slope`].
fn aq_max_delta() -> f32 {
    crate::tuning::get().aq_max_delta
}

/// Variance Boost configuration, carried from the encoder option down to the
/// per-tile [`AqCtx`]. `enabled == false` selects the classic whole-SB AQ (the
/// shipping default); when enabled the per-SB target comes from the octile of the
/// 64 8x8-subblock variances instead. Knobs mirror `av2/mod.rs::Tuning`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VarianceBoost {
    pub enabled: bool,
    pub octile: u8,
    pub strength: f32,
    pub boost_only: bool,
    /// Dark-structured-detail protection, combined with the variance boost by `max`.
    /// Independent of `enabled`: fires in both the Variance Boost and classic-AQ paths.
    pub dark: DarkAq,
    /// Mid-band base-q shift this frame was encoded with (see [`baseq_shift`]);
    /// the AQ protection side anchors to `base_q - base_shift`.
    pub base_shift: i32,
    pub qm: QmLevels,
}

impl VarianceBoost {
    /// Disabled — classic whole-SB AQ, byte-identical to the pre-VB encoder.
    pub(crate) fn off() -> Self {
        VarianceBoost {
            enabled: false,
            octile: 6,
            strength: 1.0,
            boost_only: false,
            dark: DarkAq::off(),
            base_shift: 0,
            qm: QmLevels::FLAT,
        }
    }

    pub(crate) fn on() -> Self {
        VarianceBoost {
            enabled: true,
            octile: 6,
            strength: 1.0,
            boost_only: false,
            dark: DarkAq::on(),
            base_shift: 0,
            qm: QmLevels::FLAT,
        }
    }
}

#[inline(never)]
pub(crate) fn sb_activity(
    yp: &[u16],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
) -> f32 {
    let h = height.saturating_sub(sb_y).min(64);
    let w = width.saturating_sub(sb_x).min(64);
    if h == 0 || w == 0 {
        return 0.0;
    }
    let (sum, sum_sq) = block_moments_i32(yp, pw, sb_x, sb_y, w, h);
    let n = (h * w) as f32;
    let mean = sum as f32 / n;
    let var = (sum_sq as f32 / n - mean * mean).max(0.0);
    dirty_log1pf(var)
}

fn tile_ref_activity(yp: &[u16], pw: usize, w: usize, h: usize) -> f32 {
    let mut sum = 0f32;
    let mut cnt = 0f32;
    for sb_y in (0..h).step_by(64) {
        for sb_x in (0..w).step_by(64) {
            sum += sb_activity(yp, pw, sb_y, sb_x, w, h);
            cnt += 1.0;
        }
    }
    if cnt > 0.0 { sum / cnt } else { 5.0 }
}

/// Tile mean of log1p(octile-6 8x8 subblock variance) — the cut-side anchor
/// in the SAME domain as the per-SB `picked` value.
fn tile_ref_picked(yp: &[u16], pw: usize, w: usize, h: usize, octile: u8, bd: u8) -> f32 {
    let var_scale = 1.0 / (1u32 << (2 * (bd - 8))) as f32;
    let mut sum = 0f32;
    let mut cnt = 0f32;
    for sb_y in (0..h).step_by(64) {
        for sb_x in (0..w).step_by(64) {
            let mut subvars = [0f32; 64];
            let filled = aq_sb_subblock_variances(yp, pw, sb_y, sb_x, w, h, &mut subvars);
            if filled == 0 {
                continue;
            }
            let picked = crate::aq_common::sb_octile_variance(&mut subvars, octile);
            sum += dirty_log1pf(picked * var_scale);
            cnt += 1.0;
        }
    }
    if cnt > 0.0 { sum / cnt } else { 5.0 }
}

fn aq_params() -> (f32, f32, f32) {
    // (slope, max delta, coarsen scale). Coarsen scale 1.0 == pure variance.
    (aq_slope(), aq_max_delta(), 1.0)
}

fn aq_target_qidx(base_q: i32, activity: f32, ref_act: f32) -> i32 {
    let (slope, maxd, coarsen) = aq_params();
    let mut delta = (activity - ref_act) * slope;
    if delta > 0.0 {
        // Busy/textured: quantization error is masked, so coarsening is "free"
        // perceptually. `coarsen` < 1 spends fewer of those saved bits there and
        // more refining flats (better for perceptual/SSIM); = 1 is pure variance.
        delta *= coarsen;
    }
    let delta = delta.clamp(-maxd, maxd);
    (base_q + delta.fast_round() as i32).clamp(1, 255)
}

fn aq_sb_subblock_variances(
    yp: &[u16],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    width: usize,
    height: usize,
    out: &mut [f32; 64],
) -> usize {
    let mut filled = 0usize;
    let mut acc = 0f32;
    for (by, row) in out.as_chunks_mut::<8>().0.iter_mut().take(8).enumerate() {
        for (bx, out) in row.iter_mut().enumerate() {
            let y0 = sb_y + by * 8;
            let x0 = sb_x + bx * 8;
            let h = height.saturating_sub(y0).min(8);
            let w = width.saturating_sub(x0).min(8);
            if h == 0 || w == 0 {
                *out = f32::NAN; // out-of-frame, patched below
                continue;
            }
            let (sum, sum_sq) = block_moments_i32(yp, pw, x0, y0, w, h);
            let n = (h * w) as f32;
            let mean = sum as f32 / n;
            let var = (sum_sq as f32 / n - mean * mean).max(0.0);
            *out = var;
            acc += var;
            filled += 1;
        }
    }
    if filled == 0 {
        out.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    let mean = acc / filled as f32;
    for v in out.iter_mut() {
        if v.is_nan() {
            *v = mean;
        }
    }
    filled
}

/// Per-tile adaptive-quantization state held on the [`LossyTile`]. When
/// `enabled` is false every method is a no-op and the tile quantizes at the
/// fixed base, byte-identical to the pre-AQ encoder.
struct AqCtx {
    enabled: bool,
    /// frame `base_q_idx`: the anchor the per-SB deltas are measured from and the
    /// value `CurrentQIndex` is reset to at tile start.
    base_q: u8,
    /// `delta_q_res` (log2): the per-SB step is `1 << res_log2` qindex units.
    res_log2: u8,
    /// decoder `CurrentQIndex`, updated by each signaled delta; reset to `base_q`
    /// at the start of the tile (delta-Q does not persist across tiles).
    cur_qidx: i32,
    /// Decoder qindex on entry to the current superblock. A whole-SB skip does
    /// not execute `read_delta_qindex`, so its tentative AQ change is rolled
    /// back to this value.
    prev_qidx: i32,
    /// tile mean activity, the zero-delta reference (see [`tile_ref_activity`]).
    ref_act: f32,
    /// Same-domain cut reference: tile mean of log1p(octile-6 8x8 variance) —
    /// the anchor the coarsen side compares against (the whole-SB `ref_act`
    /// is systematically higher than the octile domain, which had silently
    /// disabled cuts; see `variance_boost_delta`).
    ref_pick: f32,
    /// armed at the start of each superblock; the first coded block emits the
    /// `read_delta_qindex` token and disarms it (spec `ReadDeltas`).
    read_deltas: bool,
    /// `reducedDeltaQIndex` (pre-`<<res`) to emit at the first block of the SB.
    pending: i32,
    /// When true, use the Variance Boost scheme (octile of 8x8 subblock variances)
    /// instead of the classic whole-SB variance for the per-SB target. Gated behind
    /// the `variance_boost` encoder option; off => classic AQ, byte-identical.
    vb_enabled: bool,
    /// Variance Boost selectivity octile (1..=8). Default 6 (SVT-AV1-PSY default).
    vb_octile: u8,
    /// Variance Boost strength multiplier (1.0 = nominal).
    vb_strength: f32,
    /// When true, only boost low-variance SBs (net-negative, spends bits). When
    /// false, also coarsen high-variance SBs to keep the rate roughly matched.
    vb_boost_only: bool,
    /// Dark-structured-detail protection (see [`DarkAq`]); independent of `vb_enabled`.
    dark: DarkAq,
    /// Mid-band base-shift the frame was encoded with; protection deltas
    /// anchor to `base_q - base_shift` (see `aq_sb_target`).
    base_shift: i32,
}

impl AqCtx {
    fn off() -> Self {
        AqCtx {
            enabled: false,
            base_q: 0,
            res_log2: 0,
            cur_qidx: 0,
            prev_qidx: 0,
            ref_act: 0.0,
            ref_pick: 0.0,
            read_deltas: false,
            pending: 0,
            vb_enabled: false,
            vb_octile: 6,
            vb_strength: 1.0,
            vb_boost_only: false,
            dark: DarkAq::off(),
            base_shift: 0,
        }
    }
}

/// One superblock's precomputed AQ state: the post-clamp qindex the SB codes
/// with and the signaled `reducedDeltaQIndex` step count that got it there.
/// Produced in raster order by `precompute_aq_grid`; indexed `row * cols + col`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AqCell {
    /// Raster-chain qindex BEFORE this SB (the anchor). Cells must start from
    /// this — NOT the worker's own accumulator: the clamp(1, 255) at
    /// near-lossless qindex breaks the shared-lattice property that otherwise
    /// makes any-prev derivations converge (kodak20 q99 t8 silent corruption).
    prev: u8,
    newq: u8,
    steps: i32,
}

/// Whole-frame lossy encoder state.
struct LossyTile<'a> {
    /// One-entry cache of the current superblock's activity, for the
    /// LOCAL masking reference (see `perceptual_rd_scale`). Blocks are coded
    /// in SB order so a single slot hits almost always. Key is (sb_x, sb_y).
    sb_act_cache: std::cell::Cell<(u32, u32, f32)>,
    /// Pre-resolved per-encode forward-DCT implementations.
    dct: DctDispatch,
    /// Pre-resolved per-encode inverse-transform implementations.
    idct: IdctDispatch,
    intrapred: IntraPredDispatch,
    kmeans: KmeansDispatch,
    rd: crate::rd_sse::RdDispatch,
    bd: u8,
    quant: Quant,
    cquant: Quant,
    w: usize,
    h: usize,
    cw: usize,   // chroma plane width (= w for 4:4:4, w/2 for 4:2:2 and 4:2:0)
    ss422: bool, // chroma horizontally subsampled (4:2:2)
    ss420: bool, // chroma horizontally + vertically subsampled (4:2:0)
    mono: bool,  // monochrome: code luma only (NumPlanes=1, no chroma syntax)
    /// Frame-level U/V weight used by the shared-tree partition proxy.
    chroma_part_weight: f32,
    allow_intrabc: bool,
    /// Search the screen-content tools (palette). Frame-level; see
    /// [`crate::EncodeConfig::with_screen_content`].
    screen_content: bool,
    ibc_mv: Vec<Option<(i16, i16)>>,
    src: &'a [Vec<u16>; 3],
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
    /// Whether the current superblock already recorded its `read_cdef()` trace
    /// point (the first non-skip block carries the per-unit `cdef_idx`).
    cdef_point_marked: bool,
    /// Memo for the partition proxy's palette candidate (rd_cost_square):
    /// keyed by packed (px, py, dim). The bottom-up partition search re-prices
    /// the same square up to ~3x; the palette candidate (histogram + weighted
    /// k-means + map + distortion) is the costliest per-call piece, and its
    /// inputs (source, quant, decision CDFs) are immutable per tile.
    pal_est_cache: std::cell::RefCell<HashMap<u64, [(f32, f32); 3]>>,
    /// Lazily built flat f32 cost tables for the decision `palette_y_color`
    /// CDFs ([size-2][5 map ctx][8 symbols]): the index-map rate walk touches
    /// one entry per map cell, and the triple-Vec `cdf_cost` pointer chase was
    /// the walk's hottest load. Values are exactly `cdf_cost(..)`, so sums are
    /// bit-identical. Decision CDFs are immutable per tile (the same
    /// assumption `pal_est_cache` already relies on).
    #[allow(clippy::type_complexity)]
    pal_y_cost: std::cell::RefCell<Option<Box<[[[f32; 8]; 5]; 7]>>>,
    /// Exact, epoch-stamped memo for derived chroma block RD costs shared by
    /// multiple partition families under the same perceptual multiplier.
    chroma_rd_cache: std::cell::RefCell<HashMap<u128, (u64, f32)>>,
    /// Memo for the full 16-level partition decision (`rd_choice_16`): the
    /// bottom-up pricing evaluates each 16-block up to ~3x (64-level refine,
    /// 32-level SPLIT leg, then decode_sb's own descend). Epoch stamping keeps
    /// reuse bit-exact after any emitted block changes prediction context.
    rd16_cache: std::cell::RefCell<HashMap<u32, (u64, Part16, f32)>>,
    /// Epoch-stamped memo for the decision-stage rect16 leaf proxy
    #[allow(clippy::type_complexity)]
    rect_leaf_cache: std::cell::RefCell<HashMap<(u64, u32), (u64, f32, f32)>>,
    /// Reusable trial buffers (see [`CoderScratch`]); `Rc<RefCell>` so the
    /// [`SBuf`] guards and the `&self` decision paths can reach the pool
    /// without borrowing the tile.
    scratch: std::rc::Rc<std::cell::RefCell<CoderScratch>>,
    /// Exact 8x8-as-four-4x4 luma proxy, reused by the 16-level decision and
    /// the eventual 8-level decision within one reconstruction epoch.
    split4_rd_cache: std::cell::RefCell<HashMap<u128, (u64, f32)>>,
    /// Bumped on every emitted (or sink-captured) leaf block.
    emit_epoch: std::cell::Cell<u64>,
    /// Tile-wide IntraBC exact-match index (see `LossyIbcIndex`), owned by
    /// `encode_one_tile`.
    ibc_index: Option<&'a std::sync::OnceLock<LossyIbcIndex>>,
    /// Wavefront-capture read view of the shared finished-recon planes.
    ibc_shared: Option<IbcSharedRecon>,
    enc: OdEcEncoder,
    cdfs: Cdfs,
    updating_cdf: bool,
    /// Frozen frame-initial CDF snapshot used by every DECISION-side rate
    /// estimate (RDOQ trellis, filter-intra / angle-delta mode costs).
    dec_cdfs: Box<Cdfs>,
    /// Decision capture/replay mode (see `coder/replay.rs`). `Off` in
    /// production until the wavefront lands.
    sb_mode: SbMode,
    /// Captured (or replayed-from) RD decisions, call-order aligned.
    rec: DecisionRecord,
    /// Replay read positions into `rec`.
    cur: RecordCursor,
    /// RDO effort: [`Speed::Slow`] (default) or [`Speed::Fast`] (winner-only
    /// RDOQ, DCT-only transform choice, reduced intra mode set).
    speed: Speed,
    /// Adaptive-quantization state; `AqCtx::off()` unless enabled per tile.
    aq: AqCtx,
    /// Global luma Wiener filter to signal per superblock (`read_lr`), or `None`
    /// for RESTORE_NONE (no per-SB LR syntax emitted). When set, every 64x64
    /// restoration unit codes `use_wiener = 1` with these taps, delta-coded
    /// against the running reference `lr_ref_*` (spec 5.11.58).
    wiener: Option<crate::wiener::WienerUnit>,
    /// Running Wiener tap reference for delta coding (horizontal, vertical), one
    /// per coded tap. Reset to the spec midpoints at the start of the tile.
    lr_ref_h: [i32; 3],
    lr_ref_v: [i32; 3],
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

// Keep the state type and shared imports in this module while splitting its
// implementation by coding responsibility. `include!` preserves private field
// access without widening the encoder's internal API.

include!("coder/scratch.rs");
include!("coder/replay.rs");
include!("coder/lossy_state.rs");
include!("coder/palette.rs");
include!("coder/partition_search.rs");
include!("coder/block16.rs");
include!("coder/block8.rs");
include!("coder/block32.rs");
include!("coder/block64.rs");
include!("coder/superblock.rs");

#[allow(clippy::too_many_arguments)]
#[inline]
fn sse_recon<const N: usize, const D: usize>(
    rd: &crate::rd_sse::RdDispatch,
    pred: &[i32; N],
    resid: &[i32; N],
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    bd: u8,
) -> i64 {
    debug_assert_eq!(N, D * D);
    rd.sse_recon(pred, resid, src, stride, px, py, D, D, bd)
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
    0.4
}

fn prdo_k_444() -> f32 {
    0.5
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

fn none32_split_bias() -> f32 {
    crate::tuning::get().none32_split_bias
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
/// Same split-favoring thumb for the 64x64 SB NONE-vs-SPLIT decision (see
/// `choose_64`). BLOCK_64X64 shares one prediction over a large area, so the
/// distortion proxy is even coarser than at 32x32; the bias guards detail.
fn none64_split_bias() -> f32 {
    crate::tuning::get().none64_split_bias
}
/// Master switch for the whole-superblock BLOCK_64X64 intra path (4:2:0 only).
pub static BLOCK64_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[inline]
fn chroma_part_rd_weight(ss420: bool, ss422: bool, _src: &[Vec<u16>; 3], _bd: u8) -> f32 {
    if ss420 || ss422 { 0.0625 } else { 0.03125 }
}

pub(crate) fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Pad a `w`×`h` plane to `w8`×`h8` (≥ originals) by replicating the last
/// in-frame row/column. AV1's coded block grid is always 8-pixel aligned
/// (`MiCols = ((w+7)>>3)<<1`), so frames whose dimensions are not multiples of 8
/// are coded on the padded grid and the decoder crops back to the signaled
/// frame size. Edge replication keeps the (cropped-away) padding cheap to code.
pub(crate) fn pad_to_mult8<T: Copy>(src: &[T], w: usize, h: usize, w8: usize, h8: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w8 * h8);
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        out.extend_from_slice(row);
        out.resize(out.len() + (w8 - w), row[w - 1]);
    }
    for _ in h..h8 {
        out.extend_from_within((h - 1) * w8..h * w8);
    }
    out
}

/// Smallest `k` such that `(blk << k) >= target` (AV1 spec `tile_log2`).
fn tile_log2(blk: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

/// AV1 `increment_*_log2` bit sequence signaling `target` to a decoder that
/// starts at `min` and reads bits while its running value is `< max`: a `1` for
/// each step up, then a terminating `0` when `target < max` (at `max` the
/// decoder's loop ends on its own and reads no further bit).
fn increment_bits(min: u32, max: u32, target: u32) -> Vec<bool> {
    let mut v = Vec::new();
    let mut cur = min;
    while cur < max {
        if cur < target {
            v.push(true);
            cur += 1;
        } else {
            v.push(false);
            break;
        }
    }
    v
}

/// Full tiling decision: the chosen `(TileColsLog2, TileRowsLog2)` plus the
/// `increment_*_log2` bit sequences the frame header must emit to signal them.
pub(crate) struct Tiling {
    tcl: u32,
    trl: u32,
    cols_incr: Vec<bool>,
    rows_incr: Vec<bool>,
}

/// Pick a tiling for a frame of `sb_cols` x `sb_rows` superblocks.
fn plan_tiling(sb_cols: u32, sb_rows: u32, target_tiles: usize) -> Tiling {
    const MAX_TILE_WIDTH_SB: u32 = 4096 / 64; // 64
    const MAX_TILE_AREA_SB: u32 = (4096 * 2304) / (64 * 64); // 2304
    let min_log2_tile_cols = tile_log2(MAX_TILE_WIDTH_SB, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(MAX_TILE_AREA_SB, sb_rows * sb_cols));

    // Start at the spec minimum.
    let mut tcl = min_log2_tile_cols.min(max_log2_tile_cols);
    let mut trl = min_log2_tiles.saturating_sub(tcl).min(max_log2_tile_rows);

    // Climb toward target_tiles, splitting whichever side currently has the
    // larger tiles so the grid stays balanced.
    let target = target_tiles.max(1) as u32;
    while (1u32 << (tcl + trl)) < target {
        let can_col = tcl < max_log2_tile_cols;
        let can_row = trl < max_log2_tile_rows;
        if !can_col && !can_row {
            break;
        }
        let col_span = sb_cols >> tcl; // SBs per tile column (approx)
        let row_span = sb_rows >> trl; // SBs per tile row (approx)
        if can_col && (!can_row || col_span >= row_span) {
            tcl += 1;
        } else {
            trl += 1;
        }
    }

    let cols_incr = increment_bits(min_log2_tile_cols, max_log2_tile_cols, tcl);
    // The decoder derives its row minimum from the (now decoded) TileColsLog2.
    let min_log2_tile_rows = min_log2_tiles.saturating_sub(tcl);
    let rows_incr = increment_bits(min_log2_tile_rows, max_log2_tile_rows, trl);
    Tiling {
        tcl,
        trl,
        cols_incr,
        rows_incr,
    }
}

/// Uniform-spacing tile start offsets (in SB units), matching the decoder's
/// `for (startSb = 0; startSb < sbs; startSb += sizeSb)` loop. The returned vec
/// has one entry per tile; the implied end of tile `i` is `starts[i+1]` (or
/// `sbs` for the last). The tile count may be **less** than `1 << log2`.
fn tile_starts_sb(sbs: u32, log2: u32) -> Vec<u32> {
    let size_sb = sbs.div_ceil(1 << log2);
    let mut starts = Vec::new();
    let mut s = 0;
    while s < sbs {
        starts.push(s);
        s += size_sb;
    }
    starts
}

fn crop_plane<T: Copy>(
    src: &[T],
    full_w: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(tw * th);
    for r in 0..th {
        let s = (y0 + r) * full_w + x0;
        out.extend_from_slice(&src[s..s + tw]);
    }
    out
}

fn stitch_plane(
    dst: &mut [u16],
    full_w: usize,
    x0: usize,
    y0: usize,
    tile: &[u16],
    tw: usize,
    th: usize,
) {
    for r in 0..th {
        let d = (y0 + r) * full_w + x0;
        dst[d..d + tw].copy_from_slice(&tile[r * tw..(r + 1) * tw]);
    }
}

/// Default (untrained) inverse CDF for the `use_wiener` flag, shared by tile
/// entropy init and the LR replay path.
pub(crate) fn wiener_restore_icdf() -> Vec<u16> {
    icdf(&[11570])
}

/// `read_lr` symbols owed by the superblock at tile-local `(sb_x, sb_y)` (spec
/// 5.11.57); geometry rationale on `LossyTile::emit_lr_sb`.
#[allow(clippy::too_many_arguments)]
fn emit_lr_sb_syms(
    enc: &mut OdEcEncoder,
    wr_cdf: &mut [u16],
    lr_ref_v: &mut [i32; 3],
    lr_ref_h: &mut [i32; 3],
    unit: &crate::wiener::WienerUnit,
    frame_x0: usize,
    frame_y0: usize,
    frame_w: usize,
    frame_h: usize,
    sb_x: usize,
    sb_y: usize,
) {
    const UNIT: usize = 64;
    const MI: usize = 4;
    let count_units = |frame: usize| -> usize { 1.max((frame + (UNIT >> 1)) / UNIT) };
    let unit_rows = count_units(frame_h);
    let unit_cols = count_units(frame_w);
    // Frame-absolute superblock position in 4x4 MI units (luma).
    let r = (frame_y0 + sb_y) / MI;
    let c = (frame_x0 + sb_x) / MI;
    let sb_mi = UNIT / MI; // 16
    let urs = (r * MI).div_ceil(UNIT);
    let ure = unit_rows.min(((r + sb_mi) * MI).div_ceil(UNIT));
    let ucs = (c * MI).div_ceil(UNIT);
    let uce = unit_cols.min(((c + sb_mi) * MI).div_ceil(UNIT));
    for _ur in urs..ure {
        for _uc in ucs..uce {
            emit_lr_unit_syms(enc, wr_cdf, lr_ref_v, lr_ref_h, unit);
        }
    }
}

/// One `read_lr_unit` for a RESTORE_WIENER luma unit (spec 5.11.58): `use_wiener`,
/// then v/h taps signed-subexp coded against (and updating) the running refs.
fn emit_lr_unit_syms(
    enc: &mut OdEcEncoder,
    wr_cdf: &mut [u16],
    lr_ref_v: &mut [i32; 3],
    lr_ref_h: &mut [i32; 3],
    unit: &crate::wiener::WienerUnit,
) {
    use crate::wiener::{WIENER_TAPS_K, WIENER_TAPS_MAX, WIENER_TAPS_MIN};
    enc.encode_symbol(1, wr_cdf);
    for axis in 0..2 {
        let (taps, refs) = if axis == 0 {
            (unit.v, &mut *lr_ref_v)
        } else {
            (unit.h, &mut *lr_ref_h)
        };
        for j in 0..3usize {
            let lo = WIENER_TAPS_MIN[j];
            let hi = WIENER_TAPS_MAX[j] + 1; // exclusive high
            let k = WIENER_TAPS_K[j] as u32;
            enc.encode_signed_subexp_with_ref(taps[j], lo, hi, k, refs[j]);
            refs[j] = taps[j];
        }
    }
}

/// Pixel rectangle of one tile, in both luma and (subsampled) chroma coords.
#[derive(Clone, Copy)]
struct TileRect {
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
    cx0: usize,
    cy0: usize,
    ctw: usize,
    cth: usize,
}

/// Encoded output of one tile: its entropy-coded payload plus the tile-local
/// reconstruction (luma `tw*th`, chroma `ctw*cth`). Owned + `Send`, so it can be
/// produced on a worker thread and moved back to the caller.
struct TileOut {
    payload: Vec<u8>,
    trace: Option<Box<crate::odec::SymbolTrace>>,
    recon: [Vec<u16>; 3],
    skip8: Vec<bool>, // per-8x8 luma-unit skip flag (tile-local, row-major over ceil(tw/8))
    blk4: Vec<u8>,    // per-4x4 luma block WIDTH map (tile-local), for frame-level deblocking
    blk4h: Vec<u8>,   // per-4x4 luma block HEIGHT map (tile-local), for frame-level deblocking
    blk4v: Vec<bool>, // per-4x4 actual luma vertical-edge map
    blk4t: Vec<bool>, // per-4x4 actual luma horizontal-edge map
    pblk4: Vec<u8>,   // per-4x4 luma PREDICTION-block width map (chroma deblock)
    pblk4h: Vec<u8>,  // per-4x4 luma PREDICTION-block height map
    pblk4v: Vec<bool>,
    pblk4t: Vec<bool>,
}

/// Completed output of one parallel superblock decision. Unlike the old
/// `DecisionRecord`, this contains no search winners, coefficient copies, or
/// reconstruction copies: only the compact syntax operations consumed by the
/// raster entropy lane.
struct CapturedSb {
    tokens: crate::odec::EntropyTokens,
}

struct WavefrontPlanes {
    recon: [Vec<u16>; 3],
    skip8: Vec<bool>,
    blk4: Vec<u8>,
    blk4h: Vec<u8>,
    blk4v: Vec<bool>,
    blk4t: Vec<bool>,
    pblk4: Vec<u8>,
    pblk4h: Vec<u8>,
    pblk4v: Vec<bool>,
    pblk4t: Vec<bool>,
}

/// Parallel SB-wavefront capture pass over one tile: decide every superblock's
/// RD choices out of raster order (schedule `d = 2r + c`, so top, left AND
/// above-right neighbors are always finished), publish final reconstruction
/// and filter maps directly into disjoint shared regions, and stream compact
/// entropy operations to the raster packer. The packer restores full CDF
/// adaptivity without traversing blocks a second time.
///
/// Safety of the parallelism rests on three proven properties:
/// - decisions read reconstruction only inside the halo bands;
/// - every ctx-array read stays inside the SB's exact own column/row segment
///   (zero margin—neighboring `l_*` rows have a different last writer under
///   diagonal order than under raster order);
/// - decisions never read the adaptive CDFs or encoder state (frozen
///   `dec_cdfs`, Phase 1) — so each worker captures into a sink encoder.
///
/// Shared state is written disjointly: each cell writes only its own SB block
/// of the `done` recon planes and its own segments of the ctx handoff arrays
/// (top→bottom handoff for `a_*`, left→right for `l_*`), exactly mirroring
/// what the serial raster loop would have left there for that reader.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // `p` indexes recon planes AND their writers in lockstep
fn wavefront_capture(
    nthreads: usize,
    base_q_idx: u8,
    bd: u8,
    full_w: usize,
    full_h: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    tsrc: &[Vec<u16>; 3],
    r: &TileRect,
    speed: Speed,
    dct: DctDispatch,
    idct: IdctDispatch,
    intrapred: IntraPredDispatch,
    kmeans: KmeansDispatch,
    rd: crate::rd_sse::RdDispatch,
    aq: bool,
    vb: &VarianceBoost,
    updating_cdf: bool,
    allow_intrabc: bool,
    ibc_index: Option<&std::sync::OnceLock<LossyIbcIndex>>,
    screen_content: bool,
    tx: std::sync::mpsc::Sender<(usize, CapturedSb)>,
) -> WavefrontPlanes {
    use crate::av2::helpers::{PlaneWriter, par_wavefront_pool_with};
    const HB: usize = 4; // halo band thickness (AV1 intra needs 1 px; generous)
    let sb_rows = r.th.div_ceil(64);
    let sb_cols = r.tw.div_ceil(64);
    // `mk_tile` runs once per worker.  Computing this inside it made every
    // worker rescan the complete source tile when AQ/variance boost was on.
    // The reference activity is immutable and identical for all workers.
    let ref_act = aq.then(|| tile_ref_activity(&tsrc[0], r.tw, r.tw, r.th));
    let mk_tile = || {
        let mut t = if mono {
            LossyTile::new_mono(base_q_idx, bd, r.tw, r.th, tsrc, vb.qm)
        } else {
            match (sub_x, sub_y) {
                (0, 0) => LossyTile::new(base_q_idx, bd, r.tw, r.th, tsrc, vb.qm),
                (1, 0) => LossyTile::new_422(base_q_idx, bd, r.tw, r.th, tsrc, vb.qm),
                _ => LossyTile::new_420(base_q_idx, bd, r.tw, r.th, tsrc, vb.qm),
            }
        }
        .with_dispatch(dct, idct, intrapred, kmeans, rd)
        .with_speed(speed)
        .with_intrabc(allow_intrabc, ibc_index)
        .with_screen_content(screen_content)
        .with_updating_cdf(updating_cdf);
        t.frame_x0 = r.x0;
        t.frame_y0 = r.y0;
        t.frame_w = full_w;
        t.frame_h = full_h;
        if let Some(ref_act) = ref_act {
            t.enable_aq(base_q_idx, ref_act, vb);
        }
        // Decisions run normally, but their final syntax is captured directly
        // as semantic entropy tokens. There is no winner/coeff replay record.
        t.sb_mode = SbMode::Off;
        let slots = t.cdfs.semantic_slots();
        t.enc.set_semantic_cdfs(&slots);
        // Capture records the table identity + symbol. Only the raster packer
        // consumes live probabilities, so adapting worker-private CDF copies
        // is duplicate work (decision RDO reads `dec_cdfs`, never `cdfs`).
        t.enc.updating_cdf = false;
        t.enc.sink = true;
        t
    };
    // Prototype: geometry, frame-initial ctx array contents, and the AQ grid
    // (identical to the serial pass's — bit-exact by `aq_grid_matches_serial`).
    let proto = mk_tile();
    let aq_grid = proto.precompute_aq_grid();
    let mut done = [
        vec![0u16; proto.recon[0].len()],
        vec![0u16; proto.recon[1].len()],
        vec![0u16; proto.recon[2].len()],
    ];
    let stride4 = r.tw / 4;
    let h4 = r.th / 4;
    let stride8 = r.tw.div_ceil(8);
    let h8 = r.th.div_ceil(8);
    let mut skip8 = vec![true; stride8 * h8];
    let mut blk4 = vec![0u8; stride4 * h4];
    let mut blk4h = vec![0u8; stride4 * h4];
    let mut blk4v = vec![false; stride4 * h4];
    let mut blk4t = vec![false; stride4 * h4];
    let mut pblk4 = vec![0u8; stride4 * h4];
    let mut pblk4h = vec![0u8; stride4 * h4];
    let mut pblk4v = vec![false; stride4 * h4];
    let mut pblk4t = vec![false; stride4 * h4];
    // Ctx handoff arrays, frame-initial (clones of a fresh tile's arrays).
    let mut h_arrs: Vec<Vec<u8>> = vec![
        proto.a_coef[0].clone(),
        proto.a_coef[1].clone(),
        proto.a_coef[2].clone(),
        proto.l_coef[0].clone(),
        proto.l_coef[1].clone(),
        proto.l_coef[2].clone(),
        proto.a_part.clone(),
        proto.l_part.clone(),
        proto.a_skip.clone(),
        proto.l_skip.clone(),
        proto.a_mode.clone(),
        proto.l_mode.clone(),
        proto.a_uv_mode.clone(),
        proto.l_uv_mode.clone(),
    ];
    let (ss422, ss420) = (proto.ss422, proto.ss420);
    let cw = proto.cw;
    let allow_ibc = proto.allow_intrabc;
    let (w4, h4) = (r.tw / 4, r.th / 4);
    let proto_a_tx: Vec<u8> = proto.a_tx.iter().map(|&v| v as u8).collect();
    let proto_l_tx: Vec<u8> = proto.l_tx.iter().map(|&v| v as u8).collect();
    drop(proto);
    // Shared IntraBC MV plane: each cell writes its own SB region after
    // finishing (packed (dy<<16)|dx, 0 = none — a zero DV cannot exist);
    // cells preload their predictor window from it (finished neighbors) so
    // the DV predictor sees exactly the serial pass's state.
    let mut ibc_plane: Vec<i32> = vec![0; if allow_ibc { w4 * h4 } else { 0 }];
    // Shared palette-context planes: the above/left neighbor palettes
    // (a_palette / l_palette), packed 9 x i32 per 4-sample position
    // ([len, c0..c7]). Handing these across cells lets the palette DECISION
    // pricing use the real color cache + mode context (it was deliberately
    // context-free before, which underprices palette in dense regions).
    let mut apal_plane: Vec<i32> = vec![0; w4 * 9];
    let mut lpal_plane: Vec<i32> = vec![0; h4 * 9];
    // The UV palette context needs the same handoff as the luma one. Without
    // it `a_palette_uv`/`l_palette_uv` kept whatever the WORKER last left in
    // them, so the UV palette symbols were coded against a context that
    // depended on the cell->worker assignment. The decoder recomputes the
    // correct context, so the stream desynced (volcanic.png, 4:4:4 q97: the
    // recon is bit-identical run to run, only the coded bits move).
    // Allocated only when there is chroma; mono never reads these.
    let uvpal = cw > 0;
    let mut apal_uv_plane: Vec<i32> = vec![0; if uvpal { w4 * 9 } else { 0 }];
    let mut lpal_uv_plane: Vec<i32> = vec![0; if uvpal { h4 * 9 } else { 0 }];
    let mut atx_plane: Vec<u8> = proto_a_tx;
    let mut ltx_plane: Vec<u8> = proto_l_tx;
    // Disjoint-write views. Plane p stride: luma w, chroma cw.
    let dws: Vec<PlaneWriter<u16>> = {
        let mut it = done.iter_mut();
        let l = it.next().unwrap();
        let u = it.next().unwrap();
        let v = it.next().unwrap();
        vec![
            PlaneWriter::new(l, r.tw),
            PlaneWriter::new(u, cw.max(1)),
            PlaneWriter::new(v, cw.max(1)),
        ]
    };
    let ibcw = PlaneWriter::new(&mut ibc_plane, w4.max(1));
    let apalw = PlaneWriter::new(&mut apal_plane, (w4 * 9).max(1));
    let lpalw = PlaneWriter::new(&mut lpal_plane, (h4 * 9).max(1));
    let apaluvw = PlaneWriter::new(&mut apal_uv_plane, (w4 * 9).max(1));
    let lpaluvw = PlaneWriter::new(&mut lpal_uv_plane, (h4 * 9).max(1));
    let atxw = PlaneWriter::new(&mut atx_plane, w4.max(1));
    let ltxw = PlaneWriter::new(&mut ltx_plane, h4.max(1));
    let hws: Vec<PlaneWriter<u8>> = h_arrs
        .iter_mut()
        .map(|a| {
            let stride = a.len().max(1);
            PlaneWriter::new(a, stride)
        })
        .collect();
    let skip8w = PlaneWriter::new(&mut skip8, stride8.max(1));
    let blk4w = PlaneWriter::new(&mut blk4, stride4.max(1));
    let blk4hw = PlaneWriter::new(&mut blk4h, stride4.max(1));
    let blk4vw = PlaneWriter::new(&mut blk4v, stride4.max(1));
    let blk4tw = PlaneWriter::new(&mut blk4t, stride4.max(1));
    let pblk4w = PlaneWriter::new(&mut pblk4, stride4.max(1));
    let pblk4hw = PlaneWriter::new(&mut pblk4h, stride4.max(1));
    let pblk4vw = PlaneWriter::new(&mut pblk4v, stride4.max(1));
    let pblk4tw = PlaneWriter::new(&mut pblk4t, stride4.max(1));
    par_wavefront_pool_with(
        nthreads,
        sb_rows,
        sb_cols,
        /* needs_above_right= */ true,
        mk_tile,
        |t: &mut LossyTile, row: usize, col: usize| {
            let (sb_x, sb_y) = (col * 64, row * 64);
            // Per-plane geometry: (subsample x shift, subsample y shift).
            let shift = |p: usize| -> (usize, usize) {
                if p == 0 {
                    (0, 0)
                } else {
                    (((ss420 || ss422) as usize), (ss420 as usize))
                }
            };
            // --- copy-in: recon halo bands from the finished-SB planes ---
            for p in 0..3 {
                if t.recon[p].is_empty() {
                    continue;
                }
                let (sx, sy) = shift(p);
                let pw = if p == 0 { t.w } else { t.cw };
                let ph = t.recon[p].len() / pw;
                let (bx, by) = (sb_x >> sx, sb_y >> sy);
                let (bw, bh) = (64usize >> sx, 64usize >> sy);
                let x0 = bx.saturating_sub(HB);
                let x1 = (bx + 2 * bw).min(pw);
                let y0 = by.saturating_sub(HB);
                // A worker may receive an unrelated next cell, so initialize
                // only samples that are unavailable to that cell. The valid
                // top and left halos are overwritten from `done` below; zeroing
                // them as part of the previous cell's teardown was redundant.
                for row2 in by..(by + bh).min(ph) {
                    t.recon[p][row2 * pw + bx..row2 * pw + x1].fill(0);
                }
                // SAFETY: halo regions belong to earlier diagonals (finished,
                // not being written); regions are in-plane.
                unsafe {
                    if by > y0 {
                        dws[p].copy_region_to(&mut t.recon[p], y0, x0, by - y0, x1 - x0);
                    }
                    if bx > x0 {
                        let yh = (by + bh).min(ph) - by;
                        dws[p].copy_region_to(&mut t.recon[p], by, x0, yh, bx - x0);
                    }
                }
            }
            // --- copy-in: exact own ctx segments from the handoff arrays ---
            let cx = sb_x >> ((ss420 || ss422) as usize);
            let cy = sb_y >> (ss420 as usize);
            let segs: [(usize, usize); 14] = [
                (sb_x / 4, 16), // a_coef[0] (luma 4x4 cols, 16 per SB)
                (cx / 4, 16 >> ((ss420 || ss422) as usize)),
                (cx / 4, 16 >> ((ss420 || ss422) as usize)),
                (sb_y / 4, 16), // l_coef[0]
                (cy / 4, 16 >> (ss420 as usize)),
                (cy / 4, 16 >> (ss420 as usize)),
                (sb_x / 8, 8),  // a_part
                (sb_y / 8, 8),  // l_part
                (sb_x / 4, 16), // a_skip
                (sb_y / 4, 16), // l_skip
                (sb_x / 4, 16), // a_mode
                (sb_y / 4, 16), // l_mode
                (sb_x / 4, 16), // a_uv_mode
                (sb_y / 4, 16), // l_uv_mode
            ];
            let arr_of = |t: &mut LossyTile<'_>, i: usize| -> *mut Vec<u8> {
                match i {
                    0 => &mut t.a_coef[0],
                    1 => &mut t.a_coef[1],
                    2 => &mut t.a_coef[2],
                    3 => &mut t.l_coef[0],
                    4 => &mut t.l_coef[1],
                    5 => &mut t.l_coef[2],
                    6 => &mut t.a_part,
                    7 => &mut t.l_part,
                    8 => &mut t.a_skip,
                    9 => &mut t.l_skip,
                    10 => &mut t.a_mode,
                    11 => &mut t.l_mode,
                    12 => &mut t.a_uv_mode,
                    _ => &mut t.l_uv_mode,
                }
            };
            for (i, &(s, n)) in segs.iter().enumerate() {
                // SAFETY: arr_of returns a field pointer used immediately,
                // no aliasing (one array at a time).
                let arr = unsafe { &mut *arr_of(t, i) };
                if arr.is_empty() {
                    continue;
                }
                let e = (s + n).min(arr.len());
                if e > s {
                    // SAFETY: the segment was last written by a finished cell
                    // (above SB for a_*, left SB for l_*) or is frame-initial.
                    unsafe { hws[i].copy_region_to(&mut arr[..], 0, s, 1, e - s) };
                }
            }
            // --- IntraBC: preload the DV-predictor window from the shared
            // MV plane and expose the finished-recon read view ---
            if allow_ibc {
                let (ibp, _ibl, ibs) = ibcw.read_view();
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let x0 = sbx4.saturating_sub(8);
                let x1 = (sbx4 + 24).min(w4);
                let y0 = sby4.saturating_sub(8);
                let y1 = (sby4 + 24).min(h4);
                for y in y0..y1 {
                    for x in x0..x1 {
                        // SAFETY: finished cells hold real values; unfinished
                        // cells hold the initial zeros, which decode to None —
                        // exactly the serial "not yet coded" state.
                        let v = unsafe { *ibp.add(y * ibs + x) };
                        t.ibc_mv[y * (t.w / 4) + x] = if v == 0 {
                            None
                        } else {
                            Some(((v >> 16) as i16, v as i16))
                        };
                    }
                }
                t.ibc_shared = Some(IbcSharedRecon {
                    planes: [dws[0].read_view(), dws[1].read_view(), dws[2].read_view()],
                });
            }
            // --- palette-context copy-in: own segments, packed 9xi32 ---
            {
                let (ap, _al, _s1) = apalw.read_view();
                let (lp, _ll, _s2) = lpalw.read_view();
                let (apu, _aul, _s3) = apaluvw.read_view();
                let (lpu, _lul, _s4) = lpaluvw.read_view();
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                // Same loop, four arrays: the UV pair is skipped entirely when
                // there is no chroma, so mono pays nothing for this.
                let uvn = if uvpal { 4 } else { 2 };
                for (arr, base, ptr, lim) in [
                    (&mut t.a_palette, sbx4, ap, w4),
                    (&mut t.l_palette, sby4, lp, h4),
                    (&mut t.a_palette_uv, sbx4, apu, w4),
                    (&mut t.l_palette_uv, sby4, lpu, h4),
                ]
                .into_iter()
                .take(uvn)
                {
                    for i in 0..16usize {
                        let pos = base + i;
                        if pos >= lim {
                            break;
                        }
                        // SAFETY: own segment, last written by the finished
                        // above/left cell (or frame-initial zeros = empty).
                        let n = unsafe { *ptr.add(pos * 9) } as usize;
                        let v = &mut arr[pos];
                        v.clear();
                        for k in 0..n.min(8) {
                            v.push(unsafe { *ptr.add(pos * 9 + 1 + k) });
                        }
                    }
                }
            }
            // --- tx-size ctx copy-in: own segments (i8 as raw bytes) ---
            {
                let (atp, _al, _s1) = atxw.read_view();
                let (ltp, _ll, _s2) = ltxw.read_view();
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let na = 16usize.min(w4.saturating_sub(sbx4));
                for i in 0..na {
                    // SAFETY: own segment, last written by the finished above
                    // cell (or frame-initial).
                    t.a_tx[sbx4 + i] = unsafe { *atp.add(sbx4 + i) } as i8;
                }
                let nl = 16usize.min(h4.saturating_sub(sby4));
                for i in 0..nl {
                    // SAFETY: own segment (left cell / frame-initial).
                    t.l_tx[sby4 + i] = unsafe { *ltp.add(sby4 + i) } as i8;
                }
            }
            // --- per-cell resets ---
            t.enc.begin_semantic_sink();
            t.cur = RecordCursor::default();
            t.cdef_point_marked = false;
            if !aq_grid.is_empty() {
                t.aq_begin_sb_cell(&aq_grid[row * sb_cols + col]);
            }
            t.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
            // --- write-out: copy each own recon block directly into the
            // finished planes, without a transient contiguous SB scratch ---
            for p in 0..3 {
                if t.recon[p].is_empty() {
                    continue;
                }
                let (sx, sy) = shift(p);
                let pw = if p == 0 { t.w } else { t.cw };
                let ph = t.recon[p].len() / pw;
                let (bx, by) = (sb_x >> sx, sb_y >> sy);
                let (bw, bh) = (64usize >> sx, 64usize >> sy);
                let w2 = (bx + bw).min(pw) - bx;
                let h2 = (by + bh).min(ph) - by;
                // SAFETY: own SB block — no other concurrent writer.
                unsafe { dws[p].copy_region_from(&t.recon[p], pw, by, bx, h2, w2) };
            }
            if allow_ibc {
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let bw = (sbx4 + 16).min(w4) - sbx4;
                let bh = (sby4 + 16).min(h4) - sby4;
                let mut buf = [0i32; 16 * 16];
                for y in 0..bh {
                    for x in 0..bw {
                        buf[y * bw + x] = match t.ibc_mv[(sby4 + y) * (t.w / 4) + sbx4 + x] {
                            Some((dy, dx)) => ((dy as u16 as i32) << 16) | (dx as u16 as i32),
                            None => 0,
                        };
                    }
                }
                // SAFETY: own SB region — no other concurrent writer.
                unsafe { ibcw.write_block(sby4, sbx4, bh, bw, &buf[..bw * bh]) };
            }
            {
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let uvn = if uvpal { 4 } else { 2 };
                for (arr, base, w, lim) in [
                    (&t.a_palette, sbx4, &apalw, w4),
                    (&t.l_palette, sby4, &lpalw, h4),
                    (&t.a_palette_uv, sbx4, &apaluvw, w4),
                    (&t.l_palette_uv, sby4, &lpaluvw, h4),
                ]
                .into_iter()
                .take(uvn)
                {
                    let mut buf = [0i32; 9 * 16];
                    let n = 16usize.min(lim.saturating_sub(base));
                    for i in 0..n {
                        let v = &arr[base + i];
                        buf[i * 9] = v.len() as i32;
                        for (k, &c) in v.iter().take(8).enumerate() {
                            buf[i * 9 + 1 + k] = c;
                        }
                    }
                    if n > 0 {
                        // SAFETY: own segment — no other concurrent writer.
                        unsafe { w.write_block(0, base * 9, 1, n * 9, &buf[..n * 9]) };
                    }
                }
            }
            // --- write-out: own ctx segments into the handoff arrays ---
            for (i, &(s, n)) in segs.iter().enumerate() {
                let arr = unsafe { &mut *arr_of(t, i) };
                if arr.is_empty() {
                    continue;
                }
                let e = (s + n).min(arr.len());
                if e > s {
                    // SAFETY: own exact segment — disjoint from every
                    // concurrent cell's segment.
                    unsafe { hws[i].write_block(0, s, 1, e - s, &arr[s..e]) };
                }
            }
            // --- tx-size ctx write-out: own segments ---
            {
                let (sbx4, sby4) = (sb_x / 4, sb_y / 4);
                let na = 16usize.min(w4.saturating_sub(sbx4));
                if na > 0 {
                    let mut buf = [0u8; 16];
                    for (dst, &src) in buf.iter_mut().zip(&t.a_tx[sbx4..sbx4 + na]) {
                        *dst = src as u8;
                    }
                    // SAFETY: own exact segment — disjoint writers.
                    unsafe { atxw.write_block(0, sbx4, 1, na, &buf[..na]) };
                }
                let nl = 16usize.min(h4.saturating_sub(sby4));
                if nl > 0 {
                    let mut buf = [0u8; 16];
                    for (dst, &src) in buf.iter_mut().zip(&t.l_tx[sby4..sby4 + nl]) {
                        *dst = src as u8;
                    }
                    // SAFETY: own exact segment — disjoint writers.
                    unsafe { ltxw.write_block(0, sby4, 1, nl, &buf[..nl]) };
                }
            }
            let x4 = sb_x / 4;
            let y4 = sb_y / 4;
            let stride4 = t.w / 4;
            let bw4 = 16usize.min(stride4.saturating_sub(x4));
            let bh4 = 16usize.min((t.h / 4).saturating_sub(y4));
            let x8 = sb_x / 8;
            let y8 = sb_y / 8;
            let stride8 = t.w.div_ceil(8);
            let bw8 = 8usize.min(stride8.saturating_sub(x8));
            let bh8 = 8usize.min(t.h.div_ceil(8).saturating_sub(y8));
            // SAFETY: every writer covers only this cell's disjoint map block.
            unsafe {
                skip8w.copy_region_from(&t.skip8, stride8, y8, x8, bh8, bw8);
                blk4w.copy_region_from(&t.blk4, stride4, y4, x4, bh4, bw4);
                blk4hw.copy_region_from(&t.blk4h, stride4, y4, x4, bh4, bw4);
                blk4vw.copy_region_from(&t.blk4v, stride4, y4, x4, bh4, bw4);
                blk4tw.copy_region_from(&t.blk4t, stride4, y4, x4, bh4, bw4);
                pblk4w.copy_region_from(&t.pblk4, stride4, y4, x4, bh4, bw4);
                pblk4hw.copy_region_from(&t.pblk4h, stride4, y4, x4, bh4, bw4);
                pblk4vw.copy_region_from(&t.pblk4v, stride4, y4, x4, bh4, bw4);
                pblk4tw.copy_region_from(&t.pblk4t, stride4, y4, x4, bh4, bw4);
            }
            let cell = CapturedSb {
                tokens: t.enc.take_semantic(),
            };
            tx.send((row * sb_cols + col, cell))
                .expect("wavefront entropy receiver dropped");
        },
    );
    WavefrontPlanes {
        recon: done,
        skip8,
        blk4,
        blk4h,
        blk4v,
        blk4t,
        pblk4,
        pblk4h,
        pblk4v,
        pblk4t,
    }
}

/// Copy a superblock's packed per-plane recon blocks INTO the tile's planes
/// (pure-emit replay install). Geometry mirrors `blocks_geom_extract`.
#[allow(clippy::needless_range_loop)] // `p` indexes tile planes AND blocks in lockstep
fn blocks_geom_apply(tile: &mut LossyTile, sb_x: usize, sb_y: usize, blocks: &[Vec<u16>; 3]) {
    for p in 0..3 {
        if tile.recon[p].is_empty() || blocks[p].is_empty() {
            continue;
        }
        let sx = if p == 0 {
            0
        } else {
            (tile.ss420 || tile.ss422) as usize
        };
        let sy = if p == 0 { 0 } else { tile.ss420 as usize };
        let pw = if p == 0 { tile.w } else { tile.cw };
        let ph = tile.recon[p].len() / pw;
        let (bx, by) = (sb_x >> sx, sb_y >> sy);
        let (bw, bh) = (64usize >> sx, 64usize >> sy);
        let w2 = (bx + bw).min(pw) - bx;
        let h2 = (by + bh).min(ph) - by;
        debug_assert_eq!(blocks[p].len(), w2 * h2);
        for row2 in 0..h2 {
            tile.recon[p][(by + row2) * pw + bx..][..w2]
                .copy_from_slice(&blocks[p][row2 * w2..row2 * w2 + w2]);
        }
    }
}

/// Extract a superblock's per-plane recon blocks OUT of the tile's planes
/// (capture side of the pure-emit record).
#[allow(clippy::needless_range_loop)] // `p` indexes tile planes AND out blocks in lockstep
fn blocks_geom_extract(tile: &LossyTile, sb_x: usize, sb_y: usize, out: &mut [Vec<u16>; 3]) {
    for p in 0..3 {
        if tile.recon[p].is_empty() {
            continue;
        }
        let sx = if p == 0 {
            0
        } else {
            (tile.ss420 || tile.ss422) as usize
        };
        let sy = if p == 0 { 0 } else { tile.ss420 as usize };
        let pw = if p == 0 { tile.w } else { tile.cw };
        let ph = tile.recon[p].len() / pw;
        let (bx, by) = (sb_x >> sx, sb_y >> sy);
        let (bw, bh) = (64usize >> sx, 64usize >> sy);
        let w2 = (bx + bw).min(pw) - bx;
        let h2 = (by + bh).min(ph) - by;
        let mut v = Vec::with_capacity(w2 * h2);
        for row2 in 0..h2 {
            v.extend_from_slice(&tile.recon[p][(by + row2) * pw + bx..][..w2]);
        }
        out[p] = v;
    }
}

fn pack_wavefront_tile(
    base_q_idx: u8,
    r: &TileRect,
    record: bool,
    updating_cdf: bool,
    rx: &std::sync::mpsc::Receiver<(usize, CapturedSb)>,
) -> TileOut {
    let sb_cols = r.tw.div_ceil(64);
    let sb_rows = r.th.div_ceil(64);
    let sb_count = sb_cols * sb_rows;
    let mut pending: Vec<Option<CapturedSb>> = (0..sb_count).map(|_| None).collect();

    let mut enc = OdEcEncoder::new().with_updating_cdf(updating_cdf);
    if record {
        enc.begin_trace();
    }
    let mut cdfs = Cdfs::new(crate::coef_q::qcat(base_q_idx));
    let mut slots = cdfs.semantic_slots();

    for sb_i in 0..sb_count {
        while pending[sb_i].is_none() {
            let (i, cell) = rx.recv().expect("wavefront capture stopped early");
            assert!(i < pending.len(), "wavefront cell index out of range");
            assert!(pending[i].is_none(), "duplicate wavefront cell");
            pending[i] = Some(cell);
        }
        let cell = pending[sb_i].take().unwrap();

        enc.trace_mark();
        enc.replay_semantic(&cell.tokens, &mut slots);
    }

    let trace = enc.take_trace();
    let payload = enc.done();
    TileOut {
        payload,
        trace,
        recon: Default::default(),
        skip8: Vec::new(),
        blk4: Vec::new(),
        blk4h: Vec::new(),
        blk4v: Vec::new(),
        blk4t: Vec::new(),
        pblk4: Vec::new(),
        pblk4h: Vec::new(),
        pblk4v: Vec::new(),
        pblk4t: Vec::new(),
    }
}

/// Encode a single tile as an independent sub-frame. Pure function of its inputs
/// (no shared mutable state), so it is safe to run on any thread. When `mono`,
/// only the luma plane is coded (`src[1]`/`src[2]` ignored, chroma recon empty).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // halo gate: `p` indexes recon planes AND `done` in lockstep
fn encode_one_tile(
    base_q_idx: u8,
    bd: u8,
    full_w: usize,
    full_h: usize,
    cw8: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    src: &[Vec<u16>; 3],
    r: &TileRect,
    speed: Speed,
    dct: DctDispatch,
    idct: IdctDispatch,
    intrapred: IntraPredDispatch,
    kmeans: KmeansDispatch,
    rd: crate::rd_sse::RdDispatch,
    aq: bool,
    vb: &VarianceBoost,
    record: bool,
    wf_threads: usize,
    allow_intrabc: bool,
    updating_cdf: bool,
    screen_content: bool,
) -> TileOut {
    let tsrc = if mono {
        [
            crop_plane(&src[0], full_w, r.x0, r.y0, r.tw, r.th),
            Vec::new(),
            Vec::new(),
        ]
    } else {
        [
            crop_plane(&src[0], full_w, r.x0, r.y0, r.tw, r.th),
            crop_plane(&src[1], cw8, r.cx0, r.cy0, r.ctw, r.cth),
            crop_plane(&src[2], cw8, r.cx0, r.cy0, r.ctw, r.cth),
        ]
    };
    // The IntraBC exact-match index is a pure function of the tile source, so
    // ONE cell is shared by every consumer instead of each building its own:
    // the wavefront spawns a `LossyTile` per worker and the decouple check runs
    // a second Replay pass, and all of them used to rebuild the identical table
    // (12 threads x 34 ms on a 3450x1900 screenshot — most of the
    // multithreaded IntraBC regression). Still lazy, because Fast prices no
    // IntraBC candidate at all on many frames and must not pay the build.
    let ibc_cell: std::sync::OnceLock<LossyIbcIndex> = std::sync::OnceLock::new();
    let ibc_index = allow_intrabc.then_some(&ibc_cell);
    // One full decide+emit pass over the tile in the given decision mode.
    // `Capture` fills a fresh record; `Replay` consumes `rec_in` (skipping the
    // partition searches). Returns the coded tile plus the record so the
    // decouple check below can chain Capture -> Replay.
    let run = |sb_mode: SbMode,
               rec_in: DecisionRecord,
               stream_rx: Option<&std::sync::mpsc::Receiver<(usize, DecisionRecord)>>|
     -> (TileOut, DecisionRecord) {
        let mut tile = if mono {
            LossyTile::new_mono(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm)
        } else {
            match (sub_x, sub_y) {
                (0, 0) => LossyTile::new(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm),
                (1, 0) => LossyTile::new_422(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm),
                _ => LossyTile::new_420(base_q_idx, bd, r.tw, r.th, &tsrc, vb.qm),
            }
        }
        .with_dispatch(dct, idct, intrapred, kmeans, rd)
        .with_speed(speed)
        .with_intrabc(allow_intrabc, ibc_index)
        .with_screen_content(screen_content)
        .with_updating_cdf(updating_cdf);
        tile.sb_mode = sb_mode;
        tile.rec = rec_in;
        // Loop restoration is frame-relative: record this tile's frame-absolute luma
        // origin and the full frame luma size so `read_lr` is computed in frame
        // coordinates regardless of tiling.
        tile.frame_x0 = r.x0;
        tile.frame_y0 = r.y0;
        tile.frame_w = full_w;
        tile.frame_h = full_h;
        if aq {
            // Center the per-SB deltas on this tile's mean activity so the average
            // quantizer tracks base_q_idx (zero-mean deltas => ~rate-neutral).
            let ref_act = tile_ref_activity(&tile.src[0], tile.w, tile.w, tile.h);
            tile.enable_aq(base_q_idx, ref_act, vb);
        }
        if record {
            tile.enc.begin_trace();
        }
        // The whole per-SB AQ qindex sequence is a pure function of the source, so
        // it is precomputed up front (bit-exact vs the serial accumulator — see
        // `aq_grid_matches_serial`). This is what lets the wavefront decide SBs out
        // of raster order later: each SB reads its cell instead of advancing shared
        // state. Empty when AQ is off.
        let aq_grid = tile.precompute_aq_grid();
        let sb_count = r.tw.div_ceil(64) * r.th.div_ceil(64);
        let mut pending: Vec<Option<DecisionRecord>> = vec![None; sb_count];
        let mut streamed_rec = DecisionRecord::default();
        let mut sb_i = 0usize;
        for sb_y in (0..r.th).step_by(64) {
            for sb_x in (0..r.tw).step_by(64) {
                if let Some(rx) = stream_rx {
                    while pending[sb_i].is_none() {
                        let (i, rec) = rx.recv().expect("wavefront capture stopped early");
                        assert!(i < pending.len(), "wavefront cell index out of range");
                        assert!(pending[i].is_none(), "duplicate wavefront cell");
                        pending[i] = Some(rec);
                    }
                    tile.rec = pending[sb_i].take().unwrap();
                    tile.cur = RecordCursor::default();
                }
                // Pure-emit replay: install this SB's captured recon blocks
                // up front. Converted leaves skip prediction/recon entirely;
                // unconverted (DC) leaves read correct neighbors and rewrite
                // identical pixels.
                if sb_mode == SbMode::Replay {
                    let idx = if stream_rx.is_some() { 0 } else { sb_i };
                    if idx < tile.rec.recon.len() {
                        let blocks = std::mem::take(&mut tile.rec.recon[idx]);
                        blocks_geom_apply(&mut tile, sb_x, sb_y, &blocks);
                        tile.rec.recon[idx] = blocks;
                    }
                }
                // The mark sits exactly where a replay would interleave the LR
                // symbols owed by this superblock (`emit_lr_sb` is a no-op here).
                tile.enc.trace_mark();
                tile.cdef_point_marked = false;
                tile.emit_lr_sb(sb_x, sb_y);
                if !aq_grid.is_empty() {
                    tile.aq_begin_sb_cell(&aq_grid[sb_i]);
                }
                tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
                if sb_mode == SbMode::Capture {
                    let mut cell_recon: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                    blocks_geom_extract(&tile, sb_x, sb_y, &mut cell_recon);
                    tile.rec.recon.push(cell_recon);
                }
                if stream_rx.is_some() {
                    debug_assert_eq!(tile.cur.parts, tile.rec.parts.len());
                    debug_assert_eq!(tile.cur.luma, tile.rec.luma.len());
                    debug_assert_eq!(tile.cur.uv, tile.rec.uv.len());
                    let cell = std::mem::take(&mut tile.rec);
                    streamed_rec.parts.extend(cell.parts);
                    streamed_rec.luma.extend(cell.luma);
                    streamed_rec.uv.extend(cell.uv);
                }
                sb_i += 1;
            }
        }
        // NOTE: the in-loop deblocking filter is deliberately NOT applied here.
        // In AV1 the deblocking filter is a frame-level post-filter that operates
        // ACROSS tile boundaries (only entropy coding and prediction are
        // tile-independent). Applying it per tile leaves the inter-tile edges
        // unfiltered, diverging from the decoder at every tile boundary. The filter
        // is instead applied once on the stitched frame in `encode_lossy_tilegroup`.
        // Intra prediction already used the unfiltered recon during `decode_sb`, so
        // deferring the filter does not change any coding decision.
        let skip8 = std::mem::take(&mut tile.skip8);
        let blk4 = std::mem::take(&mut tile.blk4);
        let blk4h = std::mem::take(&mut tile.blk4h);
        let blk4v = std::mem::take(&mut tile.blk4v);
        let blk4t = std::mem::take(&mut tile.blk4t);
        let pblk4 = std::mem::take(&mut tile.pblk4);
        let pblk4h = std::mem::take(&mut tile.pblk4h);
        let pblk4v = std::mem::take(&mut tile.pblk4v);
        let pblk4t = std::mem::take(&mut tile.pblk4t);
        let trace = tile.enc.take_trace();
        let payload = tile.enc.done();
        let rec = if stream_rx.is_some() {
            streamed_rec
        } else {
            std::mem::take(&mut tile.rec)
        };
        (
            TileOut {
                payload,
                trace,
                recon: std::mem::take(&mut tile.recon),
                skip8,
                blk4,
                blk4h,
                blk4v,
                blk4t,
                pblk4,
                pblk4h,
                pblk4v,
                pblk4t,
            },
            rec,
        )
    };
    let (out, _) = if wf_threads > 1 {
        // Pipeline the lightweight adaptive-CDF token packer behind parallel
        // capture. The packer does no block traversal or reconstruction.
        let out = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            // The entropy lane spends most of its life recv()-blocked on
            // capture, so capture keeps the full requested worker budget.
            let capture_threads = wf_threads;
            let tsrc_ref = &tsrc;
            let capture = scope.spawn(move || {
                wavefront_capture(
                    capture_threads,
                    base_q_idx,
                    bd,
                    full_w,
                    full_h,
                    sub_x,
                    sub_y,
                    mono,
                    tsrc_ref,
                    r,
                    speed,
                    dct,
                    idct,
                    intrapred,
                    kmeans,
                    rd,
                    aq,
                    vb,
                    updating_cdf,
                    allow_intrabc,
                    ibc_index,
                    screen_content,
                    tx,
                )
            });
            let mut packed = pack_wavefront_tile(base_q_idx, r, record, updating_cdf, &rx);
            let planes = capture.join().expect("wavefront capture panicked");
            packed.recon = planes.recon;
            packed.skip8 = planes.skip8;
            packed.blk4 = planes.blk4;
            packed.blk4h = planes.blk4h;
            packed.blk4v = planes.blk4v;
            packed.blk4t = planes.blk4t;
            packed.pblk4 = planes.pblk4;
            packed.pblk4h = planes.pblk4h;
            packed.pblk4v = planes.pblk4v;
            packed.pblk4t = planes.pblk4t;
            packed
        });
        (out, DecisionRecord::default())
    } else {
        run(SbMode::Off, DecisionRecord::default(), None)
    };
    out
}

/// Per-64x64-unit CDEF replay context: the frame-level on/off grid plus its
/// column count. Each SB with a recorded `read_cdef()` point gets its unit's
/// index inserted there as a raw `L(1)` literal.
struct CdefReplay<'a> {
    grid: &'a [u8],
    unit_cols: usize,
    /// cdef_bits: width of the per-unit raw index literal (1..=3).
    bits: u32,
}

/// Replay a tile's recorded symbols with the frame-level filter syntax
/// interleaved: the Wiener `read_lr` symbols at each SB start (when `lr` is
/// set) and the per-unit `cdef_idx` literal at each SB's `read_cdef()` point
/// (when `cdef` is set). LR touches only its own CDF + raw bits and `cdef_idx`
/// is an equiprobable literal, so this is byte-identical to a re-encode.
fn replay_tile_with_filters(
    r: &TileRect,
    trace: &crate::odec::SymbolTrace,
    lr: Option<&crate::wiener::WienerUnit>,
    cdef: Option<&CdefReplay>,
    frame_w: usize,
    frame_h: usize,
    updating_cdf: bool,
) -> Vec<u8> {
    let mut enc = OdEcEncoder::new().with_updating_cdf(updating_cdf);
    let mut wr_cdf = wiener_restore_icdf();
    let mut lr_ref_v = crate::wiener::WIENER_TAPS_MID;
    let mut lr_ref_h = crate::wiener::WIENER_TAPS_MID;
    let mut i = 0usize;
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            if let Some(unit) = lr {
                emit_lr_sb_syms(
                    &mut enc,
                    &mut wr_cdf,
                    &mut lr_ref_v,
                    &mut lr_ref_h,
                    unit,
                    r.x0,
                    r.y0,
                    frame_w,
                    frame_h,
                    sb_x,
                    sb_y,
                );
            }
            match cdef {
                Some(c) => {
                    let (pre, post) = trace.sb_ops_split(i);
                    enc.replay(pre);
                    if let Some(post) = post {
                        // This SB has a non-skip block: its 64x64 unit carries a
                        // cdef_idx literal at the recorded read_cdef() point.
                        let u = ((r.y0 + sb_y) / 64) * c.unit_cols + (r.x0 + sb_x) / 64;
                        let idx = c.grid.get(u).copied().unwrap_or(0);
                        enc.encode_literal(idx as u32, c.bits);
                        enc.replay(post);
                    }
                }
                None => enc.replay(trace.sb_ops(i)),
            }
            i += 1;
        }
    }
    debug_assert_eq!(i, trace.sb_count(), "trace/SB iteration mismatch");
    enc.done()
}

/// Resolve the requested thread count: `0` => all available cores (fallback 1),
/// otherwise the value as-is. The caller still caps this at the tile count.
pub(crate) fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

/// Whether a single-tile AV1 wavefront is mathematically too narrow to feed
/// the requested worker count. Replay owns one lane; capture is bounded by
/// both total work and the left/above/above-right dependency chain.
fn wavefront_should_use_tiles(sb_cols: usize, sb_rows: usize, threads: usize) -> bool {
    if threads <= 1 || sb_cols == 0 || sb_rows == 0 {
        return false;
    }
    let cells = sb_cols * sb_rows;
    // Small frames never tile: the thread count would change the TILING PLAN
    // and thus the bitstream (t2/t3/t8 on a 6x9-SB frame produced 2/4/8-tile
    // grids, +6% bytes at t8), breaking the -tN == -t1 invariant for frames
    // that encode in ~a second anyway. 256 SBs = ~1MP.
    if cells < 256 {
        return false;
    }
    let wave_work_floor = cells.div_ceil(threads - 1);
    let wave_dependency_floor = sb_cols + 2 * sb_rows.saturating_sub(1);
    let wave_floor = wave_work_floor.max(wave_dependency_floor);
    let tile_floor = cells.div_ceil(threads);
    wave_floor * 100 > tile_floor * 135
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_tilegroup(
    base_q_idx: u8,
    bd: u8,
    w8: usize,
    h8: usize,
    disp_w: usize,
    disp_h: usize,
    src: &[Vec<u16>; 3],
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    context: &EncodingContext<'_>,
    aq: bool,
    cdef_on: bool,
    wiener_on: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc_allowed: bool,
) -> (
    Vec<u8>,
    Tiling,
    Option<crate::obu::CdefParams>,
    Option<crate::obu::LrParams>,
    bool,
) {
    let pool = context.thread_pool;
    let speed = context.speed;
    let vb = &context.boost;
    let dct = context.dct_dispatch();
    let idct = context.idct_dispatch();
    let intrapred = context.intrapred;
    let kmeans = context.kmeans;
    let rd = context.rd;
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;

    // Aim for ~one tile per worker so small frames can be paralleled too.
    // `threads == 1` -> target 1 -> spec-minimum tiling (single tile for small
    // frames, byte-identical to the untiled output).
    let want = pool.width();
    // Prefer minimal tiling + SB wavefront when the dependency graph can feed
    // the requested worker count. Otherwise, use ordinary parallel tiles: a
    // narrow WPP graph cannot manufacture parallelism, and forcing it was up
    // to 2-3x slower on small/medium images.
    let multitile =
        want > 1 && wavefront_should_use_tiles(sb_cols as usize, sb_rows as usize, want);
    let tile_target = if multitile { want } else { 1 };
    let plan = plan_tiling(sb_cols, sb_rows, tile_target);
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    let (cw8, ch8) = (w8 >> sub_x, h8 >> sub_y);

    // Tile rectangles in raster order (top-to-bottom, left-to-right).
    let mut rects: Vec<TileRect> = Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        let th = y1 - y0;
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            let tw = x1 - x0;
            rects.push(TileRect {
                x0,
                y0,
                tw,
                th,
                cx0: x0 >> sub_x,
                cy0: y0 >> sub_y,
                ctw: tw >> sub_x,
                cth: th >> sub_y,
            });
        }
    }

    let allow_intrabc = intrabc_allowed
        && !mono
        && rects.iter().any(|r| {
            // Coverage threshold: IntraBC costs the whole frame its loop
            // filters, so one duplicate pair is NOT enough (h_vvc's letterbox
            // bars measured +3.3% BD from lost deblock). Count NON-FLAT
            // aligned 16x16 blocks with an exact earlier duplicate; require
            // a meaningful fraction of the tile before paying the trade.
            let exact16 = |ax: usize, ay: usize, bx: usize, by: usize| {
                (0..3).all(|plane| {
                    let sx = usize::from(plane != 0 && sub_x != 0);
                    let sy = usize::from(plane != 0 && sub_y != 0);
                    let stride = if plane == 0 { w8 } else { cw8 };
                    let (bw, bh) = (16 >> sx, 16 >> sy);
                    let (ax, ay) = ((r.x0 + ax) >> sx, (r.y0 + ay) >> sy);
                    let (bx, by) = ((r.x0 + bx) >> sx, (r.y0 + by) >> sy);
                    (0..bh).all(|row| {
                        src[plane][(ay + row) * stride + ax..][..bw]
                            == src[plane][(by + row) * stride + bx..][..bw]
                    })
                })
            };
            let nly = r.th.saturating_sub(15).div_ceil(16);
            let nlx = r.tw.saturating_sub(15).div_ceil(16);
            let total = nly * nlx;
            if total == 0 {
                return false;
            }
            // dup * 20 >= total  <=>  dup >= ceil(total / 20); with the >= 8
            // floor this is the exact pass threshold, so the raster walk below
            // can stop the moment it is reached (positive early-out).
            let need = 8usize.max(total.div_ceil(20));
            // The content hash per block is ~99% of the gate's work and each
            // block is independent, so hash rows in parallel; the duplicate
            // count needs raster ("earlier block") semantics and stays serial.
            let hash_row = |row: usize| -> Vec<(bool, u64)> {
                let ly = row * 16;
                (0..nlx)
                    .map(|col| {
                        let lx = col * 16;
                        let (x0, y0) = (r.x0 + lx, r.y0 + ly);
                        // Flat blocks (single luma value) duplicate trivially —
                        // black bars, blank sky — and gain nothing from a copy
                        // that DC prediction doesn't already provide.
                        let first = src[0][y0 * w8 + x0];
                        let flat = (0..16).all(|row| {
                            src[0][(y0 + row) * w8 + x0..][..16]
                                .iter()
                                .all(|&v| v == first)
                        });
                        if flat {
                            return (true, 0);
                        }
                        let mut h = 0xcbf2_9ce4_8422_2325u64;
                        for (plane, sp) in src.iter().enumerate() {
                            let sx = usize::from(plane != 0 && sub_x != 0);
                            let sy = usize::from(plane != 0 && sub_y != 0);
                            let stride = if plane == 0 { w8 } else { cw8 };
                            let (x, y) = ((r.x0 + lx) >> sx, (r.y0 + ly) >> sy);
                            let (bw, bh) = (16 >> sx, 16 >> sy);
                            for row in 0..bh {
                                for &v in &sp[(y + row) * stride + x..][..bw] {
                                    h ^= v as u64;
                                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                                }
                            }
                        }
                        (false, h)
                    })
                    .collect()
            };
            let rows = pool.map_indexed(pool.width(), nly, hash_row);
            let mut seen: HashMap<u64, (usize, usize)> = HashMap::new();
            let mut dup = 0usize;
            for (row, cells) in rows.iter().enumerate() {
                for (col, &(flat, h)) in cells.iter().enumerate() {
                    if flat {
                        continue;
                    }
                    let (lx, ly) = (col * 16, row * 16);
                    if let Some(&(ox, oy)) = seen.get(&h) {
                        if exact16(ox, oy, lx, ly) {
                            dup += 1;
                            if dup >= need {
                                return true;
                            }
                        }
                    } else {
                        seen.insert(h, (lx, ly));
                    }
                }
            }
            false
        });

    let n = rects.len();
    let nthreads = want.clamp(1, n.max(1));

    // SB-wavefront: parallelize WITHIN each tile — parallel decision capture
    // (d = 2r+c wavefront) plus a lightweight raster entropy lane,
    // byte-identical per tile.
    // Tiles are then encoded SEQUENTIALLY with the full thread budget inside
    // each (wavefront capture spawns its own scoped workers; nesting it under
    // the tile pool would oversubscribe — same no-nesting rule as the AV2
    // wavefront). Wide frames get mandatory column tiles (MAX_TILE_WIDTH), so
    // per-tile is the only shape that covers them.
    let wf_threads = if want > 1 && !multitile { want } else { 0 };

    // Recording the symbol trace lets a winning Wiener unit or a per-unit CDEF
    // grid be signaled by a cheap replay instead of a second full encode of
    // every tile.
    let record = !allow_intrabc && (wiener_on || cdef_on) && base_q_idx != 0;
    let mut outs: Vec<TileOut> = if wf_threads > 1 && n < want {
        // Split the budget: `outer` tiles in flight × `inner`-wide wavefront
        // inside each (outer*inner == want, no oversubscription — the inner
        // scoped workers replace their outer worker's idle time while other
        // tiles are packed). Single-tile frames get the full
        // budget inside the one tile.
        // Capture/packing pipelining needs at least three lanes per tile: one
        // packing lane and two decision lanes. If there are already at least
        // as many geometry-stable tiles as workers, the branch below simply
        // runs those tiles directly without an unnecessary inner wavefront.
        let outer = n.min((wf_threads / 3).max(1));
        let inner = (wf_threads / outer).max(2);
        pool.map_indexed(outer, n, |i| {
            encode_one_tile(
                base_q_idx,
                bd,
                w8,
                h8,
                cw8,
                sub_x,
                sub_y,
                mono,
                src,
                &rects[i],
                speed,
                dct,
                idct,
                intrapred,
                kmeans,
                rd,
                aq,
                vb,
                record,
                inner,
                allow_intrabc,
                updating_cdf,
                screen_content,
            )
        })
    } else {
        pool.map_indexed(nthreads, n, |i| {
            encode_one_tile(
                base_q_idx,
                bd,
                w8,
                h8,
                cw8,
                sub_x,
                sub_y,
                mono,
                src,
                &rects[i],
                speed,
                dct,
                idct,
                intrapred,
                kmeans,
                rd,
                aq,
                vb,
                record,
                0,
                allow_intrabc,
                updating_cdf,
                screen_content,
            )
        })
    };

    let mut payloads: Vec<Vec<u8>> = outs
        .iter_mut()
        .map(|o| std::mem::take(&mut o.payload))
        .collect();
    let traces: Vec<_> = outs.iter_mut().map(|o| o.trace.take()).collect();

    // Small per-8x8 / per-4x4 maps: stitched serially (they are tiny).
    // Monochrome has only a luma plane; chroma recon stays empty.
    let mut recon = if mono {
        [vec![0u16; w8 * h8], Vec::new(), Vec::new()]
    } else {
        [
            vec![0u16; w8 * h8],
            vec![0u16; cw8 * ch8],
            vec![0u16; cw8 * ch8],
        ]
    };
    let sb8w = w8.div_ceil(8);
    let sb8h = h8.div_ceil(8);
    let mut skip8 = vec![true; sb8w * sb8h];
    // Frame-level luma block-size map (4x4 units), assembled from every tile so
    // the deblocking filter can run on the stitched frame (across tile edges).
    let nc4f = w8 / 4;
    let nr4f = h8 / 4;
    let mut blk4f = vec![0u8; nc4f * nr4f];
    let mut blk4hf = vec![0u8; nc4f * nr4f];
    let mut blk4vf = vec![false; nc4f * nr4f];
    let mut blk4tf = vec![false; nc4f * nr4f];
    let mut pblk4f = vec![0u8; nc4f * nr4f];
    let mut pblk4hf = vec![0u8; nc4f * nr4f];
    let mut pblk4vf = vec![false; nc4f * nr4f];
    let mut pblk4tf = vec![false; nc4f * nr4f];
    for (r, out) in rects.iter().zip(outs.iter()) {
        let tsb8w = r.tw.div_ceil(8);
        let (ox8, oy8) = (r.x0 / 8, r.y0 / 8);
        for ty in 0..r.th.div_ceil(8) {
            for tx in 0..tsb8w {
                let (fx, fy) = (ox8 + tx, oy8 + ty);
                if fx < sb8w && fy < sb8h {
                    skip8[fy * sb8w + fx] = out.skip8[ty * tsb8w + tx];
                }
            }
        }
        let tnc4 = r.tw / 4;
        let (ox4, oy4) = (r.x0 / 4, r.y0 / 4);
        for ty in 0..(r.th / 4) {
            for tx in 0..tnc4 {
                let (fx, fy) = (ox4 + tx, oy4 + ty);
                if fx < nc4f && fy < nr4f {
                    blk4f[fy * nc4f + fx] = out.blk4[ty * tnc4 + tx];
                    blk4hf[fy * nc4f + fx] = out.blk4h[ty * tnc4 + tx];
                    blk4vf[fy * nc4f + fx] = out.blk4v[ty * tnc4 + tx];
                    blk4tf[fy * nc4f + fx] = out.blk4t[ty * tnc4 + tx];
                    pblk4f[fy * nc4f + fx] = out.pblk4[ty * tnc4 + tx];
                    pblk4hf[fy * nc4f + fx] = out.pblk4h[ty * tnc4 + tx];
                    pblk4vf[fy * nc4f + fx] = out.pblk4v[ty * tnc4 + tx];
                    pblk4tf[fy * nc4f + fx] = out.pblk4t[ty * tnc4 + tx];
                }
            }
        }
    }

    // Pixel planes: every tile row owns a disjoint horizontal band of each
    // plane, so (plane, tile row) pairs stitch in parallel.
    let ncols = col_starts.len();
    {
        let mut items: Vec<(usize, usize, &mut [u16])> = Vec::new();
        for (pl, plane) in recon.iter_mut().enumerate() {
            if plane.is_empty() {
                continue;
            }
            let pw = if pl == 0 { w8 } else { cw8 };
            let mut rest = &mut plane[..];
            let mut consumed = 0usize;
            for ti in 0..row_starts.len() {
                let r0 = &rects[ti * ncols];
                let (py0, pth) = if pl == 0 {
                    (r0.y0, r0.th)
                } else {
                    (r0.cy0, r0.cth)
                };
                debug_assert_eq!(consumed, py0 * pw);
                let (band, r2) = std::mem::take(&mut rest).split_at_mut(pth * pw);
                rest = r2;
                consumed += band.len();
                items.push((pl, ti, band));
            }
        }
        pool.for_each(nthreads, items, |(pl, ti, band)| {
            for (r, out) in rects[ti * ncols..(ti + 1) * ncols]
                .iter()
                .zip(&outs[ti * ncols..(ti + 1) * ncols])
            {
                let (pw, px0, ptw, pth) = if pl == 0 {
                    (w8, r.x0, r.tw, r.th)
                } else {
                    (cw8, r.cx0, r.ctw, r.cth)
                };
                stitch_plane(band, pw, px0, 0, &out.recon[pl], ptw, pth);
            }
        });
    }
    drop(outs);

    // Frame-level in-loop deblocking filter, applied once on the stitched
    // reconstruction so that inter-tile edges are filtered exactly as the
    // decoder does (deblocking is not tile-independent in AV1). `filter_plane`
    // is a no-op when the derived level is 0 (e.g. lossless).
    if !allow_intrabc {
        let (lvl_y, lvl_uv) = crate::obu::loop_filter_levels(base_q_idx);
        frame_deblock(
            &context.loopfilter,
            pool,
            &mut recon,
            w8,
            h8,
            cw8,
            ch8,
            disp_w,
            disp_h,
            &blk4f,
            &blk4hf,
            &blk4vf,
            &blk4tf,
            &pblk4f,
            &pblk4hf,
            &pblk4vf,
            &pblk4tf,
            nc4f,
            sub_x,
            sub_y,
            mono,
            lvl_y,
            lvl_uv,
            bd,
        );
    }

    // Frame-level CDEF (R-D searched; may pick per-64x64-unit signaling).
    let cdef_decision = if !allow_intrabc && cdef_on && base_q_idx != 0 {
        frame_cdef(
            &mut recon, src, &skip8, sb8w, w8, h8, cw8, ch8, disp_w, disp_h, sub_x, sub_y, mono,
            base_q_idx, bd, speed, pool,
        )
    } else {
        None
    };

    // Frame-level luma Wiener loop restoration (searched on the CDEF-filtered
    // recon, matching the decoder's filter order).
    let lr_unit = if !allow_intrabc && wiener_on && base_q_idx != 0 {
        frame_wiener_search(&recon[0], &src[0], w8, h8, bd, pool)
    } else {
        None
    };

    // Signal the winning filter syntax by replaying each tile's recorded
    // symbols with the read_lr symbols / per-unit cdef_idx literals
    // interleaved (byte-identical to a full re-encode; recon is unchanged
    // because both only add their own symbols).
    let cdef_replay = cdef_decision
        .as_ref()
        .filter(|d| d.params.bits > 0)
        .map(|d| CdefReplay {
            grid: &d.grid,
            unit_cols: d.unit_cols,
            bits: d.params.bits as u32,
        });
    if lr_unit.is_some() || cdef_replay.is_some() {
        payloads = pool.map_indexed(nthreads, n, |i| {
            let trace = traces[i]
                .as_deref()
                .expect("trace recorded for filter replay");
            replay_tile_with_filters(
                &rects[i],
                trace,
                lr_unit.as_ref(),
                cdef_replay.as_ref(),
                w8,
                h8,
                updating_cdf,
            )
        });
    }
    let lr = lr_unit.map(|_| crate::obu::LrParams { luma_wiener: true });

    // Debug: dump the final reconstruction (post-deblock/CDEF) for the
    // dav1d conformance oracle (bench/rd/oscan.sh + oraclecmp.py). This
    // hook was lost in a refactor once (2026-07-26) and the oracle failed
    // 0/90 silently — keep it adjacent to tilegroup assembly.
    if let Ok(path) = std::env::var("MT_DUMP_RECON") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create(&path) {
            // ceil, not floor: for odd display dims the last chroma col/row
            // is real coded content (AV1 chroma dims are ceil(w >> sub)).
            let (cdw, cdh) = (disp_w.div_ceil(1 << sub_x), disp_h.div_ceil(1 << sub_y));
            let planes: [(usize, usize, usize); 3] =
                [(disp_w, disp_h, w8), (cdw, cdh, cw8), (cdw, cdh, cw8)];
            let np = if mono { 1 } else { 3 };
            let _ = writeln!(f, "MTREC {disp_w} {disp_h} {cdw} {cdh} {np}");
            for (pl, &(pw, ph, stride)) in planes.iter().enumerate().take(np) {
                for y in 0..ph {
                    let row: Vec<u8> = (0..pw)
                        .map(|x| (recon[pl][y * stride + x] >> (bd - 8)).clamp(0, 255) as u8)
                        .collect();
                    let _ = f.write_all(&row);
                }
            }
        }
    }

    let tilegroup = assemble_tilegroup(payloads);
    (
        tilegroup,
        plan,
        cdef_decision.map(|d| d.params),
        lr,
        allow_intrabc,
    )
}

/// CDEF damping derived from the base quantizer (spec range 3..=6); higher q ->
/// stronger ringing -> a touch more damping. Kept simple and deterministic.
fn cdef_damping(base_q_idx: u8) -> u8 {
    3 + ((base_q_idx as u32) / 64).min(3) as u8
}

fn frame_wiener_search(
    recon: &[u16],
    src: &[u16],
    w: usize,
    h: usize,
    bd: u8,
    pool: &Pool,
) -> Option<crate::wiener::WienerUnit> {
    use crate::wiener::{WienerKernel, wiener_filter_plane};
    let sse = |a: &[u16]| -> i64 {
        let mut s = 0i64;
        for i in 0..w * h {
            let d = (a[i] - src[i]) as i64;
            s += d * d;
        }
        s
    };
    let base = sse(recon);
    // Small candidate set of separable low-pass Wiener kernels (coded taps
    // [t0,t1,t2]); the identity is implicit via `base`. These span gentle to
    // moderate smoothing within the spec tap ranges.
    const CANDS: [[i32; 3]; 4] = [[0, 0, 1], [-1, 2, 2], [0, 1, 3], [1, -3, 5]];
    let cands: Vec<(&[i32; 3], &[i32; 3])> = CANDS
        .iter()
        .flat_map(|h_taps| CANDS.iter().map(move |v_taps| (h_taps, v_taps)))
        .collect();
    // Each candidate filters into its own buffer; the reduce below walks the
    // original (h, v) order so ties break exactly as the sequential loop did.
    let want = pool.width().min(cands.len());
    let sses: Vec<i64> = pool.map_indexed(want, cands.len(), |i| {
        let (h_taps, v_taps) = cands[i];
        let hk = WienerKernel::from_coded(*h_taps);
        let vk = WienerKernel::from_coded(*v_taps);
        let mut tmp = vec![0u16; w * h];
        wiener_filter_plane(&mut tmp, recon, w, h, &hk, &vk, bd);
        sse(&tmp)
    });
    let mut best: Option<(i64, crate::wiener::WienerUnit)> = None;
    for (&(h_taps, v_taps), &s) in cands.iter().zip(sses.iter()) {
        if s < base && best.as_ref().is_none_or(|b| s < b.0) {
            best = Some((
                s,
                crate::wiener::WienerUnit {
                    h: *h_taps,
                    v: *v_taps,
                },
            ));
        }
    }
    best.map(|b| b.1)
}

/// Default directional-variance gate threshold (see `frame_cdef`).
const UNIT_DIR_VAR_THRESH_DEFAULT: i64 = 15000;
/// Default per-mille per-unit margin (see `frame_cdef`).
const MARGIN_DEFAULT: i64 = 22;

/// Frame CDEF decision: header params plus, when `params.bits == 1`, the
/// per-64x64-unit on/off grid (frame unit raster order; 1 = index 1 = filtered).
/// Units whose 8x8 blocks are all skip carry no `cdef_idx` and are never
/// filtered; their grid entry stays 0.
pub(crate) struct CdefFrameDecision {
    pub(crate) params: crate::obu::CdefParams,
    pub(crate) grid: Vec<u8>,
    pub(crate) unit_cols: usize,
}

#[allow(clippy::too_many_arguments)]
fn frame_cdef(
    recon: &mut [Vec<u16>; 3],
    src: &[Vec<u16>; 3],
    skip8: &[bool],
    sb8w: usize,
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    disp_w: usize,
    disp_h: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    base_q_idx: u8,
    bd: u8,
    speed: Speed,
    pool: &Pool,
) -> Option<CdefFrameDecision> {
    use crate::cdef;
    let signaled_damping = cdef_damping(base_q_idx) as i32;
    let damping = signaled_damping + (bd as i32 - 8);

    // The decoder reconstructs and reads REAL pixels across the whole
    // mi-aligned (coded) area — CDEF taps see the coded padding columns/rows as
    // ordinary neighbors, and only samples beyond the mi extent are
    // CDEF_VERY_LARGE (handled by the stride bounds in `sample`). The padding is
    // never deblocked (see `filter_plane`'s visible clip), so the snapshot below
    // matches the decoder's pre-CDEF frame everywhere. Distortion, however, is
    // measured over the VISIBLE window only — invisible pixels must not drive
    // the decision.
    let (cw_vis, ch_vis) = (disp_w.div_ceil(1 << sub_x), disp_h.div_ceil(1 << sub_y));
    let snap_y = recon[0].clone();

    let nbx = w8.div_ceil(8);
    let nby = h8.div_ceil(8);
    let mut ldirs = vec![0usize; nbx * nby];
    let mut lvars = vec![0i32; nbx * nby];
    {
        // Directions read the decoder's frame buffer, which contains REAL
        // reconstructed pixels across the whole mi-aligned (coded) area — the
        // padding columns/rows are coded and reconstructed, only CDEF taps
        // beyond them are masked. So the direction search uses the unmarked
        // recon.
        let luma = &recon[0];
        #[allow(clippy::type_complexity)]
        let items: Vec<(usize, (&mut [usize], &mut [i32]))> = ldirs
            .chunks_mut(nbx)
            .zip(lvars.chunks_mut(nbx))
            .enumerate()
            .collect();
        pool.for_each(pool.width(), items, |(by, (drow, vrow))| {
            for bx in 0..nbx {
                if bx * 8 < w8 && by * 8 < h8 {
                    let (d, v) = cdef::cdef_direction(luma, w8, bx * 8, by * 8, bd);
                    drow[bx] = d;
                    vrow[bx] = v;
                }
            }
        });
    }

    // Per-8x8 luma skip map: CDEF is NOT applied to skip blocks (the decoder
    // leaves them untouched), so the encoder must mirror that exactly. `skip8`
    // is already a per-8x8 frame map.
    let lskip: Vec<bool> = (0..nbx * nby)
        .map(|i| {
            let (bx, by) = (i % nbx, i / nbx);
            skip8.get(by * sb8w + bx).copied().unwrap_or(true)
        })
        .collect();

    // 64x64 CDEF unit grid; a unit is "signaled" (carries a cdef_idx literal)
    // iff it contains at least one non-skip 8x8.
    let uc = w8.div_ceil(64);
    let ur = h8.div_ceil(64);
    let n_units = uc * ur;
    let mut signaled = vec![false; n_units];
    for by in 0..nby {
        for bx in 0..nbx {
            if !lskip[by * nbx + bx] {
                signaled[(by / 8) * uc + bx / 8] = true;
            }
        }
    }
    let n_sig = signaled.iter().filter(|&&s| s).count();
    if n_sig == 0 {
        return None;
    }

    // Decision knobs:
    //   gate_thresh  directional-variance gate threshold (0 disables)
    //   perceptual   plain SSE vs perceptual cdef_dist
    //   margin per mille distortion margin a unit must clear to filter
    let gate_thresh = UNIT_DIR_VAR_THRESH_DEFAULT;
    let perceptual = true;
    let margin = MARGIN_DEFAULT;

    // Directional-variance gate for the GLOBAL (filter-everything) option only:
    // that mode has no per-unit off-switch, so it is offered solely when most
    // units show a dominant edge direction (where ringing lives — CDEF's
    // target). Isotropic fine detail (fractals) has no dominant direction;
    // there the variance-masked metric collapses to plain SSE and rewards
    // smoothing SSIMULACRA2 punishes — measured on the corpus as a −0.2 SS2
    // regression whenever global filtering engages on such content. Per-unit
    // decisions are NOT gated: the distortion margin protects them.
    let mut dv_sum = vec![0i64; n_units];
    let mut dv_cnt = vec![0i64; n_units];
    for by in 0..nby {
        for bx in 0..nbx {
            if !lskip[by * nbx + bx] {
                let u = (by / 8) * uc + bx / 8;
                dv_sum[u] += lvars[by * nbx + bx] as i64;
                dv_cnt[u] += 1;
            }
        }
    }
    let trusted: Vec<bool> = (0..n_units)
        .map(|u| gate_thresh == 0 || dv_sum[u] / dv_cnt[u].max(1) >= gate_thresh)
        .collect();
    let n_trusted = trusted.iter().filter(|&&t| t).count();

    // ---- Candidate strengths ------------------------------------------------
    // Slow: two-stage full search — all 15 primary-only strengths, then the
    // top-K primaries crossed with every secondary (libaom pickcdef coverage).
    // Fast: the small legacy set. Index 0 is always (0,0) = unfiltered.
    let luma_tab = |pri: i32, sec: i32| -> Vec<i64> {
        cdef_luma_unit_dists(
            &snap_y, &src[0], w8, h8, disp_w, disp_h, &ldirs, &lvars, &lskip, nbx, uc, n_units,
            pri, sec, damping, bd, perceptual,
        )
    };
    // A unit only counts as gained when filtering clears the margin: a small
    // guaranteed improvement, standing in for the SSE/SSIMULACRA2 mismatch.
    let clears = |off: i64, on: i64| -> i64 {
        let thr = off - off.saturating_mul(margin) / 1000;
        if on < thr { off - on } else { 0 }
    };

    let slow = speed == Speed::Slow;
    let mut cands: Vec<(i32, i32)> = vec![(0, 0)];
    let mut ly: Vec<Vec<i64>> = vec![luma_tab(0, 0)];
    let luma_off: Vec<i64> = ly[0].clone();
    if slow {
        // Stage A: primary-only sweep. Strengths above 4 are deliberately NOT
        // searched: the corpus shows the decision metric over-rates strong
        // smoothing (SSIMULACRA2 regressions up to −0.4 when pri 5..15 entries
        // were offered), matching the legacy candidate cap.
        let pri_list: Vec<i32> = vec![1, 2, 4];
        let want = pool.width().min(pri_list.len());
        let tabs = pool.map_indexed(want, pri_list.len(), |i| luma_tab(pri_list[i], 0));
        let mut ranked: Vec<(i64, usize)> = tabs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    (0..n_units).map(|u| clears(luma_off[u], t[u])).sum::<i64>(),
                    i,
                )
            })
            .collect();
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push((pri_list[i], 0));
            ly.push(t);
        }
        // Stage B: top-K primaries (plus pri 0) crossed with the secondaries.
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut stage_b: Vec<(i32, i32)> = Vec::new();
        for sec in [1, 2] {
            stage_b.push((0, sec));
            for &(gain, i) in ranked.iter().take(4) {
                if gain > 0 {
                    stage_b.push((pri_list[i], sec));
                }
            }
        }
        let want = pool.width().min(stage_b.len().max(1));
        let tabs = pool.map_indexed(want, stage_b.len(), |i| {
            luma_tab(stage_b[i].0, stage_b[i].1)
        });
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push(stage_b[i]);
            ly.push(t);
        }
    } else {
        let list: Vec<(i32, i32)> = cdef::PRI_CANDIDATES
            .iter()
            .flat_map(|&pri| cdef::SEC_CANDIDATES.iter().map(move |&sec| (pri, sec)))
            .filter(|&(pri, sec)| !(pri == 0 && sec == 0))
            .collect();
        let want = pool.width().min(list.len().max(1));
        let tabs = pool.map_indexed(want, list.len(), |i| luma_tab(list[i].0, list[i].1));
        for (i, t) in tabs.into_iter().enumerate() {
            cands.push(list[i]);
            ly.push(t);
        }
    }

    // Chroma reuses the LUMA direction (remapped via uv_dir) and damping-1, and
    // is filtered at chroma sub-block granularity tied to the covering luma 8x8:
    // 8x8 for 4:4:4, 4x8 for 4:2:2, 4x4 for 4:2:0 — exactly as the decoders do
    // (dav1d cdef.fb[I444 - layout]). No per-block variance scaling for chroma.
    let uv_dir: [usize; 8] = if sub_x == 1 && sub_y == 0 {
        [7, 0, 2, 4, 5, 6, 6, 6] // 4:2:2
    } else {
        [0, 1, 2, 3, 4, 5, 6, 7] // 4:2:0 / 4:4:4 (identity)
    };
    let chroma_damping = damping - 1;
    let snap_uv: [Vec<u16>; 2] = if mono {
        [Vec::new(), Vec::new()]
    } else {
        [recon[1].clone(), recon[2].clone()]
    };
    // Chroma candidate space: (0,0) plus the same list as luma (its tables are
    // cheap relative to luma). Monochrome keeps only (0,0).
    let (c_cands, lc): (Vec<(i32, i32)>, Vec<Vec<i64>>) = if mono {
        (vec![(0, 0)], vec![vec![0; n_units]])
    } else {
        let unit_sse = |pri: i32, sec: i32| -> Vec<i64> {
            let u = cdef_chroma_unit_sse(
                &snap_uv[0],
                &src[1],
                cw8,
                ch8,
                cw_vis,
                ch_vis,
                &ldirs,
                &uv_dir,
                &lskip,
                nbx,
                nby,
                uc,
                n_units,
                sub_x,
                sub_y,
                pri,
                sec,
                chroma_damping,
                bd,
            );
            let v = cdef_chroma_unit_sse(
                &snap_uv[1],
                &src[2],
                cw8,
                ch8,
                cw_vis,
                ch_vis,
                &ldirs,
                &uv_dir,
                &lskip,
                nbx,
                nby,
                uc,
                n_units,
                sub_x,
                sub_y,
                pri,
                sec,
                chroma_damping,
                bd,
            );
            (0..n_units).map(|i| u[i] + v[i]).collect()
        };
        let want = pool.width().min(cands.len().max(1));
        let tabs = pool.map_indexed(want, cands.len(), |i| {
            if i == 0 {
                unit_sse(0, 0)
            } else {
                unit_sse(cands[i].0, cands[i].1)
            }
        });
        (cands.clone(), tabs)
    };
    let chroma_off: Vec<i64> = lc[0].clone();

    let d_off: Vec<i64> = (0..n_units).map(|u| luma_off[u] + chroma_off[u]).collect();
    // Margin threshold each unit must clear to filter (see `clears`).
    let thr: Vec<i64> = d_off
        .iter()
        .map(|&o| o - o.saturating_mul(margin) / 1000)
        .collect();

    // Greedy set construction over all strength PAIRS, exactly libaom's
    // `joint_strength_search_dual`: each added entry minimizes the frame total
    // when every unit picks its best entry (with the off-fallback margin rule
    // once the set contains the all-zero entry). The greedy is deterministic
    // and incremental, so one run to 8 entries yields the candidate sets for
    // every cdef_bits at its prefixes of size 1, 2, 4, 8.
    let nc = c_cands.len();
    let n_pairs = cands.len() * nc;
    let d_pair = |p: usize, u: usize| ly[p / nc][u] + lc[p % nc][u];
    let want = pool.width().min(n_pairs);

    // Global (cdef_bits = 0) candidate: the raw frame-total minimizer — every
    // signaled unit filters with it, no off-fallback.
    let raw_totals: Vec<i64> = pool.map_indexed(want, n_pairs, |p| {
        (0..n_units)
            .map(|u| if signaled[u] { d_pair(p, u) } else { d_off[u] })
            .sum()
    });
    let mut global_entry = 0usize;
    let mut prefix_tot = [i64::MAX; 4]; // totals at set sizes 1, 2, 4, 8
    for (p, &t) in raw_totals.iter().enumerate() {
        if t < prefix_tot[0] {
            prefix_tot[0] = t;
            global_entry = p;
        }
    }

    // Per-unit sets are seeded with the all-zero entry: a unit must always be
    // able to signal OFF (the Phase-1 {off, strength} table generalized), and
    // the margin rule sends units that don't clearly gain back to it.
    let mut cur_min: Vec<i64> = d_off.clone();
    let mut entries: Vec<usize> = vec![0];
    while entries.len() < 8 {
        let scores: Vec<i64> = pool.map_indexed(want, n_pairs, |p| {
            let mut tot = 0i64;
            for u in 0..n_units {
                if !signaled[u] {
                    tot += d_off[u];
                    continue;
                }
                let b = cur_min[u].min(d_pair(p, u));
                tot += if b >= thr[u] { d_off[u] } else { b };
            }
            tot
        });
        let mut best_p = 0usize;
        let mut best_tot = i64::MAX;
        for (p, &t) in scores.iter().enumerate() {
            if t < best_tot {
                best_tot = t;
                best_p = p;
            }
        }
        entries.push(best_p);
        for (u, m) in cur_min.iter_mut().enumerate() {
            *m = (*m).min(d_pair(best_p, u));
        }
        let n = entries.len();
        if n.is_power_of_two() && n <= 8 {
            prefix_tot[n.trailing_zeros() as usize] = best_tot;
        }
    }

    // A unit does not take its raw metric-minimizing entry: with several
    // entries available, per-unit metric errors accumulate (measured as SS2
    // regressions on detail-rich content). Instead each unit takes the MILDEST
    // entry that still clears the margin against off — least filtering among
    // the options the metric is confident in. Units clearing nothing take the
    // all-zero entry when present (else the raw minimum). The R-D totals are
    // computed from this realized assignment, not the greedy's raw minimum.
    let strength_mag = |p: usize| -> i32 {
        let (yp, ys) = cands[p / nc];
        let (up, us) = c_cands[p % nc];
        yp + ys + up + us
    };
    let assign = |set: &[usize]| -> (Vec<u8>, i64) {
        let z_in_set = set.iter().position(|&p| p == 0);
        let mut total = 0i64;
        let grid: Vec<u8> = (0..n_units)
            .map(|u| {
                if !signaled[u] {
                    total += d_off[u];
                    return 0;
                }
                let mut pick: Option<(i32, i64, usize)> = None; // (mag, d, e)
                let mut raw_best = (i64::MAX, 0usize);
                for (e, &p) in set.iter().enumerate() {
                    let d = d_pair(p, u);
                    if d < raw_best.0 {
                        raw_best = (d, e);
                    }
                    if d < thr[u] {
                        let mag = strength_mag(p);
                        if pick.is_none_or(|(m, pd, _)| mag < m || (mag == m && d < pd)) {
                            pick = Some((mag, d, e));
                        }
                    }
                }
                match (pick, z_in_set) {
                    (Some((_, d, e)), _) => {
                        total += d;
                        e as u8
                    }
                    (None, Some(z)) => {
                        total += d_off[u];
                        z as u8
                    }
                    (None, None) => {
                        total += raw_best.0;
                        raw_best.1 as u8
                    }
                }
            })
            .collect();
        (grid, total)
    };

    // Rate deltas against the always-written noop CDEF header: each strength
    // entry past the first costs 12 header bits (6 mono), and every signaled
    // unit carries a raw `cdef_idx` literal of `cdef_bits` bits.
    let lambda = mode_lambda_q(dc_q(base_q_idx, bd) as f32);
    let total_off: i64 = d_off.iter().sum();
    let cost_off = total_off as f32;
    let per_entry_bits = if mono { 6.0f32 } else { 12.0f32 };
    let mut best_bits: Option<u8> = None;
    let mut best_cost = cost_off;
    let max_bits = 3u8;
    for b in 0..=max_bits {
        let nb = 1usize << b;
        let total = if b == 0 {
            // Global: every signaled unit filters with `global_entry`.
            prefix_tot[0]
        } else {
            assign(&entries[..nb]).1
        };
        if total == i64::MAX {
            continue;
        }
        // Global filtering (cdef_bits = 0) has no per-unit off-switch, so it is
        // offered only when most units show a dominant edge direction (the
        // regime the metric is trusted in) and the whole frame clears the
        // margin. Per-unit modes are protected by the margin rule instead.
        if b == 0
            && (global_entry == 0
                || n_trusted * 4 < n_sig * 3
                || total >= total_off - total_off.saturating_mul(margin) / 1000)
        {
            continue;
        }
        let cost =
            total as f32 + lambda * (per_entry_bits * (nb - 1) as f32 + b as f32 * n_sig as f32);
        if cost < best_cost {
            best_cost = cost;
            best_bits = Some(b);
        }
    }

    let bits = best_bits?;
    let nb = 1usize << bits;
    let global_set = [global_entry];
    let set: &[usize] = if bits == 0 {
        &global_set
    } else {
        &entries[..nb]
    };
    let grid: Vec<u8> = if bits == 0 {
        vec![0; n_units]
    } else {
        assign(set).0
    };

    let strengths: Vec<(u8, u8, u8, u8)> = set
        .iter()
        .map(|&p| {
            let (yp, ys) = cands[p / nc];
            let (up, us) = c_cands[p % nc];
            (yp as u8, ys as u8, up as u8, us as u8)
        })
        .collect();
    // Per-unit effective strengths for the apply pass (constant for bits == 0;
    // all-skip units never filter regardless).
    let unit_y: Vec<(i32, i32)> = (0..n_units)
        .map(|u| {
            let (yp, ys, _, _) = strengths[grid[u] as usize];
            (yp as i32, ys as i32)
        })
        .collect();
    let unit_uv: Vec<(i32, i32)> = (0..n_units)
        .map(|u| {
            let (_, _, up, us) = strengths[grid[u] as usize];
            (up as i32, us as i32)
        })
        .collect();

    let decision = CdefFrameDecision {
        params: crate::obu::CdefParams {
            bits,
            damping: signaled_damping as u8,
            strengths,
        },
        grid: if bits > 0 { grid } else { Vec::new() },
        unit_cols: uc,
    };

    // Apply the decision to the reconstruction, reading every tap from the
    // pre-CDEF snapshot exactly as the decoder does.
    if unit_y.iter().any(|&(p, s)| p != 0 || s != 0) {
        apply_cdef_plane(
            &mut recon[0],
            &snap_y,
            w8,
            h8,
            &ldirs,
            &lvars,
            &lskip,
            nbx,
            uc,
            &unit_y,
            damping,
            bd,
            pool,
        );
    }
    if !mono && unit_uv.iter().any(|&(p, s)| p != 0 || s != 0) {
        for plane in 1..3 {
            apply_cdef_chroma(
                &mut recon[plane],
                &snap_uv[plane - 1],
                cw8,
                ch8,
                &ldirs,
                &uv_dir,
                skip8,
                sb8w,
                nbx,
                nby,
                sub_x,
                sub_y,
                uc,
                &unit_uv,
                chroma_damping,
                bd,
                pool,
            );
        }
    }

    Some(decision)
}

#[allow(clippy::too_many_arguments)]
fn cdef_block_dist_vis(
    src: &[u16],
    dst: &[u16],
    stride: usize,
    vis_w: usize,
    vis_h: usize,
    x: usize,
    y: usize,
    coeff_shift: u32,
    perceptual: bool,
) -> i64 {
    if perceptual && x + 8 <= vis_w && y + 8 <= vis_h {
        let rows = dst.len() / stride;
        return crate::cdef::cdef_dist_8x8(src, dst, stride, rows, x, y, coeff_shift);
    }
    let mut s = 0i64;
    for yy in y..(y + 8).min(vis_h) {
        for xx in x..(x + 8).min(vis_w) {
            let d = (dst[yy * stride + xx] - src[yy * stride + xx]) as i64;
            s += d * d;
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn cdef_luma_unit_dists(
    recon: &[u16],
    src: &[u16],
    w: usize,
    h: usize,
    disp_w: usize,
    disp_h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    n_units: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
    perceptual: bool,
) -> Vec<i64> {
    use crate::cdef;
    let coeff_shift = (bd - 8) as u32;
    let filtering = pri != 0 || sec != 0;
    let mut tmp = if filtering {
        recon.to_vec()
    } else {
        Vec::new()
    };
    let mut out = vec![0i64; n_units];
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let bi = (y / 8) * nbx + x / 8;
            if skip.get(bi).copied().unwrap_or(true) {
                continue;
            }
            if x >= disp_w || y >= disp_h {
                continue; // fully outside the visible frame
            }
            let u = (y / 64) * uc + x / 64;
            let dist = if filtering {
                // adjust_pri must be applied to the bit-depth-shifted strength
                // (matches the decoders' adjust_strength, which scales the
                // already-shifted level); scaling then shifting does not commute
                // because of the `+8 >> 4` rounding.
                let apri = cdef::adjust_pri(pri << (bd - 8), vars[bi]);
                cdef::cdef_filter_8x8(
                    &mut tmp,
                    recon,
                    w,
                    x,
                    y,
                    apri,
                    sec << (bd - 8),
                    // Decoders pass dir 0 when the SIGNALED pri strength is 0
                    // (`pri_strength ? dir : 0`) — sec-only filtering then runs
                    // on direction 0, not the block's estimated direction.
                    if pri == 0 { 0 } else { dirs[bi] },
                    damping,
                    bd,
                );
                cdef_block_dist_vis(src, &tmp, w, disp_w, disp_h, x, y, coeff_shift, perceptual)
            } else {
                cdef_block_dist_vis(src, recon, w, disp_w, disp_h, x, y, coeff_shift, perceptual)
            };
            out[u] += dist;
        }
    }
    out
}

/// Per-64x64-unit chroma SSE for one candidate strength (`0,0` = baseline),
/// accumulated into the covering LUMA unit (the cdef_idx is shared with luma).
/// One chroma sub-block per non-skip luma 8x8: 8x8 (4:4:4), 4x8 (4:2:2),
/// 4x4 (4:2:0). Plain SSE — chroma sub-blocks are too small for the 8x8
/// variance calibration of the perceptual metric (libaom also uses MSE here).
#[allow(clippy::too_many_arguments)]
fn cdef_chroma_unit_sse(
    recon: &[u16],
    src: &[u16],
    cw: usize,
    _ch: usize,
    cw_vis: usize,
    ch_vis: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    lskip: &[bool],
    nbx: usize,
    nby: usize,
    uc: usize,
    n_units: usize,
    sub_x: usize,
    sub_y: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    bd: u8,
) -> Vec<i64> {
    use crate::cdef;
    let cbw = 8 >> sub_x;
    let cbh = 8 >> sub_y;
    let filtering = pri != 0 || sec != 0;
    let mut tmp = if filtering {
        recon.to_vec()
    } else {
        Vec::new()
    };
    let mut out = vec![0i64; n_units];
    for lby in 0..nby {
        for lbx in 0..nbx {
            if lskip.get(lby * nbx + lbx).copied().unwrap_or(true) {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            let cy = (lby * 8) >> sub_y;
            if cx >= cw_vis || cy >= ch_vis {
                continue; // fully outside the visible chroma frame
            }
            let u = (lby / 8) * uc + lbx / 8;
            let cand: &[u16] = if filtering {
                // dir 0 when the signaled pri strength is 0 (see luma above).
                let dir = if pri == 0 {
                    0
                } else {
                    uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)]
                };
                cdef::cdef_filter_block(
                    &mut tmp,
                    0,
                    recon,
                    cw,
                    cx,
                    cy,
                    cbw,
                    cbh,
                    pri << (bd - 8),
                    sec << (bd - 8),
                    dir,
                    damping,
                    bd,
                );
                &tmp
            } else {
                recon
            };
            let mut sse = 0i64;
            for yy in cy..(cy + cbh).min(ch_vis) {
                for xx in cx..(cx + cbw).min(cw_vis) {
                    let d = (cand[yy * cw + xx] - src[yy * cw + xx]) as i64;
                    sse += d * d;
                }
            }
            out[u] += sse;
        }
    }
    out
}

/// Apply the luma CDEF strength to a whole plane in place, reading from the
/// MARKED pre-CDEF snapshot (out-of-frame samples = `CDEF_VERY_LARGE`) so every
/// 8x8 filters the same source pixels the decoder does. With a per-unit `grid`
/// (`cdef_bits = 1`), only 8x8s inside on-units are filtered.
#[allow(clippy::too_many_arguments)]
fn apply_cdef_plane(
    plane: &mut [u16],
    snapshot: &[u16],
    w: usize,
    _h: usize,
    dirs: &[usize],
    vars: &[i32],
    skip: &[bool],
    nbx: usize,
    uc: usize,
    unit_pri_sec: &[(i32, i32)],
    damping: i32,
    bd: u8,
    pool: &Pool,
) {
    use crate::cdef;
    // Every 8x8 reads the pre-CDEF snapshot and writes only its own rows, so
    // 8-row bands filter in parallel with identical output to the serial pass.
    let items: Vec<(usize, &mut [u16])> = plane.chunks_mut(8 * w).enumerate().collect();
    pool.for_each(pool.width(), items, |(bi8, band)| {
        let y = bi8 * 8;
        for x in (0..w).step_by(8) {
            let bxi = x / 8;
            let byi = y / 8;
            let bi = byi * nbx + bxi;
            // Skip blocks are left untouched, matching the decoder.
            if skip.get(bi).copied().unwrap_or(true) {
                continue;
            }
            // The unit's signaled strength; (0,0) units stay untouched.
            let (pri, sec) = unit_pri_sec[(y / 64) * uc + x / 64];
            if pri == 0 && sec == 0 {
                continue;
            }
            // Use the precomputed direction/variance (same pre-CDEF snapshot the
            // search used). Identical to the old in-loop recompute, but free.
            // Decoders force dir 0 when the signaled pri strength is 0.
            let dir = if pri == 0 { 0 } else { dirs[bi] };
            let var = vars[bi];
            // adjust_pri on the bit-depth-shifted strength (see search above).
            let apri = cdef::adjust_pri(pri << (bd - 8), var);
            cdef::cdef_filter_block(
                band,
                y,
                snapshot,
                w,
                x,
                y,
                8,
                8,
                apri,
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn apply_cdef_chroma(
    plane: &mut [u16],
    snapshot: &[u16],
    cw: usize,
    ch: usize,
    ldirs: &[usize],
    uv_dir: &[usize; 8],
    skip8: &[bool],
    sb8w: usize,
    nbx: usize,
    nby: usize,
    sub_x: usize,
    sub_y: usize,
    uc: usize,
    unit_pri_sec: &[(i32, i32)],
    damping: i32,
    bd: u8,
    pool: &Pool,
) {
    use crate::cdef;
    let cbw = 8 >> sub_x; // chroma sub-block width per luma 8x8
    let cbh = 8 >> sub_y;
    // One band per luma block-row (cbh chroma rows): disjoint writes, shared
    // pre-CDEF snapshot reads — identical output to the serial pass.
    let items: Vec<(usize, &mut [u16])> = plane.chunks_mut(cbh * cw).enumerate().collect();
    pool.for_each(pool.width(), items, |(lby, band)| {
        if lby >= nby {
            return;
        }
        let cy = (lby * 8) >> sub_y;
        for lbx in 0..nbx {
            // Skip if the covering luma 8x8 is a skip block.
            if skip8.get(lby * sb8w + lbx).copied().unwrap_or(true) {
                continue;
            }
            // The unit's signaled chroma strength; (0,0) units stay untouched.
            let (pri, sec) = unit_pri_sec[(lby / 8) * uc + lbx / 8];
            if pri == 0 && sec == 0 {
                continue;
            }
            let cx = (lbx * 8) >> sub_x;
            if cx >= cw || cy >= ch {
                continue;
            }
            // Decoders force dir 0 when the signaled pri strength is 0.
            let dir = if pri == 0 {
                0
            } else {
                uv_dir[ldirs.get(lby * nbx + lbx).copied().unwrap_or(0)]
            };
            cdef::cdef_filter_block(
                band,
                cy,
                snapshot,
                cw,
                cx,
                cy,
                cbw,
                cbh,
                pri << (bd - 8),
                sec << (bd - 8),
                dir,
                damping,
                bd,
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn cdblk_variant() -> u32 {
    1
}

#[allow(clippy::too_many_arguments)]
fn frame_deblock(
    loopfilter: &crate::loopfilter::LoopFilterDispatch,
    pool: &Pool,
    recon: &mut [Vec<u16>; 3],
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    disp_w: usize,
    disp_h: usize,
    blk4: &[u8],    // luma block width map (vertical edges)
    blk4h: &[u8],   // luma block height map (horizontal edges)
    blk4v: &[bool], // luma block starts at this 4x4 column
    blk4t: &[bool], // luma block starts at this 4x4 row
    pblk4: &[u8],   // luma PREDICTION-block width map (chroma edges derive from this)
    pblk4h: &[u8],  // luma PREDICTION-block height map
    pblk4v: &[bool],
    pblk4t: &[bool],
    nc4: usize, // luma 4-col count == w8/4
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    level_y: i32,
    level_uv: i32,
    bd: u8,
) {
    if level_y > 0 {
        crate::loopfilter::filter_plane_parallel(
            loopfilter,
            &mut recon[0],
            w8,
            h8,
            disp_w,
            disp_h,
            blk4,
            blk4h,
            blk4v,
            blk4t,
            nc4,
            level_y,
            true,
            16, // 64px superblock -> 16 4-unit rows
            bd,
            pool,
        );
    }
    if mono || level_uv <= 0 {
        return;
    }
    let ss_hor = sub_x;
    let ss_ver = sub_y;
    let cw = cw8;
    let ch = ch8;
    let cnc4 = cw / 4;
    let cnr4 = ch / 4;
    // Chroma transform geometry comes from the PREDICTION-block map, not the
    // luma transform map: chroma carries no tx_depth here, so one chroma
    // transform spans the whole chroma block even where luma split its own
    // transform. Deriving from `blk4` (transform-granular) invented interior
    // chroma edges the decoder never filters. The block-start flags are carried
    // across too — the alignment fallback in `filter_plane` cannot place edges
    // for blocks whose origin is not a multiple of their size (asymmetric and
    // rectangular partitions).
    let mut cbw4 = vec![0u8; cnc4 * cnr4];
    let mut cbh4 = vec![0u8; cnc4 * cnr4];
    let mut cbv4 = vec![false; cnc4 * cnr4];
    let mut cbt4 = vec![false; cnc4 * cnr4];
    for cr in 0..cnr4 {
        for cc in 0..cnc4 {
            let lr = cr << ss_ver;
            let lc = cc << ss_hor;
            let li = lr * nc4 + lc;
            let ci = cr * cnc4 + cc;
            // Variant matrix: bit0 = use PREDICTION geometry for the
            // size map (else the luma TRANSFORM map, as before); bit1 = pass
            // explicit block-start flags (else the alignment fallback).
            let v = cdblk_variant();
            let (sw, sh) = if v & 1 != 0 {
                (pblk4[li], pblk4h[li])
            } else {
                (blk4[li], blk4h[li])
            };
            // AV1 caps a chroma transform at 32x32 (`av1_get_max_uv_txsize`).
            // In 4:2:2 a 64x64 luma block yields a 32x64 chroma block, which is
            // coded as stacked 32x32 transforms — so the deblock geometry must
            // clamp to 8 4-units or it misses the interior chroma edge.
            const MAX_UV_TX4: u8 = 8; // 32 px
            cbw4[ci] = (sw >> ss_hor).max(1).min(MAX_UV_TX4);
            cbh4[ci] = (sh >> ss_ver).max(1).min(MAX_UV_TX4);
            if v & 2 != 0 {
                cbv4[ci] = if v & 1 != 0 { pblk4v[li] } else { blk4v[li] };
                cbt4[ci] = if v & 1 != 0 { pblk4t[li] } else { blk4t[li] };
            }
        }
    }
    let csb = 16 >> ss_ver;
    let cvis_w = disp_w.div_ceil(1 << ss_hor);
    let cvis_h = disp_h.div_ceil(1 << ss_ver);
    let cbv4 = if cdblk_variant() & 2 != 0 {
        cbv4.as_slice()
    } else {
        &[]
    };
    let cbt4 = if cdblk_variant() & 2 != 0 {
        cbt4.as_slice()
    } else {
        &[]
    };

    for pixels in &mut recon[1..] {
        crate::loopfilter::filter_plane_parallel(
            loopfilter, pixels, cw, ch, cvis_w, cvis_h, &cbw4, &cbh4, cbv4, cbt4, cnc4, level_uv,
            false, csb, bd, pool,
        );
    }
}

/// Concatenate per-tile payloads into a tile-group. A single tile is returned
/// verbatim (no header byte, no size prefix). For `NumTiles > 1` the spec
/// `tile_group_obu` prepends `tile_start_and_end_present_flag = 0` followed by
/// `byte_alignment()` (one `0x00` byte), then every tile except the last is
/// prefixed with `tile_size_minus_1` as `TileSizeBytes = 4` little-endian bytes.
fn assemble_tilegroup(payloads: Vec<Vec<u8>>) -> Vec<u8> {
    if payloads.len() == 1 {
        return payloads.into_iter().next().unwrap();
    }
    let mut out = Vec::new();
    out.push(0u8);
    let last = payloads.len() - 1;
    for (i, p) in payloads.iter().enumerate() {
        if i != last {
            let sz_minus_1 = (p.len() - 1) as u32; // TileSizeBytes = 4
            out.extend_from_slice(&sz_minus_1.to_le_bytes());
        }
        out.extend_from_slice(p);
    }
    out
}

/// Build the frame OBU(s) that follow the sequence header. A single tile is
/// emitted as one combined `OBU_FRAME` (type 6) — byte-identical to the previous
/// output. Multi-tile frames are emitted as a separate `OBU_FRAME_HEADER`
/// (type 3) + `OBU_TILE_GROUP` (type 4), which strict parsers (ffmpeg's
/// `av1_frame_merge` BSF) handle reliably where a multi-tile combined
/// `OBU_FRAME` does not.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_frame_obus(
    base_q_idx: u8,
    qm: QmLevels,
    uac_delta: i32,
    udc_delta: i32,
    plan: &Tiling,
    tilegroup: &[u8],
    mono: bool,
    aq: bool,
    allow_intrabc: bool,
    cdef: Option<&crate::obu::CdefParams>,
    lr: Option<&crate::obu::LrParams>,
    updating_cdf: bool,
) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = frame_header_lossy_multitile_th(
            base_q_idx,
            qm,
            uac_delta,
            udc_delta,
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            mono,
            aq,
            allow_intrabc,
            cdef,
            lr,
            updating_cdf,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = frame_header_lossy_multitile(
            base_q_idx,
            qm,
            uac_delta,
            udc_delta,
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            mono,
            aq,
            allow_intrabc,
            cdef,
            lr,
            updating_cdf,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossless_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let pool = Pool::new(threads);
    let (tilegroup, plan) = encode_lossless_tilegroup(
        bd,
        w8,
        h8,
        visible_w,
        visible_h,
        src,
        &pool,
        speed,
        updating_cdf,
    );
    assemble_lossless_frame_obus(&plan, &tilegroup, updating_cdf)
}

#[allow(clippy::too_many_arguments)]
fn encode_one_lossless_tile(
    bd: u8,
    full_w: usize,
    visible_w: usize,
    visible_h: usize,
    src: &[Vec<i16>; 3],
    r: &(usize, usize, usize, usize),
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(&src[0], full_w, x0, y0, tw, th);
    let p1 = crop_plane(&src[1], full_w, x0, y0, tw, th);
    let p2 = crop_plane(&src[2], full_w, x0, y0, tw, th);
    let tile_visible_w = visible_w.saturating_sub(x0).min(tw);
    let tile_visible_h = visible_h.saturating_sub(y0).min(th);
    crate::tile::encode_tile_lossless(
        tw,
        th,
        tile_visible_w,
        tile_visible_h,
        bd,
        [&p0, &p1, &p2],
        speed,
        updating_cdf,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_lossless_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    src: &[Vec<i16>; 3],
    pool: &Pool,
    speed: Speed,
    updating_cdf: bool,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = pool.width();
    let tile_target = want.min((sb_cols as usize) * (sb_rows as usize)).max(1);
    let plan = plan_tiling(sb_cols, sb_rows, tile_target);
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    // Tile pixel rectangles in raster order (top-to-bottom, left-to-right).
    let mut rects: Vec<(usize, usize, usize, usize)> =
        Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            rects.push((x0, y0, x1 - x0, y1 - y0));
        }
    }

    let n = rects.len();
    let nthreads = want.clamp(1, n.max(1));
    let payloads: Vec<Vec<u8>> = pool.map_indexed(nthreads, n, |i| {
        encode_one_lossless_tile(
            bd,
            w8,
            visible_w,
            visible_h,
            src,
            &rects[i],
            speed,
            updating_cdf,
        )
    });

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a lossless tile group with the matching frame header: a single tile uses
/// a combined `OBU_FRAME` (type 6); multiple tiles use a separate
/// `OBU_FRAME_HEADER` (type 3) + `OBU_TILE_GROUP` (type 4), the layout strict
/// parsers (ffmpeg's cbs_av1) accept.
fn assemble_lossless_frame_obus(plan: &Tiling, tilegroup: &[u8], updating_cdf: bool) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            updating_cdf,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = crate::obu::frame_header_lossless_multitile(
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            updating_cdf,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Crop the single luma plane to a tile rect and encode it as a mono lossless
/// tile. Pure function of its inputs (safe on any thread).
#[allow(clippy::too_many_arguments)]
fn encode_one_lossless_tile_mono(
    bd: u8,
    full_w: usize,
    visible_w: usize,
    visible_h: usize,
    luma: &[i16],
    r: &(usize, usize, usize, usize),
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(luma, full_w, x0, y0, tw, th);
    let tile_visible_w = visible_w.saturating_sub(x0).min(tw);
    let tile_visible_h = visible_h.saturating_sub(y0).min(th);
    crate::tile::encode_tile_lossless_mono(
        tw,
        th,
        tile_visible_w,
        tile_visible_h,
        bd,
        &p0,
        speed,
        updating_cdf,
    )
}

/// Monochrome counterpart of [`encode_lossless_tilegroup`]: a single full-res
/// luma plane (`w8*h8`, padded to a multiple of 8) tiled identically to the
/// 4:4:4 path. Byte-identical output for a fixed tiling regardless of thread
/// count.
#[allow(clippy::too_many_arguments)]
fn encode_lossless_mono_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    luma: &[i16],
    pool: &Pool,
    speed: Speed,
    updating_cdf: bool,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = pool.width();
    // One tile per worker, spec- and superblock-clamped. See
    // `encode_lossless_tilegroup` for the full rationale.
    let tile_target = want.min((sb_cols as usize) * (sb_rows as usize)).max(1);
    let plan = plan_tiling(sb_cols, sb_rows, tile_target);
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    let mut rects: Vec<(usize, usize, usize, usize)> =
        Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            rects.push((x0, y0, x1 - x0, y1 - y0));
        }
    }

    let n = rects.len();
    let nthreads = want.clamp(1, n.max(1));
    let payloads: Vec<Vec<u8>> = pool.map_indexed(nthreads, n, |i| {
        encode_one_lossless_tile_mono(
            bd,
            w8,
            visible_w,
            visible_h,
            luma,
            &rects[i],
            speed,
            updating_cdf,
        )
    });

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a mono lossless tile group with a `mono_chrome = 1` lossless frame
/// header (single tile ⇒ combined `OBU_FRAME`; multi-tile ⇒ `OBU_FRAME_HEADER` +
/// `OBU_TILE_GROUP`).
fn assemble_lossless_mono_frame_obus(
    plan: &Tiling,
    tilegroup: &[u8],
    updating_cdf: bool,
) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_mono_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            updating_cdf,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = crate::obu::frame_header_lossless_mono_multitile(
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            updating_cdf,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Encode a monochrome lossless frame's OBU portion from a padded `w8*h8` luma
/// plane. Caller prepends temporal delimiter, sequence header, metadata.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossless_mono_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    luma: &[i16],
    threads: usize,
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let pool = Pool::new(threads);
    let (tilegroup, plan) = encode_lossless_mono_tilegroup(
        bd,
        w8,
        h8,
        visible_w,
        visible_h,
        luma,
        &pool,
        speed,
        updating_cdf,
    );
    assemble_lossless_mono_frame_obus(&plan, &tilegroup, updating_cdf)
}

#[cfg(test)]
mod aq_tests {
    use super::*;

    fn emit_semantic_fixture(enc: &mut OdEcEncoder, cdfs: &mut Cdfs) {
        for i in 0..64 {
            let skip_ctx = i % cdfs.skip.len();
            enc.encode_symbol(i & 1, &mut cdfs.skip[skip_ctx]);
            enc.encode_bool(i & 2 != 0, 4096 + (i as u16 * 257));

            let part_ctx = i % cdfs.part_split[0].len();
            let part_n = cdfs.part_split[0][part_ctx].len() - 1;
            enc.encode_symbol(i % part_n, &mut cdfs.part_split[0][part_ctx]);
            enc.encode_gathered_partition(i & 4 != 0, &cdfs.part_split[0][part_ctx]);

            enc.encode_symbol_noupdate(i & 1, &cdfs.intrabc);
        }
    }

    #[test]
    fn semantic_tokens_match_live_entropy_coding() {
        for updating_cdf in [false, true] {
            let mut capture_cdfs = Cdfs::new(crate::coef_q::qcat(140));
            let capture_slots = capture_cdfs.semantic_slots();
            let mut capture = OdEcEncoder::new().with_updating_cdf(false);
            capture.set_semantic_cdfs(&capture_slots);
            capture.begin_semantic_sink();
            emit_semantic_fixture(&mut capture, &mut capture_cdfs);
            let tokens = capture.take_semantic();

            let mut reference_cdfs = Cdfs::new(crate::coef_q::qcat(140));
            let mut reference = OdEcEncoder::new().with_updating_cdf(updating_cdf);
            emit_semantic_fixture(&mut reference, &mut reference_cdfs);
            let expected = reference.done();

            let mut packed_cdfs = Cdfs::new(crate::coef_q::qcat(140));
            let mut packed_slots = packed_cdfs.semantic_slots();
            let mut packed = OdEcEncoder::new().with_updating_cdf(updating_cdf);
            packed.replay_semantic(&tokens, &mut packed_slots);

            assert_eq!(packed.done(), expected, "updating_cdf={updating_cdf}");
            assert_eq!(packed_cdfs.skip, reference_cdfs.skip);
            assert_eq!(packed_cdfs.part_split, reference_cdfs.part_split);
        }
    }

    #[test]
    fn directional_top_k_keeps_three_lowest_costs() {
        let mut top = DirectionalTopK::new();
        for (mode, cost) in [(1, 50), (2, 10), (3, 30), (4, 20), (5, 40)] {
            top.insert(mode, cost);
        }
        assert!(top.contains(2));
        assert!(top.contains(3));
        assert!(top.contains(4));
        assert!(!top.contains(1));
        assert!(!top.contains(5));
    }

    #[test]
    fn satd_sad_proxy_is_zero_only_for_equal_blocks() {
        let rd = crate::rd_sse::RdDispatch::selected();
        let src = [7u16; 16];
        let pred = [7i32; 16];
        assert_eq!(rd.satd_sad_proxy(&src, 4, &pred, 4, 4, 4), 0);
        let mut pred2 = pred;
        pred2[5] += 3;
        assert!(rd.satd_sad_proxy(&src, 4, &pred2, 4, 4, 4) > 0);
    }

    #[test]
    fn moment_grid_matches_independent_4x4_blocks() {
        const STRIDE: usize = 23;
        let plane: Vec<i32> = (0..STRIDE * 19)
            .map(|i| ((i * 37 + i / STRIDE * 11) & 1023) as i32)
            .collect();
        let (px, py) = (3, 2);
        let cells = block_moment_grid_16x16(&plane, STRIDE, px, py);
        for cy in 0..4 {
            for cx in 0..4 {
                assert_eq!(
                    cells[cy * 4 + cx],
                    block_moments_i32(&plane, STRIDE, px + cx * 4, py + cy * 4, 4, 4),
                );
            }
        }
    }

    #[test]
    fn raw_sse_guard_tracks_the_shadow_rd_choice() {
        assert!(raw_sse_guard_choice(
            "test",
            RawSseGuard::FilterIntra,
            100,
            99,
            20.0,
            10.0,
            true,
        ));
        assert!(!raw_sse_guard_choice(
            "test",
            RawSseGuard::FilterIntra,
            100,
            101,
            20.0,
            10.0,
            false,
        ));
    }

    #[test]
    fn wavefront_falls_back_only_when_the_sb_graph_is_too_narrow() {
        assert!(wavefront_should_use_tiles(24, 14, 12));
        assert!(!wavefront_should_use_tiles(26, 17, 8));
        assert!(!wavefront_should_use_tiles(110, 73, 12));
        assert!(!wavefront_should_use_tiles(24, 14, 1));
    }

    /// `precompute_aq_grid` must reproduce the serial `aq_begin_sb` accumulator
    /// walk bit-exactly — every cell's qindex, signaled steps, and the resulting
    /// quantizer state — on a padded (non-64-aligned) frame with mixed
    /// dark/flat/textured content, with and without Variance Boost, so the
    /// wavefront can consume cells out of raster order.
    #[test]
    fn aq_grid_matches_serial() {
        // 200x136 luma: 4x3 SB grid with 8px right / bottom partial SBs.
        let (w, h) = (200usize, 136usize);
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for c in 0..w {
                // dark gradient + texture patches + flat bands
                let base = ((r * 255) / h) as i32 / 3;
                let tex = if (r / 32 + c / 32) % 3 == 0 {
                    (((r * 7 + c * 13) % 53) as i32) - 26
                } else {
                    0
                };
                y[r * w + c] = (base + tex).clamp(0, 255) as u16;
            }
        }
        let src = [y, vec![128; w * h], vec![128; w * h]];
        for vb_enabled in [false, true] {
            for base_q in [60u8, 140, 220] {
                let mk = || {
                    let mut t = LossyTile::new(base_q, 8, w, h, &src, QmLevels::FLAT);
                    let ref_act = tile_ref_activity(&t.src[0], t.w, t.w, t.h);
                    let vb = VarianceBoost {
                        enabled: vb_enabled,
                        octile: 6,
                        strength: 1.0,
                        boost_only: false,
                        dark: DarkAq::on(),
                        base_shift: 0,
                        qm: QmLevels::FLAT,
                    };
                    t.enable_aq(base_q, ref_act, &vb);
                    t
                };
                let mut serial = mk();
                let grid_tile = mk();
                let grid = grid_tile.precompute_aq_grid();
                let (rows, cols) = (h.div_ceil(64), w.div_ceil(64));
                assert_eq!(grid.len(), rows * cols);
                let mut i = 0;
                for sb_y in (0..h).step_by(64) {
                    for sb_x in (0..w).step_by(64) {
                        serial.aq_begin_sb(sb_x, sb_y);
                        let cell = grid[i];
                        assert_eq!(
                            cell.newq as i32, serial.aq.cur_qidx,
                            "qidx sb {i} vb {vb_enabled} q {base_q}"
                        );
                        assert_eq!(
                            cell.steps, serial.aq.pending,
                            "steps sb {i} vb {vb_enabled} q {base_q}"
                        );
                        i += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn skipped_whole_64_does_not_consume_delta_q() {
        let src = [
            vec![128u16; 64 * 64],
            vec![128u16; 32 * 32],
            vec![128u16; 32 * 32],
        ];
        let make_tile = || {
            let mut tile = LossyTile::new_420(100, 8, 64, 64, &src, QmLevels::FLAT);
            tile.aq.enabled = true;
            tile.aq.prev_qidx = 100;
            tile.aq.cur_qidx = 98;
            tile.aq.read_deltas = true;
            tile.aq.pending = -2;
            tile
        };

        let mut skipped = make_tile();
        skipped.code_skip_and_sb_tokens_64(true, 0);
        assert!(
            skipped.aq.read_deltas,
            "a skipped whole superblock must return before read_delta_qindex"
        );
        assert_eq!(skipped.aq.cur_qidx, 100, "skipped SB must roll AQ back");

        let mut coded = make_tile();
        coded.code_skip_and_sb_tokens_64(false, 0);
        assert!(
            !coded.aq.read_deltas,
            "a non-skipped whole superblock must consume its delta-Q"
        );
        assert_eq!(
            coded.aq.cur_qidx, 98,
            "coded SB keeps its signaled AQ qindex"
        );
    }

    /// Dark protection over a full 64×64 SB of the given i32 plane, with the AV1
    /// 8-bit normalization scale for bit depth `bd`.
    fn dark_prot(d: &DarkAq, base_q: i32, yp: &[i32], bd: u8) -> i32 {
        let scale = 1.0 / (1u32 << (bd - 8)) as f32;
        crate::aq_common::dark_protection(d, base_q, yp, 64, 0, 0, 64, 64, scale)
    }

    #[test]
    fn dark_protection_targets_dark_structure_only() {
        // 64x64 SBs at three luma levels, each with a coherent striped texture (survives
        // the 2x downsample, so it registers as structure not noise). Ported from
        // `av2/aq.rs`, on the i32 luma plane the AV1 encoder uses.
        let build = |mean: i32| -> Vec<i32> {
            let mut p = vec![0i32; 64 * 64];
            for r in 0..64 {
                for c in 0..64 {
                    // 4px-wide stripes: strong at both scales.
                    p[r * 64 + c] = mean + if (c / 4) % 2 == 0 { 12 } else { -12 };
                }
            }
            p
        };
        let d = DarkAq::on(); // enabled, min_q = 90 (2026-07-24 gate ship)
        let dark = build(36);
        let bright = build(180);
        let flat_dark = vec![24i32; 64 * 64]; // dark but no structure

        let base_q = 190; // in the gated range (>= 150)
        let d_dark = dark_prot(&d, base_q, &dark, 8);
        let d_bright = dark_prot(&d, base_q, &bright, 8);
        let d_flat = dark_prot(&d, base_q, &flat_dark, 8);
        assert!(
            d_dark > 0,
            "dark structured SB should be protected, got {d_dark}"
        );
        assert_eq!(d_bright, 0, "bright structured SB must not be protected");
        assert_eq!(
            d_flat, 0,
            "dark flat SB (no structure) must not be protected"
        );

        // Gate: disabled below min_q (high quality / low qindex).
        assert_eq!(dark_prot(&d, 60, &dark, 8), 0, "gated out below min_q");
        // Disabled config is always inert.
        assert_eq!(
            dark_prot(&DarkAq::off(), base_q, &dark, 8),
            0,
            "disabled dark AQ must not protect"
        );
    }

    #[test]
    fn dark_protection_bitdepth_normalized() {
        // A 10-bit dark structured SB (all values << 2) must score the SAME protection
        // as its 8-bit counterpart: the stats normalize native depth to 8-bit range.
        let build = |mean: i32, shift: u32| -> Vec<i32> {
            let mut p = vec![0i32; 64 * 64];
            for r in 0..64 {
                for c in 0..64 {
                    p[r * 64 + c] = (mean + if (c / 4) % 2 == 0 { 12 } else { -12 }) << shift;
                }
            }
            p
        };
        let d = DarkAq::on();
        let base_q = 190;
        let p8 = build(36, 0);
        let p10 = build(36, 2);
        let d8 = dark_prot(&d, base_q, &p8, 8);
        let d10 = dark_prot(&d, base_q, &p10, 10);
        assert!(d8 > 0);
        assert_eq!(d8, d10, "dark protection must be bit-depth invariant");
    }

    #[test]
    fn chroma_rd_can_favor_a_split_on_color_detail() {
        let mut src = [
            vec![128u16; 32 * 32],
            vec![128u16; 32 * 32],
            vec![128u16; 32 * 32],
        ];
        for y in 0..16 {
            for x in 0..16 {
                let quadrant = (x >= 8) as usize + 2 * (y >= 8) as usize;
                src[1][y * 32 + x] = [24, 232, 216, 40][quadrant];
                src[2][y * 32 + x] = [224, 48, 32, 240][quadrant];
            }
        }
        let tile = LossyTile::new(160, 8, 32, 32, &src, QmLevels::FLAT);
        let none = tile.rd_cost_chroma_partition(0, 0, 16, Part16::None, 1.0, false);
        let split = tile.rd_cost_chroma_partition(0, 0, 16, Part16::Split, 1.0, false);
        assert!(
            split < none,
            "four color-homogeneous chroma blocks should beat one mixed block: split={split}, none={none}"
        );
    }

    #[test]
    fn uv_rate_matches_emitted_cdf_transaction() {
        let src = [
            vec![128u16; 16 * 16],
            vec![128u16; 16 * 16],
            vec![128u16; 16 * 16],
        ];
        let tile = LossyTile::new(120, 8, 16, 16, &src, QmLevels::FLAT);
        let y_mode = PAETH_PRED;
        assert_eq!(
            tile.uv_mode_bits(y_mode, DC_PRED, None),
            cdf_cost(&tile.dcdf().uv_mode[13 + y_mode], DC_PRED)
        );
        let a = [2, -3];
        let expected = cdf_cost(&tile.dcdf().uv_mode[13 + y_mode], CFL_PRED)
            + cdf_cost(&tile.dcdf().cfl_sign, 6)
            + cdf_cost(&tile.dcdf().cfl_alpha[4], 1)
            + cdf_cost(&tile.dcdf().cfl_alpha[2], 2);
        assert_eq!(tile.uv_mode_bits(y_mode, CFL_PRED, Some(a)), expected);
    }

    #[test]
    fn local_chroma_weight_responds_to_local_color_detail() {
        let flat = [
            vec![128u16; 16 * 16],
            vec![128u16; 16 * 16],
            vec![128u16; 16 * 16],
        ];
        let mut color = flat.clone();
        for y in 0..16 {
            for x in 0..16 {
                color[1][y * 16 + x] = if x < 8 { 24 } else { 232 };
                color[2][y * 16 + x] = if y < 8 { 224 } else { 32 };
            }
        }
        let flat_tile = LossyTile::new(120, 8, 16, 16, &flat, QmLevels::FLAT);
        let color_tile = LossyTile::new(120, 8, 16, 16, &color, QmLevels::FLAT);
        assert!(
            color_tile.chroma_partition_weight_at(0, 0, 16, 16)
                > flat_tile.chroma_partition_weight_at(0, 0, 16, 16)
        );
    }

    #[test]
    fn block64_mode_trial_restores_reconstruction() {
        let mut src = [
            vec![0u16; 64 * 64],
            vec![128u16; 64 * 64],
            vec![128u16; 64 * 64],
        ];
        for y in 0..64 {
            for x in 0..64 {
                src[0][y * 64 + x] = ((x * 5 + y * 3 + (x ^ y)) & 255) as u16;
            }
        }
        let mut tile =
            LossyTile::new(160, 8, 64, 64, &src, QmLevels::FLAT).with_speed(Speed::Medium);
        let before = tile.recon[0].clone();
        let (_, cost) = tile.rd_pick_luma64(0, 0, true, false, 1.0);
        assert!(cost.is_finite());
        assert_eq!(tile.recon[0], before);
    }
}
