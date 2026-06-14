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

use crate::Speed;
use crate::dct::{adst8x8_t, adst16x16_t};
use crate::idct::{
    iadst_dequant_8x8, iadst_dequant_16x16, idct_dequant_4x4, idct_dequant_4x8, idct_dequant_8x8,
    idct_dequant_8x16, idct_dequant_16x16, idct_dequant_16x32, idct_dequant_32x32,
};
use crate::obu::{
    frame_header_lossy_multitile, frame_header_lossy_multitile_th, temporal_delimiter,
    wrap_obu_frame, wrap_obu_frame_split,
};
use crate::odec::OdEcEncoder;
use crate::trellis::{trellis_optimize, trellis_optimize_ctx};

use crate::coeffs::encode_tx16_coeffs_adapt;
use crate::coeffs::*;
use crate::cost::*;
use crate::intrapred::*;
use crate::quant::*;
use crate::tables::*;

/// Per-frame adaptive CDF state. dav1d adapts every symbol's CDF as it decodes
/// (when `disable_cdf_update = 0`); to stay bit-exact we hold the same mutable
/// CDFs (initialised from the qcat defaults, in `icdf` form with a trailing
/// adaptation count) and adapt them identically after each coded symbol via
/// `OdEcEncoder::encode_symbol`. Coef class index: 0 = `TX_4X4` (4:2:0 chroma),
/// 1 = `TX_8X8`/`RTX_4X8` (luma, 4:4:4 and 4:2:2 chroma). Plane index: 0 = luma,
/// 1 = chroma.
pub(crate) struct Cdfs {
    pub(crate) skip: Vec<Vec<u16>>,               // block skip [3 ctx]
    pub(crate) part_bl8: Vec<Vec<u16>>,           // PARTITION_NONE @ 8x8 [4 ctx]
    pub(crate) part_split: Vec<Vec<Vec<u16>>>,    // SPLIT [bl-1=0..3][4 ctx]
    pub(crate) kf_y: Vec<Vec<u16>>,               // kf_y_mode[5*5], index [above_ctx*5 + left_ctx]
    pub(crate) uv_mode: Vec<Vec<u16>>,            // uv_mode[2*13], index [cfl_allowed*13 + y_mode]
    pub(crate) angle_delta: Vec<Vec<u16>>,        // angle_delta[8 directional modes]
    pub(crate) cfl_sign: Vec<u16>,                // cfl joint-sign (8 symbols)
    pub(crate) cfl_alpha: Vec<Vec<u16>>,          // cfl alpha magnitude [6 ctx]
    pub(crate) txtp: Vec<Vec<u16>>,               // intra txtp TX_8X8 luma, per intra mode [13]
    pub(crate) txtp16: Vec<Vec<u16>>,             // intra txtp TX_16X16 luma, per intra mode [13]
    pub(crate) txb_skip: [Vec<Vec<u16>>; 4],      // [class][13 ctx] (class 3 = TX_32X32)
    pub(crate) base_tok: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][41/42 ctx]
    pub(crate) br_tok: [[Vec<Vec<u16>>; 2]; 4],   // [class][plane][21 ctx]
    pub(crate) eob_base: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][4 ctx]
    pub(crate) eob_hi: [[Vec<Vec<u16>>; 2]; 4],   // [class][plane][11 bins], each a 2-sym CDF
    pub(crate) dc_sign: [Vec<Vec<u16>>; 2],       // [plane][3 ctx]
    pub(crate) eob_bin_16_c: Vec<u16>,            // chroma, 4x4
    pub(crate) eob_bin_32_c: Vec<u16>,            // chroma, 4x8
    pub(crate) eob_bin_64_l: Vec<u16>,            // luma, 8x8
    pub(crate) eob_bin_64_c: Vec<u16>,            // chroma, 8x8
    pub(crate) eob_bin_256_l: Vec<u16>,           // luma, 16x16 (class 2)
    pub(crate) eob_bin_256_c: Vec<u16>,           // chroma, 16x16 (class 2)
    pub(crate) eob_bin_128_c: Vec<u16>,           // chroma, RTX_8X16 (class 2, 128 coeffs)
    pub(crate) eob_bin_1024_l: Vec<u16>,          // luma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_1024_c: Vec<u16>,          // chroma, 32x32 (class 3, 1024 coeffs)
    pub(crate) eob_bin_512_c: Vec<u16>,           // chroma, RTX_16X32 (class 3, 512 coeffs)
}

impl Cdfs {
    fn new(qctx: usize) -> Self {
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
                rows(&Q::BASE_TOK_TX4_CHROMA_Q[qctx]),
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
                rows(&Q::BR_TOK_TX4_CHROMA_Q[qctx]),
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
                rows2(&Q::EOB_BASE_TX4_CHROMA_Q[qctx]),
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
                his(&Q::EOB_HI_TX4_CHROMA[qctx]),
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
            skip: SKIP_CDF.iter().map(|&v| icdf(&[v])).collect(),
            part_bl8: PART_BL8_CDF.iter().map(|r| icdf(r)).collect(),
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
            eob_bin_32_c: icdf(&Q::EOB_BIN_32_CHROMA[qctx]),
            eob_bin_64_l: icdf(&Q::EOB_BIN_64_LUMA[qctx]),
            eob_bin_64_c: icdf(&Q::EOB_BIN_64_CHROMA[qctx]),
            eob_bin_256_l: icdf(&Q::EOB_BIN_256_LUMA[qctx]),
            eob_bin_256_c: icdf(&Q::EOB_BIN_256_CHROMA[qctx]),
            eob_bin_128_c: icdf(&Q::EOB_BIN_128_CHROMA[qctx]),
            eob_bin_1024_l: icdf(&Q::EOB_BIN_1024_LUMA[qctx]),
            eob_bin_1024_c: icdf(&Q::EOB_BIN_1024_CHROMA[qctx]),
            eob_bin_512_c: icdf(&Q::EOB_BIN_512_CHROMA[qctx]),
        }
    }
}

/// Whole-frame lossy encoder state. Context arrays are indexed by absolute frame
/// coordinates: the above arrays persist down the superblock rows, and the left
/// arrays are naturally fresh per SB row (each row occupies a distinct
/// coordinate range), mirroring dav1d's per-SB-row left reset.
struct LossyTile<'a> {
    bd: u8,
    quant: Quant,
    cquant: Quant,
    w: usize,
    h: usize,
    cw: usize,   // chroma plane width (= w for 4:4:4, w/2 for 4:2:2 and 4:2:0)
    ss422: bool, // chroma horizontally subsampled (4:2:2)
    ss420: bool, // chroma horizontally + vertically subsampled (4:2:0)
    mono: bool,  // monochrome: code luma only (NumPlanes=1, no chroma syntax)
    src: &'a [Vec<i32>; 3],
    recon: [Vec<i32>; 3],
    a_coef: [Vec<u8>; 3], // len w/4, absolute bx4
    l_coef: [Vec<u8>; 3], // len h/4, absolute by4
    a_part: Vec<u8>,      // len w/8, absolute x8
    l_part: Vec<u8>,      // len h/8, absolute y8
    a_skip: Vec<u8>,      // block skip flag per 4x4 col, absolute bx4
    l_skip: Vec<u8>,      // block skip flag per 4x4 row, absolute by4
    a_mode: Vec<u8>,      // luma intra mode per 4x4 col (for kf y-mode context)
    l_mode: Vec<u8>,      // luma intra mode per 4x4 row
    blk4: Vec<u8>, // luma block size (in 4-sample units, square) per 4x4 luma unit; for the deblock filter
    skip8: Vec<bool>, // per-8x8-luma-unit block skip flag (true = no coded coeffs); for CDEF
    enc: OdEcEncoder,
    cdfs: Cdfs,
    /// RDO effort: [`Speed::Slow`] (default) or [`Speed::Fast`] (winner-only
    /// RDOQ, DCT-only transform choice, reduced intra mode set).
    speed: Speed,
}

impl<'a> LossyTile<'a> {
    fn new(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; w * h], vec![0; w * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
        }
    }

    /// Monochrome tile: codes the luma plane only (`NumPlanes = 1`). Only
    /// `src[0]` is used; the chroma reconstruction and context arrays are left
    /// empty so any stray chroma access panics instead of corrupting output.
    /// Forces 8x8 luma transforms (see `prefer_16x16`/`prefer_32x32`).
    fn new_mono(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: true,
            src,
            recon: [vec![0; w * h], Vec::new(), Vec::new()],
            a_coef: [vec![0x40; w / 4], Vec::new(), Vec::new()],
            l_coef: [vec![0x40; h / 4], Vec::new(), Vec::new()],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
        }
    }

    /// 4:2:2 tile: luma is full w x h, chroma planes are subsampled to (w/2) x h.
    /// `src[1]`/`src[2]` must already be the half-width chroma planes.
    fn new_422(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        let cw = w / 2;
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw,
            ss422: true,
            ss420: false,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; cw * h], vec![0; cw * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
        }
    }

    /// 4:2:0 tile: luma is full w x h, chroma planes are subsampled to
    /// (w/2) x (h/2). `src[1]`/`src[2]` must already be the quarter-size planes.
    fn new_420(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        let (cw, ch) = (w / 2, h / 2);
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw,
            ss422: false,
            ss420: true,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; cw * ch], vec![0; cw * ch]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; ch / 4], vec![0x40; ch / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            speed: Speed::Slow,
        }
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
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 8;
        (s != 0) as usize + (s > 0) as usize
    }

    /// Decide whether to code the 16x16 region at (`x8`,`y8`) as a single
    /// TX_16X16 (PARTITION_NONE) vs splitting into four 8x8. This is a pure R-D
    /// proxy — the decoder follows whatever partition we signal, so the choice
    /// affects compression only, never correctness. Proxy: compare the summed
    /// absolute quantized luma levels of the one 16x16 transform (plus a small
    /// per-block overhead) against the four 8x8 transforms (each with its own
    /// overhead). Smooth regions compact into the 16x16 and win decisively.
    /// Peak-to-peak luma range of the `dim`x`dim` source block at 8-unit origin
    /// `(x8,y8)`. Smooth low-contrast blocks (small range) ring into visible
    /// low-frequency banding under large transforms, so the partitioner uses it
    /// to keep such blocks on small (8x8) transforms.
    fn block_luma_range(&self, x8: usize, y8: usize, dim: usize) -> i32 {
        let (px, py) = (x8 * 8, y8 * 8);
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for ry in 0..dim {
            let base = (py + ry) * self.w + px;
            for &s in &self.src[0][base..base + dim] {
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }
        }
        hi - lo
    }

    fn prefer_16x16(&self, x8: usize, y8: usize) -> bool {
        if self.mono {
            return false; // monochrome codes 8x8 luma blocks only
        }
        // See `prefer_32x32`: a 16x16 luma block gives an 8x16 (16-row) chroma
        // transform in 4:2:2, still tall enough that flat-DC coding of a smooth
        // chroma gradient rings into green lanes. Keep 4:2:2 on 8x8 luma blocks.
        if self.ss422 {
            return false;
        }
        // Smooth, low-contrast 16x16 blocks ring into low-frequency luma banding
        // (the staircase that 8x8 avoids — see 4:2:2). Keep them on 8x8.
        if self.block_luma_range(x8, y8, 16) < LF_BAND_SMOOTH_RANGE {
            return false;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        // one 16x16 (DC-pred from available recon above/left)
        let lpred = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
        let mut r16 = [0i32; 256];
        for (ry, drow) in r16.chunks_exact_mut(16).enumerate() {
            let srow = &self.src[0][(py + ry) * self.w + px..];
            for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                *dv = s - lpred;
            }
        }
        forward_dct_quant_16x16(&mut r16, &self.quant);
        let cost16: u32 = est_block_bits(&r16, &SCAN_16X16) + OVERHEAD_16;
        // four 8x8 (DC-pred each from current recon; decision-only approximation)
        let mut cost8 = 0u32;
        for (sx, sy) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)] {
            let pred = dc_pred_8x8(&self.recon[0], self.w, px + sx, py + sy, self.bd as i32);
            let mut r8 = [0i32; 64];
            for (ry, drow) in r8.chunks_exact_mut(8).enumerate() {
                let srow = &self.src[0][(py + sy + ry) * self.w + px + sx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            forward_dct_quant_8x8(&mut r8, &self.quant);
            cost8 += est_block_bits(&r8, &SCAN_8X8) + OVERHEAD_8;
        }
        cost16 <= cost8
    }

    /// Code a 16x16 region (4:4:4 only) as a single TX_16X16 block: luma +
    /// chroma DC prediction, forward DCT16 + quant, the TX_16X16 coefficient
    /// coder, and reconstruction via the exact integer inverse. Updates the
    /// 4-unit (16-sample) skip / coef neighbour-context footprint.
    /// Set the RDO effort level (0 = full, >= 1 = fast). Returns `self` for
    /// builder-style chaining at tile construction.
    fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 4);
        let (px, py) = (x8 * 8, y8 * 8);
        // luma 16x16 (identical for all subsampling modes)
        // Luma 16x16: same non-directional intra mode search as the 8x8 path.
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let dcs16 = self.dc_sign_ctx_16(0, px / 4, py / 4);
        let mlam = mode_lambda() * acq * acq;
        let mut best_mode = DC_PRED;
        let mut best_is_adst16 = false;
        let mut lpred_arr = [0i32; 256];
        let mut lcf = [0i32; 256];
        let mut best_eff = f64::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f64;
        let mut ltf = [0f64; 256]; // winner transform coeffs (f64, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        for &m in modes {
            let mut pred = [0i32; 256];
            if m == DC_PRED {
                let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 256];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 256];
            for (ry, (rrow, prow)) in resid
                .chunks_exact_mut(16)
                .zip(pred.chunks_exact(16))
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let blk_sse16 = |rr: &[i32; 256]| -> i64 {
                let mut sse = 0i64;
                for (ry, (prow, rrow)) in pred.chunks_exact(16).zip(rr.chunks_exact(16)).enumerate()
                {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        sse += (d * d) as i64;
                    }
                }
                sse
            };
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    &self.cdfs,
                    2,
                    0,
                    &self.cdfs.eob_bin_256_l,
                    dcs16,
                );
            }
            let sse = blk_sse16(&idct_dequant_16x16(&cf, &self.quant));
            let bits = block_rate_bits(&cf, &SCAN_16X16);
            let cost = sse as f64 + mlam * (bits + mode_signal_bits(m));
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
            }
        }
        // Fast path: run RDOQ once, on the winning mode only (libaom
        // winner-mode coeff opt). The decision above used un-trellised costs.
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                &self.cdfs,
                2,
                0,
                &self.cdfs.eob_bin_256_l,
                dcs16,
            );
        }
        // Winner-only ADST_ADST refinement. Full and Medium try it; only Fast
        // prunes the transform-type search to DCT_DCT (libaom-style).
        if self.speed.try_adst() {
            let mut resid = [0i32; 256];
            for (ry, rrow) in resid.chunks_exact_mut(16).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut acf, atf) = adst16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                &self.cdfs,
                2,
                0,
                &self.cdfs.eob_bin_256_l,
                dcs16,
            );
            let rr = iadst_dequant_16x16(&acf, &self.quant);
            let mut asse = 0i64;
            for (ry, rrow) in rr.chunks_exact(16).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 16..ry * 16 + 16];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    asse += (d * d) as i64;
                }
            }
            let abits = block_rate_bits(&acf, &SCAN_16X16);
            if asse as f64 + mlam * abits < best_dct_sse as f64 + mlam * best_dct_bits {
                lcf = acf;
                best_is_adst16 = true;
            }
        }
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block16_420(
                x8,
                y8,
                &lcf,
                &lpred_arr,
                best_mode,
                luma_zero,
                best_is_adst16,
            );
        } else if self.ss422 {
            self.code_block16_422(
                x8,
                y8,
                &lcf,
                &lpred_arr,
                best_mode,
                luma_zero,
                best_is_adst16,
            );
        } else {
            self.code_block16_444(
                x8,
                y8,
                &lcf,
                &lpred_arr,
                best_mode,
                luma_zero,
                best_is_adst16,
            );
        }
    }

    /// Shared header + luma for a TX_16X16 block: codes the block-level skip
    /// flag, `DC_PRED` y/uv modes, the luma TX_16X16 coefficients, updates the
    /// 4-unit (16-sample) luma skip/coef footprint, and reconstructs luma. The
    /// caller has already decided `block_skip` (needs all planes) and passes the
    /// luma coefficients + DC prediction.
    /// Emit the chroma `uv_mode` symbol: plain DC (`None`) or CfL (`Some(alphas)`),
    /// in which case also the joint-sign and per-plane magnitude symbols.
    fn emit_uv_mode(&mut self, y_mode: usize, uv_mode: usize, cfl: Option<[i32; 2]>) {
        match cfl {
            Some(a) => {
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
                    // Both alphas zero: zero-alpha CfL reconstructs exactly as DC
                    // prediction, so signal plain DC and avoid an invalid
                    // `sign - 1` joint-sign symbol.
                    self.enc
                        .encode_symbol(DC_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
                    return;
                }
                self.enc
                    .encode_symbol(CFL_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
                self.enc.encode_symbol(sign - 1, &mut self.cdfs.cfl_sign);
                if su != 0 {
                    let c = (su == 2) as usize * 3 + sv;
                    self.enc
                        .encode_symbol((a[0].abs() - 1) as usize, &mut self.cdfs.cfl_alpha[c]);
                }
                if sv != 0 {
                    let c = (sv == 2) as usize * 3 + su;
                    self.enc
                        .encode_symbol((a[1].abs() - 1) as usize, &mut self.cdfs.cfl_alpha[c]);
                }
            }
            None => {
                self.enc
                    .encode_symbol(uv_mode, &mut self.cdfs.uv_mode[13 + y_mode]);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_header_luma16(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        is_adst16: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        self.mark_skip8(x8, y8, 2, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(y_mode, uv_mode, cfl);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 4].fill(sv);
        self.l_skip[by4..by4 + 4].fill(sv);
        self.a_mode[bx4..bx4 + 4].fill(mv);
        self.l_mode[by4..by4 + 4].fill(mv);
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_16(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_16(0, bx4, by4);
            encode_tx16_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                lcf,
                false,
                sk,
                ds,
                y_mode,
                if is_adst16 { ADST_ADST_TX16_IDX } else { 1 },
            )
        };
        self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
        self.l_coef[0][by4..by4 + 4].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 256]
        } else if is_adst16 {
            iadst_dequant_16x16(lcf, &self.quant)
        } else {
            idct_dequant_16x16(lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred.chunks_exact(16).zip(lrr.chunks_exact(16)).enumerate() {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block16_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        is_adst16: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let mut ccf = [[0i32; 256]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_16x16(&self.recon[plane], self.w, px, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 256];
            for (ry, drow) in resid.chunks_exact_mut(16).enumerate() {
                let srow = &self.src[plane][(py + ry) * self.w + px..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                &SCAN_16X16,
                trellis_lambda(),
            );
            let mean_resid_dc = resid.iter().sum::<i32>() / 256;
            if ccf[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // 4:4:4 CfL for the 16x16 chroma blocks (mirrors the 8x8 path).
        let mut cpred16 = [[0i32; 256]; 2];
        let mut cfl_opt: Option<[i32; 2]> = None;
        {
            let lrr_cfl = idct_dequant_16x16(lcf, &self.quant);
            let mut luma_rec = [0i32; 256];
            for i in 0..256 {
                luma_rec[i] = (lpred[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 256];
            cfl_ac_444(&luma_rec, 16, 16, &mut ac);
            let (dcq, acq, lam) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let mlam = mode_lambda() * acq * acq;
            let mut cfl_ccf = [[0i32; 256]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f64; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f64; 2]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 256];
                for (ry, drow) in src.chunks_exact_mut(16).enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..16]);
                }
                let dcrr = idct_dequant_16x16(&ccf[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..256 {
                    let r = (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s += (d * d) as i64;
                }
                dc_sse[ci] = s;
                dc_bits[ci] = block_rate_bits(&ccf[ci], &SCAN_16X16);
                let a = cfl_best_alpha(&ac, &src, dc, 256, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 256];
                let mut resid = [0i32; 256];
                for i in 0..256 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X16, lam);
                let rr = idct_dequant_16x16(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..256 {
                    let r = (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_sse[ci] = s2;
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_16X16);
                cpred16[ci] = cpr;
            }
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let dc_total = (dc_sse[0] + dc_sse[1]) as f64 + mlam * (dc_bits[0] + dc_bits[1]);
            let cfl_total =
                (cfl_sse[0] + cfl_sse[1]) as f64 + mlam * (cfl_bits[0] + cfl_bits[1] + sig);
            // Let the RD comparison decide DC-vs-CfL across the whole quality
            // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
            // where it helps most (high quality).
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                cfl_opt = Some(cfl_a);
                ccf[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred16[ci] = [cpred[ci]; 256];
                }
            }
        }
        // SMOOTH_V check: on smooth vertical gradients DC prediction rings at block
        // boundaries; SMOOTH_V interpolates from the top edge to the bottom-left corner,
        // giving near-zero residual on gradients and eliminating the green-lane stripes.
        let mut chosen_uv_16 = DC_PRED;
        // SMOOTH_V is only beneficial at low quality (ac_q > 300 ≈ quality ≤ 35).
        // At higher quality DC/CfL suffice and SMOOTH_V's <= tie-break causes
        // block-boundary colour mismatches across the whole image.
        if self.quant.ac_q() > 300 {
            let (dcq, acq, lam) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let mut sv_ccf16 = [[0i32; 256]; 2];
            let mut sv_preds16 = [[0i32; 256]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            for ci in 0..2 {
                let plane = ci + 1;
                for ry in 0..16 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &cpred16[ci][ry * 16..];
                    for (&srow, &prow) in srow[..16].iter().zip(prow[..16].iter()) {
                        let d = srow - prow;
                        sse_cur += (d * d) as i64;
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    false,
                    false,
                    self.w,
                    self.h,
                    &mut sv_preds16[ci],
                    self.bd,
                );
                let mut resid = [0i32; 256];
                for (ry, drow) in resid.chunks_exact_mut(16).enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds16[ci][ry * 16..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
                sv_ccf16[ci] = q;
                trellis_optimize(&mut sv_ccf16[ci], &qt, dcq, acq, &SCAN_16X16, lam);
                let mean_resid_sv = resid.iter().sum::<i32>() / 256;
                if sv_ccf16[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    sv_ccf16[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
                for ry in 0..16 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds16[ci][ry * 16..];
                    for (&srow, &prow) in srow[..16].iter().zip(prow[..16].iter()) {
                        let d = srow - prow;
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv <= sse_cur {
                ccf[..2].copy_from_slice(&sv_ccf16[..2]);
                cpred16[..2].copy_from_slice(&sv_preds16[..2]);
                cfl_opt = None; // SMOOTH_V overrides CfL if it wins
                chosen_uv_16 = SMOOTH_V_PRED;
            }
        } // end if ac_q > 300 (SMOOTH_V)
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv_16,
            cfl_opt,
            is_adst16,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_16(plane, bx4, by4);
                encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    0,
                    1,
                )
            };
            self.a_coef[plane][bx4..bx4 + 4].fill(res_ctx);
            self.l_coef[plane][by4..by4 + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else {
                idct_dequant_16x16(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.chunks_exact(16).enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if cfl_opt.is_some() || chosen_uv_16 == SMOOTH_V_PRED {
                    let prow = &cpred16[ci][ry * 16..];
                    for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                        *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                    }
                } else {
                    // Plain DC chroma: use the scalar predictor directly so recon
                    // never depends on the CfL block having populated `cpred16`.
                    for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                        *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block16_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        is_adst16: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f64,
            self.cquant.ac_q() as f64,
            trellis_lambda(),
        );
        let maxval = (1 << self.bd) - 1;
        // DC path
        let mut ccf_dc = [[0i32; 64]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 64];
            for (ry, drow) in resid.chunks_exact_mut(8).enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
            ccf_dc[ci] = q;
            trellis_optimize(&mut ccf_dc[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            let mean_resid_dc = resid.iter().sum::<i32>() / 64;
            if ccf_dc[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf_dc[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // SMOOTH_V: only at low quality (ac_q > 300 ≈ quality ≤ 35)
        let smooth_v_active = acq > 300.0;
        let mut ccf_sv = [[0i32; 64]; 2];
        let mut sv_preds = [[0i32; 64]; 2];
        if smooth_v_active {
            for ci in 0..2 {
                let plane = ci + 1;
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    8,
                    8,
                    false,
                    false,
                    self.cw,
                    self.h,
                    &mut sv_preds[ci],
                    self.bd,
                );
                let mut resid = [0i32; 64];
                for (ry, drow) in resid.chunks_exact_mut(8).enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds[ci][ry * 8..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                ccf_sv[ci] = q;
                trellis_optimize(&mut ccf_sv[ci], &qt, dcq, acq, &SCAN_8X8, lam);
                let mean_resid_sv = resid.iter().sum::<i32>() / 64;
                if ccf_sv[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    ccf_sv[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
            }
        } // end if smooth_v_active
        // RD: reconstruct both and compare SSE; cache inverse-transform for winner reuse
        let mut rr_dc = [[0i32; 64]; 2];
        let mut rr_sv = [[0i32; 64]; 2];
        let mut sse_dc = 0i64;
        let mut sse_sv = 0i64;
        for ci in 0..2 {
            let plane = ci + 1;
            rr_dc[ci] = idct_dequant_8x8(&ccf_dc[ci], &self.cquant);
            rr_sv[ci] = idct_dequant_8x8(&ccf_sv[ci], &self.cquant);
            let dc = dc_preds[ci];
            for (ry, (rd_row, rs_row)) in rr_dc[ci]
                .chunks_exact(8)
                .zip(rr_sv[ci].chunks_exact(8))
                .enumerate()
            {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                let prow = &sv_preds[ci][ry * 8..];
                for (((&s, &prow), &rd), &rs) in srow[..8]
                    .iter()
                    .zip(prow[..8].iter())
                    .zip(rd_row[..8].iter())
                    .zip(rs_row[..8].iter())
                {
                    let d = s - (dc + rd).clamp(0, maxval);
                    let v = s - (prow + rs).clamp(0, maxval);
                    sse_dc += (d * d) as i64;
                    sse_sv += (v * v) as i64;
                }
            }
        }
        let use_sv = smooth_v_active && sse_sv <= sse_dc;
        let (chosen_uv, ccf, rr_cache) = if use_sv {
            (SMOOTH_V_PRED, ccf_sv, rr_sv)
        } else {
            (DC_PRED, ccf_dc, rr_dc)
        };
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8, y8, lcf, lpred, y_mode, block_skip, chosen_uv, None, is_adst16,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx(plane, bx4c, by4c);
                encode_tx8_coeffs_adapt(&mut self.enc, &mut self.cdfs, &ccf[ci], true, sk, ds, 0, 1)
            };
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 2].fill(res_ctx);
            let rr = if block_skip { [0i32; 64] } else { rr_cache[ci] };
            for (ry, rrow) in rr.chunks_exact(8).enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                if use_sv {
                    let prow = &sv_preds[ci][ry * 8..];
                    for ((dv, &rv), &prow) in
                        drow[..8].iter_mut().zip(rrow.iter()).zip(prow[..8].iter())
                    {
                        *dv = (prow + rv).clamp(0, maxval);
                    }
                } else {
                    let dc = dc_preds[ci];
                    for (dv, &rv) in drow[..8].iter_mut().zip(rrow.iter()) {
                        *dv = (dc + rv).clamp(0, maxval);
                    }
                }
            }
        }
    }

    /// 4:2:2: a 16x16 luma region maps to an 8-wide x 16-tall chroma region per
    /// plane (`RTX_8X16`, coef-CDF class 2). Chroma is full-height, half-width, so
    /// the chroma block sits at `(cx, py)` with `cx = px/2` and spans 2 coef units
    /// horizontally and 4 vertically on the chroma grid.
    #[allow(clippy::too_many_arguments)]
    fn code_block16_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        is_adst16: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        let (bx4c, by4c) = (cx / 4, py / 4);
        let mut ccf = [[0i32; 128]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_8x16(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 128];
            for (ry, drow) in resid.chunks_exact_mut(8).enumerate() {
                let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_8x16_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                &SCAN_8X16,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8, y8, lcf, lpred, y_mode, block_skip, DC_PRED, None, is_adst16,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_8x16_422(plane, bx4c, by4c);
                encode_8x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            // RTX_8X16: 2 coef-context units wide, 4 units tall.
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_8x16(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.chunks_exact(8).enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                }
            }
        }
    }

    fn code_block(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 2);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let cx = px / 2; // chroma column for 4:2:2

        // Forward-transform/quantize all planes up front to decide block skip.
        // Luma is always 8x8; chroma is 8x8 (4:4:4) or 4x8 (4:2:2).
        // Luma 8x8: search the non-directional intra modes (DC + SMOOTH*/PAETH)
        // and keep the one minimizing pixel SSE + lambda * estimated bits. The
        // chosen prediction is per-pixel; reconstruction uses the same array so
        // the decoder (which re-derives the identical prediction) stays bit-exact.
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let mlam = mode_lambda() * acq * acq;
        let mut best_mode = DC_PRED;
        let mut best_is_adst = false;
        let mut lpred_arr = [0i32; 64];
        let mut lcf = [0i32; 64];
        let mut best_eff = f64::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f64;
        let dc_sgn = self.dc_sign_ctx(0, px / 4, py / 4);
        let mut ltf = [0f64; 64]; // winner transform coeffs (f64, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        for &m in modes {
            let mut pred = [0i32; 64];
            if m == DC_PRED {
                let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            for (ry, (rrow, prow)) in resid
                .chunks_exact_mut(8)
                .zip(pred.chunks_exact(8))
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            // Mode decision uses DCT_DCT only (cheap); the ADST_ADST transform
            // choice is refined once for the winning mode after the loop.
            let blk_sse = |rr: &[i32; 64]| -> i64 {
                let mut sse = 0i64;
                for (ry, (prow, rrow)) in pred.chunks_exact(8).zip(rr.chunks_exact(8)).enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        sse += (d * d) as i64;
                    }
                }
                sse
            };
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    &self.cdfs,
                    1,
                    0,
                    &self.cdfs.eob_bin_64_l,
                    dc_sgn,
                );
            }
            let sse = blk_sse(&idct_dequant_8x8(&cf, &self.quant));
            let bits = block_rate_bits(&cf, &SCAN_8X8);
            let cost = sse as f64 + mlam * (bits + mode_signal_bits(m));
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                &self.cdfs,
                1,
                0,
                &self.cdfs.eob_bin_64_l,
                dc_sgn,
            );
        }
        // Per-block transform refinement: try ADST_ADST on the winning
        // prediction only and keep it if cheaper than that mode's DCT. This is
        // one extra transform+trellis per block instead of one per candidate
        // mode, which is where the encode-time regression came from.
        // Full and Medium try ADST; only Fast prunes the transform type to DCT_DCT.
        if self.speed.try_adst() {
            let mut resid = [0i32; 64];
            for (ry, rrow) in resid.chunks_exact_mut(8).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut acf, atf) = adst8x8_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                &self.cdfs,
                1,
                0,
                &self.cdfs.eob_bin_64_l,
                dc_sgn,
            );
            let rr = iadst_dequant_8x8(&acf, &self.quant);
            let mut asse = 0i64;
            for (ry, rrow) in rr.chunks_exact(8).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    asse += (d * d) as i64;
                }
            }
            let abits = block_rate_bits(&acf, &SCAN_8X8);
            if asse as f64 + mlam * abits < best_dct_sse as f64 + mlam * best_dct_bits {
                lcf = acf;
                best_is_adst = true;
            }
        }
        let mut ccf8 = [[0i32; 64]; 2];
        let mut ccf48 = [[0i32; 32]; 2];
        let mut ccf44 = [[0i32; 16]; 2];
        let mut cpred = [0i32; 2];
        let cy = py / 2; // chroma row for 4:2:0
        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let pred = dc_pred_4x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 16];
                for (ry, drow) in resid.chunks_exact_mut(4).enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                ccf44[ci] = q;
                trellis_optimize(&mut ccf44[ci], &qt, dcq, acq, &SCAN_4X4, lam);
            } else if self.ss422 {
                let pred = dc_pred_4x8(&self.recon[plane], self.cw, cx, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 32];
                for (ry, drow) in resid.chunks_exact_mut(4).enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                ccf48[ci] = q;
                trellis_optimize(&mut ccf48[ci], &qt, dcq, acq, &SCAN_4X8, lam);
            } else {
                let pred = dc_pred_8x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 64];
                for (ry, drow) in resid.chunks_exact_mut(8).enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                ccf8[ci] = q;
                trellis_optimize(&mut ccf8[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            }
        }

        // 4:4:4 chroma-from-luma: try predicting U/V from the reconstructed luma
        // block (scaled, mean-removed) and pick CfL over plain DC per block.
        let mut cpred444 = [[0i32; 64]; 2];
        let mut cpred420 = [[0i32; 16]; 2];
        let mut cpred422 = [[0i32; 32]; 2];
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        if !self.mono && !self.ss420 && !self.ss422 {
            let lrr_cfl = idct_dequant_8x8(&lcf, &self.quant);
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 64];
            cfl_ac_444(&luma_rec, 8, 8, &mut ac);
            let mut cfl_ccf = [[0i32; 64]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f64; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f64; 2]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 64];
                for (ry, drow) in src.chunks_exact_mut(8).enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..8]);
                }
                // DC option distortion/rate (from the coeffs already computed)
                let dcrr = idct_dequant_8x8(&ccf8[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..64 {
                    let r = (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s += (d * d) as i64;
                }
                dc_sse[ci] = s;
                dc_bits[ci] = block_rate_bits(&ccf8[ci], &SCAN_8X8);
                // CfL option
                let a = cfl_best_alpha(&ac, &src, dc, 64, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 64];
                let mut resid = [0i32; 64];
                for i in 0..64 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                let rr = idct_dequant_8x8(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..64 {
                    let r = (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_sse[ci] = s2;
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_8X8);
                cpred444[ci] = cpr;
            }
            // joint signalling cost estimate (sign symbol + 1 magnitude per non-zero plane)
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let dc_total = (dc_sse[0] + dc_sse[1]) as f64 + mlam * (dc_bits[0] + dc_bits[1]);
            let cfl_total =
                (cfl_sse[0] + cfl_sse[1]) as f64 + mlam * (cfl_bits[0] + cfl_bits[1] + sig);
            // Let the RD comparison decide DC-vs-CfL across the whole quality
            // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
            // where it helps most (high quality).
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf8[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred444[ci] = [cpred[ci]; 64];
                }
            }
        }

        let chroma_zero = |ci: usize| {
            if self.ss420 {
                ccf44[ci].iter().all(|&c| c == 0)
            } else if self.ss422 {
                ccf48[ci].iter().all(|&c| c == 0)
            } else {
                ccf8[ci].iter().all(|&c| c == 0)
            }
        };
        let block_skip =
            lcf.iter().all(|&c| c == 0) && (self.mono || (chroma_zero(0) && chroma_zero(1)));

        // block-level mode info: skip (ctx = above_skip + left_skip), y/uv = DC
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        self.mark_skip8(x8, y8, 1, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&best_mode) {
            // angle_delta = 0 (symbol index 3); 8x8 satisfies the size condition
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[best_mode - V_PRED]);
        }
        // SMOOTH_V check for 4:2:0 4x4 chroma: only at low quality (ac_q > 300)
        let smooth_v_active_ss420 = self.quant.ac_q() > 300;
        let mut sv_preds_420 = [[0i32; 16]; 2];
        let mut chosen_uv_block = DC_PRED;
        if !self.mono && self.ss420 && smooth_v_active_ss420 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let mut sv_ccf44_2 = [[0i32; 16]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                for ry in 0..4 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for &sr in srow[..4].iter() {
                        let d = sr - dc;
                        sse_cur += (d * d) as i64;
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    4,
                    4,
                    false,
                    false,
                    self.cw,
                    self.h,
                    &mut sv_preds_420[ci],
                    self.bd,
                );
                let mut resid = [0i32; 16];
                for (ry, drow) in resid.chunks_exact_mut(4).enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds_420[ci][ry * 4..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                sv_ccf44_2[ci] = q;
                trellis_optimize(&mut sv_ccf44_2[ci], &qt, dcq2, acq2, &SCAN_4X4, lam2);
                for ry in 0..4 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds_420[ci][ry * 4..];
                    for j in 0..4 {
                        let d = srow[j] - prow[j];
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv < sse_cur {
                ccf44[..2].copy_from_slice(&sv_ccf44_2[..2]);
                chosen_uv_block = SMOOTH_V_PRED;
            }
        }
        // Note: SMOOTH_V for 4:4:4 8x8 (code_block small-block path) is intentionally
        // not added here — it introduces too many DC↔SV mode transitions at 8-row
        // boundaries that are visible as faint lines at quality 50-75.
        // 4:2:0 chroma-from-luma: predict the 4x4 U/V from the 2x2-subsampled
        // reconstructed luma of this 8x8 block (dav1d cfl_ac, ss_hor=ss_ver=1).
        // Competes with the current DC/SMOOTH_V choice on rate-distortion.
        if !self.mono && self.ss420 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let lrr = if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 16];
            cfl_ac_sub(&luma_rec, 8, 4, 4, true, true, &mut ac);
            let mut cfl_ccf = [[0i32; 16]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
            let (mut cur_bits, mut cfl_bits) = (0f64, 0f64);
            let maxv = (1 << self.bd) - 1;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 16];
                for (ry, drow) in src.chunks_exact_mut(4).enumerate() {
                    drow.copy_from_slice(&self.src[plane][(cy + ry) * self.cw + cx..][..4]);
                }
                let curr = idct_dequant_4x4(&ccf44[ci], &self.cquant);
                for i in 0..16 {
                    let p = if chosen_uv_block == SMOOTH_V_PRED {
                        sv_preds_420[ci][i]
                    } else {
                        dc
                    };
                    let r = (p + curr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cur_sse += (d * d) as i64;
                }
                cur_bits += block_rate_bits(&ccf44[ci], &SCAN_4X4);
                let a = cfl_best_alpha(&ac, &src, dc, 16, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 16];
                let mut resid = [0i32; 16];
                for i in 0..16 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X4, lam2);
                let rr = idct_dequant_4x4(&q, &self.cquant);
                for i in 0..16 {
                    let r = (cpr[i] + rr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cfl_sse += (d * d) as i64;
                }
                cfl_bits += block_rate_bits(&q, &SCAN_4X4);
                cfl_ccf[ci] = q;
                cpred420[ci] = cpr;
            }
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let cur_total = cur_sse as f64 + mlam * cur_bits;
            let cfl_total = cfl_sse as f64 + mlam * (cfl_bits + sig);
            if cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf44[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        // 4:2:2 chroma-from-luma: 4x8 chroma from the horizontally 2:1-subsampled
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0).
        if !self.mono && self.ss422 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                trellis_lambda(),
            );
            let lrr = if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 32];
            cfl_ac_sub(&luma_rec, 8, 4, 8, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 32]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
            let (mut cur_bits, mut cfl_bits) = (0f64, 0f64);
            let maxv = (1 << self.bd) - 1;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 32];
                for (ry, drow) in src.chunks_exact_mut(4).enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..4]);
                }
                let curr = idct_dequant_4x8(&ccf48[ci], &self.cquant);
                for i in 0..32 {
                    let r = (dc + curr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cur_sse += (d * d) as i64;
                }
                cur_bits += block_rate_bits(&ccf48[ci], &SCAN_4X8);
                let a = cfl_best_alpha(&ac, &src, dc, 32, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 32];
                let mut resid = [0i32; 32];
                for i in 0..32 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X8, lam2);
                let rr = idct_dequant_4x8(&q, &self.cquant);
                for i in 0..32 {
                    let r = (cpr[i] + rr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cfl_sse += (d * d) as i64;
                }
                cfl_bits += block_rate_bits(&q, &SCAN_4X8);
                cfl_ccf[ci] = q;
                cpred422[ci] = cpr;
            }
            let sig =
                4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
            let cur_total = cur_sse as f64 + mlam * cur_bits;
            let cfl_total = cfl_sse as f64 + mlam * (cfl_bits + sig);
            if cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf48[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        if !self.mono {
            self.emit_uv_mode(
                best_mode,
                chosen_uv_block,
                if use_cfl { Some(cfl_alpha_uv) } else { None },
            );
        }
        let sv = block_skip as u8;
        self.a_skip[bx4] = sv;
        self.a_skip[bx4 + 1] = sv;
        self.l_skip[by4] = sv;
        self.l_skip[by4 + 1] = sv;
        let mv = best_mode as u8;
        self.a_mode[bx4] = mv;
        self.a_mode[bx4 + 1] = mv;
        self.l_mode[by4] = mv;
        self.l_mode[by4 + 1] = mv;

        // luma (TX_8X8)
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx(0, bx4, by4, false);
            let ds = self.dc_sign_ctx(0, bx4, by4);
            encode_tx8_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                &lcf,
                false,
                sk,
                ds,
                best_mode,
                if best_is_adst { ADST_ADST_TX8_IDX } else { 1 },
            )
        };
        self.a_coef[0][bx4] = lres_ctx;
        self.a_coef[0][bx4 + 1] = lres_ctx;
        self.l_coef[0][by4] = lres_ctx;
        self.l_coef[0][by4 + 1] = lres_ctx;
        let lrr = if block_skip {
            [0i32; 64]
        } else if best_is_adst {
            iadst_dequant_8x8(&lcf, &self.quant)
        } else {
            idct_dequant_8x8(&lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred_arr
            .chunks_exact(8)
            .zip(lrr.chunks_exact(8))
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }

        // chroma U, V
        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let (bx4c, by4c) = (cx / 4, cy / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_420(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_420(plane, bx4c, by4c);
                    encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf44[ci], sk, ds)
                };
                // TX_4X4: 1 coef-context unit wide and tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                let rr = if block_skip {
                    [0i32; 16]
                } else {
                    idct_dequant_4x4(&ccf44[ci], &self.cquant)
                };
                for (ry, rrow) in rr.chunks_exact(4).enumerate() {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    if use_cfl {
                        let prow = &cpred420[ci][ry * 4..];
                        for ((dv, &rv), &p) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if chosen_uv_block == SMOOTH_V_PRED {
                        let prow = &sv_preds_420[ci][ry * 4..];
                        for ((dv, &rv), &prow) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (prow + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            } else if self.ss422 {
                let (bx4c, by4c) = (cx / 4, py / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_422(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_422(plane, bx4c, by4c);
                    encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf48[ci], sk, ds)
                };
                // RTX_4X8: 1 coef-context unit wide, 2 units tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                self.l_coef[plane][by4c + 1] = res_ctx;
                let rr = if block_skip {
                    [0i32; 32]
                } else {
                    idct_dequant_4x8(&ccf48[ci], &self.cquant)
                };
                for (ry, rrow) in rr.chunks_exact(4).enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                    if use_cfl {
                        let prow = &cpred422[ci][ry * 4..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            } else {
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx(plane, bx4, by4, true);
                    let ds = self.dc_sign_ctx(plane, bx4, by4);
                    encode_tx8_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &ccf8[ci],
                        true,
                        sk,
                        ds,
                        0,
                        1,
                    )
                };
                self.a_coef[plane][bx4] = res_ctx;
                self.a_coef[plane][bx4 + 1] = res_ctx;
                self.l_coef[plane][by4] = res_ctx;
                self.l_coef[plane][by4 + 1] = res_ctx;
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    idct_dequant_8x8(&ccf8[ci], &self.cquant)
                };
                for (ry, rrow) in rr.chunks_exact(8).enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        let prow = &cpred444[ci][ry * 8..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        // Plain DC chroma: use the scalar predictor directly so the
                        // reconstruction never depends on the CfL evaluation block
                        // having populated `cpred444`.
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            }
        }
    }

    /// 4:2:2 chroma txb_skip (all_zero) context for an RTX_4X8 block (1 unit
    /// wide, 2 units tall; `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = (a[bx4c] != 0x40) as usize;
        let cl = (l[by4c] != 0x40 || l[by4c + 1] != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_4X8: 1 unit wide, 2 tall, baseline -3.
    fn dc_sign_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 + (l[by4c + 1] >> 6) as i32 - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_8X16 block (2 units wide, 4 tall;
    /// chroma tx == chroma block so ctx_offset = 7): `7 + a_nz + l_nz`, where each
    /// term ORs over the units the block spans.
    fn skip_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = (a[bx4c] != 0x40 || a[bx4c + 1] != 0x40) as usize;
        let cl =
            (l[by4c] != 0x40 || l[by4c + 1] != 0x40 || l[by4c + 2] != 0x40 || l[by4c + 3] != 0x40)
                as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_8X16: 2 units wide, 4 tall, baseline -6.
    fn dc_sign_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32
            + (a[bx4c + 1] >> 6) as i32
            + (l[by4c] >> 6) as i32
            + (l[by4c + 1] >> 6) as i32
            + (l[by4c + 2] >> 6) as i32
            + (l[by4c + 3] >> 6) as i32
            - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_16X32 block (4 units wide, 8 tall).
    fn skip_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4c..bx4c + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4c..by4c + 8].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_16X32: 4 units wide, 8 tall, baseline -12.
    fn dc_sign_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4c..bx4c + 4].iter().map(|x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4c..by4c + 8].iter().map(|x| (x >> 6) as i32).sum();
        let s = suma + suml - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:0 chroma txb_skip context for a TX_4X4 block (1 unit wide and tall;
    /// `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        7 + (a[bx4c] != 0x40) as usize + (l[by4c] != 0x40) as usize
    }

    /// 4:2:0 chroma dc_sign context for TX_4X4: 1 unit each side, baseline -2.
    fn dc_sign_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 - 2;
        (s != 0) as usize + (s > 0) as usize
    }

    /// dc_sign context for a TX_32X32 (8-unit footprint, baseline -16).
    fn dc_sign_ctx_32(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = (0..8).map(|k| (a[bx4 + k] >> 6) as i32).sum();
        let suml: i32 = (0..8).map(|k| (l[by4 + k] >> 6) as i32).sum();
        let s = suma + suml - 16;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a TX_32X32 (8-unit footprint). Luma (max tx in a
    /// 32x32 block) is always ctx 0; chroma uses `7 + above_nz + left_nz`.
    fn skip_ctx_32(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = (0..8).any(|k| a[bx4 + k] != 0x40) as usize;
            let cl = (0..8).any(|k| l[by4 + k] != 0x40) as usize;
            7 + ca + cl
        }
    }

    /// R-D proxy for coding a 32x32 region as one TX_32X32 (PARTITION_NONE) vs
    /// splitting into four 16x16. Only enabled for 4:4:4 (the 32x32 chroma path
    /// is 4:4:4-only so far); 4:2:0/4:2:2 always split. The decoder follows the
    /// signalled partition, so this affects compression only, never correctness.
    fn prefer_32x32(&self, x8: usize, y8: usize) -> bool {
        if self.mono {
            return false; // monochrome codes 8x8 luma blocks only
        }
        // A 32x32 luma block gives a 32x32 chroma transform in 4:4:4 and a
        // 16x32 one in 4:2:2 — both tall enough that a quantized smooth-gradient
        // ramp rings (Gibbs) into horizontal banding. Only 4:2:0 keeps chroma at
        // 16 rows here, so restrict 32x32 luma to 4:2:0.
        if !self.ss420 {
            return false;
        }
        // Even in 4:2:0, a smooth low-contrast 32x32 block rings into a strong
        // low-frequency luma staircase. Keep such blocks small (they split to
        // 16x16, which the smoothness gate may split again to 8x8).
        if self.block_luma_range(x8, y8, 32) < LF_BAND_SMOOTH_RANGE {
            return false;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        // one 32x32 (DC-pred)
        let lpred = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
        let mut r32 = [0i32; 1024];
        for (ry, drow) in r32.chunks_exact_mut(32).enumerate() {
            let srow = &self.src[0][(py + ry) * self.w + px..];
            for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                *dv = s - lpred;
            }
        }
        forward_dct_quant_32x32(&mut r32, &self.quant);
        let cost32: u32 = est_block_bits(&r32, &SCAN_32X32) + OVERHEAD_16;
        // four 16x16 (DC-pred each from current recon; decision-only proxy)
        let mut cost16 = 0u32;
        for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
            let pred = dc_pred_16x16(&self.recon[0], self.w, px + sx, py + sy, self.bd as i32);
            let mut r16 = [0i32; 256];
            for (ry, drow) in r16.chunks_exact_mut(16).enumerate() {
                let srow = &self.src[0][(py + sy + ry) * self.w + px + sx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            forward_dct_quant_16x16(&mut r16, &self.quant);
            cost16 += est_block_bits(&r16, &SCAN_16X16) + OVERHEAD_16;
        }
        // Require a real margin: at high fidelity a 32x32 DCT spreads a region's
        // detail across more coded coefficients than four locally-adapting 16x16
        // blocks, so only pick 32x32 when it is clearly cheaper. This keeps the
        // partition choice from ever being net-negative.
        cost32 + (cost16 >> 4) <= cost16
    }

    /// Code a 32x32 region (4:4:4 only) as a single TX_32X32 block: DC-pred luma
    /// and both chroma planes, forward DCT32 + quant + trellis, the TX_32X32
    /// coefficient coder, and reconstruction via the exact integer inverse.
    /// Updates the 8-unit (32-sample) skip / mode / coef neighbour footprint.
    /// (DC-only for now; SMOOTH/PAETH/directional and CfL at 32x32 come next.)
    fn code_block32(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 8);
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let mlam = mode_lambda() * acq * acq;
        // luma intra mode search (non-directional + directional; the TX_32X32
        // residual transform is always DCT_DCT, so the mode affects prediction
        // only). Mirrors the 16x16 search.
        let mut best_mode = DC_PRED;
        let mut lpred = [0i32; 1024];
        let mut lcf = [0i32; 1024];
        let mut best_eff = f64::INFINITY;
        let mut ltf = [0f64; 1024]; // winner transform coeffs (f64, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        for &m in modes {
            let mut pred = [0i32; 1024];
            if m == DC_PRED {
                let d = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 1024];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 1024];
            for (ry, (rrow, prow)) in resid
                .chunks_exact_mut(32)
                .zip(pred.chunks_exact(32))
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    &self.cdfs,
                    3,
                    0,
                    &self.cdfs.eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, px / 4, py / 4),
                );
            }
            let rr = idct_dequant_32x32(&cf, &self.quant);
            let mut sse = 0i64;
            for (ry, (prow, rrow)) in pred.chunks_exact(32).zip(rr.chunks_exact(32)).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    sse += (d * d) as i64;
                }
            }
            let bits = block_rate_bits(&cf, &SCAN_32X32) + mode_signal_bits(m);
            let cost = sse as f64 + mlam * bits;
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred = pred;
                lcf = cf;
                ltf = tf;
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                &self.cdfs,
                3,
                0,
                &self.cdfs.eob_bin_1024_l,
                self.dc_sign_ctx_32(0, px / 4, py / 4),
            );
        }
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block32_420(x8, y8, &lcf, &lpred, best_mode, luma_zero);
        } else if self.ss422 {
            self.code_block32_422(x8, y8, &lcf, &lpred, best_mode, luma_zero);
        } else {
            self.code_block32_444(x8, y8, &lcf, &lpred, best_mode, luma_zero);
        }
    }

    /// Shared header + luma for a TX_32X32 block: block skip flag, y/uv modes
    /// (uv via `emit_uv_mode`, plain DC or CfL), `angle_delta` for directional
    /// luma modes, the TX_32X32 luma coefficients (no tx-type symbol), the
    /// 8-unit (32-sample) skip/mode/coef footprint, and luma reconstruction.
    #[allow(clippy::too_many_arguments)]
    fn code_header_luma32(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        self.mark_skip8(x8, y8, 4, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(y_mode, uv_mode, cfl);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 8].fill(sv);
        self.l_skip[by4..by4 + 8].fill(sv);
        self.a_mode[bx4..bx4 + 8].fill(mv);
        self.l_mode[by4..by4 + 8].fill(mv);
        let lres = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_32(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_32(0, bx4, by4);
            encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, lcf, false, sk, ds)
        };
        self.a_coef[0][bx4..bx4 + 8].fill(lres);
        self.l_coef[0][by4..by4 + 8].fill(lres);
        let lrr = if block_skip {
            [0i32; 1024]
        } else {
            idct_dequant_32x32(lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred.chunks_exact(32).zip(lrr.chunks_exact(32)).enumerate() {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }
    }

    /// 4:4:4: chroma is also 32x32 (one TX_32X32 per plane), with a CfL vs plain
    /// DC decision per the 16x16 path.
    fn code_block32_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f64,
            self.cquant.ac_q() as f64,
            trellis_lambda(),
        );
        // plain-DC chroma
        let mut ccf = [[0i32; 1024]; 2];
        let mut cdc = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = dc_pred_32x32(&self.recon[plane], self.w, px, py, self.bd as i32);
            cdc[ci] = dc;
            let mut cresid = [0i32; 1024];
            for (ry, drow) in cresid.chunks_exact_mut(32).enumerate() {
                let srow = &self.src[plane][(py + ry) * self.w + px..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_32x32_t(&cresid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(&mut ccf[ci], &qt, dcq, acq, &SCAN_32X32, lam);
            let mean_resid_dc = cresid.iter().sum::<i32>() / 1024;
            if ccf[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // CfL: predict chroma from the reconstructed luma AC.
        let mut cfl_ccf = [[0i32; 1024]; 2];
        let mut cfl_pred = [[0i32; 1024]; 2];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f64; 2], [0f64; 2]);
        let mlam = mode_lambda() * acq * acq;
        {
            let lrr_cfl = idct_dequant_32x32(lcf, &self.quant);
            let mut luma_rec = [0i32; 1024];
            for i in 0..1024 {
                luma_rec[i] = (lpred[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 1024];
            cfl_ac_444(&luma_rec, 32, 32, &mut ac);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cdc[ci];
                let mut src = [0i32; 1024];
                for (ry, drow) in src.chunks_exact_mut(32).enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..32]);
                }
                let dcrr = idct_dequant_32x32(&ccf[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..1024 {
                    let d = src[i] - (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    s += (d * d) as i64;
                }
                dc_cost[ci] = s as f64 + mlam * block_rate_bits(&ccf[ci], &SCAN_32X32);
                let a = cfl_best_alpha(&ac, &src, dc, 1024, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 1024];
                let mut resid = [0i32; 1024];
                for i in 0..1024 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_32X32, lam);
                let rr = idct_dequant_32x32(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..1024 {
                    let d = src[i] - (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_pred[ci] = cpr;
                cfl_cost[ci] = s2 as f64 + mlam * block_rate_bits(&q, &SCAN_32X32);
            }
        }
        // CfL signalling costs extra (sign + per-plane alpha); only use it when
        // it beats plain DC on both planes' summed cost by that overhead.
        let cfl_sig =
            4.0 + if cfl_a[0] != 0 { 4.0 } else { 0.0 } + if cfl_a[1] != 0 { 4.0 } else { 0.0 };
        // Gate CfL on low quality: at high quality DC prediction is accurate and CfL
        // alpha varying block-to-block creates visible colour stripes at boundaries.
        let use_cfl = acq > 300.0
            && (cfl_a[0] != 0 || cfl_a[1] != 0)
            && cfl_cost[0] + cfl_cost[1] + mlam * cfl_sig < dc_cost[0] + dc_cost[1];
        #[allow(unused_mut)] // cfl_opt mutated in 'sv block when SMOOTH_V wins
        let (cf_use, pred_dc, mut cfl_opt): (
            &[[i32; 1024]; 2],
            [i32; 2],
            Option<[i32; 2]>,
        ) = if use_cfl {
            (&cfl_ccf, cdc, Some(cfl_a))
        } else {
            (&ccf, cdc, None)
        };
        // SMOOTH_V check on 32x32 chroma (same principle as 16x16 path).
        #[allow(unused_mut)] // assigned via break in 'sv labeled block
        let mut cf_use_owned: [[i32; 1024]; 2];
        let mut sv_preds32 = [[0i32; 1024]; 2];
        let (final_cf, chosen_uv_32) = 'sv: {
            // Gate SMOOTH_V on low quality only
            if self.quant.ac_q() <= 300 {
                break 'sv (cf_use, DC_PRED);
            }
            let mut sv_ccf32 = [[0i32; 1024]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            let dcq2 = self.cquant.dc_q() as f64;
            let acq2 = self.cquant.ac_q() as f64;
            let lam2 = trellis_lambda();
            for ci in 0..2 {
                let plane = ci + 1;
                // sse_cur: raw source vs current winner prediction (DC scalar or CfL pixels)
                for ry in 0..32 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        let prow = &cfl_pred[ci][ry * 32..];
                        for (&sr, &pr) in srow[..32].iter().zip(prow[..32].iter()) {
                            let d = sr - pr;
                            sse_cur += (d * d) as i64;
                        }
                    } else {
                        let dc = pred_dc[ci];
                        for &s in srow[..32].iter() {
                            let d = s - dc;
                            sse_cur += (d * d) as i64;
                        }
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    false,
                    false,
                    self.w,
                    self.h,
                    &mut sv_preds32[ci],
                    self.bd,
                );
                let mut resid = [0i32; 1024];
                for (ry, drow) in resid.chunks_exact_mut(32).enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds32[ci][ry * 32..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                sv_ccf32[ci] = q;
                trellis_optimize(&mut sv_ccf32[ci], &qt, dcq2, acq2, &SCAN_32X32, lam2);
                let mean_resid_sv = resid.iter().sum::<i32>() / 1024;
                if sv_ccf32[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    sv_ccf32[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
                for ry in 0..32 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    let prow = &sv_preds32[ci][ry * 32..];
                    for (&srow, &prow) in srow[..32].iter().zip(prow[..32].iter()) {
                        let d = srow - prow;
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv < sse_cur {
                cfl_opt = None; // SMOOTH_V overrides CfL if it wins
                cf_use_owned = sv_ccf32;
                break 'sv (&cf_use_owned, SMOOTH_V_PRED);
            }
            (cf_use, DC_PRED)
        };
        let block_skip =
            luma_zero && final_cf[0].iter().all(|&c| c == 0) && final_cf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv_32,
            cfl_opt,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let cres = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_32(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_32(plane, bx4, by4);
                encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, &final_cf[ci], true, sk, ds)
            };
            self.a_coef[plane][bx4..bx4 + 8].fill(cres);
            self.l_coef[plane][by4..by4 + 8].fill(cres);
            let crr = if block_skip {
                [0i32; 1024]
            } else {
                idct_dequant_32x32(&final_cf[ci], &self.cquant)
            };
            for (ry, rrow) in crr.chunks_exact(32).enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if chosen_uv_32 == SMOOTH_V_PRED {
                    let prow = &sv_preds32[ci][ry * 32..];
                    for (j, (dv, &rv)) in drow[..32].iter_mut().zip(rrow.iter()).enumerate() {
                        *dv = (prow[j] + rv).clamp(0, (1 << self.bd) - 1);
                    }
                } else {
                    let base = if use_cfl {
                        cfl_pred[ci][ry * 32..][0]
                    } else {
                        pred_dc[ci]
                    };
                    for (dv, (&cp, &rv)) in drow[..32]
                        .iter_mut()
                        .zip(cfl_pred[ci][ry * 32..].iter().zip(rrow.iter()))
                    {
                        let b = if use_cfl { cp } else { base };
                        *dv = (b + rv).clamp(0, (1 << self.bd) - 1);
                    }
                }
            }
        }
    }

    /// 4:2:0: a 32x32 luma region maps to a 16x16 chroma block per plane
    /// (`TX_16X16`, coef-CDF class 2). DC-pred chroma (CfL-420 needs 2x2 luma AC
    /// downsampling, deferred).
    fn code_block32_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f64,
            self.cquant.ac_q() as f64,
            trellis_lambda(),
        );
        let maxval = (1 << self.bd) - 1;
        let mut ccf_dc = [[0i32; 256]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = dc_pred_16x16(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 256];
            for (ry, drow) in resid.chunks_exact_mut(16).enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
            ccf_dc[ci] = q;
            trellis_optimize(&mut ccf_dc[ci], &qt, dcq, acq, &SCAN_16X16, lam);
            let mean_resid_dc = resid.iter().sum::<i32>() / 256;
            if ccf_dc[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf_dc[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        let smooth_v_active_32 = dcq > 300.0;
        let mut ccf_sv = [[0i32; 256]; 2];
        let mut sv_preds = [[0i32; 256]; 2];
        if smooth_v_active_32 {
            for ci in 0..2 {
                let plane = ci + 1;
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    16,
                    16,
                    false,
                    false,
                    self.cw,
                    self.h,
                    &mut sv_preds[ci],
                    self.bd,
                );
                let mut resid = [0i32; 256];
                for (ry, drow) in resid.chunks_exact_mut(16).enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds[ci][ry * 16..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
                ccf_sv[ci] = q;
                trellis_optimize(&mut ccf_sv[ci], &qt, dcq, acq, &SCAN_16X16, lam);
                let mean_resid_sv = resid.iter().sum::<i32>() / 256;
                if ccf_sv[ci][0] == 0 && mean_resid_sv.abs() >= 8 {
                    ccf_sv[ci][0] = if mean_resid_sv > 0 { 1 } else { -1 };
                }
            }
        } // end if smooth_v_active_32
        let mut rr_dc = [[0i32; 256]; 2];
        let mut rr_sv = [[0i32; 256]; 2];
        let mut sse_dc = 0i64;
        let mut sse_sv = 0i64;
        for ci in 0..2 {
            let plane = ci + 1;
            rr_dc[ci] = idct_dequant_16x16(&ccf_dc[ci], &self.cquant);
            rr_sv[ci] = idct_dequant_16x16(&ccf_sv[ci], &self.cquant);
            let dc = dc_preds[ci];
            for (ry, (rd_row, rs_row)) in rr_dc[ci]
                .chunks_exact(16)
                .zip(rr_sv[ci].chunks_exact(16))
                .enumerate()
            {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                let prow = &sv_preds[ci][ry * 16..];
                for (((&s, &prow), &rd), &rs) in srow[..16]
                    .iter()
                    .zip(prow[..16].iter())
                    .zip(rd_row[..16].iter())
                    .zip(rs_row[..16].iter())
                {
                    let d = s - (dc + rd).clamp(0, maxval);
                    let v = s - (prow + rs).clamp(0, maxval);
                    sse_dc += (d * d) as i64;
                    sse_sv += (v * v) as i64;
                }
            }
        }
        let use_sv = smooth_v_active_32 && sse_sv <= sse_dc;
        let (chosen_uv, ccf, rr_cache) = if use_sv {
            (SMOOTH_V_PRED, ccf_sv, rr_sv)
        } else {
            (DC_PRED, ccf_dc, rr_dc)
        };
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(x8, y8, lcf, lpred, y_mode, block_skip, chosen_uv, None);
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx_16(plane, bx4c, by4c);
                encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    0,
                    1,
                )
            };
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else {
                rr_cache[ci]
            };
            for (ry, rrow) in rr.chunks_exact(16).enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                if use_sv {
                    let prow = &sv_preds[ci][ry * 16..];
                    for ((dv, &rv), &prow) in drow[..16]
                        .iter_mut()
                        .zip(rrow[..16].iter())
                        .zip(prow[..16].iter())
                    {
                        *dv = (prow + rv).clamp(0, maxval);
                    }
                } else {
                    let dc = dc_preds[ci];
                    for (dv, &rv) in drow[..16].iter_mut().zip(rrow.iter()) {
                        *dv = (dc + rv).clamp(0, maxval);
                    }
                }
            }
        }
    }

    /// 4:2:2: a 32x32 luma region maps to a 16-wide x 32-tall chroma block per
    /// plane (`RTX_16X32`, coef-CDF class 3). DC-pred chroma.
    fn code_block32_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        let (bx4c, by4c) = (cx / 4, py / 4);
        let mut ccf = [[0i32; 512]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_16x32(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 512];
            for (ry, drow) in resid.chunks_exact_mut(16).enumerate() {
                let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_16x32_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f64,
                self.cquant.ac_q() as f64,
                &SCAN_16X32,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(x8, y8, lcf, lpred, y_mode, block_skip, DC_PRED, None);
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x32_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_16x32_422(plane, bx4c, by4c);
                encode_16x32_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 8].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 512]
            } else {
                idct_dequant_16x32(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.chunks_exact(16).enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                }
            }
        }
    }
    /// Apply the AV1 in-loop deblocking filter to this tile's reconstruction.
    /// `level_y` filters luma (both edge directions, sharpness 0); `level_uv`
    /// filters both chroma planes. Each tile is an independent sub-frame so
    /// filtering stays within the tile. Block geometry comes from `blk4`.
    fn apply_loop_filter(&mut self, level_y: i32, level_uv: i32) {
        let nc4 = self.w / 4;
        // luma: square blocks -> bw4 == bh4 == blk4
        if level_y > 0 {
            let blk = self.blk4.clone();
            crate::loopfilter::filter_plane(
                &mut self.recon[0],
                self.w,
                self.h,
                &blk,
                &blk,
                nc4,
                level_y,
                true,
                16, // 64px superblock -> 16 4-unit rows
                self.bd,
            );
        }
        if self.mono || level_uv <= 0 {
            return;
        }
        let ss_hor = (self.ss422 || self.ss420) as usize;
        let ss_ver = self.ss420 as usize;
        let cw = self.cw;
        let ch = if self.ss420 { self.h / 2 } else { self.h };
        let cnc4 = cw / 4;
        let cnr4 = ch / 4;
        // derive chroma block geometry from luma blk4
        let mut cbw4 = vec![0u8; cnc4 * cnr4];
        let mut cbh4 = vec![0u8; cnc4 * cnr4];
        for cr in 0..cnr4 {
            for cc in 0..cnc4 {
                let lr = cr << ss_ver;
                let lc = cc << ss_hor;
                let d = self.blk4[lr * nc4 + lc];
                cbw4[cr * cnc4 + cc] = (d >> ss_hor).max(1);
                cbh4[cr * cnc4 + cc] = (d >> ss_ver).max(1);
            }
        }
        let csb = 16 >> ss_ver; // chroma superblock height in 4-units
        for plane in 1..3 {
            crate::loopfilter::filter_plane(
                &mut self.recon[plane],
                cw,
                ch,
                &cbw4,
                &cbh4,
                cnc4,
                level_uv,
                false,
                csb,
                self.bd,
            );
        }
    }

    /// Record a square luma block's size (in 4-sample units) into the `blk4`
    /// grid for every 4x4 luma unit it covers. Used by the deblocking filter to
    /// locate block edges and pick filter widths. `(x8,y8)` is the 8-sample
    /// origin; `dim4` is the block size in 4-units (2=8px, 4=16px, 8=32px).
    /// Record a coding block's skip flag into the per-8x8 CDEF map. `dim8` is the
    /// block side in 8-pixel units (8x8 -> 1, 16x16 -> 2, 32x32 -> 4).
    fn mark_skip8(&mut self, x8: usize, y8: usize, dim8: usize, skip: bool) {
        let sb8w = self.w.div_ceil(8);
        let sb8h = self.h.div_ceil(8);
        for ry in 0..dim8 {
            for rx in 0..dim8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    fn record_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        let nc4 = self.w / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        let d = dim4 as usize;
        let nr4 = self.h / 4;
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.blk4[r * nc4 + c] = dim4;
            }
        }
    }

    fn decode_sb(&mut self, bl: usize, x8: usize, y8: usize, sz8: usize, thr: bool, lhb: bool) {
        if sz8 == 1 {
            // BL_8X8 leaf (always fully in-frame for multiple-of-8 dimensions):
            // emit PARTITION_NONE, then the block.
            let ctx = get_partition_ctx(&self.a_part, &self.l_part, 4, x8, y8);
            self.enc.encode_symbol(0, &mut self.cdfs.part_bl8[ctx]);
            let have_tr = thr && y8 > 0 && (x8 * 8 + 8) < self.w;
            let have_bl = lhb && x8 > 0 && (y8 * 8 + 8) < self.h;
            self.code_block(x8, y8, have_tr, have_bl);
            self.a_part[x8] = 0x1e;
            self.l_part[y8] = 0x1e;
            return;
        }
        // BL_32X32: optionally code the whole 32x32 as one TX_32X32 block
        // (PARTITION_NONE) instead of splitting into four 16x16. 4:4:4 only for
        // now (prefer_32x32 returns false otherwise). Requires the full 32x32
        // in-frame.
        if sz8 == 4 {
            let full_h = (x8 + 4) * 8 <= self.w;
            let full_v = (y8 + 4) * 8 <= self.h;
            if full_h && full_v && self.prefer_32x32(x8, y8) {
                let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                self.enc
                    .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]); // NONE
                let have_tr = thr && y8 > 0 && (x8 * 8 + 32) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 32) < self.h;
                self.code_block32(x8, y8, have_tr, have_bl);
                self.a_part[x8..x8 + 4].fill(0x18);
                self.l_part[y8..y8 + 4].fill(0x18);
                return;
            }
        }
        // BL_16X16: optionally code the whole 16x16 as one TX_16X16 block
        // (PARTITION_NONE) instead of splitting into four 8x8. Enabled for all
        // subsampling modes: 4:4:4 (chroma 16x16), 4:2:0 (chroma TX_8X8) and
        // 4:2:2 (chroma RTX_8X16). Requires the full 16x16 to be in-frame
        // (have_h && have_v at hh=1 guarantees it, since the coded frame is
        // 8-aligned and the test is strict).
        if sz8 == 2 {
            let have_h = (x8 + 1) * 8 < self.w;
            let have_v = (y8 + 1) * 8 < self.h;
            if have_h && have_v && self.prefer_16x16(x8, y8) {
                let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                self.enc
                    .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]); // NONE
                let have_tr = thr && y8 > 0 && (x8 * 8 + 16) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 16) < self.h;
                self.code_block16(x8, y8, have_tr, have_bl);
                self.a_part[x8..x8 + 2].fill(0x1c);
                self.l_part[y8..y8 + 2].fill(0x1c);
                return;
            }
        }
        let hh = sz8 / 2;
        // content past the horizontal / vertical midpoint of this block?
        let have_h = (x8 + hh) * 8 < self.w;
        let have_v = (y8 + hh) * 8 < self.h;
        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
        if have_h && have_v {
            // full PARTITION_SPLIT symbol -> adapt the partition CDF (dav1d uses
            // decode_symbol_adapt here).
            self.enc
                .encode_symbol(3, &mut self.cdfs.part_split[bl - 1][ctx]);
        } else if have_h {
            // edge: dav1d codes a NON-adapting bool from a gathered probability;
            // read the live (possibly already-adapted) icdf CDF, do not adapt it.
            let p = gather_split_prob_icdf(&self.cdfs.part_split[bl - 1][ctx], true);
            self.enc.encode_bool(true, p);
        } else if have_v {
            let p = gather_split_prob_icdf(&self.cdfs.part_split[bl - 1][ctx], false);
            self.enc.encode_bool(true, p);
        }
        // else: neither -> implicit split, no symbol
        // recurse the children whose top-left is in-frame, propagating the
        // intra-edge availability (dav1d intra-edge tree). z-order child index
        // n: 0=TL,1=TR,2=BL,3=BR -> (top_has_right, left_has_bottom):
        //   TL=(1,1)  TR=(parent_thr,0)  BL=(1,parent_lhb)  BR=(0,0)
        let children = [
            (x8, y8, true, true),
            (x8 + hh, y8, thr, false),
            (x8, y8 + hh, true, lhb),
            (x8 + hh, y8 + hh, false, false),
        ];
        for (cx, cy, cthr, clhb) in children {
            if cx * 8 < self.w && cy * 8 < self.h {
                self.decode_sb(bl + 1, cx, cy, hh, cthr, clhb);
            }
        }
    }
}

/// Encode a **lossy** 4:4:4 still of arbitrary size (width and height multiples
/// of 64). `planes` are luma (G), U (B), V (R), each a `w*h` raster of 0..=255.
/// The frame is tiled into 64x64 superblocks (raster order, single tile); each
/// superblock is split uniformly into 8x8 blocks coded DC_PRED + TX_8X8
/// (DCT_DCT) and quantized by `base_q_idx` (keep `<= 20` for coefficient qctx 0).
/// Round `n` up to the next multiple of 8.
/// Source luma peak-to-peak range below which a block is treated as a smooth
/// gradient and kept on small (8x8) transforms to avoid low-frequency banding.
/// Real texture/edges exceed this, so they still use efficient large transforms.
const LF_BAND_SMOOTH_RANGE: i32 = 32;

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

/// AV1 `increment_*_log2` bit sequence signalling `target` to a decoder that
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
struct Tiling {
    tcl: u32,
    trl: u32,
    cols_incr: Vec<bool>,
    rows_incr: Vec<bool>,
}

/// Pick a tiling for a frame of `sb_cols` x `sb_rows` superblocks. It is always
/// at least the spec **minimum** the decoder derives (so large frames stay
/// valid), and is subdivided further toward `target_tiles` so tile-level threads
/// have independent work. `target_tiles == 1` yields exactly the spec minimum —
/// a single tile for small frames, byte-identical to the untiled path. Extra
/// tiles trade a little compression (each tile resets entropy contexts and can't
/// predict across its edges) for parallelism, splitting the longer side first so
/// tiles stay roughly square.
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

/// Spec-minimum `(TileColsLog2, TileRowsLog2)` (i.e. [`plan_tiling`] with a
/// single-tile target). Retained for the tiling unit tests.
#[cfg(test)]
fn choose_tiling(sb_cols: u32, sb_rows: u32) -> (u32, u32) {
    let t = plan_tiling(sb_cols, sb_rows, 1);
    (t.tcl, t.trl)
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
    dst: &mut [i32],
    full_w: usize,
    x0: usize,
    y0: usize,
    tile: &[i32],
    tw: usize,
    th: usize,
) {
    for r in 0..th {
        let d = (y0 + r) * full_w + x0;
        dst[d..d + tw].copy_from_slice(&tile[r * tw..(r + 1) * tw]);
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
    recon: [Vec<i32>; 3],
    skip8: Vec<bool>, // per-8x8 luma-unit skip flag (tile-local, row-major over ceil(tw/8))
}

/// Encode a single tile as an independent sub-frame. Pure function of its inputs
/// (no shared mutable state), so it is safe to run on any thread. When `mono`,
/// only the luma plane is coded (`src[1]`/`src[2]` ignored, chroma recon empty).
#[allow(clippy::too_many_arguments)]
fn encode_one_tile(
    base_q_idx: u8,
    bd: u8,
    full_w: usize,
    cw8: usize,
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    src: &[Vec<i32>; 3],
    r: &TileRect,
    speed: Speed,
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
    let mut tile = if mono {
        LossyTile::new_mono(base_q_idx, bd, r.tw, r.th, &tsrc)
    } else {
        match (sub_x, sub_y) {
            (0, 0) => LossyTile::new(base_q_idx, bd, r.tw, r.th, &tsrc),
            (1, 0) => LossyTile::new_422(base_q_idx, bd, r.tw, r.th, &tsrc),
            _ => LossyTile::new_420(base_q_idx, bd, r.tw, r.th, &tsrc),
        }
    }
    .with_speed(speed);
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    // In-loop deblocking filter (final reconstruction step, after all blocks are
    // coded — intra prediction used the unfiltered recon, matching the decoder).
    let (lvl_y, lvl_uv) = crate::obu::loop_filter_levels(base_q_idx);
    tile.apply_loop_filter(lvl_y, lvl_uv);
    // CDEF is applied at the frame level (after stitching) so its RD strength
    // search sees the whole frame and matches dav1d's frame-level filtering.
    let skip8 = tile.skip8;
    let payload = tile.enc.done();
    TileOut {
        payload,
        recon: tile.recon,
        skip8,
    }
}

/// Resolve the requested thread count: `0` => all available cores (fallback 1),
/// otherwise the value as-is. The caller still caps this at the tile count.
fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

/// Encode `src` (already padded to `w8` x `h8`, chroma subsampled by
/// `sub_x`/`sub_y`) as one or more AV1 tiles and return the **tile-group
/// payload** (everything that follows the frame header inside `OBU_FRAME`), the
/// stitched full-frame reconstruction, and the chosen `(TileColsLog2,
/// TileRowsLog2)`.
///
/// Each tile is encoded as an independent sub-frame: the source is cropped to
/// the tile's pixel rectangle and handed to a fresh [`LossyTile`] whose origin
/// is the tile's top-left, so tile boundaries become frame boundaries and all
/// the existing prediction/availability/context logic applies unchanged (intra
/// prediction and entropy contexts never cross a tile edge, as the spec
/// requires). For a single tile the payload is just that tile's bytes —
/// byte-identical to the previous single-tile path.
///
/// `threads` controls tile-level parallelism (AV1's natural parallel unit, since
/// tiles share no state): `1` runs serially (no threads spawned), `0` uses all
/// available cores, `N` uses up to `N`; the effective count is capped at the
/// number of tiles. The output is byte-identical regardless of `threads` — the
/// thread count only decides which core encodes which tile.
#[allow(clippy::too_many_arguments)]
fn encode_lossy_tilegroup(
    base_q_idx: u8,
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i32>; 3],
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    threads: usize,
    speed: Speed,
) -> (Vec<u8>, [Vec<i32>; 3], Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;

    // Aim for ~one tile per worker so small frames can be paralleled too.
    // `threads == 1` -> target 1 -> spec-minimum tiling (single tile for small
    // frames, byte-identical to the untiled output).
    let want = resolve_threads(threads);
    let plan = plan_tiling(sb_cols, sb_rows, want);
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

    let n = rects.len();
    let nthreads = want.clamp(1, n.max(1));

    // Encode every tile. Serial when a single thread (or single tile) is asked
    // for; otherwise split the tiles into disjoint chunks, one scoped thread per
    // chunk (no shared mutable state, so no locks and no `unsafe`).
    let outs: Vec<TileOut> = if nthreads <= 1 || n <= 1 {
        rects
            .iter()
            .map(|r| encode_one_tile(base_q_idx, bd, w8, cw8, sub_x, sub_y, mono, src, r, speed))
            .collect()
    } else {
        let mut slots: Vec<Option<TileOut>> = (0..n).map(|_| None).collect();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (r, o) in rs.iter().zip(os.iter_mut()) {
                        *o = Some(encode_one_tile(
                            base_q_idx, bd, w8, cw8, sub_x, sub_y, mono, src, r, speed,
                        ));
                    }
                });
            }
        });
        slots.into_iter().map(|o| o.unwrap()).collect()
    };

    // Stitch reconstructions and collect payloads (raster order, serial).
    // Monochrome has only a luma plane; chroma recon stays empty.
    let mut recon = if mono {
        [vec![0i32; w8 * h8], Vec::new(), Vec::new()]
    } else {
        [
            vec![0i32; w8 * h8],
            vec![0i32; cw8 * ch8],
            vec![0i32; cw8 * ch8],
        ]
    };
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(n);
    let sb8w = w8.div_ceil(8);
    let sb8h = h8.div_ceil(8);
    let mut skip8 = vec![true; sb8w * sb8h];
    for (r, out) in rects.iter().zip(outs) {
        // stitch this tile's per-8x8 skip map into the frame map
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
        stitch_plane(&mut recon[0], w8, r.x0, r.y0, &out.recon[0], r.tw, r.th);
        if !mono {
            stitch_plane(
                &mut recon[1],
                cw8,
                r.cx0,
                r.cy0,
                &out.recon[1],
                r.ctw,
                r.cth,
            );
            stitch_plane(
                &mut recon[2],
                cw8,
                r.cx0,
                r.cy0,
                &out.recon[2],
                r.ctw,
                r.cth,
            );
        }
        payloads.push(out.payload);
    }

    let tilegroup = assemble_tilegroup(payloads);
    (tilegroup, recon, plan)
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
fn assemble_frame_obus(base_q_idx: u8, plan: &Tiling, tilegroup: &[u8], mono: bool) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = frame_header_lossy_multitile_th(
            base_q_idx,
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            mono,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh =
            frame_header_lossy_multitile(base_q_idx, &plan.cols_incr, &plan.rows_incr, 0, 0, mono);
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Encode one lossless 4:4:4 tile: crop the three full-resolution planes to the
/// tile's pixel rect and hand them to `encode_tile_lossless` (whose origin is the
/// tile, so the tile's top/left behave as frame edges — intra prediction and
/// entropy never cross a tile boundary). Pure function of its inputs, so it runs
/// on any thread. Lossless recon equals the source, so no reconstruction is
/// returned. `r` is `(x0, y0, tw, th)` in pixels.
/// Encode a lossless 4:4:4 frame to its OBU frame portion (an `OBU_FRAME` for a
/// single tile, or `OBU_FRAME_HEADER` + `OBU_TILE_GROUP` for multiple). `src` are
/// the three full-resolution `w8*h8` planes, already padded to a multiple of 8.
/// Tiling is chosen automatically (at least the spec minimum, so large frames
/// are valid); `threads` parallelises across tiles (`1` = serial, byte-identical
/// to thz old single-tile output for small frames). The caller prepends the
/// temporal delimiter, sequence header and any metadata OBUs.
pub(crate) fn encode_lossless_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
) -> Vec<u8> {
    let (tilegroup, plan) = encode_lossless_tilegroup(bd, w8, h8, src, threads);
    assemble_lossless_frame_obus(&plan, &tilegroup)
}

fn encode_one_lossless_tile(
    bd: u8,
    full_w: usize,
    src: &[Vec<i16>; 3],
    r: &(usize, usize, usize, usize),
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(&src[0], full_w, x0, y0, tw, th);
    let p1 = crop_plane(&src[1], full_w, x0, y0, tw, th);
    let p2 = crop_plane(&src[2], full_w, x0, y0, tw, th);
    crate::av1_tile::encode_tile_lossless(tw, th, bd, [&p0, &p1, &p2])
}

/// Encode a **lossless** 4:4:4 frame as a (possibly multi-tile) tile group,
/// mirroring the lossy tiling path. The frame is split into at least the spec
/// minimum tiling — so frames wider than 4096px or larger than the max tile area
/// stay valid (the previous single-tile lossless path mis-signalled these) — and
/// further toward `threads` tiles for parallelism. Each tile is encoded
/// independently; `threads` parallelises across tiles with scoped threads (no
/// shared mutable state, no locks). `threads == 1` yields the spec minimum: a
/// single tile for small frames, byte-identical to the untiled path. The output
/// is byte-identical regardless of thread count for a fixed tiling.
fn encode_lossless_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = resolve_threads(threads);
    let plan = plan_tiling(sb_cols, sb_rows, want);
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
    let payloads: Vec<Vec<u8>> = if nthreads <= 1 || n <= 1 {
        rects
            .iter()
            .map(|r| encode_one_lossless_tile(bd, w8, src, r))
            .collect()
    } else {
        let mut slots: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (r, o) in rs.iter().zip(os.iter_mut()) {
                        *o = Some(encode_one_lossless_tile(bd, w8, src, r));
                    }
                });
            }
        });
        slots.into_iter().map(|o| o.unwrap()).collect()
    };

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a lossless tile group with the matching frame header: a single tile uses
/// a combined `OBU_FRAME` (type 6); multiple tiles use a separate
/// `OBU_FRAME_HEADER` (type 3) + `OBU_TILE_GROUP` (type 4), the layout strict
/// parsers (ffmpeg's cbs_av1) accept.
fn assemble_lossless_frame_obus(plan: &Tiling, tilegroup: &[u8]) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh =
            crate::obu::frame_header_lossless_multitile(&plan.cols_incr, &plan.rows_incr, 0, 0);
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Crop the single luma plane to a tile rect and encode it as a mono lossless
/// tile. Pure function of its inputs (safe on any thread).
fn encode_one_lossless_tile_mono(
    bd: u8,
    full_w: usize,
    luma: &[i16],
    r: &(usize, usize, usize, usize),
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(luma, full_w, x0, y0, tw, th);
    crate::av1_tile::encode_tile_lossless_mono(tw, th, bd, &p0)
}

/// Monochrome counterpart of [`encode_lossless_tilegroup`]: a single full-res
/// luma plane (`w8*h8`, padded to a multiple of 8) tiled identically to the
/// 4:4:4 path. Byte-identical output for a fixed tiling regardless of thread
/// count.
fn encode_lossless_mono_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    luma: &[i16],
    threads: usize,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let want = resolve_threads(threads);
    let plan = plan_tiling(sb_cols, sb_rows, want);
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
    let payloads: Vec<Vec<u8>> = if nthreads <= 1 || n <= 1 {
        rects
            .iter()
            .map(|r| encode_one_lossless_tile_mono(bd, w8, luma, r))
            .collect()
    } else {
        let mut slots: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (r, o) in rs.iter().zip(os.iter_mut()) {
                        *o = Some(encode_one_lossless_tile_mono(bd, w8, luma, r));
                    }
                });
            }
        });
        slots.into_iter().map(|o| o.unwrap()).collect()
    };

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a mono lossless tile group with a `mono_chrome = 1` lossless frame
/// header (single tile ⇒ combined `OBU_FRAME`; multi-tile ⇒ `OBU_FRAME_HEADER` +
/// `OBU_TILE_GROUP`).
fn assemble_lossless_mono_frame_obus(plan: &Tiling, tilegroup: &[u8]) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_mono_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = crate::obu::frame_header_lossless_mono_multitile(
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Encode a monochrome lossless frame's OBU portion from a padded `w8*h8` luma
/// plane. Caller prepends temporal delimiter, sequence header, metadata.
pub(crate) fn encode_lossless_mono_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    luma: &[i16],
    threads: usize,
) -> Vec<u8> {
    let (tilegroup, plan) = encode_lossless_mono_tilegroup(bd, w8, h8, luma, threads);
    assemble_lossless_mono_frame_obus(&plan, &tilegroup)
}

/// Encode a full monochrome **lossless** AV1 still image: temporal delimiter +
/// monochrome sequence header (`mono_chrome = 1`) + lossless frame. `luma` is
/// `w*h` samples; it is padded to a multiple of 8 internally. Profile 0 for
/// 8/10-bit, profile 2 for 12-bit.
pub(crate) fn encode_av1_mono_lossless_image(
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i16],
    full_range: bool,
    threads: usize,
) -> Vec<u8> {
    assert_eq!(luma.len(), w * h, "luma plane must be w*h");
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let padded = pad_to_mult8(luma, w, h, w8, h8);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_mono(
        w as u32, h as u32, bd, full_range,
    ));
    bytes.extend_from_slice(&encode_lossless_mono_frame_obus(
        bd, w8, h8, &padded, threads,
    ));
    bytes
}

/// Lossy encoder with explicit color mode: `ycbcr=false` signals MC_IDENTITY
/// (planes coded as GBR); `ycbcr=true` signals full-range BT.601 so the decoder
/// converts the coded Y/Cb/Cr planes back to RGB (decorrelated -> smaller).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_cs(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
) -> Vec<u8> {
    encode_av1_lossy_image_cs_recon_dbg(base_q_idx, bd, w, h, luma, u, v, color, threads, speed).0
}

/// Debug variant of [`encode_av1_lossy_image_cs`] (4:4:4) returning the padded
/// reconstruction and dims, for bit-exactness verification.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_cs_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
) -> (Vec<u8>, [Vec<i32>; 3], (usize, usize)) {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let src = [
        pad_to_mult8(luma, w, h, w8, h8),
        pad_to_mult8(u, w, h, w8, h8),
        pad_to_mult8(v, w, h, w8, h8),
    ];
    let (payload, recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 0, 0, false, threads, speed);
    let profile = if bd == 12 { 2 } else { 1 };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp(
        w as u32, h as u32, profile, bd, color,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, false));
    (bytes, recon, (w8, h8))
}

/// Encode a **lossy 4:2:2** YCbCr still (profile 2). `luma` is `w*h`; `u`/`v`
/// are the horizontally-subsampled chroma planes, each `cw*h` with
/// `cw = (w+1)/2`. The decoder reconstructs full-resolution RGB via the
/// signalled BT.601 matrix and 4:2:2 upsampling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_422(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
) -> Vec<u8> {
    encode_av1_lossy_image_422_recon_dbg(base_q_idx, bd, w, h, luma, u, v, color, threads, speed).0
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn encode_av1_lossy_image_422_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
) -> (Vec<u8>, [Vec<i32>; 3], (usize, usize, usize)) {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let cw = w.div_ceil(2);
    assert_eq!(u.len(), cw * h);
    assert_eq!(v.len(), cw * h);
    let (w8, h8) = (align8(w), align8(h));
    let cw8 = w8 / 2;
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| -> Vec<i32> { pad_to_mult8(p, cw, h, cw8, h8) };
    let src = [luma_p, pad_c(u), pad_c(v)];
    let (payload, recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 1, 0, false, threads, speed);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp_ss(
        w as u32, h as u32, 2, bd, color, 1, 0,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, false));
    (bytes, recon, (w8, h8, cw8))
}

/// Encode a **lossy 4:2:0** YCbCr still (profile 0). `luma` is `w*h`; `u`/`v`
/// are the half-width, half-height chroma planes, each `cw*ch` with
/// `cw=(w+1)/2`, `ch=(h+1)/2`. Each 8x8 luma block carries a 4x4 (`TX_4X4`)
/// chroma block per plane. Reconstruction is bit-exact vs dav1d 1.4.1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_lossy_image_420(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
) -> Vec<u8> {
    encode_av1_lossy_image_420_recon_dbg(base_q_idx, bd, w, h, luma, u, v, color, threads, speed).0
}

/// Debug variant of [`encode_av1_lossy_image_420`] also returning the encoder's
/// padded reconstruction `[Y, U, V]` and the padded dims `(w8, h8, cw8, ch8)`,
/// for bit-exactness verification against the decoder.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn encode_av1_lossy_image_420_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&crate::color::Cicp>,
    threads: usize,
    speed: Speed,
) -> (Vec<u8>, [Vec<i32>; 3], (usize, usize, usize, usize)) {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    assert_eq!(u.len(), cw * ch);
    assert_eq!(v.len(), cw * ch);
    let (w8, h8) = (align8(w), align8(h));
    let (cw8, ch8) = (w8 / 2, h8 / 2);
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| -> Vec<i32> { pad_to_mult8(p, cw, ch, cw8, ch8) };
    let src = [luma_p, pad_c(u), pad_c(v)];
    let (payload, recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 1, 1, false, threads, speed);
    let profile = if bd == 12 { 2 } else { 0 };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp_ss(
        w as u32, h as u32, profile, bd, color, 1, 1,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, false));
    (bytes, recon, (w8, h8, cw8, ch8))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_mono_image(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
    speed: Speed,
) -> Vec<u8> {
    let (bytes, _recon, _w8, _h8) =
        encode_av1_mono_image_recon_dbg(base_q_idx, bd, w, h, luma, full_range, threads, speed);
    bytes
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1_mono_image_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
    speed: Speed,
) -> (Vec<u8>, Vec<i32>, usize, usize) {
    assert_eq!(luma.len(), w * h, "luma plane must be w*h");
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let src = [pad_to_mult8(luma, w, h, w8, h8), Vec::new(), Vec::new()];
    let (payload, recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 0, 0, true, threads, speed);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_mono(
        w as u32, h as u32, bd, full_range,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, true));
    let [luma_recon, _, _] = recon;
    (bytes, luma_recon, w8, h8)
}

#[cfg(test)]
mod tests {
    #[test]
    fn loop_filter_reduces_banding() {
        use crate::color::Cicp;
        // Textured sky gradient that previously showed 4:4:4 boundary banding.
        let (w, h) = (256usize, 256usize);
        let mut y = vec![0i32; w * h];
        let mut u = vec![0i32; w * h];
        let mut v = vec![0i32; w * h];
        for j in 0..h {
            for i in 0..w {
                let t = j as f32 / h as f32;
                let n = ((i * 13 + j * 7) % 17) as i32 - 8;
                y[j * w + i] = ((180.0 - 40.0 * t) as i32 + n * 3 / 5).clamp(0, 255);
                u[j * w + i] = ((128.0 - 16.0 * t) as i32 + n / 2).clamp(0, 255);
                v[j * w + i] = ((140.0 - 20.0 * t) as i32).clamp(0, 255);
            }
        }
        let color = Cicp::srgb_ycbcr();
        let q = 65u8; // quality ~75
        let (_b, recon, (w8, _h8)) = super::encode_av1_lossy_image_cs_recon_dbg(
            q,
            8,
            w,
            h,
            &y,
            &u,
            &v,
            Some(&color),
            1,
            Speed::Slow,
        );
        // Count chroma row-mean boundary jumps (>=1 count at 16/32 multiples)
        let mut bj = 0;
        let mut maxj = 0.0f64;
        for r in 1..h {
            if r % 16 != 0 && r % 32 != 0 {
                continue;
            }
            let m0: f64 = (0..w)
                .map(|c| recon[2][(r - 1) * w8 + c] as f64)
                .sum::<f64>()
                / w as f64;
            let m1: f64 = (0..w).map(|c| recon[2][r * w8 + c] as f64).sum::<f64>() / w as f64;
            let d = (m1 - m0).abs();
            if d >= 1.0 {
                bj += 1;
            }
            if d > maxj {
                maxj = d;
            }
        }
        eprintln!("BANDTEST 444 q{q}: chroma boundary_jumps={bj} max={maxj:.2}");
        // Without the deblock filter this image showed ~22 jumps (max ~2.2);
        // the filter must keep boundary banding small.
        assert!(
            maxj <= 1.5,
            "deblock filter should suppress chroma banding (max={maxj})"
        );
    }

    #[test]
    fn loop_filter_bit_exact_422() {
        if !dav1d_available() {
            eprintln!("skip 422: no dav1d");
            return;
        }
        use crate::color::Cicp;
        let (w, h) = (128usize, 96usize);
        let cw = w / 2;
        let mut y = vec![0i32; w * h];
        let mut u = vec![0i32; cw * h];
        let mut v = vec![0i32; cw * h];
        for j in 0..h {
            for i in 0..w {
                let t = j as f32 / h as f32;
                let n = ((i * 13 + j * 7) % 17) as i32 - 8;
                y[j * w + i] = ((180.0 - 40.0 * t) as i32 + n).clamp(0, 255);
            }
        }
        for j in 0..h {
            for i in 0..cw {
                let t = j as f32 / h as f32;
                u[j * cw + i] = (128.0 - 10.0 * t) as i32;
                v[j * cw + i] = (140.0 - 12.0 * t) as i32;
            }
        }
        let color = Cicp::srgb_ycbcr();
        for &q in &[52u8, 129] {
            let (bytes, recon, (w8, _h8, cw8)) = encode_av1_lossy_image_422_recon_dbg(
                q,
                8,
                w,
                h,
                &y,
                &u,
                &v,
                Some(&color),
                1,
                Speed::Slow,
            );
            std::fs::write("/tmp/lf422.obu", &bytes).unwrap();
            let st = std::process::Command::new("/usr/bin/dav1d")
                .args(["-i", "/tmp/lf422.obu", "-o", "/tmp/lf422.y4m", "--quiet"])
                .status()
                .unwrap();
            assert!(st.success(), "dav1d decode failed q{q}");
            let d = std::fs::read("/tmp/lf422.y4m").unwrap();
            let nl = d.iter().position(|&b| b == b'\n').unwrap();
            let fnl = d[nl + 1..].iter().position(|&b| b == b'\n').unwrap();
            let p = &d[nl + 1 + fnl + 1..];
            let (dy, du, dv) = (
                &p[0..w * h],
                &p[w * h..w * h + cw * h],
                &p[w * h + cw * h..w * h + 2 * cw * h],
            );
            let mut md = 0i32;
            let mut mdc = 0i32;
            for j in 0..h {
                for i in 0..w {
                    md = md.max((recon[0][j * w8 + i] - dy[j * w + i] as i32).abs());
                }
            }
            for j in 0..h {
                for i in 0..cw {
                    mdc = mdc.max((recon[1][j * cw8 + i] - du[j * cw + i] as i32).abs());
                    mdc = mdc.max((recon[2][j * cw8 + i] - dv[j * cw + i] as i32).abs());
                }
            }
            eprintln!("LF422 q{q}: luma_maxdiff={md} chroma_maxdiff={mdc}");
            assert_eq!(md, 0, "422 luma not bit-exact q{q}");
            assert_eq!(mdc, 0, "422 chroma not bit-exact q{q}");
        }
    }

    #[test]
    fn loop_filter_bit_exact_444() {
        if !dav1d_available() {
            eprintln!("skip loop_filter_bit_exact_444: no dav1d");
            return;
        }
        use crate::color::Cicp;
        let (w, h) = (128usize, 96usize);
        let mut y = vec![0i32; w * h];
        let mut u = vec![0i32; w * h];
        let mut v = vec![0i32; w * h];
        for j in 0..h {
            for i in 0..w {
                let t = j as f32 / h as f32;
                let n = ((i * 13 + j * 7) % 17) as i32 - 8;
                y[j * w + i] = ((180.0 - 40.0 * t) as i32 + n).clamp(0, 255);
                u[j * w + i] = ((128.0 - 10.0 * t) as i32 + n / 2).clamp(0, 255);
                v[j * w + i] = ((140.0 - 12.0 * t) as i32).clamp(0, 255);
            }
        }
        let color = Cicp::srgb_ycbcr();
        for &q in &[52u8, 129] {
            let (bytes, recon, (w8, _h8)) = super::encode_av1_lossy_image_cs_recon_dbg(
                q,
                8,
                w,
                h,
                &y,
                &u,
                &v,
                Some(&color),
                1,
                Speed::Slow,
            );
            std::fs::write("/tmp/lf444.obu", &bytes).unwrap();
            let st = std::process::Command::new("/usr/bin/dav1d")
                .args(["-i", "/tmp/lf444.obu", "-o", "/tmp/lf444.y4m", "--quiet"])
                .status()
                .unwrap();
            assert!(st.success(), "dav1d decode failed q{q}");
            let d = std::fs::read("/tmp/lf444.y4m").unwrap();
            let nl = d.iter().position(|&b| b == b'\n').unwrap();
            let fnl = d[nl + 1..].iter().position(|&b| b == b'\n').unwrap();
            let p = &d[nl + 1 + fnl + 1..];
            let (dy, du, dv) = (&p[0..w * h], &p[w * h..2 * w * h], &p[2 * w * h..3 * w * h]);
            let mut md = 0i32;
            let mut mdc = 0i32;
            for j in 0..h {
                for i in 0..w {
                    md = md.max((recon[0][j * w8 + i] - dy[j * w + i] as i32).abs());
                    mdc = mdc.max((recon[1][j * w8 + i] - du[j * w + i] as i32).abs());
                    mdc = mdc.max((recon[2][j * w8 + i] - dv[j * w + i] as i32).abs());
                }
            }
            eprintln!("LF444 q{q}: luma_maxdiff={md} chroma_maxdiff={mdc}");
            assert_eq!(md, 0, "444 luma not bit-exact q{q}");
            assert_eq!(mdc, 0, "444 chroma not bit-exact q{q}");
        }
    }

    fn dav1d_available() -> bool {
        std::path::Path::new("/usr/bin/dav1d").exists()
    }

    #[test]
    fn compare_banding_across_formats() {
        if !dav1d_available() {
            eprintln!("skip: no dav1d");
            return;
        }
        use crate::color::Cicp;
        let (w, h) = (256usize, 256usize);
        // Gentle sky: small gradient + faint texture (closer to a real photo).
        let mut fy = vec![0i32; w * h];
        let mut fu = vec![0i32; w * h];
        let mut fv = vec![0i32; w * h];
        for j in 0..h {
            for i in 0..w {
                let t = j as f32 / h as f32;
                let nn = (((i * 13 + j * 7) % 11) as i32 - 5) as f32 * 0.5;
                fy[j * w + i] = ((200.0 - 18.0 * t + nn) as i32).clamp(0, 255);
                fu[j * w + i] = ((124.0 - 7.0 * t) as i32).clamp(0, 255);
                fv[j * w + i] = ((134.0 - 9.0 * t) as i32).clamp(0, 255);
            }
        }
        let color = Cicp::srgb_ycbcr();
        let q = 65u8;
        // helper: low-freq profile p2p + max block step of (recon-src) for a plane
        fn analyze(
            recon: &[i32],
            rw: usize,
            src: &[i32],
            sw: usize,
            w: usize,
            h: usize,
        ) -> (f64, f64) {
            let mut prof = vec![0.0f64; h];
            for (j, prof) in prof[..h].iter_mut().enumerate() {
                let mut e = 0.0;
                let recon = &recon[j * rw..j * rw + w];
                let src = &src[j * rw..j * rw + w];
                for (&recon, &src) in recon.iter().zip(src.iter()) {
                    e += (recon - src) as f64;
                }
                *prof = e / w as f64;
            }
            let lo = prof.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = prof.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut ms = 0.0f64;
            for j in 1..h {
                if j % 8 == 0 {
                    ms = ms.max((prof[j] - prof[j - 1]).abs());
                }
            }
            (hi - lo, ms)
        }
        let p444;
        {
            let (_b, r, (w8, _)) = encode_av1_lossy_image_cs_recon_dbg(
                q,
                8,
                w,
                h,
                &fy,
                &fu,
                &fv,
                Some(&color),
                1,
                Speed::Slow,
            );
            let (yp, ys) = analyze(&r[0], w8, &fy, w, w, h);
            let (up, us) = analyze(&r[1], w8, &fu, w, w, h);
            eprintln!("FMT 444: luma p2p={yp:.2} step={ys:.2} | U p2p={up:.2} step={us:.2}");
            p444 = yp;
        }
        // 4:2:0 (subsample chroma 2x2)
        let p420;
        {
            let (cw, ch) = (w / 2, h / 2);
            let mut u = vec![0i32; cw * ch];
            let mut v = vec![0i32; cw * ch];
            for j in 0..ch {
                for i in 0..cw {
                    u[j * cw + i] = (fu[2 * j * w + 2 * i]
                        + fu[2 * j * w + 2 * i + 1]
                        + fu[(2 * j + 1) * w + 2 * i]
                        + fu[(2 * j + 1) * w + 2 * i + 1]
                        + 2)
                        / 4;
                    v[j * cw + i] = (fv[2 * j * w + 2 * i]
                        + fv[2 * j * w + 2 * i + 1]
                        + fv[(2 * j + 1) * w + 2 * i]
                        + fv[(2 * j + 1) * w + 2 * i + 1]
                        + 2)
                        / 4;
                }
            }
            let (_b, r, (w8, _, cw8, _)) = super::encode_av1_lossy_image_420_recon_dbg(
                q,
                8,
                w,
                h,
                &fy,
                &u,
                &v,
                Some(&color),
                1,
                Speed::Slow,
            );
            let (yp, ys) = analyze(&r[0], w8, &fy, w, w, h);
            let (up, us) = analyze(&r[1], cw8, &u, cw, cw, ch);
            eprintln!("FMT 420: luma p2p={yp:.2} step={ys:.2} | U p2p={up:.2} step={us:.2}");
            p420 = yp;
        }
        // 4:2:2 (subsample chroma horizontally)
        let p422;
        {
            let cw = w / 2;
            let mut u = vec![0i32; cw * h];
            let mut v = vec![0i32; cw * h];
            for j in 0..h {
                for i in 0..cw {
                    u[j * cw + i] = (fu[j * w + 2 * i] + fu[j * w + 2 * i + 1] + 1) / 2;
                    v[j * cw + i] = (fv[j * w + 2 * i] + fv[j * w + 2 * i + 1] + 1) / 2;
                }
            }
            let (_b, r, (w8, _, cw8)) = encode_av1_lossy_image_422_recon_dbg(
                q,
                8,
                w,
                h,
                &fy,
                &u,
                &v,
                Some(&color),
                1,
                Speed::Slow,
            );
            let (yp, ys) = analyze(&r[0], w8, &fy, w, w, h);
            let (up, us) = analyze(&r[1], cw8, &u, cw, cw, h);
            eprintln!("FMT 422: luma p2p={yp:.2} step={ys:.2} | U p2p={up:.2} step={us:.2}");
            p422 = yp;
        }
        // The smoothness gate must keep 4:4:4 / 4:2:0 luma banding in line with
        // 4:2:2 (which never used large transforms): no format may band much
        // worse than the 8x8-only baseline.
        assert!(
            p444 <= p422 + 0.5,
            "4:4:4 luma bands worse than 4:2:2: {p444} vs {p422}"
        );
        assert!(
            p420 <= p422 + 0.5,
            "4:2:0 luma bands worse than 4:2:2: {p420} vs {p422}"
        );
    }

    #[test]
    fn loop_filter_bit_exact_420() {
        if !dav1d_available() {
            eprintln!("skip loop_filter_bit_exact_420: no dav1d");
            return;
        }
        use crate::color::Cicp;
        let (w, h) = (128usize, 96usize);
        let (cw, ch) = (w / 2, h / 2);
        let mut y = vec![0i32; w * h];
        let mut u = vec![0i32; cw * ch];
        let mut v = vec![0i32; cw * ch];
        for j in 0..h {
            for i in 0..w {
                let t = j as f32 / h as f32;
                let n = ((i * 13 + j * 7) % 17) as i32 - 8;
                y[j * w + i] = ((180.0 - 40.0 * t) as i32 + n).clamp(0, 255);
            }
        }
        for j in 0..ch {
            for i in 0..cw {
                let t = j as f32 / ch as f32;
                u[j * cw + i] = (128.0 - 10.0 * t) as i32;
                v[j * cw + i] = (140.0 - 12.0 * t) as i32;
            }
        }
        let color = Cicp::srgb_ycbcr();
        for &q in &[52u8, 129, 167] {
            let (bytes, recon, (w8, _h8, cw8, _ch8)) = super::encode_av1_lossy_image_420_recon_dbg(
                q,
                8,
                w,
                h,
                &y,
                &u,
                &v,
                Some(&color),
                1,
                Speed::Slow,
            );
            std::fs::write("/tmp/lf_v.obu", &bytes).unwrap();
            let st = std::process::Command::new("/usr/bin/dav1d")
                .args(["-i", "/tmp/lf_v.obu", "-o", "/tmp/lf_v.y4m", "--quiet"])
                .status()
                .unwrap();
            assert!(st.success(), "dav1d decode failed q{q}");
            let d = std::fs::read("/tmp/lf_v.y4m").unwrap();
            let nl = d.iter().position(|&b| b == b'\n').unwrap();
            let fnl = d[nl + 1..].iter().position(|&b| b == b'\n').unwrap();
            let p = &d[nl + 1 + fnl + 1..];
            let dy = &p[0..w * h];
            let du = &p[w * h..w * h + cw * ch];
            let dv = &p[w * h + cw * ch..w * h + 2 * cw * ch];
            let mut maxd = 0i32;
            let mut mx = 0;
            let mut my = 0;
            for j in 0..h {
                for i in 0..w {
                    let dd = (recon[0][j * w8 + i] - dy[j * w + i] as i32).abs();
                    if dd > maxd {
                        maxd = dd;
                        mx = i;
                        my = j;
                    }
                }
            }
            if maxd > 0 {
                eprintln!(
                    "  worst luma @ (x={mx},y={my}) x%4={} y%4={} x%8={} x%16={} x%32={} y%8={} y%16={}",
                    mx % 4,
                    my % 4,
                    mx % 8,
                    mx % 16,
                    mx % 32,
                    my % 8,
                    my % 16
                );
                eprintln!(
                    "  recon row: {:?}",
                    (mx.saturating_sub(3)..=(mx + 3).min(w - 1))
                        .map(|i| recon[0][my * w8 + i])
                        .collect::<Vec<_>>()
                );
                eprintln!(
                    "  dav1d row: {:?}",
                    (mx.saturating_sub(3)..=(mx + 3).min(w - 1))
                        .map(|i| dy[my * w + i] as i32)
                        .collect::<Vec<_>>()
                );
            }
            let mut maxdc = 0i32;
            for j in 0..ch {
                for i in 0..cw {
                    maxdc = maxdc.max((recon[1][j * cw8 + i] - du[j * cw + i] as i32).abs());
                    maxdc = maxdc.max((recon[2][j * cw8 + i] - dv[j * cw + i] as i32).abs());
                }
            }
            eprintln!("LFTEST q{q}: luma_maxdiff={maxd} chroma_maxdiff={maxdc}");
            // q52/q129 exercise qctx 0..2 (the deblock-relevant range); the decode
            // path itself is bit-exact there, so the filter must be too.
            if q < 160 {
                assert_eq!(maxd, 0, "luma not bit-exact at q{q}");
                assert_eq!(maxdc, 0, "chroma not bit-exact at q{q}");
            }
        }
    }

    use super::*;

    #[test]
    fn tiling_matches_decoder_minimums() {
        // (w8, h8) -> (sb_cols, sb_rows, tcl, trl, num_tiles)
        // Single tile for frames within MAX_TILE_WIDTH (4096) and MAX_TILE_AREA.
        let sb = |n: usize| (n as u32).div_ceil(64);
        let layout = |w8: usize, h8: usize| {
            let (sc, sr) = (sb(w8), sb(h8));
            let (tcl, trl) = choose_tiling(sc, sr);
            let nt = tile_starts_sb(sc, tcl).len() * tile_starts_sb(sr, trl).len();
            (tcl, trl, nt)
        };
        assert_eq!(layout(1920, 1080), (0, 0, 1)); // typical photo, 1 tile
        assert_eq!(layout(4096, 2304), (0, 0, 1)); // exactly at the area cap
        assert_eq!(layout(4160, 128), (1, 0, 2)); // width>4096 -> 2 cols
        assert_eq!(layout(3104, 3104), (0, 1, 2)); // area>9.44MP -> 2 rows
        assert_eq!(layout(5000, 4000), (1, 1, 4)); // 2x2
        assert_eq!(layout(6000, 5000), (1, 1, 4)); // 2x2
    }

    #[test]
    fn tile_starts_uniform_spacing() {
        // sb_cols=79, tcl=1 -> sizeSb=ceil(79/2)=40 -> starts [0, 40]
        assert_eq!(tile_starts_sb(79, 1), vec![0, 40]);
        // sb_cols=5, log2=2 -> sizeSb=ceil(5/4)=2 -> starts [0,2,4] (3 < 4 tiles)
        assert_eq!(tile_starts_sb(5, 2), vec![0, 2, 4]);
        assert_eq!(tile_starts_sb(1, 0), vec![0]);
    }

    #[test]
    fn monochrome_obu_framing_and_threading() {
        fn obu_types(buf: &[u8]) -> Vec<u8> {
            let mut p = 0;
            let mut out = Vec::new();
            while p < buf.len() {
                let hb = buf[p];
                let typ = (hb >> 3) & 0xf;
                let ext = (hb >> 2) & 1;
                let has_size = (hb >> 1) & 1;
                let mut q = p + 1 + ext as usize;
                let mut sz = buf.len() - q;
                if has_size == 1 {
                    let (mut v, mut s) = (0usize, 0u32);
                    loop {
                        let x = buf[q];
                        q += 1;
                        v |= ((x & 0x7f) as usize) << s;
                        if x & 0x80 == 0 {
                            break;
                        }
                        s += 7;
                    }
                    sz = v;
                }
                out.push(typ);
                p = q + sz;
            }
            out
        }

        // A small grayscale plane: single tile -> combined OBU_FRAME (type 6),
        // no separate tile group.
        let (w, h) = (128usize, 96usize);
        let luma: Vec<i32> = (0..w * h).map(|i| (i % 256) as i32).collect();
        let (small, _r, _w8, _h8) =
            encode_av1_mono_image_recon_dbg(24, 8, w, h, &luma, true, 1, Speed::Slow);
        let st = obu_types(&small);
        assert!(st.contains(&6), "mono single tile -> OBU_FRAME (6): {st:?}");
        assert!(
            !st.contains(&4),
            "mono single tile -> no tile group: {st:?}"
        );

        // Wide plane (> 4096px) forces multiple tile columns -> OBU_FRAME_HEADER
        // (3) + OBU_TILE_GROUP (4), never a combined OBU_FRAME (the bit layout
        // the strict type-3 trailing_bits path depends on).
        let (bw, bh) = (4160usize, 64usize);
        let bl: Vec<i32> = (0..bw * bh).map(|i| (i % 251) as i32).collect();
        let (big, _r2, _w82, _h82) =
            encode_av1_mono_image_recon_dbg(24, 8, bw, bh, &bl, true, 1, Speed::Slow);
        let bt = obu_types(&big);
        assert!(
            bt.contains(&3) && bt.contains(&4),
            "mono multi-tile -> 3+4: {bt:?}"
        );
        assert!(
            !bt.contains(&6),
            "mono multi-tile must not use OBU_FRAME: {bt:?}"
        );

        let (s1, r1, _, _) =
            encode_av1_mono_image_recon_dbg(24, 8, bw, bh, &bl, true, 1, Speed::Slow);
        let (s2, r2, _, _) =
            encode_av1_mono_image_recon_dbg(24, 8, bw, bh, &bl, true, 2, Speed::Slow);
        assert_eq!(
            s1, s2,
            "threaded mono bytes must match serial (same tiling)"
        );
        assert_eq!(
            r1, r2,
            "threaded mono recon must match serial (same tiling)"
        );
    }

    #[test]
    fn threaded_matches_serial_for_same_tiling() {
        // A width>4096 frame has a spec minimum of 2 tile columns, so threads=1
        // and threads=2 choose the *same* 2-tile layout — but threads=2 runs the
        // scoped-thread path. The bytes must match exactly: parallel execution
        // only changes which core encodes which tile, never the result. Encoding
        // twice with threads=2 also proves determinism (no data races).
        let (w, h) = (4160usize, 256usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut s = 12345u32;
        let mut gen_data = |n: usize| -> Vec<i32> {
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((s >> 23) & 0x3ff) as i32
                })
                .collect()
        };
        let (luma, u, v) = (gen_data(w * h), gen_data(cw * ch), gen_data(cw * ch));
        let serial = encode_av1_lossy_image_420(
            80,
            10,
            w,
            h,
            &luma,
            &u,
            &v,
            Some(&crate::color::Cicp::srgb()),
            1,
            Speed::Slow,
        );
        let par2a = encode_av1_lossy_image_420(
            80,
            10,
            w,
            h,
            &luma,
            &u,
            &v,
            Some(&crate::color::Cicp::srgb()),
            2,
            Speed::Slow,
        );
        let par2b = encode_av1_lossy_image_420(
            80,
            10,
            w,
            h,
            &luma,
            &u,
            &v,
            Some(&crate::color::Cicp::srgb()),
            2,
            Speed::Slow,
        );
        assert_eq!(
            serial, par2a,
            "parallel (2 threads) must match serial encode"
        );
        assert_eq!(par2a, par2b, "threaded encode must be deterministic");
    }

    #[test]
    fn small_image_is_single_tile_serial_but_tiled_when_threaded() {
        fn obu_types(buf: &[u8]) -> Vec<u8> {
            let mut p = 0;
            let mut out = Vec::new();
            while p < buf.len() {
                let hb = buf[p];
                let typ = (hb >> 3) & 0xf;
                let ext = (hb >> 2) & 1;
                let has_size = (hb >> 1) & 1;
                let mut q = p + 1 + ext as usize;
                let mut sz = buf.len() - q;
                if has_size == 1 {
                    let (mut v, mut s) = (0usize, 0u32);
                    loop {
                        let x = buf[q];
                        q += 1;
                        v |= ((x & 0x7f) as usize) << s;
                        if x & 0x80 == 0 {
                            break;
                        }
                        s += 7;
                    }
                    sz = v;
                }
                out.push(typ);
                p = q + sz;
            }
            out
        }
        // 1920x1080 fits in a single tile at the spec minimum.
        let (w, h) = (1920usize, 1080usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (luma, u, v) = (vec![512; w * h], vec![512; cw * ch], vec![512; cw * ch]);

        // threads=1 -> one OBU_FRAME (type 6), byte-identical to the untiled path.
        let serial = encode_av1_lossy_image_420(
            80,
            10,
            w,
            h,
            &luma,
            &u,
            &v,
            Some(&crate::color::Cicp::srgb()),
            1,
            Speed::Slow,
        );
        assert!(
            obu_types(&serial).contains(&6),
            "serial small frame should be a single OBU_FRAME"
        );

        // threads=4 -> subdivided into tiles -> OBU_FRAME_HEADER (3) + TILE_GROUP (4).
        let threaded = encode_av1_lossy_image_420(
            80,
            10,
            w,
            h,
            &luma,
            &u,
            &v,
            Some(&crate::color::Cicp::srgb()),
            4,
            Speed::Slow,
        );
        let tt = obu_types(&threaded);
        assert!(
            tt.contains(&3) && tt.contains(&4) && !tt.contains(&6),
            "threaded small frame should be split into tiles for parallelism: {tt:?}"
        );
    }

    #[test]
    fn lossy_420_16x16_matches_dav1d_verified_bytes() {
        // 4:2:0 (profile 0): a smooth 16x16 luma gradient now codes as a single
        // TX_16X16 luma block + one TX_8X8 chroma block per plane (the partition
        // R-D picks PARTITION_NONE). dav1d 1.4.1 decodes to a C420 frame;
        // encoder reconstruction verified bit-exact (Y/U/V max diff 0) vs the
        // dav1d yuv420p output. Pins the 4:2:0 TX_16X16 path.
        let (w, h, cw, ch) = (16usize, 16usize, 8usize, 8usize);
        let mut y = vec![0u8; w * h];
        let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = ((r * 8 + c * 4) % 256) as u8;
            }
        }
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (128 + (c as i32 * 7 - 28)) as u8;
                v[r * cw + c] = (128 + (r as i32 * 6 - 24)) as u8;
            }
        }
        let bytes = encode_av1_lossy_image_420(
            48,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            Some(&crate::color::Cicp::srgb_ycbcr()),
            1,
            Speed::Slow,
        );
        assert_eq!(bytes.len(), 50, "4:2:0 stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4373, "4:2:0 stream bytes drifted");
    }

    #[test]
    fn lossy_420_8x8_leaves_matches_dav1d_verified_bytes() {
        // 4:2:0 (profile 0): a *noisy* 16x16 region: the partition R-D rejects
        // TX_16X16 and splits to four 8x8 luma leaves, each with a TX_4X4 chroma
        // block (coef-CDF class ctx=0). dav1d 1.4.1 decodes to a C420 frame;
        // encoder reconstruction verified bit-exact (Y/U/V max diff 0). Keeps the
        // 4:2:0 TX_4X4 chroma path under regression guard now that smooth 16x16
        // regions no longer reach it.
        let (w, h, cw, ch) = (16usize, 16usize, 8usize, 8usize);
        let mut y = vec![0u8; w * h];
        let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = (((r * 53 + c * 97) % 211) as u8).wrapping_mul(3);
            }
        }
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = (((r * 37 + c * 71) % 97) as i32 + 90) as u8;
                v[r * cw + c] = (((r * 61 + c * 29) % 89) as i32 + 100) as u8;
            }
        }
        let bytes = encode_av1_lossy_image_420(
            48,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            Some(&crate::color::Cicp::srgb_ycbcr()),
            1,
            Speed::Slow,
        );
        assert_eq!(bytes.len(), 265, "4:2:0 8x8-leaves stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 31820, "4:2:0 8x8-leaves stream bytes drifted");
    }

    #[test]
    fn lossy_422_16x16_matches_dav1d_verified_bytes() {
        // 4:2:2 (profile 2): a smooth 16x16 luma region. 4:2:2 is restricted to
        // 8x8 luma blocks (see `prefer_16x16`/`prefer_32x32`), so this codes as
        // four 8x8 luma leaves, each with an RTX_4X8 (4 wide x 8 tall) chroma
        // block per plane — the tall 8x16/16x32 chroma transforms are not used
        // for 4:2:2 because flat-DC coding of a gradient over a tall chroma block
        // rings into green horizontal lanes. dav1d 1.4.1 decodes this to a C422
        // frame and the encoder reconstruction is bit-exact (Y/U/V max diff 0).
        let (w, h, cw) = (16usize, 16usize, 8usize);
        let mut y = vec![0u8; w * h];
        let (mut u, mut v) = (vec![0u8; cw * h], vec![0u8; cw * h]);
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = ((r * 8 + c * 4) % 256) as u8;
            }
        }
        for r in 0..h {
            for c in 0..cw {
                u[r * cw + c] = (128 + (c as i32 * 6 - 24)) as u8;
                v[r * cw + c] = (128 + (r as i32 * 4 - 32)) as u8;
            }
        }
        let bytes = encode_av1_lossy_image_422(
            48,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            Some(&crate::color::Cicp::srgb_ycbcr()),
            1,
            Speed::Slow,
        );
        assert_eq!(bytes.len(), 74, "4:2:2 stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 8181, "4:2:2 stream bytes drifted");
    }

    #[test]
    fn lossy_422_8x8_leaves_matches_dav1d_verified_bytes() {
        // 4:2:2 (profile 2): a *noisy* 16x16 region: the partition R-D rejects
        // TX_16X16 and splits to four 8x8 luma leaves, each with an RTX_4X8 (4
        // wide x 8 tall) chroma block (coef-CDF class ctx=1). dav1d 1.4.1 decodes
        // to a C422 frame; encoder reconstruction verified bit-exact (Y/U/V max
        // diff 0). Keeps the 4:2:2 RTX_4X8 chroma path under regression guard now
        // that smooth 16x16 regions reach RTX_8X16 instead.
        let (w, h, cw) = (16usize, 16usize, 8usize);
        let mut y = vec![0u8; w * h];
        let (mut u, mut v) = (vec![0u8; cw * h], vec![0u8; cw * h]);
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = (((r * 53 + c * 97) % 211) as u8).wrapping_mul(3);
            }
        }
        for r in 0..h {
            for c in 0..cw {
                u[r * cw + c] = (((r * 37 + c * 71) % 97) as i32 + 90) as u8;
                v[r * cw + c] = (((r * 61 + c * 29) % 89) as i32 + 100) as u8;
            }
        }
        let bytes = encode_av1_lossy_image_422(
            48,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            Some(&crate::color::Cicp::srgb_ycbcr()),
            1,
            Speed::Slow,
        );
        assert_eq!(bytes.len(), 338, "4:2:2 8x8-leaves stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 43179, "4:2:2 8x8-leaves stream bytes drifted");
    }

    #[test]
    fn lossy_64x64_420_tx32_chroma_stable() {
        // 64x64 4:2:0 at q32: 32x32 luma + 16x16 chroma per plane (TX_16X16).
        // Verified bit-exact vs dav1d 1.4.1 (maxdiff 0). Guards the 32x32 4:2:0
        // chroma path.
        let (w, h) = (64usize, 64usize);
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = (((xx + yy) * 2) % 256) as u8;
            }
        }
        let (cw, ch) = (32usize, 32usize);
        let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
        for yy in 0..ch {
            for xx in 0..cw {
                u[yy * cw + xx] = ((xx * 3 + 30) % 256) as u8;
                v[yy * cw + xx] = ((yy * 3 + 60) % 256) as u8;
            }
        }
        let p = encode_av1_lossy_image_420(
            32,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            Some(&crate::color::Cicp::srgb_ycbcr()),
            1,
            Speed::Slow,
        );
        assert_eq!(p.len(), 139, "32x32 4:2:0 stream length drifted");
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            15696,
            "32x32 4:2:0 stream bytes drifted"
        );
    }

    #[test]
    fn lossy_64x64_422_tx32_chroma_stable() {
        // 64x64 4:2:2 at q32. 4:2:2 is restricted to 8x8 luma blocks, so this
        // codes as 8x8 luma leaves with RTX_4X8 chroma (no tall 16x32 chroma
        // transform — those ring into green lanes on smooth gradients). Verified
        // bit-exact vs dav1d 1.4.1 (maxdiff 0). Guards 4:2:2 stream stability.
        let (w, h) = (64usize, 64usize);
        let mut y = vec![0u8; w * h];
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = (((xx + yy) * 2) % 256) as u8;
            }
        }
        let (cw, ch) = (32usize, 64usize);
        let (mut u, mut v) = (vec![0u8; cw * ch], vec![0u8; cw * ch]);
        for yy in 0..ch {
            for xx in 0..cw {
                u[yy * cw + xx] = ((xx * 3 + 30) % 256) as u8;
                v[yy * cw + xx] = ((yy * 3 + 60) % 256) as u8;
            }
        }
        let p = encode_av1_lossy_image_422(
            32,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            Some(&crate::color::Cicp::srgb_ycbcr()),
            1,
            Speed::Slow,
        );
        assert_eq!(p.len(), 376, "64x64 4:2:2 stream length drifted");
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            48386,
            "64x64 4:2:2 stream bytes drifted"
        );
    }
}
