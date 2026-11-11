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

// Adaptive CDF state (`Cdfs`) and the emitted all-zero statistics behind the adaptive skip price.

/// Emitted `all_zero` symbol counts `[tx class][ctx][symbol]`.
pub(crate) type SkipStats = [[[u32; 2]; 13]; 5];

pub(crate) fn skip_stats_add(acc: &mut SkipStats, a: &SkipStats) {
    for (ac, ar) in acc.iter_mut().zip(a.iter()) {
        for (bc, br) in ac.iter_mut().zip(ar.iter()) {
            bc[0] += br[0];
            bc[1] += br[1];
        }
    }
}

pub(crate) fn skip_stats_sub(a: &SkipStats, b: &SkipStats) -> SkipStats {
    let mut out = [[[0u32; 2]; 13]; 5];
    for cls in 0..5 {
        for ctx in 0..13 {
            out[cls][ctx][0] = a[cls][ctx][0] - b[cls][ctx][0];
            out[cls][ctx][1] = a[cls][ctx][1] - b[cls][ctx][1];
        }
    }
    out
}

/// Deterministic all-zero statistics available to the decision of SB `(row, col)`:
/// every SB the wavefront guarantees finished (left, top and above-right
/// dependencies, transitively `(row - k, <= col + k)`), summed from per-row
/// prefix sums. The serial path uses the same cone so `-t1 == -tN` holds.
pub(crate) fn skip_stats_cone(
    prefix: &[SkipStats],
    row: usize,
    col: usize,
    cols: usize,
) -> SkipStats {
    let mut acc = [[[0u32; 2]; 13]; 5];
    if col > 0 {
        skip_stats_add(&mut acc, &prefix[row * cols + col - 1]);
    }
    for k in 1..=row {
        let c = (col + k).min(cols - 1);
        skip_stats_add(&mut acc, &prefix[(row - k) * cols + c]);
    }
    acc
}

/// Record SB `(row, col)`'s own all-zero counts into the per-row prefix sums
/// [`skip_stats_cone`] reads: its slot becomes its left neighbour's slot plus
/// `own`.
pub(crate) fn skip_prefix_store(
    prefix: &mut [SkipStats],
    row: usize,
    col: usize,
    cols: usize,
    own: &SkipStats,
) {
    let i = row * cols + col;
    let mut acc = if col > 0 {
        prefix[i - 1]
    } else {
        [[[0u32; 2]; 13]; 5]
    };
    skip_stats_add(&mut acc, own);
    prefix[i] = acc;
}

/// dav1d default `txpart` (inter `txfm_split`) CDFs, [cat][a + l].
const TXPART_CDF: [[u16; 3]; 7] = [
    [28581, 23846, 20847],
    [24315, 18196, 12133],
    [18791, 10887, 11005],
    [27179, 20004, 11281],
    [26549, 19308, 14224],
    [28015, 21546, 14400],
    [28165, 22401, 16088],
];

#[derive(Clone)]
pub(crate) struct Cdfs {
    /// Trellis frequency-tilt strength for this tile's format — see
    /// `trellis_tilt_mag_cap()` in trellis.rs.
    pub(crate) band_tilt: f32,
    pub(crate) skip: Vec<Vec<u16>>,             // block skip [3 ctx]
    pub(crate) intrabc: Vec<u16>,               // use_intrabc
    pub(crate) txpart: Vec<Vec<Vec<u16>>>,      // inter txfm_split [7 cat][3 ctx]
    pub(crate) txtp_inter2: Vec<u16>,           // inter tx type, set 2 (TX_16X16): 12 types
    pub(crate) txtp_inter3: Vec<Vec<u16>>, // inter tx type, set 3 (TX_32X32) [t_dim.min]: DCT vs IDTX
    pub(crate) mv_joint: Vec<u16>,         // MV_JOINT
    pub(crate) mv_sign: [Vec<u16>; 2],     // vertical, horizontal
    pub(crate) mv_classes: [Vec<u16>; 2],  // MV class 0..10
    pub(crate) mv_class0: [Vec<u16>; 2],   // class-zero integer bit
    pub(crate) mv_class_n: [[Vec<u16>; 10]; 2], // class-N integer bits
    pub(crate) part_bl8: Vec<Vec<u16>>,    // PARTITION_NONE @ 8x8 [4 ctx]
    pub(crate) part_split: Vec<Vec<Vec<u16>>>, // SPLIT [bl-1=0..3][4 ctx]
    pub(crate) kf_y: Vec<Vec<u16>>,        // kf_y_mode[5*5], index [above_ctx*5 + left_ctx]
    pub(crate) uv_mode: Vec<Vec<u16>>,     // uv_mode[2*13], index [cfl_allowed*13 + y_mode]
    pub(crate) angle_delta: Vec<Vec<u16>>, // angle_delta[8 directional modes]
    pub(crate) filter_intra: Vec<Vec<u16>>, // use_filter_intra [BLOCK_SIZES_ALL]
    pub(crate) filter_intra_mode: Vec<u16>, // five filter-intra predictors
    pub(crate) palette_y_mode: Vec<Vec<Vec<u16>>>, // [7 bsize ctx][3 neighbor ctx]
    pub(crate) palette_y_size: Vec<Vec<u16>>, // [7 bsize ctx], sizes 2..8
    pub(crate) palette_uv_mode: [Vec<u16>; 2], // luma palette absent/present
    pub(crate) palette_y_color: Vec<Vec<Vec<u16>>>, // [size-2][5 map ctx]
    pub(crate) palette_uv_size: Vec<Vec<u16>>, // [7 bsize ctx], sizes 2..8
    pub(crate) palette_uv_color: Vec<Vec<Vec<u16>>>, // [size-2][5 map ctx]
    pub(crate) cfl_sign: Vec<u16>,         // cfl joint-sign (8 symbols)
    pub(crate) cfl_alpha: Vec<Vec<u16>>,   // cfl alpha magnitude [6 ctx]
    pub(crate) txsz: [Vec<Vec<u16>>; 4],   // intra tx_depth [t_dim.max-1][3 ctx]
    pub(crate) txtp: Vec<Vec<u16>>,        // intra txtp TX_8X8 luma, per intra mode [13]
    pub(crate) txtp4: Vec<Vec<u16>>,       // intra txtp TX_4X4 luma, per intra mode [13]
    pub(crate) txtp16: Vec<Vec<u16>>,      // intra txtp TX_16X16 luma, per intra mode [13]
    pub(crate) txb_skip: [Vec<Vec<u16>>; 5], // [class][13 ctx] (class 3 = TX_32X32, 4 = TX_64X64)
    /// Running count of emitted `all_zero` symbols `[class][ctx][symbol]` (see
    /// `SkipStats`); the superblock drivers difference it per SB.
    pub(crate) skip_stats: SkipStats,
    pub(crate) base_tok: [[Vec<Vec<u16>>; 2]; 5], // [class][plane][41/42 ctx]
    pub(crate) br_tok: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][21 ctx]; TX_64X64 shares class 3 (dav1d min(ctx, 3))
    pub(crate) eob_base: [[Vec<Vec<u16>>; 2]; 5], // [class][plane][4 ctx]
    pub(crate) eob_hi: [[Vec<Vec<u16>>; 2]; 5], // [class][plane][11 bins], each a 2-sym CDF
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
    /// Code one `all_zero` flag and count it for the adaptive skip price.
    #[inline]
    pub(crate) fn encode_skip(&mut self, enc: &mut OdEcEncoder, cls: usize, ctx: usize, s: usize) {
        self.skip_stats[cls][ctx][s] += 1;
        enc.encode_symbol(s, &mut self.txb_skip[cls][ctx]);
    }
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
        for cat in &mut self.txpart {
            for v in cat {
                push(v);
            }
        }
        push(&mut self.txtp_inter2);
        for v in &mut self.txtp_inter3 {
            push(v);
        }
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

    /// Frozen frame-initial snapshot used for decision-side rate estimates.
    pub(crate) fn decision_snapshot(qctx: usize) -> Box<Self> {
        Box::new(Self::new(qctx))
    }

    /// 4:2:2 control: its photo ladder regressed with the size-specific prior,
    /// so retain the historical neutral decision price for this format.
    pub(crate) fn decision_snapshot_422(qctx: usize) -> Box<Self> {
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
            Q::SKIP_TX64[qctx]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect::<Vec<_>>(),
        ];
        // Class 4 (TX_64X64) is luma-only: chroma transforms clamp to 32, so
        // its chroma slot is an unused placeholder.
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
            [rows(&Q::BASE_TOK_TX64_LUMA_Q[qctx]), Vec::new()],
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
            [rows2(&Q::EOB_BASE_TX64_LUMA_Q[qctx]), Vec::new()],
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
            [his(&Q::EOB_HI_TX64_LUMA[qctx]), Vec::new()],
        ];
        Cdfs {
            band_tilt: 0.0,
            skip: SKIP_CDF.iter().map(|&v| icdf(&[v])).collect(),
            intrabc: icdf(&[30531]),
            txpart: TXPART_CDF
                .iter()
                .map(|cat| cat.iter().map(|&v| icdf(&[v])).collect())
                .collect(),
            txtp_inter2: icdf(&[
                770, 2421, 5225, 12907, 15819, 18927, 21561, 24089, 26595, 28526, 30529,
            ]),
            txtp_inter3: [16384u16, 4167, 1998, 748]
                .iter()
                .map(|&v| icdf(&[v]))
                .collect(),
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
            skip_stats: [[[0u32; 2]; 13]; 5],
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
