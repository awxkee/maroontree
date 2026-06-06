//! Minimal *real-AV1* still-image encoder for a single 64x64 intra keyframe.
//!
//! This is the first path that targets actual dav1d decodability (distinct from
//! the self-consistent codec in `encoder.rs`). The tile codes exactly one
//! 64x64 block: PARTITION_NONE, skip=1 (no residual), DC_PRED luma and chroma.
//! With skip and no neighbors, the decoded image is flat mid-grey — not our
//! input, but a genuine AV1 frame that proves the whole stack (framing, headers,
//! `od_ec`, mode syntax, default CDFs) is decoder-valid.
//!
//! CDF values are the AV1 spec defaults, taken from dav1d's `cdf.c`
//! (`CDFn(a..) = 32768 - a`). At the single top-left block every context is 0,
//! so we need only: partition[64x64][0], skip[0], kf_y_mode[0][0], uv_mode[0][0].

use crate::dct::{
    dct4x4_t, dct4x8_t, dct8x8, dct8x8_t, dct8x16_t, dct16x16, dct16x16_t, dct16x32_t, dct32x32,
    dct32x32_t,
};
use crate::idct::{
    idct_dequant_4x4, idct_dequant_4x8, idct_dequant_8x8, idct_dequant_8x16, idct_dequant_16x16,
    idct_dequant_16x32, idct_dequant_32x32,
};
use crate::obu::{
    frame_header_lossy_multitile, frame_header_lossy_multitile_th, temporal_delimiter,
    wrap_obu_frame, wrap_obu_frame_split,
};
use crate::odec::OdEcEncoder;

/// Build a length-N inverse CDF (rav1e/od_ec layout) from dav1d's `CDFn` args.
/// `args` are the N-1 direct cumulative boundaries of an N-symbol model; the
/// returned array is `[32768-a_0, .., 32768-a_{N-2}, count=0]`.
fn icdf(args: &[u16]) -> Vec<u16> {
    let mut v: Vec<u16> = args.iter().map(|&a| 32768 - a).collect();
    v.push(0); // adaptation counter (also acts as the terminal, since 0>>6==0)
    v
}

/// Block-level skip flag CDF, indexed by ctx = above_skip + left_skip (0..2).
/// (dav1d `default_cdf.m.skip`.)
static SKIP_CDF: [u16; 3] = [31671, 16515, 4576];

/// Per-frame adaptive CDF state. dav1d adapts every symbol's CDF as it decodes
/// (when `disable_cdf_update = 0`); to stay bit-exact we hold the same mutable
/// CDFs (initialised from the qcat defaults, in `icdf` form with a trailing
/// adaptation count) and adapt them identically after each coded symbol via
/// `OdEcEncoder::encode_symbol`. Coef class index: 0 = `TX_4X4` (4:2:0 chroma),
/// 1 = `TX_8X8`/`RTX_4X8` (luma, 4:4:4 and 4:2:2 chroma). Plane index: 0 = luma,
/// 1 = chroma.
struct Cdfs {
    skip: Vec<Vec<u16>>,               // block skip [3 ctx]
    part_bl8: Vec<Vec<u16>>,           // PARTITION_NONE @ 8x8 [4 ctx]
    part_split: Vec<Vec<Vec<u16>>>,    // SPLIT [bl-1=0..3][4 ctx]
    kf_y: Vec<Vec<u16>>,               // kf_y_mode[5*5], index [above_ctx*5 + left_ctx]
    uv_mode: Vec<Vec<u16>>,            // uv_mode[2*13], index [cfl_allowed*13 + y_mode]
    angle_delta: Vec<Vec<u16>>,        // angle_delta[8 directional modes]
    cfl_sign: Vec<u16>,                // cfl joint-sign (8 symbols)
    cfl_alpha: Vec<Vec<u16>>,          // cfl alpha magnitude [6 ctx]
    txtp: Vec<Vec<u16>>,               // intra txtp TX_8X8 luma, per intra mode [13]
    txtp16: Vec<Vec<u16>>,             // intra txtp TX_16X16 luma, per intra mode [13]
    txb_skip: [Vec<Vec<u16>>; 4],      // [class][13 ctx] (class 3 = TX_32X32)
    base_tok: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][41/42 ctx]
    br_tok: [[Vec<Vec<u16>>; 2]; 4],   // [class][plane][21 ctx]
    eob_base: [[Vec<Vec<u16>>; 2]; 4], // [class][plane][4 ctx]
    eob_hi: [[Vec<Vec<u16>>; 2]; 4],   // [class][plane][11 bins], each a 2-sym CDF
    dc_sign: [Vec<Vec<u16>>; 2],       // [plane][3 ctx]
    eob_bin_16_c: Vec<u16>,            // chroma, 4x4
    eob_bin_32_c: Vec<u16>,            // chroma, 4x8
    eob_bin_64_l: Vec<u16>,            // luma, 8x8
    eob_bin_64_c: Vec<u16>,            // chroma, 8x8
    eob_bin_256_l: Vec<u16>,           // luma, 16x16 (class 2)
    eob_bin_256_c: Vec<u16>,           // chroma, 16x16 (class 2)
    eob_bin_128_c: Vec<u16>,           // chroma, RTX_8X16 (class 2, 128 coeffs)
    eob_bin_1024_l: Vec<u16>,          // luma, 32x32 (class 3, 1024 coeffs)
    eob_bin_1024_c: Vec<u16>,          // chroma, 32x32 (class 3, 1024 coeffs)
    eob_bin_512_c: Vec<u16>,           // chroma, RTX_16X32 (class 3, 512 coeffs)
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

/// `default_kf_y_mode_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]` (libaom), the
/// keyframe luma intra-mode CDFs indexed by `[above_ctx][left_ctx]` (each ctx is
/// `INTRA_MODE_CTX[neighbour_mode]`). `[0][0]` equals the former single
/// `kf_y_mode_dc_dc()` CDF, so all-DC output is unchanged.
pub(crate) static KF_Y_MODE_CDF: [[[u16; 12]; 5]; 5] = [
    [
        [
            15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466,
        ],
        [
            12016, 18066, 19516, 20303, 20719, 21444, 21888, 23032, 24434, 28658, 30172, 31409,
        ],
        [
            10052, 10771, 22296, 22788, 23055, 23239, 24133, 25620, 26160, 29336, 29929, 31567,
        ],
        [
            14091, 15406, 16442, 18808, 19136, 19546, 19998, 22096, 24746, 29585, 30958, 32462,
        ],
        [
            12122, 13265, 15603, 16501, 18609, 20033, 22391, 25583, 26437, 30261, 31073, 32475,
        ],
    ],
    [
        [
            10023, 19585, 20848, 21440, 21832, 22760, 23089, 24023, 25381, 29014, 30482, 31436,
        ],
        [
            5983, 24099, 24560, 24886, 25066, 25795, 25913, 26423, 27610, 29905, 31276, 31794,
        ],
        [
            7444, 12781, 20177, 20728, 21077, 21607, 22170, 23405, 24469, 27915, 29090, 30492,
        ],
        [
            8537, 14689, 15432, 17087, 17408, 18172, 18408, 19825, 24649, 29153, 31096, 32210,
        ],
        [
            7543, 14231, 15496, 16195, 17905, 20717, 21984, 24516, 26001, 29675, 30981, 31994,
        ],
    ],
    [
        [
            12613, 13591, 21383, 22004, 22312, 22577, 23401, 25055, 25729, 29538, 30305, 32077,
        ],
        [
            9687, 13470, 18506, 19230, 19604, 20147, 20695, 22062, 23219, 27743, 29211, 30907,
        ],
        [
            6183, 6505, 26024, 26252, 26366, 26434, 27082, 28354, 28555, 30467, 30794, 32086,
        ],
        [
            10718, 11734, 14954, 17224, 17565, 17924, 18561, 21523, 23878, 28975, 30287, 32252,
        ],
        [
            9194, 9858, 16501, 17263, 18424, 19171, 21563, 25961, 26561, 30072, 30737, 32463,
        ],
    ],
    [
        [
            12602, 14399, 15488, 18381, 18778, 19315, 19724, 21419, 25060, 29696, 30917, 32409,
        ],
        [
            8203, 13821, 14524, 17105, 17439, 18131, 18404, 19468, 25225, 29485, 31158, 32342,
        ],
        [
            8451, 9731, 15004, 17643, 18012, 18425, 19070, 21538, 24605, 29118, 30078, 32018,
        ],
        [
            7714, 9048, 9516, 16667, 16817, 16994, 17153, 18767, 26743, 30389, 31536, 32528,
        ],
        [
            8843, 10280, 11496, 15317, 16652, 17943, 19108, 22718, 25769, 29953, 30983, 32485,
        ],
    ],
    [
        [
            12578, 13671, 15979, 16834, 19075, 20913, 22989, 25449, 26219, 30214, 31150, 32477,
        ],
        [
            9563, 13626, 15080, 15892, 17756, 20863, 22207, 24236, 25380, 29653, 31143, 32277,
        ],
        [
            8356, 8901, 17616, 18256, 19350, 20106, 22598, 25947, 26466, 29900, 30523, 32261,
        ],
        [
            10835, 11815, 13124, 16042, 17018, 18039, 18947, 22753, 24615, 29489, 30883, 32482,
        ],
        [
            7618, 8288, 9859, 10509, 15386, 18657, 22903, 28776, 29180, 31355, 31802, 32593,
        ],
    ],
];

/// `default_uv_mode_cdf[0]` (CfL disallowed), indexed by luma `y_mode`; 13-symbol
/// CDFs. Row 0 equals the former `uv_mode_nocfl_dc()`.
pub(crate) static UV_MODE_NOCFL_CDF: [[u16; 12]; 13] = [
    [
        22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923, 28244, 30059, 30941, 31961,
    ],
    [
        9513, 26881, 26973, 27046, 27118, 27664, 27739, 27824, 28359, 29505, 29800, 31796,
    ],
    [
        9845, 9915, 28663, 28704, 28757, 28780, 29198, 29822, 29854, 30764, 31777, 32029,
    ],
    [
        13639, 13897, 14171, 25331, 25606, 25727, 25953, 27148, 28577, 30612, 31355, 32493,
    ],
    [
        9764, 9835, 9930, 9954, 25386, 27053, 27958, 28148, 28243, 31101, 31744, 32363,
    ],
    [
        11825, 13589, 13677, 13720, 15048, 29213, 29301, 29458, 29711, 31161, 31441, 32550,
    ],
    [
        14175, 14399, 16608, 16821, 17718, 17775, 28551, 30200, 30245, 31837, 32342, 32667,
    ],
    [
        12885, 13038, 14978, 15590, 15673, 15748, 16176, 29128, 29267, 30643, 31961, 32461,
    ],
    [
        12026, 13661, 13874, 15305, 15490, 15726, 15995, 16273, 28443, 30388, 30767, 32416,
    ],
    [
        19052, 19840, 20579, 20916, 21150, 21467, 21885, 22719, 23174, 28861, 30379, 32175,
    ],
    [
        18627, 19649, 20974, 21219, 21492, 21816, 22199, 23119, 23527, 27053, 31397, 32148,
    ],
    [
        17026, 19004, 19997, 20339, 20586, 21103, 21349, 21907, 22482, 25896, 26541, 31819,
    ],
    [
        12124, 13759, 14959, 14992, 15007, 15051, 15078, 15166, 15255, 15753, 16039, 16606,
    ],
];

/// `default_uv_mode_cdf[1]` (CfL allowed), indexed by luma `y_mode`; 14-symbol
/// CDFs. Row 0 equals the former `uv_mode_cfl_dc()`.
static UV_MODE_CFL_CDF: [[u16; 13]; 13] = [
    [
        10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656, 15986, 20086, 20995, 22455, 24212,
    ],
    [
        4532, 19780, 20057, 20215, 20428, 21071, 21199, 21451, 22099, 24228, 24693, 27032, 29472,
    ],
    [
        5273, 5379, 20177, 20270, 20385, 20439, 20949, 21695, 21774, 23138, 24256, 24703, 26679,
    ],
    [
        6740, 7167, 7662, 14152, 14536, 14785, 15034, 16741, 18371, 21520, 22206, 23389, 24182,
    ],
    [
        4987, 5368, 5928, 6068, 19114, 20315, 21857, 22253, 22411, 24911, 25380, 26027, 26376,
    ],
    [
        5370, 6889, 7247, 7393, 9498, 21114, 21402, 21753, 21981, 24780, 25386, 26517, 27176,
    ],
    [
        4816, 4961, 7204, 7326, 8765, 8930, 20169, 20682, 20803, 23188, 23763, 24455, 24940,
    ],
    [
        6608, 6740, 8529, 9049, 9257, 9356, 9735, 18827, 19059, 22336, 23204, 23964, 24793,
    ],
    [
        5998, 7419, 7781, 8933, 9255, 9549, 9753, 10417, 18898, 22494, 23139, 24764, 25989,
    ],
    [
        10660, 11298, 12550, 12957, 13322, 13624, 14040, 15004, 15534, 20714, 21789, 23443, 24861,
    ],
    [
        10522, 11530, 12552, 12963, 13378, 13779, 14245, 15235, 15902, 20102, 22696, 23774, 25838,
    ],
    [
        10099, 10691, 12639, 13049, 13386, 13665, 14125, 15163, 15636, 19676, 20474, 23519, 25208,
    ],
    [
        3144, 5087, 7382, 7504, 7593, 7690, 7801, 8064, 8232, 9248, 9875, 10521, 29048,
    ],
];

/// `txtp_intra1[TX_8X8][mode]` (dav1d) — the 7-type intra tx-type CDFs for an
/// 8x8 luma block, indexed by luma intra mode. Row 0 (DC) equals the former
/// single `txtp_intra1_tx8_dc()`.
static TXTP_INTRA1_TX8: [[u16; 6]; 13] = [
    [1870, 13742, 14530, 16498, 23770, 27698],
    [326, 8796, 14632, 15079, 19272, 27486],
    [484, 7576, 7712, 14443, 19159, 22591],
    [1126, 15340, 15895, 17023, 20896, 30279],
    [655, 4854, 5249, 5913, 22099, 27138],
    [1299, 6458, 8885, 9290, 14851, 25497],
    [311, 5295, 5552, 6885, 16107, 22672],
    [883, 8059, 8270, 11258, 17289, 21549],
    [741, 7580, 9318, 10345, 16688, 29046],
    [110, 7406, 7915, 9195, 16041, 23329],
    [363, 7974, 9357, 10673, 15629, 24474],
    [153, 7647, 8112, 9936, 15307, 19996],
    [3511, 6332, 11165, 15335, 19323, 23594],
];

/// `txtp_intra2[TX_16X16][mode]` (dav1d) — the 5-type intra tx-type CDFs for a
/// 16x16 luma block, indexed by luma intra mode. Row 0 (DC) equals the former
/// single `txtp16` init.
static TXTP_INTRA2_TX16: [[u16; 4]; 13] = [
    [1127, 12814, 22772, 27483],
    [145, 6761, 11980, 26667],
    [362, 5887, 11678, 16725],
    [385, 15213, 18587, 30693],
    [25, 2914, 23134, 27903],
    [60, 4470, 11749, 23991],
    [37, 3332, 14511, 21448],
    [157, 6320, 13036, 17439],
    [119, 6719, 12906, 29396],
    [47, 5537, 12576, 21499],
    [269, 6076, 11258, 23115],
    [83, 5615, 12001, 17228],
    [1968, 5556, 12023, 18547],
];

/// `default_angle_delta_cdf[DIRECTIONAL_MODES]` (libaom): the angle-delta CDFs
/// for the 8 directional modes (V, H, then the diagonals), 7 symbols each
/// (delta -3..=3). Indexed by `y_mode - VERT_PRED`. Only V/H are used now; the
/// rest are stored for the directional follow-up.
pub(crate) static ANGLE_DELTA_CDF: [[u16; 6]; 8] = [
    [2180, 5032, 7567, 22776, 26989, 30217],
    [2301, 5608, 8801, 23487, 26974, 30330],
    [3780, 11018, 13699, 19354, 23083, 31286],
    [4581, 11226, 15147, 17138, 21834, 28397],
    [1737, 10927, 14509, 19588, 22745, 28823],
    [2664, 10176, 12485, 17650, 21600, 30495],
    [2240, 11096, 15453, 20341, 22561, 28917],
    [3605, 10428, 12459, 17676, 21244, 30655],
];

const DC_PRED: usize = 0;

const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_RANGE: i32 = 12;

/// Encode `read_golomb` value `v` (>=0) as unary-prefixed binary via bypass bits.
fn encode_golomb(enc: &mut OdEcEncoder, v: u32) {
    let x = v + 1;
    let length = 32 - x.leading_zeros();
    for _ in 0..length - 1 {
        enc.encode_bool(false, 16384); // unary zeros
    }
    for i in (0..length).rev() {
        enc.encode_bool((x >> i) & 1 == 1, 16384); // x MSB-first (MSB terminates unary)
    }
}

/// DC dequant value `dav1d_dq_tbl[bd][q][0]` for bit_depth 8/10/12.
pub(crate) fn dc_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::DC_QLOOKUP_10,
        12 => &crate::coef_q::DC_QLOOKUP_12,
        _ => &crate::coef_q::DC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}
/// AC dequant value `dav1d_dq_tbl[bd][q][1]` for bit_depth 8/10/12.
pub(crate) fn ac_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::AC_QLOOKUP_10,
        12 => &crate::coef_q::AC_QLOOKUP_12,
        _ => &crate::coef_q::AC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}

/// Inverse-transform clip bounds for `bit_depth`, matching dav1d's `itx_tmpl.c`:
/// returns `(row_min, row_max, col_min, col_max, cf_max)`. 8-bit uses `INT16`
/// for both row and col; for higher depth the row clip is `±2^(bd+7)`, the
/// column clip is `±2^(bd+5)`, and `cf_max == row_max == ~(~127 << bd)`.
fn itx_clips(bd: u8) -> (i32, i32, i32, i32, i32) {
    if bd <= 8 {
        let (mn, mx) = (i16::MIN as i32, i16::MAX as i32);
        (mn, mx, mn, mx, 32767)
    } else {
        let row_max = (1i32 << (bd + 7)) - 1;
        let col_max = (1i32 << (bd + 5)) - 1;
        (!row_max, row_max, !col_max, col_max, row_max)
    }
}

/// Source of the dequant coefficients and inverse-transform clip bounds the
/// DCT/IDCT drivers need. Implemented by [`Quant`], which computes them once
/// per (base_q_idx, bit_depth) so the transforms read them from `self` instead
/// of indexing `dav1d_dq_tbl` and recomputing the clips on every block.
pub(crate) trait Dct {
    /// DC dequant step (`dav1d_dq_tbl[bd][q][0]`). Used by the inverse transform
    /// and as the trellis distortion weight.
    fn dc_q(&self) -> i32;
    /// AC dequant step (`dav1d_dq_tbl[bd][q][1]`).
    fn ac_q(&self) -> i32;
    /// Inverse-transform clips `(row_min, row_max, col_min, col_max, cf_max)`.
    fn clips(&self) -> (i32, i32, i32, i32, i32);
    /// Forward-quantisation multiplier for DC, `round(65536 / dc_q)`, so that the
    /// integer forward DCT's `mul_q16(coeff, q_mult_dc()) ≈ coeff / dc_q` (the
    /// inverse multiplies the level back by `dc_q`, so this round-trips).
    fn q_mult_dc(&self) -> i32;
    /// Forward-quantisation multiplier for AC, `round(65536 / ac_q)`.
    fn q_mult_ac(&self) -> i32;
}

/// Precomputed dequant coefficients + inverse-transform clips for one
/// (base_q_idx, bit_depth). Cheap to copy; build once and hand to the transforms.
#[derive(Clone, Copy)]
pub(crate) struct Quant {
    dc: i32,
    ac: i32,
    q_mult_dc: i32,
    q_mult_ac: i32,
    rmin: i32,
    rmax: i32,
    cmin: i32,
    cmax: i32,
    cf_max: i32,
}

impl Quant {
    pub(crate) fn new(base_q_idx: u8, bd: u8) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        let dc = dc_q(base_q_idx, bd) as i32;
        let ac = ac_q(base_q_idx, bd) as i32;
        Quant {
            dc,
            ac,
            q_mult_dc: (65536.0_f64 / dc as f64).round() as i32,
            q_mult_ac: (65536.0_f64 / ac as f64).round() as i32,
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
        }
    }
}

impl Dct for Quant {
    #[inline]
    fn dc_q(&self) -> i32 {
        self.dc
    }
    #[inline]
    fn ac_q(&self) -> i32 {
        self.ac
    }
    #[inline]
    fn clips(&self) -> (i32, i32, i32, i32, i32) {
        (self.rmin, self.rmax, self.cmin, self.cmax, self.cf_max)
    }
    #[inline]
    fn q_mult_dc(&self) -> i32 {
        self.q_mult_dc
    }
    #[inline]
    fn q_mult_ac(&self) -> i32 {
        self.q_mult_ac
    }
}

/// Forward-DCT + quantize an 8x8 residual block into AV1 quantized coefficient
/// levels (raster order, for `encode_tx8_luma_coeffs`). The dav1d 8x8 inverse
/// DCT equals (1/8) x orthonormal DCT, so forward `cf = 8 * orthonormalDCT2(R)`,
/// quantized by dc_q (DC) / ac_q (AC), transposed (`rc = u*8 + v`) to dav1d's
/// coefficient layout. (Calibrated against dav1d: round-trip max error ~1 at q=16.)
pub(crate) fn forward_dct_quant_8x8(residual: &mut [i32; 64], q: &impl Dct) {
    dct8x8(residual, q)
}

/// Trellis (RDOQ) forward 8x8: the integer DCT levels plus the unrounded
/// per-coefficient targets. `.0` is bit-identical to `forward_dct_quant_8x8`.
pub(crate) fn forward_dct_quant_8x8_t(
    residual: &[i32; 64],
    q: &impl Dct,
) -> ([i32; 64], [f64; 64]) {
    dct8x8_t(residual, q)
}

/// AV1 up-right diagonal scan for an 8x8 transform (`scan_8x8`).
static SCAN_16X16: [usize; 256] = crate::coef_q::SCAN_16X16;
static SCAN_32X32: [usize; 1024] = crate::coef_q::SCAN_32X32;
static SCAN_8X16: [usize; 128] = crate::coef_q::SCAN_8X16;
static SCAN_16X32: [usize; 512] = crate::coef_q::SCAN_16X32;
/// R-D proxy per-block overhead (level-sum units) for the 16x16-vs-4x8x8
/// partition decision in `prefer_16x16`. Tuned so smooth regions pick 16x16.
const OVERHEAD_8: u32 = 8;
const OVERHEAD_16: u32 = 8;
static SCAN_8X8: [usize; 64] = [
    0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34,
    41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31,
    38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];
/// `dav1d_lo_ctx_offsets[0]` (square, w==h) — position offset for coeff_base ctx.
static LO_CTX_OFF: [[u32; 5]; 5] = [
    [0, 1, 6, 6, 21],
    [1, 6, 6, 21, 21],
    [6, 6, 21, 21, 21],
    [6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21],
];

/// The byte dav1d stores in its `levels` map for a coefficient of magnitude `m`,
/// used by the neighbour-context model: `m*0x41` for m<=2, else `min(m,15)+0xC0`.
fn level_byte(m: u32) -> u8 {
    if m == 0 {
        0
    } else if m <= 2 {
        (m * 0x41) as u8
    } else {
        (m.min(15) + (3 << 6)) as u8
    }
}

const EOB_BITW: u32 = 10;

fn est_block_bits(cf: &[i32], scan: &[usize]) -> u32 {
    let mut eob_idx: i32 = -1;
    for (i, &rc) in scan.iter().enumerate() {
        if cf[rc] != 0 {
            eob_idx = i as i32;
        }
    }
    if eob_idx < 0 {
        return 1;
    }
    let mag: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    mag + EOB_BITW * (eob_idx as u32 + 1)
}

/// Static per-level bit estimate for the AV1 coefficient token structure (base
/// token 0..3, the base-range ladder for levels >= 3, a golomb tail for levels
/// >= 15, plus one sign bit for any nonzero). Used only by the encoder's trellis
/// > quantizer to compare candidate levels — it need not be exact, since only the
/// > *relative* costs drive the decision.
pub(crate) fn coef_rate_bits(level: u32) -> f64 {
    match level {
        0 => 0.9, // a "0" base token coded in the interior run
        1 => 1.7 + 1.0,
        2 => 2.6 + 1.0,
        _ => {
            let mut b = 3.0 + 1.0; // base token == 3 (escape) + sign
            let total_br = ((level as i32) - 3).min(COEFF_BASE_RANGE); // 0..12
            let steps = (total_br / 3 + 1) as f64; // base-range symbols actually coded
            b += steps * 1.3;
            if level >= 15 {
                let v = level - 15;
                b += 2.0 * ((32 - (v + 1).leading_zeros()) as f64) - 1.0; // ~exp-golomb
            }
            b
        }
    }
}

/// `lambda0` for the trellis quantizer (R-D tradeoff, in `ac_q^2` units so the
/// behaviour is q-adaptive). Calibrated so the per-coefficient round-down and
/// EOB-trim land on the R-D frontier: meaningfully smaller streams for a
/// negligible PSNR cost, beating the naive "raise q" baseline.
const TRELLIS_LAMBDA0: f64 = 0.05;

/// Current trellis lambda0.
#[inline]
fn trellis_lambda() -> f64 {
    TRELLIS_LAMBDA0
}

/// Rate-distortion optimized quantization (trellis / RDOQ). Given baseline
/// rounded levels `cf` and the matching pre-round real targets `tf` (the value
/// `cf` was rounded from, = forward coefficient * SCALE / quant-step), this
/// lowers coefficient magnitudes and trims the end-of-block wherever doing so
/// reduces `D + lambda*R`, where `D` is the dequantized-domain squared error
/// (orthonormal transform, so proportional to pixel SSE) and `R` the estimated
/// coded bits. Only the written levels change, so the decoder — which simply
/// inverse-transforms whatever levels are coded — stays bit-exact; the caller
/// reconstructs from this same `cf`.
fn trellis_optimize(
    cf: &mut [i32],
    tf: &[f64],
    dc_q: f64,
    ac_q: f64,
    scan: &[usize],
    lambda0: f64,
) {
    if lambda0 <= 0.0 {
        return; // trellis disabled
    }
    let n = scan.len();
    let lambda = lambda0 * ac_q * ac_q;
    let dqf = |rc: usize| if rc == 0 { dc_q } else { ac_q };
    let d = |rc: usize, lev: i32| {
        let dq = dqf(rc);
        let t = tf[rc].abs();
        dq * dq * (t - lev.unsigned_abs() as f64).powi(2)
    };

    let mut eob_idx: i32 = -1;
    for i in 0..n {
        if cf[scan[i]] != 0 {
            eob_idx = i as i32;
        }
    }
    if eob_idx < 0 {
        return; // already all-zero
    }

    // Step A: per-coefficient round-down (toward zero) by local R-D.
    for &rc in scan[..=eob_idx as usize].iter() {
        let l = cf[rc].unsigned_abs();
        if l == 0 {
            continue;
        }
        let cost_l = d(rc, l as i32) + lambda * coef_rate_bits(l);
        let cost_dn = d(rc, (l - 1) as i32) + lambda * coef_rate_bits(l - 1);
        if cost_dn < cost_l {
            let s = if cf[rc] < 0 { -1 } else { 1 };
            cf[rc] = s * (l as i32 - 1);
        }
    }

    // Step B: choose the last-nonzero (EOB) position that minimises total cost,
    // dropping every coefficient after it.
    let mut suf0 = vec![0f64; n + 1]; // distortion of zeroing coeffs from i..n
    for i in (0..n).rev() {
        suf0[i] = suf0[i + 1] + d(scan[i], 0);
    }
    let mut pre = vec![0f64; n + 1]; // interior cost of coeffs strictly before i
    for i in 0..n {
        let rc = scan[i];
        pre[i + 1] = pre[i] + d(rc, cf[rc]) + lambda * coef_rate_bits(cf[rc].unsigned_abs());
    }
    let eob_sig = |e: usize| -> f64 {
        let bin = if e < 2 {
            e
        } else {
            (32 - (e as u32).leading_zeros()) as usize
        };
        let extra = if bin > 1 { bin - 2 } else { 0 };
        (bin as f64) * 0.9 + extra as f64 + 2.0 // eob_pt + extra bits + eob_base token
    };

    let mut best_e: i32 = -1;
    let mut best_cost = f64::INFINITY;
    for e in 0..n {
        let rc = scan[e];
        if cf[rc] == 0 {
            continue; // the EOB must land on a nonzero
        }
        let c = pre[e] + d(rc, cf[rc]) + lambda * eob_sig(e) + suf0[e + 1];
        if c < best_cost {
            best_cost = c;
            best_e = e as i32;
        }
    }
    let skip_cost = suf0[0] + lambda * 1.0; // zero everything + the txb_skip flag
    if best_e < 0 || skip_cost < best_cost {
        for &rc in scan.iter() {
            cf[rc] = 0;
        }
        return;
    }
    for i in (best_e as usize + 1)..n {
        cf[scan[i]] = 0;
    }
}

/// Probability -> bit-cost table. `COST_Q[p]` holds `-log2(p / 32768)` in
/// Q22 fixed point (1/2^22 bit units) for every CDF partition `p` in
/// `[1, 32768]`. Built once; replaces a per-call `log2()` (a libm transcendental
/// that dominated the trellis) with a single array load. Q22 keeps the rounding
/// error ~1e-7 bits, far below anything the R-D comparison can resolve, so the
/// chosen levels are identical to the float version.
const COST_Q_FRAC: u32 = 22;
const COST_Q_SCALE_INV: f64 = 1.0 / (1u32 << COST_Q_FRAC) as f64;

fn cost_q_table() -> &'static [u32; 32769] {
    static TABLE: std::sync::OnceLock<Box<[u32; 32769]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u32; 32769]);
        for (p, slot) in t.iter_mut().enumerate().skip(1) {
            let bits = -((p as f64) * (1.0 / 32768.0)).log2();
            *slot = (bits * (1u32 << COST_Q_FRAC) as f64).round() as u32;
        }
        t
    })
}

/// Bits to code symbol `s` against an (inverse-form) CDF: `-log2(p)` where the
/// probability is `(cdf[s-1] - cdf[s]) / 32768` (with `cdf[-1] = 32768`). This
/// matches the MSAC's symbol partition (ignoring the negligible `EC_MIN_PROB`
/// term), so it is the same rate libaom's cost tables approximate. The `-log2`
/// is a precomputed fixed-point table lookup (see [`cost_q_table`]).
#[inline]
fn cdf_cost(cdf: &[u16], s: usize) -> f64 {
    let fl = if s > 0 { cdf[s - 1] as i32 } else { 32768 };
    let fh = cdf[s] as i32;
    let p = (fl - fh).max(1) as usize;
    cost_q_table()[p] as f64 * COST_Q_SCALE_INV
}

/// Bypass bits for the Exp-Golomb tail coding `v` (level ≥ 15 carries `v=L-15`).
#[inline]
fn golomb_cost(v: u32) -> f64 {
    let len = 32 - (v + 1).leading_zeros();
    (2 * len - 1) as f64
}

/// Accurate bit cost of the base-range (hi_tok) ladder for magnitude `m` (≥ 3)
/// against `br_cdf`, plus the Exp-Golomb tail when `m ≥ 15`.
fn hi_tok_cost(m: u32, br_cdf: &[u16]) -> f64 {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0i32;
    let mut bits = 0.0;
    for _ in 0..(COEFF_BASE_RANGE / 3) {
        let s = (total_br - coded).min(3);
        bits += cdf_cost(br_cdf, s as usize);
        coded += s;
        if s < 3 {
            break;
        }
    }
    if m >= 15 {
        bits += golomb_cost(m - 15);
    }
    bits
}

/// Context-accurate RDOQ for the 2D square luma transforms (TX_8X8/16X16/32X32,
/// `cls` 1/2/3). Unlike [`trellis_optimize`], the per-coefficient rate is the
/// *real* coding cost: the base token cost depends on the neighbour-magnitude
/// context (so coefficients in sparse regions are correctly cheap), the
/// base-range/Golomb tail and the EOB signalling are priced from the live CDFs.
/// Level reduction runs in reverse scan order so forward-neighbour contexts are
/// stable; a final pass re-selects the EOB with accurate `eob_pt`/`eob_base`
/// costs. Only the chosen levels change, so the result still decodes exactly.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn trellis_optimize_ctx(
    cf: &mut [i32],
    tf: &[f64],
    dc_q: f64,
    ac_q: f64,
    scan: &[usize],
    lambda0: f64,
    w: usize,
    cdfs: &Cdfs,
    cls: usize,
    plane: usize,
    eob_bin_cdf: &[u16],
    dcs_ctx: usize,
) {
    if lambda0 <= 0.0 {
        return;
    }
    let n = scan.len();
    let lambda = lambda0 * ac_q * ac_q;
    let log2w = w.trailing_zeros() as usize;
    let stride = w;
    // Hoist the per-(class, plane) CDF tables once for clarity (and to avoid
    // re-walking the nested arrays on every coefficient).
    let base_tok = &cdfs.base_tok[cls][plane];
    let br_tok = &cdfs.br_tok[cls][plane];
    let eob_hi = &cdfs.eob_hi[cls][plane];
    let eob_base = &cdfs.eob_base[cls][plane];
    let dc_sign = &cdfs.dc_sign[plane];
    let dq2_dc = dc_q * dc_q;
    let dq2_ac = ac_q * ac_q;
    let dist = |rc: usize, lev: i32| {
        let dq2 = if rc == 0 { dq2_dc } else { dq2_ac };
        let e = tf[rc].abs() - (lev.abs() as f64);
        dq2 * (e * e)
    };

    // Precompute the base-range (hi_tok) ladder cost for every br context and
    // every total_br in 0..=12, once per call. `hi_tok_cost` otherwise reruns a
    // 4-step `cdf_cost` ladder for every level-3+ coefficient (hot at high
    // quality). Only worth the ~0.5us setup for the larger transforms (n >= 256),
    // where it is called hundreds of times; small blocks use the direct path.
    // Accumulation order matches `hi_tok_cost` exactly, so the chosen levels are
    // identical either way.
    let use_br_table = n >= 256;
    let mut br_cum = [[0f64; 13]; 21];
    if use_br_table {
        for (bc, row) in br_cum.iter_mut().enumerate() {
            let br = &br_tok[bc];
            let c = [
                cdf_cost(br, 0),
                cdf_cost(br, 1),
                cdf_cost(br, 2),
                cdf_cost(br, 3),
            ];
            for (j, slot) in row.iter_mut().enumerate() {
                let mut coded = 0i32;
                let mut bits = 0.0;
                for _ in 0..(COEFF_BASE_RANGE / 3) {
                    let s = (j as i32 - coded).min(3);
                    bits += c[s as usize];
                    coded += s;
                    if s < 3 {
                        break;
                    }
                }
                *slot = bits;
            }
        }
    }
    // Base-range tail cost for magnitude `m` (>= 3) in br context `bc`.
    let hi_cost = |m: u32, bc: usize| -> f64 {
        if use_br_table {
            let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
            let mut bits = br_cum[bc][total_br as usize];
            if m >= 15 {
                bits += golomb_cost(m - 15);
            }
            bits
        } else {
            hi_tok_cost(m, &br_tok[bc])
        }
    };

    // Last nonzero in scan order. `rposition` scans from the end and stops at the
    // first hit, and iterating `scan` drops its bounds check (only `cf[rc]` is a
    // random access). Same value as the forward max-index scan.
    let eob: i32 = scan
        .iter()
        .rposition(|&rc| cf[rc] != 0)
        .map_or(-1, |i| i as i32);
    if eob < 0 {
        return;
    }
    let eu = eob as usize;

    // Reuse scratch allocations across calls (this runs once per coded transform
    // block, so per-call alloc+zero+free of `levels`/`pre`/`suf0`/`irate` shows
    // up in profiles). Buffers are taken from a thread-local pool and returned at
    // the single exit below. Entries are fully overwritten before use except the
    // cumulative seed (`suf0[n]`) and `levels` (a sparse magnitude map
    // that must start zeroed), which are reset explicitly.

    thread_local! {
        static SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<f64>, Vec<f64>, Vec<f64>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
    }
    let (mut levels, mut pre, mut suf0, mut irate) = SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        (
            std::mem::take(&mut b.0),
            std::mem::take(&mut b.1),
            std::mem::take(&mut b.2),
            std::mem::take(&mut b.3),
        )
    });
    levels.clear();
    levels.resize(w * (w + 4), 0);
    let set_level = |levels: &mut [u8], rc: usize, m: u32| {
        levels[(rc >> log2w) * stride + (rc & (w - 1))] = level_byte(m);
    };
    for &rc in &scan[..eu + 1] {
        set_level(&mut levels, rc, cf[rc].unsigned_abs());
    }

    // Interior base-token context + br context for a position, from `levels`.
    let interior_ctx = |levels: &[u8], rc: usize| -> (usize, usize) {
        let (x, y) = (rc >> log2w, rc & (w - 1));
        let (ctx, hi_mag) = get_lo_ctx_2d(levels, x, y, &LO_CTX_OFF, stride);
        let mag = hi_mag & 63;
        let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
        (ctx, bc as usize)
    };
    let dc_brc = |levels: &[u8]| -> usize {
        let mag = (levels[1] as u32 + levels[stride] as u32 + levels[stride + 1] as u32) & 63;
        if mag > 12 {
            6
        } else {
            ((mag + 1) >> 1) as usize
        }
    };
    // Rate of an interior coefficient at level k (base_tok + br + AC sign).
    let interior_rate = |ctx: usize, bc: usize, k: u32| -> f64 {
        if k == 0 {
            return cdf_cost(&base_tok[ctx], 0);
        }
        let tok = k.min(3);
        let mut b = cdf_cost(&base_tok[ctx], tok as usize);
        if tok == 3 {
            b += hi_cost(k, bc);
        }
        b + 1.0 // AC sign (bypass)
    };

    // Step A: reverse-scan per-coefficient RD-best level (interior), then DC.
    for i in (1..(eob as usize)).rev() {
        let rc = scan[i];
        let l = cf[rc].unsigned_abs();
        if l == 0 {
            continue;
        }
        let (ctx, bc) = interior_ctx(&levels, rc);
        // Hoist the four base-token costs (tok 0..=3) out of the k-loop; only the
        // br/Golomb tail (k >= 3) and distortion vary per candidate. Float-op
        // order matches `interior_rate` exactly so the choice is unchanged.
        let bt = &base_tok[ctx];
        let bt0 = cdf_cost(bt, 0);
        let bt1 = cdf_cost(bt, 1);
        let bt2 = cdf_cost(bt, 2);
        let bt3 = cdf_cost(bt, 3);
        let rate_k = |k: u32| -> f64 {
            match k {
                0 => bt0,
                1 => bt1 + 1.0,
                2 => bt2 + 1.0,
                _ => (bt3 + hi_cost(k, bc)) + 1.0,
            }
        };
        let mut best_k = l;
        let mut best_c = dist(rc, l as i32) + lambda * rate_k(l);
        for k in (0..l).rev() {
            let dk = dist(rc, k as i32);
            // dist grows monotonically as k falls below l (l <= |tf|), and the
            // rate is non-negative, so once dist alone reaches best_c no smaller
            // level can win. Exact, just stops the scan early.
            if dk >= best_c {
                break;
            }
            let c = dk + lambda * rate_k(k);
            if c < best_c {
                best_c = c;
                best_k = k;
            }
        }
        if best_k != l {
            cf[rc] = if cf[rc] < 0 {
                -(best_k as i32)
            } else {
                best_k as i32
            };
            set_level(&mut levels, rc, best_k);
        }
    }
    {
        let rc = scan[0];
        let l = cf[rc].unsigned_abs();
        if l != 0 {
            let bc = dc_brc(&levels);
            let sgn = (cf[rc] < 0) as usize;
            let dc_rate = |k: u32| -> f64 {
                if k == 0 {
                    return cdf_cost(&base_tok[0], 0);
                }
                let tok = k.min(3);
                let mut b = cdf_cost(&base_tok[0], tok as usize);
                if tok == 3 {
                    b += hi_cost(k, bc);
                }
                b + cdf_cost(&dc_sign[dcs_ctx], sgn)
            };
            let mut best_k = l;
            let mut best_c = dist(rc, l as i32) + lambda * dc_rate(l);
            for k in (0..l).rev() {
                let dk = dist(rc, k as i32);
                if dk >= best_c {
                    break;
                }
                let c = dk + lambda * dc_rate(k);
                if c < best_c {
                    best_c = c;
                    best_k = k;
                }
            }
            if best_k != l {
                cf[rc] = if cf[rc] < 0 {
                    -(best_k as i32)
                } else {
                    best_k as i32
                };
                set_level(&mut levels, rc, best_k);
            }
        }
    }

    // Step B: EOB-position selection with accurate eob_pt / eob_base costs.
    let eob_pt_cost = |e: usize| -> f64 {
        let bin = if e < 2 {
            e
        } else {
            32 - (e as u32).leading_zeros() as usize
        };
        let mut c = cdf_cost(eob_bin_cdf, bin);
        if bin > 1 {
            let nbits = bin - 2;
            c += cdf_cost(&eob_hi[bin], (e >> nbits) & 1);
            c += nbits as f64; // remaining eob offset bits (bypass)
        }
        c
    };
    let eob_coeff_cost = |e: usize, m: u32| -> f64 {
        let ctx_e = 1 + (e > n / 8) as usize + (e > n / 4) as usize;
        let tok = m.min(3);
        let mut c = cdf_cost(&eob_base[ctx_e], tok as usize - 1);
        if tok == 3 {
            let rc = scan[e];
            let (ex, ey) = (rc >> log2w, rc & (w - 1));
            let bc = if (ex | ey) > 1 { 14 } else { 7 };
            c += hi_cost(m, bc);
        }
        c + 1.0 // sign
    };

    // Interior (base_tok) rate of each position at its current level, for the
    // running prefix; positions are priced as interior even if they will end up
    // being the EOB (corrected by swapping in eob_coeff_cost at the candidate).
    // Driven by zipped slice iterators so the sequential index checks drop out;
    // accumulation order (`acc + lambda*r + d`) matches the indexed form exactly.
    pre.resize(n + 1, 0.0);
    irate.resize(n, 0.0);
    let mut acc = 0.0f64; // pre[1]: empty prefix
    // Interior positions [1, eob]: priced with neighbour context.
    for ((&rc, ir), p) in scan[1..eu + 1]
        .iter()
        .zip(irate[1..eu + 1].iter_mut())
        .zip(pre[2..eu + 2].iter_mut())
    {
        let (ctx, bc) = interior_ctx(&levels, rc);
        let r = interior_rate(ctx, bc, cf[rc].unsigned_abs());
        *ir = r;
        acc = (acc + lambda * r) + dist(rc, cf[rc]);
        *p = acc;
    }
    // Trailing positions (eob, n): coded as zeros, distortion only.
    for (&rc, p) in scan[eu + 1..n].iter().zip(pre[eu + 2..n + 1].iter_mut()) {
        acc += dist(rc, 0);
        *p = acc;
    }
    suf0.resize(n + 1, 0.0);
    suf0[n] = 0.0; // suffix seed (read as suf0[n]; not written by the loop below)
    let mut sacc = 0.0f64;
    for (&rc, s) in scan[1..n].iter().rev().zip(suf0[1..n].iter_mut().rev()) {
        sacc += dist(rc, 0);
        *s = sacc;
    }
    // DC contribution (rate + distortion), constant across EOB choices ≥ 1.
    let dc_rc = scan[0];
    let dc_m = cf[dc_rc].unsigned_abs();
    let dc_cost = if dc_m == 0 {
        lambda * cdf_cost(&base_tok[0], 0)
    } else {
        let bc = dc_brc(&levels);
        let tok = dc_m.min(3);
        let mut b = cdf_cost(&base_tok[0], tok as usize);
        if tok == 3 {
            b += hi_cost(dc_m, bc);
        }
        b += cdf_cost(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize);
        lambda * b
    } + dist(dc_rc, cf[dc_rc]);

    let mut best_e: i32 = -1;
    let mut best_cost = f64::INFINITY;
    for e in 1..n {
        let rc = scan[e];
        if cf[rc] == 0 {
            continue; // EOB must land on a nonzero
        }
        // pre[e] prices position e as interior; replace with eob_coeff cost.
        // A nonzero at `e` implies `e <= eob`, so `irate[e]` was filled above
        // (identical value to interior_rate here, just cached).
        let interior_e = lambda * irate[e];
        let c = dc_cost
            + (pre[e + 1] - interior_e)
            + lambda * (eob_pt_cost(e) + eob_coeff_cost(e, cf[rc].unsigned_abs()))
            + suf0[e + 1];
        if c < best_cost {
            best_cost = c;
            best_e = e as i32;
        }
    }
    // EOB at DC (only DC nonzero) and the all-zero (txb_skip) alternative.
    if dc_m != 0 {
        let ctx_e = 1usize; // e == 0
        let tok = dc_m.min(3);
        let mut c0 = cdf_cost(eob_bin_cdf, 0) + cdf_cost(&eob_base[ctx_e], tok as usize - 1);
        if tok == 3 {
            c0 += hi_cost(dc_m, dc_brc(&levels));
        }
        c0 += cdf_cost(&dc_sign[dcs_ctx], (cf[dc_rc] < 0) as usize);
        let total0 = lambda * c0 + dist(dc_rc, cf[dc_rc]) + suf0[1];
        if total0 < best_cost {
            best_cost = total0;
            best_e = 0;
        }
    }
    let skip_cost = suf0[1] + dist(dc_rc, 0) + lambda * 1.0;
    if best_e < 0 || skip_cost < best_cost {
        for &rc in scan.iter() {
            cf[rc] = 0;
        }
    } else {
        for i in (best_e as usize + 1)..n {
            cf[scan[i]] = 0;
        }
    }
    SCRATCH.with(|s| {
        let mut b = s.borrow_mut();
        b.0 = std::mem::take(&mut levels);
        b.1 = std::mem::take(&mut pre);
        b.2 = std::mem::take(&mut suf0);
        b.3 = std::mem::take(&mut irate);
    });
}
/// directional modes (V/H and the diagonals, 1..=8) are intentionally omitted
/// from this set: they would also require the `angle_delta` symbol on >= 8x8
/// blocks. These four need no extra symbols beyond `y_mode` itself.
const SMOOTH_PRED: usize = 9;
const SMOOTH_V_PRED: usize = 10;
const SMOOTH_H_PRED: usize = 11;
const PAETH_PRED: usize = 12;
/// Directional modes added in this increment (axis-aligned only, at
/// `angle_delta = 0`): pure vertical / horizontal copy. They sit in the
/// directional range 1..=8 (`VERT_LEFT_PRED`), so they emit an `angle_delta`
/// symbol — but at delta 0 (angle 90/180) the decoder maps them straight to the
/// plain copy predictors, with no top-right/bottom-left edge extension.
const V_PRED: usize = 1;
const H_PRED: usize = 2;
/// Z2 diagonal modes (down-right directions). At `angle_delta = 0` their angles
/// are 135/113/157, all in (90,180) -> the dav1d `ipred_z2` path, which reads
/// only top + left + corner (no top-right/bottom-left edge extension), so they
/// reuse the same reference arrays as the other modes. The Z1 (D45/D67) and Z3
/// (D203) diagonals — which DO need the extension samples — are still deferred.
const D135_PRED: usize = 4;
const D113_PRED: usize = 5;
const D157_PRED: usize = 6;
const VERT_LEFT_PRED: usize = 8;
/// Z1 diagonals (up-right): `D45` (45 deg) and `D67` (= `VERT_LEFT_PRED`, 67 deg).
/// They read the top row extended to the right (top-right samples). Z3 diagonal
/// (down-left): `D203` (203 deg), reading the left column extended downward
/// (bottom-left samples). These need the neighbour-availability derivation
/// (dav1d's intra-edge tree) and the extended reference arrays.
const D45_PRED: usize = 3;
const D203_PRED: usize = 7;
/// Chroma-from-luma. Signalled as a `uv_mode` symbol; its tx-type is **not** in
/// `txtp_from_uvmode`, so it defaults to `DCT_DCT` — i.e. CfL needs no ADST.
const CFL_PRED: usize = 13;

/// `default_cfl_sign_cdf` (libaom/dav1d): joint sign of the U/V alphas, 8 symbols.
static CFL_SIGN_CDF: [u16; 7] = [1418, 2123, 13340, 18405, 26972, 28343, 32294];
/// `default_cfl_alpha_cdf[6]`: per-plane alpha magnitude (1..=16 -> symbols 0..=15),
/// indexed by a context derived from the joint sign.
static CFL_ALPHA_CDF: [[u16; 15]; 6] = [
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
static DR_INTRA_DERIVATIVE: [i32; 44] = [
    0, 1023, 0, 547, 372, 0, 0, 273, 215, 0, 178, 151, 0, 132, 116, 0, 102, 0, 90, 80, 0, 71, 64,
    0, 57, 51, 0, 45, 0, 40, 35, 0, 31, 27, 0, 23, 19, 0, 15, 0, 11, 0, 7, 3,
];

/// `dav1d_intra_mode_context` — maps an intra mode to its keyframe y-mode CDF
/// context (0..=4), used for both the above and left neighbours.
pub(crate) static INTRA_MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `dav1d_sm_weights` slice for a given block dimension (SMOOTH predictors).
fn sm_weights(n: usize) -> &'static [i32] {
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

/// Build the AV1 intra reference edges from the reconstructed plane and predict
/// `mode` into `out` (row-major `bw*bh`). Bit-exact with dav1d's non-directional
/// predictors (`ipred_{paeth,smooth,smooth_v,smooth_h}_c`) and the default-fill
/// rules of `dav1d_prepare_intra_edges` (single-tile raster order: above/left
/// availability = not at the frame's top/left edge). `recon`/`stride` is the
/// reconstructed plane; `(ox, oy)` the block's pixel origin. DC is handled by
/// the dedicated `dc_pred_*` helpers, not here.
/// CfL luma-AC for 4:4:4: the reconstructed luma block scaled by 8 with its mean
/// removed, exactly as dav1d's `cfl_ac` with `ss_hor = ss_ver = 0`.
fn cfl_ac_444(luma_rec: &[i32], w: usize, h: usize, ac: &mut [i32]) {
    let n = w * h;
    for (ac, luma) in ac[..n].iter_mut().zip(luma_rec[..n].iter()) {
        *ac = *luma << 3;
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

/// CfL prediction combine (dav1d `cfl_pred`): `dc + sign(diff)*((|diff|+32)>>6)`.
#[inline]
fn cfl_pred_pixel(dc: i32, ac: i32, alpha: i32, bd: u8) -> i32 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 32) >> 6;
    let s = if diff < 0 { -mag } else { mag };
    (dc + s).clamp(0, (1 << bd) - 1)
}

/// Energy-minimising CfL alpha for one plane, in dav1d alpha units (the predictor
/// applies `alpha/64` after the <<3 AC scaling). Returns the best of the analytic
/// optimum and its +/-1 neighbours by pre-quantisation residual energy, clamped to
/// the signalled range [-16, 16] (0 means "CfL useless for this plane").
fn cfl_best_alpha(ac: &[i32], src: &[i32], dc: i32, n: usize, bd: u8) -> i32 {
    let mut num: i64 = 0;
    let mut den: i64 = 0;
    for i in 0..n {
        num += (src[i] - dc) as i64 * ac[i] as i64;
        den += ac[i] as i64 * ac[i] as i64;
    }
    if den == 0 {
        return 0;
    }
    let a0 = ((64 * num + (den >> 1) * num.signum()) / den).clamp(-16, 16) as i32;
    let mut best_a = 0i32;
    let mut best_e = i64::MAX;
    for cand in [a0 - 1, a0, a0 + 1] {
        if !(-16..=16).contains(&cand) {
            continue;
        }
        let mut e: i64 = 0;
        for i in 0..n {
            let d = (src[i] - cfl_pred_pixel(dc, ac[i], cand, bd)) as i64;
            e += d * d;
        }
        if e < best_e {
            best_e = e;
            best_a = cand;
        }
    }
    best_a
}

#[allow(clippy::too_many_arguments)]
fn intra_predict_nd(
    mode: usize,
    recon: &[i32],
    stride: usize,
    ox: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    have_tr: bool,
    have_bl: bool,
    fw: usize,
    fh: usize,
    out: &mut [i32],
    bd: u8,
) {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let base = 1i32 << (bd - 1);
    // Sized for the directional reach at the largest supported block (32): top
    // row + top-right extension (Z1) and left column + bottom-left extension
    // (Z3), each up to 2x the block dim, so 2*32 = 64.
    let mut top = [0i32; 64];
    let mut left = [0i32; 64];
    if have_top {
        for i in 0..bw {
            top[i] = recon[(oy - 1) * stride + ox + i];
        }
    } else {
        let fill = if have_left {
            recon[oy * stride + ox - 1]
        } else {
            base - 1
        };
        top[..bw].fill(fill);
    }
    if have_left {
        for j in 0..bh {
            left[j] = recon[(oy + j) * stride + ox - 1];
        }
    } else {
        let fill = if have_top {
            recon[(oy - 1) * stride + ox]
        } else {
            base + 1
        };
        left[..bh].fill(fill);
    }
    let corner = if have_left {
        if have_top {
            recon[(oy - 1) * stride + ox - 1]
        } else {
            recon[oy * stride + ox - 1]
        }
    } else if have_top {
        recon[(oy - 1) * stride + ox]
    } else {
        base
    };
    // Top-right extension (top[bw..2bw]) for Z1, and bottom-left extension
    // (left[bh..2bh]) for Z3, following dav1d_prepare_intra_edges: copy the
    // available samples (clamped to the frame edge) then replicate, or — when
    // the neighbour is unavailable — replicate the last edge sample.
    if have_tr {
        let px_have = bw.min(fw - (ox + bw));
        for i in 0..px_have {
            top[bw + i] = recon[(oy - 1) * stride + ox + bw + i];
        }
        let fill = top[bw + px_have - 1];
        for i in px_have..bw {
            top[bw + i] = fill;
        }
    } else {
        let fill = top[bw - 1];
        for i in 0..bw {
            top[bw + i] = fill;
        }
    }
    if have_bl {
        let px_have = bh.min(fh - (oy + bh));
        for i in 0..px_have {
            left[bh + i] = recon[(oy + bh + i) * stride + ox - 1];
        }
        let fill = left[bh + px_have - 1];
        for i in px_have..bh {
            left[bh + i] = fill;
        }
    } else {
        let fill = left[bh - 1];
        for i in 0..bh {
            left[bh + i] = fill;
        }
    }

    match mode {
        V_PRED => {
            for orow in out.chunks_exact_mut(bw) {
                orow.copy_from_slice(&top[..bw]);
            }
        }
        H_PRED => {
            for (orow, &lv) in out.chunks_exact_mut(bw).zip(left.iter()) {
                orow.iter_mut().for_each(|o| *o = lv);
            }
        }
        D45_PRED | VERT_LEFT_PRED => {
            // dav1d ipred_z1 (edge filter/upsampling off): project from the top
            // row (extended with top-right samples). D45 -> 45 deg, D67 -> 67 deg.
            let angle: i32 = if mode == D45_PRED { 45 } else { 67 };
            let dx = DR_INTRA_DERIVATIVE[(angle >> 1) as usize];
            let max_base_x = (bw + bw.min(bh) - 1) as i32;
            for y in 0..bh {
                let xpos = dx * (y as i32 + 1);
                let frac = xpos & 0x3E;
                let mut bx = xpos >> 6;
                #[allow(clippy::explicit_counter_loop)]
                for x in 0..bw {
                    if bx < max_base_x {
                        let v = top[bx as usize] * (64 - frac) + top[(bx + 1) as usize] * frac;
                        out[y * bw + x] = (v + 32) >> 6;
                    } else {
                        let fill = top[max_base_x as usize];
                        for xx in x..bw {
                            out[y * bw + xx] = fill;
                        }
                        break;
                    }
                    bx += 1;
                }
            }
        }
        D203_PRED => {
            // dav1d ipred_z3 (edge filter/upsampling off): project from the left
            // column (extended with bottom-left samples). D203 -> 203 deg.
            let angle: i32 = 203;
            let dy = DR_INTRA_DERIVATIVE[((270 - angle) >> 1) as usize];
            let max_base_y = (bh + bw.min(bh) - 1) as i32;
            for x in 0..bw {
                let ypos = dy * (x as i32 + 1);
                let frac = ypos & 0x3E;
                let mut by = ypos >> 6;
                #[allow(clippy::explicit_counter_loop)]
                for y in 0..bh {
                    if by < max_base_y {
                        let v = left[by as usize] * (64 - frac) + left[(by + 1) as usize] * frac;
                        out[y * bw + x] = (v + 32) >> 6;
                    } else {
                        let fill = left[max_base_y as usize];
                        for yy in y..bh {
                            out[yy * bw + x] = fill;
                        }
                        break;
                    }
                    by += 1;
                }
            }
        }
        D135_PRED | D113_PRED | D157_PRED => {
            // dav1d ipred_z2 with edge filter/upsampling disabled: pure angular
            // projection from the top row, left column and corner.
            let angle: i32 = match mode {
                D135_PRED => 135,
                D113_PRED => 113,
                _ => 157,
            };
            let dy = DR_INTRA_DERIVATIVE[((angle - 90) >> 1) as usize];
            let dx = DR_INTRA_DERIVATIVE[((180 - angle) >> 1) as usize];
            // topleft[idx]: idx 0 = corner, idx>=1 = top[idx-1], idx<0 = left[-idx-1]
            let tl = |idx: i32| -> i32 {
                if idx >= 0 {
                    if idx == 0 {
                        corner
                    } else {
                        top[((idx - 1) as usize).min(bw - 1)]
                    }
                } else {
                    left[((-idx - 1) as usize).min(bh - 1)]
                }
            };
            for y in 0..bh as i32 {
                let xpos = (1 << 6) - dx * (y + 1);
                let mut base_x = xpos >> 6;
                let frac_x = xpos & 0x3E;
                let mut ypos = (y << 6) - dy;
                #[allow(clippy::explicit_counter_loop)]
                for x in 0..bw as i32 {
                    let v = if base_x >= 0 {
                        tl(base_x) * (64 - frac_x) + tl(base_x + 1) * frac_x
                    } else {
                        let base_y = ypos >> 6;
                        let frac_y = ypos & 0x3E;
                        tl(-1 - base_y) * (64 - frac_y) + tl(-2 - base_y) * frac_y
                    };
                    out[(y * bw as i32 + x) as usize] = (v + 32) >> 6;
                    base_x += 1;
                    ypos -= dy;
                }
            }
        }
        PAETH_PRED => {
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
        SMOOTH_PRED => {
            let (wv, wh) = (sm_weights(bh), sm_weights(bw));
            let (right, bottom) = (top[bw - 1], left[bh - 1]);
            for ((orow, &wvy), &lv) in out.chunks_exact_mut(bw).zip(wv.iter()).zip(left.iter()) {
                for (o, (&tv, &whx)) in orow.iter_mut().zip(top.iter().zip(wh.iter())) {
                    let pred = wvy * tv + (256 - wvy) * bottom + whx * lv + (256 - whx) * right;
                    *o = (pred + 256) >> 9;
                }
            }
        }
        SMOOTH_V_PRED => {
            let wv = sm_weights(bh);
            let bottom = left[bh - 1];
            for (orow, &wvy) in out.chunks_exact_mut(bw).zip(wv.iter()) {
                for (o, &tv) in orow.iter_mut().zip(top.iter()) {
                    let pred = wvy * tv + (256 - wvy) * bottom;
                    *o = (pred + 128) >> 8;
                }
            }
        }
        SMOOTH_H_PRED => {
            let wh = sm_weights(bw);
            let right = top[bw - 1];
            for (orow, &lv) in out.chunks_exact_mut(bw).zip(left.iter()) {
                for (o, &whx) in orow.iter_mut().zip(wh.iter()) {
                    let pred = whx * lv + (256 - whx) * right;
                    *o = (pred + 128) >> 8;
                }
            }
        }
        _ => unreachable!("intra_predict_nd called with mode {}", mode),
    }
}

/// Candidate non-directional luma modes evaluated by the mode search, in CDF
/// symbol order (DC first).
/// Estimated coded bits for a quantized block, for the intra mode search. Unlike
/// `est_block_bits` (a partition-time proxy whose EOB term wrongly penalises the
/// many-small-coefficient residuals that good prediction produces), this sums
/// the calibrated per-level token cost over the coded prefix, so it tracks the
/// real entropy cost and ranks predictors correctly.
fn block_rate_bits(cf: &[i32], scan: &[usize]) -> f64 {
    let mut eob: i32 = -1;
    for (i, &rc) in scan.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i as i32;
        }
    }
    if eob < 0 {
        return 1.0; // all-zero: just the txb_skip flag
    }
    let mut bits = 2.0; // eob_pt / skip-flag overhead
    for &rc in scan.iter().take(eob as usize + 1) {
        bits += coef_rate_bits(cf[rc].unsigned_abs());
    }
    bits
}

static ND_LUMA_MODES: [usize; 13] = [
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
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
];

/// Candidate luma modes evaluated by the intra mode search.
fn nd_modes() -> &'static [usize] {
    &ND_LUMA_MODES
}

/// R-D weight for the intra luma mode search (cost = pixel SSE + lambda * proxy
/// bits, with `lambda = MODE_LAMBDA0 * ac_q^2` so it tracks the quantizer).
const MODE_LAMBDA0: f64 = 0.02;
#[inline]
fn mode_lambda() -> f64 {
    MODE_LAMBDA0
}
/// Rough extra bits to *signal* a non-DC luma mode (DC is the most probable
/// symbol; the others cost a little more). Keeps the search from switching modes
/// for a negligible residual gain.
/// Estimated cost (in bits) of *choosing* a non-DC luma mode: the rare y_mode
/// symbol, the shift of the uv_mode CDF context (chroma still codes DC, but
/// under a less-favourable context), and CDF-adaptation churn. DC is free. This
/// is what makes the search only leave DC for a clear net win.
#[inline]
fn mode_signal_bits(m: usize) -> f64 {
    if m == DC_PRED { 0.0 } else { MODE_SIGNAL_BITS }
}
const MODE_SIGNAL_BITS: f64 = 30.0;

fn get_lo_ctx_2d(
    levels: &[u8],
    x: usize,
    y: usize,
    off: &[[u32; 5]; 5],
    stride: usize,
) -> (usize, u32) {
    let g = |dx: usize, dy: usize| levels[(x + dx) * stride + (y + dy)] as u32;
    let hi_mag = g(0, 1) + g(1, 0) + g(1, 1);
    let mag = hi_mag + g(0, 2) + g(2, 0);
    let offset = off[y.min(4)][x.min(4)];
    let ctx = offset + if mag > 512 { 4 } else { (mag + 64) >> 7 };
    (ctx as usize, hi_mag)
}

/// Encode the hi_tok (base-range) ladder for magnitude `m` (>=3) with `br_cdf`.
fn encode_hi_tok(enc: &mut OdEcEncoder, m: u32, br_cdf: &mut [u16]) {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0;
    for _ in 0..(COEFF_BASE_RANGE / 3) {
        let s = (total_br - coded).min(3);
        enc.encode_symbol(s as usize, br_cdf);
        coded += s;
        if s < 3 {
            break;
        }
    }
}

/// Adaptive eob==0 DC-only tail (shared by all adaptive coef coders). Codes the
/// eob_pt=0 symbol, the DC base token, optional hi_tok ladder, the dc_sign, and
/// the Golomb residual — adapting each CDF.
fn encode_dc_tail(
    enc: &mut OdEcEncoder,
    level: i32,
    eob_bin_cdf: &mut [u16],
    base_eob: &mut [u16],
    dc_sign: &mut [u16],
    br0: &mut [u16],
) {
    enc.encode_symbol(0, eob_bin_cdf);
    let m = level.unsigned_abs();
    let base = m.min(3);
    enc.encode_symbol(base as usize - 1, base_eob);
    if base == 3 {
        encode_hi_tok(enc, m, br0);
    }
    enc.encode_symbol((level < 0) as usize, dc_sign);
    if m >= 15 {
        encode_golomb(enc, m - 15);
    }
}

/// **Adaptive** `TX_8X8` coefficient encoder (dav1d-compatible CDF adaptation):
/// every symbol is coded against the persistent CDF in `cdfs` and adapts it.
/// Used by the full-image path. TX_8X8 is coef class 1.
fn encode_tx8_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
) -> u8 {
    let pl = chroma as usize;
    let mut eob = 0usize;
    for (i, &rc) in SCAN_8X8.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    if !chroma {
        enc.encode_symbol(1, &mut cdfs.txtp[y_mode]); // luma: txtp idx 1 -> DCT_DCT
    }
    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        let eb = if chroma {
            &mut cdfs.eob_bin_64_c
        } else {
            &mut cdfs.eob_bin_64_l
        };
        encode_dc_tail(
            enc,
            cf[0],
            eb,
            &mut cdfs.eob_base[1][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[1][pl][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    {
        let eb = if chroma {
            &mut cdfs.eob_bin_64_c
        } else {
            &mut cdfs.eob_bin_64_l
        };
        enc.encode_symbol(eob_bin, eb);
    }
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384); // extra eob bits, equiprobable
        }
    }
    let mut levels = [0u8; 80];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_8X8[eob];
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1; // 0,1,2
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][pl][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][pl][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X8[i];
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][pl][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][pl][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][pl][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X8[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// TX_16X16 coefficient coder (class 2). Parameterized port of
/// `encode_tx8_coeffs_adapt`: 256 coeffs in `SCAN_16X16` order, stride 16, coef
/// CDF class 2, eob_pt over 256 (`eob_bin_256`), eob ctx thresholds 256>>3=32 /
/// 256>>2=64, and the 2D coeff-base context reuses `LO_CTX_OFF` + `get_lo_ctx_2d`
/// at stride 16 (the libaom 16x16 offset table equals that 5x5 region). Used for
/// 4:4:4 luma (chroma=false) and 4:4:4 chroma (chroma=true).
fn encode_tx16_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 256],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
) -> u8 {
    let pl = chroma as usize;
    let mut eob = 0usize;
    for (i, &rc) in SCAN_16X16.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 1
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 0
    if !chroma {
        enc.encode_symbol(1, &mut cdfs.txtp16[y_mode]); // luma TX_16X16: DTT4_IDTX idx 1 -> DCT_DCT
    }
    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        let eb = if chroma {
            &mut cdfs.eob_bin_256_c
        } else {
            &mut cdfs.eob_bin_256_l
        };
        encode_dc_tail(
            enc,
            cf[0],
            eb,
            &mut cdfs.eob_base[2][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[2][pl][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    {
        let eb = if chroma {
            &mut cdfs.eob_bin_256_c
        } else {
            &mut cdfs.eob_bin_256_l
        };
        enc.encode_symbol(eob_bin, eb);
    }
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 320]; // stride 16, neighbour reads up to (x+2)*16+(y+2)=289
    let ctx_e = 1 + (eob > 32) as usize + (eob > 64) as usize;
    let rc = SCAN_16X16[eob];
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1; // 0,1,2
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][pl][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][pl][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X16[i];
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][pl][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][pl][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][pl][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X16[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// TX_32X32 coefficient coder (class 3). Mirror of `encode_tx16_coeffs_adapt`
/// for 1024 coeffs in `SCAN_32X32` order, stride 32, coef CDF class 3, eob_pt
/// over 1024 (`eob_bin_1024`), eob ctx thresholds 1024>>3=128 / 1024>>2=256.
/// Intra TX_32X32 codes NO tx-type symbol: dav1d derives `t_dim->max + intra >=
/// TX_64X64` -> DCT_DCT, so unlike the 16x16 luma path there is no `txtp`
/// symbol. The 2D coeff-base context reuses `LO_CTX_OFF` + `get_lo_ctx_2d` at
/// stride 32 (the libaom offset table saturates to 21 outside the 5x5 corner,
/// identical for every square transform). Used for 4:4:4 luma and chroma.
fn encode_tx32_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 1024],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let pl = chroma as usize;
    let mut eob = 0usize;
    for (i, &rc) in SCAN_32X32.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]); // all_zero = 1
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]); // all_zero = 0
    // NO txtp symbol for intra TX_32X32 (DCT_DCT implied).
    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        let eb = if chroma {
            &mut cdfs.eob_bin_1024_c
        } else {
            &mut cdfs.eob_bin_1024_l
        };
        encode_dc_tail(
            enc,
            cf[0],
            eb,
            &mut cdfs.eob_base[3][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[3][pl][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    {
        let eb = if chroma {
            &mut cdfs.eob_bin_1024_c
        } else {
            &mut cdfs.eob_bin_1024_l
        };
        enc.encode_symbol(eob_bin, eb);
    }
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 1156]; // stride 32, neighbour reads up to (x+2)*32+(y+2)
    let ctx_e = 1 + (eob > 128) as usize + (eob > 256) as usize;
    let rc = SCAN_32X32[eob];
    let (ex, ey) = (rc >> 5, rc & 31);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1; // 0,1,2
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][pl][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][pl][bc]);
    }
    levels[ex * 32 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_32X32[i];
        let (x, y) = (rc_i >> 5, rc_i & 31);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 32);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][pl][bc as usize]);
        }
        levels[x * 32 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][pl][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[32] as u32 + levels[33] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][pl][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_32X32[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// DC-only tail for the eob==0 path with a raw signed level.
/// 4:2:2 chroma coefficient coder for an `RTX_4X8` block (4 wide x 8 tall, 32
/// coeffs, `cf[fx*8+fy]`). `RTX_4X8` shares coef-CDF class ctx=1 with `TX_8X8`,
/// so the base/br/eob-base/eob-hi/dc-sign/skip CDFs are reused; only the eob_pt
/// CDF (`eob_bin_32`), the scan, the lo-ctx offsets (w<h) and the eob-ctx
/// thresholds differ. Returns the dav1d coef neighbour-context byte.
fn encode_4x8_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 32],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let mut eob = 0usize;
    for (i, &rc) in SCAN_4X8.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_32_c,
            &mut cdfs.eob_base[1][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[1][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_32_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob coeff: eob-ctx thresholds use sw*sh = imin(w,8)*imin(h,8) = 1*2 = 2
    let ctx_e = 1 + (eob > 4) as usize + (eob > 8) as usize;
    let rc = SCAN_4X8[eob];
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X8[i];
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X8[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// 4:2:2 chroma coefficient coder for an `RTX_8X16` block (8 wide x 16 tall,
/// 128 coeffs, `cf[fx*16+fy]`). `txsize_sqr_up[TX_8X16] = TX_16X16`, so it uses
/// the coef-CDF class ctx=2 (the same base/br/eob/skip CDFs as luma TX_16X16),
/// the chroma `eob_multi128` bins, the `w<h` level-offset table at stride 16,
/// and eob-ctx thresholds N>>3 / N>>2 = 16 / 32. Chroma infers the transform
/// type (no txtp symbol). Returns the coef neighbour byte.
fn encode_8x16_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 128],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let mut eob = 0usize;
    for (i, &rc) in SCAN_8X16.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 1
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_128_c,
            &mut cdfs.eob_base[2][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[2][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_128_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 200];
    // eob coeff: 128 coeffs -> thresholds 128>>3 = 16, 128>>2 = 32
    let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
    let rc = SCAN_8X16[eob];
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][1][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X16[i];
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][1][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X16[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

fn encode_4x4_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 16],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let mut eob = 0usize;
    for (i, &rc) in SCAN_4X4.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[0][skip_ctx]);
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[0][skip_ctx]);

    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_16_c,
            &mut cdfs.eob_base[0][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[0][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_16_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[0][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob-ctx thresholds: sw*sh = imin(w,8)*imin(h,8) = 1*1 = 1
    let ctx_e = 1 + (eob > 2) as usize + (eob > 4) as usize;
    let rc = SCAN_4X4[eob];
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[0][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[0][1][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X4[i];
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[0][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[0][1][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[0][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[5] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[0][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X4[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

static PART_SPLIT_CDF: [[[u16; 9]; 4]; 3] = [
    [
        [
            20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724,
        ],
        [6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104],
        [5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238],
        [870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332],
    ],
    [
        [
            18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544,
        ],
        [7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052],
        [6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047],
        [1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899],
    ],
    [
        [
            15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902,
        ],
        [7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834],
        [5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117],
        [2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171],
    ],
];
static PART_BL8_CDF: [[u16; 3]; 4] = [
    [19132, 25510, 30392],
    [13928, 19855, 28540],
    [12522, 23679, 28629],
    [9896, 18783, 25853],
];

/// 8x8 DC_PRED from a reconstructed plane (stride 64). `(ox, oy)` pixel origin.
fn dc_pred_8x8(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(8)
                    .sum::<i32>();
            (s + 8) >> 4
        }
        (true, false) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            (s + 4) >> 3
        }
        (false, true) => {
            let mut s = 0i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            (s + 4) >> 3
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 4x4 chroma block (dav1d `dc_gen`, 8-bit). w==h==4 is a
/// power of two so no reciprocal multiply is needed.
fn dc_pred_4x4(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 4i32; // (4+4)>>1
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(4)
                    .sum::<i32>();
            s >> 3 // ctz(8)
        }
        (true, false) => {
            let mut s = 2i32;
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>();
            s >> 2
        }
        (false, true) => {
            let mut s = 2i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(4)
                .sum::<i32>();
            s >> 2
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 4-wide x 8-tall chroma block (dav1d `dc_gen`, 8-bit).
/// w+h = 12 is not a power of two, so the both-edges case uses the reciprocal
/// multiply (ctz(12)=2 shift, then *0x5556>>16 since 8 is not > 2*4).
fn dc_pred_4x8(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 6i32; // (4+8)>>1
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            s >>= 2; // ctz(4+8)
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 2i32; // 4>>1
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>();
            s >> 2 // ctz(4)
        }
        (false, true) => {
            let mut s = 4i32; // 8>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            s >> 3 // ctz(8)
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC predictor for an 8-wide x 16-tall chroma block (4:2:2 `RTX_8X16`). Mirrors
/// dav1d/AV1 DC_PRED: average of the 8 above + 16 left reconstructed neighbours
/// (w+h = 24 = 8*3, so `>>3` then the `*0x5556>>16` divide-by-3); single-edge
/// and no-edge cases fall back to the available average or 128.
fn dc_pred_8x16(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 12i32; // (8+16)>>1
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            s >>= 3; // ctz(8+16) = ctz(24) = 3
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 4i32; // 8>>1
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            s >> 3 // ctz(8)
        }
        (false, true) => {
            let mut s = 8i32; // 16>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            s >> 4 // ctz(16)
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// `forward_dct_quant_8x8`: orthonormal float DCT (rows then cols), scaled, then
/// /q (dc_q for the (0,0) coefficient, ac_q otherwise). Output in dav1d order
/// `cf[u*16+v]`. The scale is calibrated so the round-trip through the exact
/// integer inverse recovers the residual; only the encoder uses this (recon is
/// the exact inverse), so its precision does not affect bit-exactness.
pub(crate) fn forward_dct_quant_16x16(residual: &mut [i32; 256], q: &impl Dct) {
    // forward_dct_quant_16x16_t(residual, q).0
    dct16x16(residual, q)
}

/// As [`forward_dct_quant_16x16`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_16x16_t(
    residual: &[i32; 256],
    q: &impl Dct,
) -> ([i32; 256], [f64; 256]) {
    dct16x16_t(residual, q)
}

/// DC prediction for a 16x16 block (mirror of `dc_pred_8x8`).
fn dc_pred_16x16(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(16)
                    .sum::<i32>();
            (s + 16) >> 5
        }
        (true, false) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            (s + 8) >> 4
        }
        (false, true) => {
            let mut s = 0i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            (s + 8) >> 4
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// Forward DCT + quantize a 32x32 residual via the shared integer DCT in
/// `crate::dct`. Recon is the exact integer inverse.
pub(crate) fn forward_dct_quant_32x32(residual: &mut [i32; 1024], q: &impl Dct) {
    dct32x32(residual, q)
}

pub(crate) fn forward_dct_quant_32x32_t(
    residual: &[i32; 1024],
    q: &impl Dct,
) -> ([i32; 1024], [f64; 1024]) {
    dct32x32_t(residual, q)
}

/// DC prediction for a 32x32 block (mirror of `dc_pred_16x16`).
fn dc_pred_32x32(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..32].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(32)
                    .sum::<i32>();
            (s + 32) >> 6
        }
        (true, false) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..32].iter().sum::<i32>();
            (s + 16) >> 5
        }
        (false, true) => {
            let mut s = 0i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(32)
                .sum::<i32>();
            (s + 16) >> 5
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// As [`forward_dct_quant_4x8`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_4x8_t(
    residual: &[i32; 32],
    q: &impl Dct,
) -> ([i32; 32], [f64; 32]) {
    dct4x8_t(residual, q)
}

/// As [`forward_dct_quant_8x16`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_8x16_t(
    residual: &[i32; 128],
    q: &impl Dct,
) -> ([i32; 128], [f64; 128]) {
    dct8x16_t(residual, q)
}

pub(crate) fn forward_dct_quant_16x32_t(
    residual: &[i32; 512],
    q: &impl Dct,
) -> ([i32; 512], [f64; 512]) {
    dct16x32_t(residual, q)
}

/// DC predictor for a 16-wide x 32-tall chroma block (4:2:2 `RTX_16X32`).
/// Mirrors `dc_pred_8x16`: sum 16 above + 32 left = 48 = 16*3 samples.
fn dc_pred_16x32(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 24i32; // (16+32)>>1
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(32)
                .sum::<i32>();
            s >>= 4; // ctz(48) = 4
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 8i32; // 16>>1
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            s >> 4
        }
        (false, true) => {
            let mut s = 16i32; // 32>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(32)
                .sum::<i32>();
            s >> 5
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// 4:2:2 chroma coefficient coder for an `RTX_16X32` block (16 wide x 32 tall,
/// 512 coeffs, `cf[fx*32+fy]`). `txsize_sqr_up[RTX_16X32] = TX_32X32`, so the
/// base/br/eob-base/eob-hi/dc-sign/skip CDFs are coef-CDF class `ctx=3`; only
/// the eob_pt CDF (`eob_bin_512`), the scan, the w<h lo-ctx offsets and the
/// eob-ctx thresholds (512>>3=64, 512>>2=128) differ from TX_32X32.
fn encode_16x32_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 512],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let mut eob = 0usize;
    for (i, &rc) in SCAN_16X32.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]);
        return 0x40;
    }
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]);
    let cul: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_512_c,
            &mut cdfs.eob_base[3][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[3][1][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_512_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 640]; // stride 32, max read (15+2)*32+(31+2)=577
    let ctx_e = 1 + (eob > 64) as usize + (eob > 128) as usize;
    let rc = SCAN_16X32[eob];
    let (ex, ey) = (rc >> 5, rc & 31);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][1][bc]);
    }
    levels[ex * 32 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X32[i];
        let (x, y) = (rc_i >> 5, rc_i & 31);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 32);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][1][bc as usize]);
        }
        levels[x * 32 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[32] as u32 + levels[33] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X32[i]];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// As [`forward_dct_quant_4x4`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_4x4_t(
    residual: &[i32; 16],
    q: &impl Dct,
) -> ([i32; 16], [f64; 16]) {
    dct4x4_t(residual, q)
}

/// 4x4 scan order (dav1d `scan_4x4`): scan index -> raster rc = fx*4 + fy.
static SCAN_4X4: [usize; 16] = [0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15];

/// 4x8 scan order (dav1d `scan_4x8`): scan index -> raster rc = fx*8 + fy.
static SCAN_4X8: [usize; 32] = [
    0, 8, 1, 16, 9, 2, 24, 17, 10, 3, 25, 18, 11, 4, 26, 19, 12, 5, 27, 20, 13, 6, 28, 21, 14, 7,
    29, 22, 15, 30, 23, 31,
];

/// `dav1d_lo_ctx_offsets[2]` (w < h) — coeff_base position offset for RTX_4X8.
static LO_CTX_OFF_WLH: [[u32; 5]; 5] = [
    [0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [6, 6, 21, 21, 21],
    [6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21],
];

fn get_partition_ctx(a: &[u8], l: &[u8], bl: usize, x8: usize, y8: usize) -> usize {
    let sh = 4 - bl;
    ((a[x8] >> sh) & 1) as usize + ((((l[y8] >> sh) & 1) as usize) << 1)
}

/// Probability (0..32768) for the binary `is_split` decision dav1d reads at a
/// frame edge when only one of have_h/have_v is set. `top` selects
/// `gather_top_partition_prob` (have_h only → split-or-horz), else
/// `gather_left_partition_prob` (have_v only → split-or-vert). Operates on the
/// 9-value default partition CDF for the relevant block level/context.
fn gather_split_prob_icdf(cdf: &[u16], top: bool) -> u16 {
    let v = |s: usize| cdf[s] as i32; // live CDF is already icdf form
    let out = if top {
        (v(1) - v(4)) + v(5) + (v(8) - v(7))
    } else {
        (v(0) - v(1)) + (v(2) - v(6)) + (v(7) - v(8))
    };
    out.clamp(1, 32767) as u16
}

/// Whole-frame lossy encoder state. Context arrays are indexed by absolute frame
/// coordinates: the above arrays persist down the superblock rows, and the left
/// arrays are naturally fresh per SB row (each row occupies a distinct
/// coordinate range), mirroring dav1d's per-SB-row left reset.
struct LossyTile<'a> {
    bd: u8,
    quant: Quant,
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
    enc: OdEcEncoder,
    cdfs: Cdfs,
}

impl<'a> LossyTile<'a> {
    fn new(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
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
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
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
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
        }
    }

    /// 4:2:2 tile: luma is full w x h, chroma planes are subsampled to (w/2) x h.
    /// `src[1]`/`src[2]` must already be the half-width chroma planes.
    fn new_422(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        let cw = w / 2;
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
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
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
        }
    }

    /// 4:2:0 tile: luma is full w x h, chroma planes are subsampled to
    /// (w/2) x (h/2). `src[1]`/`src[2]` must already be the quarter-size planes.
    fn new_420(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3]) -> Self {
        let (cw, ch) = (w / 2, h / 2);
        LossyTile {
            bd,
            quant: Quant::new(q, bd),
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
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
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
            let ca = (0..4).any(|k| a[bx4 + k] != 0x40) as usize;
            let cl = (0..4).any(|k| l[by4 + k] != 0x40) as usize;
            7 + ca + cl
        }
    }

    /// dc_sign context for a 16x16 transform (4-unit footprint, baseline -8).
    fn dc_sign_ctx_16(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = (0..4).map(|k| (a[bx4 + k] >> 6) as i32).sum();
        let suml: i32 = (0..4).map(|k| (l[by4 + k] >> 6) as i32).sum();
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
    fn prefer_16x16(&self, x8: usize, y8: usize) -> bool {
        if self.mono {
            return false; // monochrome codes 8x8 luma blocks only
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
    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
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
        let mut lpred_arr = [0i32; 256];
        let mut lcf = [0i32; 256];
        let mut best_eff = f64::INFINITY;
        for &m in nd_modes() {
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
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
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
            let rr = idct_dequant_16x16(&cf, &self.quant);
            let mut sse = 0i64;
            for (ry, (prow, rrow)) in pred.chunks_exact(16).zip(rr.chunks_exact(16)).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    sse += (d * d) as i64;
                }
            }
            let bits = block_rate_bits(&cf, &SCAN_16X16) + mode_signal_bits(m);
            let cost = sse as f64 + mlam * bits;
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
            }
        }
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block16_420(x8, y8, &lcf, &lpred_arr, best_mode, luma_zero);
        } else if self.ss422 {
            self.code_block16_422(x8, y8, &lcf, &lpred_arr, best_mode, luma_zero);
        } else {
            self.code_block16_444(x8, y8, &lcf, &lpred_arr, best_mode, luma_zero);
        }
    }

    /// Shared header + luma for a TX_16X16 block: codes the block-level skip
    /// flag, `DC_PRED` y/uv modes, the luma TX_16X16 coefficients, updates the
    /// 4-unit (16-sample) luma skip/coef footprint, and reconstructs luma. The
    /// caller has already decided `block_skip` (needs all planes) and passes the
    /// luma coefficients + DC prediction.
    /// Emit the chroma `uv_mode` symbol: plain DC (`None`) or CfL (`Some(alphas)`),
    /// in which case also the joint-sign and per-plane magnitude symbols.
    fn emit_uv_mode(&mut self, y_mode: usize, cfl: Option<[i32; 2]>) {
        match cfl {
            Some(a) => {
                self.enc
                    .encode_symbol(CFL_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
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
                let sign = su * 3 + sv; // 1..=8 (both-zero excluded by construction)
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
                    .encode_symbol(DC_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
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
        cfl: Option<[i32; 2]>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(y_mode, cfl);
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
            encode_tx16_coeffs_adapt(&mut self.enc, &mut self.cdfs, lcf, false, sk, ds, y_mode)
        };
        self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
        self.l_coef[0][by4..by4 + 4].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 256]
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

    /// 4:4:4: chroma is also 16x16 (one TX_16X16 per plane).
    fn code_block16_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
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
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.quant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.quant.dc_q() as f64,
                self.quant.ac_q() as f64,
                &SCAN_16X16,
                trellis_lambda(),
            );
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
                self.quant.dc_q() as f64,
                self.quant.ac_q() as f64,
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
                let dcrr = idct_dequant_16x16(&ccf[ci], &self.quant);
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
                let (mut q, qt) = forward_dct_quant_16x16_t(&resid, &self.quant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X16, lam);
                let rr = idct_dequant_16x16(&q, &self.quant);
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
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                cfl_opt = Some(cfl_a);
                ccf[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred16[ci] = [cpred[ci]; 256];
                }
            }
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(x8, y8, lcf, lpred, y_mode, block_skip, cfl_opt);
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_16(plane, bx4, by4);
                encode_tx16_coeffs_adapt(&mut self.enc, &mut self.cdfs, &ccf[ci], true, sk, ds, 0)
            };
            self.a_coef[plane][bx4..bx4 + 4].fill(res_ctx);
            self.l_coef[plane][by4..by4 + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else {
                idct_dequant_16x16(&ccf[ci], &self.quant)
            };
            for (ry, (prow, rrow)) in cpred16[ci]
                .chunks_exact(16)
                .zip(rr.chunks_exact(16))
                .enumerate()
            {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                    *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                }
            }
        }
    }

    /// 4:2:0: a 16x16 luma region maps to an 8x8 chroma region per plane, coded
    /// with the existing `TX_8X8` chroma path (coef-CDF class 1). The chroma tx
    /// equals the chroma block size, so the txb_skip ctx offset is 7 — identical
    /// to the 4:4:4 8x8 chroma case but indexed on the chroma 4-unit grid (which,
    /// in 4:2:0, lands at `bx4c = x8`, `by4c = y8`, a 2-unit footprint).
    fn code_block16_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let mut ccf = [[0i32; 64]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 64];
            for (ry, drow) in resid.chunks_exact_mut(8).enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.quant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.quant.dc_q() as f64,
                self.quant.ac_q() as f64,
                &SCAN_8X8,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(x8, y8, lcf, lpred, y_mode, block_skip, None);
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                // 8x8 chroma footprint on the chroma grid: reuse the 8x8 helpers.
                let sk = self.skip_ctx(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx(plane, bx4c, by4c);
                encode_tx8_coeffs_adapt(&mut self.enc, &mut self.cdfs, &ccf[ci], true, sk, ds, 0)
            };
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 2].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 64]
            } else {
                idct_dequant_8x8(&ccf[ci], &self.quant)
            };
            for (ry, rrow) in rr.chunks_exact(8).enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                }
            }
        }
    }

    /// 4:2:2: a 16x16 luma region maps to an 8-wide x 16-tall chroma region per
    /// plane (`RTX_8X16`, coef-CDF class 2). Chroma is full-height, half-width, so
    /// the chroma block sits at `(cx, py)` with `cx = px/2` and spans 2 coef units
    /// horizontally and 4 vertically on the chroma grid.
    fn code_block16_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
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
            let (q, qt) = forward_dct_quant_8x16_t(&resid, &self.quant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.quant.dc_q() as f64,
                self.quant.ac_q() as f64,
                &SCAN_8X16,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(x8, y8, lcf, lpred, y_mode, block_skip, None);
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
                idct_dequant_8x16(&ccf[ci], &self.quant)
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
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let cx = px / 2; // chroma column for 4:2:2

        // Forward-transform/quantize all planes up front to decide block skip.
        // Luma is always 8x8; chroma is 8x8 (4:4:4) or 4x8 (4:2:2).
        // Luma 8x8: search the non-directional intra modes (DC + SMOOTH*/PAETH)
        // and keep the one minimising pixel SSE + lambda * estimated bits. The
        // chosen prediction is per-pixel; reconstruction uses the same array so
        // the decoder (which re-derives the identical prediction) stays bit-exact.
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
            trellis_lambda(),
        );
        let mlam = mode_lambda() * acq * acq;
        let mut best_mode = DC_PRED;
        let mut lpred_arr = [0i32; 64];
        let mut lcf = [0i32; 64];
        let mut best_eff = f64::INFINITY;
        for &m in nd_modes() {
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
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
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
                self.dc_sign_ctx(0, px / 4, py / 4),
            );
            let rr = idct_dequant_8x8(&cf, &self.quant);
            let mut sse = 0i64;
            for (ry, (prow, rrow)) in pred.chunks_exact(8).zip(rr.chunks_exact(8)).enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    sse += (d * d) as i64;
                }
            }
            let bits = block_rate_bits(&cf, &SCAN_8X8) + mode_signal_bits(m);
            let cost = sse as f64 + mlam * bits;
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
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
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.quant);
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
                let (q, qt) = forward_dct_quant_4x8_t(&resid, &self.quant);
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
                let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.quant);
                ccf8[ci] = q;
                trellis_optimize(&mut ccf8[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            }
        }

        // 4:4:4 chroma-from-luma: try predicting U/V from the reconstructed luma
        // block (scaled, mean-removed) and pick CfL over plain DC per block.
        let mut cpred444 = [[0i32; 64]; 2];
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
                let dcrr = idct_dequant_8x8(&ccf8[ci], &self.quant);
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
                let (mut q, qt) = forward_dct_quant_8x8_t(&resid, &self.quant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                let rr = idct_dequant_8x8(&q, &self.quant);
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
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&best_mode) {
            // angle_delta = 0 (symbol index 3); 8x8 satisfies the size condition
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[best_mode - V_PRED]);
        }
        if !self.mono {
            self.emit_uv_mode(best_mode, if use_cfl { Some(cfl_alpha_uv) } else { None });
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
            )
        };
        self.a_coef[0][bx4] = lres_ctx;
        self.a_coef[0][bx4 + 1] = lres_ctx;
        self.l_coef[0][by4] = lres_ctx;
        self.l_coef[0][by4 + 1] = lres_ctx;
        let lrr = if block_skip {
            [0i32; 64]
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
                    idct_dequant_4x4(&ccf44[ci], &self.quant)
                };
                for (ry, rrow) in rr.chunks_exact(4).enumerate() {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                        *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
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
                    idct_dequant_4x8(&ccf48[ci], &self.quant)
                };
                for (ry, rrow) in rr.chunks_exact(4).enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                    for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                        *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
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
                    )
                };
                self.a_coef[plane][bx4] = res_ctx;
                self.a_coef[plane][bx4 + 1] = res_ctx;
                self.l_coef[plane][by4] = res_ctx;
                self.l_coef[plane][by4 + 1] = res_ctx;
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    idct_dequant_8x8(&ccf8[ci], &self.quant)
                };
                for (ry, (prow, rrow)) in cpred444[ci]
                    .chunks_exact(8)
                    .zip(rr.chunks_exact(8))
                    .enumerate()
                {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                        *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
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
        let ca = (0..4).any(|k| a[bx4c + k] != 0x40) as usize;
        let cl = (0..8).any(|k| l[by4c + k] != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_16X32: 4 units wide, 8 tall, baseline -12.
    fn dc_sign_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = (0..4).map(|k| (a[bx4c + k] >> 6) as i32).sum();
        let suml: i32 = (0..8).map(|k| (l[by4c + k] >> 6) as i32).sum();
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
        for &m in nd_modes() {
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
            }
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
        cfl: Option<[i32; 2]>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(y_mode, cfl);
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
            self.quant.dc_q() as f64,
            self.quant.ac_q() as f64,
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
            let (q, qt) = forward_dct_quant_32x32_t(&cresid, &self.quant);
            ccf[ci] = q;
            trellis_optimize(&mut ccf[ci], &qt, dcq, acq, &SCAN_32X32, lam);
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
                let dcrr = idct_dequant_32x32(&ccf[ci], &self.quant);
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
                let (mut q, qt) = forward_dct_quant_32x32_t(&resid, &self.quant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_32X32, lam);
                let rr = idct_dequant_32x32(&q, &self.quant);
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
        let use_cfl = (cfl_a[0] != 0 || cfl_a[1] != 0)
            && cfl_cost[0] + cfl_cost[1] + mlam * cfl_sig < dc_cost[0] + dc_cost[1];
        let (cf_use, pred_dc, cfl_opt): (&[[i32; 1024]; 2], [i32; 2], Option<[i32; 2]>) = if use_cfl
        {
            (&cfl_ccf, cdc, Some(cfl_a))
        } else {
            (&ccf, cdc, None)
        };
        let block_skip =
            luma_zero && cf_use[0].iter().all(|&c| c == 0) && cf_use[1].iter().all(|&c| c == 0);
        self.code_header_luma32(x8, y8, lcf, lpred, y_mode, block_skip, cfl_opt);
        for ci in 0..2 {
            let plane = ci + 1;
            let cres = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_32(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_32(plane, bx4, by4);
                encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, &cf_use[ci], true, sk, ds)
            };
            self.a_coef[plane][bx4..bx4 + 8].fill(cres);
            self.l_coef[plane][by4..by4 + 8].fill(cres);
            let crr = if block_skip {
                [0i32; 1024]
            } else {
                idct_dequant_32x32(&cf_use[ci], &self.quant)
            };
            for (ry, (prow, rrow)) in cfl_pred[ci]
                .chunks_exact(32)
                .zip(crr.chunks_exact(32))
                .enumerate()
            {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                for ((dv, &cp), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                    let base = if use_cfl { cp } else { pred_dc[ci] };
                    *dv = (base + rv).clamp(0, (1 << self.bd) - 1);
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
        let mut ccf = [[0i32; 256]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = dc_pred_16x16(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 256];
            for (ry, drow) in resid.chunks_exact_mut(16).enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.quant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.quant.dc_q() as f64,
                self.quant.ac_q() as f64,
                &SCAN_16X16,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(x8, y8, lcf, lpred, y_mode, block_skip, None);
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx_16(plane, bx4c, by4c);
                encode_tx16_coeffs_adapt(&mut self.enc, &mut self.cdfs, &ccf[ci], true, sk, ds, 0)
            };
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else {
                idct_dequant_16x16(&ccf[ci], &self.quant)
            };
            for (ry, rrow) in rr.chunks_exact(16).enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
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
            let (q, qt) = forward_dct_quant_16x32_t(&resid, &self.quant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.quant.dc_q() as f64,
                self.quant.ac_q() as f64,
                &SCAN_16X32,
                trellis_lambda(),
            );
        }
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(x8, y8, lcf, lpred, y_mode, block_skip, None);
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
                idct_dequant_16x32(&ccf[ci], &self.quant)
            };
            for (ry, rrow) in rr.chunks_exact(16).enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                    *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                }
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
pub(crate) fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Pad a `w`×`h` plane to `w8`×`h8` (≥ originals) by replicating the last
/// in-frame row/column. AV1's coded block grid is always 8-pixel aligned
/// (`MiCols = ((w+7)>>3)<<1`), so frames whose dimensions are not multiples of 8
/// are coded on the padded grid and the decoder crops back to the signalled
/// frame size. Edge replication keeps the (cropped-away) padding cheap to code.
pub(crate) fn pad_to_mult8<T: Copy>(src: &[T], w: usize, h: usize, w8: usize, h8: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w8 * h8);
    for y in 0..h8 {
        let sy = y.min(h - 1);
        let row = &src[sy * w..sy * w + w];
        for x in 0..w8 {
            out.push(row[x.min(w - 1)]);
        }
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
    };
    for sb_y in (0..r.th).step_by(64) {
        for sb_x in (0..r.tw).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let payload = tile.enc.done();
    TileOut {
        payload,
        recon: tile.recon,
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
            .map(|r| encode_one_tile(base_q_idx, bd, w8, cw8, sub_x, sub_y, mono, src, r))
            .collect()
    } else {
        let mut slots: Vec<Option<TileOut>> = (0..n).map(|_| None).collect();
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (rs, os) in rects.chunks(chunk).zip(slots.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (r, o) in rs.iter().zip(os.iter_mut()) {
                        *o = Some(encode_one_tile(
                            base_q_idx, bd, w8, cw8, sub_x, sub_y, mono, src, r,
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
    for (r, out) in rects.iter().zip(outs) {
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
    ycbcr: bool,
    threads: usize,
) -> Vec<u8> {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    // Code on the 8-pixel-aligned grid; signal the exact frame size in the header.
    let (w8, h8) = (align8(w), align8(h));
    let src = [
        pad_to_mult8(luma, w, h, w8, h8),
        pad_to_mult8(u, w, h, w8, h8),
        pad_to_mult8(v, w, h, w8, h8),
    ];
    let (payload, _recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 0, 0, false, threads);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    let mc = if ycbcr { 6 } else { 0 };
    let profile = if bd == 12 { 2 } else { 1 };
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32, h as u32, profile, bd, mc, 0, 0,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, false));
    bytes
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
    threads: usize,
) -> Vec<u8> {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let cw = w.div_ceil(2);
    assert_eq!(u.len(), cw * h);
    assert_eq!(v.len(), cw * h);
    let (w8, h8) = (align8(w), align8(h));
    let cw8 = w8 / 2; // chroma coded width (aligned to luma 8x8 -> 4px chroma)
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| -> Vec<i32> { pad_to_mult8(p, cw, h, cw8, h8) };
    let src = [luma_p, pad_c(u), pad_c(v)];
    let (payload, _recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 1, 0, false, threads);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    // profile 2 @ 8-bit => I422 (forced), BT.601 full-range YCbCr
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32, h as u32, 2, bd, 6, 1, 0,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, false));
    bytes
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
    threads: usize,
) -> Vec<u8> {
    assert_eq!(luma.len(), w * h);
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    assert_eq!(u.len(), cw * ch);
    assert_eq!(v.len(), cw * ch);
    let (w8, h8) = (align8(w), align8(h));
    let (cw8, ch8) = (w8 / 2, h8 / 2); // chroma coded size (4px chroma per 8px luma)
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| -> Vec<i32> { pad_to_mult8(p, cw, ch, cw8, ch8) };
    let src = [luma_p, pad_c(u), pad_c(v)];
    let (payload, _recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 1, 1, false, threads);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    // profile 0 @ 8-bit => I420 (forced), BT.601 full-range YCbCr
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32,
        h as u32,
        if bd == 12 { 2 } else { 0 },
        bd,
        6,
        1,
        1,
    ));
    bytes.extend_from_slice(&assemble_frame_obus(base_q_idx, &plan, &payload, false));
    bytes
}

/// Encode a **monochrome** (single luma plane) AV1 still — the form AVIF uses
/// for an alpha auxiliary image. `luma` is the `w*h` grayscale plane (e.g. the
/// alpha channel). `full_range` sets `color_range` (alpha is normally full
/// range). `base_q_idx` controls quality (use a small value, or the lossless
/// path, for exact alpha). `threads`: `0` = all cores, `1` = serial, `N` = up to
/// N (tiles are subdivided toward the thread count, exactly like the color
/// encoders, so large alpha planes tile and parallelise too).
pub(crate) fn encode_av1_mono_image(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
) -> Vec<u8> {
    let (bytes, _recon, _w8, _h8) =
        encode_av1_mono_image_recon_dbg(base_q_idx, bd, w, h, luma, full_range, threads);
    bytes
}

/// Debug variant of [`encode_av1_mono_image`] also returning the encoder's
/// reconstruction (luma, `w8*h8`) and the padded dimensions, for bit-exactness
/// checks against a decoder.
#[doc(hidden)]
pub(crate) fn encode_av1_mono_image_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
) -> (Vec<u8>, Vec<i32>, usize, usize) {
    assert_eq!(luma.len(), w * h, "luma plane must be w*h");
    assert!(w > 0 && h > 0, "width/height must be non-zero");
    let (w8, h8) = (align8(w), align8(h));
    let src = [pad_to_mult8(luma, w, h, w8, h8), Vec::new(), Vec::new()];
    let (payload, recon, plan) =
        encode_lossy_tilegroup(base_q_idx, bd, w8, h8, &src, 0, 0, true, threads);
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
        let (small, _r, _w8, _h8) = encode_av1_mono_image_recon_dbg(24, 8, w, h, &luma, true, 1);
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
        let (big, _r2, _w82, _h82) = encode_av1_mono_image_recon_dbg(24, 8, bw, bh, &bl, true, 1);
        let bt = obu_types(&big);
        assert!(
            bt.contains(&3) && bt.contains(&4),
            "mono multi-tile -> 3+4: {bt:?}"
        );
        assert!(
            !bt.contains(&6),
            "mono multi-tile must not use OBU_FRAME: {bt:?}"
        );

        let (s1, r1, _, _) = encode_av1_mono_image_recon_dbg(24, 8, bw, bh, &bl, true, 1);
        let (s2, r2, _, _) = encode_av1_mono_image_recon_dbg(24, 8, bw, bh, &bl, true, 2);
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
        let serial = encode_av1_lossy_image_420(80, 10, w, h, &luma, &u, &v, 1);
        let par2a = encode_av1_lossy_image_420(80, 10, w, h, &luma, &u, &v, 2);
        let par2b = encode_av1_lossy_image_420(80, 10, w, h, &luma, &u, &v, 2);
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
        let serial = encode_av1_lossy_image_420(80, 10, w, h, &luma, &u, &v, 1);
        assert!(
            obu_types(&serial).contains(&6),
            "serial small frame should be a single OBU_FRAME"
        );

        // threads=4 -> subdivided into tiles -> OBU_FRAME_HEADER (3) + TILE_GROUP (4).
        let threaded = encode_av1_lossy_image_420(80, 10, w, h, &luma, &u, &v, 4);
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
            1,
        );
        assert_eq!(bytes.len(), 48, "4:2:0 stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4401, "4:2:0 stream bytes drifted");
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
            1,
        );
        assert_eq!(bytes.len(), 264, "4:2:0 8x8-leaves stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 34928, "4:2:0 8x8-leaves stream bytes drifted");
    }

    #[test]
    fn lossy_422_16x16_matches_dav1d_verified_bytes() {
        // 4:2:2 (profile 2): a smooth 16x16 luma region now codes as one TX_16X16
        // luma block + one RTX_8X16 (8 wide x 16 tall) chroma block per plane. dav1d
        // 1.4.1 decodes this to a C422 frame ("Decoded 1/1"), and the encoder
        // reconstruction was verified bit-exact (Y/U/V max diff 0) vs the dav1d
        // yuv422p output. Pins the 4:2:2 RTX_8X16 chroma path.
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
            1,
        );
        assert_eq!(bytes.len(), 50, "4:2:2 stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4912, "4:2:2 stream bytes drifted");
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
            1,
        );
        assert_eq!(bytes.len(), 315, "4:2:2 8x8-leaves stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 40504, "4:2:2 8x8-leaves stream bytes drifted");
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
            1,
        );
        assert_eq!(p.len(), 141, "32x32 4:2:0 stream length drifted");
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            17360,
            "32x32 4:2:0 stream bytes drifted"
        );
    }

    #[test]
    fn lossy_64x64_422_tx32_chroma_stable() {
        // 64x64 4:2:2 at q32: 32x32 luma + 16-wide x 32-tall chroma per plane
        // (RTX_16X32, coef class 3). Verified bit-exact vs dav1d 1.4.1 (maxdiff
        // 0). Guards the new RTX_16X32 transform + coefficient coder.
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
            1,
        );
        assert_eq!(
            p.len(),
            162,
            "32x32 4:2:2 (RTX_16X32) stream length drifted"
        );
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            19503,
            "32x32 4:2:2 (RTX_16X32) stream bytes drifted"
        );
    }
}
