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

use crate::dct::{dct8x8, dct8x16_i32, dct16x16, dct32x32};
use crate::dct_trellis::forward_dct_quant_8x8_t;
use crate::obu::{
    frame_header_lossless, frame_header_lossy, frame_header_lossy_tiled, sequence_header_444_8bit,
    temporal_delimiter, wrap_obu_frame,
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

// AV1 spec default CDFs (from dav1d cdf.c), the four needed at context 0.
fn partition_64x64_ctx0() -> Vec<u16> {
    icdf(&[
        20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724,
    ])
}
fn skip_ctx0() -> Vec<u16> {
    icdf(&[31671])
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
                for m in 0..13 {
                    v.push(icdf(&UV_MODE_NOCFL_CDF[m]));
                }
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

fn kf_y_mode_dc_dc() -> Vec<u16> {
    icdf(&[
        15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466,
    ])
}
fn uv_mode_nocfl_dc() -> Vec<u16> {
    icdf(&[
        22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923, 28244, 30059, 30941, 31961,
    ])
}
/// CfL-allowed uv_mode DC CDF (`uv_mode[1][DC]`, 14 symbols). Used in the lossy
/// path where `cfl_allowed_mask` includes BS_8X8 (unlike lossless).
fn uv_mode_cfl_dc() -> Vec<u16> {
    icdf(&[
        10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656, 15986, 20086, 20995, 22455, 24212,
    ])
}

/// `default_kf_y_mode_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]` (libaom), the
/// keyframe luma intra-mode CDFs indexed by `[above_ctx][left_ctx]` (each ctx is
/// `INTRA_MODE_CTX[neighbour_mode]`). `[0][0]` equals the former single
/// `kf_y_mode_dc_dc()` CDF, so all-DC output is unchanged.
pub static KF_Y_MODE_CDF: [[[u16; 12]; 5]; 5] = [
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
pub static UV_MODE_NOCFL_CDF: [[u16; 12]; 13] = [
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
pub static ANGLE_DELTA_CDF: [[u16; 6]; 8] = [
    [2180, 5032, 7567, 22776, 26989, 30217],
    [2301, 5608, 8801, 23487, 26974, 30330],
    [3780, 11018, 13699, 19354, 23083, 31286],
    [4581, 11226, 15147, 17138, 21834, 28397],
    [1737, 10927, 14509, 19588, 22745, 28823],
    [2664, 10176, 12485, 17650, 21600, 30495],
    [2240, 11096, 15453, 20341, 22561, 28917],
    [3605, 10428, 12459, 17676, 21244, 30655],
];

const PARTITION_NONE: usize = 0;
const DC_PRED: usize = 0;

// --- Coefficient CDFs (qctx 0 = lossless), from dav1d cdf.c ---
// partition CDF for BL_8X8, ctx 0 (4 symbols: NONE/H/V/SPLIT). PARTITION_NONE=0.
fn partition_bl8_ctx0() -> Vec<u16> {
    icdf(&[19132, 25510, 30392])
}
// txb_skip (all_zero) by context: skip[TX_4X4][ctx], ctx in 0..13
fn txb_skip(ctx: usize) -> Vec<u16> {
    const T: [u16; 13] = [
        31849, 5892, 12112, 21935, 20289, 27473, 32487, 7654, 19473, 29984, 9961, 30242, 32117,
    ];
    icdf(&[T[ctx]])
}
fn eob_bin_luma() -> Vec<u16> {
    icdf(&[840, 1039, 1980, 4895])
}
fn eob_bin_chroma() -> Vec<u16> {
    icdf(&[3247, 4950, 9688, 14563])
}
fn base_eob_luma() -> Vec<u16> {
    icdf(&[17837, 29055])
}
fn base_eob_chroma() -> Vec<u16> {
    icdf(&[21365, 30026])
}
fn br_luma() -> Vec<u16> {
    icdf(&[14298, 20718, 24174])
}
fn br_chroma() -> Vec<u16> {
    icdf(&[15967, 22905, 26286])
}
fn dc_sign_luma() -> Vec<u16> {
    icdf(&[16000])
}
fn dc_sign_chroma() -> Vec<u16> {
    icdf(&[15232])
}
/// dc_sign CDF for a plane and neighbour-derived context (0..2).

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

/// Encode one TX_4X4 block that has a single nonzero DC coefficient `level`.
/// CDFs are passed per-plane. Bitstream order matches AV1 coeffs():
/// all_zero, eob_pt, coeff_base_eob, coeff_br*, dc_sign, golomb.
#[allow(clippy::too_many_arguments)]
fn encode_dc_tx(
    enc: &mut OdEcEncoder,
    level: i32,
    txb_skip: &[u16],
    eob_bin: &[u16],
    base_eob: &[u16],
    br: &[u16],
    dc_sign: &[u16],
) {
    if level == 0 {
        enc.encode_symbol_noupdate(1, txb_skip); // all_zero = 1
        return;
    }
    enc.encode_symbol_noupdate(0, txb_skip); // all_zero = 0
    enc.encode_symbol_noupdate(0, eob_bin); // eob_bin=0 -> eob=0 -> dc-only path
    let mag = level.unsigned_abs() as i32;
    let base = mag.min(NUM_BASE_LEVELS + 1); // 1..=3
    enc.encode_symbol_noupdate((base - 1) as usize, base_eob);
    if base == NUM_BASE_LEVELS + 1 {
        // coeff_br: code min(mag-3, 12) in up to 4 chunks of 3
        let total_br = (mag - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
        let mut coded = 0;
        for _ in 0..(COEFF_BASE_RANGE / 3) {
            let s = (total_br - coded).min(3);
            enc.encode_symbol_noupdate(s as usize, br);
            coded += s;
            if s < 3 {
                break;
            }
        }
    }
    enc.encode_symbol_noupdate((level < 0) as usize, dc_sign);
    if mag > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
        // golomb codes the excess above the base+br cap (15)
        encode_golomb(enc, (mag - (NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1)) as u32);
    }
}

/// Encode the 4 TX_4X4 blocks of one plane within the inferred 8x8 block.
/// Only TX0 (top-left, the part that maps to the 4x4 output) carries the DC
/// coefficient `level`; TX1..3 are all-zero padding. txb_skip contexts shift
/// for TX1/TX2 once TX0 is non-zero (neighbour cul_level >= 4 -> skip_ctx idx 4).
fn encode_plane_4tx(
    enc: &mut OdEcEncoder,
    level: i32,
    is_luma: bool,
    eob_bin: &[u16],
    base_eob: &[u16],
    br: &[u16],
    dc_sign: &[u16],
) {
    let nz = level != 0;
    // txb_skip contexts for [TX0, TX1, TX2, TX3]
    let ctxs: [usize; 4] = if is_luma {
        // luma: skip_ctx[above][left]; TX0=skip_ctx[0][0]=1.
        // when TX0 nonzero: TX1=skip_ctx[0][4]=3, TX2=skip_ctx[4][0]=3, TX3=skip_ctx[0][0]=1
        [1, if nz { 3 } else { 1 }, if nz { 3 } else { 1 }, 1]
    } else {
        // chroma (4:4:4, 8x8 block): base = 7 + not_one_blk(1)*3 = 10; +ca +cl
        [10, if nz { 11 } else { 10 }, if nz { 11 } else { 10 }, 10]
    };
    // TX0 carries the residual; TX1..3 are all-zero
    encode_dc_tx(
        enc,
        level,
        &txb_skip(ctxs[0]),
        eob_bin,
        base_eob,
        br,
        dc_sign,
    );
    for &c in &ctxs[1..] {
        encode_dc_tx(enc, 0, &txb_skip(c), eob_bin, base_eob, br, dc_sign);
    }
}

// ===========================================================================
//  LOSSY PATH — dav1d-validated (8x8 DC frame)
// ===========================================================================
//
// A lossy still: an 8x8 frame coded as one 8x8 block with one TX_8X8 (DCT_DCT)
// per plane carrying a single DC coefficient. Verified to decode to the exact
// target colour in stock dav1d 1.4.1 (see tests). Differences from lossless:
//
//   * frame_header_lossy(q): base_q != 0 -> CodedLossless false, so loop-filter
//     (levels 0), delta_q_present bit and tx_mode_select (= TX_LARGEST, TX size
//     inferred) are coded.
//   * uv_mode uses the 14-symbol CfL CDF (BS_8X8 is in cfl_allowed_mask, unlike
//     lossless which keys off cbw4==cbh4==1).
//   * luma codes a transform-type symbol (txtp_intra1[TX_8X8][DC], encode idx 1
//     -> DCT_DCT); chroma infers txtp from uv_mode (no symbol).
//   * coefficient CDFs use the TX_8X8 contexts (txb_skip sctx 0 luma / 7 chroma,
//     eob_bin_64, eob_base_tok[1], br_tok[1]); keep base_q_idx <= 20 for qctx 0.
//   * dequant cf[0] = dc_q[base_q_idx]*level (dq_shift 0), then the 8x8 inverse
//     DCT yields residual = (cf + 32) >> 6. level_for_residual() inverts this.
//
// Tile symbol order (per the dav1d recon trace): partition(BL_8X8)=NONE, skip=0,
// kf_y_mode=DC, uv_mode=DC(CfL), then per plane's TX_8X8: txb_skip, [luma txtp],
// eob_pt=0, base, br (hi_tok), dc_sign, golomb.

/// txb_skip (coeff all_zero) CDF for `TX_8X8`, qctx 0, by `sctx`.
fn txb_skip_tx8(ctx: usize) -> Vec<u16> {
    const T: [u16; 13] = [
        31548, 1549, 10130, 16656, 18591, 26308, 32537, 5403, 18096, 30003, 16384, 16384, 16384,
    ];
    icdf(&[T[ctx]])
}
/// Intra transform-type CDF, `txtp_intra1[TX_8X8][DC_PRED]` (7 symbols). Encode
/// idx 1 to select DCT_DCT (`tx_types_per_set[1 + 5] == DCT_DCT`).
fn txtp_intra1_tx8_dc() -> Vec<u16> {
    icdf(&[1870, 13742, 14530, 16498, 23770, 27698])
}
fn eob_bin_64_luma() -> Vec<u16> {
    icdf(&[329, 498, 1101, 1784, 3265, 7758])
}
fn eob_bin_64_chroma() -> Vec<u16> {
    icdf(&[3505, 5304, 10086, 13814, 17684, 23370])
}
fn base_eob_tx8_luma() -> Vec<u16> {
    icdf(&[5717, 26477])
}
fn base_eob_tx8_chroma() -> Vec<u16> {
    icdf(&[12608, 27820])
}
fn br_tx8_luma() -> Vec<u16> {
    icdf(&[14406, 20862, 24414])
}
fn br_tx8_chroma() -> Vec<u16> {
    icdf(&[15460, 21696, 25469])
}
// --- eob>0 (AC) path CDFs (TX_8X8, qctx 0) ---
/// `eob_base_tok[TX_8X8][luma][ctx=1]` — coeff_base_eob for the last (eob) coeff
/// when eob>0 (ctx = 1 + (eob>2*sw*sh) + (eob>4*sw*sh); 1 for small eob).
fn eob_base_tok_tx8_luma_c1() -> Vec<u16> {
    icdf(&[30491, 31703])
}
/// `base_tok[TX_8X8][luma][ctx=0]` — coeff_base for the DC in the eob>0 path
/// (the 2D DC always uses ctx 0). 4-symbol.
fn base_tok_tx8_luma_c0() -> Vec<u16> {
    icdf(&[4536, 10072, 14001])
}
/// `base_tok[TX_8X8][luma][ctx=1]` — coeff_base for an AC coeff whose neighbour
/// template sums to 0 at a position with `lo_ctx_offsets` offset 1.
fn base_tok_tx8_luma_c1() -> Vec<u16> {
    icdf(&[25459, 31416, 32206])
}
/// `eob_hi_bit[TX_8X8][luma][eob_bin=2]` — the high bit of the eob value when
/// `eob_bin > 1` (here eob_bin=2 -> eob = 2 | hi_bit).
fn eob_hi_bit_tx8_luma_b2() -> Vec<u16> {
    icdf(&[20401])
}
/// DC dequant value `dav1d_dq_tbl[bd][q][0]` for bit_depth 8/10/12.
pub fn dc_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::DC_QLOOKUP_10,
        12 => &crate::coef_q::DC_QLOOKUP_12,
        _ => &crate::coef_q::DC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}
/// AC dequant value `dav1d_dq_tbl[bd][q][1]` for bit_depth 8/10/12.
pub fn ac_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::AC_QLOOKUP_10,
        12 => &crate::coef_q::AC_QLOOKUP_12,
        _ => &crate::coef_q::AC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}
/// DC dequant value `dav1d_dq_tbl[8bpc][q][0]` (full range, any base_q_idx).
pub fn dc_q_8bit(base_q_idx: u8) -> u16 {
    dc_q(base_q_idx, 8)
}
/// AC dequant value `dav1d_dq_tbl[8bpc][q][1]` (full range, any base_q_idx).
pub fn ac_q_8bit(base_q_idx: u8) -> u16 {
    ac_q(base_q_idx, 8)
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
pub trait Dct {
    /// DC dequant step (`dav1d_dq_tbl[bd][q][0]`).
    fn dc_q(&self) -> i32;
    /// AC dequant step (`dav1d_dq_tbl[bd][q][1]`).
    fn ac_q(&self) -> i32;
    /// Inverse-transform clips `(row_min, row_max, col_min, col_max, cf_max)`.
    fn clips(&self) -> (i32, i32, i32, i32, i32);
    /// Dequant step for raster/scan position `rc` (DC at 0, AC otherwise).
    #[inline]
    fn step(&self, rc: usize) -> i32 {
        if rc == 0 { self.dc_q() } else { self.ac_q() }
    }
}

/// Precomputed dequant coefficients + inverse-transform clips for one
/// (base_q_idx, bit_depth). Cheap to copy; build once and hand to the transforms.
#[derive(Clone, Copy)]
pub struct Quant {
    dc: i32,
    ac: i32,
    rmin: i32,
    rmax: i32,
    cmin: i32,
    cmax: i32,
    cf_max: i32,
}

impl Quant {
    pub fn new(base_q_idx: u8, bd: u8) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        Quant {
            dc: dc_q(base_q_idx, bd) as i32,
            ac: ac_q(base_q_idx, bd) as i32,
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
}

/// Map a human-facing **quality** value in `0..=100` to an AV1 `base_q_idx`
/// (`1..=255`) for the lossy encoder. Higher quality means finer quantization
/// (a smaller `base_q_idx` and a larger file); lower quality means coarser.
///
/// The scale is **perceptually even** rather than linear in the index. Perceived
/// distortion tracks the *logarithm* of the quantizer step size, so equal
/// quality steps should change the step by a constant *ratio* (a geometric
/// progression), not a constant amount. This function therefore interpolates the
/// AC quant step geometrically between the finest usable index (`q=1`, AC step
/// 8) and the coarsest (`q=255`, AC step 1828), then returns the index whose
/// `ac_q` is closest to that target step. Because `ac_q` is monotonic, the
/// result is monotonically non-increasing in `quality`.
///
/// Endpoints: `quality=100` → `base_q_idx 1` (finest lossy; true lossless has
/// its own path), `quality=0` → `255` (coarsest). Inputs above 100 are clamped.
/// This is a perceptual calibration, not a bitrate target — it can later be
/// re-tuned against a reference codec (e.g. to align "quality 75" with JPEG q75)
/// without affecting decoder correctness, since `base_q_idx` is just signalled.
pub fn quality_to_base_q_idx(quality: u8) -> u8 {
    let q = quality.min(100) as f64;
    let (lo, hi) = (1u8, 255u8);
    let ac = |i: u8| ac_q_8bit(i) as f64;
    // Geometric target step: step(lo) at quality 100, step(hi) at quality 0.
    let target = ac(lo) * (ac(hi) / ac(lo)).powf((100.0 - q) / 100.0);
    // ac_q is monotonic non-decreasing; pick the closest index (ties favour the
    // higher-quality / lower index).
    let mut best = lo;
    let mut best_err = f64::INFINITY;
    for i in lo..=hi {
        let err = (ac(i) - target).abs();
        if err < best_err {
            best_err = err;
            best = i;
        }
    }
    best
}

/// Forward-DCT + quantize an 8x8 residual block into AV1 quantized coefficient
/// levels (raster order, for `encode_tx8_luma_coeffs`). The dav1d 8x8 inverse
/// DCT equals (1/8) x orthonormal DCT, so forward `cf = 8 * orthonormalDCT2(R)`,
/// quantized by dc_q (DC) / ac_q (AC), transposed (`rc = u*8 + v`) to dav1d's
/// coefficient layout. (Calibrated against dav1d: round-trip max error ~1 at q=16.)
pub fn forward_dct_quant_8x8(residual: &mut [i32; 64], q: &impl Dct) {
    dct8x8(residual, q)
}

/// Encode an 8x8 luma image (`pixels`, 0..=255) as a lossy AV1 still: forward
/// DCT + quantize the residual (pixel - 128, since DC_PRED with no neighbours
/// predicts 128), then the general coefficient encoder. Chroma flat (`r_u`,`r_v`).
/// The decoded luma approximates `pixels`, lossily per `base_q_idx`.
pub fn encode_av1_lossy_luma_image_8x8(
    base_q_idx: u8,
    pixels: &[u8; 64],
    r_u: i32,
    r_v: i32,
) -> Vec<u8> {
    let mut residual = [0i32; 64];
    for i in 0..64 {
        residual[i] = pixels[i] as i32 - 128;
    }
    forward_dct_quant_8x8(&mut residual, &Quant::new(base_q_idx, 8));
    encode_av1_lossy_luma_block_8x8(base_q_idx, &residual, r_u, r_v)
}

/// Encode one `TX_8X8` block with a single DC coefficient `level` (lossy path).
/// Luma codes the intra transform-type symbol; chroma infers it from uv_mode,
/// so `code_txtp` must be true only for the luma plane. Order matches the dav1d
/// recon trace: all_zero, [txtp], eob_pt=0, base, br, dc_sign, golomb.
#[allow(clippy::too_many_arguments)]
fn encode_tx8_dc(
    enc: &mut OdEcEncoder,
    level: i32,
    code_txtp: bool,
    txb_skip: &[u16],
    eob_bin: &[u16],
    base_eob: &[u16],
    br: &[u16],
    dc_sign: &[u16],
) {
    if level == 0 {
        enc.encode_symbol_noupdate(1, txb_skip); // all_zero = 1
        return;
    }
    enc.encode_symbol_noupdate(0, txb_skip); // all_zero = 0
    if code_txtp {
        enc.encode_symbol_noupdate(1, &txtp_intra1_tx8_dc()); // idx 1 -> DCT_DCT
    }
    enc.encode_symbol_noupdate(0, eob_bin); // eob_pt = 0 -> dc-only path
    let mag = level.unsigned_abs() as i32;
    let base = mag.min(NUM_BASE_LEVELS + 1);
    enc.encode_symbol_noupdate((base - 1) as usize, base_eob);
    if base == NUM_BASE_LEVELS + 1 {
        let total_br = (mag - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
        let mut coded = 0;
        for _ in 0..(COEFF_BASE_RANGE / 3) {
            let s = (total_br - coded).min(3);
            enc.encode_symbol_noupdate(s as usize, br);
            coded += s;
            if s < 3 {
                break;
            }
        }
    }
    enc.encode_symbol_noupdate((level < 0) as usize, dc_sign);
    if mag > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
        encode_golomb(enc, (mag - (NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1)) as u32);
    }
}

/// Pick the DC coefficient `level` that makes dav1d's 8x8 inverse DCT decode to
/// (approximately) the target residual `r`. The 8x8 inverse DCT of a DC-only
/// coefficient yields `residual = (dc_q*level + 32) >> 6`, so the level that
/// best hits `r` is `round(64*|r| / dc_q)`, with the sign carried separately.
fn level_for_residual(r: i32, dc_q: u16) -> i32 {
    if r == 0 {
        return 0;
    }
    let mag = ((64 * r.unsigned_abs() + dc_q as u32 / 2) / dc_q as u32) as i32;
    if r < 0 { -mag } else { mag }
}

/// A tiny **lossy** AV1 keyframe: an 8x8 frame coded as one 8x8 block with one
/// `TX_8X8` (DCT_DCT) per plane. `r_y/r_u/r_v` are the *target* residuals; the
/// decoded flat colour is approximately `(128+r_y, 128+r_u, 128+r_v)` (lossy:
/// the value is quantized by `base_q_idx`). Keep `base_q_idx` <= 20 for qctx 0.
/// dav1d dequantizes `cf[0] = dc_q[base_q_idx] * level` then inverse-DCTs:
/// `residual = (cf + 32) >> 6`.
pub fn encode_av1_lossy_dc_8x8(base_q_idx: u8, r_y: i32, r_u: i32, r_v: i32) -> Vec<u8> {
    let dc_q = dc_q_8bit(base_q_idx);
    let (level_y, level_u, level_v) = (
        level_for_residual(r_y, dc_q),
        level_for_residual(r_u, dc_q),
        level_for_residual(r_v, dc_q),
    );
    let mut enc = OdEcEncoder::new();
    enc.encode_symbol_noupdate(PARTITION_NONE, &partition_bl8_ctx0());
    enc.encode_symbol_noupdate(0, &skip_ctx0()); // skip = 0
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc());
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_cfl_dc()); // lossy 8x8: CfL allowed -> 14-sym CDF
    encode_tx8_dc(
        &mut enc,
        level_y,
        true,
        &txb_skip_tx8(0),
        &eob_bin_64_luma(),
        &base_eob_tx8_luma(),
        &br_tx8_luma(),
        &dc_sign_luma(),
    );
    encode_tx8_dc(
        &mut enc,
        level_u,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    encode_tx8_dc(
        &mut enc,
        level_v,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    let tile = enc.done();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(8, 8));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossy(base_q_idx), &tile));
    bytes
}

/// Encode a luma `TX_8X8` block with a DC coefficient plus ONE AC coefficient
/// at scan position 1 (eob=1) — the minimal step into the `eob>0` path. Both
/// levels must be in 1..=2 (DC) / 1..=2 (AC) so no hi_tok/golomb is needed and
/// the AC reverse-scan loop (which would need neighbor-context modelling) stays
/// empty. dav1d places the AC at scan[1], producing a gradient, not a flat tile.
/// Symbol order (dav1d recon): all_zero, txtp, eob_pt=1, coeff_base_eob(ctx 1),
/// coeff_base DC(ctx 0), then dequant-loop signs: dc_sign (CDF), ac_sign (equi).
fn encode_tx8_luma_dc_ac(enc: &mut OdEcEncoder, dc_level: i32, ac_level: i32) {
    enc.encode_symbol_noupdate(0, &txb_skip_tx8(0)); // all_zero = 0
    enc.encode_symbol_noupdate(1, &txtp_intra1_tx8_dc()); // txtp idx 1 -> DCT_DCT
    enc.encode_symbol_noupdate(1, &eob_bin_64_luma()); // eob_pt = 1 -> eob = 1
    let a = ac_level.unsigned_abs() as usize; // 1..=2
    enc.encode_symbol_noupdate(a - 1, &eob_base_tok_tx8_luma_c1()); // eob coeff base
    let d = dc_level.unsigned_abs() as usize; // 0..=2
    enc.encode_symbol_noupdate(d, &base_tok_tx8_luma_c0()); // DC coeff base
    // dequant-loop signs: DC (CDF) first, then AC (equiprobable)
    if d != 0 {
        enc.encode_symbol_noupdate((dc_level < 0) as usize, &dc_sign_luma());
    }
    enc.encode_bool(ac_level < 0, 16384); // AC sign (equiprobable)
}

/// A lossy 8x8 still whose **luma** block carries a DC plus one AC coefficient
/// (a gradient), with chroma flat (`r_u`, `r_v` target residuals). Proves the
/// `eob>0` coefficient path end-to-end against dav1d. `dc_level`/`ac_level` are
/// raw quantized coefficient levels (1..=2) at base_q_idx 16.
pub fn encode_av1_lossy_luma_ac_8x8(
    base_q_idx: u8,
    dc_level: i32,
    ac_level: i32,
    r_u: i32,
    r_v: i32,
) -> Vec<u8> {
    let dc_q = dc_q_8bit(base_q_idx);
    let (lu, lv) = (level_for_residual(r_u, dc_q), level_for_residual(r_v, dc_q));
    let mut enc = OdEcEncoder::new();
    enc.encode_symbol_noupdate(PARTITION_NONE, &partition_bl8_ctx0());
    enc.encode_symbol_noupdate(0, &skip_ctx0());
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc());
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_cfl_dc());
    encode_tx8_luma_dc_ac(&mut enc, dc_level, ac_level);
    encode_tx8_dc(
        &mut enc,
        lu,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    encode_tx8_dc(
        &mut enc,
        lv,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    let tile = enc.done();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(8, 8));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossy(base_q_idx), &tile));
    bytes
}

/// Encode a luma `TX_8X8` block with THREE coefficients (eob=2): DC at scan[0],
/// an AC at scan[1] (=position 8), and the eob coeff at scan[2] (=position 1).
/// This is the first case that runs the reverse-scan loop and therefore the
/// neighbour-context model `get_lo_ctx`. For these scan positions the scan[1]
/// coeff's 5-neighbour template (positions 9,16,17,10,24) excludes the only
/// other decoded coeff (scan[2]=1), so its magnitude sum is 0 and its context
/// is the position offset `lo_ctx_offsets[0][0][1] = 1` -> base_tok ctx 1.
/// Levels are kept in 1..=2 so no hi_tok/golomb is needed.
fn encode_tx8_luma_eob2(enc: &mut OdEcEncoder, dc: i32, ac1: i32, ac2: i32) {
    enc.encode_symbol_noupdate(0, &txb_skip_tx8(0)); // all_zero = 0
    enc.encode_symbol_noupdate(1, &txtp_intra1_tx8_dc()); // txtp idx 1 -> DCT_DCT
    enc.encode_symbol_noupdate(2, &eob_bin_64_luma()); // eob_pt = 2
    enc.encode_symbol_noupdate(0, &eob_hi_bit_tx8_luma_b2()); // eob_hi_bit = 0 -> eob = 2
    // eob coeff at scan[2] (ctx 1), then AC at scan[1] (ctx 1), then DC (ctx 0)
    enc.encode_symbol_noupdate(ac2.unsigned_abs() as usize - 1, &eob_base_tok_tx8_luma_c1());
    enc.encode_symbol_noupdate(ac1.unsigned_abs() as usize, &base_tok_tx8_luma_c1());
    enc.encode_symbol_noupdate(dc.unsigned_abs() as usize, &base_tok_tx8_luma_c0());
    // dequant-loop signs: DC (CDF), then chain order scan[1], scan[2] (equi)
    enc.encode_symbol_noupdate((dc < 0) as usize, &dc_sign_luma());
    enc.encode_bool(ac1 < 0, 16384);
    enc.encode_bool(ac2 < 0, 16384);
}

/// A lossy 8x8 still whose luma block carries THREE coefficients (DC + 2 AC,
/// eob=2) — the first multi-AC block, exercising the neighbour-context model.
/// Chroma is flat (`r_u`, `r_v`). `dc`/`ac1`/`ac2` are quantized levels (1..=2).
pub fn encode_av1_lossy_luma_eob2_8x8(
    base_q_idx: u8,
    dc: i32,
    ac1: i32,
    ac2: i32,
    r_u: i32,
    r_v: i32,
) -> Vec<u8> {
    let dc_q = dc_q_8bit(base_q_idx);
    let (lu, lv) = (level_for_residual(r_u, dc_q), level_for_residual(r_v, dc_q));
    let mut enc = OdEcEncoder::new();
    enc.encode_symbol_noupdate(PARTITION_NONE, &partition_bl8_ctx0());
    enc.encode_symbol_noupdate(0, &skip_ctx0());
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc());
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_cfl_dc());
    encode_tx8_luma_eob2(&mut enc, dc, ac1, ac2);
    encode_tx8_dc(
        &mut enc,
        lu,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    encode_tx8_dc(
        &mut enc,
        lv,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    let tile = enc.done();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(8, 8));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossy(base_q_idx), &tile));
    bytes
}

// --- General eob>0 luma coefficient coding (TX_8X8, 2D, qctx 0) -------------
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

/// Replicate dav1d `get_lo_ctx` for TX_CLASS_2D: returns (base ctx, hi_mag).
/// `levels` is the padded magnitude map with the given `stride` (8 for TX_8X8 /
/// RTX_4X8, 4 for TX_4X4); (x,y) is the coefficient position. `off` is the
/// position-offset table (`LO_CTX_OFF` square, or `LO_CTX_OFF_WLH` for w<h).
/// Cheap coded-size proxy for one transform block used by the partition
/// decision: the sum of |levels| (entropy roughly tracks coefficient magnitude)
/// plus `EOB_BITW * (eob_index + 1)`, where the EOB index is the scan position of
/// the last nonzero coefficient. The EOB term matters because every coefficient
/// up to the last nonzero must be signalled, so a block whose energy is spread
/// far out in the scan (e.g. a 16x16 DCT of a smooth gradient, or a lone stray
/// coefficient at coarse quant) is dearer than a low-EOB alternative of equal
/// total magnitude. All-zero blocks cost ~1 (just the skip flag). This keeps the
/// 16x16-vs-four-8x8 choice from regressing at the quality extremes.
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
/// quantizer to compare candidate levels — it need not be exact, since only the
/// *relative* costs drive the decision.
pub fn coef_rate_bits(level: u32) -> f64 {
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
    for i in 0..=(eob_idx as usize) {
        let rc = scan[i];
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

/// Bits to code symbol `s` against an (inverse-form) CDF: `-log2(p)` where the
/// probability is `(cdf[s-1] - cdf[s]) / 32768` (with `cdf[-1] = 32768`). This
/// matches the MSAC's symbol partition (ignoring the negligible `EC_MIN_PROB`
/// term), so it is the same rate libaom's cost tables approximate.
#[inline]
fn cdf_cost(cdf: &[u16], s: usize) -> f64 {
    let fl = if s > 0 { cdf[s - 1] as i32 } else { 32768 };
    let fh = cdf[s] as i32;
    let p = (fl - fh).max(1) as f64;
    -(p * (1.0 / 32768.0)).log2()
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
#[allow(clippy::too_many_arguments)]
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
    let dqf = |rc: usize| if rc == 0 { dc_q } else { ac_q };
    let dist = |rc: usize, lev: i32| {
        let dq = dqf(rc);
        (dq * dq) * (tf[rc].abs() - (lev.abs() as f64)).powi(2)
    };

    let mut eob: i32 = -1;
    for i in 0..n {
        if cf[scan[i]] != 0 {
            eob = i as i32;
        }
    }
    if eob < 0 {
        return;
    }

    let mut levels = vec![0u8; w * (w + 4)];
    let set_level = |levels: &mut [u8], rc: usize, m: u32| {
        levels[(rc >> log2w) * stride + (rc & (w - 1))] = level_byte(m);
    };
    for i in 0..=(eob as usize) {
        let rc = scan[i];
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
            return cdf_cost(&cdfs.base_tok[cls][plane][ctx], 0);
        }
        let tok = k.min(3);
        let mut b = cdf_cost(&cdfs.base_tok[cls][plane][ctx], tok as usize);
        if tok == 3 {
            b += hi_tok_cost(k, &cdfs.br_tok[cls][plane][bc]);
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
        let mut best_k = l;
        let mut best_c = dist(rc, l as i32) + lambda * interior_rate(ctx, bc, l);
        for k in (0..l).rev() {
            let c = dist(rc, k as i32) + lambda * interior_rate(ctx, bc, k);
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
                    return cdf_cost(&cdfs.base_tok[cls][plane][0], 0);
                }
                let tok = k.min(3);
                let mut b = cdf_cost(&cdfs.base_tok[cls][plane][0], tok as usize);
                if tok == 3 {
                    b += hi_tok_cost(k, &cdfs.br_tok[cls][plane][bc]);
                }
                b + cdf_cost(&cdfs.dc_sign[plane][dcs_ctx], sgn)
            };
            let mut best_k = l;
            let mut best_c = dist(rc, l as i32) + lambda * dc_rate(l);
            for k in (0..l).rev() {
                let c = dist(rc, k as i32) + lambda * dc_rate(k);
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
            c += cdf_cost(&cdfs.eob_hi[cls][plane][bin], (e >> nbits) & 1);
            c += nbits as f64; // remaining eob offset bits (bypass)
        }
        c
    };
    let eob_coeff_cost = |e: usize, m: u32| -> f64 {
        let ctx_e = 1 + (e > n / 8) as usize + (e > n / 4) as usize;
        let tok = m.min(3);
        let mut c = cdf_cost(&cdfs.eob_base[cls][plane][ctx_e], tok as usize - 1);
        if tok == 3 {
            let rc = scan[e];
            let (ex, ey) = (rc >> log2w, rc & (w - 1));
            let bc = if (ex | ey) > 1 { 14 } else { 7 };
            c += hi_tok_cost(m, &cdfs.br_tok[cls][plane][bc]);
        }
        c + 1.0 // sign
    };

    // Interior (base_tok) rate of each position at its current level, for the
    // running prefix; positions are priced as interior even if they will end up
    // being the EOB (corrected by swapping in eob_coeff_cost at the candidate).
    let mut pre = vec![0f64; n + 1]; // sum over positions [1, i) of (rate + dist)
    for i in 1..n {
        let rc = scan[i];
        let (rate, d) = if i <= eob as usize {
            let (ctx, bc) = interior_ctx(&levels, rc);
            (
                interior_rate(ctx, bc, cf[rc].unsigned_abs()),
                dist(rc, cf[rc]),
            )
        } else {
            (0.0, dist(rc, 0))
        };
        pre[i + 1] = pre[i] + lambda * rate + d;
    }
    let mut suf0 = vec![0f64; n + 1]; // distortion of zeroing positions [i, n)
    for i in (1..n).rev() {
        suf0[i] = suf0[i + 1] + dist(scan[i], 0);
    }
    // DC contribution (rate + distortion), constant across EOB choices ≥ 1.
    let dc_rc = scan[0];
    let dc_m = cf[dc_rc].unsigned_abs();
    let dc_cost = if dc_m == 0 {
        lambda * cdf_cost(&cdfs.base_tok[cls][plane][0], 0)
    } else {
        let bc = dc_brc(&levels);
        let tok = dc_m.min(3);
        let mut b = cdf_cost(&cdfs.base_tok[cls][plane][0], tok as usize);
        if tok == 3 {
            b += hi_tok_cost(dc_m, &cdfs.br_tok[cls][plane][bc]);
        }
        b += cdf_cost(&cdfs.dc_sign[plane][dcs_ctx], (cf[dc_rc] < 0) as usize);
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
        let (ctx, bc) = interior_ctx(&levels, rc);
        let interior_e = lambda * interior_rate(ctx, bc, cf[rc].unsigned_abs());
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
        let mut c0 = cdf_cost(eob_bin_cdf, 0)
            + cdf_cost(&cdfs.eob_base[cls][plane][ctx_e], tok as usize - 1);
        if tok == 3 {
            c0 += hi_tok_cost(dc_m, &cdfs.br_tok[cls][plane][dc_brc(&levels)]);
        }
        c0 += cdf_cost(&cdfs.dc_sign[plane][dcs_ctx], (cf[dc_rc] < 0) as usize);
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
        return;
    }
    for i in (best_e as usize + 1)..n {
        cf[scan[i]] = 0;
    }
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
pub static INTRA_MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

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
    for i in 0..n {
        ac[i] = luma_rec[i] << 3;
    }
    let log2sz = w.trailing_zeros() + h.trailing_zeros();
    let mut sum: i64 = (1i64 << log2sz) >> 1;
    for i in 0..n {
        sum += ac[i] as i64;
    }
    let mean = (sum >> log2sz) as i32;
    for i in 0..n {
        ac[i] -= mean;
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

/// General luma `TX_8X8` coefficient encoder for arbitrary quantized levels
/// `cf` (raster order, `cf[row*8+col]`). Replicates dav1d's eob>0 coefficient
/// path: eob_pt/hi_bit/extra, reverse-scan coeff_base with `get_lo_ctx`, br
/// (hi_tok), the DC, then dequant-loop signs (DC adaptive, AC equiprobable in
/// chain order) and golomb tails. eob==0 (DC only) delegates to encode_tx8_dc.
fn encode_tx8_luma_coeffs(enc: &mut OdEcEncoder, cf: &[i32; 64]) {
    encode_tx8_coeffs(enc, cf, false);
}

/// Isolated-block wrapper (static CDFs): luma skip-ctx 0, chroma skip-ctx 7.
fn encode_tx8_coeffs(enc: &mut OdEcEncoder, cf: &[i32; 64], chroma: bool) -> u8 {
    encode_tx8_coeffs_ctx(enc, cf, chroma, if chroma { 7 } else { 0 }, 0)
}

/// Static (non-adapting) `TX_8X8` coefficient encoder, used by the isolated
/// single-block demo APIs (which set `disable_cdf_update = 1`). The full-image
/// path uses [`encode_tx8_coeffs_adapt`] instead.
pub fn encode_tx8_coeffs_ctx(
    enc: &mut OdEcEncoder,
    cf: &[i32; 64],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    use crate::coef_q as Q;
    let qctx = 0usize; // isolated APIs always use base_q -> qcat 0 inputs
    let bt = |c: usize| {
        icdf(if chroma {
            &Q::BASE_TOK_TX8_CHROMA_Q[qctx][c]
        } else {
            &Q::BASE_TOK_TX8_LUMA_Q[qctx][c]
        })
    };
    let br = |c: usize| {
        icdf(if chroma {
            &Q::BR_TOK_TX8_CHROMA_Q[qctx][c]
        } else {
            &Q::BR_TOK_TX8_LUMA_Q[qctx][c]
        })
    };
    let eob_base = |c: usize| {
        icdf(if chroma {
            &Q::EOB_BASE_TX8_CHROMA_Q[qctx][c]
        } else {
            &Q::EOB_BASE_TX8_LUMA_Q[qctx][c]
        })
    };
    let eob_hi = |b: usize| {
        icdf(&[if chroma {
            Q::EOB_HI_TX8_CHROMA[qctx][b]
        } else {
            Q::EOB_HI_TX8_LUMA[qctx][b]
        }])
    };
    let eob_bin_cdf = icdf(if chroma {
        &Q::EOB_BIN_64_CHROMA[qctx]
    } else {
        &Q::EOB_BIN_64_LUMA[qctx]
    });
    let dc_sign = icdf(&[Q::DC_SIGN_Q[qctx][chroma as usize][dcs_ctx]]);
    let mut eob = 0usize;
    for (i, &rc) in SCAN_8X8.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i;
        }
    }
    if cf.iter().all(|&c| c == 0) {
        enc.encode_symbol_noupdate(1, &icdf(&[Q::SKIP_TX8[qctx][skip_ctx]]));
        return 0x40;
    }
    enc.encode_symbol_noupdate(0, &icdf(&[Q::SKIP_TX8[qctx][skip_ctx]]));
    if !chroma {
        enc.encode_symbol_noupdate(1, &txtp_intra1_tx8_dc());
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
        enc.encode_symbol_noupdate(0, &eob_bin_cdf);
        let m = cf[0].unsigned_abs();
        let base = m.min(3);
        enc.encode_symbol_noupdate(base as usize - 1, &eob_base(0));
        if base == 3 {
            encode_hi_tok_static(enc, m, &br(0));
        }
        enc.encode_symbol_noupdate((cf[0] < 0) as usize, &dc_sign);
        if m >= 15 {
            encode_golomb(enc, m - 15);
        }
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol_noupdate(eob_bin, &eob_bin_cdf);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        enc.encode_symbol_noupdate((eob >> nbits) & 1, &eob_hi(eob_bin));
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_8X8[eob];
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    enc.encode_symbol_noupdate(m.min(3) as usize - 1, &eob_base(ctx_e));
    if m.min(3) == 3 {
        encode_hi_tok_static(enc, m, &br(if (ex | ey) > 1 { 14 } else { 7 }));
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X8[i];
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol_noupdate(tok as usize, &bt(ctx));
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok_static(enc, m, &br(bc as usize));
        }
        levels[x * 8 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol_noupdate(dc_tok as usize, &bt(0));
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        encode_hi_tok_static(
            enc,
            dm,
            &br(if mag > 12 {
                6
            } else {
                ((mag + 1) >> 1) as usize
            }),
        );
    }
    if cf[0] != 0 {
        enc.encode_symbol_noupdate((cf[0] < 0) as usize, &dc_sign);
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

/// Static hi_tok ladder (non-adapting), for the isolated single-block coder.
fn encode_hi_tok_static(enc: &mut OdEcEncoder, m: u32, br_cdf: &[u16]) {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0;
    for _ in 0..(COEFF_BASE_RANGE / 3) {
        let s = (total_br - coded).min(3);
        enc.encode_symbol_noupdate(s as usize, br_cdf);
        coded += s;
        if s < 3 {
            break;
        }
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

/// A lossy 8x8 still whose luma block carries an arbitrary quantized coefficient
/// block `cf` (raster order), with chroma flat. Exercises the full general
/// eob>0 coefficient path (arbitrary positions/levels, computed contexts).
pub fn encode_av1_lossy_luma_block_8x8(
    base_q_idx: u8,
    cf: &[i32; 64],
    r_u: i32,
    r_v: i32,
) -> Vec<u8> {
    let dc_q = dc_q_8bit(base_q_idx);
    let (lu, lv) = (level_for_residual(r_u, dc_q), level_for_residual(r_v, dc_q));
    let mut enc = OdEcEncoder::new();
    enc.encode_symbol_noupdate(PARTITION_NONE, &partition_bl8_ctx0());
    enc.encode_symbol_noupdate(0, &skip_ctx0());
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc());
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_cfl_dc());
    encode_tx8_luma_coeffs(&mut enc, cf);
    encode_tx8_dc(
        &mut enc,
        lu,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    encode_tx8_dc(
        &mut enc,
        lv,
        false,
        &txb_skip_tx8(7),
        &eob_bin_64_chroma(),
        &base_eob_tx8_chroma(),
        &br_tx8_chroma(),
        &dc_sign_chroma(),
    );
    let tile = enc.done();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(8, 8));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossy(base_q_idx), &tile));
    bytes
}

/// Encode a full-colour 8x8 AV1 still: luma, U and V each forward-DCT'd,
/// quantized, and run through the general coefficient encoder. The decoded
/// frame approximates the three input planes (lossy per `base_q_idx`). Chroma
/// uses the same TX_8X8 machinery as luma but its own CDFs and infers txtp.
pub fn encode_av1_lossy_color_image_8x8(
    base_q_idx: u8,
    y_px: &[u8; 64],
    u_px: &[u8; 64],
    v_px: &[u8; 64],
) -> Vec<u8> {
    let resid = |p: &[u8; 64]| {
        let mut r = [0i32; 64];
        for i in 0..64 {
            r[i] = p[i] as i32 - 128;
        }
        r
    };
    let mut cf_y = resid(y_px);
    let mut cf_u = resid(u_px);
    let mut cf_v = resid(v_px);
    forward_dct_quant_8x8(&mut cf_y, &Quant::new(base_q_idx, 8));
    forward_dct_quant_8x8(&mut cf_u, &Quant::new(base_q_idx, 8));
    forward_dct_quant_8x8(&mut cf_v, &Quant::new(base_q_idx, 8));
    let mut enc = OdEcEncoder::new();
    enc.encode_symbol_noupdate(PARTITION_NONE, &partition_bl8_ctx0());
    enc.encode_symbol_noupdate(0, &skip_ctx0());
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc());
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_cfl_dc());
    encode_tx8_coeffs(&mut enc, &cf_y, false);
    encode_tx8_coeffs(&mut enc, &cf_u, true);
    encode_tx8_coeffs(&mut enc, &cf_v, true);
    let tile = enc.done();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(8, 8));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossy(base_q_idx), &tile));
    bytes
}

/// A complete tiny AV1 keyframe carrying a real flat colour. The frame is 4x4,
/// but dav1d decodes it as one inferred 8x8 block with 12 TX_4X4 blocks (4 per
/// plane). The top-left TX of each plane carries DC = 4*r (iwht divides by 4), so the decoded 4x4
/// output is the constant colour (128+r_y, 128+r_u, 128+r_v). Lossless / WHT.
pub fn encode_av1_dc_keyframe_4x4(r_y: i32, r_u: i32, r_v: i32) -> Vec<u8> {
    let mut enc = OdEcEncoder::new();
    // At BL_8X8 the partition is coded (PARTITION_NONE keeps the 8x8 block).
    enc.encode_symbol_noupdate(0, &partition_bl8_ctx0()); // PARTITION_NONE
    // block: skip = 0 (has residual), DC luma + chroma modes
    enc.encode_symbol_noupdate(0, &skip_ctx0()); // skip = 0
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc());
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_nocfl_dc()); // 8x8 block: cbw4==2 -> CfL not allowed
    // coeffs: luma (4 TX), then U (4 TX), then V (4 TX)
    encode_plane_4tx(
        &mut enc,
        4 * r_y,
        true,
        &eob_bin_luma(),
        &base_eob_luma(),
        &br_luma(),
        &dc_sign_luma(),
    );
    encode_plane_4tx(
        &mut enc,
        4 * r_u,
        false,
        &eob_bin_chroma(),
        &base_eob_chroma(),
        &br_chroma(),
        &dc_sign_chroma(),
    );
    encode_plane_4tx(
        &mut enc,
        4 * r_v,
        false,
        &eob_bin_chroma(),
        &base_eob_chroma(),
        &br_chroma(),
        &dc_sign_chroma(),
    );
    let tile = enc.done();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(4, 4));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossless(), &tile));
    bytes
}

/// Encode the single-block skip tile and return its `od_ec` byte payload.
fn encode_skip_tile() -> Vec<u8> {
    let mut enc = OdEcEncoder::new();
    // decode_partition(BLOCK_64X64): PARTITION_NONE
    enc.encode_symbol_noupdate(PARTITION_NONE, &partition_64x64_ctx0());
    // decode_block -> intra_frame_mode_info:
    enc.encode_symbol_noupdate(1, &skip_ctx0()); // skip = 1
    enc.encode_symbol_noupdate(DC_PRED, &kf_y_mode_dc_dc()); // luma intra mode
    enc.encode_symbol_noupdate(DC_PRED, &uv_mode_nocfl_dc()); // chroma intra mode
    enc.done()
}

/// Build a complete, dav1d-targeted AV1 still image (64x64, mid-grey result).
pub fn encode_av1_skip_keyframe_64x64() -> Vec<u8> {
    let tile = encode_skip_tile();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&sequence_header_444_8bit(64, 64));
    bytes.extend_from_slice(&wrap_obu_frame(&frame_header_lossless(), &tile));
    bytes
}

// ---------------------------------------------------------------------------
// Lossy 64x64: a full superblock split uniformly into 8x8 blocks, each coded
// DC_PRED + TX_8X8 (DCT_DCT), quantized. Reuses the validated TX_8X8 coefficient
// coder and forward DCT; adds the partition tree, per-block mode info, a
// reconstruction loop (so DC prediction matches the decoder), and the coef
// neighbour-context bookkeeping that ties the 64 blocks together.
// ---------------------------------------------------------------------------

// partition[bl][ctx] CDFs from dav1d cdf.c. Split levels emit PARTITION_SPLIT
// (symbol 3) with a 10-symbol CDF; BL_8X8 emits PARTITION_NONE (symbol 0) with a
// 4-symbol CDF. Index 0=BL_64X64, 1=BL_32X32, 2=BL_16X16.
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

/// Inverse of `forward_dct_quant_8x8`: dequantize `levels` and apply the inverse
/// DCT, returning the reconstructed residual (raster). Float approximation of
/// dav1d's integer transform — close enough that DC-prediction drift stays tiny.
/// dav1d's exact integer 1-D inverse DCT4 (`inv_dct4_1d_internal_c`, tx64=0),
/// operating on `c[0], c[s], c[2s], c[3s]` with clip to `[min,max]`.
fn inv_dct4_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    let (in0, in1, in2, in3) = (c[0], c[s], c[2 * s], c[3 * s]);
    let t0 = ((in0 + in2) * 181 + 128) >> 8;
    let t1 = ((in0 - in2) * 181 + 128) >> 8;
    let t2 = ((in1 * 1567 - in3 * (3784 - 4096) + 2048) >> 12) - in3;
    let t3 = ((in1 * (3784 - 4096) + in3 * 1567 + 2048) >> 12) + in1;
    c[0] = clip(t0 + t3);
    c[s] = clip(t1 + t2);
    c[2 * s] = clip(t1 - t2);
    c[3 * s] = clip(t0 - t3);
}

/// dav1d's exact integer 1-D inverse DCT8 (`inv_dct8_1d_internal_c`, tx64=0).
fn inv_dct8_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    inv_dct4_1d(c, 2 * s, min, max); // even positions c[0],c[2s],c[4s],c[6s]
    let (in1, in3, in5, in7) = (c[s], c[3 * s], c[5 * s], c[7 * s]);
    let t4a = ((in1 * 799 - in7 * (4017 - 4096) + 2048) >> 12) - in7;
    let mut t5a = (in5 * 1703 - in3 * 1138 + 1024) >> 11;
    let mut t6a = (in5 * 1138 + in3 * 1703 + 1024) >> 11;
    let t7a = ((in1 * (4017 - 4096) + in7 * 799 + 2048) >> 12) + in1;
    let t4 = clip(t4a + t5a);
    t5a = clip(t4a - t5a);
    let t7 = clip(t7a + t6a);
    t6a = clip(t7a - t6a);
    let t5 = ((t6a - t5a) * 181 + 128) >> 8;
    let t6 = ((t6a + t5a) * 181 + 128) >> 8;
    let (t0, t1, t2, t3) = (c[0], c[2 * s], c[4 * s], c[6 * s]);
    c[0] = clip(t0 + t7);
    c[s] = clip(t1 + t6);
    c[2 * s] = clip(t2 + t5);
    c[3 * s] = clip(t3 + t4);
    c[4 * s] = clip(t3 - t4);
    c[5 * s] = clip(t2 - t5);
    c[6 * s] = clip(t1 - t6);
    c[7 * s] = clip(t0 - t7);
}

/// Reconstruct an 8x8 residual from quantized levels using dav1d's EXACT integer
/// inverse transform (TX_8X8 DCT_DCT, 8-bit, shift=1), so the encoder's
/// reconstruction is bit-identical to the decoder's. This eliminates DC-pred
/// drift across blocks (the float inverse accumulated error on smooth content).
fn idct_dequant_8x8(levels: &[i32; 64], q: &impl Dct) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    // dequant: coeff[rc] = clamp(|level|*q, cf_max) with sign (dq_shift=0 for TX_8X8)
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // tmp[y*8+x] = coeff[y + x*8]; row inv_dct8, >>1 (rnd 1) clip, col inv_dct8, >>4
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for y in 0..8 {
        inv_dct8_1d(&mut tmp[y * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for x in 0..8 {
        inv_dct8_1d(&mut tmp[x..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d-exact integer inverse 16-point DCT (`dav1d_inv_dct16_1d_c`, tx64=0
/// branch of `inv_dct16_1d_internal_c` in src/itx_1d.c). Operates in place on
/// `c[0], c[s], .., c[15*s]`. Even positions are handled by `inv_dct8_1d`; the
/// odd-position stages use the AV1 rotation constants verbatim. Stage-named
/// locals avoid the in-place variable reuse of the C source.
fn inv_dct16_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    inv_dct8_1d(c, 2 * s, min, max); // even positions c[0],c[2s],..,c[14s]
    let (in1, in3, in5, in7) = (c[s], c[3 * s], c[5 * s], c[7 * s]);
    let (in9, in11, in13, in15) = (c[9 * s], c[11 * s], c[13 * s], c[15 * s]);
    // stage 1 (odd inputs -> t8a..t15a)
    let t8a = ((in1 * 401 - in15 * (4076 - 4096) + 2048) >> 12) - in15;
    let t9a = (in9 * 1583 - in7 * 1299 + 1024) >> 11;
    let t10a = ((in5 * 1931 - in11 * (3612 - 4096) + 2048) >> 12) - in11;
    let t11a = ((in13 * (3920 - 4096) - in3 * 1189 + 2048) >> 12) + in13;
    let t12a = ((in13 * 1189 + in3 * (3920 - 4096) + 2048) >> 12) + in3;
    let t13a = ((in5 * (3612 - 4096) + in11 * 1931 + 2048) >> 12) + in5;
    let t14a = (in9 * 1299 + in7 * 1583 + 1024) >> 11;
    let t15a = ((in1 * (4076 - 4096) + in15 * 401 + 2048) >> 12) + in1;
    // stage 2 (butterflies)
    let t8 = clip(t8a + t9a);
    let t9 = clip(t8a - t9a);
    let t10 = clip(t11a - t10a);
    let t11 = clip(t11a + t10a);
    let t12 = clip(t12a + t13a);
    let t13 = clip(t12a - t13a);
    let t14 = clip(t15a - t14a);
    let t15 = clip(t15a + t14a);
    // stage 3 (rotations)
    let t9a = ((t14 * 1567 - t9 * (3784 - 4096) + 2048) >> 12) - t9;
    let t14a = ((t14 * (3784 - 4096) + t9 * 1567 + 2048) >> 12) + t14;
    let t10a = ((-(t13 * (3784 - 4096) + t10 * 1567) + 2048) >> 12) - t13;
    let t13a = ((t13 * 1567 - t10 * (3784 - 4096) + 2048) >> 12) - t10;
    // stage 4 (butterflies)
    let t8a = clip(t8 + t11);
    let t9 = clip(t9a + t10a);
    let t10 = clip(t9a - t10a);
    let t11a = clip(t8 - t11);
    let t12a = clip(t15 - t12);
    let t13 = clip(t14a - t13a);
    let t14 = clip(t14a + t13a);
    let t15a = clip(t15 + t12);
    // stage 5 (181/256 rotations)
    let t10a = ((t13 - t10) * 181 + 128) >> 8;
    let t13a = ((t13 + t10) * 181 + 128) >> 8;
    let t11 = ((t12a - t11a) * 181 + 128) >> 8;
    let t12 = ((t12a + t11a) * 181 + 128) >> 8;
    // even part (already transformed, in c at even positions)
    let (t0, t1, t2, t3) = (c[0], c[2 * s], c[4 * s], c[6 * s]);
    let (t4, t5, t6, t7) = (c[8 * s], c[10 * s], c[12 * s], c[14 * s]);
    c[0] = clip(t0 + t15a);
    c[s] = clip(t1 + t14);
    c[2 * s] = clip(t2 + t13a);
    c[3 * s] = clip(t3 + t12);
    c[4 * s] = clip(t4 + t11);
    c[5 * s] = clip(t5 + t10a);
    c[6 * s] = clip(t6 + t9);
    c[7 * s] = clip(t7 + t8a);
    c[8 * s] = clip(t7 - t8a);
    c[9 * s] = clip(t6 - t9);
    c[10 * s] = clip(t5 - t10a);
    c[11 * s] = clip(t4 - t11);
    c[12 * s] = clip(t3 - t12);
    c[13 * s] = clip(t2 - t13a);
    c[14 * s] = clip(t1 - t14);
    c[15 * s] = clip(t0 - t15a);
}

/// Reconstruct a 16x16 residual from quantized levels via dav1d's EXACT integer
/// inverse (TX_16X16 DCT_DCT, 8-bit). dq_shift = max(0, ctx-2) = 0 for TX_16X16
/// (same as TX_8X8); 2D shift = 2 (`inv_txfm_fn16(16,16,2)`): row inv_dct16,
/// (t+2)>>2 clip int16, col inv_dct16, (t+8)>>4.
fn idct_dequant_16x16(levels: &[i32; 256], q: &impl Dct) -> [i32; 256] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 256];
    for rc in 0..256 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 256];
    for y in 0..16 {
        for x in 0..16 {
            tmp[y * 16 + x] = coeff[y + x * 16];
        }
    }
    for y in 0..16 {
        inv_dct16_1d(&mut tmp[y * 16..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 2) >> 2).clamp(cmin, cmax);
    }
    for x in 0..16 {
        inv_dct16_1d(&mut tmp[x..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d-exact integer inverse 32-point DCT (`inv_dct32_1d_internal_c`, tx64=0).
/// Even positions are handled by `inv_dct16_1d`; the 16 odd-position inputs go
/// through the AV1 rotation/butterfly stages verbatim. Mutable locals are
/// reassigned in the exact order of the C source so the sequential semantics
/// match.
fn inv_dct32_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    inv_dct16_1d(c, 2 * s, min, max); // even positions c[0],c[2s],..,c[30s]

    let (in1, in3, in5, in7) = (c[s], c[3 * s], c[5 * s], c[7 * s]);
    let (in9, in11, in13, in15) = (c[9 * s], c[11 * s], c[13 * s], c[15 * s]);
    let (in17, in19, in21, in23) = (c[17 * s], c[19 * s], c[21 * s], c[23 * s]);
    let (in25, in27, in29, in31) = (c[25 * s], c[27 * s], c[29 * s], c[31 * s]);

    // stage 1
    let mut t16a = ((in1 * 201 - in31 * (4091 - 4096) + 2048) >> 12) - in31;
    let mut t17a = ((in17 * (3035 - 4096) - in15 * 2751 + 2048) >> 12) + in17;
    let mut t18a = ((in9 * 1751 - in23 * (3703 - 4096) + 2048) >> 12) - in23;
    let mut t19a = ((in25 * (3857 - 4096) - in7 * 1380 + 2048) >> 12) + in25;
    let mut t20a = ((in5 * 995 - in27 * (3973 - 4096) + 2048) >> 12) - in27;
    let mut t21a = ((in21 * (3513 - 4096) - in11 * 2106 + 2048) >> 12) + in21;
    let mut t22a = (in13 * 1220 - in19 * 1645 + 1024) >> 11;
    let mut t23a = ((in29 * (4052 - 4096) - in3 * 601 + 2048) >> 12) + in29;
    let mut t24a = ((in29 * 601 + in3 * (4052 - 4096) + 2048) >> 12) + in3;
    let mut t25a = (in13 * 1645 + in19 * 1220 + 1024) >> 11;
    let mut t26a = ((in21 * 2106 + in11 * (3513 - 4096) + 2048) >> 12) + in11;
    let mut t27a = ((in5 * (3973 - 4096) + in27 * 995 + 2048) >> 12) + in5;
    let mut t28a = ((in25 * 1380 + in7 * (3857 - 4096) + 2048) >> 12) + in7;
    let mut t29a = ((in9 * (3703 - 4096) + in23 * 1751 + 2048) >> 12) + in9;
    let mut t30a = ((in17 * 2751 + in15 * (3035 - 4096) + 2048) >> 12) + in15;
    let mut t31a = ((in1 * (4091 - 4096) + in31 * 201 + 2048) >> 12) + in1;

    // stage 2
    let mut t16 = clip(t16a + t17a);
    let mut t17 = clip(t16a - t17a);
    let mut t18 = clip(t19a - t18a);
    let mut t19 = clip(t19a + t18a);
    let mut t20 = clip(t20a + t21a);
    let mut t21 = clip(t20a - t21a);
    let mut t22 = clip(t23a - t22a);
    let mut t23 = clip(t23a + t22a);
    let mut t24 = clip(t24a + t25a);
    let mut t25 = clip(t24a - t25a);
    let mut t26 = clip(t27a - t26a);
    let mut t27 = clip(t27a + t26a);
    let mut t28 = clip(t28a + t29a);
    let mut t29 = clip(t28a - t29a);
    let mut t30 = clip(t31a - t30a);
    let mut t31 = clip(t31a + t30a);

    // stage 3
    t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    // stage 4
    t16a = clip(t16 + t19);
    t17 = clip(t17a + t18a);
    t18 = clip(t17a - t18a);
    t19a = clip(t16 - t19);
    t20a = clip(t23 - t20);
    t21 = clip(t22a - t21a);
    t22 = clip(t22a + t21a);
    t23a = clip(t23 + t20);
    t24a = clip(t24 + t27);
    t25 = clip(t25a + t26a);
    t26 = clip(t25a - t26a);
    t27a = clip(t24 - t27);
    t28a = clip(t31 - t28);
    t29 = clip(t30a - t29a);
    t30 = clip(t30a + t29a);
    t31a = clip(t31 + t28);

    // stage 5
    t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    // stage 6
    t16 = clip(t16a + t23a);
    t17a = clip(t17 + t22);
    t18 = clip(t18a + t21a);
    t19a = clip(t19 + t20);
    t20a = clip(t19 - t20);
    t21 = clip(t18a - t21a);
    t22a = clip(t17 - t22);
    t23 = clip(t16a - t23a);
    t24 = clip(t31a - t24a);
    t25a = clip(t30 - t25);
    t26 = clip(t29a - t26a);
    t27a = clip(t28 - t27);
    t28a = clip(t28 + t27);
    t29 = clip(t29a + t26a);
    t30a = clip(t30 + t25);
    t31 = clip(t31a + t24a);

    // stage 7 (181/256 rotations)
    t20 = ((t27a - t20a) * 181 + 128) >> 8;
    t27 = ((t27a + t20a) * 181 + 128) >> 8;
    t21a = ((t26 - t21) * 181 + 128) >> 8;
    t26a = ((t26 + t21) * 181 + 128) >> 8;
    t22 = ((t25a - t22a) * 181 + 128) >> 8;
    t25 = ((t25a + t22a) * 181 + 128) >> 8;
    t23a = ((t24 - t23) * 181 + 128) >> 8;
    t24a = ((t24 + t23) * 181 + 128) >> 8;

    // even results (in c at positions 0,2s,..,30s)
    let (t0, t1, t2, t3) = (c[0], c[2 * s], c[4 * s], c[6 * s]);
    let (t4, t5, t6, t7) = (c[8 * s], c[10 * s], c[12 * s], c[14 * s]);
    let (t8, t9, t10, t11) = (c[16 * s], c[18 * s], c[20 * s], c[22 * s]);
    let (t12, t13, t14, t15) = (c[24 * s], c[26 * s], c[28 * s], c[30 * s]);

    c[0] = clip(t0 + t31);
    c[s] = clip(t1 + t30a);
    c[2 * s] = clip(t2 + t29);
    c[3 * s] = clip(t3 + t28a);
    c[4 * s] = clip(t4 + t27);
    c[5 * s] = clip(t5 + t26a);
    c[6 * s] = clip(t6 + t25);
    c[7 * s] = clip(t7 + t24a);
    c[8 * s] = clip(t8 + t23a);
    c[9 * s] = clip(t9 + t22);
    c[10 * s] = clip(t10 + t21a);
    c[11 * s] = clip(t11 + t20);
    c[12 * s] = clip(t12 + t19a);
    c[13 * s] = clip(t13 + t18);
    c[14 * s] = clip(t14 + t17a);
    c[15 * s] = clip(t15 + t16);
    c[16 * s] = clip(t15 - t16);
    c[17 * s] = clip(t14 - t17a);
    c[18 * s] = clip(t13 - t18);
    c[19 * s] = clip(t12 - t19a);
    c[20 * s] = clip(t11 - t20);
    c[21 * s] = clip(t10 - t21a);
    c[22 * s] = clip(t9 - t22);
    c[23 * s] = clip(t8 - t23a);
    c[24 * s] = clip(t7 - t24a);
    c[25 * s] = clip(t6 - t25);
    c[26 * s] = clip(t5 - t26a);
    c[27 * s] = clip(t4 - t27);
    c[28 * s] = clip(t3 - t28a);
    c[29 * s] = clip(t2 - t29);
    c[30 * s] = clip(t1 - t30a);
    c[31 * s] = clip(t0 - t31);
}

/// Reconstruct a 32x32 residual from quantized levels via dav1d's EXACT integer
/// inverse (TX_32X32 DCT_DCT, 8-bit). `dq_shift = max(0, ctx-2) = 1` for
/// TX_32X32; 2D shift: row inv_dct32, (t+2)>>2 clip int16, col inv_dct32,
/// (t+8)>>4.
fn idct_dequant_32x32(levels: &[i32; 1024], q: &impl Dct) -> [i32; 1024] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 1024];
    for rc in 0..1024 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        // mask to 24 bits, then dq_shift = 1, then clamp to cf_max
        let mag = (((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) >> 1) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            tmp[y * 32 + x] = coeff[y + x * 32];
        }
    }
    for y in 0..32 {
        inv_dct32_1d(&mut tmp[y * 32..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 2) >> 2).clamp(cmin, cmax);
    }
    for x in 0..32 {
        inv_dct32_1d(&mut tmp[x..], 32, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// `forward_dct_quant_8x8`: orthonormal float DCT (rows then cols), scaled, then
/// /q (dc_q for the (0,0) coefficient, ac_q otherwise). Output in dav1d order
/// `cf[u*16+v]`. The scale is calibrated so the round-trip through the exact
/// integer inverse recovers the residual; only the encoder uses this (recon is
/// the exact inverse), so its precision does not affect bit-exactness.
pub fn forward_dct_quant_16x16(residual: &mut [i32; 256], q: &impl Dct) {
    // forward_dct_quant_16x16_t(residual, q).0
    dct16x16(residual, q)
}

/// As [`forward_dct_quant_16x16`] but also returns the pre-round real targets.
pub fn forward_dct_quant_16x16_t(residual: &[i32; 256], q: &impl Dct) -> ([i32; 256], [f64; 256]) {
    const N: usize = 16;
    const SCALE: f64 = 8.0; // G16=1/128, ortho-DC=16V -> SCALE=128/16=8 (same as 8x8)
    let mut m = [[0.0f64; N]; N];
    for k in 0..N {
        let s: f64 = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / N as f64).sqrt();
        for n in 0..N {
            m[k][n] =
                (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / (2.0 * N as f64)).cos() * s;
        }
    }
    let mut tmp = [[0.0f64; N]; N]; // tmp[v][x] = sum_y M[v][y] * R[y][x]
    for v in 0..N {
        for x in 0..N {
            let mut acc = 0.0;
            for y in 0..N {
                acc += m[v][y] * residual[y * N + x] as f64;
            }
            tmp[v][x] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 256];
    let mut tf = [0.0f64; 256];
    for v in 0..N {
        for u in 0..N {
            let mut c = 0.0;
            for x in 0..N {
                c += m[u][x] * tmp[v][x];
            }
            c *= SCALE;
            let dq = if v == 0 && u == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[u * N + v] = q;
            cf[u * N + v] = q.round() as i32;
        }
    }
    (cf, tf)
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

/// Forward DCT + quantize a 32x32 residual (`residual[row*32+col]`). `SCALE` is
/// calibrated so the round-trip through the exact integer inverse (which for
/// TX_32X32 includes the extra `dq_shift = 1`) recovers the residual. Only the
/// encoder uses this; recon is the exact inverse, so its precision does not
/// affect bit-exactness.
const FDCT32_SCALE: f64 = 8.0;
pub fn forward_dct_quant_32x32(residual: &mut [i32; 1024], q: &impl Dct) {
    // forward_dct_quant_32x32_scaled(residual, q, FDCT32_SCALE).0
    dct32x32(residual, q)
}

pub fn forward_dct_quant_32x32_t(
    residual: &[i32; 1024],
    q: &impl Dct,
) -> ([i32; 1024], [f64; 1024]) {
    forward_dct_quant_32x32_scaled(residual, q, FDCT32_SCALE)
}
fn forward_dct_quant_32x32_scaled(
    residual: &[i32; 1024],
    q: &impl Dct,
    scale: f64,
) -> ([i32; 1024], [f64; 1024]) {
    const N: usize = 32;
    let mut m = [[0.0f64; N]; N];
    for k in 0..N {
        let s: f64 = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / N as f64).sqrt();
        for n in 0..N {
            m[k][n] =
                (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / (2.0 * N as f64)).cos() * s;
        }
    }
    let mut tmp = vec![[0.0f64; N]; N];
    for v in 0..N {
        for x in 0..N {
            let mut acc = 0.0;
            for y in 0..N {
                acc += m[v][y] * residual[y * N + x] as f64;
            }
            tmp[v][x] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 1024];
    let mut tf = [0.0f64; 1024];
    for v in 0..N {
        for u in 0..N {
            let mut c = 0.0;
            for x in 0..N {
                c += m[u][x] * tmp[v][x];
            }
            c *= scale;
            let dq = if v == 0 && u == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[u * N + v] = q;
            cf[u * N + v] = q.round() as i32;
        }
    }
    (cf, tf)
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

/// Forward DCT + quantize a 4-wide x 8-tall chroma residual (`residual[row*4+col]`,
/// row 0..8, col 0..4) for 4:2:2 chroma (`RTX_4X8`). Returns quantized levels in
/// dav1d coefficient order `cf[fx*8 + fy]` (fx = horizontal freq 0..4, fy =
/// vertical freq 0..8) so the integer inverse below reconstructs them exactly.
pub fn forward_dct_quant_4x8(residual: &[i32; 32], q: &impl Dct) -> [i32; 32] {
    forward_dct_quant_4x8_t(residual, q).0
}

/// As [`forward_dct_quant_4x8`] but also returns the pre-round real targets.
pub fn forward_dct_quant_4x8_t(residual: &[i32; 32], q: &impl Dct) -> ([i32; 32], [f64; 32]) {
    // width-4 and height-8 orthonormal DCT bases
    let mut m4 = [[0.0f64; 4]; 4];
    for k in 0..4 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 4.0).sqrt();
        for n in 0..4 {
            m4[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 8.0).cos() * s;
        }
    }
    let mut m8 = [[0.0f64; 8]; 8];
    for k in 0..8 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 8.0).sqrt();
        for n in 0..8 {
            m8[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos() * s;
        }
    }
    // tmp[fy][col] = sum_row m8[fy][row] * resid[row*4+col]
    let mut tmp = [[0.0f64; 4]; 8];
    for fy in 0..8 {
        for col in 0..4 {
            let mut acc = 0.0;
            for row in 0..8 {
                acc += m8[fy][row] * residual[row * 4 + col] as f64;
            }
            tmp[fy][col] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 32];
    let mut tf = [0.0f64; 32];
    for fx in 0..4 {
        for fy in 0..8 {
            let mut c = 0.0;
            for col in 0..4 {
                c += m4[fx][col] * tmp[fy][col];
            }
            c *= 8.0; // same overall scale as TX_8X8 (the rect2 *181 in the inverse compensates)
            let dq = if fx == 0 && fy == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[fx * 8 + fy] = q;
            cf[fx * 8 + fy] = q.round() as i32;
        }
    }
    (cf, tf)
}

/// dav1d's EXACT integer inverse for `RTX_4X8` (4 wide x 8 tall, 8-bit, shift=0,
/// is_rect2): dequant `level*q` (clamped int16), the rect2 `*181>>8` prescale,
/// a width-4 row `inv_dct4`, then a height-8 column `inv_dct8`, then `(+8)>>4`.
/// Bit-identical to dav1d's chroma reconstruction. `levels[fx*8+fy]`; output
/// residual `r[row*4+col]`.
fn idct_dequant_4x8(levels: &[i32; 32], q: &impl Dct) -> [i32; 32] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 32];
    for rc in 0..32 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // tmp[row*4+col] = (coeff[row + col*8] * 181 + 128) >> 8   (is_rect2 prescale)
    let mut tmp = [0i32; 32];
    for row in 0..8 {
        for col in 0..4 {
            tmp[row * 4 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    // row transform: width-4 inv_dct4 (stride 1) over each of the 8 rows
    for row in 0..8 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    // shift = 0 => only clip
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    // column transform: height-8 inv_dct8 (stride 4) over each of the 4 columns
    for col in 0..4 {
        inv_dct8_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// Forward DCT + quantize an 8-wide x 16-tall chroma residual
/// (`residual[row*8 + col]`, row = 0..16) for 4:2:2 `RTX_8X16`. Returns levels
/// in dav1d's transposed layout `cf[fx*16 + fy]` (fx = horizontal freq 0..8,
/// fy = vertical freq 0..16). The dav1d inverse applies an `is_rect2` `*181>>8`
/// rescale, so the forward uses the same overall scale as the square transforms
/// (SCALE = 8, calibrated by flat round-trip).
pub fn forward_dct_quant_8x16(residual: &mut [i32; 128], q: &impl Dct) {
    dct8x16_i32(residual, q)
}

/// As [`forward_dct_quant_8x16`] but also returns the pre-round real targets.
pub fn forward_dct_quant_8x16_t(residual: &[i32; 128], q: &impl Dct) -> ([i32; 128], [f64; 128]) {
    const SCALE: f64 = 8.0;
    // width-8 and height-16 orthonormal DCT bases
    let mut m8 = [[0.0f64; 8]; 8];
    for k in 0..8 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 8.0).sqrt();
        for n in 0..8 {
            m8[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos() * s;
        }
    }
    let mut m16 = [[0.0f64; 16]; 16];
    for k in 0..16 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 16.0).sqrt();
        for n in 0..16 {
            m16[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 32.0).cos() * s;
        }
    }
    // tmp[fy][col] = sum_row m16[fy][row] * resid[row*8 + col]
    let mut tmp = [[0.0f64; 8]; 16];
    for fy in 0..16 {
        for col in 0..8 {
            let mut acc = 0.0;
            for row in 0..16 {
                acc += m16[fy][row] * residual[row * 8 + col] as f64;
            }
            tmp[fy][col] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 128];
    let mut tf = [0.0f64; 128];
    for fx in 0..8 {
        for fy in 0..16 {
            let mut c = 0.0;
            for col in 0..8 {
                c += m8[fx][col] * tmp[fy][col];
            }
            c *= SCALE;
            let dq = if fx == 0 && fy == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[fx * 16 + fy] = q;
            cf[fx * 16 + fy] = q.round() as i32;
        }
    }
    (cf, tf)
}

/// dav1d's EXACT integer inverse for `RTX_8X16` (8 wide x 16 tall, 8-bit,
/// `is_rect2`, mid-shift = 1): dequant `level*q` (clamped int16), the rect2
/// `*181>>8` prescale, a width-8 row `inv_dct8`, the `(+1)>>1` mid-shift, a
/// height-16 column `inv_dct16`, then `(+8)>>4`. (`inv_txfm_fn84(8,16,1)`.)
fn idct_dequant_8x16(levels: &[i32; 128], q: &impl Dct) -> [i32; 128] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 128];
    for rc in 0..128 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // rect2 prescale + transpose: tmp[row*8+col] = (coeff[row + col*16]*181+128)>>8
    let mut tmp = [0i32; 128];
    for row in 0..16 {
        for col in 0..8 {
            tmp[row * 8 + col] = (coeff[row + col * 16] * 181 + 128) >> 8;
        }
    }
    // row transform: width-8 inv_dct8 (stride 1) over each of the 16 rows
    for row in 0..16 {
        inv_dct8_1d(&mut tmp[row * 8..], 1, rmin, rmax);
    }
    // mid shift = 1: (t + 1) >> 1, clipped to int16
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    // column transform: height-16 inv_dct16 (stride 8) over each of the 8 columns
    for col in 0..8 {
        inv_dct16_1d(&mut tmp[col..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d's EXACT integer inverse for `RTX_16X32` (16 wide x 32 tall, 8-bit,
/// `is_rect2`, dequant `dq_shift = 1` since `txsize_sqr_up = TX_32X32` so
/// `ctx = 3`, row/mid shift = 1, final shift = 4). Mirrors `idct_dequant_8x16`
/// scaled up: dequant `level*q` (>>1, clamped int16), the rect2 `181/256`
/// prescale + transpose, width-16 row DCTs, the mid shift, then height-32
/// column DCTs. `levels`/coeffs are in dav1d order `cf[fx*32 + fy]`.
fn idct_dequant_16x32(levels: &[i32; 512], q: &impl Dct) -> [i32; 512] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 512];
    for rc in 0..512 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = (((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) >> 1) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // rect2 prescale + transpose: tmp[row*16+col] = (coeff[row + col*32]*181+128)>>8
    let mut tmp = [0i32; 512];
    for row in 0..32 {
        for col in 0..16 {
            tmp[row * 16 + col] = (coeff[row + col * 32] * 181 + 128) >> 8;
        }
    }
    // row transform: width-16 inv_dct16 (stride 1) over each of the 32 rows
    for row in 0..32 {
        inv_dct16_1d(&mut tmp[row * 16..], 1, rmin, rmax);
    }
    // mid shift = 1: (t + 1) >> 1, clipped to int16
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    // column transform: height-32 inv_dct32 (stride 16) over each of the 16 columns
    for col in 0..16 {
        inv_dct32_1d(&mut tmp[col..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// Forward DCT + quantize a 16-wide x 32-tall residual (`residual[row*16+col]`,
/// row 0..32, col 0..16) for 4:2:2 `RTX_16X32`. Returns quantized levels in
/// dav1d coef order `cf[fx*32 + fy]` (fx = horizontal freq 0..16, fy = vertical
/// freq 0..32). `SCALE` is calibrated so the round-trip through the exact
/// integer inverse (which includes the `dq_shift = 1`) recovers the residual.
const FDCT1632_SCALE: f64 = 8.0;
pub fn forward_dct_quant_16x32(residual: &[i32; 512], q: &impl Dct) -> [i32; 512] {
    forward_dct_quant_16x32_scaled(residual, q, FDCT1632_SCALE).0
}
pub fn forward_dct_quant_16x32_t(residual: &[i32; 512], q: &impl Dct) -> ([i32; 512], [f64; 512]) {
    forward_dct_quant_16x32_scaled(residual, q, FDCT1632_SCALE)
}
fn forward_dct_quant_16x32_scaled(
    residual: &[i32; 512],
    q: &impl Dct,
    scale: f64,
) -> ([i32; 512], [f64; 512]) {
    let mut m16 = [[0.0f64; 16]; 16];
    for k in 0..16 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 16.0).sqrt();
        for n in 0..16 {
            m16[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 32.0).cos() * s;
        }
    }
    let mut m32 = [[0.0f64; 32]; 32];
    for k in 0..32 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 32.0).sqrt();
        for n in 0..32 {
            m32[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 64.0).cos() * s;
        }
    }
    // tmp[fy][col] = sum_row m32[fy][row] * resid[row*16 + col]
    let mut tmp = [[0.0f64; 16]; 32];
    for fy in 0..32 {
        for col in 0..16 {
            let mut acc = 0.0;
            for row in 0..32 {
                acc += m32[fy][row] * residual[row * 16 + col] as f64;
            }
            tmp[fy][col] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 512];
    let mut tf = [0.0f64; 512];
    for fx in 0..16 {
        for fy in 0..32 {
            let mut c = 0.0;
            for col in 0..16 {
                c += m16[fx][col] * tmp[fy][col];
            }
            c *= scale;
            let dq = if fx == 0 && fy == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[fx * 32 + fy] = q;
            cf[fx * 32 + fy] = q.round() as i32;
        }
    }
    (cf, tf)
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

/// 4:2:0 chroma (`TX_4X4`). Returns levels in dav1d order `cf[fx*4+fy]`.
pub fn forward_dct_quant_4x4(residual: &[i32; 16], q: &impl Dct) -> [i32; 16] {
    forward_dct_quant_4x4_t(residual, q).0
}

/// As [`forward_dct_quant_4x4`] but also returns the pre-round real targets.
pub fn forward_dct_quant_4x4_t(residual: &[i32; 16], q: &impl Dct) -> ([i32; 16], [f64; 16]) {
    let mut m = [[0.0f64; 4]; 4];
    for k in 0..4 {
        let s = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 4.0).sqrt();
        for n in 0..4 {
            m[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 8.0).cos() * s;
        }
    }
    // tmp[fy][col] = sum_row m[fy][row] * resid[row*4+col]
    let mut tmp = [[0.0f64; 4]; 4];
    for fy in 0..4 {
        for col in 0..4 {
            let mut acc = 0.0;
            for row in 0..4 {
                acc += m[fy][row] * residual[row * 4 + col] as f64;
            }
            tmp[fy][col] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 16];
    let mut tf = [0.0f64; 16];
    for fx in 0..4 {
        for fy in 0..4 {
            let mut c = 0.0;
            for col in 0..4 {
                c += m[fx][col] * tmp[fy][col];
            }
            c *= 8.0;
            let dq = if fx == 0 && fy == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[fx * 4 + fy] = q;
            cf[fx * 4 + fy] = q.round() as i32;
        }
    }
    (cf, tf)
}

/// dav1d's EXACT integer inverse for `TX_4X4` (8-bit, shift=0, square/no rect2):
/// dequant `level*q` (clamped int16), a row `inv_dct4`, a column `inv_dct4`,
/// then `(+8)>>4`. `levels[fx*4+fy]`; output residual `r[row*4+col]`.
fn idct_dequant_4x4(levels: &[i32; 16], q: &impl Dct) -> [i32; 16] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut tmp = [0i32; 16];
    for rc in 0..16 {
        let lvl = levels[rc];
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        // c[x] = coeff[y + x*4]; here rc = fx*4 + fy => place at tmp[fy*4+fx]
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    // row transform: inv_dct4 (stride 1) over each of the 4 rows; shift=0
    for row in 0..4 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    // column transform: inv_dct4 (stride 4) over each of the 4 columns
    for col in 0..4 {
        inv_dct4_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
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
        if y_mode >= V_PRED && y_mode <= VERT_LEFT_PRED {
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
                for ci in 0..2 {
                    ccf[ci] = cfl_ccf[ci];
                }
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
        for ci in 0..2 {
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
        if !self.ss420 && !self.ss422 {
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
                for ci in 0..2 {
                    ccf8[ci] = cfl_ccf[ci];
                }
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
        let block_skip = lcf.iter().all(|&c| c == 0) && chroma_zero(0) && chroma_zero(1);

        // block-level mode info: skip (ctx = above_skip + left_skip), y/uv = DC
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
        if best_mode >= V_PRED && best_mode <= VERT_LEFT_PRED {
            // angle_delta = 0 (symbol index 3); 8x8 satisfies the size condition
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[best_mode - V_PRED]);
        }
        self.emit_uv_mode(best_mode, if use_cfl { Some(cfl_alpha_uv) } else { None });
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
        for ci in 0..2 {
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
        if y_mode >= V_PRED && y_mode <= VERT_LEFT_PRED {
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

pub fn encode_av1_lossy_image(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
) -> Vec<u8> {
    let (li, ui, vi): (Vec<i32>, Vec<i32>, Vec<i32>) = (
        luma.iter().map(|&x| x as i32).collect(),
        u.iter().map(|&x| x as i32).collect(),
        v.iter().map(|&x| x as i32).collect(),
    );
    encode_av1_lossy_image_cs(base_q_idx, bd, w, h, &li, &ui, &vi, false)
}

/// Lossy encoder with explicit colour mode: `ycbcr=false` signals MC_IDENTITY
/// (planes coded as GBR); `ycbcr=true` signals full-range BT.601 so the decoder
/// converts the coded Y/Cb/Cr planes back to RGB (decorrelated -> smaller).
pub fn encode_av1_lossy_image_cs(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    ycbcr: bool,
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
    let mut tile = LossyTile::new(base_q_idx, bd, w8, h8, &src);
    // superblock raster order; partial edge superblocks are split by decode_sb
    for sb_y in (0..h8).step_by(64) {
        for sb_x in (0..w8).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let payload = tile.enc.done();
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    let mc = if ycbcr { 6 } else { 0 };
    let profile = if bd == 12 { 2 } else { 1 };
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32, h as u32, profile, bd, mc, 0, 0,
    ));
    bytes.extend_from_slice(&wrap_obu_frame(
        &frame_header_lossy_tiled(base_q_idx, sb_cols, sb_rows),
        &payload,
    ));
    bytes
}

/// Debug helper: identity (GBR) lossy 4:4:4 encode that also returns the
/// encoder's reconstructed planes (aligned `w8*h8`) so callers can verify the
/// stream is bit-exact against dav1d's decoded output. Not part of the public API.
#[doc(hidden)]
pub fn encode_av1_lossy_image_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
) -> (Vec<u8>, [Vec<i32>; 3], usize, usize) {
    let (w8, h8) = (align8(w), align8(h));
    let src = [
        pad_to_mult8(luma, w, h, w8, h8),
        pad_to_mult8(u, w, h, w8, h8),
        pad_to_mult8(v, w, h, w8, h8),
    ];
    let mut tile = LossyTile::new(base_q_idx, bd, w8, h8, &src);
    for sb_y in (0..h8).step_by(64) {
        for sb_x in (0..w8).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let recon = tile.recon.clone();
    let payload = tile.enc.done();
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    let profile = if bd == 12 { 2 } else { 1 };
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32, h as u32, profile, bd, 0, 0, 0,
    ));
    bytes.extend_from_slice(&wrap_obu_frame(
        &frame_header_lossy_tiled(base_q_idx, sb_cols, sb_rows),
        &payload,
    ));
    (bytes, recon, w8, h8)
}

/// Debug variant of [`encode_av1_lossy_image_420`] returning the encoder's
/// reconstruction (luma + the two subsampled chroma planes) and the coded
/// dimensions, for bit-exactness checks against a decoder.
#[doc(hidden)]
pub fn encode_av1_lossy_image_420_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
) -> (Vec<u8>, [Vec<i32>; 3], usize, usize, usize, usize) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let (w8, h8) = (align8(w), align8(h));
    let (cw8, ch8) = (w8 / 2, h8 / 2);
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| pad_to_mult8(p, cw, ch, cw8, ch8);
    let src = [luma_p, pad_c(u), pad_c(v)];
    let mut tile = LossyTile::new_420(base_q_idx, bd, w8, h8, &src);
    for sb_y in (0..h8).step_by(64) {
        for sb_x in (0..w8).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let recon = tile.recon.clone();
    let payload = tile.enc.done();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32,
        h as u32,
        if bd == 12 { 2 } else { 0 },
        bd,
        6,
        1,
        1,
    ));
    bytes.extend_from_slice(&wrap_obu_frame(
        &frame_header_lossy_tiled(base_q_idx, w8.div_ceil(64) as u32, h8.div_ceil(64) as u32),
        &payload,
    ));
    (bytes, recon, w8, h8, cw8, ch8)
}

/// Debug variant of [`encode_av1_lossy_image_422`] returning the encoder's
/// reconstruction and coded dimensions (chroma is `cw8` x `h8`).
#[doc(hidden)]
pub fn encode_av1_lossy_image_422_recon_dbg(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
) -> (Vec<u8>, [Vec<i32>; 3], usize, usize, usize) {
    let cw = w.div_ceil(2);
    let (w8, h8) = (align8(w), align8(h));
    let cw8 = w8 / 2;
    let luma_p: Vec<i32> = pad_to_mult8(luma, w, h, w8, h8);
    let pad_c = |p: &[i32]| pad_to_mult8(p, cw, h, cw8, h8);
    let src = [luma_p, pad_c(u), pad_c(v)];
    let mut tile = LossyTile::new_422(base_q_idx, bd, w8, h8, &src);
    for sb_y in (0..h8).step_by(64) {
        for sb_x in (0..w8).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let recon = tile.recon.clone();
    let payload = tile.enc.done();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32, h as u32, 2, bd, 6, 1, 0,
    ));
    bytes.extend_from_slice(&wrap_obu_frame(
        &frame_header_lossy_tiled(base_q_idx, w8.div_ceil(64) as u32, h8.div_ceil(64) as u32),
        &payload,
    ));
    (bytes, recon, w8, h8, cw8)
}

/// Encode a **lossy 4:2:2** YCbCr still (profile 2). `luma` is `w*h`; `u`/`v`
/// are the horizontally-subsampled chroma planes, each `cw*h` with
/// `cw = (w+1)/2`. The decoder reconstructs full-resolution RGB via the
/// signalled BT.601 matrix and 4:2:2 upsampling.
pub fn encode_av1_lossy_image_422(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
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
    let mut tile = LossyTile::new_422(base_q_idx, bd, w8, h8, &src);
    for sb_y in (0..h8).step_by(64) {
        for sb_x in (0..w8).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let payload = tile.enc.done();
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    // profile 2 @ 8-bit => I422 (forced), BT.601 full-range YCbCr
    bytes.extend_from_slice(&crate::obu::sequence_header_mc(
        w as u32, h as u32, 2, bd, 6, 1, 0,
    ));
    bytes.extend_from_slice(&wrap_obu_frame(
        &frame_header_lossy_tiled(base_q_idx, sb_cols, sb_rows),
        &payload,
    ));
    bytes
}

/// Encode a **lossy 4:2:0** YCbCr still (profile 0). `luma` is `w*h`; `u`/`v`
/// are the half-width, half-height chroma planes, each `cw*ch` with
/// `cw=(w+1)/2`, `ch=(h+1)/2`. Each 8x8 luma block carries a 4x4 (`TX_4X4`)
/// chroma block per plane. Reconstruction is bit-exact vs dav1d 1.4.1.
pub fn encode_av1_lossy_image_420(
    base_q_idx: u8,
    bd: u8,
    w: usize,
    h: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
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
    let mut tile = LossyTile::new_420(base_q_idx, bd, w8, h8, &src);
    for sb_y in (0..h8).step_by(64) {
        for sb_x in (0..w8).step_by(64) {
            tile.decode_sb(1, sb_x / 8, sb_y / 8, 8, true, false);
        }
    }
    let payload = tile.enc.done();
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
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
    bytes.extend_from_slice(&wrap_obu_frame(
        &frame_header_lossy_tiled(base_q_idx, sb_cols, sb_rows),
        &payload,
    ));
    bytes
}

/// Convenience wrapper: lossy 64x64 still (see [`encode_av1_lossy_image`]).
pub fn encode_av1_lossy_image_64x64(
    base_q_idx: u8,
    luma: &[u8; 4096],
    u: &[u8; 4096],
    v: &[u8; 4096],
) -> Vec<u8> {
    let (li, ui, vi): (Vec<i32>, Vec<i32>, Vec<i32>) = (
        luma.iter().map(|&x| x as i32).collect(),
        u.iter().map(|&x| x as i32).collect(),
        v.iter().map(|&x| x as i32).collect(),
    );
    encode_av1_lossy_image(base_q_idx, 8, 64, 64, &li, &ui, &vi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct16x32_roundtrip_and_scale_calibration() {
        let make = |kind: u8| -> [i32; 512] {
            let mut r = [0i32; 512];
            for y in 0..32 {
                for x in 0..16 {
                    r[y * 16 + x] = match kind {
                        0 => 40,
                        1 => (x as i32 - 8) * 3 + (y as i32 - 16) * 2,
                        2 => (((x * 7 + y * 13) % 23) as i32 - 11) * 4,
                        _ => 0,
                    };
                }
            }
            r
        };
        for &scale in &[4.0f64, 5.6569, 8.0, 11.3137, 16.0] {
            let mut worst = 0.0f64;
            for kind in 0..3 {
                let r = make(kind);
                let (cf, _) = forward_dct_quant_16x32_scaled(&r, &Quant::new(32, 8), scale);
                let rec = idct_dequant_16x32(&cf, &Quant::new(32, 8));
                let mut se = 0.0;
                for i in 0..512 {
                    let d = (r[i] - rec[i]) as f64;
                    se += d * d;
                }
                worst = worst.max((se / 512.0).sqrt());
            }
            eprintln!("RTX_16X32 SCALE={:>8.4} worst RMSE={:.4}", scale, worst);
        }
        let r = make(1);
        let cf = forward_dct_quant_16x32(&r, &Quant::new(24, 8));
        let rec = idct_dequant_16x32(&cf, &Quant::new(24, 8));
        let maxe = (0..512).map(|i| (r[i] - rec[i]).abs()).max().unwrap();
        eprintln!("RTX_16X32 round-trip max abs err (q24) = {}", maxe);
        assert!(maxe < 8, "16x32 round-trip error too large: {}", maxe);
    }

    #[test]
    fn skip_keyframe_matches_dav1d_verified_bytes() {
        // This exact 20-byte stream was decoded by dav1d 1.4.1
        // ("Decoded 1/1 frames"), producing a 64x64 4:4:4 frame of all-128.
        // Frame header is 90 00 00 (disable_cdf_update = 1).
        let expected = [
            0x12, 0x00, // temporal delimiter
            0x0a, 0x09, 0x38, 0x15, 0x7f, 0xfc, 0x04, 0x04, 0x34, 0x00,
            0x80, // seq header OBU
            0x32, 0x05, 0x90, 0x00, 0x00, 0x98,
            0x80, // OBU_FRAME: hdr(90 00 00) + tile(98 80)
        ];
        assert_eq!(encode_av1_skip_keyframe_64x64(), expected);
    }

    #[test]
    fn dc_keyframe_gray_matches_dav1d_verified_bytes() {
        // 4x4 all-grey (r=0): dav1d 1.4.1 decodes "Decoded 1/1", Y=U=V=128.
        let expected = [
            0x12, 0x00, // temporal delimiter
            0x0a, 0x08, 0x38, 0x04, 0x7c, 0x04, 0x04, 0x34, 0x00, 0x80, // seq header (4x4)
            0x32, 0x05, 0x90, 0x00, 0x00, 0x2f, 0x80, // OBU_FRAME: hdr + tile(2f 80)
        ];
        assert_eq!(encode_av1_dc_keyframe_4x4(0, 0, 0), expected);
    }

    #[test]
    fn dc_keyframe_color_is_stable() {
        // A non-128 colour (carries real pixels via the DC coefficient chain).
        // Verified separately with dav1d 1.4.1: decodes to Y,U,V = 129,130,131.
        let bytes = encode_av1_dc_keyframe_4x4(1, 2, 3);
        // Frame is well-formed: TD + seq OBU + frame OBU, last tile byte present.
        assert_eq!(&bytes[0..2], &[0x12, 0x00]);
        assert!(bytes.len() > 20); // carries coefficient data
    }

    #[test]
    fn lossy_dc_8x8_matches_dav1d_verified_bytes() {
        // Lossy 8x8 still, base_q_idx=16, target residuals (10,-5,7).
        // Verified with dav1d 1.4.1: decodes to Y,U,V = 138,123,135 (= 128+r).
        let bytes = encode_av1_lossy_dc_8x8(16, 10, -5, 7);
        assert_eq!(
            bytes,
            vec![
                0x12, 0x00, 0x0a, 0x08, 0x38, 0x08, 0xbf, 0x01, 0x01, 0x0d, 0x00, 0x20, 0x32, 0x0f,
                0x91, 0x00, 0x00, 0x00, 0x00, 0x01, 0x5d, 0x92, 0x6a, 0x9f, 0x91, 0x02, 0x2d, 0x3f,
                0x9a,
            ]
        );
    }

    #[test]
    fn lossy_level_for_residual_inverts_dct_scaling() {
        // residual = (dc_q*level + 32) >> 6 ; level_for_residual inverts it.
        let dc_q = dc_q_8bit(16); // = 20
        for r in [1, 5, 10, 20, 40, -7, -30] {
            let level = level_for_residual(r, dc_q);
            let cf = dc_q as i32 * level;
            let decoded = (cf.abs() + 32) >> 6;
            let decoded = if cf < 0 { -decoded } else { decoded };
            assert_eq!(decoded, r, "r={} level={} cf={}", r, level, cf);
        }
    }

    #[test]
    fn lossy_luma_ac_8x8_matches_dav1d_verified_bytes() {
        // Luma DC+AC (eob=1, gradient), chroma flat. base_q=16, dc=2, ac=2.
        // Verified with dav1d 1.4.1: luma row decodes to [130,129,129,129,
        // 128,128,128,128] (a gradient, not flat); chroma stays 128.
        let bytes = encode_av1_lossy_luma_ac_8x8(16, 2, 2, 0, 0);
        assert_eq!(
            bytes,
            vec![
                0x12, 0x00, 0x0a, 0x08, 0x38, 0x08, 0xbf, 0x01, 0x01, 0x0d, 0x00, 0x20, 0x32, 0x09,
                0x91, 0x00, 0x00, 0x00, 0x00, 0x01, 0x64, 0xef, 0x80,
            ]
        );
    }

    #[test]
    fn lossy_luma_eob2_8x8_matches_dav1d_verified_bytes() {
        // Luma DC + 2 AC (eob=2), chroma flat. base_q=16, dc=ac1=ac2=2.
        // Exercises get_lo_ctx (scan[1] coeff resolves to base_tok ctx 1).
        // Verified with dav1d 1.4.1: decodes to a 2D gradient (131..127).
        let bytes = encode_av1_lossy_luma_eob2_8x8(16, 2, 2, 2, 0, 0);
        assert_eq!(
            bytes,
            vec![
                0x12, 0x00, 0x0a, 0x08, 0x38, 0x08, 0xbf, 0x01, 0x01, 0x0d, 0x00, 0x20, 0x32, 0x09,
                0x91, 0x00, 0x00, 0x00, 0x00, 0x01, 0x7c, 0x95, 0x04,
            ]
        );
    }

    #[test]
    fn lossy_luma_block_8x8_matches_dav1d_verified_bytes() {
        // General eob>0 path: eob=4 with coeffs at scan[0..4] (positions
        // 0,8,1,2,9), levels 2,2,1,2,1. Verified with dav1d 1.4.1: decodes with
        // get_lo_ctx contexts 1,6,3,2 (hand-checked against the per-symbol trace).
        let mut cf = [0i32; 64];
        cf[0] = 2;
        cf[8] = 2;
        cf[1] = 1;
        cf[2] = 2;
        cf[9] = 1;
        let bytes = encode_av1_lossy_luma_block_8x8(16, &cf, 0, 0);
        assert_eq!(
            bytes,
            vec![
                0x12, 0x00, 0x0a, 0x08, 0x38, 0x08, 0xbf, 0x01, 0x01, 0x0d, 0x00, 0x20, 0x32, 0x0a,
                0x91, 0x00, 0x00, 0x00, 0x00, 0x01, 0x95, 0x0b, 0xae, 0x80,
            ]
        );
    }

    #[test]
    fn lossy_luma_image_8x8_matches_dav1d_verified_bytes() {
        // 8x8 luma gradient image -> forward DCT + quantize + general encoder.
        // Verified with dav1d 1.4.1: decodes back to the gradient (max err 1).
        let mut px = [0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                px[y * 8 + x] = (120 + 4 * x as i32 + 3 * y as i32).clamp(0, 255) as u8;
            }
        }
        let bytes = encode_av1_lossy_luma_image_8x8(16, &px, 0, 0);
        assert_eq!(
            bytes,
            vec![
                0x12, 0x00, 0x0a, 0x08, 0x38, 0x08, 0xbf, 0x01, 0x01, 0x0d, 0x00, 0x20, 0x32, 0x10,
                0x91, 0x00, 0x00, 0x00, 0x00, 0x02, 0x45, 0xd3, 0x3f, 0x99, 0x1b, 0xfe, 0xe3, 0x31,
                0x06, 0xc7,
            ]
        );
    }

    #[test]
    fn lossy_color_image_8x8_matches_dav1d_verified_bytes() {
        // Full colour: luma gradient, U horizontal ramp, V vertical ramp.
        // Verified with dav1d 1.4.1: Y max err 1, U and V decode exactly.
        let (mut y, mut u, mut v) = ([0u8; 64], [0u8; 64], [0u8; 64]);
        for j in 0..8 {
            for i in 0..8 {
                y[j * 8 + i] = (120 + 4 * i as i32 + 3 * j as i32).clamp(0, 255) as u8;
                u[j * 8 + i] = (140 - 3 * i as i32).clamp(0, 255) as u8;
                v[j * 8 + i] = (110 + 4 * j as i32).clamp(0, 255) as u8;
            }
        }
        let bytes = encode_av1_lossy_color_image_8x8(16, &y, &u, &v);
        assert_eq!(
            bytes,
            vec![
                0x12, 0x00, 0x0a, 0x08, 0x38, 0x08, 0xbf, 0x01, 0x01, 0x0d, 0x00, 0x20, 0x32, 0x1e,
                0x91, 0x00, 0x00, 0x00, 0x00, 0x02, 0x45, 0xd3, 0x3f, 0x99, 0x1b, 0xfe, 0xe3, 0x31,
                0x06, 0xc4, 0x83, 0xdc, 0x95, 0x87, 0xee, 0x9a, 0x45, 0xd5, 0xad, 0x58, 0x17, 0xbb,
                0xc9, 0x70,
            ]
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
        );
        assert_eq!(bytes.len(), 47, "4:2:0 stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4358, "4:2:0 stream bytes drifted");
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
        );
        assert_eq!(bytes.len(), 267, "4:2:0 8x8-leaves stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 33620, "4:2:0 8x8-leaves stream bytes drifted");
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
        );
        assert_eq!(bytes.len(), 48, "4:2:2 stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 4399, "4:2:2 stream bytes drifted");
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
        );
        assert_eq!(bytes.len(), 321, "4:2:2 8x8-leaves stream length drifted");
        let sum: u32 = bytes.iter().map(|&x| x as u32).sum();
        assert_eq!(sum, 40775, "4:2:2 8x8-leaves stream bytes drifted");
    }

    #[test]
    fn lossy_64x64_flat_matches_dav1d_verified_bytes() {
        // 64x64 superblock, flat content: the partition R-D picks PARTITION_NONE
        // at each 16x16 (one TX_16X16 block instead of four TX_8X8). The first
        // 16x16 codes a tiny DC residual; the rest predict from it and skip.
        // dav1d 1.4.1 decodes to exact flat colour (verified bit-exact vs the
        // encoder reconstruction).
        let (y, u, v) = ([130u8; 4096], [120u8; 4096], [140u8; 4096]);
        let bytes = encode_av1_lossy_image_64x64(16, &y, &u, &v);
        assert_eq!(
            bytes,
            vec![
                18, 0, 10, 9, 56, 21, 127, 252, 4, 4, 52, 0, 128, 50, 17, 17, 0, 0, 0, 0, 180, 77,
                152, 109, 246, 233, 28, 168, 147, 66, 231, 172
            ]
        );
    }

    #[test]
    fn lossy_128x64_flat_matches_dav1d_verified_bytes() {
        // Two 64x64 superblocks side by side (multi-SB tiling + cross-SB
        // contexts), flat content coded as TX_16X16 blocks. Verified to decode
        // exactly in dav1d 1.4.1 and ffmpeg.
        let (w, h) = (128usize, 64usize);
        let (y, u, v) = (vec![130u8; w * h], vec![120u8; w * h], vec![140u8; w * h]);
        let bytes = encode_av1_lossy_image(
            16,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        );
        assert_eq!(
            bytes,
            vec![
                18, 0, 10, 9, 56, 25, 127, 254, 2, 2, 26, 0, 64, 50, 19, 16, 128, 0, 0, 0, 180, 77,
                152, 109, 246, 233, 28, 168, 147, 66, 231, 169, 14, 144
            ]
        );
    }

    /// Truly-arbitrary-size (non-multiple-of-8) lossy regression guard. 70×50 is
    /// padded to the 72×56 coded grid; the header signals 70×50 and the decoder
    /// crops. Verified to decode in dav1d 1.4.1 / ffmpeg at exactly 70×50.
    #[test]
    fn lossy_70x50_padded_stable() {
        let (w, h) = (70usize, 50usize);
        let (y, u, v) = (vec![130u8; w * h], vec![120u8; w * h], vec![140u8; w * h]);
        let p = encode_av1_lossy_image(
            16,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        );
        assert_eq!(p.len(), 40);
        assert_eq!(p.iter().map(|&x| x as u64).sum::<u64>(), 3212);
        assert_eq!(&p[..6], &[18, 0, 10, 9, 56, 25]);
    }

    /// Arbitrary-size (non-multiple-of-64) lossy regression guard. 80×80 has one
    /// full superblock plus right/bottom edge regions split to 8×8 leaves via the
    /// frame-edge partition logic. Verified to decode in dav1d 1.4.1 / ffmpeg
    /// (PSNR ~59 dB, max err 1).
    #[test]
    fn lossy_80x80_edge_stable() {
        let (w, h) = (80usize, 80usize);
        let (y, u, v) = (vec![130u8; w * h], vec![120u8; w * h], vec![140u8; w * h]);
        let p = encode_av1_lossy_image(
            16,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        );
        assert_eq!(p.len(), 37);
        assert_eq!(p.iter().map(|&x| x as u64).sum::<u64>(), 2873);
        assert_eq!(&p[..6], &[18, 0, 10, 9, 56, 25]);
    }

    #[test]
    fn lossy_64x64_gradient_tx32_coef_path_stable() {
        // 64x64 4:4:4 gradient at q32: the partition R-D picks PARTITION_NONE at
        // 32x32, so all three planes code real TX_32X32 coefficients (not just
        // the flat skip/DC path). Verified bit-exact against dav1d 1.4.1
        // (maxdiff 0, decoded vs encoder reconstruction). This guards the
        // TX_32X32 transform + coefficient coder against regressions.
        let (w, h) = (64usize, 64usize);
        let (mut y, mut u, mut v) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
        for yy in 0..h {
            for xx in 0..w {
                y[yy * w + xx] = ((xx + yy) * 2) as u8;
                u[yy * w + xx] = (xx * 3) as u8;
                v[yy * w + xx] = (yy * 3) as u8;
            }
        }
        let p = encode_av1_lossy_image(
            32,
            8,
            w,
            h,
            &y.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &u.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
            &v.iter().map(|&x| x as i32).collect::<Vec<i32>>(),
        );
        assert_eq!(p.len(), 170, "TX_32X32 gradient stream length drifted");
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            19690,
            "TX_32X32 gradient stream bytes drifted"
        );
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
        );
        assert_eq!(p.len(), 132, "32x32 4:2:0 stream length drifted");
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            15824,
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
        );
        assert_eq!(
            p.len(),
            151,
            "32x32 4:2:2 (RTX_16X32) stream length drifted"
        );
        assert_eq!(
            p.iter().map(|&x| x as u64).sum::<u64>(),
            17284,
            "32x32 4:2:2 (RTX_16X32) stream bytes drifted"
        );
    }

    #[test]
    fn quality_to_base_q_idx_endpoints_and_monotonic() {
        // endpoints: best quality -> finest lossy index, worst -> coarsest
        assert_eq!(quality_to_base_q_idx(100), 1);
        assert_eq!(quality_to_base_q_idx(0), 255);
        // clamped above 100
        assert_eq!(quality_to_base_q_idx(200), 1);
        // monotonic non-increasing in quality, and always in the lossy range
        let mut prev = 255u8;
        for qual in 0..=100u8 {
            let idx = quality_to_base_q_idx(qual);
            assert!((1..=255).contains(&idx));
            assert!(idx <= prev, "non-monotonic at quality {}", qual);
            prev = idx;
        }
        // perceptual spread: higher quality is finer (smaller ac_q step)
        assert!(ac_q_8bit(quality_to_base_q_idx(90)) < ac_q_8bit(quality_to_base_q_idx(50)));
        assert!(ac_q_8bit(quality_to_base_q_idx(50)) < ac_q_8bit(quality_to_base_q_idx(10)));
        // the step ratio between adjacent decades is roughly constant (geometric);
        // check it stays within a sane band rather than collapsing at either end
        let step = |qual: u8| ac_q_8bit(quality_to_base_q_idx(qual)) as f64;
        for d in [(80u8, 60u8), (60, 40), (40, 20)] {
            let ratio = step(d.1) / step(d.0); // coarser / finer > 1
            assert!(
                ratio > 1.5 && ratio < 6.0,
                "decade ratio {:.2} out of band",
                ratio
            );
        }
    }
}
