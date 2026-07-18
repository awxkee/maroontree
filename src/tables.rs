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
#![allow(clippy::all)]

pub(crate) fn icdf(args: &[u16]) -> Vec<u16> {
    let mut v: Vec<u16> = args.iter().map(|&a| 32768 - a).collect();
    v.push(0); // adaptation counter (also acts as the terminal, since 0>>6==0)
    v
}

pub(crate) static SKIP_CDF: [u16; 3] = [31671, 16515, 4576];

pub(crate) static FILTER_INTRA_MODE_CDF: [u16; 4] = [8949, 12776, 17211, 29558];

pub(crate) static FILTER_INTRA_CDF: [u16; 22] = [
    4621, 6743, 5893, 7866, 12551, 9394, 12408, 14301, 12756, 22343, 16384, 16384, 16384, 16384,
    16384, 16384, 12770, 10368, 20229, 18101, 16384, 16384,
];

pub(crate) const INTRA_FILTER_SCALE_BITS: u32 = 4;

/// AV1 section 9 filter-intra taps. The eighth coefficient is padding in the
/// reference table and is omitted here.
pub(crate) static INTRA_FILTER_TAPS: [[[i8; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// `default_kf_y_mode_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]` (libaom), the
/// keyframe luma intra-mode CDFs indexed by `[above_ctx][left_ctx]` (each ctx is
/// `INTRA_MODE_CTX[neighbor_mode]`). `[0][0]` equals the former single
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
pub(crate) static UV_MODE_CFL_CDF: [[u16; 13]; 13] = [
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
pub(crate) static TXTP_INTRA1_TX8: [[u16; 6]; 13] = [
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
pub(crate) static TXTP_INTRA2_TX16: [[u16; 4]; 13] = [
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

pub(crate) const NUM_BASE_LEVELS: i32 = 2;
pub(crate) const COEFF_BASE_RANGE: i32 = 12;

/// AV1 up-right diagonal scan for an 8x8 transform (`scan_8x8`).
pub(crate) static SCAN_16X16: [u32; 256] = crate::coef_q::SCAN_16X16;
pub(crate) static SCAN_32X32: [u32; 1024] = crate::coef_q::SCAN_32X32;
pub(crate) static SCAN_8X16: [u32; 128] = crate::coef_q::SCAN_8X16;
pub(crate) static SCAN_16X8: [u32; 128] = crate::coef_q::SCAN_16X8;
pub(crate) static SCAN_16X32: [u32; 512] = crate::coef_q::SCAN_16X32;
pub(crate) static SCAN_32X16: [u32; 512] = crate::coef_q::SCAN_32X16;
pub(crate) static SCAN_8X8: [u32; 64] = [
    0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34,
    41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31,
    38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];
/// `dav1d_lo_ctx_offsets[0]` (square, w==h) — position offset for coeff_base ctx.
pub(crate) static LO_CTX_OFF: [[u32; 5]; 5] = [
    [0, 1, 6, 6, 21],
    [1, 6, 6, 21, 21],
    [6, 6, 21, 21, 21],
    [6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21],
];

/// The byte dav1d stores in its `levels` map for a coefficient of magnitude `m`,
/// used by the neighbor-context model: `m*0x41` for m<=2, else `min(m,15)+0xC0`.
#[inline]
pub(crate) fn level_byte(m: u32) -> u8 {
    if m == 0 {
        0
    } else if m <= 2 {
        (m * 0x41) as u8
    } else {
        (m.min(15) + (3 << 6)) as u8
    }
}

pub(crate) const EOB_BITW: u32 = 10;

pub(crate) static PART_SPLIT_CDF: [[[u16; 9]; 4]; 3] = [
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
pub(crate) static PART_BL8_CDF: [[u16; 3]; 4] = [
    [19132, 25510, 30392],
    [13928, 19855, 28540],
    [12522, 23679, 28629],
    [9896, 18783, 25853],
];

// dav1d `txsz` defaults (raw CDF args; apply icdf() at load): the intra
// `tx_depth` symbol of spec `read_tx_size`, indexed [t_dim.max - 1][ctx].
// Category 0 (max TX 8x8-class: TX_8X8/RTX_8X4/RTX_4X8) is a 2-symbol CDF
// (depth 0 or 1); categories 1 (16x16-class) and 2 (32x32-class) are 3-symbol
// (depth 0, 1 or 2). ctx in 0..=2 from the above/left coded TX dims
// (dav1d env.h `get_tx_ctx`).
pub(crate) static TXSZ_CAT0_CDF: [[u16; 1]; 3] = [[19968], [19968], [24320]];
pub(crate) static TXSZ_CAT1_CDF: [[u16; 2]; 3] = [[12272, 30172], [12272, 30172], [18677, 30848]];
pub(crate) static TXSZ_CAT2_CDF: [[u16; 2]; 3] = [[12986, 15180], [12986, 15180], [24302, 25602]];
// Category 3 (max TX 64x64-class): 3-symbol CDF (depth 0/1/2 -> TX_64X64 /
// TX_32X32 / TX_16X16). libaom `default_tx_size_cdf[3]` (av1/common/token_cdfs
// family). Enables intra BLOCK_64X64 with a signaled tx_depth.
pub(crate) static TXSZ_CAT3_CDF: [[u16; 2]; 3] = [[5782, 11475], [5782, 11475], [16803, 22759]];

// dav1d 1.4.1 txtp_intra1[TX_4X4] (raw CDF6 args; apply icdf() at load).
// Verified: txtp_intra1[TX_8X8] sibling == existing TXTP_INTRA1_TX8.
pub(crate) static TXTP_INTRA1_TX4: [[u16; 6]; 13] = [
    [1535, 8035, 9461, 12751, 23467, 27825],
    [564, 3335, 9709, 10870, 18143, 28094],
    [672, 3247, 3676, 11982, 19415, 23127],
    [5279, 13885, 15487, 18044, 23527, 30252],
    [4423, 6074, 7985, 10416, 25693, 29298],
    [1486, 4241, 9460, 10662, 16456, 27694],
    [439, 2838, 3522, 6737, 18058, 23754],
    [1190, 4233, 4855, 11670, 20281, 24377],
    [1045, 4312, 8647, 10159, 18644, 29335],
    [202, 3734, 4747, 7298, 17127, 24016],
    [447, 4312, 6819, 8884, 16010, 23858],
    [277, 4369, 5255, 8905, 16465, 22271],
    [3409, 5436, 10599, 15599, 19687, 24040],
];

/// 4x4 scan order (dav1d `scan_4x4`): scan index -> raster rc = fx*4 + fy.
pub(crate) static SCAN_4X4: [u32; 16] = [0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15];

/// 4x8 scan order (dav1d `scan_4x8`): scan index -> raster rc = fx*8 + fy.
pub(crate) static SCAN_4X8: [u32; 32] = [
    0, 8, 1, 16, 9, 2, 24, 17, 10, 3, 25, 18, 11, 4, 26, 19, 12, 5, 27, 20, 13, 6, 28, 21, 14, 7,
    29, 22, 15, 30, 23, 31,
];
pub(crate) static SCAN_8X4: [u32; 32] = [
    0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 16, 11, 14, 17, 20, 15, 18, 21, 24, 19, 22, 25, 28,
    23, 26, 29, 27, 30, 31,
];

/// `dav1d_lo_ctx_offsets[2]` (w < h) — coeff_base position offset for RTX_4X8.
pub(crate) static LO_CTX_OFF_WLH: [[u32; 5]; 5] = [
    [0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [6, 6, 21, 21, 21],
    [6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21],
];

/// 2D coeff-base context offsets for non-square transforms with width > height
/// (e.g. RTX_16X8). dav1d `dav1d_lo_ctx_offsets[1]`, indexed `[y][x]`.
pub(crate) static LO_CTX_OFF_WGH: [[u32; 5]; 5] = [
    [0, 16, 6, 6, 21],
    [16, 16, 6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
];
