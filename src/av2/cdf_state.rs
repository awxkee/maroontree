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

use crate::av2::cdf_para;
use crate::av2::cdfs_qctx::*;
use crate::av2::cdfs_uv_qcx as uvq;
use crate::av2::cdfx_4tx::*;
use crate::av2::cfl;
use crate::av2::tables_tx32::TX_PART_2D_64;
use hashbrown::HashMap;

/// Expand a static ICDF slice to include counter + PARA rate words.
/// `nsyms` = alphabet size = number of entries before the sentinel.
pub(crate) fn expand(src: &[u16], nsyms_mt: usize) -> Vec<u16> {
    let has_sentinel = nsyms_mt > 0 && src[nsyms_mt - 1] == 0;
    let nsyms_avm = if has_sentinel { nsyms_mt } else { nsyms_mt + 1 };
    let (r0, r1, r2) = if nsyms_avm <= 3 {
        (2u16, 3, 4)
    } else {
        (3u16, 4, 5)
    };
    let mut v = Vec::with_capacity(nsyms_avm + 4);
    v.extend_from_slice(&src[..nsyms_mt]); // ICDF values (incl. sentinel if present)
    if !has_sentinel {
        v.push(0); // append sentinel -> index [nsyms_mt] = [nsyms_avm - 1]
    }
    v.push(0); // counter   -> index [nsyms_avm]
    v.push(r0); // PARA ti=0  -> index [nsyms_avm + 1]
    v.push(r1); // PARA ti=1
    v.push(r2); // PARA ti=2
    v
}

pub(crate) fn expand_para(src: &[u16], nsyms_mt: usize, para: (u8, u8, u8)) -> Vec<u16> {
    let has_sentinel = nsyms_mt > 0 && src[nsyms_mt - 1] == 0;
    let nsyms_avm = if has_sentinel { nsyms_mt } else { nsyms_mt + 1 };
    let mut v = Vec::with_capacity(nsyms_avm + 4);
    v.extend_from_slice(&src[..nsyms_mt]);
    if !has_sentinel {
        v.push(0); // sentinel
    }
    v.push(0); // counter
    v.push(para.0 as u16); // PARA ti=0  (exact AVM value)
    v.push(para.1 as u16); // PARA ti=1
    v.push(para.2 as u16); // PARA ti=2
    v
}

/// Build a 2-symbol adaptive working copy from a maroontree bool probability
/// `icdf0` (the ICDF of symbol 0) and the matching AVM PARA tuple.
/// Layout CDF_SIZE(2)=6: [icdf0, sentinel=0, counter, p0, p1, p2].
pub(crate) fn bool_wc(icdf0: u16, para: (u8, u8, u8)) -> Vec<u16> {
    vec![icdf0, 0, 0, para.0 as u16, para.1 as u16, para.2 as u16]
}

/// Expand a 1-D table of ICDF entries, each `W` u16s wide.
fn e1d<const W: usize>(src: &[[u16; W]]) -> Vec<Vec<u16>> {
    // W is nsyms_mt (the array holds exactly W ICDF values, no sentinel).
    src.iter().map(|r| expand(r, W)).collect()
}

/// Expand a 1-D token table applying the matching per-context AVM PARA words.
/// `para_qc` is the PARA row for the current q-context (one tuple per context).
fn e1d_para<const W: usize>(src: &[[u16; W]], para_qc: &[(u8, u8, u8)]) -> Vec<Vec<u16>> {
    src.iter()
        .enumerate()
        .map(|(i, r)| expand_para(r, W, para_qc[i]))
        .collect()
}

/// `update_cdf` — exact port of AVM prob.h update_cdf().
///
/// `cdf[nsyms]` is the adaptation counter; `cdf[nsyms+1..nsyms+3]` are the
/// three PARA rate-offset words.
#[inline(always)]
pub(crate) fn update_cdf(cdf: &mut [u16], val: usize, nsyms: usize) {
    let count = cdf[nsyms] as usize;
    let ti = if count > 31 {
        2
    } else if count > 15 {
        1
    } else {
        0
    };
    let rate = 2 + cdf[nsyms + 1 + ti] as usize;
    let mut tmp: u32 = 32768; // CDF_PROB_TOP = AVM_ICDF(0)
    for (i, cdf) in cdf[..nsyms - 1].iter_mut().enumerate() {
        if i == val {
            tmp = 0;
        }
        let ci = *cdf as u32;
        *cdf = if tmp < ci {
            (ci - ((ci - tmp) >> rate)) as u16
        } else {
            (ci + ((tmp - ci) >> rate)) as u16
        };
    }
    if count < 32 {
        cdf[nsyms] += 1;
    }
}

pub(crate) static Y_SET_INIT: [u16; 3] = [3905, 1746, 1044];
pub(crate) static Y_IDX0_INIT: [[u16; 7]; 3] = [
    [17593, 12693, 11040, 8670, 6363, 5113, 3908],
    [22654, 17811, 15953, 13641, 12621, 7185, 5599],
    [27132, 23764, 22312, 20646, 20024, 12443, 7161],
];
pub(crate) static Y_IDX1_INIT: [[u16; 5]; 3] = [
    [20025, 14596, 12574, 9120, 6349],
    [23792, 16684, 11941, 8173, 4272],
    [23984, 18212, 13058, 7865, 4044],
];
pub(crate) static UV_MODE_INIT: [[u16; 7]; 2] = [
    [23405, 11811, 9903, 8015, 6357, 4785, 2340],
    [11486, 9158, 4560, 3457, 2420, 1610, 1277],
];
pub(crate) static DELTA_Q_INIT: [u16; 7] = [16174, 9443, 6344, 4543, 3410, 2669, 2155];
pub(crate) static TX_PART_64X32_INIT: [u16; 6] = [28067, 19266, 7810, 6355, 4602, 2639];
pub(crate) static TX_PART_32X64_INIT: [u16; 6] = [30413, 15167, 11065, 6718, 4887, 1371];
pub(crate) static TX_PART_64X16_INIT: [u16; 6] = [30452, 28054, 10037, 8971, 3254, 2691];
pub(crate) static TX_PART_16X64_INIT: [u16; 6] = [31215, 16608, 14089, 4785, 3176, 629];
pub(crate) static INTRA_EXT_TX16_INIT: [u16; 6] = [19009, 6660, 5080, 2975, 2503, 1192];
// AVM default_intra_ext_tx_cdf[INTRA_TX_SET1][TX_8X8]: AVM_CDF7(14910,25257,26964,
// 29323,30237,31535) -> icdf, with AVM_PARA7(0,0,0) -> stored (3,4,5). This MUST be an
// adaptive slot (like intra_ext_tx16): the decoder's intra_ext_tx_cdf[set1][TX_8X8]
// adapts per block, so a static encoder cdf desyncs after the first 8x8 leaf.
pub(crate) static INTRA_EXT_TX8_INIT: [u16; 6] = [17858, 7511, 5804, 3445, 2531, 1233];
pub(crate) static TX_SHORT_SIDE_INIT: [[u16; 3]; 2] = [
    [6068, 608, 20], // ctx 0: intra_ext_tx_short_side_cdf index 1 (8x32 / 32x8 leaves)
    [5853, 357, 20], // ctx 1: index 2 (TX_16X16 / 16x32 / 32x16 leaves)
];

/// Every field is a mutable working copy of the corresponding static table,
/// extended with [counter, para_r0, para_r1, para_r2] words.
#[allow(non_snake_case)]
pub(crate) struct CdfState {
    pub(crate) qc: usize,

    pub(crate) luma8_hf: Vec<Vec<u16>>, // [20] × (nsyms=2, words=3+4)
    pub(crate) luma8_lf: Vec<Vec<u16>>, // [33] × (nsyms=4, words=5+4)
    pub(crate) luma8_eob_hf: Vec<Vec<u16>>, // [4]  × (nsyms=1, words=2+4)
    pub(crate) luma8_eob_lf: Vec<Vec<u16>>, // [4]  × (nsyms=3, words=4+4)
    pub(crate) luma16_hf: Vec<Vec<u16>>,
    pub(crate) luma16_lf: Vec<Vec<u16>>,
    pub(crate) luma16_eob_hf: Vec<Vec<u16>>,
    pub(crate) luma16_eob_lf: Vec<Vec<u16>>,
    pub(crate) luma32_hf: Vec<Vec<u16>>,
    pub(crate) luma32_lf: Vec<Vec<u16>>,
    pub(crate) luma32_eob_hf: Vec<Vec<u16>>,
    pub(crate) luma32_eob_lf: Vec<Vec<u16>>,
    pub(crate) br_hf: Vec<Vec<u16>>, // [7]
    pub(crate) br: Vec<Vec<u16>>,    // [14]
    pub(crate) eob_bin: Vec<u16>,
    pub(crate) eob_bin_inter: Vec<u16>, // inter luma TX32 eob (pl_ctx=1)
    pub(crate) eob64_luma: Vec<u16>,
    pub(crate) eob128_luma: Vec<u16>,
    pub(crate) eob256: Vec<u16>,
    pub(crate) eob256_inter: Vec<u16>,
    pub(crate) eob512: Vec<u16>,
    pub(crate) chr_eob_bin: Vec<u16>,
    pub(crate) chr_eob32: Vec<u16>,
    pub(crate) chr_eob64: Vec<u16>,
    pub(crate) chr_eob128: Vec<u16>,
    pub(crate) chr_eob256: Vec<u16>,
    pub(crate) chr_eob512: Vec<u16>,
    pub(crate) eob16_q0: Vec<Vec<u16>>,         // [3] × (nsyms=3)
    pub(crate) base_lf_tx4: Vec<Vec<Vec<u16>>>, // [33][2] × (nsyms=4)
    pub(crate) base_tx4: Vec<Vec<Vec<u16>>>,    // [20][2] × (nsyms=2)
    pub(crate) base_lf_eob_tx4: Vec<Vec<u16>>,  // [4] × (nsyms=3)
    pub(crate) base_eob_tx4: Vec<Vec<u16>>,     // [4] × (nsyms=1)
    pub(crate) br_lf_q0: Vec<Vec<u16>>,         // [14] × (nsyms=2)
    pub(crate) br_q0: Vec<Vec<u16>>,            // [7]  × (nsyms=2)
    pub(crate) base_lf_uv: Vec<Vec<u16>>,       // [12] × (nsyms=4)
    pub(crate) base_uv: Vec<Vec<u16>>,          // [12] × (nsyms=2)
    pub(crate) br_uv: Vec<Vec<u16>>,            // [4]  × (nsyms=2)
    pub(crate) base_lf_eob_uv: Vec<Vec<u16>>,   // [4]  × (nsyms=3)
    pub(crate) base_eob_uv: Vec<Vec<u16>>,      // [4]  × (nsyms=1)
    pub(crate) y_set: Vec<u16>,                 // nsyms=2
    pub(crate) y_idx0: Vec<Vec<u16>>,           // [3] × nsyms=6
    pub(crate) y_idx1: Vec<Vec<u16>>,           // [3] × nsyms=4
    pub(crate) uv_mode: Vec<Vec<u16>>,          // [2] × nsyms=6
    pub(crate) delta_q: Vec<u16>,               // nsyms=6
    pub(crate) tx_part_64: Vec<u16>,            // nsyms=5
    pub(crate) tx_part_64x32: Vec<u16>,
    pub(crate) tx_part_32x64: Vec<u16>,
    pub(crate) tx_part_64x16: Vec<u16>,
    pub(crate) tx_part_16x64: Vec<u16>,
    pub(crate) tx_short_side: Vec<Vec<u16>>, // [2 ctx] nsyms=2
    pub(crate) cfl_sign: Vec<u16>,           // nsyms=7
    pub(crate) cfl_alpha: Vec<Vec<u16>>,     // [6] × nsyms=7
    pub(crate) intra_ext_tx16: Vec<u16>,     // nsyms=5
    pub(crate) intra_ext_tx8: Vec<u16>,      // nsyms=6 (TX_8X8 intra ext-tx)
    pub(crate) txb_skip: Vec<Vec<u16>>,      // [10] TX32 skip (avm txb_skip tx3)
    pub(crate) txb_skip_inter: Vec<Vec<u16>>, // [10] TX32 inter skip (pred_mode_ctx=1)
    /// inter tx_type set 3 (flat 2-way) per eob_ctx, TX32 (avm inter_ext_tx_cdf[3][*][3]).
    pub(crate) inter_ext_tx3: Vec<Vec<u16>>,
    /// TX16 inter transform selection: set selector and the first 8-way bank.
    pub(crate) inter_tx16_set: Vec<Vec<u16>>,
    pub(crate) inter_tx16_idx: Vec<Vec<u16>>,
    pub(crate) skip_tx16: Vec<Vec<u16>>, // [10] TX16 skip (avm txb_skip tx2)
    pub(crate) skip_tx16_inter: Vec<Vec<u16>>, // [10] TX16 inter skip (pred_mode_ctx=1)
    pub(crate) skip_tx8: Vec<Vec<u16>>,  // [10] TX8 skip (avm txb_skip tx1)
    pub(crate) skip_tx8_inter: Vec<Vec<u16>>,
    pub(crate) skip_tx4: Vec<Vec<u16>>, // [10] TX4 skip (avm txb_skip tx0)
    pub(crate) cfl_is: Vec<Vec<u16>>,   // [3] cfl_is/use flag (avm cfl_cdf)
    pub(crate) cfl_index: Vec<u16>,     // cfl_index flag (avm cfl_index_cdf)
    pub(crate) cfl_mhccp: Vec<u16>,     // cfl_mhccp_switch flag (avm cfl_mhccp_switch_cdf)
    pub(crate) filter_dir: Vec<Vec<u16>>, // [4 ctx] mh_dir (avm filter_dir_cdf), nsyms=3
    pub(crate) skip_tx64: Vec<Vec<u16>>, // [10] TX64 skip (avm txb_skip tx4)
    pub(crate) skip_tx64_inter: Vec<Vec<u16>>, // [10] TX64 inter skip (pred_mode_ctx=1)
    pub(crate) skip_v: Vec<Vec<u16>>, // [12] V skip (avm v_txb_skip — shared by all V blocks incl 4x4)
    pub(crate) dc_sign: Vec<Vec<u16>>, // [3 dc_sign_ctx]
    pub(crate) eob_extra: Vec<u16>,   // [1]
    pub(crate) do_split: Vec<Vec<u16>>, // [64 partition_ctx]
    pub(crate) do_square_split: Vec<Vec<u16>>, // [8 square_split_ctx]
    pub(crate) rect_type: Vec<Vec<u16>>, // [64 partition_ctx]
    pub(crate) txfm_bools: HashMap<u16, Vec<u16>>,
    /// CCSO per-superblock on/off flag, adaptive 2-symbol CDF. Indexed
    /// [plane][ctx] with CCSO_PLANES=3, CCSO_CONTEXT=4. Phase 1 only uses
    /// plane 1 (U), but all three planes are initialized to the AVM defaults.
    pub(crate) ccso: Vec<Vec<Vec<u16>>>,
    /// Per-CDEF-unit "strength is index 0" flag, adaptive 2-symbol CDF, indexed
    /// [ctx] with CDEF_STRENGTH_INDEX0_CTX=4 (AVM default_cdef_strength_index0_cdf).
    /// Only used when nb_cdef_strengths > 1 (per-block CDEF); s=1 => index 0 (off).
    pub(crate) cdef: Vec<Vec<u16>>,
    /// intra_inter flag [4 ctx] (avm intra_inter_cdf). s=0 => intra.
    pub(crate) intra_inter: Vec<Vec<u16>>,
    /// single-ref inter mode [5 ctx] (avm inter_single_mode_cdf, 3 syms).
    pub(crate) inter_single_mode: Vec<Vec<u16>>,
    pub(crate) drl: Vec<Vec<u16>>, // drl_cdf[0][ctx] (idx0)
    /// single-ref rank bit [3 ctx] (avm single_ref_cdf[ctx][0]); coded per inter
    /// block only when the frame lists two references. bit=1 selects rank 0.
    pub(crate) single_ref: Vec<Vec<u16>>,
    // Adaptive MVD (QTR_PEL) CDFs — mirror AVM nmv_context so encoder/decoder adapt
    // in lockstep (static CDFs desync after the first motion vector in a frame).
    pub(crate) mvd_shell_set: Vec<u16>, // joint_shell_set_cdf [2]
    pub(crate) mvd_shell_class0: Vec<u16>, // joint_shell_class_cdf_0[QTR] [8]
    pub(crate) mvd_shell_class1: Vec<u16>, // joint_shell_class_cdf_1[QTR] [8]
    pub(crate) mvd_offset_low: Vec<Vec<u16>>, // shell_offset_low_class_cdf [2][2]
    pub(crate) mvd_offset_class2: Vec<u16>, // shell_offset_class2_cdf [2]
    pub(crate) mvd_offset_other: Vec<Vec<u16>>, // shell_offset_other_class_cdf[0][16][2]
    pub(crate) mvd_col_greater: Vec<Vec<u16>>, // col_mv_greater_flags_cdf [2][2]
    pub(crate) mvd_col_index: Vec<Vec<u16>>, // col_mv_index_cdf [4][2]
    /// block-level skip_txfm [6 ctx] (avm skip_txfm_cdfs); inter blocks only.
    pub(crate) skip_txfm_blk: Vec<Vec<u16>>,
    /// inter 64x64 tx do_partition [ctx 0][1][7] (avm txfm_do_partition_cdf).
    pub(crate) tx_do_partition: Vec<u16>,
    /// inter 64x64 4-way tx partition type [0][1][9], 7 syms (SPLIT=0).
    pub(crate) tx_part_type: Vec<u16>,
}

impl CdfState {
    pub(crate) fn new(qc: usize) -> Self {
        CdfState {
            qc,
            luma8_hf: e1d_para(
                &LUMA8_BASE_TOK_HF_QC[qc],
                &cdf_para::PARA_LUMA8_BASE_TOK_HF[qc],
            ),
            luma8_lf: e1d_para(
                &LUMA8_BASE_TOK_LF_QC[qc],
                &cdf_para::PARA_LUMA8_BASE_TOK_LF[qc],
            ),
            luma8_eob_hf: e1d_para(
                &LUMA8_EOB_TOK_HF_QC[qc],
                &cdf_para::PARA_LUMA8_EOB_TOK_HF[qc],
            ),
            luma8_eob_lf: e1d_para(
                &LUMA8_EOB_TOK_LF_QC[qc],
                &cdf_para::PARA_LUMA8_EOB_TOK_LF[qc],
            ),
            luma16_hf: e1d_para(
                &LUMA16_BASE_TOK_HF_QC[qc],
                &cdf_para::PARA_LUMA16_BASE_TOK_HF[qc],
            ),
            luma16_lf: e1d_para(
                &LUMA16_BASE_TOK_LF_QC[qc],
                &cdf_para::PARA_LUMA16_BASE_TOK_LF[qc],
            ),
            luma16_eob_hf: e1d_para(
                &LUMA16_EOB_TOK_HF_QC[qc],
                &cdf_para::PARA_LUMA16_EOB_TOK_HF[qc],
            ),
            luma16_eob_lf: e1d_para(
                &LUMA16_EOB_TOK_LF_QC[qc],
                &cdf_para::PARA_LUMA16_EOB_TOK_LF[qc],
            ),
            luma32_hf: e1d_para(
                &LUMA32_BASE_TOK_HF_QC[qc],
                &cdf_para::PARA_LUMA32_BASE_TOK_HF[qc],
            ),
            luma32_lf: e1d_para(
                &LUMA32_BASE_TOK_LF_QC[qc],
                &cdf_para::PARA_LUMA32_BASE_TOK_LF[qc],
            ),
            luma32_eob_hf: e1d_para(
                &LUMA32_EOB_TOK_HF_QC[qc],
                &cdf_para::PARA_LUMA32_EOB_TOK_HF[qc],
            ),
            luma32_eob_lf: e1d_para(
                &LUMA32_EOB_TOK_LF_QC[qc],
                &cdf_para::PARA_LUMA32_EOB_TOK_LF[qc],
            ),
            br_hf: e1d_para(&BR_TOK_HF_QC[qc], &cdf_para::PARA_BR_TOK_HF[qc]),
            br: e1d_para(&BR_TOK_QC[qc], &cdf_para::PARA_BR_TOK[qc]),
            eob_bin: expand_para(&EOB_BIN_QC[qc], 7, cdf_para::PARA_EOB_BIN[qc]),
            eob_bin_inter: expand_para(&EOB_BIN_INTER_QC[qc], 7, cdf_para::PARA_EOB_BIN_INTER[qc]),
            eob64_luma: expand_para(&EOB64_LUMA_QC[qc], 6, cdf_para::PARA_EOB64_LUMA[qc]),
            eob128_luma: expand_para(&EOB128_LUMA_QC[qc], 7, cdf_para::PARA_EOB128_LUMA[qc]),
            eob256: expand_para(&EOB256_QC[qc], 7, cdf_para::PARA_EOB256[qc]),
            eob256_inter: expand_para(&INTER_EOB256_QC[qc], 7, cdf_para::PARA_INTER_EOB256[qc]),
            eob512: expand_para(&EOB512_QC[qc], 7, cdf_para::PARA_EOB512[qc]),
            chr_eob_bin: expand_para(&CHROMA_EOB_BIN_QC[qc], 7, cdf_para::PARA_CHROMA_EOB_BIN[qc]),
            chr_eob32: expand_para(&CHROMA_EOB32_QC[qc], 5, cdf_para::PARA_CHROMA_EOB32[qc]),
            chr_eob64: expand_para(&CHROMA_EOB64_QC[qc], 6, cdf_para::PARA_CHROMA_EOB64[qc]),
            chr_eob128: expand_para(&CHROMA_EOB128_QC[qc], 7, cdf_para::PARA_CHROMA_EOB128[qc]),
            chr_eob256: expand_para(&CHROMA_EOB256_QC[qc], 7, cdf_para::PARA_CHROMA_EOB256[qc]),
            chr_eob512: expand_para(&CHROMA_EOB512_QC[qc], 7, cdf_para::PARA_CHROMA_EOB512[qc]),
            eob16_q0: e1d(&EOB16_Q0),
            base_lf_tx4: BASE_LF_TX4_Q0
                .iter()
                .map(|outer| outer.iter().map(|r| expand(r, 5)).collect())
                .collect(),
            base_tx4: BASE_TX4_Q0
                .iter()
                .map(|outer| outer.iter().map(|r| expand(r, 3)).collect())
                .collect(),
            base_lf_eob_tx4: e1d(&BASE_LF_EOB_TX4_Q0),
            base_eob_tx4: e1d(&BASE_EOB_TX4_Q0),
            br_lf_q0: e1d(&BR_LF_Q0),
            br_q0: e1d(&BR_Q0),
            base_lf_uv: e1d_para(&uvq::BASE_LF_UV_QCX[qc], &uvq::PARA_BASE_LF_UV_QCX[qc]),
            base_uv: e1d_para(&uvq::BASE_UV_QCX[qc], &uvq::PARA_BASE_UV_QCX[qc]),
            br_uv: e1d_para(&uvq::BR_UV_QCX[qc], &uvq::PARA_BR_UV_QCX[qc]),
            base_lf_eob_uv: e1d_para(
                &uvq::BASE_LF_EOB_UV_QCX[qc],
                &uvq::PARA_BASE_LF_EOB_UV_QCX[qc],
            ),
            base_eob_uv: e1d_para(&uvq::BASE_EOB_UV_QCX[qc], &uvq::PARA_BASE_EOB_UV_QCX[qc]),
            y_set: expand_para(&Y_SET_INIT, 3, cdf_para::PARA_Y_SET),
            y_idx0: e1d_para(&Y_IDX0_INIT, &cdf_para::PARA_Y_IDX0),
            y_idx1: e1d_para(&Y_IDX1_INIT, &cdf_para::PARA_Y_IDX1),
            uv_mode: e1d_para(&UV_MODE_INIT, &cdf_para::PARA_UV_MODE),
            delta_q: expand_para(&DELTA_Q_INIT, 7, cdf_para::PARA_DELTA_Q),
            tx_part_64: expand_para(&TX_PART_2D_64, 6, (2, 3, 3)),
            tx_part_64x32: expand_para(&TX_PART_64X32_INIT, 6, (2, 3, 3)),
            tx_part_32x64: expand_para(&TX_PART_32X64_INIT, 6, (2, 3, 3)),
            tx_part_64x16: expand_para(&TX_PART_64X16_INIT, 6, (2, 2, 3)),
            tx_part_16x64: expand_para(&TX_PART_16X64_INIT, 6, (2, 3, 3)),
            tx_short_side: (0..2)
                .map(|c| expand_para(&TX_SHORT_SIDE_INIT[c], 3, cdf_para::PARA_TX_SHORT_SIDE))
                .collect(),
            cfl_sign: expand_para(&cfl::CFL_SIGN_ICDF, 8, (1, 2, 3)),
            cfl_alpha: cfl::CFL_ALPHA_ICDF
                .iter()
                .map(|r| expand_para(r, 8, (1, 2, 3)))
                .collect(),
            intra_ext_tx16: expand_para(&INTRA_EXT_TX16_INIT, 6, (2, 2, 5)),
            intra_ext_tx8: expand_para(&INTRA_EXT_TX8_INIT, 6, (3, 4, 5)),
            // Build 2-symbol working copies from maroontree's static skip/sign CDF
            // values (which equal AVM defaults) plus the matching AVM PARA words.
            txb_skip: (0..10)
                .map(|c| bool_wc(CHROMA_SKIP_TX32_QC[qc][c], cdf_para::PARA_SKIP_TX32[qc][c]))
                .collect(),
            txb_skip_inter: (0..10)
                .map(|c| {
                    bool_wc(
                        INTER_SKIP_TX32_QC[qc][c],
                        cdf_para::PARA_INTER_SKIP_TX32[qc][c],
                    )
                })
                .collect(),
            // inter_ext_tx_cdf[3][eob_ctx][TX32]=CDF2(16384) PARA2(0,0,0) flat 2-way.
            inter_ext_tx3: (0..3).map(|_| bool_wc(16384, (2, 3, 4))).collect(),
            // eset=2 (EXT_TX_SET_DTT9_IDTX_1DDCT), square TX16. DCT_DCT is
            // symbol 3 in the first 8-way bank, selected by tx_set=0.
            inter_tx16_set: [11933, 2048, 3911]
                .into_iter()
                .zip([(2, 3, 4), (2, 3, 5), (2, 4, 2)])
                .map(|(cdf, para)| bool_wc(cdf, para))
                .collect(),
            inter_tx16_idx: [
                [31628, 31043, 30444, 18115, 13696, 9150, 4659],
                [32710, 32507, 32181, 451, 212, 142, 60],
                [15364, 15099, 14365, 8716, 6375, 4262, 2092],
            ]
            .iter()
            .zip([(2, 3, 4), (3, 4, 3), (1, 3, 3)])
            .map(|(cdf, para)| expand_para(cdf, 7, para))
            .collect(),
            skip_tx16: (0..10)
                .map(|c| bool_wc(SKIP_TX16_QC[qc][c], cdf_para::PARA_SKIP_TX16[qc][c]))
                .collect(),
            skip_tx16_inter: (0..10)
                .map(|c| {
                    bool_wc(
                        INTER_SKIP_TX16_QC[qc][c],
                        cdf_para::PARA_INTER_SKIP_TX16[qc][c],
                    )
                })
                .collect(),
            skip_tx8: (0..10)
                .map(|c| bool_wc(SKIP_TX8_QC[qc][c], cdf_para::PARA_SKIP_TX8[qc][c]))
                .collect(),
            skip_tx8_inter: (0..10)
                .map(|c| {
                    bool_wc(
                        INTER_SKIP_TX8_QC[qc][c],
                        cdf_para::PARA_INTER_SKIP_TX8[qc][c],
                    )
                })
                .collect(),
            skip_tx4: (0..10)
                .map(|c| bool_wc(SKIP_TX4_QC[qc][c], cdf_para::PARA_SKIP_TX4[qc][c]))
                .collect(),
            cfl_is: (0..3)
                .map(|c| bool_wc(cfl::CFL_IS_CDF[c], cdf_para::PARA_CFL_IS[c]))
                .collect(),
            cfl_index: bool_wc(cfl::CFL_INDEX_CDF, cdf_para::PARA_CFL_INDEX),
            cfl_mhccp: bool_wc(cfl::CFL_MHCCP_SWITCH_CDF, cdf_para::PARA_CFL_INDEX),
            filter_dir: (0..4)
                .map(|c| {
                    expand_para(
                        &cfl::FILTER_DIR_CDF[c],
                        3,
                        (
                            cfl::FILTER_DIR_PARA[c].0,
                            cfl::FILTER_DIR_PARA[c].1,
                            cfl::FILTER_DIR_PARA[c].2,
                        ),
                    )
                })
                .collect(),
            skip_tx64: (0..10)
                .map(|c| bool_wc(CHROMA_SKIP_TX64_QC[qc][c], cdf_para::PARA_SKIP_TX64[qc][c]))
                .collect(),
            skip_tx64_inter: (0..10)
                .map(|c| {
                    bool_wc(
                        INTER_SKIP_TX64_QC[qc][c],
                        cdf_para::PARA_INTER_SKIP_TX64[qc][c],
                    )
                })
                .collect(),
            skip_v: (0..12)
                .map(|c| bool_wc(V_SKIP_TX4_QC[qc][c], cdf_para::PARA_V_SKIP_TX4[qc][c]))
                .collect(),
            dc_sign: (0..3)
                .map(|c| bool_wc(DC_SIGN_QC[qc][c], cdf_para::PARA_DC_SIGN[qc][c]))
                .collect(),
            eob_extra: bool_wc(EOB_HI_BIT_QC[qc], cdf_para::PARA_EOB_EXTRA[qc]),
            do_split: (0..64)
                .map(|c| {
                    bool_wc(
                        crate::av2::partition::DO_SPLIT_CDF0[c] as u16,
                        cdf_para::PARA_DO_SPLIT[c],
                    )
                })
                .collect(),
            do_square_split: (0..8)
                .map(|c| {
                    bool_wc(
                        crate::av2::partition::DO_SQUARE_SPLIT_CDF0[c] as u16,
                        cdf_para::PARA_DO_SQUARE_SPLIT[c],
                    )
                })
                .collect(),
            rect_type: (0..64)
                .map(|c| {
                    bool_wc(
                        crate::av2::partition::RECT_TYPE_CDF0[c] as u16,
                        cdf_para::PARA_RECT_TYPE[c],
                    )
                })
                .collect(),
            txfm_bools: HashMap::new(),
            // CCSO blk on/off flag CDFs. AVM default_ccso_cdf[plane][ctx]:
            // CDF2 value a0 -> icdf0 = 32768 - a0; PARA2(a,b,c) = (a+2,b+3,c+4).
            ccso: {
                // [plane][ctx] = (a0, (para0, para1, para2))
                #[allow(clippy::type_complexity)]
                let defs: [[(u16, (u8, u8, u8)); 4]; 3] = [
                    [
                        (18469, (0, 1, 2)),
                        (16384, (2, 3, 4)),
                        (4949, (1, 1, 2)),
                        (16384, (2, 3, 4)),
                    ],
                    [
                        (23470, (1, 1, 2)),
                        (16384, (2, 3, 4)),
                        (6666, (1, 1, 2)),
                        (16384, (2, 3, 4)),
                    ],
                    [
                        (22914, (1, 1, 2)),
                        (16384, (2, 3, 4)),
                        (6993, (1, 1, 2)),
                        (16384, (2, 3, 4)),
                    ],
                ];
                defs.iter()
                    .map(|plane| {
                        plane
                            .iter()
                            .map(|&(a0, para)| bool_wc(32768 - a0, para))
                            .collect()
                    })
                    .collect()
            },
            // CDEF per-unit index0 flag CDFs. AVM default_cdef_strength_index0_cdf[ctx]:
            // CDF2(a0), PARA2(a,b,c) = (a+2,b+3,c+4).
            cdef: {
                let defs: [(u16, (u8, u8, u8)); 4] = [
                    (29034, (1, 2, 2)),
                    (16472, (1, 2, 2)),
                    (5751, (1, 2, 2)),
                    (3115, (1, 2, 3)),
                ];
                defs.iter().map(|&(a0, p)| bool_wc(32768 - a0, p)).collect()
            },
            intra_inter: {
                // avm default_intra_inter_cdf; para = AVM_PARA2(a,b,c)=(a+2,b+3,c+4).
                let defs: [(u16, (u8, u8, u8)); 4] = [
                    (1522, (2, 3, 3)),
                    (14381, (2, 3, 4)),
                    (10455, (1, 3, 4)),
                    (27796, (2, 3, 4)),
                ];
                defs.iter().map(|&(a0, p)| bool_wc(32768 - a0, p)).collect()
            },
            inter_single_mode: {
                // avm default_inter_single_mode_cdf; AVM_CDF3(a,b) icdf=[32768-a,32768-b];
                // para = AVM_PARA3(x,y,z)=(x+2,y+3,z+4).
                let defs: [([u16; 2], (u8, u8, u8)); 5] = [
                    ([10043, 11100], (2, 2, 3)),
                    ([21561, 21758], (2, 3, 3)),
                    ([25411, 25714], (2, 3, 4)),
                    ([14117, 14341], (2, 3, 4)),
                    ([18288, 18577], (2, 3, 4)),
                ];
                defs.iter()
                    .map(|&(ab, p)| expand_para(&[32768 - ab[0], 32768 - ab[1]], 2, p))
                    .collect()
            },
            drl: {
                // avm default_drl_cdf[0][ctx] (idx0). icdf + AVM_PARA2 offset.
                let defs: [(u16, (u8, u8, u8)); 5] = [
                    (17047, (3, 4, 4)),
                    (11653, (2, 3, 4)),
                    (13201, (2, 3, 3)),
                    (15166, (3, 4, 5)),
                    (19449, (3, 4, 5)),
                ];
                defs.iter().map(|&(a, p)| bool_wc(a, p)).collect()
            },
            single_ref: {
                // avm default_single_ref_cdf[ctx][0]; para = AVM_PARA2(a,b,c)=(a+2,b+3,c+4).
                let defs: [(u16, (u8, u8, u8)); 3] =
                    [(26469, (2, 3, 4)), (13631, (2, 2, 3)), (2599, (2, 3, 4))];
                defs.iter().map(|&(a0, p)| bool_wc(32768 - a0, p)).collect()
            },
            // AVM default_nmv_context (QTR_PEL). icdf0 = 32768 - AVM_CDF arg;
            // PARA stored AVM_PARA-encoded (arg + base offset), always >= 0.
            mvd_shell_set: bool_wc(1189, (1, 3, 4)),
            mvd_shell_class0: expand_para(
                &[25271, 15467, 8920, 5330, 3373, 1889, 765],
                7,
                (2, 3, 4),
            ),
            mvd_shell_class1: expand_para(&[14299, 5341, 1206, 116, 44, 40, 36], 7, (2, 3, 4)),
            mvd_offset_low: vec![bool_wc(18181, (1, 1, 3)), bool_wc(11802, (3, 3, 4))],
            mvd_offset_class2: bool_wc(19579, (2, 3, 4)),
            mvd_offset_other: {
                let defs: [(u16, (u8, u8, u8)); 16] = [
                    (14825, (3, 4, 5)),
                    (13834, (3, 4, 5)),
                    (13840, (3, 4, 5)),
                    (14072, (3, 4, 5)),
                    (13724, (3, 4, 5)),
                    (12406, (3, 4, 5)),
                    (12342, (3, 4, 5)),
                    (10205, (3, 4, 5)),
                    (10578, (3, 4, 5)),
                    (9310, (3, 4, 4)),
                    (6541, (2, 3, 2)),
                    (2003, (0, 3, 4)),
                    (16384, (2, 3, 4)),
                    (16384, (2, 3, 4)),
                    (16384, (2, 3, 4)),
                    (16384, (2, 3, 4)),
                ];
                defs.iter().map(|&(a, p)| bool_wc(a, p)).collect()
            },
            mvd_col_greater: vec![bool_wc(27105, (1, 3, 4)), bool_wc(27912, (3, 4, 4))],
            mvd_col_index: {
                let defs: [(u16, (u8, u8, u8)); 4] = [
                    (19323, (2, 3, 3)),
                    (19227, (2, 3, 3)),
                    (18723, (2, 3, 3)),
                    (19880, (1, 2, 3)),
                ];
                defs.iter().map(|&(a, p)| bool_wc(a, p)).collect()
            },
            skip_txfm_blk: {
                // avm default_skip_txfm_cdfs; AVM_CDF2(a) icdf0=32768-a, AVM_PARA2 offset.
                let defs: [(u16, (u8, u8, u8)); 6] = [
                    (25865, (1, 3, 4)),
                    (14316, (2, 3, 4)),
                    (4598, (2, 3, 4)),
                    (25612, (2, 2, 3)),
                    (12366, (2, 3, 3)),
                    (3320, (3, 4, 4)),
                ];
                defs.iter().map(|&(a0, p)| bool_wc(32768 - a0, p)).collect()
            },
            // do_partition[0][1][7]=CDF2(27351) PARA2(0,0,-1); 4way[0][1][9]=
            // CDF7(6404,..,28421) PARA7(-2,-2,-2). AVM_PARA offset (a+2,b+3,c+4).
            tx_do_partition: bool_wc(32768 - 27351, (2, 3, 3)),
            tx_part_type: expand_para(
                &[
                    32768 - 6404,
                    32768 - 12066,
                    32768 - 16173,
                    32768 - 20041,
                    32768 - 24512,
                    32768 - 28421,
                ],
                6,
                (1, 2, 3),
            ),
        }
    }

    pub(crate) fn skip_slot_of(&self, val: u16) -> (u8, usize) {
        if let Some(i) = CHROMA_SKIP_TX32_QC[self.qc].iter().position(|&v| v == val) {
            return (0, i);
        }
        if let Some(i) = CHROMA_SKIP_TX64_QC[self.qc].iter().position(|&v| v == val) {
            return (1, i);
        }
        if let Some(i) = V_SKIP_TX4_QC[self.qc].iter().position(|&v| v == val) {
            return (2, i);
        }
        // Luma TX_16X16 skip (avm txb_skip txs_ctx=2) — distinct adaptation slot from
        // the TX_32X32 table reused for `txb_skip`. Checked last because its ctx0 seed
        // can otherwise be confused with a chroma value at some q-contexts.
        if let Some(i) = SKIP_TX16_QC[self.qc].iter().position(|&v| v == val) {
            return (3, i);
        }
        if let Some(i) = INTER_SKIP_TX16_QC[self.qc].iter().position(|&v| v == val) {
            return (6, i);
        }
        // TX_8X8 skip (avm txb_skip txs_ctx=1) — its own adaptation slot.
        if let Some(i) = SKIP_TX8_QC[self.qc].iter().position(|&v| v == val) {
            return (4, i);
        }
        (0, 0)
    }

    pub(crate) fn dc_sign_ctx_of(&self, val: u16) -> usize {
        DC_SIGN_QC[self.qc]
            .iter()
            .position(|&v| v == val)
            .unwrap_or(0)
    }

    pub(crate) fn do_split_ctx_of(&self, val: u16) -> usize {
        crate::av2::partition::DO_SPLIT_CDF0
            .iter()
            .position(|&v| v as u16 == val)
            .unwrap_or(0)
    }

    pub(crate) fn do_square_split_ctx_of(&self, val: u16) -> usize {
        crate::av2::partition::DO_SQUARE_SPLIT_CDF0
            .iter()
            .position(|&v| v as u16 == val)
            .unwrap_or(0)
    }

    pub(crate) fn txfm_bool(&mut self, val: u16) -> &mut Vec<u16> {
        self.txfm_bools.entry(val).or_insert_with(|| {
            let para = cdf_para::BOOL_PARA_LUT
                .binary_search_by_key(&val, |&(v, _)| v)
                .map(|i| cdf_para::BOOL_PARA_LUT[i].1)
                .unwrap_or((0, 2, 3));
            bool_wc(val, para)
        })
    }
}
