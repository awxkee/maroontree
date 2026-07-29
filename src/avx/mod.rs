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
mod dct;
mod idct;
mod intrapred;
mod kmeans;
mod loopfilter;
mod qmatrix;
mod rd;
mod trellis;
mod wht;

pub(crate) use dct::{
    adst4x4_avx2_quant_t, adst4x8_avx2_quant_t, adst8x8_avx2_quant_t, adst8x16_avx2_quant_t,
    adst16x8_avx2_quant_t, adst16x16_avx2_quant_t, adstdct4x4_avx2_quant_t,
    adstdct4x8_avx2_quant_t, adstdct8x8_avx2_quant_t, adstdct8x16_avx2_quant_t,
    adstdct16x8_avx2_quant_t, adstdct16x16_avx2_quant_t, dct4x4_avx2_quant_t, dct4x8_avx2_quant_t,
    dct4x16_avx2_quant_t, dct8x4_avx2_quant_t, dct8x8_avx2_quant_t, dct8x16_avx2_i32,
    dct8x16_avx2_quant_t, dct16x4_avx2_quant_t, dct16x8_avx2_quant_t, dct16x16_avx2_quant_t,
    dct16x32_avx2_quant_t, dct32x16_avx2_quant_t, dct32x32_avx2_i32, dct32x32_avx2_quant_t,
    dctadst4x4_avx2_quant_t, dctadst4x8_avx2_quant_t, dctadst8x8_avx2_quant_t,
    dctadst8x16_avx2_quant_t, dctadst16x8_avx2_quant_t, dctadst16x16_avx2_quant_t,
    fhdct4x4_avx2_quant_t, fhdct8x8_avx2_quant_t, fhdct8x16_avx2_quant_t, fhdct16x8_avx2_quant_t,
    fvdct4x4_avx2_quant_t, fvdct8x8_avx2_quant_t, fvdct8x16_avx2_quant_t, fvdct16x8_avx2_quant_t,
};
pub(crate) use idct::*;
pub(crate) use intrapred::{
    cfl_ac_444_u16_avx2, cfl_ac_sub_u16_avx2, cfl_best_alpha_u16_avx2, cfl_pred_avx2, dc_pred_avx2,
    dr_predict_avx2, edge_conv5_avx2, filter_intra_cells_avx2, horizontal_avx2, paeth_avx2,
    smooth_avx2, smooth_h_avx2, smooth_v_avx2, vertical_avx2,
};
pub(crate) use kmeans::{luma_nearest_indices_avx2, uv_nearest_indices_avx2};
pub(crate) use loopfilter::{loop_filter_avx2, loop_filter_batch_avx2};
pub(crate) use qmatrix::apply_qmatrix_avx2;
pub(crate) use rd::{
    all_zero_i32_avx2, luma_satd_avx2, reconstruct_avx2, residual_dc_avx2, residual_pred_avx2,
    satd_sad_proxy_avx2, sse_recon_avx2, sse_u16_avx2, sum_i32_avx2, sum_u16_avx2,
    sum_u16_strided_avx2,
};
pub(crate) use trellis::{trellis_dist_current_zero_scan_avx2, trellis_round_down_scan_avx2};
pub(crate) use wht::fwht_raw_avx2;
