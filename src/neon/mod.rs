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
    adst4x4_neon_quant_t, adst4x8_neon_quant_t, adst8x8_neon_quant_t, adst8x16_neon_quant_t,
    adst16x8_neon_quant_t, adst16x16_neon_quant_t, adstdct4x4_neon_quant_t,
    adstdct4x8_neon_quant_t, adstdct8x8_neon_quant_t, adstdct8x16_neon_quant_t,
    adstdct16x8_neon_quant_t, adstdct16x16_neon_quant_t, dct4x4_neon_quant_t, dct4x8_neon_quant_t,
    dct4x16_neon_quant_t, dct8x4_neon_quant_t, dct8x8_neon_quant_t, dct8x16_neon_i32,
    dct8x16_neon_quant_t, dct16x4_neon_quant_t, dct16x8_neon_quant_t, dct16x16_neon_quant_t,
    dct16x32_neon_quant_t, dct32x16_neon_quant_t, dct32x32_neon_i32, dct32x32_neon_quant_t,
    dctadst4x4_neon_quant_t, dctadst4x8_neon_quant_t, dctadst8x8_neon_quant_t,
    dctadst8x16_neon_quant_t, dctadst16x8_neon_quant_t, dctadst16x16_neon_quant_t,
    fhdct4x4_neon_quant_t, fhdct8x8_neon_quant_t, fhdct8x16_neon_quant_t, fhdct16x8_neon_quant_t,
    fvdct4x4_neon_quant_t, fvdct8x8_neon_quant_t, fvdct8x16_neon_quant_t, fvdct16x8_neon_quant_t,
};
pub(crate) use idct::*;
pub(crate) use intrapred::{
    cfl_ac_444_u16_neon, cfl_ac_sub_u16_neon, cfl_best_alpha_u16_neon, cfl_pred_neon, dc_pred_neon,
    dr_predict_neon, edge_conv5_neon, filter_intra_cells_neon, horizontal_neon, paeth_neon,
    smooth_h_neon, smooth_neon, smooth_v_neon, vertical_neon,
};
pub(crate) use kmeans::{luma_nearest_indices_neon, uv_nearest_indices_neon};
pub(crate) use loopfilter::{loop_filter_batch_neon, loop_filter_neon};
pub(crate) use qmatrix::apply_qmatrix_neon;
pub(crate) use rd::{
    all_zero_i32_neon, luma_satd_neon, reconstruct_neon, residual_dc_neon, residual_pred_neon,
    satd_sad_proxy_neon, sse_recon_neon, sse_u16_neon, sum_i32_neon, sum_u16_neon,
    sum_u16_strided_neon,
};
pub(crate) use trellis::{trellis_dist_current_zero_scan_neon, trellis_round_down_scan_neon};
pub(crate) use wht::fwht_raw_neon;
