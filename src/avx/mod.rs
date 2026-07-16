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
mod rd;
mod trellis;
mod wht;

pub(crate) use dct::{
    adst16x16_avx2_quant_t, adstdct16x16_avx2_quant_t, dct8x8_avx2_quant_t, dct8x16_avx2_i32,
    dct8x16_avx2_quant_t, dct16x16_avx2_i32, dct16x16_avx2_quant_t, dct16x32_avx2_quant_t,
    dct32x16_avx2_quant_t, dct32x32_avx2_i32, dct32x32_avx2_quant_t, dctadst16x16_avx2_quant_t,
};
pub(crate) use idct::{
    iadst_dequant_16x16_avx2, iadstdct_dequant_16x16_avx2, idct_dequant_8x8_avx2,
    idct_dequant_16x16_avx2, idct_dequant_32x32_avx2, idctadst_dequant_16x16_avx2,
};
pub(crate) use rd::{residual_dc_avx2, residual_pred_avx2, sse_recon_avx2};
pub(crate) use trellis::{trellis_dist_current_zero_scan_avx2, trellis_round_down_scan_avx2};
pub(crate) use wht::fwht_raw_avx2;
