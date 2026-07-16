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

use crate::dct::{dct4x4_t, dct4x8_t, dct8x4_t, dct8x16_t, dct16x8_t, dct16x32_t, dct32x16_t};
use crate::idct::{
    idct_dequant_4x4, idct_dequant_4x8, idct_dequant_8x4, idct_dequant_8x16, idct_dequant_16x8,
    idct_dequant_16x16, idct_dequant_16x32, idct_dequant_32x16,
};
use crate::intrapred::{
    dc_pred_4x4, dc_pred_4x8, dc_pred_8x4, dc_pred_8x16, dc_pred_16x8, dc_pred_16x16,
    dc_pred_16x32, dc_pred_32x16,
};
use crate::quant::{Quant, forward_dct_quant_16x16_t};
use crate::tables::{
    SCAN_4X4, SCAN_4X8, SCAN_8X4, SCAN_8X16, SCAN_16X8, SCAN_16X16, SCAN_16X32, SCAN_32X16,
};

pub(crate) fn chroma_dc_rect(
    recon: &[i32],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    match (width, height) {
        (32, 16) => dc_pred_32x16(recon, stride, x, y, bit_depth),
        (16, 32) => dc_pred_16x32(recon, stride, x, y, bit_depth),
        (16, 8) => dc_pred_16x8(recon, stride, x, y, bit_depth),
        (8, 16) => dc_pred_8x16(recon, stride, x, y, bit_depth),
        _ => dc_pred_16x16(recon, stride, x, y, bit_depth),
    }
}

pub(crate) fn scan_rect(width: usize, height: usize) -> &'static [u32] {
    match (width, height) {
        (32, 16) => &SCAN_32X16,
        (16, 32) => &SCAN_16X32,
        (16, 8) => &SCAN_16X8,
        (8, 16) => &SCAN_8X16,
        _ => &SCAN_16X16,
    }
}

pub(crate) fn fwd_chroma_rect(
    width: usize,
    height: usize,
    residual: &[i32; 512],
    quantizer: &Quant,
) -> ([i32; 512], [f32; 512]) {
    let mut coefficients = [0i32; 512];
    let mut targets = [0.0f32; 512];
    match (width, height) {
        (32, 16) => return dct32x16_t(residual, quantizer),
        (16, 32) => return dct16x32_t(residual, quantizer),
        (16, 8) => {
            let mut input = [0i32; 128];
            input.copy_from_slice(&residual[..128]);
            let (output, target) = dct16x8_t(&input, quantizer);
            coefficients[..128].copy_from_slice(&output);
            targets[..128].copy_from_slice(&target);
        }
        (8, 16) => {
            let mut input = [0i32; 128];
            input.copy_from_slice(&residual[..128]);
            let (output, target) = dct8x16_t(&input, quantizer);
            coefficients[..128].copy_from_slice(&output);
            targets[..128].copy_from_slice(&target);
        }
        _ => {
            let mut input = [0i32; 256];
            input.copy_from_slice(&residual[..256]);
            let (output, target) = forward_dct_quant_16x16_t(&input, quantizer);
            coefficients[..256].copy_from_slice(&output);
            targets[..256].copy_from_slice(&target);
        }
    }
    (coefficients, targets)
}

pub(crate) fn inv_chroma_rect(
    width: usize,
    height: usize,
    coefficients: &[i32; 512],
    quantizer: &Quant,
) -> [i32; 512] {
    let mut residual = [0i32; 512];
    match (width, height) {
        (32, 16) => return idct_dequant_32x16(coefficients, quantizer),
        (16, 32) => return idct_dequant_16x32(coefficients, quantizer),
        (16, 8) => {
            let mut input = [0i32; 128];
            input.copy_from_slice(&coefficients[..128]);
            residual[..128].copy_from_slice(&idct_dequant_16x8(&input, quantizer));
        }
        (8, 16) => {
            let mut input = [0i32; 128];
            input.copy_from_slice(&coefficients[..128]);
            residual[..128].copy_from_slice(&idct_dequant_8x16(&input, quantizer));
        }
        _ => {
            let mut input = [0i32; 256];
            input.copy_from_slice(&coefficients[..256]);
            residual[..256].copy_from_slice(&idct_dequant_16x16(&input, quantizer));
        }
    }
    residual
}

pub(crate) fn chroma_dc_rect8(
    recon: &[i32],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    match (width, height) {
        (8, 4) => dc_pred_8x4(recon, stride, x, y, bit_depth),
        (4, 8) => dc_pred_4x8(recon, stride, x, y, bit_depth),
        _ => dc_pred_4x4(recon, stride, x, y, bit_depth),
    }
}

pub(crate) fn scan_rect8(width: usize, height: usize) -> &'static [u32] {
    match (width, height) {
        (8, 4) => &SCAN_8X4,
        (4, 8) => &SCAN_4X8,
        _ => &SCAN_4X4,
    }
}

pub(crate) fn fwd_chroma_rect8(
    width: usize,
    height: usize,
    residual: &[i32; 64],
    quantizer: &Quant,
) -> ([i32; 64], [f32; 64]) {
    let mut coefficients = [0i32; 64];
    let mut targets = [0.0f32; 64];
    match (width, height) {
        (8, 4) => {
            let mut input = [0i32; 32];
            input.copy_from_slice(&residual[..32]);
            let (output, target) = dct8x4_t(&input, quantizer);
            coefficients[..32].copy_from_slice(&output);
            targets[..32].copy_from_slice(&target);
        }
        (4, 8) => {
            let mut input = [0i32; 32];
            input.copy_from_slice(&residual[..32]);
            let (output, target) = dct4x8_t(&input, quantizer);
            coefficients[..32].copy_from_slice(&output);
            targets[..32].copy_from_slice(&target);
        }
        _ => {
            let mut input = [0i32; 16];
            input.copy_from_slice(&residual[..16]);
            let (output, target) = dct4x4_t(&input, quantizer);
            coefficients[..16].copy_from_slice(&output);
            targets[..16].copy_from_slice(&target);
        }
    }
    (coefficients, targets)
}

pub(crate) fn inv_chroma_rect8(
    width: usize,
    height: usize,
    coefficients: &[i32; 64],
    quantizer: &Quant,
) -> [i32; 64] {
    let mut residual = [0i32; 64];
    match (width, height) {
        (8, 4) => {
            let mut input = [0i32; 32];
            input.copy_from_slice(&coefficients[..32]);
            residual[..32].copy_from_slice(&idct_dequant_8x4(&input, quantizer));
        }
        (4, 8) => {
            let mut input = [0i32; 32];
            input.copy_from_slice(&coefficients[..32]);
            residual[..32].copy_from_slice(&idct_dequant_4x8(&input, quantizer));
        }
        _ => {
            let mut input = [0i32; 16];
            input.copy_from_slice(&coefficients[..16]);
            residual[..16].copy_from_slice(&idct_dequant_4x4(&input, quantizer));
        }
    }
    residual
}
