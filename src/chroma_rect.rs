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

use crate::dct::DctDispatch;
use crate::idct::IdctDispatch;
use crate::intrapred::IntraPredDispatch;
use crate::quant::Quant;
use crate::tables::{
    SCAN_4X4, SCAN_4X8, SCAN_8X4, SCAN_8X16, SCAN_16X8, SCAN_16X16, SCAN_16X32, SCAN_32X16,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_dc_rect(
    intrapred: &IntraPredDispatch,
    recon: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    intrapred.dc_pred(recon, stride, x, y, width, height, bit_depth)
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
    dct: &DctDispatch,
    width: usize,
    height: usize,
    residual: &[i32; 512],
    quantizer: &Quant,
) -> ([i32; 512], [f32; 512]) {
    let mut coefficients = [0i32; 512];
    let mut targets = [0.0f32; 512];
    match (width, height) {
        (32, 16) => return dct.dct32x16_t(residual, quantizer),
        (16, 32) => return dct.dct16x32_t(residual, quantizer),
        (16, 8) => {
            let (output, target) = dct.dct16x8_t(residual.first_chunk::<128>().unwrap(), quantizer);
            coefficients[..128].copy_from_slice(&output);
            targets[..128].copy_from_slice(&target);
        }
        (8, 16) => {
            let (output, target) = dct.dct8x16_t(residual.first_chunk::<128>().unwrap(), quantizer);
            coefficients[..128].copy_from_slice(&output);
            targets[..128].copy_from_slice(&target);
        }
        _ => {
            let (output, target) =
                dct.dct16x16_t(residual.first_chunk::<256>().unwrap(), quantizer);
            coefficients[..256].copy_from_slice(&output);
            targets[..256].copy_from_slice(&target);
        }
    }
    (coefficients, targets)
}

pub(crate) fn inv_chroma_rect(
    idct: &IdctDispatch,
    width: usize,
    height: usize,
    coefficients: &[i32; 512],
    quantizer: &Quant,
) -> [i32; 512] {
    let mut residual = [0i32; 512];
    match (width, height) {
        (32, 16) => return idct.idct_dequant_32x16(coefficients, quantizer),
        (16, 32) => return idct.idct_dequant_16x32(coefficients, quantizer),
        (16, 8) => {
            residual[..128].copy_from_slice(
                &idct.idct_dequant_16x8(coefficients.first_chunk::<128>().unwrap(), quantizer),
            );
        }
        (8, 16) => {
            residual[..128].copy_from_slice(
                &idct.idct_dequant_8x16(coefficients.first_chunk::<128>().unwrap(), quantizer),
            );
        }
        _ => {
            residual[..256].copy_from_slice(
                &idct.idct_dequant_16x16(coefficients.first_chunk::<256>().unwrap(), quantizer),
            );
        }
    }
    residual
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_dc_rect8(
    intrapred: &IntraPredDispatch,
    recon: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    intrapred.dc_pred(recon, stride, x, y, width, height, bit_depth)
}

pub(crate) fn scan_rect8(width: usize, height: usize) -> &'static [u32] {
    match (width, height) {
        (8, 4) => &SCAN_8X4,
        (4, 8) => &SCAN_4X8,
        _ => &SCAN_4X4,
    }
}

pub(crate) fn fwd_chroma_rect8(
    dct: &DctDispatch,
    width: usize,
    height: usize,
    residual: &[i32; 64],
    quantizer: &Quant,
) -> ([i32; 64], [f32; 64]) {
    let mut coefficients = [0i32; 64];
    let mut targets = [0.0f32; 64];
    match (width, height) {
        (8, 4) => {
            let (output, target) = dct.dct8x4_t(residual.first_chunk::<32>().unwrap(), quantizer);
            coefficients[..32].copy_from_slice(&output);
            targets[..32].copy_from_slice(&target);
        }
        (4, 8) => {
            let (output, target) = dct.dct4x8_t(residual.first_chunk::<32>().unwrap(), quantizer);
            coefficients[..32].copy_from_slice(&output);
            targets[..32].copy_from_slice(&target);
        }
        _ => {
            let (output, target) = dct.dct4x4_t(residual.first_chunk::<16>().unwrap(), quantizer);
            coefficients[..16].copy_from_slice(&output);
            targets[..16].copy_from_slice(&target);
        }
    }
    (coefficients, targets)
}

pub(crate) fn inv_chroma_rect8(
    idct: &IdctDispatch,
    width: usize,
    height: usize,
    coefficients: &[i32; 64],
    quantizer: &Quant,
) -> [i32; 64] {
    let mut residual = [0i32; 64];
    match (width, height) {
        (8, 4) => {
            residual[..32].copy_from_slice(
                &idct.idct_dequant_8x4(coefficients.first_chunk::<32>().unwrap(), quantizer),
            );
        }
        (4, 8) => {
            residual[..32].copy_from_slice(
                &idct.idct_dequant_4x8(coefficients.first_chunk::<32>().unwrap(), quantizer),
            );
        }
        _ => {
            residual[..16].copy_from_slice(
                &idct.idct_dequant_4x4(coefficients.first_chunk::<16>().unwrap(), quantizer),
            );
        }
    }
    residual
}
