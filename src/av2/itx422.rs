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
use crate::util::FastRound;

/// avm chroma/DC-leaf tx_scale: `0` when `log2(w)+log2(h) <= 8`, else `(sum-7)/2`.
/// Validated against every `reconstruct_chroma_rect` size and the luma tx_scales.
fn dc_tx_scale(w: usize, h: usize) -> i32 {
    let s = (w.trailing_zeros() + h.trailing_zeros()) as i32;
    if s <= 8 { 0 } else { (s - 7) / 2 }
}

fn dc_tx_index(w: usize, h: usize) -> usize {
    use crate::av2::av2_itx::tx_size::*;
    match (w, h) {
        (4, 4) => TX_4X4,
        (8, 8) => TX_8X8,
        (16, 16) => TX_16X16,
        (32, 32) => TX_32X32,
        (64, 64) => TX_64X64,
        (4, 8) => RTX_4X8,
        (8, 4) => RTX_8X4,
        (8, 16) => RTX_8X16,
        (16, 8) => RTX_16X8,
        (16, 32) => RTX_16X32,
        (32, 16) => RTX_32X16,
        (32, 64) => RTX_32X64,
        (64, 32) => RTX_64X32,
        (4, 16) => RTX_4X16,
        (16, 4) => RTX_16X4,
        (8, 32) => RTX_8X32,
        (32, 8) => RTX_32X8,
        (16, 64) => RTX_16X64,
        (64, 16) => RTX_64X16,
        (4, 32) => RTX_4X32,
        (32, 4) => RTX_32X4,
        (8, 64) => RTX_8X64,
        (64, 8) => RTX_64X8,
        (4, 64) => RTX_4X64,
        (64, 4) => RTX_64X4,
        _ => unreachable!("unsupported chroma/DC tx {w}x{h}"),
    }
}

pub(crate) fn reconstruct_chroma(
    pred: f32,
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    w: usize,
    h: usize,
    bd: i32,
) -> Vec<f32> {
    let (cw, ch) = (w.min(32), h.min(32));
    let txs = dc_tx_scale(w, h);
    let tx = dc_tx_index(w, h);
    // Dequantize scan-ordered levels directly into dav2d's transposed coeff
    // layout (`coeff[col*ch + row]`), skipping the intermediate grid + transpose.
    let mut coeff = vec![0i32; cw * ch];
    for (&l, &rc) in lev.iter().zip(scan.iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> txs) as i32;
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let p = pred.fast_round() as i32;
    let mut out = vec![0f32; w * h];
    crate::av2::av2_itx::inv_txfm_recon_f32(&mut out, &coeff, 0, tx, bd, |_| p);
    out
}

/// As `reconstruct_chroma`, but the prediction base is a per-pixel block (`pred[i]`)
/// rather than a flat DC scalar — used for CfL, whose predictor is dc + alpha*luma_ac.
/// `pred` must be `w*h` in raster order and already clipped to bit depth.
pub(crate) fn reconstruct_chroma_cfl(
    pred: &[i32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    w: usize,
    h: usize,
    bd: i32,
) -> Vec<f32> {
    let (cw, ch) = (w.min(32), h.min(32));
    let txs = dc_tx_scale(w, h);
    let tx = dc_tx_index(w, h);
    let mut coeff = vec![0i32; cw * ch];
    for (&l, &rc) in lev.iter().zip(scan.iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> txs) as i32;
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = vec![0f32; w * h];
    crate::av2::av2_itx::inv_txfm_recon_f32(&mut out, &coeff, 0, tx, bd, |i| pred[i]);
    out
}

#[rustfmt::skip]
pub(crate) fn reconstruct_luma16_adst(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    row_adst: bool,
    col_adst: bool,
    bd: i32,
) -> [f32; 256] {
    let mut coeff = [0i32; 256];
    for (&l, &rc) in lev[..256].iter().zip(scan.iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = rounded as i32;
            coeff[col * 16 + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    // txtp = hor | (ver << 5); hor = row transform, ver = col transform. ADST=2, DCT=0.
    let txtp = (if row_adst { 2 } else { 0 }) | ((if col_adst { 2 } else { 0 }) << 5);
    let mut out = [0f32; 256];
    crate::av2::av2_itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        txtp,
        crate::av2::av2_itx::tx_size::TX_16X16,
        bd,
        |i| (pred[i] + 0.5) as i32,
    );
    out
}

pub(crate) fn reconstruct_luma16(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    bd: i32,
) -> [f32; 256] {
    // Dequantize directly into the transposed coeff layout (`coeff[col*sh + row]`).
    let mut coeff = [0i32; 256];
    for (&l, &rc) in lev[..256].iter().zip(scan[..256].iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = rounded as i32; // tx_scale(TX_16X16)=0
            coeff[col * 16 + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    // DCT_DCT (txtp 0), TX_16X16 — fused reconstruct (add pred + clip + cast in one pass).
    let mut out = [0f32; 256];
    crate::av2::av2_itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::av2_itx::tx_size::TX_16X16,
        bd,
        |i| (pred[i] + 0.5) as i32,
    );
    out
}

/// Bit-exact TX_16X64 luma inverse with per-pixel prediction. Mirrors the (16,64)
/// branch of `reconstruct_chroma_rect` (shifts {6,13}, tx_scale=1, no sqrt2; coeff
/// region 16×32 then nearest vertical upsample 32→64), but adds a per-pixel pred
/// block instead of a scalar DC pred. `scan` is SCAN16X32 (rc: col=rc>>5,row=rc&31).
/// Bit-exact TX_64X16 luma inverse with per-pixel prediction. Mirrors AVM's
/// `inv_txfm_c` for a wide block: transform width clamped to 32, then nearest
/// horizontal upsample 32→64 by column duplication. Shifts {6,13}, tx_scale=1, no
/// sqrt2 (log2(64)+log2(16)=10 even). `scan` is SCAN32X16 (rc: col=rc>>5,row=rc&31).
pub(crate) fn reconstruct_luma_64x16(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    bd: i32,
) -> [f32; 1024] {
    let h = 16usize;
    let mut coeff = [0i32; 512];
    for (&l, &rc) in lev[..512].iter().zip(scan[..512].iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> 1) as i32; // tx_scale(TX_64X16)=1
            coeff[col * h + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = [0f32; 1024];
    crate::av2::av2_itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::av2_itx::tx_size::RTX_64X16,
        bd,
        |i| (pred[i] + 0.5) as i32,
    );
    out
}

pub(crate) fn reconstruct_luma_16x64(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    bd: i32,
) -> [f32; 1024] {
    let ch = 32usize;
    // Zero-initialize: the inverse transform reads the full coefficient block,
    // so every uncoded (zero) coefficient position must be 0, not stack garbage.
    let mut coeff = [0i32; 512];
    for (&l, &rc) in lev[..512].iter().zip(scan[..512].iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> 1) as i32; // tx_scale(TX_16X64)=1
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = [0f32; 1024];
    crate::av2::av2_itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::av2_itx::tx_size::RTX_16X64,
        bd,
        |i| (pred[i] + 0.5) as i32,
    );
    out
}

pub(crate) fn reconstruct_luma(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    bd: i32,
) -> [f32; 1024] {
    // Zero-initialize: the inverse transform reads the full coefficient block,
    // so every uncoded (zero) coefficient position must be 0, not stack garbage.
    let mut coeff = [0i32; 1024];
    for (&l, &rc) in lev[..1024].iter().zip(scan[..1024].iter()) {
        if l != 0.0 {
            let (col, row) = (rc as usize >> 5, rc as usize & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> 1) as i32; // >> tx_scale (TX_32X32 => 1)
            coeff[col * 32 + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = [0f32; 1024];
    crate::av2::av2_itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::av2_itx::tx_size::TX_32X32,
        bd,
        |i| (pred[i] + 0.5) as i32,
    );
    out
}
