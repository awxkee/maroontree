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

/// Integer DC-prediction inverse transform shared by chroma (all subsampling formats)
/// and the scalar-DC-pred luma rect/16 leaves. Routes through the ported dav2d driver
/// (`inv_txfm_add`), so it is bit-exact with avmdec and bit-depth correct (`bd`),
/// replacing the float `Basis::reconstruct[_scan]` path whose synthesis drifts. `pred`
/// is the scalar DC prediction; `lev` the scan-ordered quantised levels; `(w,h)` the
/// transform geometry (coefficient region capped at 32; the driver handles 64-upsample).
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
    for (k, &l) in lev.iter().enumerate() {
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> txs) as i32;
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let p = pred.round() as i32;
    let mut out = vec![0f32; w * h];
    crate::av2::av2_itx::inv_txfm_recon_f32(&mut out, &coeff, 0, tx, bd, |_| p);
    out
}

/// Reconstruct a 32x32 luma block bit-exactly: `clip(pred[i] + inv_tx(dequant(lev)))`.
/// `pred` = per-pixel prediction (1024 samples, row-major); `lev` = scan-ordered
/// levels; `qstep` = ac/dc dqv; `scan` = the TX_32X32 scan. avm dequant:
/// `ROUND_POWER_OF_TWO(|lev|*dqv,3) >> tx_scale`, tx_scale=av2_get_tx_scale(TX_32X32)=1,
/// sign applied last.
/// avm's inverse 16-pt ADST (DST-VII) kernel `tx_kernel_adst_size16[INV_TXFM]`
/// (`av2/common/txb_common.c`), row-major `[k][j]`: pixel `j` accumulates
/// `coeff[k] * ADST16[k*16+j]`. Used for intra TX_16X16 tx_types ADST_ADST /
/// ADST_DCT / DCT_ADST (DST7 1D); `replace_adst_by_ddt` is false for intra so the
/// matrix path (not DDTX) applies.
#[rustfmt::skip]
pub(crate) static ADST16: [i32; 256] = [
     8, 17, 25, 33, 41, 48, 55, 62, 67, 73, 77, 81, 84, 87, 88, 89,
    25, 48, 67, 81, 88, 88, 81, 67, 48, 25,  0,-25,-48,-67,-81,-88,
    41, 73, 88, 84, 62, 25,-17,-55,-81,-89,-77,-48, -8, 33, 67, 87,
    55, 87, 81, 41,-17,-67,-89,-73,-25, 33, 77, 88, 62,  8,-48,-84,
    67, 88, 48,-25,-81,-81,-25, 48, 88, 67,  0,-67,-88,-48, 25, 81,
    77, 77,  0,-77,-77,  0, 77, 77,  0,-77,-77,  0, 77, 77,  0,-77,
    84, 55,-48,-87, -8, 81, 62,-41,-88,-17, 77, 67,-33,-89,-25, 73,
    88, 25,-81,-48, 67, 67,-48,-81, 25, 88,  0,-88,-25, 81, 48,-67,
    89, -8,-88, 17, 87,-25,-84, 33, 81,-41,-77, 48, 73,-55,-67, 62,
    87,-41,-67, 73, 33,-88,  8, 84,-48,-62, 77, 25,-89, 17, 81,-55,
    81,-67,-25, 88,-48,-48, 88,-25,-67, 81,  0,-81, 67, 25,-88, 48,
    73,-84, 25, 55,-89, 48, 33,-87, 67,  8,-77, 81,-17,-62, 88,-41,
    62,-89, 67, -8,-55, 88,-73, 17, 48,-87, 77,-25,-41, 84,-81, 33,
    48,-81, 88,-67, 25, 25,-67, 88,-81, 48,  0,-48, 81,-88, 67,-25,
    33,-62, 81,-89, 84,-67, 41, -8,-25, 55,-77, 88,-87, 73,-48, 17,
    17,-33, 48,-62, 73,-81, 87,-89, 88,-84, 77,-67, 55,-41, 25, -8,
];

/// TX_16X16 luma reconstruction for the ADST family, identical dequant to
/// [`reconstruct_luma16`] (tx_scale 0) but using the ADST inverse. `row_adst`/
/// `col_adst` select the per-axis 1D transform (see [`inv_tx_16x16_adst`]).
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
    for k in 0..256 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
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

/// TX_16X16 luma reconstruction with per-pixel prediction. Dequant mirrors
/// `reconstruct_luma` but `tx_scale = av2_get_tx_scale(TX_16X16) = 0` (256 pels),
/// then the bit-exact 16×16 DCT_DCT inverse. `scan` is SCAN16 (rc = a*32 + c).
pub(crate) fn reconstruct_luma16(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    bd: i32,
) -> [f32; 256] {
    // Dequantize directly into the transposed coeff layout (`coeff[col*sh + row]`).
    let mut coeff = [0i32; 256];
    for k in 0..256 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
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
    let (cw, h) = (32usize, 16usize); // clamped transform width, height
    let mut coeff = vec![0i32; cw * h];
    for (k, &l) in lev.iter().enumerate() {
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
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
    let (w, ch) = (16usize, 32usize);
    let mut coeff = vec![0i32; w * ch];
    for (k, &l) in lev.iter().enumerate() {
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
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
    let mut coeff = [0i32; 1024];
    for k in 0..1024 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
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
