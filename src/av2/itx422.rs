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
    use crate::av2::itx::tx_size::*;
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
    // Width-w scans: rc=row*cw+col; legacy stride-32: rc=col*32+row.
    let wscan = scan.get(1).is_some_and(|&v| v as usize == cw);
    let sh = (cw as u32).trailing_zeros();
    let smask = cw - 1;
    for (&l, &rc) in lev.iter().zip(scan.iter()) {
        if l != 0.0 {
            let rc = rc as usize;
            let (row, col) = if wscan {
                (rc >> sh, rc & smask)
            } else {
                (rc & 31, rc >> 5)
            };
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> txs) as i32;
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let p = pred.fast_round() as i32;
    let mut out = vec![0f32; w * h];
    crate::av2::itx::inv_txfm_recon_f32(&mut out, &coeff, 0, tx, bd, |_| p);
    out
}

pub(crate) fn reconstruct_chroma_pred(
    pred: &[i32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    w: usize,
    h: usize,
    bd: i32,
) -> Vec<f32> {
    // Per-pixel (motion-compensated) predictor variant of reconstruct_chroma.
    // Handles w/h up to 64 (TX64 keeps only the top-left 32x32 coeffs).
    let (cw, ch) = (w.min(32), h.min(32));
    let txs = dc_tx_scale(w, h);
    let tx = dc_tx_index(w, h);
    let mut coeff = vec![0i32; cw * ch];
    let wscan = scan.get(1).is_some_and(|&v| v as usize == cw);
    let sh = (cw as u32).trailing_zeros();
    let smask = cw - 1;
    for (&l, &rc) in lev.iter().zip(scan.iter()) {
        if l != 0.0 {
            let rc = rc as usize;
            let (row, col) = if wscan {
                (rc >> sh, rc & smask)
            } else {
                (rc & 31, rc >> 5)
            };
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> txs) as i32;
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = vec![0f32; w * h];
    crate::av2::itx::inv_txfm_recon_f32(&mut out, &coeff, 0, tx, bd, |i| pred[i]);
    out
}

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
    // Width-sensitive scan mapping, matching reconstruct_chroma: width-w scans use
    // rc = row*cw + col; the legacy stride-32 packed scan uses rc = col*32 + row.
    let wscan = scan.get(1).is_some_and(|&v| v as usize == cw);
    let sh = (cw as u32).trailing_zeros();
    let smask = cw - 1;
    for (&l, &rc) in lev.iter().zip(scan.iter()) {
        if l != 0.0 {
            let rc = rc as usize;
            let (row, col) = if wscan {
                (rc >> sh, rc & smask)
            } else {
                (rc & 31, rc >> 5)
            };
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> txs) as i32;
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = vec![0f32; w * h];
    crate::av2::itx::inv_txfm_recon_f32(&mut out, &coeff, 0, tx, bd, |i| pred[i]);
    out
}

#[rustfmt::skip]
static AVM_ADST16_INV: [i16; 256] = [
    8, 17, 25, 33, 41, 48, 55, 62, 67, 73, 77, 81, 84, 87, 88, 89,
    25, 48, 67, 81, 88, 88, 81, 67, 48, 25, 0, -25, -48, -67, -81, -88,
    41, 73, 88, 84, 62, 25, -17, -55, -81, -89, -77, -48, -8, 33, 67, 87,
    55, 87, 81, 41, -17, -67, -89, -73, -25, 33, 77, 88, 62, 8, -48, -84,
    67, 88, 48, -25, -81, -81, -25, 48, 88, 67, 0, -67, -88, -48, 25, 81,
    77, 77, 0, -77, -77, 0, 77, 77, 0, -77, -77, 0, 77, 77, 0, -77,
    84, 55, -48, -87, -8, 81, 62, -41, -88, -17, 77, 67, -33, -89, -25, 73,
    88, 25, -81, -48, 67, 67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67,
    89, -8, -88, 17, 87, -25, -84, 33, 81, -41, -77, 48, 73, -55, -67, 62,
    87, -41, -67, 73, 33, -88, 8, 84, -48, -62, 77, 25, -89, 17, 81, -55,
    81, -67, -25, 88, -48, -48, 88, -25, -67, 81, 0, -81, 67, 25, -88, 48,
    73, -84, 25, 55, -89, 48, 33, -87, 67, 8, -77, 81, -17, -62, 88, -41,
    62, -89, 67, -8, -55, 88, -73, 17, 48, -87, 77, -25, -41, 84, -81, 33,
    48, -81, 88, -67, 25, 25, -67, 88, -81, 48, 0, -48, 81, -88, 67, -25,
    33, -62, 81, -89, 84, -67, 41, -8, -25, 55, -77, 88, -87, 73, -48, 17,
    17, -33, 48, -62, 73, -81, 87, -89, 88, -84, 77, -67, 55, -41, 25, -8,
];
/// AVM `tx_kernel_dct2_size16[INV_TXFM]` (DCT-II, 16-point), same layout.
#[rustfmt::skip]
static AVM_DCT16_INV: [i16; 256] = [
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    90, 87, 80, 70, 57, 43, 26, 9, -9, -26, -43, -57, -70, -80, -87, -90,
    89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
    87, 57, 9, -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87,
    83, 35, -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83,
    80, 9, -70, -87, -26, 57, 90, 43, -43, -90, -57, 26, 87, 70, -9, -80,
    75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
    70, -43, -87, 9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70,
    64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    57, -80, -26, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 26, 80, -57,
    50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
    43, -90, 57, 26, -87, 70, 9, -80, 80, -9, -70, 87, -26, -57, 90, -43,
    35, -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35,
    26, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -26,
    18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
    9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9,
];

/// One 16-point 1-D inverse pass, a verbatim port of AVM `inv_txfm_*_size16_c`:
/// `dst[j*16 + i] = clamp((Σ_k src[i*16+k]·M[k][j] + add) >> shift, cmin, cmax)`
/// (matrix multiply with transposed output, so two passes restore orientation).
#[inline]
fn adst16_pass(
    src: &[i32; 256],
    dst: &mut [i32; 256],
    mat: &[i16; 256],
    shift: i32,
    cmin: i32,
    cmax: i32,
) {
    let add = 1i32 << (shift - 1);
    for i in 0..16 {
        let s = &src[i * 16..i * 16 + 16];
        for j in 0..16 {
            let mut sum = 0i32;
            for k in 0..16 {
                sum += s[k] * mat[k * 16 + j] as i32;
            }
            dst[j * 16 + i] = ((sum + add) >> shift).clamp(cmin, cmax);
        }
    }
}

/// 16×16 luma inverse for the ADST / mixed tx_types, a direct port of AVM's
/// `inv_txfm_c` (TX_16X16): matrix-multiply passes with shifts {6, 13} and the
/// AVM per-pass clamps (row → intermediate `bd+8` range, col → `bd` range). This
/// deliberately does NOT reuse the shared `inv_txfm_recon_f32` wrapper — that
/// wrapper is only verified bit-exact for the symmetric DCT, and its ADST result
/// diverged from AVM, so the encoder's RD estimate (which scores ADST via this
/// reconstruction) did not match what the decoder produces. Mirroring AVM here
/// makes the estimate exact. `row_adst`/`col_adst` select the horizontal/vertical
/// 1-D transform (matching AVM's `g_hor_tx_type`/`g_ver_tx_type`).
pub(crate) fn reconstruct_luma16_adst(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    row_adst: bool,
    col_adst: bool,
    bd: i32,
) -> [f32; 256] {
    // Dequantize into AVM's row-major coefficient grid `block[row*16 + col]`.
    // SCAN16 stores `rc = row*16 + col` directly; the legacy stride-32 scans use
    // `rc = col*32 + row`. Clamp to the decoder's dequant range `[-2^(7+bd), 2^(7+bd)-1]`.
    let wscan = scan.get(1).is_some_and(|&v| v as usize == 16);
    let dq_min = -(1i32 << (7 + bd));
    let dq_max = (1i32 << (7 + bd)) - 1;
    let mut block = [0i32; 256];
    for (&l, &rc) in lev[..256].iter().zip(scan[..256].iter()) {
        if l != 0.0 {
            let rc = rc as usize;
            let pos = if wscan {
                rc
            } else {
                (rc >> 5) + (rc & 31) * 16
            };
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let dqmag = ((mag + (1 << 2)) >> 3) as i32;
            block[pos] = (if li < 0 { -dqmag } else { dqmag }).clamp(dq_min, dq_max);
        }
    }
    let row_mat = if row_adst {
        &AVM_ADST16_INV
    } else {
        &AVM_DCT16_INV
    };
    let col_mat = if col_adst {
        &AVM_ADST16_INV
    } else {
        &AVM_DCT16_INV
    };
    // AVM inv_tx_shift[TX_16X16] = {6, 13}; intermediate (row) range ±2^(bd+7),
    // column-output (residual) range ±2^bd (idct.c rng / col_rng).
    let rng = 1i32 << (bd + 7);
    let col_rng = 1i32 << bd;
    let mut tmp = [0i32; 256];
    let mut res = [0i32; 256];
    adst16_pass(&block, &mut tmp, row_mat, 6, -rng, rng - 1);
    adst16_pass(&tmp, &mut res, col_mat, 13, -col_rng, col_rng - 1);
    // Residual (`res`, back in row-major after the double transpose) + prediction.
    let pmax = (1i32 << bd) - 1;
    let mut out = [0f32; 256];
    for (o, (&p, &r)) in out.iter_mut().zip(pred.iter().zip(res.iter())) {
        *o = ((p + 0.5) as i32 + r).clamp(0, pmax) as f32;
    }
    out
}

pub(crate) fn reconstruct_luma16(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    bd: i32,
) -> [f32; 256] {
    // Dequantize directly into the transposed coeff layout (`coeff[col*16 + row]`).
    // `rc` is stored in the scan's native convention: width-16 scans (SCAN16) use
    // `rc = row*16 + col`; the legacy stride-32 layout uses `rc = col*32 + row`.
    // Decoding with the wrong stride transposes the coefficient grid (a DCT_DCT
    // block then reconstructs transposed) — the cause of the 4:4:4 16x16 luma-leaf
    // mismatch. Detect the convention exactly as `reconstruct_chroma` does.
    let wscan = scan.get(1).is_some_and(|&v| v as usize == 16);
    let mut coeff = [0i32; 256];
    for (&l, &rc) in lev[..256].iter().zip(scan[..256].iter()) {
        if l != 0.0 {
            let rc = rc as usize;
            let (row, col) = if wscan {
                (rc >> 4, rc & 15)
            } else {
                (rc & 31, rc >> 5)
            };
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = rounded as i32; // tx_scale(TX_16X16)=0
            coeff[col * 16 + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    // DCT_DCT (txtp 0), TX_16X16 — fused reconstruct (add pred + clip + cast in one pass).
    let mut out = [0f32; 256];
    crate::av2::itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::itx::tx_size::TX_16X16,
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
            // SCAN32X16 rc = height_freq*32 + width_freq (avm width-32 scan row-major);
            // the RTX_64X16 coeff grid is width_freq*16 + height_freq.
            let hf = rc as usize >> 5; // height freq 0..15
            let wf = rc as usize & 31; // width freq 0..31
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> 1) as i32; // tx_scale(TX_64X16)=1
            coeff[wf * h + hf] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = [0f32; 1024];
    crate::av2::itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::itx::tx_size::RTX_64X16,
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
            // SCAN16X32 is the AVM width-16 scan: rc = row*16 + col.
            let (row, col) = (rc as usize >> 4, rc as usize & 15);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> 1) as i32; // tx_scale(TX_16X64)=1
            coeff[col * ch + row] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut out = [0f32; 1024];
    crate::av2::itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::itx::tx_size::RTX_16X64,
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
    crate::av2::itx::inv_txfm_recon_f32(
        &mut out,
        &coeff,
        0,
        crate::av2::itx::tx_size::TX_32X32,
        bd,
        |i| (pred[i] + 0.5) as i32,
    );
    out
}
