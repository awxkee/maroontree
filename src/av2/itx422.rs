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

//! Bit-exact integer inverse TX_32X64 for 4:2:2 chroma, ported from avm
//! (`av2/common/idct.c` `inv_txfm_c` + `inv_txfm_dct2_size32_c`,
//! `tx_kernel_dct2_size32[INV_TXFM]`, `inv_tx_shift[TX_32X64] = {6, 12}`,
//! `NewInvSqrt2 = 2896`, `NewSqrt2Bits = 12`). The encoder's f32 separable basis
//! was only first-order-correct for this rectangular transform; the per-block
//! mismatch accumulated through intra DC prediction into a chroma drift (green
//! cast). Reconstructing with avm's exact integer transform makes the encoder's
//! reconstruction bit-match the decoder, so the prediction loop no longer drifts.

/// avm `tx_kernel_dct2_size32[INV_TXFM]`, row-major `K32[in*32 + out_group]`.
static K32: [i32; 1024] = [
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4,
    -4, -13, -22, -30, -39, -47, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90, 90, 87, 80, 70,
    57, 43, 26, 9, -9, -26, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -26, -9, 9,
    26, 43, 57, 70, 80, 87, 90, 90, 82, 67, 47, 22, -4, -30, -54, -73, -85, -90, -88, -78, -61,
    -39, -13, 13, 39, 61, 78, 88, 90, 85, 73, 54, 30, 4, -22, -47, -67, -82, -90, 89, 75, 50, 18,
    -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89,
    -89, -75, -50, -18, 18, 50, 75, 89, 88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39, 73, 90,
    85, 61, 22, -22, -61, -85, -90, -73, -39, 4, 47, 78, 90, 82, 54, 13, -30, -67, -88, 87, 57, 9,
    -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 26,
    -26, -70, -90, -80, -43, 9, 57, 87, 85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61,
    -90, -78, -30, 30, 78, 90, 61, 4, -54, -88, -82, -39, 22, 73, 90, 67, 13, -47, -85, 83, 35,
    -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35,
    83, 83, 35, -35, -83, -83, -35, 35, 83, 82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67,
    4, 73, 88, 39, -39, -88, -73, -4, 67, 90, 47, -30, -85, -78, -13, 61, 90, 54, -22, -82, 80, 9,
    -70, -87, -26, 57, 90, 43, -43, -90, -57, 26, 87, 70, -9, -80, -80, -9, 70, 87, 26, -57, -90,
    -43, 43, 90, 57, -26, -87, -70, 9, 80, 78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 30, 90, 54,
    -39, -90, -47, 47, 90, 39, -54, -90, -30, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78, 75, -18,
    -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18,
    -75, -75, 18, 89, 50, -50, -89, -18, 75, 73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61, -47,
    -88, -4, 85, 54, -54, -85, 4, 88, 47, -61, -82, 13, 90, 39, -67, -78, 22, 90, 30, -73, 70, -43,
    -87, 9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -26, 80, 57,
    -57, -80, 26, 90, 9, -87, -43, 70, 67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47,
    -73, -61, 61, 73, -47, -82, 30, 88, -13, -90, -4, 90, 22, -85, -39, 78, 54, -67, 64, -64, -64,
    64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    64, -64, -64, 64, 64, -64, -64, 64, 61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39,
    -78, 54, 67, -67, -54, 78, 39, -85, -22, 90, 4, -90, 13, 88, -30, -82, 47, 73, -61, 57, -80,
    -26, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 26, 80, -57, -57, 80, 26, -90, 9, 87, -43, -70,
    70, 43, -87, -9, 90, -26, -80, 57, 54, -85, -4, 88, -47, -61, 82, 13, -90, 39, 67, -78, -22,
    90, -30, -73, 73, 30, -90, 22, 78, -67, -39, 90, -13, -82, 61, 47, -88, 4, 85, -54, 50, -89,
    18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89,
    -50, -50, 89, -18, -75, 75, 18, -89, 50, 47, -90, 39, 54, -90, 30, 61, -88, 22, 67, -85, 13,
    73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -30, 90, -54, -39, 90, -47, 43,
    -90, 57, 26, -87, 70, 9, -80, 80, -9, -70, 87, -26, -57, 90, -43, -43, 90, -57, -26, 87, -70,
    -9, 80, -80, 9, 70, -87, 26, 57, -90, 43, 39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61,
    -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 30, 47, -90, 67, 4, -73, 88, -39, 35,
    -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83,
    -83, 35, 35, -83, 83, -35, -35, 83, -83, 35, 30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73,
    -90, 67, -13, -47, 85, -85, 47, 13, -67, 90, -73, 22, 39, -82, 88, -54, -4, 61, -90, 78, -30,
    26, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -26, -26, 70, -90, 80, -43,
    -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 26, 22, -61, 85, -90, 73, -39, -4, 47, -78, 90, -82,
    54, -13, -30, 67, -88, 88, -67, 30, 13, -54, 82, -90, 78, -47, 4, 39, -73, 90, -85, 61, -22,
    18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89,
    -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 13, -39, 61, -78, 88, -90, 85, -73, 54, -30,
    4, 22, -47, 67, -82, 90, -90, 82, -67, 47, -22, -4, 30, -54, 73, -85, 90, -88, 78, -61, 39,
    -13, 9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9, -9, 26, -43, 57,
    -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -26, 9, 4, -13, 22, -30, 39, -47, 54, -61, 67,
    -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 47, -39, 30, -22,
    13, -4,
];

const NEW_INV_SQRT2: i64 = 2896;
const NEW_SQRT2_BITS: i64 = 12;

#[inline]
fn round_shift(v: i64, bit: i64) -> i32 {
    ((v + (1i64 << (bit - 1))) >> bit) as i32
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// One size-32 inverse DCT line. `src` = 32 inputs; writes 32 outputs.
/// Equivalent (bit-exact, integer sums are associative) to avm's butterfly:
/// `dst[m] = clamp((sum_j K32[j*32+m]*src[j] + (1<<(shift-1))) >> shift)`.
// Exact even/odd butterfly factorization of the size-32 inverse DCT-II.
// Mathematically identical to the dense matrix product `dst[m] =
// Σ_j K32[j*32+m]·src[j]` (the K32 kernel is perfectly even/odd symmetric, so
// outputs 16..31 are recovered from the low half via `a[15-k] ∓ b[15-k]`),
// but with ~3× fewer multiplies and i32 arithmetic. All intermediates are
// spec-bounded to fit i32 (verified bit-exact against the dense product and
// the reference decoder).
#[inline]
fn idct32_line(src: &[i32; 32], shift: i32, lo: i32, hi: i32) -> [i32; 32] {
    let add = 1i32 << (shift - 1);
    let k = |j: usize, m: usize| K32[j * 32 + m];

    // Odd inputs (1,3,...,31) → b[0..16]
    let mut b = [0i32; 16];
    for (m, bm) in b.iter_mut().enumerate() {
        let mut s = 0i32;
        let mut j = 1;
        while j < 32 {
            s += k(j, m) * src[j];
            j += 2;
        }
        *bm = s;
    }
    // Inputs 2,6,10,...,30 → d[0..8]
    let mut d = [0i32; 8];
    for (m, dm) in d.iter_mut().enumerate() {
        let mut s = 0i32;
        let mut j = 2;
        while j < 32 {
            s += k(j, m) * src[j];
            j += 4;
        }
        *dm = s;
    }
    // Inputs 4,12,20,28 → f[0..4]
    let mut f = [0i32; 4];
    for (m, fm) in f.iter_mut().enumerate() {
        *fm = k(4, m) * src[4] + k(12, m) * src[12] + k(20, m) * src[20] + k(28, m) * src[28];
    }
    // Inputs 8,24 → h[0..2]; inputs 0,16 → g[0..2]
    let h = [
        k(8, 0) * src[8] + k(24, 0) * src[24],
        k(8, 1) * src[8] + k(24, 1) * src[24],
    ];
    let g = [
        k(0, 0) * src[0] + k(16, 0) * src[16],
        k(0, 1) * src[0] + k(16, 1) * src[16],
    ];
    let e = [g[0] + h[0], g[1] + h[1], g[1] - h[1], g[0] - h[0]];
    let mut c = [0i32; 8];
    for kk in 0..4 {
        c[kk] = e[kk] + f[kk];
        c[kk + 4] = e[3 - kk] - f[3 - kk];
    }
    let mut a = [0i32; 16];
    for kk in 0..8 {
        a[kk] = c[kk] + d[kk];
        a[kk + 8] = c[7 - kk] - d[7 - kk];
    }
    let mut dst = [0i32; 32];
    for kk in 0..16 {
        dst[kk] = clampi((a[kk] + b[kk] + add) >> shift, lo, hi);
        dst[kk + 16] = clampi((a[15 - kk] - b[15 - kk] + add) >> shift, lo, hi);
    }
    dst
}

/// Exact avm inverse of a TX_32X64 block (8-bit). `dqcoeff` is the 32x32 (raster,
/// `dq[vfreq*32 + hfreq]`) dequantized coefficient grid. Returns a 32-wide x
/// 64-tall residual (`res[y*32 + x]`).
pub(crate) fn inv_tx_32x64_8bit(dqcoeff: &[i32; 1024]) -> Vec<i32> {
    // Rectangular Sqrt2 pre-scaling (log2(32)+log2(64) is odd).
    let mut block = [0i32; 1024];
    for i in 0..1024 {
        block[i] = round_shift(dqcoeff[i] as i64 * NEW_INV_SQRT2, NEW_SQRT2_BITS);
        block[i] = clampi(block[i], -(1 << 15), (1 << 15) - 1); // clamp_buf, bd+8=16
    }
    // Pass 1: row (horizontal) transform, shift 6, clamp +/-2^15. 32 rows.
    // Reads row y (block[y*32 + 0..31]); writes transposed tmp[outx*32 + y].
    let mut tmp = [0i32; 1024];
    let mut line = [0i32; 32];
    for y in 0..32 {
        line.copy_from_slice(&block[y * 32..y * 32 + 32]);
        let o = idct32_line(&line, 6, -(1 << 15), (1 << 15) - 1);
        for x in 0..32 {
            tmp[x * 32 + y] = o[x];
        }
    }
    // Pass 2: column (vertical) transform, shift 12, clamp +/-256. 32 columns.
    // Reads column x (tmp[x*32 + 0..31]); writes transposed block[outy*32 + x].
    for x in 0..32 {
        line.copy_from_slice(&tmp[x * 32..x * 32 + 32]);
        let o = idct32_line(&line, 12, -(1 << 8), (1 << 8) - 1);
        for y in 0..32 {
            block[y * 32 + x] = o[y];
        }
    }
    // Vertical upsample 32 -> 64 by row duplication (avm nearest, no interp).
    let mut res = vec![0i32; 32 * 64];
    for y in 0..32 {
        for x in 0..32 {
            let v = block[y * 32 + x];
            res[(2 * y) * 32 + x] = v;
            res[(2 * y + 1) * 32 + x] = v;
        }
    }
    res
}

/// Exact avm inverse of a TX_32X32 block (8-bit). `dqcoeff` = 32x32 grid
/// `dq[vfreq*32 + hfreq]`. Returns a 32x32 residual `res[y*32 + x]`. No Sqrt2
/// pre-scale (log2(32)+log2(32)=10 is even); `inv_tx_shift[TX_32X32]={6,13}`; both
/// passes size-32; no upsample.
pub(crate) fn inv_tx_32x32_8bit(dqcoeff: &[i32; 1024]) -> [i32; 1024] {
    // Pass 1: row (horizontal) transform, shift 6, clamp +/-2^15. 32 rows.
    let mut tmp = [0i32; 1024];
    let mut line = [0i32; 32];
    for y in 0..32 {
        line.copy_from_slice(&dqcoeff[y * 32..y * 32 + 32]);
        let o = idct32_line(&line, 6, -(1 << 15), (1 << 15) - 1);
        for x in 0..32 {
            tmp[x * 32 + y] = o[x];
        }
    }
    // Pass 2: column (vertical) transform, shift 13, clamp +/-256. 32 columns.
    let mut block = [0i32; 1024];
    for x in 0..32 {
        line.copy_from_slice(&tmp[x * 32..x * 32 + 32]);
        let o = idct32_line(&line, 13, -(1 << 8), (1 << 8) - 1);
        for y in 0..32 {
            block[y * 32 + x] = o[y];
        }
    }
    block
}

/// Reconstruct a 32x32 luma block bit-exactly: `clip(pred[i] + inv_tx(dequant(lev)))`.
/// `pred` = per-pixel prediction (1024 samples, row-major); `lev` = scan-ordered
/// levels; `qstep` = ac/dc dqv; `scan` = the TX_32X32 scan. avm dequant:
/// `ROUND_POWER_OF_TWO(|lev|*dqv,3) >> tx_scale`, tx_scale=av2_get_tx_scale(TX_32X32)=1,
/// sign applied last.
pub(crate) fn reconstruct_luma(pred: &[f32], lev: &[f32], qstep: i32, scan: &[u16]) -> [f32; 1024] {
    let mut dq = [0i32; 1024];
    for k in 0..1024 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (c, a) = (rc & 31, rc >> 5);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> 1) as i32; // >> tx_scale (TX_32X32 => 1)
            dq[c * 32 + a] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let res = inv_tx_32x32_8bit(&dq);
    let mut out = [0f32; 1024];
    for i in 0..1024 {
        out[i] = clampi((pred[i] + 0.5) as i32 + res[i], 0, 255) as f32;
    }
    out
}

/// Bit-exact 4:2:2 chroma reconstruction for one 32x64 block: dequantize the
/// scan-ordered levels (`dq[vfreq*32+hfreq] = level * qstep`), run avm's exact
/// inverse TX_32X64, add the (DC) prediction and clip. Layout matches
/// `put_block_rect` (`out[y*32 + x]`, 32 wide x 64 tall).
pub(crate) fn reconstruct_422(pred: f32, lev: &[f32], qstep: i32, scan: &[u16]) -> [f32; 2048] {
    let mut dq = [0i32; 1024];
    for k in 0..1024 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (c, a) = (rc & 31, rc >> 5);
            // avm dequant (decodetxb.c): operate on the ABSOLUTE level, then apply
            // sign. dq = ROUND_POWER_OF_TWO(|level|*dqv & 0xffffff, QUANT_TABLE_BITS=3)
            // >> tx_scale; tx_scale = av2_get_tx_scale(TX_32X64) = 2. Sign last —
            // (neg+4)>>3>>2 != -((pos+4)>>3>>2) at rounding boundaries.
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> 2) as i32; // >> tx_scale
            dq[c * 32 + a] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let res = inv_tx_32x64_8bit(&dq);
    let p = pred.round() as i32;
    let mut out = [0f32; 2048];
    for i in 0..2048 {
        out[i] = clampi(p + res[i], 0, 255) as f32;
    }
    out
}
