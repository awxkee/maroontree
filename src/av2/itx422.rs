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

/// avm `tx_kernel_dct2_size16[INV_TXFM]`, row-major `K16[freq*16 + out]`. The AV2
/// integer DCT-2 kernels do NOT nest (size-16 ≠ even rows of size-32), so each size
/// needs its own table.
static K16: [i32; 256] = [
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 90, 87, 80, 70, 57, 43, 26, 9,
    -9, -26, -43, -57, -70, -80, -87, -90, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18,
    18, 50, 75, 89, 87, 57, 9, -43, -80, -90, -70, -26, 26, 70, 90, 80, 43, -9, -57, -87, 83, 35,
    -35, -83, -83, -35, 35, 83, 83, 35, -35, -83, -83, -35, 35, 83, 80, 9, -70, -87, -26, 57, 90,
    43, -43, -90, -57, 26, 87, 70, -9, -80, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50,
    -50, -89, -18, 75, 70, -43, -87, 9, 90, 26, -80, -57, 57, 80, -26, -90, -9, 87, 43, -70, 64,
    -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 57, -80, -26, 90, -9, -87,
    43, 70, -70, -43, 87, 9, -90, 26, 80, -57, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18,
    -75, 75, 18, -89, 50, 43, -90, 57, 26, -87, 70, 9, -80, 80, -9, -70, 87, -26, -57, 90, -43, 35,
    -83, 83, -35, -35, 83, -83, 35, 35, -83, 83, -35, -35, 83, -83, 35, 26, -70, 90, -80, 43, 9,
    -57, 87, -87, 57, -9, -43, 80, -90, 70, -26, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75,
    89, -89, 75, -50, 18, 9, -26, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 26, -9,
];

/// avm `tx_kernel_dct2_size8[INV_TXFM]`, row-major `K8[freq*8 + out]`.
static K8: [i32; 64] = [
    64, 64, 64, 64, 64, 64, 64, 64, 89, 75, 50, 18, -18, -50, -75, -89, 83, 35, -35, -83, -83, -35,
    35, 83, 75, -18, -89, -50, 50, 89, 18, -75, 64, -64, -64, 64, 64, -64, -64, 64, 50, -89, 18,
    75, -75, -18, 89, -50, 35, -83, 83, -35, -35, 83, -83, 35, 18, -50, 75, -89, 89, -75, 50, -18,
];

/// avm `tx_kernel_dct2_size4[INV_TXFM]`, row-major `K4[freq*4 + out]`.
static K4: [i32; 16] = [
    64, 64, 64, 64, 83, 35, -35, -83, 64, -64, -64, 64, 35, -83, 83, -35,
];

/// Generic direct-form 1D inverse DCT-2 of length `n` (n ∈ {4,8,16,32}), using avm's
/// per-size kernel `tx_kernel_dct2_size{n}[INV_TXFM]` (indexed `K[freq*n + out]`).
/// Full-precision accumulate, single final `>> shift` with rounding, then clamp —
/// matching avm's `inv_txfm_dct2_size{n}_c`.
fn idct_line_n(src: &[i32], n: usize, shift: i32, lo: i32, hi: i32) -> [i32; 32] {
    let add = 1i32 << (shift - 1);
    let kern = |j: usize, m: usize| -> i32 {
        match n {
            4 => K4[j * 4 + m],
            8 => K8[j * 8 + m],
            16 => K16[j * 16 + m],
            _ => K32[j * 32 + m],
        }
    };
    let mut dst = [0i32; 32];
    for (m, d) in dst.iter_mut().take(n).enumerate() {
        let mut s = 0i32;
        for (j, &sv) in src.iter().take(n).enumerate() {
            s += kern(j, m) * sv;
        }
        *d = clampi((s + add) >> shift, lo, hi);
    }
    dst
}

/// Bit-exact 4:2:2 chroma reconstruction for ANY rectangular chroma TX, matching
/// avm's `inv_txfm_c` exactly (`av2/common/idct.c`): optional NewSqrt2 pre-scale when
/// log2(w)+log2(h) is odd, horizontal pass (shift `inv_tx_shift[tx][0]`), vertical pass
/// (shift `inv_tx_shift[tx][1]`), then nearest row-duplication 32→64 for tall blocks.
/// The dequant mirrors avm `decodetxb.c`: `dq = (ROUND_POWER_OF_TWO(|lev|·qstep & 0xffffff,3)) >> tx_scale`,
/// sign applied last. Reconstructing this way makes the encoder bit-match avmdec, so the
/// chroma DC-prediction loop cannot drift (the green-cast root cause).
pub(crate) fn reconstruct_chroma_rect(
    pred: f32,
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    w: usize,
    h: usize,
) -> Vec<f32> {
    // (shift_1st, shift_2nd, sqrt2_prescale, tx_scale) per chroma TX (avm tables).
    let (s1, s2, sqrt2, txs): (i32, i32, bool, i32) = match (w, h) {
        (32, 64) => (6, 12, true, 2),
        (32, 32) => (6, 13, false, 1),
        (16, 64) => (6, 13, false, 1),
        (16, 32) => (6, 12, true, 1),
        (8, 64) => (6, 12, true, 1),
        (32, 16) => (6, 12, true, 1),
        (8, 16) => (7, 11, true, 0),
        (4, 32) => (7, 11, true, 0),
        (16, 8) => (7, 11, true, 0),
        _ => unreachable!("unsupported 4:2:2 chroma TX {w}x{h}"),
    };
    let ch = h.min(32); // coefficient/transform height (tall blocks upsample after)
    // Dequantize scan-ordered levels into the ch×w grid (`dq[row*w + col]`).
    let mut block = vec![0i32; ch * w];
    for (k, &l) in lev.iter().enumerate() {
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = (rounded >> txs) as i32;
            block[row * w + col] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    if sqrt2 {
        for v in block.iter_mut() {
            *v = clampi(
                round_shift(*v as i64 * NEW_INV_SQRT2, NEW_SQRT2_BITS),
                -(1 << 15),
                (1 << 15) - 1,
            );
        }
    }
    // Pass 1: horizontal (each row, w-point idct), shift s1, clamp ±2^15 → transposed tmp.
    let mut tmp = vec![0i32; ch * w];
    for row in 0..ch {
        let o = idct_line_n(
            &block[row * w..row * w + w],
            w,
            s1,
            -(1 << 15),
            (1 << 15) - 1,
        );
        for col in 0..w {
            tmp[col * ch + row] = o[col];
        }
    }
    // Pass 2: vertical (each col, ch-point idct), shift s2, clamp ±256 → block.
    for col in 0..w {
        let o = idct_line_n(
            &tmp[col * ch..col * ch + ch],
            ch,
            s2,
            -(1 << 8),
            (1 << 8) - 1,
        );
        for row in 0..ch {
            block[row * w + col] = o[row];
        }
    }
    let p = pred.round() as i32;
    if h > ch {
        // Tall block: nearest vertical upsample ch→h by row duplication, then add pred.
        let mut out = vec![0f32; w * h];
        for row in 0..ch {
            for col in 0..w {
                let v = clampi(p + block[row * w + col], 0, 255) as f32;
                out[(2 * row) * w + col] = v;
                out[(2 * row + 1) * w + col] = v;
            }
        }
        out
    } else {
        let mut out = vec![0f32; w * h];
        for i in 0..w * h {
            out[i] = clampi(p + block[i], 0, 255) as f32;
        }
        out
    }
}

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
/// avm's inverse 16-pt ADST (DST-VII) kernel `tx_kernel_adst_size16[INV_TXFM]`
/// (`av2/common/txb_common.c`), row-major `[k][j]`: pixel `j` accumulates
/// `coeff[k] * ADST16[k*16+j]`. Used for intra TX_16X16 tx_types ADST_ADST /
/// ADST_DCT / DCT_ADST (DST7 1D); `replace_adst_by_ddt` is false for intra so the
/// matrix path (not DDTX) applies.
#[rustfmt::skip]
pub(crate) const ADST16: [i32; 256] = [
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

/// One 16-pt inverse ADST line (matrix form, `dst[m] = Σ_j ADST16[j*16+m]·src[j]`),
/// mirroring `idct_line_n`'s rounding/clamp so the 2D flow is identical to the DCT
/// path (`inv_tx_shift[TX_16X16] = {6,13}`, intermediate ±2^15, final ±2^8).
fn iadst_line_16(src: &[i32], shift: i32, lo: i32, hi: i32) -> [i32; 16] {
    let add = 1i32 << (shift - 1);
    let mut dst = [0i32; 16];
    for (m, d) in dst.iter_mut().enumerate() {
        let mut s = 0i32;
        for (j, &sv) in src.iter().take(16).enumerate() {
            s += ADST16[j * 16 + m] * sv;
        }
        *d = clampi((s + add) >> shift, lo, hi);
    }
    dst
}

/// Bit-exact TX_16X16 inverse for the ADST family. `row_adst`/`col_adst` pick ADST
/// (DST-VII) vs DCT for the horizontal (pass 1) and vertical (pass 2) passes, so this
/// covers ADST_ADST (true,true), ADST_DCT (row DCT=false, col ADST=true) and
/// DCT_ADST (row ADST=true, col DCT=false). Pass order, shifts and clamps match
/// `inv_tx_16x16_8bit`; only the 1D kernel changes.
fn inv_tx_16x16_adst(dqcoeff: &[i32; 256], row_adst: bool, col_adst: bool) -> [i32; 256] {
    let mut tmp = [0i32; 256];
    let mut line = [0i32; 16];
    // Pass 1: row (horizontal), shift 6, clamp ±2^15.
    for y in 0..16 {
        line.copy_from_slice(&dqcoeff[y * 16..y * 16 + 16]);
        let o = if row_adst {
            iadst_line_16(&line, 6, -(1 << 15), (1 << 15) - 1)
        } else {
            let d = idct_line_n(&line, 16, 6, -(1 << 15), (1 << 15) - 1);
            let mut a = [0i32; 16];
            a.copy_from_slice(&d[..16]);
            a
        };
        for x in 0..16 {
            tmp[x * 16 + y] = o[x];
        }
    }
    // Pass 2: column (vertical), shift 13, clamp ±2^8.
    let mut block = [0i32; 256];
    for x in 0..16 {
        line.copy_from_slice(&tmp[x * 16..x * 16 + 16]);
        let o = if col_adst {
            iadst_line_16(&line, 13, -(1 << 8), (1 << 8) - 1)
        } else {
            let d = idct_line_n(&line, 16, 13, -(1 << 8), (1 << 8) - 1);
            let mut a = [0i32; 16];
            a.copy_from_slice(&d[..16]);
            a
        };
        for y in 0..16 {
            block[y * 16 + x] = o[y];
        }
    }
    block
}

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
) -> [f32; 256] {
    let mut dq = [0i32; 256];
    for k in 0..256 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (c, a) = (rc & 31, rc >> 5);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = rounded as i32;
            dq[c * 16 + a] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let res = inv_tx_16x16_adst(&dq, row_adst, col_adst);
    let mut out = [0f32; 256];
    for i in 0..256 {
        out[i] = clampi((pred[i] + 0.5) as i32 + res[i], 0, 255) as f32;
    }
    out
}

/// Bit-exact TX_16X16 DCT_DCT inverse (8-bit). Same construction as
/// `inv_tx_32x32_8bit` with n=16: `inv_tx_shift[TX_16X16] = {6, 13}` (AVM
/// common_data.h), intermediate clamp ±2^15 (bd+8), final clamp ±2^8 (bd).
pub(crate) fn inv_tx_16x16_8bit(dqcoeff: &[i32; 256]) -> [i32; 256] {
    let mut tmp = [0i32; 256];
    let mut line = [0i32; 32];
    // Pass 1: row (horizontal) 16-pt idct, shift 6, clamp ±2^15.
    for y in 0..16 {
        line[..16].copy_from_slice(&dqcoeff[y * 16..y * 16 + 16]);
        let o = idct_line_n(&line[..16], 16, 6, -(1 << 15), (1 << 15) - 1);
        for x in 0..16 {
            tmp[x * 16 + y] = o[x];
        }
    }
    // Pass 2: column (vertical) 16-pt idct, shift 13, clamp ±2^8.
    let mut block = [0i32; 256];
    for x in 0..16 {
        line[..16].copy_from_slice(&tmp[x * 16..x * 16 + 16]);
        let o = idct_line_n(&line[..16], 16, 13, -(1 << 8), (1 << 8) - 1);
        for y in 0..16 {
            block[y * 16 + x] = o[y];
        }
    }
    block
}

/// TX_16X16 luma reconstruction with per-pixel prediction. Dequant mirrors
/// `reconstruct_luma` but `tx_scale = av2_get_tx_scale(TX_16X16) = 0` (256 pels),
/// then the bit-exact 16×16 DCT_DCT inverse. `scan` is SCAN16 (rc = a*32 + c).
pub(crate) fn reconstruct_luma16(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
) -> [f32; 256] {
    let mut dq = [0i32; 256];
    for k in 0..256 {
        let l = lev[k];
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (c, a) = (rc & 31, rc >> 5);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3; // ROUND_POWER_OF_TWO(_, 3)
            let dqmag = rounded as i32; // >> tx_scale (TX_16X16 => 0)
            dq[c * 16 + a] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let res = inv_tx_16x16_8bit(&dq);
    let mut out = [0f32; 256];
    for i in 0..256 {
        out[i] = clampi((pred[i] + 0.5) as i32 + res[i], 0, 255) as f32;
    }
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
) -> [f32; 1024] {
    let (cw, h) = (32usize, 16usize); // clamped transform width, height
    let mut block = vec![0i32; h * cw];
    for (k, &l) in lev.iter().enumerate() {
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> 1) as i32; // tx_scale(TX_64X16)=1
            block[row * cw + col] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut tmp = vec![0i32; h * cw];
    for row in 0..h {
        let o = idct_line_n(
            &block[row * cw..row * cw + cw],
            cw,
            6,
            -(1 << 15),
            (1 << 15) - 1,
        );
        for col in 0..cw {
            tmp[col * h + row] = o[col];
        }
    }
    for col in 0..cw {
        let o = idct_line_n(&tmp[col * h..col * h + h], h, 13, -(1 << 8), (1 << 8) - 1);
        for row in 0..h {
            block[row * cw + col] = o[row];
        }
    }
    let mut out = [0f32; 1024];
    for row in 0..h {
        for col in 0..cw {
            let i0 = row * 64 + 2 * col;
            let i1 = row * 64 + 2 * col + 1;
            let r = block[row * cw + col];
            out[i0] = clampi((pred[i0] + 0.5) as i32 + r, 0, 255) as f32;
            out[i1] = clampi((pred[i1] + 0.5) as i32 + r, 0, 255) as f32;
        }
    }
    out
}

pub(crate) fn reconstruct_luma_16x64(
    pred: &[f32],
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
) -> [f32; 1024] {
    let (w, ch) = (16usize, 32usize);
    let mut block = vec![0i32; ch * w];
    for (k, &l) in lev.iter().enumerate() {
        if l != 0.0 {
            let rc = scan[k] as usize;
            let (col, row) = (rc >> 5, rc & 31);
            let li = l as i64;
            let mag = (li.abs() * qstep as i64) & 0xffffff;
            let rounded = (mag + (1 << 2)) >> 3;
            let dqmag = (rounded >> 1) as i32; // tx_scale(TX_16X64)=1
            block[row * w + col] = if li < 0 { -dqmag } else { dqmag };
        }
    }
    let mut tmp = vec![0i32; ch * w];
    for row in 0..ch {
        let o = idct_line_n(
            &block[row * w..row * w + w],
            w,
            6,
            -(1 << 15),
            (1 << 15) - 1,
        );
        for col in 0..w {
            tmp[col * ch + row] = o[col];
        }
    }
    for col in 0..w {
        let o = idct_line_n(
            &tmp[col * ch..col * ch + ch],
            ch,
            13,
            -(1 << 8),
            (1 << 8) - 1,
        );
        for row in 0..ch {
            block[row * w + col] = o[row];
        }
    }
    let mut out = [0f32; 1024];
    for row in 0..ch {
        for col in 0..w {
            let i0 = (2 * row) * w + col;
            let i1 = (2 * row + 1) * w + col;
            let r = block[row * w + col];
            out[i0] = clampi((pred[i0] + 0.5) as i32 + r, 0, 255) as f32;
            out[i1] = clampi((pred[i1] + 0.5) as i32 + r, 0, 255) as f32;
        }
    }
    out
}

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
