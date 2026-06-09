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
use crate::av2::cdfs_qctx::CHROMA_SKIP_TX32_QC;
use crate::av2::coder::Coeff;
use crate::av2::lossless::levels_to_coeffs_4x4;
use crate::av2::wht::fwht4x4;

pub(crate) fn sb_tu_contexts(
    tus: &[Vec<Coeff>; 4],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    qc: usize,
) -> ([u32; 4], [usize; 4]) {
    let pos = [(0usize, 0usize), (0, 32), (32, 0), (32, 32)];
    let mut skip_cdfs = [0u32; 4];
    let mut dc_sign_ctxs = [0usize; 4];
    for i in 0..4 {
        let (ty, tx) = pos[i];
        let cy = (sb_y + ty) / 4;
        let cx = (sb_x + tx) / 4;
        let a = &above[cx..cx + 8];
        let l = &left[cy..cy + 8];
        let merge_a = a.iter().fold(0u8, |acc, &b| acc | b);
        let merge_l = l.iter().fold(0u8, |acc, &b| acc | b);
        let sctx = (((merge_a & 0x3F).min(4) + (merge_l & 0x3F).min(4)) as usize + 3) >> 1;
        let dcs: i32 = a.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>()
            + l.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>();
        let sgn = dcs - 8 - 8;
        skip_cdfs[i] = CHROMA_SKIP_TX32_QC[qc][sctx] as u32;
        dc_sign_ctxs[i] = ((sgn != 0) as usize) + ((sgn > 0) as usize);
        // update context with this TU's res_ctx
        let nz: Vec<Coeff> = tus[i].iter().cloned().filter(|&(_, l)| l != 0).collect();
        let res = if nz.is_empty() {
            0x40u8
        } else {
            let cul = (nz
                .iter()
                .map(|&(_, l)| l.unsigned_abs())
                .sum::<u32>()
                .min(63)) as u8;
            let dc = nz
                .iter()
                .find(|&&(s, _)| s == 0)
                .map(|&(_, l)| l)
                .unwrap_or(0);
            let dcbits = if dc > 0 {
                0x80
            } else if dc < 0 {
                0x00
            } else {
                0x40
            };
            (cul & 0x3F) | dcbits
        };
        for k in 0..8 {
            above[cx + k] = res;
            left[cy + k] = res;
        }
    }
    (skip_cdfs, dc_sign_ctxs)
}

/// Edge-replicate a `w`x`h` plane into a `pw`x`ph` buffer. The encoder tiles the
/// frame with whole 64x64 superblocks, so dimensions that are not multiples of 64
/// must be padded up to the SB grid; replicating the last row/column keeps the
/// boundary residual small. The decoder is told the padded size and the caller
/// crops the top-left `w`x`h` region back out.
pub(crate) fn pad_plane(src: &[f32], w: usize, h: usize, pw: usize, ph: usize) -> Vec<f32> {
    if pw == w && ph == h {
        return src.to_vec();
    }
    let mut out = vec![0f32; pw * ph];
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            out[y * pw + x] = src[sy * w + x.min(w - 1)];
        }
    }
    out
}
/// SB-aligned (multiple of 64) size for a given dimension.
pub(crate) fn sb_align(n: usize) -> usize {
    (n + 63) / 64 * 64
}

/// DC prediction for a `bw`-wide × `bh`-tall block (4:2:2 chroma is 32×64).
pub(crate) fn dc_pred(rec: &[f32], w: usize, y0: usize, x0: usize, bs: usize, neutral: f32) -> f32 {
    let (ha, hl) = (y0 > 0, x0 > 0);
    let sa: i64 = if ha {
        (0..bs).map(|i| rec[(y0 - 1) * w + x0 + i] as i64).sum()
    } else {
        0
    };
    let sl: i64 = if hl {
        (0..bs).map(|i| rec[(y0 + i) * w + x0 - 1] as i64).sum()
    } else {
        0
    };
    let b = bs as i64;
    let p = if ha && hl {
        (sa + sl + b) / (2 * b)
    } else if ha {
        (sa + b / 2) / b
    } else if hl {
        (sl + b / 2) / b
    } else {
        return neutral;
    };
    p as f32
}
pub(crate) fn get_residual(
    plane: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    pred: f32,
) -> Vec<f32> {
    let mut r = vec![0f32; bs * bs];
    for yy in 0..bs {
        for (dst, &src) in r[yy * bs..yy * bs + bs]
            .iter_mut()
            .zip(plane[(y0 + yy) * w + x0..(y0 + yy) * w + x0 + bs].iter())
        {
            *dst = src - pred;
        }
    }
    r
}
pub(crate) fn put_block(plane: &mut [f32], w: usize, y0: usize, x0: usize, bs: usize, rec: &[f32]) {
    for yy in 0..bs {
        plane[(y0 + yy) * w + x0..(y0 + yy) * w + x0 + bs]
            .copy_from_slice(&rec[yy * bs..yy * bs + bs]);
    }
}
/// DC prediction for a `bw`-wide × `bh`-tall block (4:2:2 chroma is 32×64).
pub(crate) fn dc_pred_rect(
    rec: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    neutral: f32,
) -> f32 {
    let (ha, hl) = (y0 > 0, x0 > 0);
    let sa: i64 = if ha {
        (0..bw).map(|i| rec[(y0 - 1) * w + x0 + i] as i64).sum()
    } else {
        0
    };
    let sl: i64 = if hl {
        (0..bh).map(|i| rec[(y0 + i) * w + x0 - 1] as i64).sum()
    } else {
        0
    };
    let p = if ha && hl {
        (sa + sl + ((bw + bh) as i64) / 2) / ((bw + bh) as i64)
    } else if ha {
        (sa + bw as i64 / 2) / bw as i64
    } else if hl {
        (sl + bh as i64 / 2) / bh as i64
    } else {
        return neutral;
    };
    p as f32
}
/// Residual of a `bw`-wide × `bh`-tall block, row-major (`r[yy*bw + xx]`).
pub(crate) fn get_residual_rect(
    plane: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    pred: f32,
) -> Vec<f32> {
    let mut r = vec![0f32; bw * bh];
    for yy in 0..bh {
        for xx in 0..bw {
            r[yy * bw + xx] = plane[(y0 + yy) * w + x0 + xx] - pred;
        }
    }
    r
}
/// Write a `bw`-wide × `bh`-tall reconstructed block back into `plane`.
pub(crate) fn put_block_rect(
    plane: &mut [f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    rec: &[f32],
) {
    for yy in 0..bh {
        for xx in 0..bw {
            plane[(y0 + yy) * w + x0 + xx] = rec[yy * bw + xx];
        }
    }
}
pub(crate) fn levels_to_coeffs(lev: &[f32]) -> Vec<Coeff> {
    lev.iter()
        .enumerate()
        .filter(|&(_, l)| *l != 0.0)
        .map(|(k, &l)| (k, l as i32))
        .collect()
}

/// Compute the four per-TU (skip_cdf, dc_sign_ctx) for one 64x64 SB at pixel
/// origin (sb_y, sb_x), advancing the global cf-context arrays (4px granularity).
/// Per-TU neighbour contexts for a lossless 64x64 superblock coded as a 16x16 grid of
/// 4x4 transform units (raster order). Mirrors avm `get_txb_ctx` for luma with a 4x4 tx
/// inside a larger block: each TU has one above and one left neighbour unit, so
/// `txb_skip_ctx = (min(top,4)+min(left,4)+3)>>1` (== avm's `skip_contexts[top][left]`)
/// and `dc_sign_ctx` from the neighbour DC signs. `above`/`left` pack cul-level in bits
/// 0..5 and DC sign in bits 6..7 (0x40 = none/zero), exactly like [`sb_tu_contexts`].
/// Returns the 256 `(txb_skip_ctx, dc_sign_ctx)` pairs; CDF lookup is the caller's job.
pub(crate) fn sb_tu4_contexts(
    tus: &[Vec<Coeff>],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
) -> (Vec<usize>, Vec<usize>) {
    let mut skip_ctx = vec![0usize; 256];
    let mut dc_sign_ctx = vec![0usize; 256];
    for by in 0..16 {
        for bx in 0..16 {
            let i = by * 16 + bx;
            let cx = sb_x / 4 + bx;
            let cy = sb_y / 4 + by;
            let a = above[cx];
            let l = left[cy];
            let top = (a & 0x3F).min(4) as usize;
            let lft = (l & 0x3F).min(4) as usize;
            skip_ctx[i] = (top + lft + 3) >> 1;
            // each neighbour unit contributes its packed sign (0x40 -> 1 baseline)
            let dcs = ((a & 0xC0) >> 6) as i32 + ((l & 0xC0) >> 6) as i32;
            let sgn = dcs - 2; // two neutral units sum to 2
            dc_sign_ctx[i] = ((sgn != 0) as usize) + ((sgn > 0) as usize);
            // update the grid with this TU's result context
            let nz: Vec<Coeff> = tus[i].iter().cloned().filter(|&(_, l)| l != 0).collect();
            let res = if nz.is_empty() {
                0x40u8
            } else {
                let cul = (nz
                    .iter()
                    .map(|&(_, l)| l.unsigned_abs())
                    .sum::<u32>()
                    .min(63)) as u8;
                let dc = nz
                    .iter()
                    .find(|&&(s, _)| s == 0)
                    .map(|&(_, l)| l)
                    .unwrap_or(0);
                let dcbits = if dc > 0 {
                    0x80
                } else if dc < 0 {
                    0x00
                } else {
                    0x40
                };
                (cul & 0x3F) | dcbits
            };
            above[cx] = res;
            left[cy] = res;
        }
    }
    (skip_ctx, dc_sign_ctx)
}

/// Compute one superblock's 256 4x4 lossless TUs for a single plane: per 4x4 block,
/// DC-predict from the reconstruction, forward-WHT the residual, reconstruct (so later
/// blocks predict from exact neighbours), and emit the coded levels as a coeff list.
pub(crate) fn lossless_sb_tus(
    src: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    neutral: f32,
) -> Vec<Vec<Coeff>> {
    let mut tus: Vec<Vec<Coeff>> = Vec::with_capacity(256);
    for by in 0..16 {
        for bx in 0..16 {
            let (y0, x0) = (sb_y + by * 4, sb_x + bx * 4);
            let pred = dc_pred(src, pw, y0, x0, 4, neutral);
            let mut resid = [0i32; 16];
            for yy in 0..4 {
                let row = (y0 + yy) * pw + x0;
                for xx in 0..4 {
                    resid[yy * 4 + xx] = (src[row + xx] - pred) as i32;
                }
            }
            let lev = fwht4x4(&resid);
            tus.push(levels_to_coeffs_4x4(&lev));
        }
    }
    tus
}

/// U: ctx = (above_nz + left_nz) + 6, indexed into the shared txb_skip table.
/// V: ctx = (above_nz + left_nz) + 3 + (co-located U non-zero ? 6 : 0), indexed into
/// the separate v_txb_skip table. Entropy context inits to 0; grids store cul.
pub(crate) fn sb_tu4_chroma_skip(
    tus: &[Vec<Coeff>],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    plane_v: bool,
    eob_u_last: bool,
) -> Vec<usize> {
    // avm reads all U txbs then all V txbs; xd->eob_u_flag is a single field left at the
    // LAST U txb's value, so every V TU's skip context uses (last U TU nonzero), not the
    // co-located one.
    let v_off = 3 + if eob_u_last { 6 } else { 0 };
    let mut skip = vec![0usize; 256];
    for by in 0..16 {
        for bx in 0..16 {
            let i = by * 16 + bx;
            let cx = sb_x / 4 + bx;
            let cy = sb_y / 4 + by;
            let base = ((above[cx] != 0) as usize) + ((left[cy] != 0) as usize);
            skip[i] = if plane_v { base + v_off } else { base + 6 };
            let cul = tus[i]
                .iter()
                .filter(|&&(_, l)| l != 0)
                .map(|&(_, l)| l.unsigned_abs())
                .sum::<u32>()
                .min(63) as u8;
            above[cx] = cul;
            left[cy] = cul;
        }
    }
    skip
}
