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

//! Motion-compensated prediction. Full-pel copy + separable 8-tap subpel
//! (AV2 EIGHTTAP_REGULAR, av2/common/filter.h). MV in 1/8-pel units.

use crate::Pixel;
use crate::av2::video::mv::Mv;

const FILTER_BITS: i32 = 7;
/// 16-phase 8-tap regular subpel filter (av2_sub_pel_filters_8).
static SUBPEL8: [[i32; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
];

/// ROUND_POWER_OF_TWO: rounds v by `bits` (bits==0 is a no-op).
#[inline]
fn rpo(v: i32, bits: i32) -> i32 {
    if bits == 0 {
        v
    } else {
        (v + (1 << (bits - 1))) >> bits
    }
}

/// Bordered copy of `plane` (w x h, stride `stride`) with `b`-pixel edge
/// replication; (0,0) maps to (b,b). Matches AVM reference border reads.
pub fn bordered<T: Pixel>(
    plane: &[T],
    w: usize,
    h: usize,
    stride: usize,
    b: usize,
) -> (Vec<T>, usize) {
    let ns = w + 2 * b;
    let mut buf = vec![T::default(); ns * (h + 2 * b)];
    let src_rows = plane.chunks_exact(stride).take(h).map(|row| &row[..w]);
    let dst_rows = buf.chunks_exact_mut(ns).skip(b).take(h);
    for (src_row, dst_row) in src_rows.zip(dst_rows) {
        let (left, rest) = dst_row.split_at_mut(b);
        let (center, right) = rest.split_at_mut(w);
        center.copy_from_slice(src_row);
        left.fill(src_row[0]);
        right.fill(src_row[w - 1]);
    }
    for y in 0..b {
        let (top, bot) = (b * ns, (b + h - 1) * ns);
        buf.copy_within(top..top + ns, y * ns);
        buf.copy_within(bot..bot + ns, (b + h + y) * ns);
    }
    (buf, ns)
}

/// Clamp an eighth-pel MV to AVM's UMV border for a bw x bh block at pixel
/// (px,py) in a w x h frame (luma). Mirrors clamp_mv_to_umv_border_sb.
pub fn clamp_umv(mv: Mv, px: i32, py: i32, bw: i32, bh: i32, w: i32, h: i32) -> Mv {
    const EXT: i32 = 4; // AVM_INTERP_EXTEND
    let spel_l = (EXT + bw) << 4;
    let spel_t = (EXT + bh) << 4;
    let lo_c = (-px * 8) - (spel_l >> 1);
    let hi_c = ((w - bw - px) * 8) + ((spel_l - 16) >> 1);
    let lo_r = (-py * 8) - (spel_t >> 1);
    let hi_r = ((h - bh - py) * 8) + ((spel_t - 16) >> 1);
    Mv {
        row: mv.row.clamp(lo_r, hi_r),
        col: mv.col.clamp(lo_c, hi_c),
    }
}

/// Motion vector, source origin, block geometry and bit depth for one inter
/// prediction. The destination/reference planes and their strides stay explicit.
#[derive(Clone, Copy)]
pub struct MotionBlock {
    pub origin_x: isize,
    pub origin_y: isize,
    pub mv: Mv,
    pub width: usize,
    pub height: usize,
    pub bit_depth: u8,
}

/// Predict a block from `refp` at an eighth-pel motion vector. The referenced
/// source region needs a three-pixel border on all sides.
pub fn predict<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    refp: &[T],
    ref_stride: usize,
    block: &MotionBlock,
) {
    let mut tmp = Vec::new();
    predict_with_tmp(dst, dst_stride, refp, ref_stride, block, &mut tmp);
}

/// Motion compensation with caller-owned storage for the separable 2D filter.
/// Search code retains this buffer across candidates to keep allocation out of
/// the subpel loop; one-shot callers can continue using [`predict`].
pub(crate) fn predict_with_tmp<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    refp: &[T],
    ref_stride: usize,
    block: &MotionBlock,
    tmp: &mut Vec<i32>,
) {
    let MotionBlock {
        origin_x: ox,
        origin_y: oy,
        mv,
        width: bw,
        height: bh,
        bit_depth,
    } = *block;
    let ix = ox + (mv.col >> 3) as isize;
    let iy = oy + (mv.row >> 3) as isize;
    // 1/8-pel fraction -> 4-bit (1/16) phase.
    let px = ((mv.col & 7) * 2) as usize;
    let py = ((mv.row & 7) * 2) as usize;
    let maxv = (1i32 << bit_depth) - 1;
    let bd = bit_depth as i32;

    if px == 0 && py == 0 {
        let src_offset = (iy * ref_stride as isize + ix) as usize;
        let dst_rows = dst
            .chunks_exact_mut(dst_stride)
            .take(bh)
            .map(|row| &mut row[..bw]);
        let src_rows = refp[src_offset..]
            .chunks_exact(ref_stride)
            .take(bh)
            .map(|row| &row[..bw]);
        for (dst_row, src_row) in dst_rows.zip(src_rows) {
            dst_row.copy_from_slice(src_row);
        }
        return;
    }

    let hf = &SUBPEL8[px];
    let vf = &SUBPEL8[py];

    // x-only: RPO(round_0=3) then RPO(FILTER_BITS-3). av2_highbd_convolve_x_sr_c.
    if py == 0 {
        for y in 0..bh {
            let sy = iy + y as isize;
            for x in 0..bw {
                let mut acc = 0i32;
                for (t, &c) in hf.iter().enumerate() {
                    let sx = ix + x as isize + t as isize - 3;
                    acc += c * refp[(sy * ref_stride as isize + sx) as usize].to_i32();
                }
                let r = rpo(acc, 3);
                let v = rpo(r, FILTER_BITS - 3).clamp(0, maxv);
                dst[y * dst_stride + x] = T::from_i32_clamped(v, bit_depth);
            }
        }
        return;
    }
    // y-only: single pass RPO(FILTER_BITS). av2_highbd_convolve_y_sr_c.
    if px == 0 {
        for y in 0..bh {
            for x in 0..bw {
                let mut acc = 0i32;
                for (t, &c) in vf.iter().enumerate() {
                    let sy = iy + y as isize + t as isize - 3;
                    acc += c * refp[(sy * ref_stride as isize + ix + x as isize) as usize].to_i32();
                }
                let v = rpo(acc, FILTER_BITS).clamp(0, maxv);
                dst[y * dst_stride + x] = T::from_i32_clamped(v, bit_depth);
            }
        }
        return;
    }
    // 2d: biased horiz RPO(3), biased vert RPO(11)-offset, RPO(bits=0).
    let round_0 = 3;
    let round_1 = 2 * FILTER_BITS - round_0; // 11 (single-ref)
    let bits = 2 * FILTER_BITS - round_0 - round_1; // 0
    let offset_bits = bd + 2 * FILTER_BITS - round_0;
    tmp.resize((bh + 7) * bw, 0);
    let tmp = &mut tmp[..(bh + 7) * bw];
    for y in 0..bh + 7 {
        let sy = iy + y as isize - 3;
        for x in 0..bw {
            let mut sum = 1i32 << (bd + FILTER_BITS - 1);
            for (t, &c) in hf.iter().enumerate() {
                let sx = ix + x as isize + t as isize - 3;
                sum += c * refp[(sy * ref_stride as isize + sx) as usize].to_i32();
            }
            tmp[y * bw + x] = rpo(sum, round_0);
        }
    }
    for y in 0..bh {
        for x in 0..bw {
            let mut sum = 1i32 << offset_bits;
            for (t, &c) in vf.iter().enumerate() {
                sum += c * tmp[(y + t) * bw + x];
            }
            let res = rpo(sum, round_1)
                - ((1 << (offset_bits - round_1)) + (1 << (offset_bits - round_1 - 1)));
            let v = rpo(res, bits).clamp(0, maxv);
            dst[y * dst_stride + x] = T::from_i32_clamped(v, bit_depth);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullpel_is_copy() {
        let w = 8usize;
        let refp: Vec<u8> = (0..(w * w) as u8).collect();
        let mut dst = vec![0u8; 16];
        predict(
            &mut dst,
            4,
            &refp,
            w,
            &MotionBlock {
                origin_x: 2,
                origin_y: 2,
                mv: Mv::ZERO,
                width: 4,
                height: 4,
                bit_depth: 8,
            },
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], refp[(y + 2) * w + (x + 2)]);
            }
        }
    }

    #[test]
    fn flat_input_preserved() {
        // A constant plane must survive subpel filtering unchanged (taps sum 128).
        let w = 16usize;
        let refp = vec![100u8; w * w];
        let mut dst = vec![0u8; 16];
        predict(
            &mut dst,
            4,
            &refp,
            w,
            &MotionBlock {
                origin_x: 5,
                origin_y: 5,
                mv: Mv { row: 4, col: 4 },
                width: 4,
                height: 4,
                bit_depth: 8,
            },
        );
        for v in &dst {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn matches_avm_convolve() {
        // Bit-exact vs AVM av2_highbd_convolve_2d_sr_c (verified over all 256 phases
        // offline; this pins the 2d path with an LCG-seeded plane).
        let s = 48usize;
        let mut src = vec![0u8; s * s];
        let mut r: u32 = 42;
        for v in src.iter_mut() {
            r = r.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((r >> 16) & 0xff) as u8;
        }
        let mut dst = vec![0u8; 16 * 16];
        predict(
            &mut dst,
            16,
            &src,
            s,
            &MotionBlock {
                origin_x: 16,
                origin_y: 16,
                mv: Mv { row: 5, col: 3 },
                width: 16,
                height: 16,
                bit_depth: 8,
            },
        );
        assert_eq!(&dst[0..4], &[173u8, 91, 70, 255]);
    }

    // A non-origin tile must motion-compensate against the FULL reference frame at
    // frame-absolute coordinates, so an MV crossing the tile boundary fetches real
    // neighbor content. Cropping the reference to the tile (edge replication)
    // reads the wrong pixels -- the cause of the tiled-video streaking. This pins
    // that the full-frame + tile-origin fetch differs from (and is correct vs) the
    // cropped fetch for a boundary-crossing MV.
    #[test]
    fn tile_mc_reads_frame_absolute_reference() {
        // 128-wide u8 reference: left half = 50, right half = 200. Tile 1 at x0=64.
        let (fw, fh) = (128usize, 64usize);
        let mut refp = vec![0u8; fw * fh];
        for r in 0..fh {
            for c in 0..fw {
                refp[r * fw + c] = if c < 64 { 50 } else { 200 };
            }
        }
        const BRD: usize = 72;
        // (A) full frame + tile origin: block at tile-local (0,0) => frame (64,0),
        // MV = 32px left => source column 32 (left half, 50).
        let (bref_a, bs_a) = bordered(&refp, fw, fh, fw, BRD);
        let mut a = [0u8; 8 * 8];
        predict(
            &mut a,
            8,
            &bref_a,
            bs_a,
            &MotionBlock {
                origin_x: (64 + BRD) as isize,
                origin_y: BRD as isize,
                mv: Mv {
                    row: 0,
                    col: -32 * 8,
                },
                width: 8,
                height: 8,
                bit_depth: 8,
            },
        );
        // (B) cropped tile-1 reference (all 200), origin 0 -- the buggy path.
        let mut crop = vec![0u8; 64 * fh];
        for r in 0..fh {
            for c in 0..64 {
                crop[r * 64 + c] = refp[r * fw + 64 + c];
            }
        }
        let (bref_b, bs_b) = bordered(&crop, 64, fh, 64, BRD);
        let mut b = [0u8; 8 * 8];
        predict(
            &mut b,
            8,
            &bref_b,
            bs_b,
            &MotionBlock {
                origin_x: BRD as isize,
                origin_y: BRD as isize,
                mv: Mv {
                    row: 0,
                    col: -32 * 8,
                },
                width: 8,
                height: 8,
                bit_depth: 8,
            },
        );
        assert_eq!(
            a[0], 50,
            "full-frame+origin reads the true left-half content"
        );
        assert_eq!(
            b[0], 200,
            "cropped reference edge-replicates its own (wrong) content"
        );
        assert_ne!(a[0], b[0], "the fix must change the fetched reference");
    }
}
