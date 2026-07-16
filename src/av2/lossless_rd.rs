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

use crate::av2::coder::{Coeff, LosslessDpcmMode, encode_luma_tu4};
use crate::av2::entropy::RangeEncoder;
use crate::av2::helpers::{dc_pred, rect_rows};
use crate::av2::intrapred::{self, IntraRefSpec, build_refs};
use crate::av2::lossless::LumaPalette;
use crate::av2::lossless::levels_to_coeffs_4x4;
use crate::av2::wht::fwht4x4;

/// One plane participating in an intra-block-copy equality test. Coordinates
/// and block dimensions are supplied in luma pixels and shifted for this plane.
#[derive(Clone, Copy)]
pub(crate) struct IbcPlane<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) stride: usize,
    pub(crate) subsampling_x: u8,
    pub(crate) subsampling_y: u8,
}

/// Luma-space geometry of a candidate intra-BC coding block.
#[derive(Clone, Copy)]
pub(crate) struct IbcBlock {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

const DEFAULT_IBC_DX: usize = 64 + 256;

/// Test AV2's second normative BVP candidate: one superblock plus the required
/// 256-pixel IBC delay to the left (`-320, 0` for 64-pixel superblocks).
///
/// The lossless path only selects IBC when every coded plane is sample-exact, so
/// `skip_flag=1` is valid and no residual/MVD syntax is needed. Cropped boundary
/// blocks are deliberately excluded; their padded samples are not part of the
/// visible source and therefore must not influence the decision.
pub(crate) fn exact_default_ibc(
    planes: &[IbcPlane<'_>],
    frame_width: usize,
    frame_height: usize,
    block: IbcBlock,
) -> bool {
    if block.x != DEFAULT_IBC_DX
        || block.y != 0
        || block.width == 0
        || block.height == 0
        || block.width > 64
        || block.height > 64
        || block.width > 32
        || block.height > 32
        || (block.width == 64 && block.height == 64)
        || block.x + block.width > frame_width
        || block.y + block.height > frame_height
    {
        return false;
    }

    planes.iter().all(|plane| {
        let sx = plane.subsampling_x as usize;
        let sy = plane.subsampling_y as usize;
        let x = block.x >> sx;
        let y = block.y >> sy;
        let width = block.width >> sx;
        let height = block.height >> sy;
        let src_x = x - (DEFAULT_IBC_DX >> sx);

        if width == 0 || height == 0 {
            return false;
        }
        (0..height).all(|row| {
            let dst = (y + row) * plane.stride + x;
            let src = (y + row) * plane.stride + src_x;
            plane.data[dst..dst + width] == plane.data[src..src + width]
        })
    })
}

/// Cheap frame pre-pass used to avoid advertising IBC on ordinary lossless
/// images. Any exact 8x8 luma-aligned patch proves that a subsequently visited
/// eligible leaf may use the vertical default BVP. False positives are harmless
/// (the leaf is checked again); false negatives only miss a compression win.
pub(crate) fn frame_has_default_ibc(
    planes: &[IbcPlane<'_>],
    frame_width: usize,
    frame_height: usize,
) -> bool {
    if frame_width < DEFAULT_IBC_DX + 8 || frame_height < 8 {
        return false;
    }
    (0..=frame_height - 8).step_by(8).any(|y| {
        (DEFAULT_IBC_DX..=frame_width - 8).step_by(8).any(|x| {
            exact_default_ibc(
                planes,
                frame_width,
                frame_height,
                IbcBlock {
                    x,
                    y,
                    width: 8,
                    height: 8,
                },
            )
        })
    })
}

/// A full 64x64 leaf cannot signal AV2 intra-BC. Split it to the existing four
/// 32x32 leaves only when at least one child is an exact default-BVP match.
pub(crate) fn sb_needs_ibc_split(
    planes: &[IbcPlane<'_>],
    frame_width: usize,
    frame_height: usize,
    sb_x: usize,
    sb_y: usize,
) -> bool {
    [0usize, 32].into_iter().any(|dy| {
        [0usize, 32].into_iter().any(|dx| {
            exact_default_ibc(
                planes,
                frame_width,
                frame_height,
                IbcBlock {
                    x: sb_x + dx,
                    y: sb_y + dy,
                    width: 32,
                    height: 32,
                },
            )
        })
    })
}

pub(crate) struct LumaBlockCand {
    pub(crate) mode_idx: usize,
    pub(crate) tus: Vec<Vec<Coeff>>,
    pub(crate) dpcm: Option<LosslessDpcmMode>,
}

#[derive(Clone, Copy)]
pub(crate) struct LosslessBlockRd {
    pub(crate) y: usize,
    pub(crate) x: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) neutral: f32,
    pub(crate) qc: usize,
    pub(crate) y_ctx: usize,
}

fn luma_candidate_bits(tus: &[Vec<Coeff>], mode_idx: usize, y_ctx: usize, qc: usize) -> i32 {
    let mut enc = RangeEncoder::new();
    enc.qc = qc;
    enc.encode_bool(16384, 0); // use_dpcm_y = false
    enc.sym_y_set(0);
    enc.sym_y_idx0(y_ctx, mode_idx, 7);
    for coeffs in tus {
        encode_luma_tu4(&mut enc, coeffs, 16384, 0);
    }
    (enc.finish().len() as i32) * 8
}

#[derive(Clone, Copy)]
struct LosslessTuSpec {
    refs: IntraRefSpec,
    mode: usize,
}

fn tu_coeffs_for_mode(src: &[f32], spec: &LosslessTuSpec) -> Vec<Coeff> {
    let LosslessTuSpec { refs, mode } = *spec;
    let IntraRefSpec {
        stride: pw,
        y: y0,
        x: x0,
        neutral,
        ..
    } = refs;
    let pred: [f32; 16] = if mode == 0 {
        [dc_pred(src, pw, y0, x0, 4, neutral); 16]
    } else {
        let (above, left, corner) = build_refs(src, &refs);
        debug_assert_eq!(mode, 4);
        let values = intrapred::paeth(4, &above, &left, corner);
        let mut out = [0f32; 16];
        out.copy_from_slice(&values);
        out
    };
    let mut resid = [0i32; 16];
    for ((dst_row, src_row), pred_row) in resid
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(rect_rows(src, pw, y0, x0, 4, 4))
        .zip(pred.as_chunks::<4>().0)
    {
        for ((dst, &sample), &prediction) in dst_row.iter_mut().zip(src_row).zip(pred_row) {
            *dst = (sample - prediction) as i32;
        }
    }
    levels_to_coeffs_4x4(&fwht4x4(&resid))
}

/// Select between the verified lossless DC and Paeth paths. Paeth uses only the
/// immediate top, left, and top-left references, so it is independent of AVM's
/// coded-neighbor map. Smooth modes additionally consume top-right/bottom-left
/// extensions whose availability changes across coding-block, SB, tile, and
/// cropped-frame boundaries; they must remain excluded until that map is modeled.
/// Cropped transforms use DC because their out-of-frame residual fill is special.
pub(crate) fn best_luma_block(src: &[f32], pw: usize, rd: LosslessBlockRd) -> LumaBlockCand {
    let LosslessBlockRd {
        y: by0,
        x: bx0,
        rows,
        cols,
        frame_width,
        frame_height,
        neutral,
        qc,
        y_ctx,
    } = rd;
    let mut best = LumaBlockCand {
        mode_idx: 0,
        tus: Vec::new(),
        dpcm: None,
    };
    let mut best_bits = i32::MAX;
    let fully_visible = bx0 + cols * 4 <= frame_width && by0 + rows * 4 <= frame_height;
    let modes: &[usize] = if fully_visible { &[0, 4] } else { &[0] };
    for &mode in modes {
        let mut tus = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let (y, x) = (by0 + row * 4, bx0 + col * 4);
                tus.push(tu_coeffs_for_mode(
                    src,
                    &LosslessTuSpec {
                        refs: IntraRefSpec {
                            stride: pw,
                            y,
                            x,
                            block_size: 4,
                            have_above: y > 0,
                            have_left: x > 0,
                            top_right: 0,
                            bottom_left: 0,
                            neutral,
                            available_above: 4,
                            available_left: 4,
                        },
                        mode,
                    },
                ));
            }
        }
        let bits = luma_candidate_bits(&tus, mode, y_ctx, qc);
        if bits < best_bits {
            best_bits = bits;
            best = LumaBlockCand {
                mode_idx: mode,
                tus,
                dpcm: None,
            };
        }
    }
    best
}

pub(crate) struct ChromaBlockCand {
    pub(crate) u_tus: Vec<Vec<Coeff>>,
    pub(crate) v_tus: Vec<Vec<Coeff>>,
    pub(crate) dpcm: Option<LosslessDpcmMode>,
}

/// Keep chroma on the same verified DC/WHT path as luma.
pub(crate) fn best_chroma_block(base_u: &[Vec<Coeff>], base_v: &[Vec<Coeff>]) -> ChromaBlockCand {
    ChromaBlockCand {
        u_tus: base_u.to_vec(),
        v_tus: base_v.to_vec(),
        dpcm: None,
    }
}

/// Return an exact 2..=8-color palette for a complete AVM coding block.  The
/// first implementation deliberately requires an uncropped block and an empty
/// palette-neighbor context: that makes the AVM palette cache empty, while
/// covering the full normative palette-size range.
pub(crate) fn exact_luma_palette(
    src: &[f32],
    stride: usize,
    y: usize,
    x: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Option<LumaPalette> {
    if width < 8 || height < 8 || width > 64 || height > 64 {
        return None;
    }
    let max = (1u16 << bit_depth) - 1;
    let mut colors = Vec::with_capacity(8);
    for row in 0..height {
        for col in 0..width {
            let v = src[(y + row) * stride + x + col] as u16;
            if v > max || colors.contains(&v) {
                continue;
            }
            if colors.len() == 8 {
                return None;
            }
            colors.push(v);
        }
    }
    if colors.len() < 2 {
        return None;
    }
    colors.sort_unstable();
    let mut map = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            let v = src[(y + row) * stride + x + col] as u16;
            map.push(colors.binary_search(&v).unwrap() as u8);
        }
    }
    Some(LumaPalette {
        colors,
        map,
        width,
        height,
    })
}

#[cfg(test)]
mod ibc_tests {
    use super::{IbcBlock, IbcPlane, exact_default_ibc, frame_has_default_ibc, sb_needs_ibc_split};

    fn repeated_plane(width: usize, height: usize) -> Vec<f32> {
        let mut plane = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..320.min(width) {
                plane[y * width + x] = ((y * width + x) & 0xffff) as f32;
            }
            for x in 320..width {
                plane[y * width + x] = plane[y * width + x - 320];
            }
        }
        plane
    }

    #[test]
    fn exact_default_bvp_match_is_selected() {
        let width = 384;
        let height = 64;
        let plane = repeated_plane(width, height);
        let planes = [IbcPlane {
            data: &plane,
            stride: width,
            subsampling_x: 0,
            subsampling_y: 0,
        }];

        assert!(exact_default_ibc(
            &planes,
            width,
            height,
            IbcBlock {
                x: 320,
                y: 0,
                width: 32,
                height: 32,
            },
        ));
        assert!(frame_has_default_ibc(&planes, width, height));
        assert!(sb_needs_ibc_split(&planes, width, height, 320, 0));
    }

    #[test]
    fn mismatch_and_full_64_square_are_rejected() {
        let width = 384;
        let height = 64;
        let mut plane = repeated_plane(width, height);
        plane[320] += 1.0;
        let planes = [IbcPlane {
            data: &plane,
            stride: width,
            subsampling_x: 0,
            subsampling_y: 0,
        }];

        assert!(!exact_default_ibc(
            &planes,
            width,
            height,
            IbcBlock {
                x: 320,
                y: 0,
                width: 32,
                height: 32,
            },
        ));
        assert!(!exact_default_ibc(
            &planes,
            width,
            height,
            IbcBlock {
                x: 320,
                y: 0,
                width: 64,
                height: 64,
            },
        ));
    }

    #[test]
    fn all_subsampled_planes_must_match() {
        let width = 384;
        let height = 64;
        let y = repeated_plane(width, height);
        let mut u = repeated_plane(width / 2, height);
        let mut v = repeated_plane(width / 2, height);
        for row in 0..height {
            for col in 160..width / 2 {
                u[row * (width / 2) + col] = u[row * (width / 2) + col - 160];
                v[row * (width / 2) + col] = v[row * (width / 2) + col - 160];
            }
        }
        let block = IbcBlock {
            x: 320,
            y: 0,
            width: 32,
            height: 32,
        };
        let planes = [
            IbcPlane {
                data: &y,
                stride: width,
                subsampling_x: 0,
                subsampling_y: 0,
            },
            IbcPlane {
                data: &u,
                stride: width / 2,
                subsampling_x: 1,
                subsampling_y: 0,
            },
            IbcPlane {
                data: &v,
                stride: width / 2,
                subsampling_x: 1,
                subsampling_y: 0,
            },
        ];
        assert!(exact_default_ibc(&planes, width, height, block));

        u[160] += 1.0;
        let mismatched_planes = [
            IbcPlane {
                data: &y,
                stride: width,
                subsampling_x: 0,
                subsampling_y: 0,
            },
            IbcPlane {
                data: &u,
                stride: width / 2,
                subsampling_x: 1,
                subsampling_y: 0,
            },
            IbcPlane {
                data: &v,
                stride: width / 2,
                subsampling_x: 1,
                subsampling_y: 0,
            },
        ];
        assert!(!exact_default_ibc(&mismatched_planes, width, height, block,));
    }

    #[test]
    fn cropped_or_unavailable_reference_is_rejected() {
        let width = 384;
        let height = 64;
        let plane = repeated_plane(width, height);
        let planes = [IbcPlane {
            data: &plane,
            stride: width,
            subsampling_x: 0,
            subsampling_y: 0,
        }];

        assert!(!exact_default_ibc(
            &planes,
            width,
            height,
            IbcBlock {
                x: 32,
                y: 0,
                width: 32,
                height: 32,
            },
        ));
        assert!(!exact_default_ibc(
            &planes,
            width,
            height,
            IbcBlock {
                x: 368,
                y: 0,
                width: 32,
                height: 32,
            },
        ));
    }
}
