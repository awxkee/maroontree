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

use crate::av2::coder::{Coeff, LosslessDpcmMode, encode_chroma_tu4, encode_luma_tu4};
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
    pub(crate) neutral: f32,
    pub(crate) qc: usize,
    pub(crate) y_ctx: usize,
}

fn luma_candidate_bits(
    tus: &[Vec<Coeff>],
    mode_idx: usize,
    dpcm: Option<LosslessDpcmMode>,
    y_ctx: usize,
    qc: usize,
) -> i32 {
    let mut e = RangeEncoder::new();
    e.qc = qc;
    e.encode_bool(16384, dpcm.is_some() as u32);
    if let Some(mode) = dpcm {
        e.encode_bool(16384, (mode == LosslessDpcmMode::Horizontal) as u32);
    } else {
        e.sym_y_set(0);
        e.sym_y_idx0(y_ctx, mode_idx, 7);
    }
    for coeffs in tus {
        encode_luma_tu4(&mut e, coeffs, 16384, 0);
    }
    (e.finish().len() as i32) * 8
}

fn chroma_candidate_bits(
    u_tus: &[Vec<Coeff>],
    v_tus: &[Vec<Coeff>],
    dpcm: Option<LosslessDpcmMode>,
    luma_directional: bool,
    qc: usize,
) -> i32 {
    let mut e = RangeEncoder::new();
    e.qc = qc;
    e.encode_bool(16384, dpcm.is_some() as u32);
    if let Some(mode) = dpcm {
        e.encode_bool(16384, (mode == LosslessDpcmMode::Horizontal) as u32);
    } else if luma_directional {
        e.sym_uv_mode(1, 1, 7); // DC follows the co-located V/H mode
    } else {
        e.sym_uv_mode(0, 0, 7);
    }
    for coeffs in u_tus {
        encode_chroma_tu4(&mut e, coeffs, 16384, false);
    }
    for coeffs in v_tus {
        encode_chroma_tu4(&mut e, coeffs, 16384, true);
    }
    (e.finish().len() as i32) * 8
}

#[derive(Clone, Copy)]
struct LosslessTuSpec {
    refs: IntraRefSpec,
    mode: usize,
}

fn tu_coeffs_for_mode(src: &[f32], spec: &LosslessTuSpec) -> Vec<Coeff> {
    let LosslessTuSpec { refs, mode: m } = *spec;
    let IntraRefSpec {
        stride: pw,
        y: y0,
        x: x0,
        have_above: _,
        have_left: _,
        top_right: _,
        bottom_left: _,
        neutral,
        ..
    } = refs;
    let pred: [f32; 16] = if m == 0 {
        [dc_pred(src, pw, y0, x0, 4, neutral); 16]
    } else {
        let (above, left, corner) = build_refs(src, &refs);
        let v = match m {
            1 => intrapred::smooth(4, &above, &left),
            2 => intrapred::smooth_v(4, &above, &left),
            3 => intrapred::smooth_h(4, &above, &left),
            _ => intrapred::paeth(4, &above, &left, corner),
        };
        let mut a = [0f32; 16];
        a.copy_from_slice(&v[..16]);
        a
    };
    let mut resid = [0i32; 16];
    for ((dst_row, src_row), pred_row) in resid
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(rect_rows(src, pw, y0, x0, 4, 4))
        .zip(pred.as_chunks::<4>().0.iter())
    {
        for ((dst, &src), &pred) in dst_row.iter_mut().zip(src_row).zip(pred_row) {
            *dst = (src - pred) as i32;
        }
    }
    levels_to_coeffs_4x4(&fwht4x4(&resid))
}

/// Build the spatial residual that the decoder's lossless DPCM accumulation
/// reverses after the inverse WHT. The first row/column is relative to the
/// regular V/H intra predictor; the remaining samples are first differences.
fn dpcm_residual4(
    src: &[f32],
    stride: usize,
    y: usize,
    x: usize,
    neutral: f32,
    mode: LosslessDpcmMode,
) -> [i32; 16] {
    let mut residual = [0i32; 16];
    match mode {
        LosslessDpcmMode::Vertical => {
            for (row_idx, src_row) in rect_rows(src, stride, y, x, 4, 4).enumerate() {
                for (col, (&sample, dst)) in src_row
                    .iter()
                    .zip(&mut residual[row_idx * 4..row_idx * 4 + 4])
                    .enumerate()
                {
                    let pred = if row_idx > 0 {
                        src[(y + row_idx - 1) * stride + x + col]
                    } else if y > 0 {
                        src[(y - 1) * stride + x + col]
                    } else if x > 0 {
                        // With no top edge, directional prediction replicates
                        // the available left sample across AboveRow.
                        src[y * stride + x - 1]
                    } else {
                        neutral - 1.0
                    };
                    *dst = (sample - pred) as i32;
                }
            }
        }
        LosslessDpcmMode::Horizontal => {
            for (row_idx, src_row) in rect_rows(src, stride, y, x, 4, 4).enumerate() {
                let mut pred = if x > 0 {
                    src[(y + row_idx) * stride + x - 1]
                } else if y > 0 {
                    // With no left edge, directional prediction replicates the
                    // available top sample down LeftCol.
                    src[(y - 1) * stride + x]
                } else {
                    neutral + 1.0
                };
                for (&sample, dst) in src_row
                    .iter()
                    .zip(&mut residual[row_idx * 4..row_idx * 4 + 4])
                {
                    *dst = (sample - pred) as i32;
                    pred = sample;
                }
            }
        }
    }
    residual
}

fn dpcm_tu_coeffs(
    src: &[f32],
    stride: usize,
    y: usize,
    x: usize,
    neutral: f32,
    mode: LosslessDpcmMode,
) -> Vec<Coeff> {
    levels_to_coeffs_4x4(&fwht4x4(&dpcm_residual4(src, stride, y, x, neutral, mode)))
}

#[derive(Clone, Copy)]
struct DpcmRegion {
    stride: usize,
    y: usize,
    x: usize,
    rows: usize,
    cols: usize,
}

fn dpcm_tus(
    src: &[f32],
    region: DpcmRegion,
    neutral: f32,
    mode: LosslessDpcmMode,
) -> Vec<Vec<Coeff>> {
    let DpcmRegion {
        stride,
        y,
        x,
        rows,
        cols,
    } = region;
    let mut tus = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            tus.push(dpcm_tu_coeffs(
                src,
                stride,
                y + row * 4,
                x + col * 4,
                neutral,
                mode,
            ));
        }
    }
    tus
}

/// Pick the lowest-rate lossless luma representation for a coded block spanning
/// `rows`×`cols` 4×4 TUs. Search the complete AV2 non-directional set (DC,
/// Smooth, Smooth-V, Smooth-H and Paeth) plus vertical/horizontal lossless DPCM;
/// every residual is formed from the exact reconstructed source references.
pub(crate) fn best_luma_block(src: &[f32], pw: usize, rd: LosslessBlockRd) -> LumaBlockCand {
    let LosslessBlockRd {
        y: by0,
        x: bx0,
        rows,
        cols,
        neutral,
        qc,
        y_ctx,
    } = rd;
    static CANDS: [usize; 5] = [0, 1, 2, 3, 4];
    let mut best = LumaBlockCand {
        mode_idx: 0,
        tus: Vec::new(),
        dpcm: None,
    };
    let mut best_bits = i32::MAX;
    for &m in CANDS.iter() {
        let mut tus = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let (y0, x0) = (by0 + r * 4, bx0 + c * 4);
                let have_above = y0 > 0;
                let have_left = x0 > 0;
                let tr_px = if have_above && c + 1 < cols { 4 } else { 0 };
                let bl_px = if have_left && r + 1 < rows { 4 } else { 0 };
                let coeffs = tu_coeffs_for_mode(
                    src,
                    &LosslessTuSpec {
                        refs: IntraRefSpec {
                            stride: pw,
                            y: y0,
                            x: x0,
                            block_size: 4,
                            have_above,
                            have_left,
                            top_right: tr_px,
                            bottom_left: bl_px,
                            neutral,
                            available_above: 4,
                            available_left: 4,
                        },
                        mode: m,
                    },
                );
                tus.push(coeffs);
            }
        }
        let bits = luma_candidate_bits(&tus, m, None, y_ctx, qc);
        if bits < best_bits {
            best_bits = bits;
            best = LumaBlockCand {
                mode_idx: m,
                tus,
                dpcm: None,
            };
        }
    }

    for mode in [LosslessDpcmMode::Vertical, LosslessDpcmMode::Horizontal] {
        let tus = dpcm_tus(
            src,
            DpcmRegion {
                stride: pw,
                y: by0,
                x: bx0,
                rows,
                cols,
            },
            neutral,
            mode,
        );
        let bits = luma_candidate_bits(&tus, 0, Some(mode), y_ctx, qc);
        if bits < best_bits {
            best_bits = bits;
            best = LumaBlockCand {
                mode_idx: 0,
                tus,
                dpcm: Some(mode),
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

#[derive(Clone, Copy)]
pub(crate) struct ChromaBlockRd {
    pub(crate) y: usize,
    pub(crate) x: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) neutral: f32,
    pub(crate) qc: usize,
    pub(crate) luma_directional: bool,
}

/// Pick one shared U/V DPCM direction. The regular candidate keeps the existing
/// DC-predicted chroma TUs; both DPCM candidates rebuild U and V from the exact
/// reconstructed source and are compared by lossless syntax + coefficient rate.
pub(crate) fn best_chroma_block(
    u: &[f32],
    v: &[f32],
    stride: usize,
    base_u: &[Vec<Coeff>],
    base_v: &[Vec<Coeff>],
    rd: ChromaBlockRd,
) -> ChromaBlockCand {
    let ChromaBlockRd {
        y,
        x,
        rows,
        cols,
        neutral,
        qc,
        luma_directional,
    } = rd;
    let mut best = ChromaBlockCand {
        u_tus: base_u.to_vec(),
        v_tus: base_v.to_vec(),
        dpcm: None,
    };
    if rows == 0 || cols == 0 {
        return best;
    }

    let mut best_bits = chroma_candidate_bits(base_u, base_v, None, luma_directional, qc);

    for mode in [LosslessDpcmMode::Vertical, LosslessDpcmMode::Horizontal] {
        let region = DpcmRegion {
            stride,
            y,
            x,
            rows,
            cols,
        };
        let u_tus = dpcm_tus(u, region, neutral, mode);
        let v_tus = dpcm_tus(v, region, neutral, mode);
        let bits = chroma_candidate_bits(&u_tus, &v_tus, Some(mode), luma_directional, qc);
        if bits < best_bits {
            best_bits = bits;
            best = ChromaBlockCand {
                u_tus,
                v_tus,
                dpcm: Some(mode),
            };
        }
    }
    best
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

#[cfg(test)]
mod dpcm_tests {
    use super::{LosslessDpcmMode, dpcm_residual4};

    fn plane(width: usize, height: usize) -> Vec<f32> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (17 * y + 3 * x + (x * y) % 5) as f32))
            .collect()
    }

    struct Predictor<'a> {
        src: &'a [f32],
        stride: usize,
        y: usize,
        x: usize,
        neutral: f32,
        mode: LosslessDpcmMode,
    }

    impl Predictor<'_> {
        fn at(&self, row: usize, col: usize) -> f32 {
            match self.mode {
                LosslessDpcmMode::Vertical if self.y > 0 => {
                    self.src[(self.y - 1) * self.stride + self.x + col]
                }
                LosslessDpcmMode::Vertical if self.x > 0 => {
                    self.src[self.y * self.stride + self.x - 1]
                }
                LosslessDpcmMode::Vertical => self.neutral - 1.0,
                LosslessDpcmMode::Horizontal if self.x > 0 => {
                    self.src[(self.y + row) * self.stride + self.x - 1]
                }
                LosslessDpcmMode::Horizontal if self.y > 0 => {
                    self.src[(self.y - 1) * self.stride + self.x]
                }
                LosslessDpcmMode::Horizontal => self.neutral + 1.0,
            }
        }
    }

    fn assert_round_trip(
        src: &[f32],
        stride: usize,
        y: usize,
        x: usize,
        neutral: f32,
        mode: LosslessDpcmMode,
    ) {
        let mut residual = dpcm_residual4(src, stride, y, x, neutral, mode);
        match mode {
            LosslessDpcmMode::Vertical => {
                for col in 0..4 {
                    for row in 1..4 {
                        residual[row * 4 + col] += residual[(row - 1) * 4 + col];
                    }
                }
            }
            LosslessDpcmMode::Horizontal => {
                for row in 0..4 {
                    for col in 1..4 {
                        residual[row * 4 + col] += residual[row * 4 + col - 1];
                    }
                }
            }
        }
        let predictor = Predictor {
            src,
            stride,
            y,
            x,
            neutral,
            mode,
        };
        for row in 0..4 {
            for col in 0..4 {
                let pred = predictor.at(row, col) as i32;
                assert_eq!(
                    pred + residual[row * 4 + col],
                    src[(y + row) * stride + x + col] as i32,
                );
            }
        }
    }

    #[test]
    fn vertical_and_horizontal_dpcm_reverse_decoder_accumulation() {
        let src = plane(12, 12);
        assert_round_trip(&src, 12, 4, 4, 128.0, LosslessDpcmMode::Vertical);
        assert_round_trip(&src, 12, 4, 4, 128.0, LosslessDpcmMode::Horizontal);
    }

    #[test]
    fn picture_edge_predictors_match_directional_intra_rules() {
        let src = plane(8, 8);
        assert_round_trip(&src, 8, 0, 0, 128.0, LosslessDpcmMode::Vertical);
        assert_round_trip(&src, 8, 0, 0, 128.0, LosslessDpcmMode::Horizontal);
        assert_round_trip(&src, 8, 0, 4, 128.0, LosslessDpcmMode::Vertical);
        assert_round_trip(&src, 8, 4, 0, 128.0, LosslessDpcmMode::Horizontal);
    }
}
