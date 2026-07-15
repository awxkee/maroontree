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

//! Motion estimation: full-pel diamond from MV predictors, then subpel refine.

use crate::Pixel;
use crate::av2::video::mc::{MotionBlock, predict, predict_with_tmp};
use crate::av2::video::mv::{Mv, mv_cost};

/// Allocation-free predictor list for one block search. The current syntax can
/// contribute at most zero, a frame seed, and three spatial neighbors.
pub(crate) struct MeCandidates {
    values: [Mv; 5],
    len: usize,
}

impl MeCandidates {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            values: [Mv::ZERO; 5],
            len: 1,
        }
    }

    #[inline]
    pub(crate) fn push_unique(&mut self, candidate: Mv) {
        if self.values[..self.len].contains(&candidate) {
            return;
        }
        debug_assert!(self.len < self.values.len());
        if self.len < self.values.len() {
            self.values[self.len] = candidate;
            self.len += 1;
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[Mv] {
        &self.values[..self.len]
    }
}

/// Pixel-specific full-pel SAD. Production video uses f32 end-to-end and reaches
/// the platform SIMD kernel; u16 remains for representation-parity tests.
pub(crate) trait MePixel: Pixel {
    fn block_sad(
        current: &[Self],
        current_stride: usize,
        reference: &[Self],
        reference_stride: usize,
        width: usize,
        height: usize,
    ) -> u32;

    fn block_satd(
        current: &[Self],
        current_stride: usize,
        prediction: &[Self],
        prediction_stride: usize,
        width: usize,
        height: usize,
    ) -> u32;
}

fn sad_scalar<T: Pixel>(
    cur: &[T],
    cur_stride: usize,
    refp: &[T],
    ref_stride: usize,
    ref_off: isize,
    bw: usize,
    bh: usize,
) -> u32 {
    let reference = &refp[ref_off as usize..];
    let mut sum = 0u64;
    for y in 0..bh {
        for x in 0..bw {
            sum += (cur[y * cur_stride + x].to_i32() - reference[y * ref_stride + x].to_i32())
                .unsigned_abs() as u64;
        }
    }
    sum.min(u32::MAX as u64) as u32
}

impl MePixel for f32 {
    #[inline]
    fn block_sad(
        current: &[Self],
        current_stride: usize,
        reference: &[Self],
        reference_stride: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        crate::av2::metrics::sad_f32(
            current,
            current_stride,
            reference,
            reference_stride,
            width,
            height,
        )
        .min(u64::from(u32::MAX)) as u32
    }

    #[inline]
    fn block_satd(
        current: &[Self],
        current_stride: usize,
        prediction: &[Self],
        prediction_stride: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        (crate::av2::metrics::satd_f32(
            current,
            current_stride,
            prediction,
            prediction_stride,
            width,
            height,
        ) >> 2)
            .min(u64::from(u32::MAX)) as u32
    }
}

impl MePixel for u16 {
    #[inline]
    fn block_sad(
        current: &[Self],
        current_stride: usize,
        reference: &[Self],
        reference_stride: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        sad_scalar(
            current,
            current_stride,
            reference,
            reference_stride,
            0,
            width,
            height,
        )
    }

    #[inline]
    fn block_satd(
        current: &[Self],
        current_stride: usize,
        prediction: &[Self],
        prediction_stride: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        satd4x4_scalar(
            current,
            current_stride,
            prediction,
            prediction_stride,
            width,
            height,
        )
    }
}

/// Current/reference planes and their strides for motion estimation.
#[derive(Clone, Copy)]
pub(crate) struct MePlanes<'a, T: Pixel> {
    pub(crate) current: &'a [T],
    pub(crate) current_stride: usize,
    pub(crate) reference: &'a [T],
    pub(crate) reference_stride: usize,
}

/// Block geometry, search limits and MV-rate state for one motion search.
#[derive(Clone, Copy)]
pub(crate) struct MeSearchSpec {
    pub(crate) origin_x: isize,
    pub(crate) origin_y: isize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) reference_mv: Mv,
    pub(crate) lambda_mv: u32,
    pub(crate) max_dx: i32,
    pub(crate) max_dy: i32,
    /// Normalized 8-bit SAD-per-pixel threshold for trusting a predictor and
    /// skipping integer search. Zero disables this approximate speed gate.
    pub(crate) predictor_gate_sad_per_pixel: u32,
    pub(crate) integer_satd_radius: u8,
    pub(crate) bit_depth: u8,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
}

/// Reusable prediction storage for subpel candidate evaluation. One instance is
/// retained by each frame/worker decision context and resized only when a larger
/// block is searched.
pub(crate) struct MeScratch<T: Pixel> {
    prediction: Vec<T>,
    convolve_tmp: Vec<i32>,
}

impl<T: Pixel> Default for MeScratch<T> {
    fn default() -> Self {
        Self {
            prediction: Vec::new(),
            convolve_tmp: Vec::new(),
        }
    }
}

/// 4x4 Hadamard SATD between current and predicted blocks. SATD approximates
/// post-transform coding cost far better than SAD, which barely discriminates
/// between neighboring subpel candidates. Encoder-only metric: it never touches
/// the bitstream, so it cannot cause decoder drift. Blocks are processed in 4x4
/// tiles (bw,bh are multiples of 4 for all inter sizes here).
fn satd4x4_scalar<T: Pixel>(
    cur: &[T],
    cur_stride: usize,
    pred: &[T],
    pred_stride: usize,
    bw: usize,
    bh: usize,
) -> u32 {
    // Orthonormal 4-point Hadamard butterfly (returns 2x-scaled transform; the
    // constant scale cancels in candidate comparison).
    #[inline]
    fn had4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
        let (e, f, g, h) = (a + c, a - c, b + d, b - d);
        [e + g, f + h, f - h, e - g]
    }
    let mut total: u64 = 0;
    let mut ty = 0;
    while ty < bh {
        let mut tx = 0;
        while tx < bw {
            let mut m = [[0i32; 4]; 4];
            for r in 0..4 {
                let cr = &cur[(ty + r) * cur_stride + tx..];
                let pr = &pred[(ty + r) * pred_stride + tx..];
                let d: [i32; 4] = std::array::from_fn(|k| cr[k].to_i32() - pr[k].to_i32());
                m[r] = had4(d[0], d[1], d[2], d[3]);
            }
            for (((&a, &b), &c), &d) in m[0].iter().zip(&m[1]).zip(&m[2]).zip(&m[3]) {
                let col = had4(a, b, c, d);
                for v in col {
                    total += v.unsigned_abs() as u64;
                }
            }
            tx += 4;
        }
        ty += 4;
    }
    // Normalize the 4x-Hadamard gain (>>2 per 2D pass -> >>2 overall here since
    // had4 applies a single scale) so SATD is comparable to SAD-scale mv_cost.
    (total >> 2) as u32
}

/// SATD-based subpel cost: build the subpel prediction, then Hadamard-SATD vs
/// current. Replaces plain SAD for the half/quarter-pel refine, where the
/// distinction between candidates is small and transform-domain cost matters.
fn subpel_satd<T: MePixel>(
    planes: &MePlanes<'_, T>,
    block: &MotionBlock,
    pred: &mut [T],
    convolve_tmp: &mut Vec<i32>,
) -> u32 {
    debug_assert!(pred.len() >= block.width * block.height);
    let pred = &mut pred[..block.width * block.height];
    predict_with_tmp(
        pred,
        block.width,
        planes.reference,
        planes.reference_stride,
        block,
        convolve_tmp,
    );
    T::block_satd(
        planes.current,
        planes.current_stride,
        pred,
        block.width,
        block.width,
        block.height,
    )
}

/// SATD-free subpel cost: build the subpel prediction, sum abs diff vs current.
/// Retained for reference/A-B testing; the search now uses `subpel_satd`.
#[allow(dead_code)]
fn subpel_sad<T: MePixel>(planes: &MePlanes<'_, T>, block: &MotionBlock, pred: &mut [T]) -> u32 {
    debug_assert!(pred.len() >= block.width * block.height);
    let pred = &mut pred[..block.width * block.height];
    predict(
        pred,
        block.width,
        planes.reference,
        planes.reference_stride,
        block,
    );
    T::block_sad(
        planes.current,
        planes.current_stride,
        pred,
        block.width,
        block.width,
        block.height,
    )
}

/// Full-pel diamond from predictors, then half- and quarter-pel refinement.
/// Returns the best MV in 1/8-pel storage units and its distortion+mv_cost.
pub(crate) fn search<T: MePixel>(
    planes: &MePlanes<'_, T>,
    preds: &[Mv],
    spec: &MeSearchSpec,
    scratch: &mut MeScratch<T>,
) -> (Mv, u32) {
    let MePlanes {
        current: cur,
        current_stride: cur_stride,
        reference: refp,
        reference_stride: ref_stride,
    } = *planes;
    let MeSearchSpec {
        origin_x: ox,
        origin_y: oy,
        width: bw,
        height: bh,
        reference_mv: ref_mv,
        lambda_mv,
        max_dx,
        max_dy,
        predictor_gate_sad_per_pixel: predictor_gate,
        integer_satd_radius,
        bit_depth: bd,
        frame_width: fw,
        frame_height: fh,
    } = *spec;
    // In-frame test with 3px margin for the 8-tap subpel filter taps.
    let in_bounds = |mv: Mv| -> bool {
        let dx = (mv.col >> 3) as isize;
        let dy = (mv.row >> 3) as isize;
        ox + dx >= 3
            && oy + dy >= 3
            && ox + dx + bw as isize + 4 <= fw as isize
            && oy + dy + bh as isize + 4 <= fh as isize
    };
    let full = |mv: Mv| -> u32 {
        let dx = (mv.col >> 3) as isize;
        let dy = (mv.row >> 3) as isize;
        let off = (oy + dy) * ref_stride as isize + (ox + dx);
        T::block_sad(cur, cur_stride, &refp[off as usize..], ref_stride, bw, bh)
            .saturating_add(mv_cost(mv.diff(ref_mv), lambda_mv))
    };
    let mut best = if in_bounds(preds[0]) {
        preds[0]
    } else {
        Mv::ZERO
    };
    let mut best_c = full(best);
    for &p in &preds[1..] {
        if !in_bounds(p) {
            continue;
        }
        let c = full(p);
        if c < best_c {
            best_c = c;
            best = p;
        }
    }
    // A full-pel reference predictor with zero SAD reaches the global lower
    // bound: the zero-delta MV syntax cost. Diamond, dense, and subpel
    // candidates cannot beat it, so return before touching the reusable
    // subpel prediction surface.
    let lower_bound = mv_cost(Mv::ZERO, lambda_mv);
    if best == ref_mv && best_c == lower_bound && (best.row | best.col) & 7 == 0 {
        return (best, best_c);
    }
    let best_sad = best_c.saturating_sub(mv_cost(best.diff(ref_mv), lambda_mv));
    let sample_scale = 1u64 << bd.saturating_sub(8);
    let gate_limit = u64::from(predictor_gate)
        .saturating_mul((bw * bh) as u64)
        .saturating_mul(sample_scale);
    let trust_predictor = predictor_gate != 0 && u64::from(best_sad) <= gate_limit;
    if !trust_predictor {
        // Coarse-to-fine full-pel search: step from ~32px down to 1px (1/8-pel units).
        let mut step = 32 * 8;
        while step >= 8 {
            let cands = [
                Mv {
                    row: best.row - step,
                    col: best.col,
                },
                Mv {
                    row: best.row + step,
                    col: best.col,
                },
                Mv {
                    row: best.row,
                    col: best.col - step,
                },
                Mv {
                    row: best.row,
                    col: best.col + step,
                },
                Mv {
                    row: best.row - step,
                    col: best.col - step,
                },
                Mv {
                    row: best.row - step,
                    col: best.col + step,
                },
                Mv {
                    row: best.row + step,
                    col: best.col - step,
                },
                Mv {
                    row: best.row + step,
                    col: best.col + step,
                },
            ];
            let mut improved = false;
            for c in cands {
                if (c.col >> 3).abs() > max_dx || (c.row >> 3).abs() > max_dy || !in_bounds(c) {
                    continue;
                }
                let cost = full(c);
                if cost < best_c {
                    best_c = cost;
                    best = c;
                    improved = true;
                }
            }
            if !improved {
                step >>= 1;
            }
        }
        // Dense integer refine (+-8px) from the coarse-localized best. Running this
        // AFTER coarse-to-fine (not before) is proper hierarchical search: the coarse
        // pass localizes the motion region, then the dense pass refines within it and
        // recovers any full-pel offset the diamond's fixed step pattern skipped.
        let seed = best;
        for dy in -8..=8i32 {
            for dx in -8..=8i32 {
                let c = Mv {
                    row: seed.row + dy * 8,
                    col: seed.col + dx * 8,
                };
                if (c.col >> 3).abs() > max_dx || (c.row >> 3).abs() > max_dy || !in_bounds(c) {
                    continue;
                }
                let cost = full(c);
                if cost < best_c {
                    best_c = cost;
                    best = c;
                }
            }
        }
    }
    // Subpel refine: half (step 4), then quarter (step 2). SATD (not SAD)
    // discriminates subpel candidates by transform-domain cost. The current
    // low-delay syntax codes quarter-pel MVDs, so odd 1/8-pel deltas are excluded.
    // One prediction surface is reused for every subpel candidate. Motion
    // searches are frequent enough that allocating inside the candidate loop
    // materially distorts both speed and allocator contention.
    scratch.prediction.resize(bw * bh, T::default());
    let (subpel_pred, convolve_tmp) = (&mut scratch.prediction, &mut scratch.convolve_tmp);
    let mut sub = |mv: Mv| -> u32 {
        subpel_satd(
            planes,
            &MotionBlock {
                origin_x: ox,
                origin_y: oy,
                mv,
                width: bw,
                height: bh,
                bit_depth: bd,
            },
            &mut *subpel_pred,
            convolve_tmp,
        )
        .saturating_add(mv_cost(mv.diff(ref_mv), lambda_mv))
    };
    best_c = sub(best);
    if integer_satd_radius != 0 {
        let seed = best;
        let radius = i32::from(integer_satd_radius);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let candidate = Mv {
                    row: seed.row + dy * 8,
                    col: seed.col + dx * 8,
                };
                if (candidate.col >> 3).abs() > max_dx
                    || (candidate.row >> 3).abs() > max_dy
                    || !in_bounds(candidate)
                {
                    continue;
                }
                let cost = sub(candidate);
                if cost < best_c {
                    best_c = cost;
                    best = candidate;
                }
            }
        }
    }
    for step in [4i32, 2] {
        let cands = [
            Mv {
                row: best.row - step,
                col: best.col,
            },
            Mv {
                row: best.row + step,
                col: best.col,
            },
            Mv {
                row: best.row,
                col: best.col - step,
            },
            Mv {
                row: best.row,
                col: best.col + step,
            },
        ];
        for c in cands {
            if !in_bounds(c) {
                continue;
            }
            let cost = sub(c);
            if cost < best_c {
                best_c = cost;
                best = c;
            }
        }
    }
    (best, best_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_candidates_preserve_priority_and_remove_duplicates() {
        let mut candidates = MeCandidates::new();
        let frame = Mv { row: 8, col: 16 };
        let left = Mv { row: -8, col: 24 };
        candidates.push_unique(frame);
        candidates.push_unique(Mv::ZERO);
        candidates.push_unique(left);
        candidates.push_unique(frame);
        assert_eq!(candidates.as_slice(), &[Mv::ZERO, frame, left]);
    }
    #[test]
    fn prediction_scratch_retains_its_largest_allocation() {
        let mut scratch = MeScratch::<f32>::default();
        scratch.prediction.resize(64 * 64, 0.0);
        let large = scratch.prediction.as_ptr();
        scratch.prediction.resize(16 * 16, 0.0);
        let small = scratch.prediction.as_ptr();
        scratch.prediction.resize(64 * 64, 0.0);
        let large_again = scratch.prediction.as_ptr();
        assert_eq!(small, large);
        assert_eq!(large_again, large);
    }

    #[test]
    fn two_dimensional_subpel_reuses_convolution_storage() {
        let reference: Vec<f32> = (0..32 * 32).map(|i| (i * 13 % 251) as f32).collect();
        let current = vec![0f32; 8 * 8];
        let planes = MePlanes {
            current: &current,
            current_stride: 8,
            reference: &reference,
            reference_stride: 32,
        };
        let block = MotionBlock {
            origin_x: 8,
            origin_y: 8,
            mv: Mv { row: 4, col: 4 },
            width: 8,
            height: 8,
            bit_depth: 8,
        };
        let mut scratch = MeScratch::<f32>::default();
        scratch.prediction.resize(8 * 8, 0.0);
        let _ = subpel_satd(
            &planes,
            &block,
            &mut scratch.prediction,
            &mut scratch.convolve_tmp,
        );
        let first = scratch.convolve_tmp.as_ptr();
        assert!(scratch.convolve_tmp.capacity() >= (8 + 7) * 8);
        let _ = subpel_satd(
            &planes,
            &block,
            &mut scratch.prediction,
            &mut scratch.convolve_tmp,
        );
        assert_eq!(scratch.convolve_tmp.as_ptr(), first);
    }

    #[test]
    fn exact_fullpel_predictor_skips_search_and_subpel_storage() {
        let reference: Vec<f32> = (0..32 * 32).map(|i| (i * 13 % 251) as f32).collect();
        let mut current = vec![0f32; 8 * 8];
        for y in 0..8 {
            current[y * 8..y * 8 + 8]
                .copy_from_slice(&reference[(8 + y) * 32 + 8..(8 + y) * 32 + 16]);
        }
        let spec = MeSearchSpec {
            origin_x: 8,
            origin_y: 8,
            width: 8,
            height: 8,
            reference_mv: Mv::ZERO,
            lambda_mv: 16,
            max_dx: 16,
            max_dy: 16,
            predictor_gate_sad_per_pixel: 0,
            integer_satd_radius: 0,
            bit_depth: 8,
            frame_width: 32,
            frame_height: 32,
        };
        let mut scratch = MeScratch::default();
        let result = search(
            &MePlanes {
                current: &current,
                current_stride: 8,
                reference: &reference,
                reference_stride: 32,
            },
            &[Mv::ZERO, Mv { row: 8, col: 8 }],
            &spec,
            &mut scratch,
        );
        assert_eq!(result, (Mv::ZERO, mv_cost(Mv::ZERO, 16)));
        assert_eq!(scratch.prediction.capacity(), 0);
    }

    #[test]
    fn predictor_quality_gate_skips_integer_search_but_keeps_subpel_refine() {
        let reference: Vec<f32> = (0..64 * 64)
            .map(|i| ((i * 37 + i / 64 * 11) & 255) as f32)
            .collect();
        let mut current = vec![0f32; 8 * 8];
        for y in 0..8 {
            current[y * 8..y * 8 + 8]
                .copy_from_slice(&reference[(16 + y) * 64 + 20..(16 + y) * 64 + 28]);
        }
        let planes = MePlanes {
            current: &current,
            current_stride: 8,
            reference: &reference,
            reference_stride: 64,
        };
        let base = MeSearchSpec {
            origin_x: 16,
            origin_y: 16,
            width: 8,
            height: 8,
            reference_mv: Mv::ZERO,
            lambda_mv: 0,
            max_dx: 8,
            max_dy: 8,
            predictor_gate_sad_per_pixel: 0,
            integer_satd_radius: 0,
            bit_depth: 8,
            frame_width: 64,
            frame_height: 64,
        };
        let exhaustive = search(&planes, &[Mv::ZERO], &base, &mut MeScratch::default());
        let gated = search(
            &planes,
            &[Mv::ZERO],
            &MeSearchSpec {
                predictor_gate_sad_per_pixel: 255,
                ..base
            },
            &mut MeScratch::default(),
        );
        assert_eq!(exhaustive.0.col, 4 * 8);
        assert!(gated.0.col.abs() <= 6, "gated MV was {:?}", gated.0);
        assert_ne!(gated.1, exhaustive.1);

        let _ = search(
            &planes,
            &[Mv::ZERO],
            &MeSearchSpec {
                integer_satd_radius: 1,
                ..base
            },
            &mut MeScratch::default(),
        );
    }

    #[test]
    fn f32_motion_search_matches_integer_samples() {
        let reference: Vec<u16> = (0..32 * 32)
            .map(|i| ((i * 37 + i / 32 * 11) & 1023) as u16)
            .collect();
        let mut current = vec![0u16; 8 * 8];
        for y in 0..8 {
            current[y * 8..y * 8 + 8]
                .copy_from_slice(&reference[(8 + y) * 32 + 9..(8 + y) * 32 + 17]);
        }
        let spec = MeSearchSpec {
            origin_x: 8,
            origin_y: 8,
            width: 8,
            height: 8,
            reference_mv: Mv::ZERO,
            lambda_mv: 0,
            max_dx: 4,
            max_dy: 4,
            predictor_gate_sad_per_pixel: 0,
            integer_satd_radius: 0,
            bit_depth: 10,
            frame_width: 32,
            frame_height: 32,
        };
        let integer = search(
            &MePlanes {
                current: &current,
                current_stride: 8,
                reference: &reference,
                reference_stride: 32,
            },
            &[Mv::ZERO],
            &spec,
            &mut MeScratch::default(),
        );
        let current_f: Vec<f32> = current.iter().map(|&v| v as f32).collect();
        let reference_f: Vec<f32> = reference.iter().map(|&v| v as f32).collect();
        let float = search(
            &MePlanes {
                current: &current_f,
                current_stride: 8,
                reference: &reference_f,
                reference_stride: 32,
            },
            &[Mv::ZERO],
            &spec,
            &mut MeScratch::default(),
        );
        assert_eq!(float, integer);
        assert_eq!(float.0.col, 8);
    }
}
