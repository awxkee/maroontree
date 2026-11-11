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

use super::*;

#[test]
fn cdef_all_candidate_tables_match_per_candidate() {
    use crate::cdef;
    let pool = Pool::new(3);
    for bd in [8u8, 10] {
        let maxv = (1u32 << bd) - 1;
        for (w, h, vis_w, vis_h) in [(72usize, 136usize, 69usize, 133usize), (128, 64, 128, 64)] {
            let px = |i: usize, k: usize| {
                ((i * k + (i * i) % 97 + (i / w) * 7) as u32 % (maxv + 1)) as u16
            };
            let src: Vec<u16> = (0..w * h).map(|i| px(i, 31)).collect();
            // Recon close to the source, sometimes above and sometimes
            // below it, so the SSE sees errors of both signs.
            let recon: Vec<u16> = (0..w * h)
                .map(|i| (src[i] as i32 + (i % 11) as i32 - 5).clamp(0, maxv as i32) as u16)
                .collect();
            let (nbx, nby) = (w / 8, h / 8);
            let uc = w.div_ceil(64);
            let n_units = uc * h.div_ceil(64);
            let dirs: Vec<usize> = (0..nbx * nby).map(|i| (i * 5) % 8).collect();
            let vars: Vec<i32> = (0..nbx * nby).map(|i| [0, 40, 900, 70000][i % 4]).collect();
            let skip: Vec<bool> = (0..nbx * nby).map(|i| i % 9 == 0).collect();
            let damping = 3 + i32::from(bd - 8);
            for perceptual in [false, true] {
                let all = cdef_luma_unit_dists_all(
                    &crate::cdef::CdefDispatch::selected(),
                    &crate::rd_sse::RdDispatch::selected(),
                    &recon,
                    &src,
                    w,
                    h,
                    vis_w,
                    vis_h,
                    &dirs,
                    &vars,
                    &skip,
                    nbx,
                    uc,
                    damping,
                    bd,
                    if perceptual {
                        CdefMetric::CdefDist
                    } else {
                        CdefMetric::Sse
                    },
                    &pool,
                );
                for c in 0..cdef::N_CAND {
                    let (pri, sec) = (cdef::CAND_PRI[c % 4], cdef::CAND_SEC[c / 4]);
                    let want = cdef_luma_unit_dists(
                        &recon, &src, w, h, vis_w, vis_h, &dirs, &vars, &skip, nbx, uc, n_units,
                        pri, sec, damping, bd, perceptual,
                    );
                    let got: Vec<i64> = all.iter().map(|u| u[c]).collect();
                    assert_eq!(
                        got, want,
                        "luma {w}x{h} bd={bd} ({pri},{sec}) p={perceptual}"
                    );
                }
            }
            for (sub_x, sub_y) in [(0, 0), (1, 0), (1, 1)] {
                let (cw, ch) = (w >> sub_x, h >> sub_y);
                let (cw_vis, ch_vis) = (vis_w.div_ceil(1 << sub_x), vis_h.div_ceil(1 << sub_y));
                let uv_dir = if (sub_x, sub_y) == (1, 0) {
                    [7, 0, 2, 4, 5, 6, 6, 6]
                } else {
                    [0, 1, 2, 3, 4, 5, 6, 7]
                };
                let csrc: Vec<u16> = (0..cw * ch).map(|i| px(i, 17)).collect();
                let crec: Vec<u16> = (0..cw * ch)
                    .map(|i| (csrc[i] as i32 + (i % 7) as i32 - 3).clamp(0, maxv as i32) as u16)
                    .collect();
                let all = cdef_chroma_unit_sse_all(
                    &crate::cdef::CdefDispatch::selected(),
                    &crate::rd_sse::RdDispatch::selected(),
                    &crec,
                    &csrc,
                    cw,
                    cw_vis,
                    ch_vis,
                    &dirs,
                    &uv_dir,
                    &skip,
                    nbx,
                    nby,
                    uc,
                    sub_x,
                    sub_y,
                    damping - 1,
                    bd,
                    CdefMetric::Sse,
                    &pool,
                );
                for c in 0..cdef::N_CAND {
                    let (pri, sec) = (cdef::CAND_PRI[c % 4], cdef::CAND_SEC[c / 4]);
                    let want = cdef_chroma_unit_sse(
                        &crec,
                        &csrc,
                        cw,
                        ch,
                        cw_vis,
                        ch_vis,
                        &dirs,
                        &uv_dir,
                        &skip,
                        nbx,
                        nby,
                        uc,
                        n_units,
                        sub_x,
                        sub_y,
                        pri,
                        sec,
                        damping - 1,
                        bd,
                    );
                    let got: Vec<i64> = all.iter().map(|u| u[c]).collect();
                    assert_eq!(got, want, "chroma {sub_x}{sub_y} bd={bd} ({pri},{sec})");
                }
            }
        }
    }
}

#[test]
fn cdef_row_scratch_matches_full_plane_scores() {
    use crate::cdef;
    for bd in [8u8, 10, 12] {
        for (w, h, vis_w, vis_h) in [(16usize, 24usize, 13usize, 21usize), (72, 80, 69, 77)] {
            let scale = 1u16 << (bd - 8);
            let src: Vec<u16> = (0..w * h).map(|i| (i % 8) as u16 * scale).collect();
            let recon: Vec<u16> = (0..w * h)
                .map(|i| {
                    if i % w >= vis_w || i / w >= vis_h {
                        cdef::CDEF_VERY_LARGE as u16
                    } else {
                        (64 + (i * 13 % 128) as u16) * scale
                    }
                })
                .collect();
            let nbx = w / 8;
            let nby = h / 8;
            let uc = w.div_ceil(64);
            let n_units = uc * h.div_ceil(64);
            let dirs: Vec<usize> = (0..nbx * nby).map(|i| i % 8).collect();
            let vars: Vec<i32> = (0..nbx * nby)
                .map(|i| if i % 3 == 0 { 0 } else { 12345 })
                .collect();
            let skip: Vec<bool> = (0..nbx * nby).map(|i| i % 7 == 0).collect();
            let damping = 3 + i32::from(bd - 8);
            for (pri, sec) in [(0, 0), (1, 0), (0, 2), (4, 2)] {
                let mut full = recon.clone();
                for y in (0..h).step_by(8) {
                    for x in (0..w).step_by(8) {
                        let bi = y / 8 * nbx + x / 8;
                        if skip[bi] || x >= vis_w || y >= vis_h {
                            continue;
                        }
                        if pri != 0 || sec != 0 {
                            cdef::cdef_filter_block(
                                &mut full,
                                0,
                                &recon,
                                w,
                                x,
                                y,
                                8,
                                8,
                                cdef::adjust_pri(pri << (bd - 8), vars[bi]),
                                sec << (bd - 8),
                                if pri == 0 { 0 } else { dirs[bi] },
                                damping,
                                bd,
                            );
                        }
                    }
                }
                for perceptual in [false, true] {
                    let mut expected = vec![0i64; n_units];
                    for y in (0..h).step_by(8) {
                        for x in (0..w).step_by(8) {
                            if skip[y / 8 * nbx + x / 8] || x >= vis_w || y >= vis_h {
                                continue;
                            }
                            expected[y / 64 * uc + x / 64] += cdef_block_dist_vis(
                                &src,
                                &full,
                                w,
                                vis_w,
                                vis_h,
                                x,
                                y,
                                u32::from(bd - 8),
                                perceptual,
                            );
                        }
                    }
                    assert_eq!(
                        cdef_luma_unit_dists(
                            &recon, &src, w, h, vis_w, vis_h, &dirs, &vars, &skip, nbx, uc,
                            n_units, pri, sec, damping, bd, perceptual,
                        ),
                        expected,
                        "luma {w}x{h} bd={bd} strength={pri},{sec} perceptual={perceptual}",
                    );
                }
                for (sub_x, sub_y) in [(0, 0), (1, 0), (1, 1)] {
                    let (cw, ch) = (w >> sub_x, h >> sub_y);
                    let (cw_vis, ch_vis) = (vis_w.div_ceil(1 << sub_x), vis_h.div_ceil(1 << sub_y));
                    let (bw, bh) = (8 >> sub_x, 8 >> sub_y);
                    let uv_dir = if (sub_x, sub_y) == (1, 0) {
                        [7, 0, 2, 4, 5, 6, 6, 6]
                    } else {
                        [0, 1, 2, 3, 4, 5, 6, 7]
                    };
                    let csrc: Vec<u16> = (0..cw * ch).map(|i| (i % 8) as u16 * scale).collect();
                    let crecon: Vec<u16> = (0..cw * ch)
                        .map(|i| {
                            if i % cw >= cw_vis || i / cw >= ch_vis {
                                cdef::CDEF_VERY_LARGE as u16
                            } else {
                                (64 + (i * 17 % 128) as u16) * scale
                            }
                        })
                        .collect();
                    let mut full = crecon.clone();
                    let mut expected = vec![0i64; n_units];
                    for by in 0..nby {
                        for bx in 0..nbx {
                            let bi = by * nbx + bx;
                            let (x, y) = (bx * bw, by * bh);
                            if skip[bi] || x >= cw_vis || y >= ch_vis {
                                continue;
                            }
                            if pri != 0 || sec != 0 {
                                cdef::cdef_filter_block(
                                    &mut full,
                                    0,
                                    &crecon,
                                    cw,
                                    x,
                                    y,
                                    bw,
                                    bh,
                                    pri << (bd - 8),
                                    sec << (bd - 8),
                                    if pri == 0 { 0 } else { uv_dir[dirs[bi]] },
                                    damping,
                                    bd,
                                );
                            }
                            for yy in y..(y + bh).min(ch_vis) {
                                for xx in x..(x + bw).min(cw_vis) {
                                    let d = full[yy * cw + xx] as i64 - csrc[yy * cw + xx] as i64;
                                    expected[by / 8 * uc + bx / 8] += d * d;
                                }
                            }
                        }
                    }
                    assert_eq!(
                        cdef_chroma_unit_sse(
                            &crecon, &csrc, cw, ch, cw_vis, ch_vis, &dirs, &uv_dir, &skip, nbx,
                            nby, uc, n_units, sub_x, sub_y, pri, sec, damping, bd,
                        ),
                        expected,
                        "chroma {w}x{h} bd={bd} sub={sub_x},{sub_y} strength={pri},{sec}",
                    );
                }
            }
        }
    }
}

#[test]
fn decision_snapshot_preserves_filter_intra_priors_except_422() {
    let live = Cdfs::new(crate::coef_q::qcat(140));
    let decision = Cdfs::decision_snapshot(crate::coef_q::qcat(140));
    assert_eq!(decision.filter_intra, live.filter_intra);
    let decision_422 = Cdfs::decision_snapshot_422(crate::coef_q::qcat(140));
    assert!(decision_422.filter_intra.iter().all(|cdf| cdf[0] == 16384));
}

fn emit_semantic_fixture(enc: &mut OdEcEncoder, cdfs: &mut Cdfs) {
    for i in 0..64 {
        let skip_ctx = i % cdfs.skip.len();
        enc.encode_symbol(i & 1, &mut cdfs.skip[skip_ctx]);
        enc.encode_bool(i & 2 != 0, 4096 + (i as u16 * 257));

        let part_ctx = i % cdfs.part_split[0].len();
        let part_n = cdfs.part_split[0][part_ctx].len() - 1;
        enc.encode_symbol(i % part_n, &mut cdfs.part_split[0][part_ctx]);
        enc.encode_gathered_partition(i & 4 != 0, i & 8 == 0, &cdfs.part_split[0][part_ctx]);

        enc.encode_symbol_noupdate(i & 1, &cdfs.intrabc);
    }
}

#[test]
fn semantic_tokens_match_live_entropy_coding() {
    for updating_cdf in [false, true] {
        let mut capture_cdfs = Cdfs::new(crate::coef_q::qcat(140));
        let capture_slots = capture_cdfs.semantic_slots();
        let mut capture = OdEcEncoder::new().with_updating_cdf(false);
        capture.set_semantic_cdfs(&capture_slots);
        capture.begin_semantic_sink();
        emit_semantic_fixture(&mut capture, &mut capture_cdfs);
        let tokens = capture.take_semantic();

        let mut reference_cdfs = Cdfs::new(crate::coef_q::qcat(140));
        let mut reference = OdEcEncoder::new().with_updating_cdf(updating_cdf);
        emit_semantic_fixture(&mut reference, &mut reference_cdfs);
        let expected = reference.done();

        let mut packed_cdfs = Cdfs::new(crate::coef_q::qcat(140));
        let mut packed_slots = packed_cdfs.semantic_slots();
        let mut packed = OdEcEncoder::new().with_updating_cdf(updating_cdf);
        packed.replay_semantic(&tokens, &mut packed_slots);

        assert_eq!(packed.done(), expected, "updating_cdf={updating_cdf}");
        assert_eq!(packed_cdfs.skip, reference_cdfs.skip);
        assert_eq!(packed_cdfs.part_split, reference_cdfs.part_split);
    }
}

#[test]
fn angle_shortlist_keeps_predictions_paired_with_deltas() {
    fn check<const N: usize>(dim: usize) {
        let w = 64;
        let src = std::array::from_fn(|p| {
            (0..w * w)
                .map(|i| ((i * 37 + i / w * 53 + p * 17) & 1023) as u16)
                .collect()
        });
        let mut tile = LossyTile::new(Chroma::I444, 140, 10, w, w, &src, QmLevels::FLAT);
        tile.ctx.recon.clone_from(&src);
        for mode in V_PRED..=VERT_LEFT_PRED {
            for budget in 0..=2 {
                let (mut a, mut b, mut scratch) = ([0; N], [0; N], [0; N]);
                let mut preds = [&mut a, &mut b, &mut scratch];
                let deltas = tile
                    .rank_angle_deltas::<N>(mode, 8, 8, dim, dim, true, true, budget, &mut preds);
                assert_eq!(deltas.len(), budget);
                let mut previous = 0;
                for (i, &delta) in deltas.iter().enumerate() {
                    let mut expected = [0; N];
                    tile.k.intrapred.predict_nd_ad(
                        mode,
                        delta,
                        &tile.ctx.recon[0],
                        w,
                        8,
                        8,
                        dim,
                        dim,
                        true,
                        true,
                        w,
                        w,
                        tile.luma_filter_type(8, 8),
                        &mut expected,
                        10,
                    );
                    assert_eq!(
                        *preds[i], expected,
                        "{dim} mode={mode} budget={budget} delta={delta}"
                    );
                    let score =
                        tile.k
                            .rd
                            .satd_sad_proxy(tile.src_blk(0, 8, 8, dim, dim), &expected, dim);
                    assert!(score >= previous);
                    previous = score;
                }
            }
        }
    }
    check::<64>(8);
    check::<256>(16);
    check::<1024>(32);
}

#[test]
fn ordinary_luma_syntax_includes_directional_zero_delta() {
    let src = std::array::from_fn(|_| vec![128; 64 * 64]);
    let tile = LossyTile::new(Chroma::I444, 140, 8, 64, 64, &src, QmLevels::FLAT);
    for dim in [8, 16, 32] {
        for mode in V_PRED..=VERT_LEFT_PRED {
            for delta in -3..=3 {
                let expected = tile.mode_bits(8, 8, mode)
                    + cdf_cost(
                        &tile.dcdf().angle_delta[mode - V_PRED],
                        (delta + 3) as usize,
                    );
                assert_eq!(
                    tile.luma_intra_syntax_bits(8, 8, dim, dim, mode, delta),
                    expected
                );
            }
        }
        let dc = tile.mode_bits(8, 8, DC_PRED)
            + cdf_cost(&tile.dcdf().filter_intra[av1_block_size_index(dim, dim)], 0);
        assert_eq!(tile.luma_intra_syntax_bits(8, 8, dim, dim, DC_PRED, 0), dc);
        assert_eq!(
            tile.luma_intra_syntax_bits(8, 8, dim, dim, PAETH_PRED, 0),
            tile.mode_bits(8, 8, PAETH_PRED)
        );
    }
    assert_eq!(
        tile.luma_intra_syntax_bits(8, 8, 4, 4, V_PRED, 0),
        tile.mode_bits(8, 8, V_PRED)
    );
}

#[test]
fn global_angles_recover_a_family_pruned_at_delta_zero() {
    fn check<const N: usize>(dim: usize) {
        let w = 64;
        let recon: Vec<u16> = (0..w * w)
            .map(|i| ((i * 37 + i / w * 53 + (i ^ (i / w)) * 11) & 255) as u16)
            .collect();
        let dispatch = IntraPredDispatch::selected();
        let mut recovered = false;
        for mode in V_PRED..=VERT_LEFT_PRED {
            for delta in [-3, -2, -1, 1, 2, 3] {
                let mut target = [0; N];
                dispatch.predict_nd_ad(
                    mode,
                    delta,
                    &recon,
                    w,
                    8,
                    8,
                    dim,
                    dim,
                    true,
                    true,
                    w,
                    w,
                    false,
                    &mut target,
                    8,
                );
                let mut src = std::array::from_fn(|_| recon.clone());
                for y in 0..dim {
                    for x in 0..dim {
                        src[0][(8 + y) * w + 8 + x] = target[y * dim + x] as u16;
                    }
                }
                let mut tile = LossyTile::new(Chroma::I444, 140, 8, w, w, &src, QmLevels::FLAT)
                    .with_speed(Speed::ExtraSlow);
                tile.ctx.recon[0].clone_from(&recon);
                let nominal = tile.rank_luma_modes::<N>(nd_modes(), 8, 8, dim, dim, true, true, 1);
                if nominal.contains(&mode) {
                    continue;
                }
                let candidates =
                    tile.rank_luma_candidates::<N>(nd_modes(), 8, 8, dim, dim, true, true, 1);
                assert_eq!(candidates.len(), 3);
                assert_eq!(candidates[0], (nominal[0], 0));
                if candidates.contains(&(mode, delta)) {
                    recovered = true;
                    break;
                }
            }
            if recovered {
                break;
            }
        }
        assert!(
            recovered,
            "no pruned angular family recovered for {dim}x{dim}"
        );
    }
    check::<64>(8);
    check::<256>(16);
    check::<1024>(32);
}

#[test]
fn directional_top_k_keeps_three_lowest_costs() {
    let mut top = DirectionalTopK::new();
    for (mode, cost) in [(1, 50), (2, 10), (3, 30), (4, 20), (5, 40)] {
        top.insert(mode, cost);
    }
    assert!(top.contains(2));
    assert!(top.contains(3));
    assert!(top.contains(4));
    assert!(!top.contains(1));
    assert!(!top.contains(5));
}

#[test]
fn satd_sad_proxy_is_zero_only_for_equal_blocks() {
    let rd = crate::rd_sse::RdDispatch::selected();
    let src = [7u16; 16];
    let blk = crate::rd_sse::SrcBlock::new(&src, 4, 0, 0, 4, 4, 4, 4);
    let pred = [7i32; 16];
    assert_eq!(rd.satd_sad_proxy(blk, &pred, 4), 0);
    let mut pred2 = pred;
    pred2[5] += 3;
    assert!(rd.satd_sad_proxy(blk, &pred2, 4) > 0);
}

#[test]
fn moment_grid_matches_independent_4x4_blocks() {
    const STRIDE: usize = 23;
    let plane: Vec<i32> = (0..STRIDE * 19)
        .map(|i| ((i * 37 + i / STRIDE * 11) & 1023) as i32)
        .collect();
    let (px, py) = (3, 2);
    let cells = block_moment_grid_16x16(&plane, STRIDE, px, py);
    for cy in 0..4 {
        for cx in 0..4 {
            assert_eq!(
                cells[cy * 4 + cx],
                block_moments_i32(&plane, STRIDE, px + cx * 4, py + cy * 4, 4, 4),
            );
        }
    }
}

/// `precompute_aq_grid` must reproduce the serial `aq_begin_sb` accumulator
/// walk bit-exactly — every cell's qindex, signaled steps, and the resulting
/// quantizer state — on a padded (non-64-aligned) frame with mixed
/// dark/flat/textured content, with and without Variance Boost, so the
/// wavefront can consume cells out of raster order.
#[test]
fn aq_grid_matches_serial() {
    // 200x136 luma: 4x3 SB grid with 8px right / bottom partial SBs.
    let (w, h) = (200usize, 136usize);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            // dark gradient + texture patches + flat bands
            let base = ((r * 255) / h) as i32 / 3;
            let tex = if (r / 32 + c / 32) % 3 == 0 {
                (((r * 7 + c * 13) % 53) as i32) - 26
            } else {
                0
            };
            y[r * w + c] = (base + tex).clamp(0, 255) as u16;
        }
    }
    let src = [y, vec![128; w * h], vec![128; w * h]];
    for vb_enabled in [false, true] {
        for base_q in [60u8, 140, 220] {
            let mk = || {
                let mut t = LossyTile::new(Chroma::I444, base_q, 8, w, h, &src, QmLevels::FLAT);
                let ref_act = tile_ref_activity(&t.cfg.src[0], t.cfg.w, t.cfg.w, t.cfg.h);
                let vb = VarianceBoost {
                    enabled: vb_enabled,
                    octile: 6,
                    strength: 1.0,
                    boost_only: false,
                    dark: DarkAq::on(),
                    base_shift: 0,
                    qm: QmLevels::FLAT,
                    mono_large: false,
                };
                t.enable_aq(base_q, ref_act, &vb);
                t
            };
            let mut serial = mk();
            let grid_tile = mk();
            let grid = grid_tile.precompute_aq_grid();
            let (rows, cols) = (h.div_ceil(64), w.div_ceil(64));
            assert_eq!(grid.len(), rows * cols);
            let mut i = 0;
            for sb_y in (0..h).step_by(64) {
                for sb_x in (0..w).step_by(64) {
                    serial.aq_begin_sb(sb_x, sb_y);
                    let cell = grid[i];
                    assert_eq!(
                        cell.newq as i32, serial.ctx.aq.cur_qidx,
                        "qidx sb {i} vb {vb_enabled} q {base_q}"
                    );
                    assert_eq!(
                        cell.steps, serial.ctx.aq.pending,
                        "steps sb {i} vb {vb_enabled} q {base_q}"
                    );
                    i += 1;
                }
            }
        }
    }
}

#[test]
fn skipped_whole_64_does_not_consume_delta_q() {
    let src = [
        vec![128u16; 64 * 64],
        vec![128u16; 32 * 32],
        vec![128u16; 32 * 32],
    ];
    let make_tile = || {
        let mut tile = LossyTile::new(Chroma::I420, 100, 8, 64, 64, &src, QmLevels::FLAT);
        tile.ctx.aq.enabled = true;
        tile.ctx.aq.prev_qidx = 100;
        tile.ctx.aq.cur_qidx = 98;
        tile.ctx.aq.read_deltas = true;
        tile.ctx.aq.pending = -2;
        tile
    };

    let mut skipped = make_tile();
    skipped.code_skip_and_sb_tokens_64(true, 0);
    assert!(
        skipped.ctx.aq.read_deltas,
        "a skipped whole superblock must return before read_delta_qindex"
    );
    assert_eq!(skipped.ctx.aq.cur_qidx, 100, "skipped SB must roll AQ back");

    let mut coded = make_tile();
    coded.code_skip_and_sb_tokens_64(false, 0);
    assert!(
        !coded.ctx.aq.read_deltas,
        "a non-skipped whole superblock must consume its delta-Q"
    );
    assert_eq!(
        coded.ctx.aq.cur_qidx, 98,
        "coded SB keeps its signaled AQ qindex"
    );
}

/// Dark protection over a full 64×64 SB of the given i32 plane, with the AV1
/// 8-bit normalization scale for bit depth `bd`.
fn dark_prot(d: &DarkAq, base_q: i32, yp: &[i32], bd: u8) -> i32 {
    let scale = 1.0 / (1u32 << (bd - 8)) as f32;
    crate::aq_common::dark_protection(d, base_q, yp, 64, 0, 0, 64, 64, scale)
}

#[test]
fn dark_protection_targets_dark_structure_only() {
    // 64x64 SBs at three luma levels, each with a coherent striped texture (survives
    // the 2x downsample, so it registers as structure not noise). Ported from
    // `av2/aq.rs`, on the i32 luma plane the AV1 encoder uses.
    let build = |mean: i32| -> Vec<i32> {
        let mut p = vec![0i32; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                // 4px-wide stripes: strong at both scales.
                p[r * 64 + c] = mean + if (c / 4) % 2 == 0 { 12 } else { -12 };
            }
        }
        p
    };
    let d = DarkAq::on(); // enabled, min_q = 90 (2026-07-24 gate ship)
    let dark = build(36);
    let bright = build(180);
    let flat_dark = vec![24i32; 64 * 64]; // dark but no structure

    let base_q = 190; // in the gated range (>= 150)
    let d_dark = dark_prot(&d, base_q, &dark, 8);
    let d_bright = dark_prot(&d, base_q, &bright, 8);
    let d_flat = dark_prot(&d, base_q, &flat_dark, 8);
    assert!(
        d_dark > 0,
        "dark structured SB should be protected, got {d_dark}"
    );
    assert_eq!(d_bright, 0, "bright structured SB must not be protected");
    assert_eq!(
        d_flat, 0,
        "dark flat SB (no structure) must not be protected"
    );

    // Gate: disabled below min_q (high quality / low qindex).
    assert_eq!(dark_prot(&d, 60, &dark, 8), 0, "gated out below min_q");
    // Disabled config is always inert.
    assert_eq!(
        dark_prot(&DarkAq::off(), base_q, &dark, 8),
        0,
        "disabled dark AQ must not protect"
    );
}

#[test]
fn dark_protection_bitdepth_normalized() {
    // A 10-bit dark structured SB (all values << 2) must score the SAME protection
    // as its 8-bit counterpart: the stats normalize native depth to 8-bit range.
    let build = |mean: i32, shift: u32| -> Vec<i32> {
        let mut p = vec![0i32; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                p[r * 64 + c] = (mean + if (c / 4) % 2 == 0 { 12 } else { -12 }) << shift;
            }
        }
        p
    };
    let d = DarkAq::on();
    let base_q = 190;
    let p8 = build(36, 0);
    let p10 = build(36, 2);
    let d8 = dark_prot(&d, base_q, &p8, 8);
    let d10 = dark_prot(&d, base_q, &p10, 10);
    assert!(d8 > 0);
    assert_eq!(d8, d10, "dark protection must be bit-depth invariant");
}

#[test]
fn chroma_rd_can_favor_a_split_on_color_detail() {
    let mut src = [
        vec![128u16; 32 * 32],
        vec![128u16; 32 * 32],
        vec![128u16; 32 * 32],
    ];
    for y in 0..16 {
        for x in 0..16 {
            let quadrant = (x >= 8) as usize + 2 * (y >= 8) as usize;
            src[1][y * 32 + x] = [24, 232, 216, 40][quadrant];
            src[2][y * 32 + x] = [224, 48, 32, 240][quadrant];
        }
    }
    let tile = LossyTile::new(Chroma::I444, 160, 8, 32, 32, &src, QmLevels::FLAT);
    let none = tile.rd_cost_chroma_partition(0, 0, 16, Part16::None, 1.0, false);
    let split = tile.rd_cost_chroma_partition(0, 0, 16, Part16::Split, 1.0, false);
    assert!(
        split < none,
        "four color-homogeneous chroma blocks should beat one mixed block: split={split}, none={none}"
    );
}

#[test]
fn uv_rate_matches_emitted_cdf_transaction() {
    let src = [
        vec![128u16; 16 * 16],
        vec![128u16; 16 * 16],
        vec![128u16; 16 * 16],
    ];
    let tile = LossyTile::new(Chroma::I444, 120, 8, 16, 16, &src, QmLevels::FLAT);
    let y_mode = PAETH_PRED;
    assert_eq!(
        tile.uv_mode_bits(y_mode, DC_PRED, None),
        cdf_cost(&tile.dcdf().uv_mode[13 + y_mode], DC_PRED)
    );
    let a = [2, -3];
    let expected = cdf_cost(&tile.dcdf().uv_mode[13 + y_mode], CFL_PRED)
        + cdf_cost(&tile.dcdf().cfl_sign, 6)
        + cdf_cost(&tile.dcdf().cfl_alpha[4], 1)
        + cdf_cost(&tile.dcdf().cfl_alpha[2], 2);
    assert_eq!(tile.uv_mode_bits(y_mode, CFL_PRED, Some(a)), expected);
}

#[test]
fn local_chroma_weight_responds_to_local_color_detail() {
    let flat = [
        vec![128u16; 16 * 16],
        vec![128u16; 16 * 16],
        vec![128u16; 16 * 16],
    ];
    let mut color = flat.clone();
    for y in 0..16 {
        for x in 0..16 {
            color[1][y * 16 + x] = if x < 8 { 24 } else { 232 };
            color[2][y * 16 + x] = if y < 8 { 224 } else { 32 };
        }
    }
    let flat_tile = LossyTile::new(Chroma::I444, 120, 8, 16, 16, &flat, QmLevels::FLAT);
    let color_tile = LossyTile::new(Chroma::I444, 120, 8, 16, 16, &color, QmLevels::FLAT);
    assert!(
        color_tile.chroma_partition_weight_at(0, 0, 16, 16)
            > flat_tile.chroma_partition_weight_at(0, 0, 16, 16)
    );
}

#[test]
fn block64_mode_trial_restores_reconstruction() {
    let mut src = [
        vec![0u16; 64 * 64],
        vec![128u16; 64 * 64],
        vec![128u16; 64 * 64],
    ];
    for y in 0..64 {
        for x in 0..64 {
            src[0][y * 64 + x] = ((x * 5 + y * 3 + (x ^ y)) & 255) as u16;
        }
    }
    let mut tile = LossyTile::new(Chroma::I444, 160, 8, 64, 64, &src, QmLevels::FLAT)
        .with_speed(Speed::Medium);
    let before = tile.ctx.recon[0].clone();
    let (_, _, cost, _, _) = tile.rd_pick_luma64(0, 0, true, false, 1.0);
    assert!(cost.is_finite());
    assert_eq!(tile.ctx.recon[0], before);
}
