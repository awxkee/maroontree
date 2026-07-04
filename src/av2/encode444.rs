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
use crate::av2::cfl::{cfl_decide_64, cfl_prediction};

/// Resolve the requested tile grid (`Tuning::tile_cols/rows`) into (log2_cols,
/// log2_rows). Rounds each request up to a power of two and clamps to the available
/// superblock count. Returns `None` for a single tile (1x1, or fewer SBs than tiles).
pub(super) fn tile_grid_for(
    tile_cols: usize,
    tile_rows: usize,
    width: usize,
    height: usize,
) -> Option<(usize, usize)> {
    let log2 = |n: usize| {
        let mut k = 0;
        while (1usize << k) < n {
            k += 1;
        }
        k
    };
    let (mut lc, mut lr) = (log2(tile_cols.max(1)), log2(tile_rows.max(1)));
    // Clamp to the number of superblocks that `tile_starts` can actually place on each
    // axis. That loop bounds on `full_sb` (the FLOOR superblock count, mi >> 4), not the
    // ceil count, so a frame whose last superblock is partial (e.g. 484 px = 7 full + 1
    // partial SB) can hold fewer uniform tiles than `div_ceil(64)` suggests. Signaling
    // log2 against the ceil count there made the frame header advertise more tiles than
    // were emitted (8 vs 7 rows for 484 px), so avmdec read past the tile data and
    // reported a corrupt/truncated tile size. Clamp against `full_sb` to keep the
    // signaled grid in lock-step with the emitted tiles.
    let full_sb = |dim_px: usize| (((dim_px + 7) & !7) / 4) >> 4;
    let (sbc, sbr) = (full_sb(width), full_sb(height));
    while (1usize << lc) > sbc && lc > 0 {
        lc -= 1;
    }
    while (1usize << lr) > sbr && lr > 0 {
        lr -= 1;
    }
    if lc == 0 && lr == 0 {
        return None;
    }
    Some((lc, lr))
}

/// Copy the `tw x th` sub-plane at `(x0, y0)` out of a `width`-stride plane.
pub(super) fn extract_subplane(
    p: &[f32],
    width: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<f32> {
    let mut o = vec![0f32; tw * th];
    for r in 0..th {
        o[r * tw..r * tw + tw]
            .copy_from_slice(&p[(y0 + r) * width + x0..(y0 + r) * width + x0 + tw]);
    }
    o
}

/// Tile SB-start boundaries EXACTLY as the decoder computes them
/// (av2_calculate_tile_cols/rows, uniform spacing, seq SB == coding SB so scale_sb=0):
/// base = sb_count >> log2, the first `extra` tiles get one extra SB. Yields widths like
/// 3,3,2,2 for 10 cols / 4 tiles — NOT a uniform ceil, which would desync the decode.
/// Tile start SBs for one axis, matching the decoder's `av2_calculate_tile_cols/rows`
/// (av2/common/tile_common.c) EXACTLY. The base tile size and remainder come from
/// `full_sb = mi_count >> mib_size_log2` — the FLOOR of the SB count derived from the
/// (8-px-aligned) MI count — while the last tile extends to the ceil SB count. Using
/// the ceil count for base/extra mis-distributes the remainder SB whenever a frame has
/// a partial right/bottom SB (e.g. 4000px → floor 62 vs ceil 63), desyncing the tile
/// grid for non-64-aligned frames larger than one SB per tile.
fn tile_starts(dim_px: usize, log2: usize) -> Vec<usize> {
    let n = 1usize << log2;
    let mi = ((dim_px + 7) & !7) / 4; // 8-px-aligned MI count
    let full_sb = mi >> 4; // floor(mi / 16) — decoder's full_sb_cols/rows
    let ceil_sb = dim_px.div_ceil(64); // sentinel / loop bound (seq_sb)
    let base = full_sb >> log2;
    let mut extra = full_sb as isize - ((base << log2) as isize);
    let mut starts = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    // Loop bound is `full_sb` (floor SB count), matching the decoder's uniform-tiling
    // loop `while sbx < fsbw` (obu.rs). The sentinel below extends the last tile to
    // `ceil_sb`. Bounding on `ceil_sb` here instead would emit one extra tile whenever
    // the per-tile size is 1 SB and the frame has a partial last SB (e.g. 456 px = 7
    // full + 1 partial SB at log2=3: decoder makes 7 tiles, not 8), desyncing the grid.
    while start < full_sb && i < n {
        starts.push(start);
        start += base + if extra > 0 { 1 } else { 0 };
        if extra > 0 {
            extra -= 1;
        }
        i += 1;
    }
    starts.push(ceil_sb); // sentinel: tile k spans [starts[k], starts[k+1])
    starts
}

/// One 4:4:4 chroma leaf (U then V), DC-predicted unless MHCCP wins. Mirrors the
/// per-arm inline path (project_scan + reconstruct_chroma) so the eligible
/// partition arms can share it. Returns (u_present, v_present).
#[allow(clippy::too_many_arguments)]
fn code_444_chroma_leaf(
    enc: &mut RangeEncoder,
    recy: &[f32],
    recu: &mut [f32],
    recv: &mut [f32],
    up: &[f32],
    vp: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    cw: usize,
    ch: usize,
    basis: &Basis,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
    u_skip_row: &[u16],
    _qc: usize,
    neutral: f32,
    qstep: i32,
    ua: i32,
    ul: i32,
    va: i32,
    vl: i32,
    have_top: bool,
    have_left: bool,
    lambda: f64,
    mhccp_on: bool,
    bd: i32,
) -> (bool, bool) {
    // MHCCP vs DC incumbent (4:4:4 => ssx = ssy = false).
    let mhccp_choice = if mhccp_on && cfl::is_mhccp_allowed(cw / 4, ch / 4, false, false) {
        let dcu = dc_pred_rect(recu, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let dcv = dc_pred_rect(recv, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let mut suf = vec![0f32; cw * ch];
        let mut svf = vec![0f32; cw * ch];
        for r in 0..ch {
            let b = (sb_y + r) * pw + sb_x;
            for c in 0..cw {
                suf[r * cw + c] = up[b + c];
                svf[r * cw + c] = vp[b + c];
            }
        }
        crate::av2::cfl::mhccp_eval_leaf(
            recy, pw, recu, recv, pw, sb_y, sb_x, sb_y, sb_x, cw, ch, false, false, have_top,
            have_left, &suf, &svf, dcu, dcv, basis, qstep, lambda, scan, bd,
        )
    } else {
        None
    };
    let mh = mhccp_choice.as_ref().and_then(|c| c.mhccp.as_ref());
    if let Some(mh) = mh {
        enc.cfl_use = true;
        enc.mhccp_use = true;
        enc.mhccp_dir = mh.mh_dir;
        enc.mhccp_size_group = mh.size_group;
        enc.uv_mode = 0;
    }
    let win = mh.is_some();
    // U plane.
    let levu = if win {
        let ch_choice = mhccp_choice.as_ref().unwrap();
        let mut ru = vec![0f32; cw * ch];
        for j in 0..ch {
            for i in 0..cw {
                ru[j * cw + i] =
                    up[(sb_y + j) * pw + sb_x + i] - ch_choice.pred_u[j * cw + i] as f32;
            }
        }
        let levu = basis.project_scan(&ru, 0.0, scan);
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&ch_choice.pred_u, &levu, qstep, scan, cw, ch, bd),
        );
        levu
    } else {
        let predu = dc_pred_rect(recu, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let levu = basis.project_scan(
            &get_residual_rect(up, pw, sb_y, sb_x, cw, ch, predu),
            0.0,
            scan,
        );
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma(predu, &levu, qstep, scan, cw, ch, bd),
        );
        levu
    };
    // V plane.
    let levv = if win {
        let ch_choice = mhccp_choice.as_ref().unwrap();
        let mut rv = vec![0f32; cw * ch];
        for j in 0..ch {
            for i in 0..cw {
                rv[j * cw + i] =
                    vp[(sb_y + j) * pw + sb_x + i] - ch_choice.pred_v[j * cw + i] as f32;
            }
        }
        let levv = basis.project_scan(&rv, 0.0, scan);
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&ch_choice.pred_v, &levv, qstep, scan, cw, ch, bd),
        );
        levv
    } else {
        let predv = dc_pred_rect(recv, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let levv = basis.project_scan(
            &get_residual_rect(vp, pw, sb_y, sb_x, cw, ch, predv),
            0.0,
            scan,
        );
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma(predv, &levv, qstep, scan, cw, ch, bd),
        );
        levv
    };
    let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
    let cbwl = (cw.min(32) as f32).log2() as i32;
    let u_skip = u_skip_row[(6 + ua + ul) as usize] as u32;
    encode_chroma_block_rect_w(enc, &uc, u_skip, true, scan, eob_cdf, eob_hi, area, cbwl);
    let up_ = uc.iter().any(|&(_, l)| l != 0);
    let v_skip = (6 * (up_ as i32) + va + vl) as u32;
    encode_chroma_block_rect_w(enc, &vc, v_skip, false, scan, eob_cdf, eob_hi, area, cbwl);
    (up_, vc.iter().any(|&(_, l)| l != 0))
}

/// Luma tile regions in raster order: `(x0, y0, tw, th)` (all luma-pixel units, x0/y0 on
/// 64-px SB boundaries). Tile column/row boundaries are in superblock units and so are
/// identical across chroma formats.
pub(super) fn tile_specs(
    width: usize,
    height: usize,
    log2c: usize,
    log2r: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let col_starts = tile_starts(width, log2c);
    let row_starts = tile_starts(height, log2r);
    let mut specs = Vec::new();
    for tr in 0..row_starts.len() - 1 {
        let (r0, r1) = (row_starts[tr], row_starts[tr + 1]);
        let (y0, th) = (r0 * 64, (r1 * 64).min(height) - r0 * 64);
        for tc in 0..col_starts.len() - 1 {
            let (c0, c1) = (col_starts[tc], col_starts[tc + 1]);
            let (x0, tw) = (c0 * 64, (c1 * 64).min(width) - c0 * 64);
            specs.push((x0, y0, tw, th));
        }
    }
    specs
}

/// Wrap already-encoded per-tile byte streams (raster order) into a single multi-tile
/// frame: frame header with the tile grid, `tsb`-byte size prefixes before every tile
/// but the last, then the TD/SEQ/FRAME OBUs. Format-agnostic — chroma signalling lives
/// in `config`/`chroma_format`.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_multitile(
    config: &Config,
    sig_w: usize,
    sig_h: usize,
    disp_w: usize,
    disp_h: usize,
    color: &Cicp,
    log2c: usize,
    log2r: usize,
    bit_depth: u8,
    chroma_format: ChromaFormat,
    tiles_bytes: &[Vec<u8>],
) -> Av2Frame {
    let n = tiles_bytes.len();
    // Fixed TileSizeBytes = 4 (matches the reference encoders; always sufficient).
    let tsb = 4usize;
    let mut frame = frame_header(config, sig_w as u32, sig_h as u32, (log2c, log2r, tsb));
    for (i, t) in tiles_bytes.iter().enumerate() {
        if i + 1 < n {
            let v = t.len() - 1; // - AV2_MIN_TILE_SIZE_BYTES (=1)
            for b in 0..tsb {
                frame.push(((v >> (8 * b)) & 0xff) as u8);
            }
        }
        frame.extend(t);
    }
    let mut data = vec![];
    data.extend(obu(2, &[]));
    data.extend(obu(1, &sequence_header(config, sig_w as u32, sig_h as u32)));
    data.extend(obu(4, &frame));
    Av2Frame {
        data,
        width: disp_w,
        height: disp_h,
        // Coded size = the OBU-signaled size. When it exceeds the display size (the
        // padded-tiling fallback for non-boundary-exact frames) the AVIF muxer crops
        // via a `clap` box.
        coded_width: sig_w,
        coded_height: sig_h,
        bit_depth,
        color: *color,
        chroma_format,
    }
}

impl Av2Encoder {
    /// RD decision (mirrors AVM `rd_pick_partition` NONE-vs-SPLIT on the chroma cost):
    /// returns true when coding the 64x64 chroma as 4x TX_32X32 has lower RD than one
    /// TX_64X64. The 64x64 transform zeros the high-frequency 3/4 of coefficients, so on
    /// detailed chroma the split reconstructs far more accurately; the split pays the
    /// partition-signal bits plus per-leaf overhead. Distortion is real reconstructed SSE
    /// and rate is the (lambda-weighted) coefficient + signal bit estimate, matching the
    /// `RDCOST(rate,dist)` comparison AVM performs. DC prediction is used on both sides as
    /// the common baseline (CfL/MHCCP only shift both costs together).
    #[allow(clippy::too_many_arguments)]
    fn chroma_split_wins_444(
        &self,
        recu: &[f32],
        recv: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        bases: &crate::av2::proj::Bases,
        sb_qstep: i32,
        sb_resid_scale: f32,
        neutral: f32,
        _qc: usize,
    ) -> bool {
        use crate::av2::helpers::{dc_pred, get_residual};
        let bd = self.bit_depth as i32;
        let scan = &tables::SCAN;
        let lambda = crate::av2::leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
        // Cost of one plane coded as a single 64x64 transform (DC pred).
        let whole_plane = |rec: &[f32], src: &[f32]| -> (f64, f64) {
            let pred = dc_pred(rec, pw, sb_y, sb_x, 64, neutral);
            let lev = bases.chroma444.project(
                &crate::av2::aq::scale_resid(
                    &get_residual(src, pw, sb_y, sb_x, 64, pred),
                    sb_resid_scale,
                ),
                0.0,
            );
            let recon =
                crate::av2::itx422::reconstruct_chroma(pred, &lev, sb_qstep, scan, 64, 64, bd);
            let mut sse = 0f64;
            for r in 0..64 {
                let b = (sb_y + r) * pw + sb_x;
                for c in 0..64 {
                    let d = (src[b + c] - recon[r * 64 + c]) as f64;
                    sse += d * d;
                }
            }
            let bits: f64 = lev.iter().filter(|&&l| l != 0.0).count() as f64 * 4.0;
            (sse, bits)
        };
        // Cost of one plane coded as 4x 32x32 transforms (DC pred per sub-block).
        let split_plane = |rec: &[f32], src: &[f32]| -> (f64, f64) {
            let mut sse = 0f64;
            let mut bits = 0f64;
            for (dr, dc) in [(0usize, 0usize), (0, 32), (32, 0), (32, 32)] {
                let (by, bx) = (sb_y + dr, sb_x + dc);
                let pred = dc_pred(rec, pw, by, bx, 32, neutral);
                let lev = bases.chroma420.project(
                    &crate::av2::aq::scale_resid(
                        &get_residual(src, pw, by, bx, 32, pred),
                        sb_resid_scale,
                    ),
                    0.0,
                );
                let recon =
                    crate::av2::itx422::reconstruct_chroma(pred, &lev, sb_qstep, scan, 32, 32, bd);
                for r in 0..32 {
                    let b = (by + r) * pw + bx;
                    for c in 0..32 {
                        let d = (src[b + c] - recon[r * 32 + c]) as f64;
                        sse += d * d;
                    }
                }
                bits += lev.iter().filter(|&&l| l != 0.0).count() as f64 * 4.0;
            }
            (sse, bits)
        };
        let (whu_sse, whu_bits) = whole_plane(recu, up);
        let (whv_sse, whv_bits) = whole_plane(recv, vp);
        let (spu_sse, spu_bits) = split_plane(recu, up);
        let (spv_sse, spv_bits) = split_plane(recv, vp);
        // Partition signal: do_split(1) + do_square_split(1) at 64x64, and each of the
        // four 32x32 children codes do_split(0). ~1 bit each is a conservative estimate.
        let split_signal_bits = 6.0;
        // Luma is coded identically (4x TX_32X32) in both the whole-64 fast path and the
        // split, so it cancels and is excluded from this chroma-only comparison.
        // SS2-calibrated 4x: whole-64 chroma drops the >32x32 spectrum; SSE underweights it.
        let j_whole = (whu_sse + whv_sse) * 4.0 + lambda * (whu_bits + whv_bits);
        let j_split = (spu_sse + spv_sse) + lambda * (spu_bits + spv_bits + split_signal_bits);
        j_split < j_whole
    }

    /// Encode a 4:4:4 YCbCr still. `y`, `cb`, `cr` are full-resolution
    /// (`width × height`). Luma is four 32x32 transform units per 64x64 superblock;
    /// each chroma plane is one 64x64 transform per superblock.
    pub fn encode_yuv444<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        if self.base_q_idx == 0 {
            return self.encode_yuv444_lossless(planar_image, color, self.threads);
        }
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (yf, cbf, crf) = (to_plane(y), to_plane(cb), to_plane(cr));
        let (pw, ph) = (sb_align(width), sb_align(height));
        let config = self.config(Layout::I444);
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_444_tiled(
                &yf,
                &cbf,
                &crf,
                width,
                height,
                &config,
                color,
                log2c,
                log2r,
                self.threads,
            ));
        }
        let enc = self.encode_444_core(&yf, &cbf, &crf, width, height);
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// SB-loop core for one 4:4:4 region (a whole frame, or one tile treated as a
    /// sub-frame). Returns the finished entropy coder; assembly (frame header/OBU,
    /// or multi-tile concatenation) happens in the caller.
    fn encode_444_core(
        &self,
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        width: usize,
        height: usize,
    ) -> RangeEncoder {
        let bases = &self.bases;
        // Encode-time tuning (was AV2_* env). Captured once per region.
        let rdoq_lambda = self.tune.rdoq_lambda;
        let part_lambda_c = self.tune.part_lambda_c;
        let txpart = self.tune.txpart;
        let (pw, ph) = (sb_align(width), sb_align(height));
        // Native-size 444: boundary-safe non-aligned sizes can signal real W×H so the
        // decoder reconstructs the full padded SB and crops — no AVIF clap box needed.
        let native_mi = lossy_native_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        let yp = pad_plane(y, width, height, pw, ph);
        let up = pad_plane(cb, width, height, pw, ph);
        let vp = pad_plane(cr, width, height, pw, ph);

        let _layout = Layout::I444;
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pw * ph];
        let mut recv = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = false;
        enc.mhccp_ssy = false;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        // Per-mi chroma neighbor coeff-presence (mirrors the luma above/left arrays):
        // `*_above[mi_col]` / `*_left[mi_row]` hold whether the most recent TU covering
        // that column/row had U/V coeffs. Per-mi (not per-SB) so that multiple chroma
        // TUs within one SB — e.g. the two vertically stacked 8×32 residue-2 leaves —
        // see each other as neighbors.
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        // Per-mi CfL-usage neighbors for get_cfl_ctx: `cfl_above[mi_col]` / `cfl_left
        // [mi_row]` hold whether the chroma block covering that column/row used CfL
        // (uv_mode == UV_CFL_PRED). is_cfl context = above_used + left_used (0..2).
        let mut cfl_above = vec![0i32; tmc as usize + 16];
        let mut cfl_left = vec![0i32; tmr as usize + 16];
        let qstep_i = quant::qstep(self.base_q_idx as u32) as i32;
        // Bottom-edge force-split: the last SB row is 32 px tall in frame, so each
        // 64X64 force-splits HORZ (implied, no bits) into a top 64X32 leaf coded by
        // the partition leaf path. Partition context `above_pctx` persists down
        // columns; `left_pctx` is len-16 and reset per SB row.
        // Force-split partition walk. When any edge residue is 6 or 8 the right/bottom
        // SBs split into 32-family leaves (32X64 / 64X32 / 32X32); otherwise every SB is
        // a whole 64X64. The walk drives `sb_partition_ops`, which also maintains the
        // partition contexts (`above_pctx` down columns, `left_pctx` reset per SB row).
        let needs_partition = native_mi.is_some() && lossy_needs_partition(width, height);
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = vec![0u8; 16];
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(&yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        );

        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                // Fast-path SB chroma context at the SB-origin mi (col*16, row*16).
                let (fmr, fmc) = (row * 16, col * 16);
                let ua = if fmr > 0 {
                    u_above[fmc..fmc + 16].iter().any(|&x| x != 0) as i32
                } else {
                    0
                };
                let ul = if fmc > 0 {
                    u_left[fmr..fmr + 16].iter().any(|&x| x != 0) as i32
                } else {
                    0
                };
                let va = if fmr > 0 {
                    v_above[fmc..fmc + 16].iter().any(|&x| x != 0) as i32
                } else {
                    0
                };
                let vl = if fmc > 0 {
                    v_left[fmr..fmr + 16].iter().any(|&x| x != 0) as i32
                } else {
                    0
                };

                // Helper closures capture nothing mutable; chroma coeff encode is inlined
                // per leaf because basis/size/skip-table differ.
                // Per-SB: decide whether to code this full interior 64x64 as a 4x32x32
                // square split (chroma-motivated, see chroma_split_wins_444). Edge SBs and
                // dimension-forced splits keep their existing behaviour.
                // The split relies on the 32x32 intra-leaf coder, which is bit-exact against
                // the reference decoder for base_q_idx <= 97 (quality >= ~62). Below that a
                // pre-existing leaf-coder desync appears (also present in edge-SB leaf paths,
                // independent of this feature), so the split is restricted to the verified
                // range. The chroma-detail plateau this fixes only matters at higher quality,
                // which is exactly the bit-exact range, so no useful gain is lost.
                let split_q_safe = self.base_q_idx <= 97;
                // Edge SBs (incomplete 64x64) require the partition walk; full interior
                // SBs use the fast path even in unaligned frames.
                let sb_walk = needs_partition && !(sb_x + 64 <= width && sb_y + 64 <= height);
                let sb_use_split = !sb_walk
                    && self.tune.chroma_split
                    && split_q_safe
                    && sb_x + 64 <= width
                    && sb_y + 64 <= height
                    && {
                        let (sb_qstep, sb_resid_scale) =
                            aqs.per_sb_probe(&yp, pw, sb_y, sb_x, width, height);
                        self.chroma_split_wins_444(
                            &recu,
                            &recv,
                            &up,
                            &vp,
                            pw,
                            sb_y,
                            sb_x,
                            bases,
                            sb_qstep,
                            sb_resid_scale,
                            neutral,
                            qc,
                        )
                    };
                if !sb_walk && !sb_use_split {
                    // Fast path: whole 64X64 SB. RD-choose luma tx-partition between
                    // SPLIT (4×TX_32X32) and VERT4 (4×TX_16X64), cheap SSE + rate proxy.
                    let (sb_qstep, sb_resid_scale) =
                        aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height);
                    // do_split cdf for this whole-64 PARTITION_NONE, from the real partition
                    // context (12276 in an all-whole-64 frame; differs next to a split SB).
                    let none_do_split_cdf =
                        partition::sb_none_do_split_cdf(row, col, &above_pctx, &left_pctx);
                    let sse_region = |rec: &[f32]| -> f64 {
                        let mut s = 0f64;
                        for r in 0..64 {
                            let b = (sb_y + r) * pw + sb_x;
                            for c in 0..64 {
                                let d = (rec[b + c] - yp[b + c]) as f64;
                                s += d * d;
                            }
                        }
                        s
                    };
                    let rate_proxy = |tus: &[Vec<Coeff>], ovh: f64| -> f64 {
                        let mut bits = 0f64;
                        for tu in tus {
                            bits += ovh;
                            for &(_, l) in tu {
                                if l != 0 {
                                    bits += 2.0 + 2.0 * ((l.unsigned_abs() as f64) + 1.0).log2();
                                }
                            }
                        }
                        bits
                    };
                    let lambda = crate::av2::leaf::part_lambda(qstep_i, part_lambda_c);
                    // ---- SPLIT candidate (existing mode search) ----
                    let (tus_s, mode_idx, _) = encode_luma_sb(
                        &mut recy,
                        &yp,
                        pw,
                        width,
                        height,
                        sb_y,
                        sb_x,
                        &bases.luma,
                        sb_qstep,
                        sb_resid_scale,
                        &tables::SCAN,
                        neutral,
                        qc,
                        rdoq_lambda,
                        self.speed,
                        self.bit_depth as i32,
                        false, // non-directional path
                    );
                    let j_s = sse_region(&recy)
                        + lambda
                            * (rate_proxy(&tus_s, 3.0) + if mode_idx != 0 { 6.0 } else { 0.0 });
                    // partition strategy from tuning (was AV2_TXPART env)
                    // Rect tx-partition (VERT4/HORZ4) is only safe on FULL interior 64x64
                    // SBs: on a partial edge SB the rect strips cross the frame boundary
                    // and the edge-clamped coding desyncs the decoder. Restrict rect
                    // candidates to whole SBs; partial edge SBs fall back to SPLIT.
                    let whole_sb = sb_x + 64 <= width && sb_y + 64 <= height;
                    let want_vert4 = whole_sb
                        && matches!(txpart, TxPart::ThreeWay | TxPart::Rd2 | TxPart::Vert4);
                    let want_horz4 = whole_sb && matches!(txpart, TxPart::ThreeWay | TxPart::Horz4);
                    let force_vert4 = txpart == TxPart::Vert4;
                    let force_horz4 = txpart == TxPart::Horz4;
                    let mut snap_split = [0f32; 64 * 64];
                    let mut snap_best = [0f32; 64 * 64];
                    for r in 0..64 {
                        let b = (sb_y + r) * pw + sb_x;
                        snap_split[r * 64..r * 64 + 64].copy_from_slice(&recy[b..b + 64]);
                    }
                    snap_best.copy_from_slice(&snap_split);
                    let restore = |recy: &mut [f32], snap: &[f32]| {
                        for r in 0..64 {
                            let b = (sb_y + r) * pw + sb_x;
                            recy[b..b + 64].copy_from_slice(&snap[r * 64..r * 64 + 64]);
                        }
                    };
                    #[derive(PartialEq, Debug)]
                    enum Part {
                        Split,
                        Vert4,
                        Horz4,
                    }
                    let mut best = Part::Split;
                    let mut best_j = j_s;
                    let mut tus_v: [Vec<Coeff>; 4] =
                        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                    let mut tus_h: [Vec<Coeff>; 4] =
                        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                    // ---- VERT4 candidate (4× TX_16X64, strips L→R) ----
                    if want_vert4 {
                        for (i, tus_v) in tus_v[..4].iter_mut().enumerate() {
                            let x0 = sb_x + i * 16;
                            let predv = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                x0,
                                16,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma16x64.project_scan(
                                &aq::scale_resid(
                                    &get_residual_rect(&yp, pw, sb_y, x0, 16, 64, predv),
                                    sb_resid_scale,
                                ),
                                0.0,
                                &SCAN16X32,
                            );
                            let pred_flat = [predv; 1024];
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                x0,
                                16,
                                64,
                                &itx422::reconstruct_luma_16x64(
                                    &pred_flat,
                                    &lev,
                                    sb_qstep,
                                    &SCAN16X32,
                                    self.bit_depth as i32,
                                ),
                            );
                            *tus_v = levels_to_coeffs(&lev);
                        }
                        let j_v = sse_region(&recy) + lambda * rate_proxy(&tus_v, 4.0);
                        let take = force_vert4 || j_v < best_j;
                        if take {
                            best = Part::Vert4;
                            best_j = j_v;
                            snap_best.copy_from_slice(&{
                                let mut s = [0f32; 64 * 64];
                                for r in 0..64 {
                                    let b = (sb_y + r) * pw + sb_x;
                                    s[r * 64..r * 64 + 64].copy_from_slice(&recy[b..b + 64]);
                                }
                                s
                            });
                        }
                        restore(&mut recy, &snap_split);
                    }
                    // ---- HORZ4 candidate (4× TX_64X16, strips T→B) ----
                    if want_horz4 {
                        for (i, tus_h) in tus_h[..4].iter_mut().enumerate() {
                            let y0 = sb_y + i * 16;
                            let predh = dc_pred_rect(
                                &recy,
                                pw,
                                y0,
                                sb_x,
                                64,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma64x16.project_scan(
                                &aq::scale_resid(
                                    &get_residual_rect(&yp, pw, y0, sb_x, 64, 16, predh),
                                    sb_resid_scale,
                                ),
                                0.0,
                                &SCAN32X16,
                            );
                            let pred_flat = [predh; 1024];
                            put_block_rect(
                                &mut recy,
                                pw,
                                y0,
                                sb_x,
                                64,
                                16,
                                &itx422::reconstruct_luma_64x16(
                                    &pred_flat,
                                    &lev,
                                    sb_qstep,
                                    &SCAN32X16,
                                    self.bit_depth as i32,
                                ),
                            );
                            *tus_h = levels_to_coeffs(&lev);
                        }
                        let j_h = sse_region(&recy) + lambda * rate_proxy(&tus_h, 4.0);
                        let take = force_horz4 || j_h < best_j;
                        if take {
                            best = Part::Horz4;
                            // best_j no longer read past the last candidate.
                            for r in 0..64 {
                                let b = (sb_y + r) * pw + sb_x;
                                snap_best[r * 64..r * 64 + 64].copy_from_slice(&recy[b..b + 64]);
                            }
                        }
                        restore(&mut recy, &snap_split);
                    }
                    // ---- commit winner ----
                    restore(&mut recy, &snap_best);
                    // CfL decision (4:4:4 whole-64 fast path). recy is final here; recu/
                    // recv hold the neighbor reconstructions for the DC prediction. The
                    // is_cfl context comes from CfL-usage neighbors. This sets the
                    // per-block CfL state consumed by encode_intra_modes during the
                    // luma-block encode below (which emits is_cfl + alphas).
                    let cfl_a = if fmr > 0 { cfl_above[fmc] } else { 0 };
                    let cfl_l = if fmc > 0 { cfl_left[fmr] } else { 0 };
                    enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                    let cfl_choice = if enc.cfl {
                        cfl_decide_64(
                            &recy,
                            &up,
                            &vp,
                            &recu,
                            &recv,
                            pw,
                            sb_y,
                            sb_x,
                            self.bit_depth as i32,
                            neutral,
                            &bases.chroma444,
                            qstep_i,
                            lambda,
                        )
                    } else {
                        None
                    };
                    if let Some(ref ch) = cfl_choice {
                        enc.cfl_use = true;
                        enc.cfl_js = ch.js;
                        enc.cfl_mag_u = ch.mag_u;
                        enc.cfl_mag_v = ch.mag_v;
                        enc.cfl_ctx_u = ch.ctx_u;
                        enc.cfl_ctx_v = ch.ctx_v;
                    } else {
                        enc.cfl_use = false;
                        enc.cfl_signaled = false;
                    }
                    // Chroma intra mode search MUST run before the luma-block encode
                    // below, because that encode emits the uv_mode symbol. Decide the
                    // winning predictor now (when not CfL), set enc.uv_mode, and reuse
                    // the predictor when coding the chroma residual further down.
                    let uv444_pred: Option<(Vec<f32>, Vec<f32>)> =
                        if cfl_choice.is_none() && self.tune.chroma_mode_search {
                            let cand_modes: &[usize] = if self.speed.reduced_modes() {
                                &[0, 1, 4, 5, 6]
                            } else if self.speed.chroma_angle_directional() {
                                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                            } else {
                                &[0, 1, 2, 3, 4, 5, 6]
                            };
                            let mode_lambda = leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
                            let dcu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                            let dcv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                            let mut best_mode = 0usize;
                            let mut best_cost = f64::INFINITY;
                            let mut best_pred: Option<(Vec<f32>, Vec<f32>)> = None;
                            for &m in cand_modes {
                                let (pu, pv): (Vec<f32>, Vec<f32>) = if m == 0 {
                                    (vec![dcu; 64 * 64], vec![dcv; 64 * 64])
                                } else {
                                    (
                                        chroma422::predict_chroma444_whole64(
                                            &recu, pw, sb_y, sb_x, m, neutral, width, height,
                                        ),
                                        chroma422::predict_chroma444_whole64(
                                            &recv, pw, sb_y, sb_x, m, neutral, width, height,
                                        ),
                                    )
                                };
                                let pu_i: Vec<i32> =
                                    pu.iter().map(|&p| (p + 0.5).floor() as i32).collect();
                                let pv_i: Vec<i32> =
                                    pv.iter().map(|&p| (p + 0.5).floor() as i32).collect();
                                let mut ru = vec![0f32; 64 * 64];
                                let mut rv = vec![0f32; 64 * 64];
                                for r in 0..64 {
                                    let b = (sb_y + r) * pw + sb_x;
                                    for c in 0..64 {
                                        ru[r * 64 + c] = up[b + c] - pu[r * 64 + c];
                                        rv[r * 64 + c] = vp[b + c] - pv[r * 64 + c];
                                    }
                                }
                                let lu = bases
                                    .chroma444
                                    .project(&aq::scale_resid(&ru, sb_resid_scale), 0.0);
                                let lv = bases
                                    .chroma444
                                    .project(&aq::scale_resid(&rv, sb_resid_scale), 0.0);
                                let recu_b = itx422::reconstruct_chroma_cfl(
                                    &pu_i,
                                    &lu,
                                    sb_qstep,
                                    &tables::SCAN,
                                    64,
                                    64,
                                    self.bit_depth as i32,
                                );
                                let recv_b = itx422::reconstruct_chroma_cfl(
                                    &pv_i,
                                    &lv,
                                    sb_qstep,
                                    &tables::SCAN,
                                    64,
                                    64,
                                    self.bit_depth as i32,
                                );
                                let mut sse = 0f64;
                                for r in 0..64 {
                                    let b = (sb_y + r) * pw + sb_x;
                                    for c in 0..64 {
                                        let du = up[b + c] - recu_b[r * 64 + c];
                                        let dv = vp[b + c] - recv_b[r * 64 + c];
                                        sse += (du * du + dv * dv) as f64;
                                    }
                                }
                                let rate: f64 =
                                    lu.iter().chain(lv.iter()).map(|&l| l.abs() as f64).sum();
                                let mode_bits = if m == 0 { 0.0 } else { 2.0 };
                                let cost = sse + mode_lambda * (rate + mode_bits);
                                if cost < best_cost {
                                    best_cost = cost;
                                    best_mode = m;
                                    best_pred = if m == 0 { None } else { Some((pu, pv)) };
                                }
                            }
                            enc.uv_mode = best_mode;
                            best_pred
                        } else {
                            None
                        };
                    enc.delta_q_pending = enc.delta_q_present;
                    match best {
                        Part::Vert4 => {
                            let mut skip_cdfs = [0u32; 4];
                            let mut dc_sign_ctxs = [0usize; 4];
                            for i in 0..4 {
                                let (s, d) = sb_tu_contexts_rect(
                                    &tus_v[i],
                                    sb_y,
                                    sb_x + i * 16,
                                    &mut above,
                                    &mut left,
                                    qc,
                                    tmc,
                                    tmr,
                                    4,
                                    16,
                                    false,
                                );
                                skip_cdfs[i] = s;
                                dc_sign_ctxs[i] = d;
                            }
                            encode_luma_block_vert4(
                                &mut enc,
                                &tus_v,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                0,
                                true,
                                none_do_split_cdf,
                            );
                        }
                        Part::Horz4 => {
                            let mut skip_cdfs = [0u32; 4];
                            let mut dc_sign_ctxs = [0usize; 4];
                            for i in 0..4 {
                                let (s, d) = sb_tu_contexts_rect(
                                    &tus_h[i],
                                    sb_y + i * 16,
                                    sb_x,
                                    &mut above,
                                    &mut left,
                                    qc,
                                    tmc,
                                    tmr,
                                    16,
                                    4,
                                    false,
                                );
                                skip_cdfs[i] = s;
                                dc_sign_ctxs[i] = d;
                            }
                            encode_luma_block_horz4(
                                &mut enc,
                                &tus_h,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                0,
                                true,
                                none_do_split_cdf,
                            );
                        }
                        Part::Split => {
                            let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                                &tus_s, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            encode_luma_block_split(
                                &mut enc,
                                &tus_s,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                true,
                                none_do_split_cdf,
                            );
                        }
                    }
                    let bd = self.bit_depth as i32;
                    let (levu, levv) = if let Some(ref ch) = cfl_choice {
                        // CfL: residual against the per-pixel prediction; reconstruct
                        // with that prediction as the base.
                        let mut ru = [0f32; 64 * 64];
                        let mut rv = [0f32; 64 * 64];
                        cfl_prediction::<64>(pw, &up, &vp, sb_y, sb_x, &ch, &mut ru, &mut rv);
                        let levu = bases
                            .chroma444
                            .project(&aq::scale_resid(&ru, sb_resid_scale), 0.0);
                        let levv = bases
                            .chroma444
                            .project(&aq::scale_resid(&rv, sb_resid_scale), 0.0);
                        put_block(
                            &mut recu,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            &itx422::reconstruct_chroma_cfl(
                                &ch.pred_u,
                                &levu,
                                sb_qstep,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            ),
                        );
                        put_block(
                            &mut recv,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            &itx422::reconstruct_chroma_cfl(
                                &ch.pred_v,
                                &levv,
                                sb_qstep,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            ),
                        );
                        (levu, levv)
                    } else if let Some((pu, pv)) = uv444_pred.as_ref() {
                        // Non-DC chroma intra mode chosen above (enc.uv_mode already
                        // set and emitted by the luma-block encode). Code the residual
                        // against the per-pixel predictor, reconstruct with it as base.
                        let pu_i: Vec<i32> = pu.iter().map(|&p| (p + 0.5).floor() as i32).collect();
                        let pv_i: Vec<i32> = pv.iter().map(|&p| (p + 0.5).floor() as i32).collect();
                        let mut ru = vec![0f32; 64 * 64];
                        let mut rv = vec![0f32; 64 * 64];
                        for r in 0..64 {
                            let b = (sb_y + r) * pw + sb_x;
                            for c in 0..64 {
                                ru[r * 64 + c] = up[b + c] - pu[r * 64 + c];
                                rv[r * 64 + c] = vp[b + c] - pv[r * 64 + c];
                            }
                        }
                        let levu = bases
                            .chroma444
                            .project(&aq::scale_resid(&ru, sb_resid_scale), 0.0);
                        let levv = bases
                            .chroma444
                            .project(&aq::scale_resid(&rv, sb_resid_scale), 0.0);
                        put_block(
                            &mut recu,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            &itx422::reconstruct_chroma_cfl(
                                &pu_i,
                                &levu,
                                sb_qstep,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            ),
                        );
                        put_block(
                            &mut recv,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            &itx422::reconstruct_chroma_cfl(
                                &pv_i,
                                &levv,
                                sb_qstep,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            ),
                        );
                        (levu, levv)
                    } else {
                        // DC (either search disabled, or DC won the search above).
                        let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                        let levu = bases.chroma444.project(
                            &aq::scale_resid(
                                &get_residual(&up, pw, sb_y, sb_x, 64, predu),
                                sb_resid_scale,
                            ),
                            0.0,
                        );
                        put_block(
                            &mut recu,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            &itx422::reconstruct_chroma(
                                predu,
                                &levu,
                                sb_qstep,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            ),
                        );
                        let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                        let levv = bases.chroma444.project(
                            &aq::scale_resid(
                                &get_residual(&vp, pw, sb_y, sb_x, 64, predv),
                                sb_resid_scale,
                            ),
                            0.0,
                        );
                        put_block(
                            &mut recv,
                            pw,
                            sb_y,
                            sb_x,
                            64,
                            &itx422::reconstruct_chroma(
                                predv,
                                &levv,
                                sb_qstep,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            ),
                        );
                        (levu, levv)
                    };
                    let ucoeffs = levels_to_coeffs(&levu);
                    let vcoeffs = levels_to_coeffs(&levv);
                    let u_skip = (6 + ua + ul) as u32;
                    encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                    let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                    let v_skip = (6 * (u_present as i32) + va + vl) as u32;
                    encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                    let v_present = vcoeffs.iter().any(|&(_, l)| l != 0);
                    let cfl_used = cfl_choice.is_some() as i32;
                    for c in fmc..fmc + 16 {
                        u_above[c] = u_present as i32;
                        v_above[c] = v_present as i32;
                        cfl_above[c] = cfl_used;
                    }
                    for r in fmr..fmr + 16 {
                        u_left[r] = u_present as i32;
                        v_left[r] = v_present as i32;
                        cfl_left[r] = cfl_used;
                    }
                    // Maintain partition contexts for this whole-64 PARTITION_NONE so that
                    // SBs neighbouring a chroma-motivated split observe correct contexts.
                    partition::sb_none_pctx(row, col, &mut above_pctx, &mut left_pctx);
                    continue;
                }

                // Walk + dispatch. For residues {6,8} each SB yields exactly one Leaf and
                // no RectType ops; RectType is handled generically for forward-compat.
                // A chroma-motivated interior split emits a 4x32x32 PARTITION_SPLIT instead.
                let ops = if sb_use_split {
                    partition::sb_square_split_ops(row, col, &mut above_pctx, &mut left_pctx)
                } else {
                    partition::sb_partition_ops(
                        row,
                        col,
                        tmr as usize,
                        tmc as usize,
                        &mut above_pctx,
                        &mut left_pctx,
                    )
                };
                // Per-SB AQ: on the split path, commit the delta-q and reconstruct at the
                // accumulated qstep (matching the decoder), not the base qstep.
                let (split_qstep, split_resid_scale) = if sb_use_split {
                    aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height)
                } else {
                    enc.delta_q_signaled = 0;
                    aqs.current()
                };
                enc.delta_q_pending = enc.delta_q_present;
                enc.in_interior_split = sb_use_split;
                // Walk leaves code DC chroma; clear any CMS mode left by a fast-path SB.
                enc.uv_mode = 0;
                for op in &ops {
                    let (bw_mi, bh_mi, pc, lmr, lmc) = match op {
                        partition::Op::RectType { cdf, val } => {
                            enc.bool_rect_type(*cdf, *val);
                            continue;
                        }
                        partition::Op::Split {
                            do_split_cdf,
                            square_cdf,
                        } => {
                            enc.bool_do_split(*do_split_cdf, 1);
                            if *square_cdf != 0 {
                                enc.bool_do_square_split(*square_cdf, 1);
                            }
                            continue;
                        }
                        partition::Op::Leaf {
                            bw_mi,
                            bh_mi,
                            part_cdf,
                            mi_row,
                            mi_col,
                        } => (*bw_mi, *bh_mi, part_cdf.unwrap_or(12276), *mi_row, *mi_col),
                    };
                    // Per-leaf position (a single SB may contain several leaves, e.g. the
                    // two stacked 8×32 residue-2 edges). Shadow sb_y/sb_x so the arms below
                    // address the leaf, not the SB origin.
                    let sb_y = lmr * 4;
                    let sb_x = lmc * 4;
                    let ua = if lmr > 0 {
                        u_above[lmc..lmc + bw_mi].iter().any(|&x| x != 0) as i32
                    } else {
                        0
                    };
                    let ul = if lmc > 0 {
                        u_left[lmr..lmr + bh_mi].iter().any(|&x| x != 0) as i32
                    } else {
                        0
                    };
                    let va = if lmr > 0 {
                        v_above[lmc..lmc + bw_mi].iter().any(|&x| x != 0) as i32
                    } else {
                        0
                    };
                    let vl = if lmc > 0 {
                        v_left[lmr..lmr + bh_mi].iter().any(|&x| x != 0) as i32
                    } else {
                        0
                    };
                    {
                        let cfl_a = if lmr > 0 { cfl_above[lmc] } else { 0 };
                        let cfl_l = if lmc > 0 { cfl_left[lmr] } else { 0 };
                        enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                        enc.cfl_use = false;
                        enc.cfl_signaled = false;
                        enc.mhccp_use = false;
                    }
                    let (u_present, v_present) = match (bw_mi, bh_mi) {
                        (16, 16) => {
                            let (tus, mode_idx, _) = encode_luma_sb(
                                &mut recy,
                                &yp,
                                pw,
                                width,
                                height,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                split_qstep,
                                split_resid_scale,
                                &tables::SCAN,
                                neutral,
                                qc,
                                rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                                false, // non-directional path
                            );
                            let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                                &tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            // CfL decision (partition-walk whole-64 leaf). recy is final
                            // after encode_luma_sb; this sets the per-block CfL state read
                            // by encode_intra_modes during the luma encode just below.
                            let cfl_choice = if enc.cfl {
                                cfl_decide_64(
                                    &recy,
                                    &up,
                                    &vp,
                                    &recu,
                                    &recv,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    self.bit_depth as i32,
                                    neutral,
                                    &bases.chroma444,
                                    split_qstep,
                                    leaf::part_lambda(split_qstep, part_lambda_c),
                                )
                            } else {
                                None
                            };
                            if let Some(ref ch) = cfl_choice {
                                enc.cfl_use = true;
                                enc.cfl_js = ch.js;
                                enc.cfl_mag_u = ch.mag_u;
                                enc.cfl_mag_v = ch.mag_v;
                                enc.cfl_ctx_u = ch.ctx_u;
                                enc.cfl_ctx_v = ch.ctx_v;
                            }
                            encode_luma_block_split(
                                &mut enc,
                                &tus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                true,
                                pc,
                            );
                            let bd = self.bit_depth as i32;
                            let (levu, levv) = if let Some(ref ch) = cfl_choice {
                                let mut ru = [0f32; 64 * 64];
                                let mut rv = [0f32; 64 * 64];
                                cfl_prediction::<64>(
                                    pw, &up, &vp, sb_y, sb_x, &ch, &mut ru, &mut rv,
                                );
                                let levu = bases
                                    .chroma444
                                    .project(&aq::scale_resid(&ru, split_resid_scale), 0.0);
                                let levv = bases
                                    .chroma444
                                    .project(&aq::scale_resid(&rv, split_resid_scale), 0.0);
                                put_block(
                                    &mut recu,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    64,
                                    &itx422::reconstruct_chroma_cfl(
                                        &ch.pred_u,
                                        &levu,
                                        split_qstep,
                                        &tables::SCAN,
                                        64,
                                        64,
                                        bd,
                                    ),
                                );
                                put_block(
                                    &mut recv,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    64,
                                    &itx422::reconstruct_chroma_cfl(
                                        &ch.pred_v,
                                        &levv,
                                        split_qstep,
                                        &tables::SCAN,
                                        64,
                                        64,
                                        bd,
                                    ),
                                );
                                (levu, levv)
                            } else {
                                let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                                let levu = bases.chroma444.project(
                                    &aq::scale_resid(
                                        &get_residual(&up, pw, sb_y, sb_x, 64, predu),
                                        split_resid_scale,
                                    ),
                                    0.0,
                                );
                                put_block(
                                    &mut recu,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    64,
                                    &itx422::reconstruct_chroma(
                                        predu,
                                        &levu,
                                        split_qstep,
                                        &tables::SCAN,
                                        64,
                                        64,
                                        bd,
                                    ),
                                );
                                let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                                let levv = bases.chroma444.project(
                                    &aq::scale_resid(
                                        &get_residual(&vp, pw, sb_y, sb_x, 64, predv),
                                        split_resid_scale,
                                    ),
                                    0.0,
                                );
                                put_block(
                                    &mut recv,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    64,
                                    &itx422::reconstruct_chroma(
                                        predv,
                                        &levv,
                                        split_qstep,
                                        &tables::SCAN,
                                        64,
                                        64,
                                        bd,
                                    ),
                                );
                                (levu, levv)
                            };
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = (6 + ua + ul) as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (16, 8) => {
                            let (tus2, mode_idx) = encode_luma_leaf32(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                split_qstep,
                                &tables::SCAN,
                                neutral,
                                qc,
                                rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_64x32(
                                &tus2, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            encode_luma_leaf_64x32(
                                &mut enc, &tus2, &skip2, &dcs2, mode_idx, true, pc,
                            );
                            let predu = dc_pred_rect(
                                &recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levu = bases.chroma444_64x32.project(
                                &aq::scale_resid(
                                    &get_residual_rect(&up, pw, sb_y, sb_x, 64, 32, predu),
                                    split_resid_scale,
                                ),
                                0.0,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                &itx422::reconstruct_chroma(
                                    predu,
                                    &levu,
                                    split_qstep,
                                    &tables::SCAN,
                                    64,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let predv = dc_pred_rect(
                                &recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levv = bases.chroma444_64x32.project(
                                &aq::scale_resid(
                                    &get_residual_rect(&vp, pw, sb_y, sb_x, 64, 32, predv),
                                    split_resid_scale,
                                ),
                                0.0,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                &itx422::reconstruct_chroma(
                                    predv,
                                    &levv,
                                    split_qstep,
                                    &tables::SCAN,
                                    64,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = (6 + ua + ul) as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 16) => {
                            let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                split_qstep,
                                &tables::SCAN,
                                neutral,
                                qc,
                                rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0), (32, 0)],
                                &tus2,
                                sb_y,
                                sb_x,
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                false,
                            );
                            let s2 = [skip2[0], skip2[1]];
                            let d2 = [dcs2[0], dcs2[1]];
                            encode_luma_leaf_32x64(&mut enc, &tus2, &s2, &d2, mode_idx, true, pc);
                            // chroma TX_32X64 (32 wide x 64 tall): chroma422 basis, TX64 skip ctx.
                            let predu = dc_pred_rect(
                                &recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levu = bases.chroma422.project(
                                &aq::scale_resid(
                                    &get_residual_rect(&up, pw, sb_y, sb_x, 32, 64, predu),
                                    split_resid_scale,
                                ),
                                0.0,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                &itx422::reconstruct_chroma(
                                    predu,
                                    &levu,
                                    split_qstep,
                                    &tables::SCAN,
                                    32,
                                    64,
                                    self.bit_depth as i32,
                                ),
                            );
                            let predv = dc_pred_rect(
                                &recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levv = bases.chroma422.project(
                                &aq::scale_resid(
                                    &get_residual_rect(&vp, pw, sb_y, sb_x, 32, 64, predv),
                                    split_resid_scale,
                                ),
                                0.0,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                &itx422::reconstruct_chroma(
                                    predv,
                                    &levv,
                                    split_qstep,
                                    &tables::SCAN,
                                    32,
                                    64,
                                    self.bit_depth as i32,
                                ),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = (6 + ua + ul) as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 8) => {
                            let (tu, mode_idx) = encode_luma_leaf_s32x32(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                split_qstep,
                                &tables::SCAN,
                                neutral,
                                qc,
                                rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0)],
                                std::slice::from_ref(&tu),
                                sb_y,
                                sb_x,
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                true,
                            );
                            // MHCCP evaluation (4:4:4, 32x32 chroma = 32x32 luma):
                            // this block size is MHCCP-eligible. Compete MHCCP against
                            // the DC chroma incumbent; if it wins, code chroma against
                            // the MHCCP predictor and signal it (is_cfl + switch=1 +
                            // mh_dir). ssx=ssy=false for 4:4:4.
                            let bd444 = self.bit_depth as i32;
                            let mh444 = if enc.mhccp && !enc.in_interior_split {
                                let dcu = dc_pred(&recu, pw, sb_y, sb_x, 32, neutral);
                                let dcv = dc_pred(&recv, pw, sb_y, sb_x, 32, neutral);
                                let baseline_j = {
                                    // Incumbent DC J: SSE of DC-predicted residual.
                                    let ru = get_residual(&up, pw, sb_y, sb_x, 32, dcu);
                                    let rv = get_residual(&vp, pw, sb_y, sb_x, 32, dcv);
                                    let sse: f32 = ru.iter().chain(rv.iter()).map(|&r| r * r).sum();
                                    sse as f64
                                };
                                let mut suf = [0f32; 32 * 32];
                                let mut svf = [0f32; 32 * 32];
                                for r in 0..32 {
                                    let b = (sb_y + r) * pw + sb_x;
                                    for c in 0..32 {
                                        suf[r * 32 + c] = up[b + c];
                                        svf[r * 32 + c] = vp[b + c];
                                    }
                                }
                                let mctx = cfl::MhccpCtx {
                                    recy: &recy,
                                    pw,
                                    recu: &recu,
                                    recv: &recv,
                                    pcw: pw,
                                    ly: sb_y,
                                    lx: sb_x,
                                    cy: sb_y,
                                    cx: sb_x,
                                    ssx: false,
                                    ssy: false,
                                    have_top: lmr > 0,
                                    have_left: lmc > 0,
                                    is_top_sb_boundary: true,
                                    size_group: cfl::mhccp_size_group_wh4(8, 8),
                                };
                                cfl::mhccp_decide(
                                    &mctx,
                                    &suf,
                                    &svf,
                                    32,
                                    32,
                                    bd444,
                                    &bases.chroma420,
                                    split_qstep,
                                    leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                    &tables::SCAN,
                                    baseline_j,
                                )
                            } else {
                                None
                            };
                            let mh_win = mh444.as_ref().and_then(|c| c.mhccp.as_ref()).is_some();
                            // This 32x32 leaf codes chroma as DC or MHCCP, both of which use
                            // uv_mode = 0 (UV_DC_PRED). Set it explicitly so a stale uv_mode
                            // from a prior leaf can't leak into the emitted symbol.
                            enc.uv_mode = 0;
                            if mh_win && let Some(ref ch) = mh444 {
                                enc.cfl_use = true;
                                enc.mhccp_use = true;
                                if let Some(ref mh) = ch.mhccp {
                                    enc.mhccp_dir = mh.mh_dir;
                                    enc.mhccp_size_group = mh.size_group;
                                }
                                enc.uv_mode = 0;
                            }
                            encode_luma_leaf_32x32(
                                &mut enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc,
                            );
                            // chroma TX_32X32: MHCCP predictor when selected, else DC.
                            let (levu, levv) = if mh_win {
                                let ch = mh444.as_ref().unwrap();
                                let mut ru = [0f32; 32 * 32];
                                let mut rv = [0f32; 32 * 32];
                                cfl_prediction::<32>(
                                    pw, &up, &vp, sb_y, sb_x, &ch, &mut ru, &mut rv,
                                );
                                // AQ: project at split-q scale, reconstruct at split_qstep.
                                let levu = bases
                                    .chroma420
                                    .project(&aq::scale_resid(&ru, split_resid_scale), 0.0);
                                let levv = bases
                                    .chroma420
                                    .project(&aq::scale_resid(&rv, split_resid_scale), 0.0);
                                put_block(
                                    &mut recu,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    32,
                                    &itx422::reconstruct_chroma_cfl(
                                        &ch.pred_u,
                                        &levu,
                                        split_qstep,
                                        &tables::SCAN,
                                        32,
                                        32,
                                        bd444,
                                    ),
                                );
                                put_block(
                                    &mut recv,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    32,
                                    &itx422::reconstruct_chroma_cfl(
                                        &ch.pred_v,
                                        &levv,
                                        split_qstep,
                                        &tables::SCAN,
                                        32,
                                        32,
                                        bd444,
                                    ),
                                );
                                (levu, levv)
                            } else {
                                // chroma TX_32X32: DC pred; project at split scale, recon at split_qstep.
                                let predu = dc_pred(&recu, pw, sb_y, sb_x, 32, neutral);
                                let levu = bases.chroma420.project(
                                    &aq::scale_resid(
                                        &get_residual(&up, pw, sb_y, sb_x, 32, predu),
                                        split_resid_scale,
                                    ),
                                    0.0,
                                );
                                put_block(
                                    &mut recu,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    32,
                                    &itx422::reconstruct_chroma(
                                        predu,
                                        &levu,
                                        split_qstep,
                                        &tables::SCAN,
                                        32,
                                        32,
                                        self.bit_depth as i32,
                                    ),
                                );
                                let predv = dc_pred(&recv, pw, sb_y, sb_x, 32, neutral);
                                let levv = bases.chroma420.project(
                                    &aq::scale_resid(
                                        &get_residual(&vp, pw, sb_y, sb_x, 32, predv),
                                        split_resid_scale,
                                    ),
                                    0.0,
                                );
                                put_block(
                                    &mut recv,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    32,
                                    &itx422::reconstruct_chroma(
                                        predv,
                                        &levv,
                                        split_qstep,
                                        &tables::SCAN,
                                        32,
                                        32,
                                        self.bit_depth as i32,
                                    ),
                                );
                                (levu, levv)
                            };

                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = (6 + ua + ul) as u32;
                            encode_chroma_block_ex(&mut enc, &uc, u_skip, true, false);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (4, 16) => {
                            // Right-edge 16×64 luma leaf: RD single TX_16X64 vs tx-partition
                            // HORZ (2×TX_16X32, per-TU sequential prediction).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 64, neutral, bd);
                            let resid = aq::scale_resid(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 64, pred),
                                split_resid_scale,
                            );
                            let rate = |lev: &[f32]| -> f64 {
                                lev.iter()
                                    .filter(|&&v| v != 0.0)
                                    .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                                    .sum::<f64>()
                            };
                            let sse_vs = |rec: &[f32], h2: usize, yoff: usize| -> f64 {
                                let mut s = 0f64;
                                for r in 0..h2 {
                                    for c in 0..16 {
                                        let d = yp[(sb_y + yoff + r) * pw + sb_x + c] as f64
                                            - rec[r * 16 + c] as f64;
                                        s += d * d;
                                    }
                                }
                                s
                            };
                            let lambda = leaf::part_lambda(split_qstep, self.tune.part_lambda_c);
                            let lev = bases.luma16x64.project_scan(&resid, 0.0, &SCAN16X32);
                            let rec_a = itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                split_qstep,
                                &SCAN16X32,
                                16,
                                64,
                                bd,
                            );
                            let j_a = sse_vs(&rec_a, 64, 0) + lambda * rate(&lev);
                            let mut levs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                            let mut recs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                            let mut j_b = lambda * 4.0;
                            {
                                let mut scratch = recy.clone();
                                for half in 0..2 {
                                    let tuy = sb_y + half * 32;
                                    let p =
                                        dc_pred_rect(&scratch, pw, tuy, sb_x, 16, 32, neutral, bd);
                                    let r = aq::scale_resid(
                                        &get_residual_rect(&yp, pw, tuy, sb_x, 16, 32, p),
                                        split_resid_scale,
                                    );
                                    let l = bases.luma16x32.project_scan(&r, 0.0, &SCAN16X32);
                                    let rec = itx422::reconstruct_chroma(
                                        p,
                                        &l,
                                        split_qstep,
                                        &SCAN16X32,
                                        16,
                                        32,
                                        bd,
                                    );
                                    put_block_rect(&mut scratch, pw, tuy, sb_x, 16, 32, &rec);
                                    j_b += sse_vs(&rec, 32, half * 32) + lambda * rate(&l);
                                    levs_b[half] = l;
                                    recs_b[half] = rec;
                                }
                            }
                            if j_b < j_a {
                                for (half, src) in recs_b.iter().enumerate() {
                                    put_block_rect(
                                        &mut recy,
                                        pw,
                                        sb_y + half * 32,
                                        sb_x,
                                        16,
                                        32,
                                        src,
                                    );
                                }
                                let tus: [Vec<Coeff>; 2] =
                                    [levels_to_coeffs(&levs_b[0]), levels_to_coeffs(&levs_b[1])];
                                let mut skips = [0u32; 2];
                                let mut dcss = [0usize; 2];
                                for half in 0..2 {
                                    let (sk, dc) = sb_tu_contexts_rect(
                                        &tus[half],
                                        sb_y + half * 32,
                                        sb_x,
                                        &mut above,
                                        &mut left,
                                        qc,
                                        tmc,
                                        tmr,
                                        4,
                                        8,
                                        false,
                                    );
                                    skips[half] = sk;
                                    dcss[half] = dc;
                                }
                                coder::encode_luma_leaf_16x64_horz(
                                    &mut enc, &tus, &skips, &dcss, 0, true, pc,
                                );
                            } else {
                                put_block_rect(&mut recy, pw, sb_y, sb_x, 16, 64, &rec_a);
                                let tu = levels_to_coeffs(&lev);
                                let (skip, dcs) = sb_tu_contexts_rect(
                                    &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 16,
                                    true,
                                );
                                encode_luma_leaf_16x64(&mut enc, &tu, skip, dcs, 0, true, pc);
                            }
                            // chroma 16×64 (TX_16X64): reuse luma16x64 basis for projection
                            // (validity-only); chroma eob class 512, TX_32X32 skip ctx.
                            let predu = dc_pred_rect(
                                &recu,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levu = bases.luma16x64.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 16, 64, predu),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &itx422::reconstruct_chroma(
                                    predu,
                                    &levu,
                                    split_qstep,
                                    &SCAN16X32,
                                    16,
                                    64,
                                    self.bit_depth as i32,
                                ),
                            );
                            let predv = dc_pred_rect(
                                &recv,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levv = bases.luma16x64.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 16, 64, predv),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &itx422::reconstruct_chroma(
                                    predv,
                                    &levv,
                                    split_qstep,
                                    &SCAN16X32,
                                    16,
                                    64,
                                    self.bit_depth as i32,
                                ),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect_w(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN16X32,
                                EobCdf::ChrEob512,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                                4,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
                            encode_chroma_block_rect_w(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN16X32,
                                EobCdf::ChrEob512,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                                4,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (16, 4) => {
                            // Bottom-edge 64×16 luma leaf: RD between single TX_64X16 and
                            // tx-partition VERT (2×TX_32X16, no >32 zero-out).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 64, 16, neutral, bd);
                            let resid = aq::scale_resid(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 64, 16, pred),
                                split_resid_scale,
                            );
                            let rate = |lev: &[f32]| -> f64 {
                                lev.iter()
                                    .filter(|&&v| v != 0.0)
                                    .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                                    .sum::<f64>()
                            };
                            let sse_vs = |rec: &[f32], w: usize, xoff: usize| -> f64 {
                                let mut s = 0f64;
                                for r in 0..16 {
                                    for c in 0..w {
                                        let d = yp[(sb_y + r) * pw + sb_x + xoff + c] as f64
                                            - rec[r * w + c] as f64;
                                        s += d * d;
                                    }
                                }
                                s
                            };
                            let lambda = leaf::part_lambda(split_qstep, self.tune.part_lambda_c);
                            let lev = bases.luma64x16.project_scan(&resid, 0.0, &SCAN32X16);
                            let rec_a = itx422::reconstruct_chroma(
                                pred,
                                &lev,
                                split_qstep,
                                &SCAN32X16,
                                64,
                                16,
                                bd,
                            );
                            let j_a = sse_vs(&rec_a, 64, 0) + lambda * rate(&lev);
                            // Per-TU prediction (decoder predicts each sub-TU from prior
                            // recon), simulated on a scratch copy so cand A stays clean.
                            let mut levs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                            let mut recs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                            let mut j_b = lambda * 4.0;
                            {
                                let mut scratch = recy.clone();
                                for half in 0..2 {
                                    let tux = sb_x + half * 32;
                                    let p =
                                        dc_pred_rect(&scratch, pw, sb_y, tux, 32, 16, neutral, bd);
                                    let r = aq::scale_resid(
                                        &get_residual_rect(&yp, pw, sb_y, tux, 32, 16, p),
                                        split_resid_scale,
                                    );
                                    let l = bases.luma32x16.project_scan(&r, 0.0, &SCAN32X16);
                                    let rec = itx422::reconstruct_chroma(
                                        p,
                                        &l,
                                        split_qstep,
                                        &SCAN32X16,
                                        32,
                                        16,
                                        bd,
                                    );
                                    put_block_rect(&mut scratch, pw, sb_y, tux, 32, 16, &rec);
                                    j_b += sse_vs(&rec, 32, half * 32) + lambda * rate(&l);
                                    levs_b[half] = l;
                                    recs_b[half] = rec;
                                }
                            }
                            if j_b < j_a {
                                for (half, src) in recs_b.iter().enumerate() {
                                    put_block_rect(
                                        &mut recy,
                                        pw,
                                        sb_y,
                                        sb_x + half * 32,
                                        32,
                                        16,
                                        src,
                                    );
                                }
                                let tus: [Vec<Coeff>; 2] =
                                    [levels_to_coeffs(&levs_b[0]), levels_to_coeffs(&levs_b[1])];
                                let mut skips = [0u32; 2];
                                let mut dcss = [0usize; 2];
                                for half in 0..2 {
                                    let (sk, dc) = sb_tu_contexts_rect(
                                        &tus[half],
                                        sb_y,
                                        sb_x + half * 32,
                                        &mut above,
                                        &mut left,
                                        qc,
                                        tmc,
                                        tmr,
                                        8,
                                        4,
                                        false,
                                    );
                                    skips[half] = sk;
                                    dcss[half] = dc;
                                }
                                coder::encode_luma_leaf_64x16_vert(
                                    &mut enc, &tus, &skips, &dcss, 0, true, pc,
                                );
                            } else {
                                put_block_rect(&mut recy, pw, sb_y, sb_x, 64, 16, &rec_a);
                                let tu = levels_to_coeffs(&lev);
                                let (skip, dcs) = sb_tu_contexts_rect(
                                    &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 16, 4,
                                    true,
                                );
                                encode_luma_leaf_64x16(&mut enc, &tu, skip, dcs, 0, true, pc);
                            }
                            let predu = dc_pred_rect(
                                &recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levu = bases.luma64x16.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 64, 16, predu),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &itx422::reconstruct_chroma(
                                    predu,
                                    &levu,
                                    split_qstep,
                                    &SCAN32X16,
                                    64,
                                    16,
                                    self.bit_depth as i32,
                                ),
                            );
                            let predv = dc_pred_rect(
                                &recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let levv = bases.luma64x16.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 64, 16, predv),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &itx422::reconstruct_chroma(
                                    predv,
                                    &levv,
                                    split_qstep,
                                    &SCAN32X16,
                                    64,
                                    16,
                                    self.bit_depth as i32,
                                ),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN32X16,
                                EobCdf::ChrEob512,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip = (6 * (up_ as i32) + va + vl) as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN32X16,
                                EobCdf::ChrEob512,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (2, 8) => {
                            // Right-edge 8×32 luma leaf (residue 2). 8×64 would be 1:8
                            // aspect (disallowed) so the SB partitions to 8×32 leaves.
                            // TX_8X32 = entropy class 2; luma is DC-only (eob count 1 →
                            // no LONG_SIDE_32 tx_type). do_part group 8 → cdf 18958.
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma8x32.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 32, pred),
                                0.0,
                                &SCAN8X32,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &SCAN8X32,
                                    8,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 8, true,
                            );
                            encode_luma_leaf_8x32(&mut enc, &tu, skip, dcs, 0, true, pc);
                            // chroma 8×32 (TX_8X32): full AC, reuse luma8x32 basis, eob
                            // class 256, class-2 U skip / shared V skip.
                            // chroma 8x32: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &bases.luma8x32,
                                &SCAN8X32,
                                EobCdf::ChrEob256,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                                &SKIP_TX16_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (8, 2) => {
                            // Bottom-edge 32×8 luma leaf (residue 2). TX_32X8 = class 2,
                            // DC-only luma, do_part group 8 → cdf 18958, scan SCAN32X8.
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma32x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 32, 8, pred),
                                0.0,
                                &SCAN32X8,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &SCAN32X8,
                                    32,
                                    8,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 8, 2, true,
                            );
                            encode_luma_leaf_32x8(&mut enc, &tu, skip, dcs, 0, true, pc);
                            // chroma 32x8: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &bases.luma32x8,
                                &SCAN32X8,
                                EobCdf::ChrEob256,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                                &SKIP_TX16_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (4, 4) => {
                            // Bottom-right 16×16 corner leaf (residue 4 in both dims).
                            // Native TX_16X16 (entropy class 2, eob class 256). The luma
                            // tx_type is RD-chosen between DCT_DCT (idx 0) and ADST_ADST
                            // (idx 1, the mode-dependent EXT_NEW_TX_SET alternative);
                            // chroma stays DCT (tx_type is luma-only).
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let resid = get_residual_rect(&yp, pw, sb_y, sb_x, 16, 16, pred);
                            let pred_flat = [pred; 256];
                            // Source pixels for the distortion term.
                            let mut src16 = [0f32; 256];
                            for r in 0..16 {
                                for c in 0..16 {
                                    src16[r * 16 + c] = yp[(sb_y + r) * pw + sb_x + c];
                                }
                            }
                            let rate = |lev: &[f32]| -> f64 {
                                lev.iter()
                                    .filter(|&&v| v != 0.0)
                                    .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                                    .sum::<f64>()
                            };
                            let sse = |rec: &[f32]| -> f64 {
                                (0..256)
                                    .map(|i| {
                                        let d = src16[i] as f64 - rec[i] as f64;
                                        d * d
                                    })
                                    .sum()
                            };
                            let lambda =
                                crate::av2::leaf::part_lambda(split_qstep, self.tune.part_lambda_c);
                            // DCT_DCT candidate (idx 0).
                            let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                            let rec_dct = crate::av2::itx422::reconstruct_luma16(
                                &pred_flat,
                                &lev_dct,
                                split_qstep,
                                &SCAN16,
                                self.bit_depth as i32,
                            );
                            let cost_dct = sse(&rec_dct) + lambda * rate(&lev_dct);
                            // ADST_ADST candidate (idx 1, DST-VII both axes).
                            let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
                            let rec_adst = itx422::reconstruct_luma16_adst(
                                &pred_flat,
                                &lev_adst,
                                split_qstep,
                                &SCAN16,
                                true,
                                true,
                                self.bit_depth as i32,
                            );
                            let cost_adst = sse(&rec_adst) + lambda * (rate(&lev_adst) + 0.2);
                            // ADST_DCT candidate (idx 2: ADST vertical, DCT horizontal →
                            // inverse row_adst=false, col_adst=true). The tx_type symbol
                            // costs ~3.1 bits more than DCT (idx 2 in the EXT_NEW_TX_SET
                            // cdf), so it only wins on a clear distortion gain.
                            let lev_ad =
                                bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
                            let rec_ad = itx422::reconstruct_luma16_adst(
                                &pred_flat,
                                &lev_ad,
                                split_qstep,
                                &SCAN16,
                                false,
                                true,
                                self.bit_depth as i32,
                            );
                            let cost_ad = sse(&rec_ad) + lambda * (rate(&lev_ad) + 3.12);
                            // DCT_ADST candidate (idx 3: DCT vertical, ADST horizontal →
                            // inverse row_adst=true, col_adst=false; ~2.7 extra bits).
                            let lev_da =
                                bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
                            let rec_da = itx422::reconstruct_luma16_adst(
                                &pred_flat,
                                &lev_da,
                                split_qstep,
                                &SCAN16,
                                true,
                                false,
                                self.bit_depth as i32,
                            );
                            let cost_da = sse(&rec_da) + lambda * (rate(&lev_da) + 2.71);
                            // Pick the best tx_type. Tie-break preserves the original
                            // DCT_DCT-over-ADST_ADST behavior (each alternative must be
                            // STRICTLY better), so byte output is unchanged wherever the
                            // mixed transforms don't help.
                            let mut best = cost_dct;
                            let mut choice = 0usize;
                            if cost_adst < best {
                                best = cost_adst;
                                choice = 1;
                            }
                            if cost_ad < best {
                                best = cost_ad;
                                choice = 2;
                            }
                            if cost_da < best {
                                choice = 3;
                            }
                            let (lev, rec, tx_idx): (&[f32], &[f32; 256], usize) = match choice {
                                1 => (&lev_adst, &rec_adst, 1),
                                2 => (&lev_ad, &rec_ad, 2),
                                3 => (&lev_da, &rec_da, 3),
                                _ => (&lev_dct, &rec_dct, 0),
                            };
                            put_block_rect(&mut recy, pw, sb_y, sb_x, 16, 16, rec);
                            let tu: Vec<Coeff> = levels_to_coeffs(lev);
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 4, true,
                            );
                            // TX_16X16 luma skip = class-2 cdf, block_eq_tx → ctx 0.
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_16x16_full(
                                &mut enc, &tu, skip, dcs, 0, true, pc, 11074, tx_idx,
                            );
                            // chroma 16×16 (TX_16X16): full AC, reuse luma16x16 basis,
                            // chroma 16x16: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16,
                                &SCAN16,
                                EobCdf::ChrEob256,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                                &SKIP_TX16_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (2, 2) => {
                            // Both-axis residue-2 corner: 8×8 luma (TX_8X8) + 8×8 chroma per
                            // plane (4:4:4, full-res, same stride/position as luma).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 8, 8, neutral, bd);
                            let lev = bases.c8x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 8, pred),
                                0.0,
                                &SCAN8X8,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                8,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &SCAN8X8,
                                    8,
                                    8,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 2, true,
                            );
                            encode_luma_leaf_8x8(
                                &mut enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                true,
                                pc,
                                3148,
                                Some((&crate::av2::coder::TXTP_EXT8, 0, 6)),
                            );
                            // chroma 8x8: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                8,
                                &bases.c8x8,
                                &SCAN8X8,
                                EobCdf::ChrEob64,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                64,
                                &SKIP_TX8_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (2, 4) => {
                            // residue-2 W × residue-4 H: 8×16 luma + 8×16 chroma per plane
                            // (4:4:4 full-res → TX_8X16, ctx2, eob128).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
                            let lev = bases.c8x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 16, pred),
                                0.0,
                                &tables::SCAN8X16,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                16,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &tables::SCAN8X16,
                                    8,
                                    16,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 4, true,
                            );
                            coder::encode_luma_leaf_rect128(
                                &mut enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                true,
                                2,
                                4,
                                pc,
                                12348,
                                &tables::SCAN8X16,
                                Some((&coder::TXTP_EXT8, 0, 6)),
                            );
                            // chroma 8x16: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                16,
                                &bases.c8x16,
                                &tables::SCAN8X16,
                                EobCdf::ChrEob128,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                128,
                                &SKIP_TX16_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (4, 2) => {
                            // residue-4 W × residue-2 H: 16×8 luma + 16×8 chroma per plane
                            // (4:4:4 full-res → TX_16X8, ctx2, eob128).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
                            let lev = bases.c16x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 8, pred),
                                0.0,
                                &tables::SCAN16X8,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                8,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &tables::SCAN16X8,
                                    16,
                                    8,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 2, true,
                            );
                            coder::encode_luma_leaf_rect128(
                                &mut enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                true,
                                4,
                                2,
                                pc,
                                12348,
                                &tables::SCAN16X8,
                                Some((&coder::TXTP_EXT8, 0, 6)),
                            );
                            // chroma 16x8: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                8,
                                &bases.c16x8,
                                &tables::SCAN16X8,
                                EobCdf::ChrEob128,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                128,
                                &SKIP_TX16_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (4, 8) => {
                            // residue-4 W × residue-{6,8} H: 16×32 luma + 16×32 chroma per
                            // plane (4:4:4 full-res → TX_16X32, ctx3, eob512).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 32, neutral, bd);
                            let lev = bases.luma16x32.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 32, pred),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                32,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &SCAN16X32,
                                    16,
                                    32,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 8, true,
                            );
                            encode_luma_leaf_16x32(&mut enc, &tu, skip, dcs, 0, true, pc);
                            // chroma 16x32: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                32,
                                &bases.luma16x32,
                                &SCAN16X32,
                                EobCdf::ChrEob512,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                                &CHROMA_SKIP_TX32_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        (8, 4) => {
                            // residue-{6,8} W × residue-4 H: 32×16 luma + 32×16 chroma per
                            // plane (4:4:4 full-res → TX_32X16, ctx3, eob512).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 32, 16, neutral, bd);
                            let lev = bases.luma32x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 32, 16, pred),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                16,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    split_qstep,
                                    &SCAN32X16,
                                    32,
                                    16,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 8, 4, true,
                            );
                            encode_luma_leaf_32x16(&mut enc, &tu, skip, dcs, 0, true, pc);
                            // chroma 32x16: DC or MHCCP (eligible).
                            let mh_on = enc.mhccp;
                            code_444_chroma_leaf(
                                &mut enc,
                                &recy,
                                &mut recu,
                                &mut recv,
                                &up,
                                &vp,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                16,
                                &bases.luma32x16,
                                &SCAN32X16,
                                EobCdf::ChrEob512,
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                                &CHROMA_SKIP_TX32_QC[qc],
                                qc,
                                neutral,
                                split_qstep,
                                ua,
                                ul,
                                va,
                                vl,
                                lmr > 0,
                                lmc > 0,
                                leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                mh_on,
                                self.bit_depth as i32,
                            )
                        }
                        other => unreachable!("unsupported lossy leaf {:?}", other),
                    };
                    // CfL-usage neighbor update: enc.cfl_use holds this leaf's decision
                    // (true only for a (16,16) leaf that picked CfL; false otherwise).
                    let cfl_used = enc.cfl_signaled as i32;
                    for c in lmc..lmc + bw_mi {
                        u_above[c] = u_present as i32;
                        v_above[c] = v_present as i32;
                        cfl_above[c] = cfl_used;
                    }
                    for r in lmr..lmr + bh_mi {
                        u_left[r] = u_present as i32;
                        v_left[r] = v_present as i32;
                        cfl_left[r] = cfl_used;
                    }
                }
                enc.in_interior_split = false;
            }
        }
        if let Ok(p) = std::env::var("DUMP_REC") {
            let mut o = Vec::with_capacity(width * height * 3);
            for buf in [&recy, &recu, &recv] {
                for r in 0..height {
                    o.extend(
                        buf[r * pw..r * pw + width]
                            .iter()
                            .map(|&v| v.clamp(0.0, 255.0) as u8),
                    );
                }
            }
            std::fs::write(p, o).unwrap();
        }
        enc
    }

    /// Multi-tile 4:4:4 assembly. Each tile is encoded as an independent sub-frame
    /// (CDFs/contexts reset, tile boundary == frame boundary for prediction), then
    /// concatenated under one multi-tile frame header with size prefixes. Tiles are
    /// independent, so the per-tile encodes run in parallel across `threads` workers.
    #[allow(clippy::too_many_arguments)]
    fn encode_444_tiled(
        &self,
        yf: &[f32],
        cbf: &[f32],
        crf: &[f32],
        width: usize,
        height: usize,
        config: &Config,
        color: &Cicp,
        log2c: usize,
        log2r: usize,
        threads: usize,
    ) -> Av2Frame {
        // Tile column/row boundaries fall on 64-px superblock edges, so every interior
        // tile is SB-aligned; only the right-column / bottom-row tiles inherit the
        // frame's partial edge. A tile decodes correctly in-frame when its dimensions
        // are boundary-exact (lossy_native_mi is Some — SB-aligned or a supported
        // residue edge). When *every* tile is exact we signal the real frame size and
        // each tile carries its own native partial-edge entropy (byte-identical to a
        // standalone encode of that region). When some edge tile is NOT exact (e.g. a
        // residue-2 corner that would otherwise pad+clap per tile — which is invalid in
        // a shared multi-tile frame), we instead pad the WHOLE frame to SB-aligned, carve
        // tiles on the padded grid so all of them are SB-aligned, signal the padded size,
        // and let the AVIF muxer crop back to width×height with one frame-level clap.
        let native_specs = tile_specs(width, height, log2c, log2r);
        let exact = native_specs
            .iter()
            .all(|&(_, _, tw, th)| lossy_native_mi(tw, th).is_some());
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (sig_w, sig_h, stride, planes, specs) = if exact {
            (
                width,
                height,
                width,
                (yf.to_vec(), cbf.to_vec(), crf.to_vec()),
                native_specs,
            )
        } else {
            (
                pw,
                ph,
                pw,
                (
                    pad_plane(yf, width, height, pw, ph),
                    pad_plane(cbf, width, height, pw, ph),
                    pad_plane(crf, width, height, pw, ph),
                ),
                tile_specs(pw, ph, log2c, log2r),
            )
        };
        let (yf, cbf, crf) = (&planes.0, &planes.1, &planes.2);
        let n = specs.len();
        let mut tiles_bytes: Vec<Vec<u8>> = vec![Vec::new(); n];
        // Each tile is a fully independent sub-frame encode, so they run concurrently.
        // Output order (raster) is preserved because each worker writes its own slot.
        let nthreads = Self::resolve_threads(threads).min(n.max(1));
        if nthreads <= 1 || n <= 1 {
            for (slot, &(x0, y0, tw, th)) in tiles_bytes.iter_mut().zip(&specs) {
                let ty = extract_subplane(yf, stride, x0, y0, tw, th);
                let tu = extract_subplane(cbf, stride, x0, y0, tw, th);
                let tv = extract_subplane(crf, stride, x0, y0, tw, th);
                *slot = self.encode_444_core(&ty, &tu, &tv, tw, th).finish();
            }
        } else {
            let chunk = n.div_ceil(nthreads);
            let me = self;
            let (yf, cbf, crf) = (&yf, &cbf, &crf);
            std::thread::scope(|sc| {
                for (out_chunk, spec_chunk) in
                    tiles_bytes.chunks_mut(chunk).zip(specs.chunks(chunk))
                {
                    sc.spawn(move || {
                        for (slot, &(x0, y0, tw, th)) in out_chunk.iter_mut().zip(spec_chunk) {
                            let ty = extract_subplane(yf, stride, x0, y0, tw, th);
                            let tu = extract_subplane(cbf, stride, x0, y0, tw, th);
                            let tv = extract_subplane(crf, stride, x0, y0, tw, th);
                            *slot = me.encode_444_core(&ty, &tu, &tv, tw, th).finish();
                        }
                    });
                }
            });
        }
        assemble_multitile(
            config,
            sig_w,
            sig_h,
            width,
            height,
            color,
            log2c,
            log2r,
            self.bit_depth,
            ChromaFormat::Yuv444,
            &tiles_bytes,
        )
    }

    /// 4:4:4 lossless (q=0): luma + full-resolution U/V, all TX_4X4 WHT. Per superblock
    /// the block codes intra modes (incl. use_dpcm_y/uv = 0 and DC uv mode), then 256
    /// luma TUs, 256 U TUs, 256 V TUs — matching avm's shared-tree plane order.
    fn encode_yuv444_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);
        let config = self.config(Layout::I444);
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = false;
        enc.mhccp_ssy = false;
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        // mi grid is 8px-aligned; recursion handles every boundary -> always exact.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        // luma ctx grids (0x40 = neutral DC-sign packing); chroma grids store cul (init 0).
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        let mut ua = vec![0u8; pw / 4 + 16];
        let mut ul = vec![0u8; ph / 4 + 16];
        let mut va = vec![0u8; pw / 4 + 16];
        let mut vl = vec![0u8; ph / 4 + 16];

        let nsb = sb_rows * sb_cols;
        // Phase A: per-SB TU generation (DC-pred + WHT + levels). Independent across SBs
        // (lossless reconstruction == source), so this is data-parallel.
        type PackedCoeff = Vec<Coeff>;
        let mut sbtus: Vec<(Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)> = (0..nsb)
            .map(|_| (Vec::new(), Vec::new(), Vec::new()))
            .collect();
        let gen_tile =
            |idx: usize, slot: &mut (Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)| {
                let (sb_y, sb_x) = ((idx / sb_cols) * 64, (idx % sb_cols) * 64);
                let (rr, rc) = rem(idx / sb_cols, idx % sb_cols);
                *slot = (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&up, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&vp, pw, sb_y, sb_x, neutral, rr, rc),
                );
            };
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                gen_tile(idx, slot);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (yp, up, vp) = (&yp, &up, &vp);
            let (code_mc, code_mr) = (code_mc, code_mr);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (row, col) = ((base + k) / sb_cols, (base + k) % sb_cols);
                            let (sb_y, sb_x) = (row * 64, col * 64);
                            let rr = (code_mr - row * 16).min(16);
                            let rc = (code_mc - col * 16).min(16);
                            *slot = (
                                lossless_sb_tus(yp, pw, sb_y, sb_x, neutral, rr, rc),
                                lossless_sb_tus(up, pw, sb_y, sb_x, neutral, rr, rc),
                                lossless_sb_tus(vp, pw, sb_y, sb_x, neutral, rr, rc),
                            );
                        }
                    });
                }
            });
        }
        // Phase B: serial context derivation (cross-SB grids) + entropy coding.
        // Partition context arrays (av2 update_partition_context): `above` persists
        // down columns frame-wide; `left` is len-16 and zeroed per SB row.
        let mut above_pctx = vec![0u8; code_mc + 16];
        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                let (ytus, utus, vtus) = &sbtus[row * sb_cols + col];
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    code_mr,
                    code_mc,
                    &mut above_pctx,
                    &mut left_pctx,
                );
                for op in &ops {
                    match *op {
                        partition::Op::RectType { cdf, val } => {
                            enc.bool_rect_type(cdf, val);
                        }
                        partition::Op::Split {
                            do_split_cdf,
                            square_cdf,
                        } => {
                            enc.bool_do_split(do_split_cdf, 1);
                            if square_cdf != 0 {
                                enc.bool_do_square_split(square_cdf, 1);
                            }
                        }
                        partition::Op::Leaf {
                            mi_row,
                            mi_col,
                            bw_mi,
                            bh_mi,
                            part_cdf,
                        } => {
                            let lr = mi_row - row * 16;
                            let lc = mi_col - col * 16;
                            let lrows = bh_mi.min(rr - lr);
                            let lcols = bw_mi.min(rc - lc);
                            let slice = |g: &[Vec<Coeff>]| -> Vec<Vec<Coeff>> {
                                let mut v = Vec::with_capacity(lrows * lcols);
                                for i in 0..lrows {
                                    for j in 0..lcols {
                                        v.push(g[(lr + i) * rc + (lc + j)].clone());
                                    }
                                }
                                v
                            };
                            let (lytus, lutus, lvtus) = (slice(ytus), slice(utus), slice(vtus));
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let (yskip, ydcs) =
                                sb_tu4_contexts(&lytus, ly, lx, &mut ya, &mut yl, lrows, lcols);
                            let yskip_cdfs: Vec<u32> =
                                yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                            let uskip = sb_tu4_chroma_skip(
                                &lutus, ly, lx, &mut ua, &mut ul, false, false, lrows, lcols,
                            );
                            // avm's eob_u_flag is the LAST U TU of the block, used by every V TU.
                            let u_last_nz =
                                lutus.last().is_some_and(|t| t.iter().any(|&(_, l)| l != 0));
                            let vskip = sb_tu4_chroma_skip(
                                &lvtus, ly, lx, &mut va, &mut vl, true, u_last_nz, lrows, lcols,
                            );
                            // modes (incl. uv) + luma coeffs, then U, then V (shared-tree order)
                            encode_lossless_luma_sb(
                                &mut enc,
                                &lytus,
                                &yskip_cdfs,
                                &ydcs,
                                0,
                                true,
                                part_cdf,
                            );
                            for (i, tu) in lutus.iter().enumerate() {
                                encode_chroma_tu4(
                                    &mut enc,
                                    tu,
                                    TXB_SKIP_TX4_Q0[uskip[i]] as u32,
                                    false,
                                );
                            }
                            for (i, tu) in lvtus.iter().enumerate() {
                                encode_chroma_tu4(
                                    &mut enc,
                                    tu,
                                    V_TXB_SKIP_TX4_Q0[vskip[i]] as u32,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode an RGB image to 4:4:4 AV2. Converts RGB→YCbCr internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_444<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let bd = img.bit_depth;
        let maxv = (1i32 << bd.bits()) - 1;
        let off_q = (1i32 << (bd.bits() - 1)) << Q;
        let mx_i = maxv;
        let n = img.planes[0].len();
        let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        for (((((yv, cbv), crv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(cb.iter_mut())
            .zip(cr.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *cbv = ((CB_R * ri + CB_G * gi + CB_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
            *crv = ((CR_R * ri + CR_G * gi + CR_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
        }
        self.encode_yuv444(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr, Vec::new()],
            },
            color,
        )
    }
}
