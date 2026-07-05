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

pub(crate) static CCSO_OFFSET: [i32; 8] = [0, 1, -1, 3, -3, 7, -7, -10];
pub(crate) static CCSO_SCALE: [i32; 4] = [1, 2, 3, 4];

/// `quant_sz[scale_idx][quant_idx]` — the quantizer step that thresholds the
/// luma-difference classification.
pub(crate) static QUANT_SZ: [[u16; 4]; 4] = [
    [16, 8, 32, 0],
    [56, 40, 64, 128],
    [48, 24, 96, 192],
    [80, 112, 160, 256],
];

/// Number of edge intervals per `edge_clf` (3 for clf 0, 2 for clf 1).
static EDGE_INTERVAL: [usize; 2] = [3, 2];

/// `CCSO_PADDING_SIZE`: luma border extension for the edge taps.
pub(crate) const PAD: usize = 5;

/// Decode `ext_filter_support` (0..=6) into the two tap offsets `(dx, dy)`.
/// Mirrors `derive_ccso_sample_pos` (where the stride term is `dy` and the
/// constant term is `dx`).
fn tap_offsets(sup: u8) -> [(isize, isize); 2] {
    match sup {
        0 => [(0, -1), (0, 1)],
        1 => [(-1, 0), (1, 0)],
        2 => [(-1, -1), (1, 1)],
        3 => [(1, -1), (-1, 1)],
        4 => [(-2, -1), (2, 1)],
        5 => [(-2, 1), (2, -1)],
        _ => [(2, 0), (-2, 0)], // 6
    }
}

/// Classify one luma tap difference `d = luma[tap] - luma[center]` per
/// `cal_filter_support`.
#[inline]
fn classify(d: i32, q: i32, edge_clf: u8) -> usize {
    if edge_clf == 0 {
        if d > q {
            2
        } else if d < -q {
            0
        } else {
            1
        }
    } else if d < -q {
        0
    } else {
        1
    }
}

/// Build a border-extended luma plane (padding = `PAD`) the way
/// `extend_ccso_border` does: replicate the left/right edge pixels, then the
/// top/bottom rows. Returns `(buf, stride)` where the original pixel (x, y) sits
/// at `buf[(y + PAD) * stride + (x + PAD)]`.
pub(crate) fn extend_luma(recy: &[f32], w: usize, h: usize) -> (Vec<i32>, usize) {
    let stride = w + (PAD << 1);
    let mut buf = vec![0i32; stride * (h + (PAD << 1))];
    for y in 0..h {
        let dst = (y + PAD) * stride + PAD;
        for x in 0..w {
            buf[dst + x] = recy[y * w + x] as i32;
        }
        // left/right edge replicate
        let lv = buf[dst];
        let rv = buf[dst + w - 1];
        for x in 0..PAD {
            buf[dst - PAD + x] = lv;
            buf[dst + w + x] = rv;
        }
    }
    // top/bottom row replicate (full padded rows)
    for y in 0..PAD {
        let top_src = PAD * stride;
        let bot_src = (PAD + h - 1) * stride;
        let (a, b) = (y * stride, (PAD + h + y) * stride);
        buf.copy_within(top_src..top_src + stride, a);
        buf.copy_within(bot_src..bot_src + stride, b);
    }
    (buf, stride)
}

/// One chroma-plane edge-classified CCSO result.
#[derive(Clone)]
pub(crate) struct CcsoEdgeResult {
    pub(crate) scale_idx: u8,
    pub(crate) quant_idx: u8,
    pub(crate) ext_filter_support: u8,
    pub(crate) edge_clf: u8,
    pub(crate) max_band_log2: u8,
    /// Raw `ccso_offset` value per LUT entry, indexed `(band<<4)+(c0<<2)+c1`.
    /// Length = max_band * 16 (only the used `(c0, c1)` cells are non-trivial).
    pub(crate) offsets: Vec<i32>,
}

/// Co-located luma value (center) for chroma (x, y) in the extended buffer.
#[inline]
fn center(ext: &[i32], stride: usize, x: usize, y: usize, hsc: usize, vsc: usize) -> i32 {
    ext[((y << vsc) + PAD) * stride + ((x << hsc) + PAD)]
}

/// Search the best edge-classified CCSO LUT for one chroma plane. Returns `None`
/// if no configuration beats the unfiltered SSE.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_edge(
    ext: &[i32],
    stride: usize,
    src_c: &[f32],
    rec_c: &[f32],
    cw: usize,
    ch: usize,
    hsc: usize,
    vsc: usize,
    bit_depth: u32,
    activity: Option<&[f64]>,
) -> Option<CcsoEdgeResult> {
    let max_val = (1i32 << bit_depth) - 1;
    let mut base_sse = 0f64;
    for i in 0..cw * ch {
        let w = activity.map(|a| a[i]).unwrap_or(1.0);
        let d = rec_c[i] as f64 - src_c[i] as f64;
        base_sse += d * d * w;
    }
    let mut best: Option<(f64, CcsoEdgeResult)> = None;

    // Scan structure: for each (edge_clf, support, scale_idx, quant_idx) the
    // per-pixel classification (c0, c1) and the finest-resolution band are computed
    // ONCE, accumulating closed-form stats (Σe^2 w, Σe w, Σw) into 8-band bins. The
    // four max_band_log2 candidates are then evaluated by aggregating the fine bins
    // (coarser band = finer >> (3 - max_band_log2)), so the plane is scanned once
    // instead of four times. This mirrors AVM's ccso_pre_compute_class_err reuse.
    const FINE_BANDS: usize = 8; // 1 << 3 (max max_band_log2)
    for edge_clf in 0u8..2 {
        let ni = EDGE_INTERVAL[edge_clf as usize];
        for sup in 0u8..7 {
            let taps = tap_offsets(sup);
            for scale_idx in 0u8..4 {
                let scale = CCSO_SCALE[scale_idx as usize];
                let clamp_lo = -scale * 10;
                let clamp_hi = scale * 7;
                for quant_idx in 0u8..4 {
                    let q = QUANT_SZ[scale_idx as usize][quant_idx as usize] as i32;
                    if q == 0 {
                        continue; // unused table slot
                    }
                    let fine_shift = bit_depth - 3; // 8 bands
                    let nfine = FINE_BANDS * 16;
                    let mut se2 = vec![0f64; nfine];
                    let mut se = vec![0f64; nfine];
                    let mut sw = vec![0f64; nfine];
                    let mut corr: Vec<[f64; 8]> = Vec::new();
                    // Hoist the activity-weighting branch out of the per-pixel loop:
                    // when no proxy weighting is in effect (the common path) the weight
                    // is 1, so the accumulation drops a per-pixel Option check and three
                    // float multiplies.
                    let weighted = activity.is_some();
                    for y in 0..ch {
                        let row_lut_base = y * cw;
                        for x in 0..cw {
                            let c = center(ext, stride, x, y, hsc, vsc);
                            let lx = (x << hsc) + PAD;
                            let ly = (y << vsc) + PAD;
                            let t0 = ext[(ly as isize + taps[0].1) as usize * stride
                                + (lx as isize + taps[0].0) as usize];
                            let t1 = ext[(ly as isize + taps[1].1) as usize * stride
                                + (lx as isize + taps[1].0) as usize];
                            let c0 = classify(t0 - c, q, edge_clf);
                            let c1 = classify(t1 - c, q, edge_clf);
                            let fband = (c >> fine_shift) as usize;
                            let lut = (fband << 4) + (c0 << 2) + c1;
                            let r = rec_c[row_lut_base + x] as i32;
                            let s = src_c[row_lut_base + x] as f64;
                            let e = r as f64 - s;
                            if weighted {
                                let w = activity.unwrap()[row_lut_base + x];
                                se2[lut] += e * e * w;
                                se[lut] += e * w;
                                sw[lut] += w;
                            } else {
                                se2[lut] += e * e;
                                se[lut] += e;
                                sw[lut] += 1.0;
                            }
                            if r + clamp_lo < 0 || r + clamp_hi > max_val {
                                if corr.is_empty() {
                                    corr = vec![[0f64; 8]; nfine];
                                }
                                let w = if weighted {
                                    activity.unwrap()[row_lut_base + x]
                                } else {
                                    1.0
                                };
                                let cell = &mut corr[lut];
                                for (ci, &raw) in CCSO_OFFSET.iter().enumerate() {
                                    let off = raw * scale;
                                    let fc = (r + off).clamp(0, max_val) as f64 - s;
                                    let fu = e + off as f64;
                                    cell[ci] += (fc * fc - fu * fu) * w;
                                }
                            }
                        }
                    }
                    // Evaluate each band granularity by aggregating fine bins.
                    for &max_band_log2 in &[0u8, 1, 2, 3] {
                        let nband = 1usize << max_band_log2;
                        let agg = 3 - max_band_log2 as usize; // fine>>agg = coarse band
                        let nlut = nband * 16;
                        let mut offsets = vec![0i32; nlut];
                        let mut total = 0f64;
                        for b in 0..nband {
                            for c0 in 0..ni {
                                for c1 in 0..ni {
                                    let coarse = (b << 4) + (c0 << 2) + c1;
                                    // Sum the fine bins that map to this coarse band.
                                    let mut a2 = 0f64;
                                    let mut a1 = 0f64;
                                    let mut a0 = 0f64;
                                    let mut acorr = [0f64; 8];
                                    for fb in (b << agg)..((b + 1) << agg) {
                                        let fl = (fb << 4) + (c0 << 2) + c1;
                                        a2 += se2[fl];
                                        a1 += se[fl];
                                        a0 += sw[fl];
                                        if !corr.is_empty() {
                                            for ci in 0..8 {
                                                acorr[ci] += corr[fl][ci];
                                            }
                                        }
                                    }
                                    let mut best_e = f64::INFINITY;
                                    let mut best_ci = 0usize;
                                    for (ci, &raw) in CCSO_OFFSET.iter().enumerate() {
                                        let off = (raw * scale) as f64;
                                        let e = a2 + 2.0 * off * a1 + off * off * a0 + acorr[ci];
                                        if e < best_e {
                                            best_e = e;
                                            best_ci = ci;
                                        }
                                    }
                                    offsets[coarse] = CCSO_OFFSET[best_ci];
                                    total += best_e;
                                }
                            }
                        }
                        let better = match &best {
                            Some((bt, _)) => total < *bt,
                            None => true,
                        };
                        if better {
                            best = Some((
                                total,
                                CcsoEdgeResult {
                                    scale_idx,
                                    quant_idx,
                                    ext_filter_support: sup,
                                    edge_clf,
                                    max_band_log2,
                                    offsets,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    // Return the best filter whenever it reduces (weighted) SSE at all
    match best {
        Some((total, res)) if total + 1e-6 < base_sse => Some(res),
        _ => None,
    }
}

/// Unified per-plane CCSO result handed from the encoder search to the header
/// builder. Either a band-offset-only result or an edge-classified result.
#[derive(Clone)]
pub(crate) enum PlaneResult {
    Edge {
        scale_idx: u8,
        quant_idx: u8,
        ext_filter_support: u8,
        edge_clf: u8,
        max_band_log2: u8,
        offsets: Vec<i32>,
    },
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decide_blk_md(
    ext: &[i32],
    stride: usize,
    src_c: &[f32],
    rec_c: &[f32],
    cw: usize,
    ch: usize,
    hsc: usize,
    vsc: usize,
    bit_depth: u32,
    res: &CcsoEdgeResult,
    sb_cols: usize,
    sb_rows: usize,
    rd_mult: f64,
    plane: usize,
) -> (Vec<u8>, bool) {
    let max_val = (1i32 << bit_depth) - 1;
    let scale = CCSO_SCALE[res.scale_idx as usize];
    let q = QUANT_SZ[res.scale_idx as usize][res.quant_idx as usize] as i32;
    let single_band = res.max_band_log2 == 0;
    let shift_bits = bit_depth - res.max_band_log2 as u32;
    let taps = tap_offsets(res.ext_filter_support);
    // Chroma SB size in chroma pixels (64-px luma SB >> subsampling).
    let sb_cw = 64 >> hsc;
    let sb_ch = 64 >> vsc;
    let mut grid = vec![0u8; sb_cols * sb_rows];
    let mut any_on = false;
    // Running P(symbol 0 = "off") per context, in 1/32768. Seeded from the AVM
    // default ccso_cdf for this plane: ctx0 a0 = P(off), ctx2 a0 = P(off). The
    // default_ccso_cdf a0 values are the probability of symbol 0 directly. Plane 1
    // (U): ctx0 = 23470, ctx2 = 6666. Plane 2 (V): ctx0 = 22914, ctx2 = 6993. ctx1
    // and ctx3 are unused here (the bitstream context is only ever 0 or 2). The CDF
    // is adapted as the decision walks, mirroring AVM's update_cdf.
    let mut cdf_p0 = match plane {
        2 => [22914i32, 16384, 6993, 16384],
        _ => [23470i32, 16384, 6666, 16384],
    };
    for sr in 0..sb_rows {
        for sc in 0..sb_cols {
            let x0 = sc * sb_cw;
            let y0 = sr * sb_ch;
            let x1 = (x0 + sb_cw).min(cw);
            let y1 = (y0 + sb_ch).min(ch);
            let mut sse_off = 0f64; // unfiltered
            let mut sse_on = 0f64; // filtered
            for y in y0..y1 {
                for x in x0..x1 {
                    let s = src_c[y * cw + x] as f64;
                    let r = rec_c[y * cw + x] as i32;
                    let w = 1.0;
                    let du = r as f64 - s;
                    sse_off += du * du * w;
                    // filtered value
                    let c = center(ext, stride, x, y, hsc, vsc);
                    let lx = (x << hsc) + PAD;
                    let ly = (y << vsc) + PAD;
                    let t0 = ext[(ly as isize + taps[0].1) as usize * stride
                        + (lx as isize + taps[0].0) as usize];
                    let t1 = ext[(ly as isize + taps[1].1) as usize * stride
                        + (lx as isize + taps[1].0) as usize];
                    let c0 = classify(t0 - c, q, res.edge_clf);
                    let c1 = classify(t1 - c, q, res.edge_clf);
                    let band = if single_band {
                        0
                    } else {
                        (c >> shift_bits) as usize
                    };
                    let lut = (band << 4) + (c0 << 2) + c1;
                    let off = res.offsets[lut] * scale;
                    let f = (r + off).clamp(0, max_val) as f64;
                    let df = f - s;
                    sse_on += df * df;
                }
            }
            // RD decision with the real adaptive-CDF flag cost.
            let idx = sr * sb_cols + sc;
            let left_on = if sc == 0 { false } else { grid[idx - 1] != 0 };
            let ctx = if sc == 0 {
                0usize
            } else if left_on {
                2
            } else {
                0
            };
            // cdf_p0[ctx] is the running probability of symbol 0 (= "off"), in 1/32768.
            let p0 = cdf_p0[ctx] as f64 / 32768.0;
            let p1 = 1.0 - p0;
            let cost_off = -(p0.max(1e-6)).log2();
            let cost_on = -(p1.max(1e-6)).log2();
            let rd_off = sse_off + rd_mult * cost_off;
            let rd_on = sse_on + rd_mult * cost_on;
            let on = rd_on < rd_off;
            grid[idx] = on as u8;
            any_on |= on;
            // Adapt the CDF toward the chosen symbol (AVM update_cdf, rate ~ 1/16).
            let target0: f64 = if on { 0.0 } else { 32768.0 };
            cdf_p0[ctx] += ((target0 - cdf_p0[ctx] as f64) / 16.0) as i32;
            cdf_p0[ctx] = cdf_p0[ctx].clamp(64, 32768 - 64);
        }
    }
    (grid, any_on)
}

/// Apply the edge filter only to superblocks whose grid entry is on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_edge_gated(
    ext: &[i32],
    stride: usize,
    rec_c: &mut [f32],
    cw: usize,
    ch: usize,
    hsc: usize,
    vsc: usize,
    bit_depth: u32,
    res: &CcsoEdgeResult,
    grid: &[u8],
    sb_cols: usize,
) {
    let max_val = (1i32 << bit_depth) - 1;
    let scale = CCSO_SCALE[res.scale_idx as usize];
    let q = QUANT_SZ[res.scale_idx as usize][res.quant_idx as usize] as i32;
    let single_band = res.max_band_log2 == 0;
    let shift_bits = bit_depth - res.max_band_log2 as u32;
    let taps = tap_offsets(res.ext_filter_support);
    let sb_cw = 64 >> hsc;
    let sb_ch = 64 >> vsc;
    for y in 0..ch {
        let sr = y / sb_ch;
        for x in 0..cw {
            let sc = x / sb_cw;
            if grid[sr * sb_cols + sc] == 0 {
                continue;
            }
            let c = center(ext, stride, x, y, hsc, vsc);
            let lx = (x << hsc) + PAD;
            let ly = (y << vsc) + PAD;
            let t0 = ext
                [(ly as isize + taps[0].1) as usize * stride + (lx as isize + taps[0].0) as usize];
            let t1 = ext
                [(ly as isize + taps[1].1) as usize * stride + (lx as isize + taps[1].0) as usize];
            let c0 = classify(t0 - c, q, res.edge_clf);
            let c1 = classify(t1 - c, q, res.edge_clf);
            let band = if single_band {
                0
            } else {
                (c >> shift_bits) as usize
            };
            let lut = (band << 4) + (c0 << 2) + c1;
            let off = res.offsets[lut] * scale;
            let v = (rec_c[y * cw + x] as i32 + off).clamp(0, max_val);
            rec_c[y * cw + x] = v as f32;
        }
    }
}
