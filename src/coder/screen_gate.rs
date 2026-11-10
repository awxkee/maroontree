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

// Frame-level screen-content gate (IntraBC coverage / few-colour verdicts).

/// Frame-level screen-content verdicts from `screen_frame_gate`.
#[derive(Default, Clone, Copy, Debug)]
struct ScreenGate {
    /// IntraBC is expected to save more than the loop filters it costs.
    ibc_pays: bool,
    /// Enough few-colour (palette-exact) 16x16 blocks for the text palette
    /// tools (map Viterbi, colour-cache pricing) to be worth their time.
    few_colour: bool,
}

/// Decide the two frame-level screen-content bits with a rate model instead
/// of a duplicate count.
///
/// IntraBC costs the whole frame its loop filters, so it has to pay for
/// itself: the saving is the texture mass of every aligned 8x8 luma block that
/// has an EXACT earlier copy anywhere in the frame (unaligned — periodic
/// screentones / UI patterns rarely sit on a 16-grid, which is why the old
/// aligned-16x16 duplicate count never fired on line art and left s_manga 40%
/// behind libaom), less a per-copy DV overhead; the loss is a fraction of the
/// texture mass of the photo-like area (neither few-colour nor copied) that
/// the deblocking filter would have cleaned. Texture mass = SAD to the block
/// mean, the same quantity on both sides, so the quantiser cancels except in
/// the DV overhead term. Exact copies come from the IntraBC hash index
/// (`LossyIbcIndex`, every 8x8 window), built only when the cheap few-colour
/// pre-screen says the frame could be screen content at all, so photos pay
/// one histogram pass and nothing else.
///
/// `few_colour` is libaom's screen-content test in our palette's terms: the
/// share of 16x16 blocks that are palette-exact (2..=8 luma values with a
/// span of at least 64): the text palette tools are priced for that content
/// and measured as noise on photos.
#[allow(clippy::too_many_arguments)]
fn screen_frame_gate(
    src: &[Vec<u16>; 3],
    w8: usize,
    h8: usize,
    sub_x: usize,
    sub_y: usize,
    bd: u8,
    base_q_idx: u8,
    pool: &crate::par::Pool,
) -> ScreenGate {
    let t = crate::tuning::get();
    let luma = &src[0];
    let (n8x, n8y) = (w8 / 8, h8 / 8);
    if n8x < 2 || n8y < 2 {
        return ScreenGate::default();
    }
    // Stage 1: per 16x16 block, is it palette-exact (few colours), and how
    // many colours does it hold at all (lenient pre-screen for stage 2).
    let (n16x, n16y) = (w8 / 16, h8 / 16);
    let span_min = 64u32 << (bd - 8);
    // Per 16-row band: palette-exact / lenient 16x16 flags plus the window
    // hash of every non-flat aligned 8x8 (a cheap aligned-duplicate count
    // completes the pre-screen for UI/text frames whose antialiased 16x16
    // blocks hold more than 16 colours).
    let bands = pool.map_indexed(pool.width(), n16y, |row| -> (Vec<bool>, u32, Vec<u32>) {
        let mut few = Vec::with_capacity(n16x);
        let mut lenient = 0u32;
        let mut hashes = Vec::with_capacity(n16x * 4);
        for col in 0..n16x {
            let mut is_few = false;
            if let Some(hist) = block_color_histogram(luma, w8, col * 16, row * 16, 16, 16) {
                let values = hist.as_slice().iter().map(|&(v, _)| v);
                let span = (values.clone().max().unwrap_or(0) - values.min().unwrap_or(0)) as u32;
                if span >= span_min {
                    lenient += u32::from(hist.len() <= 16);
                    is_few = hist.len() <= 8;
                }
            }
            few.push(is_few);
            for (ox, oy) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
                let (x, y) = (col * 16 + ox, row * 16 + oy);
                let first = luma[y * w8 + x];
                if (0..8).any(|j| luma[(y + j) * w8 + x..][..8].iter().any(|&v| v != first)) {
                    hashes.push(LossyIbcIndex::hash8(luma, w8, x, y));
                }
            }
        }
        (few, lenient, hashes)
    });
    let total16 = (n16x * n16y).max(1) as f32;
    let few16 = bands
        .iter()
        .map(|b| b.0.iter().filter(|&&f| f).count())
        .sum::<usize>() as u32;
    let lenient16: u32 = bands.iter().map(|b| b.1).sum();
    let few_share = few16 as f32 / total16;
    let few_colour = few_share >= t.screen_few_share;
    let mut seen: HashSet<u32> = HashSet::with_capacity(n16x * n16y * 4);
    let dup8 = bands
        .iter()
        .flat_map(|b| b.2.iter())
        .filter(|&&h| !seen.insert(h))
        .count() as f32
        / (4.0 * total16);
    drop(seen);
    let lenient_share = lenient16 as f32 / total16;
    if lenient_share < t.ibc_gate_prescreen && dup8 < t.ibc_gate_prescreen {
        return ScreenGate {
            ibc_pays: false,
            few_colour,
        };
    }
    // Stage 2: copy gain per 16x16 block. An exact earlier 8x8 window (the
    // IntraBC search's own anchor) proposes a DV; the block's gain is its
    // texture mass minus what copying the whole 16x16 along that DV leaves
    // as residual (text repeats exactly at the glyph and nearly around it),
    // minus the DV overhead. Blocks that gain nothing and are not palette
    // territory are the photo-like mass that pays for the lost loop filters;
    // that loss scales with the deblocking level the q-law would apply.
    let index = LossyIbcIndex::build(src, w8, h8, (sub_x, sub_y));
    let dv_cost = (t.ibc_gate_dv * f32::from(crate::quant::ac_q(base_q_idx, bd)).powi(2)) as u64;
    // Texture mass = gradient energy (squared neighbour differences): the
    // bit-cost proxy that prices a sharp edge above a smooth ramp of the same
    // swing, as the transform coder does, and is commensurate with the SSE
    // a copy leaves behind.
    let sad8 = |x: usize, y: usize| -> u64 {
        let mut e = 0u64;
        for j in 0..8 {
            let row = &luma[(y + j) * w8 + x..][..8];
            e += row
                .array_windows::<2>()
                .map(|p| u64::from(p[0].abs_diff(p[1])).pow(2))
                .sum::<u64>();
            if j > 0 {
                let up = &luma[(y + j - 1) * w8 + x..][..8];
                e += row
                    .iter()
                    .zip(up)
                    .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                    .sum::<u64>();
            }
        }
        e
    };
    let rows = pool.map_indexed(pool.width(), n16y, |row| -> (u64, u64, u32) {
        let py = row * 16;
        let (mut gain, mut photo, mut copied) = (0u64, 0u64, 0u32);
        for (col, &parent_few) in bands[row].0.iter().enumerate() {
            let px = col * 16;
            let subs = [(0usize, 0usize), (8, 0), (0, 8), (8, 8)];
            let mass: u64 = subs.iter().map(|&(ox, oy)| sad8(px + ox, py + oy)).sum();
            if mass == 0 {
                continue;
            }
            // Best copy residual over the anchors' DVs (first few exact,
            // earlier hits per sub-block; the search itself ranks further).
            let mut best_res = u64::MAX;
            for &(ox, oy) in &subs {
                let (sx, sy) = (px + ox, py + oy);
                let exact = |ax: usize, ay: usize| {
                    (0..8)
                        .all(|j| luma[(ay + j) * w8 + ax..][..8] == luma[(sy + j) * w8 + sx..][..8])
                };
                let earlier = |ax: usize, ay: usize| ay + 8 <= sy || (ay <= sy && ax + 8 <= sx);
                for (ax, ay) in index
                    .candidates(luma, w8, sx, sy, 8, sy)
                    .filter(|&(ax, ay)| earlier(ax, ay) && exact(ax, ay))
                    .take(3)
                {
                    // DV of the whole 16x16 along this anchor's offset.
                    let (Some(rx), Some(ry)) =
                        ((px + ax).checked_sub(sx), (py + ay).checked_sub(sy))
                    else {
                        continue;
                    };
                    if rx + 16 > w8
                        || ry + 16 > h8
                        || (ry + 16 > py && rx + 16 > px && ry < py + 16 && rx < px + 16)
                    {
                        continue;
                    }
                    let res: u64 = (0..16)
                        .map(|j| {
                            luma[(py + j) * w8 + px..][..16]
                                .iter()
                                .zip(&luma[(ry + j) * w8 + rx..][..16])
                                .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                                .sum::<u64>()
                        })
                        .sum();
                    best_res = best_res.min(res);
                    if best_res == 0 {
                        break;
                    }
                }
                if best_res == 0 {
                    break;
                }
            }
            let g = if best_res == u64::MAX {
                0
            } else {
                mass.saturating_sub(best_res).saturating_sub(dv_cost)
            };
            if g > 0 {
                gain += g;
                copied += 1;
            } else if !parent_few {
                // Photo-like = what the intra coder transform-codes and the
                // deblocker then cleans: not palette territory at 16x16, and
                // per 8x8 not a text fragment either (<= 12 values spanning
                // >= 64: glyph + background + antialiasing). Smooth gradients
                // (small range) stay in — that is where blocking shows.
                for &(ox, oy) in &subs {
                    let text = block_color_histogram(luma, w8, px + ox, py + oy, 8, 8).is_some_and(
                        |hist| {
                            let values = hist.as_slice().iter().map(|&(v, _)| v);
                            let span = (values.clone().max().unwrap_or(0)
                                - values.min().unwrap_or(0))
                                as u32;
                            hist.len() <= 12 && span >= span_min
                        },
                    );
                    if !text && sad8(px + ox, py + oy) > 0 {
                        photo += 1;
                    }
                }
            }
        }
        (gain, photo, copied)
    });
    let gain: u64 = rows.iter().map(|r| r.0).sum();
    let photo: u64 = rows.iter().map(|r| r.1).sum();
    let copied: u32 = rows.iter().map(|r| r.2).sum();
    // Deblocking is worth roughly a fixed SSE per filtered block boundary
    // step, i.e. per photo-like block it scales with the quantizer step (not
    // its square: at coarse q the filter also costs detail), and nothing at
    // all once the q-law switches the filter off.
    let ac_q = f64::from(crate::quant::ac_q(base_q_idx, bd));
    let lvl_y = crate::obu::loop_filter_levels(base_q_idx, sub_x + sub_y).0;
    let loss = if lvl_y > 0 {
        f64::from(t.ibc_gate_beta) * photo as f64 * ac_q
    } else {
        0.0
    };
    let ibc_pays = copied >= 4 && gain as f64 > loss;
    ScreenGate {
        ibc_pays,
        few_colour,
    }
}
