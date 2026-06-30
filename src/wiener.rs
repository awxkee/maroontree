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

//! AV1 loop-restoration **Wiener filter**, spec §7.17.
pub(crate) const FILTER_BITS: i32 = 7;

/// The three coded taps per axis have spec-defined ranges, sub-exponential `k`
/// parameters and reference midpoints. Index 0..=2 = innermost..outermost coded
/// tap (the 7-tap kernel is `[t0, t1, t2, centre, t2, t1, t0]`).
pub(crate) const WIENER_TAPS_MIN: [i32; 3] = [-5, -23, -17];
pub(crate) const WIENER_TAPS_MAX: [i32; 3] = [10, 8, 46];
pub(crate) const WIENER_TAPS_K: [i32; 3] = [1, 2, 3];
pub(crate) const WIENER_TAPS_MID: [i32; 3] = [3, -7, 15];

/// A full 7-tap symmetric kernel built from the three coded taps. `taps[3]` is
/// the derived centre so the kernel sums to `1 << FILTER_BITS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WienerKernel {
    pub(crate) taps: [i32; 7],
}

impl WienerKernel {
    /// Build the symmetric kernel from the three coded taps `[t0, t1, t2]`.
    pub(crate) fn from_coded(c: [i32; 3]) -> Self {
        let centre = (1 << FILTER_BITS) - 2 * (c[0] + c[1] + c[2]);
        WienerKernel {
            taps: [c[0], c[1], c[2], centre, c[2], c[1], c[0]],
        }
    }
}

/// The Wiener configuration for one restoration unit: horizontal + vertical
/// coded taps (each `[t0, t1, t2]`), or `None` for RESTORE_NONE (no filtering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WienerUnit {
    pub h: [i32; 3],
    pub v: [i32; 3],
}

/// Rounding constants for the two filter passes (spec §7.17.4). For 8-bit:
/// `round0 = 3`, `round1 = 11`. They always sum to `2*FILTER_BITS = 14` and are
/// adjusted by `(bd - 8)` so higher bit depths keep the same headroom.
#[inline]
fn inter_rounds(bd: u8) -> (i32, i32) {
    let cs = (bd - 8) as i32;
    // Spec: InterRound0 = 3 + cs (clamped so it never exceeds), InterRound1 =
    // 11 - cs. For 8/10/12-bit this gives (3,11)/(5,9)/(7,7).
    (3 + cs, 11 - cs)
}

#[inline]
fn clamp3(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Read a sample from `plane`, clamping coordinates to the plane edge (the AV1
/// Wiener filter extends the restoration-unit border by replication when the
/// stripe/frame boundary is reached). `w`/`h` are the plane dimensions.
#[inline]
fn get(plane: &[i32], stride: usize, w: usize, h: usize, x: i32, y: i32) -> i32 {
    let xx = x.clamp(0, w as i32 - 1) as usize;
    let yy = y.clamp(0, h as i32 - 1) as usize;
    plane[yy * stride + xx]
}

/// Apply a Wiener filter to the rectangle `[x0, x0+rw) x [y0, y0+rh)` of `plane`
/// (reading from `src`, writing to `dst`), using the separable kernel
/// `(hk, vk)`. Vertical context is clamped to `[ytop, ybot]` (inclusive), the
/// current restoration *stripe*: AV1 loop restoration processes the frame in
/// 64-row stripes and never lets the vertical filter read across a stripe
/// boundary — it replicates the stripe-edge row instead. Horizontal context is
/// clamped to the plane edges.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wiener_filter_rect(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    rw: usize,
    rh: usize,
    ytop: usize,
    ybot: usize,
    hk: &WienerKernel,
    vk: &WienerKernel,
    bd: u8,
) {
    let (round0, round1) = inter_rounds(bd);
    let bdi = bd as i32;
    let maxv = (1i32 << bd) - 1;
    let offset = 1i32 << (bdi + FILTER_BITS - 1);
    let limit = (1i32 << (bdi + 1 + FILTER_BITS - round0)) - 1;

    // Horizontal pass into an intermediate buffer covering the rect plus 3 rows
    // of vertical context above and below, with the source row clamped to the
    // stripe `[ytop, ybot]` so the vertical pass can't cross the boundary.
    let pad = 3usize;
    let ih = rh + 2 * pad;
    let mut inter = vec![0i32; ih * rw];
    for r in 0..ih {
        let raw_sy = y0 as i32 + r as i32 - pad as i32;
        let sy = raw_sy.clamp(ytop as i32, ybot as i32);
        for c in 0..rw {
            let sx = x0 as i32 + c as i32;
            let mut s = 0i32;
            for t in 0..7 {
                let px = get(src, stride, w, h, sx + t as i32 - 3, sy);
                s += hk.taps[t] * px;
            }
            let v = clamp3((s + (1 << (round0 - 1)) + offset) >> round0, 0, limit);
            inter[r * rw + c] = v;
        }
    }

    let round1_offset = 1i32 << (round1 - 1);
    let offset_correction = offset << (FILTER_BITS - round0);
    for r in 0..rh {
        for c in 0..rw {
            let mut s = 0i32;
            for t in 0..7 {
                let iy = r as i32 + t as i32;
                s += vk.taps[t] * inter[iy as usize * rw + c];
            }
            let v = (s - offset_correction + round1_offset) >> round1;
            dst[(y0 + r) * stride + (x0 + c)] = v.clamp(0, maxv);
        }
    }
}

/// Apply a single global Wiener filter to a whole luma plane, honoring AV1's
/// 64-row restoration stripes. The first stripe is rows `0..=55` (height 56);
/// subsequent stripes are 64 rows starting at 56 (`56..=119`, `120..=183`, ...).
/// The vertical filter context is clamped within each stripe.
pub(crate) fn wiener_filter_plane(
    dst: &mut [i32],
    src: &[i32],
    w: usize,
    h: usize,
    hk: &WienerKernel,
    vk: &WienerKernel,
    bd: u8,
) {
    let mut ytop = 0usize;
    while ytop < h {
        // Stripe height: 56 for the first stripe, 64 thereafter.
        let stripe_h = if ytop == 0 { 56 } else { 64 };
        let ybot = (ytop + stripe_h).min(h);
        // AV1 makes 2 rows of the neighbouring stripe available to the vertical
        // filter (the saved boundary lines); the 3rd tap row is replicated.
        let ctop = ytop.saturating_sub(2);
        let cbot = (ybot + 2).min(h) - 1;
        wiener_filter_rect(
            dst,
            src,
            w,
            w,
            h,
            0,
            ytop,
            w,
            ybot - ytop,
            ctop,
            cbot,
            hk,
            vk,
            bd,
        );
        ytop = ybot;
    }
}
