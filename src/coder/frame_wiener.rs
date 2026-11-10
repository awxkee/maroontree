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

// Frame-level luma Wiener (loop restoration) search.

#[allow(clippy::too_many_arguments)]
fn frame_wiener_search(
    recon: &[u16],
    src: &[u16],
    w: usize,
    h: usize,
    bd: u8,
    pool: &Pool,
    rd: &crate::rd_sse::RdDispatch,
    lambda_d: f32,
) -> Option<crate::wiener::WienerUnit> {
    use crate::wiener::{WienerKernel, wiener_filter_plane_sse, wiener_filter_plane_wsse};
    // Unification V7: the decision on normalized wSSE + lambda_d x the LR
    // syntax (one `use_wiener` per 64x64 unit plus the first unit's taps;
    // later units delta-code zero against the reference). Identity = 0 bits.
    let wsse = crate::tuning::get().wiener_wsse;
    let base = if wsse {
        let mut err = vec![0i32; w * h];
        for (e, (&r, &s)) in err.iter_mut().zip(recon.iter().zip(src.iter())) {
            *e = i32::from(r) - i32::from(s);
        }
        rd.wsse_err(&err, w, h)
    } else {
        plane_sse_vis(recon, src, w, w, h)
    };
    let count_units = |frame: usize| -> usize { 1.max((frame + 32) / 64) };
    let lr_bits = if wsse {
        (count_units(w) * count_units(h)) as f32 + 6.0 * 5.0
    } else {
        0.0
    };
    let rate = (lambda_d * lr_bits).round() as i64;
    // Small candidate set of separable low-pass Wiener kernels (coded taps
    // [t0,t1,t2]); the identity is implicit via `base`. These span gentle to
    // moderate smoothing within the spec tap ranges.
    const CANDS: [[i32; 3]; 4] = [[0, 0, 1], [-1, 2, 2], [0, 1, 3], [1, -3, 5]];
    let cands: Vec<(&[i32; 3], &[i32; 3])> = CANDS
        .iter()
        .flat_map(|h_taps| CANDS.iter().map(move |v_taps| (h_taps, v_taps)))
        .collect();
    // Each candidate retains one stripe; the reduce below walks the original
    // (h, v) order so ties break exactly as the sequential loop did.
    let want = pool.width().min(cands.len());
    let sses: Vec<i64> = pool.map_indexed(want, cands.len(), |i| {
        let (h_taps, v_taps) = cands[i];
        let hk = WienerKernel::from_coded(*h_taps);
        let vk = WienerKernel::from_coded(*v_taps);
        if wsse {
            wiener_filter_plane_wsse(recon, src, w, h, &hk, &vk, bd, rd) + rate
        } else {
            wiener_filter_plane_sse(recon, src, w, h, &hk, &vk, bd)
        }
    });
    let mut best: Option<(i64, crate::wiener::WienerUnit)> = None;
    for (&(h_taps, v_taps), &s) in cands.iter().zip(sses.iter()) {
        if s < base && best.as_ref().is_none_or(|b| s < b.0) {
            best = Some((
                s,
                crate::wiener::WienerUnit {
                    h: *h_taps,
                    v: *v_taps,
                },
            ));
        }
    }
    best.map(|b| b.1)
}
