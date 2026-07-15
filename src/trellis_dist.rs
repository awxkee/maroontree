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

use std::sync::OnceLock;

use crate::cost::{coef_rate_bits, rate_cost};

pub(crate) type TrellisDistCurrentZeroScanFn =
    fn(&mut [f32], &mut [f32], &[f32], &[i32], &[u32], f32, f32);
pub(crate) type TrellisRoundDownScanFn = fn(&mut [i32], &[f32], &[u32], f32, f32, f32);

static TRELLIS_DIST_CURRENT_ZERO_SCAN: OnceLock<TrellisDistCurrentZeroScanFn> = OnceLock::new();
static TRELLIS_ROUND_DOWN_SCAN: OnceLock<TrellisRoundDownScanFn> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn trellis_dist_current_zero_scan_neon_wrap(
    dst_cur: &mut [f32],
    dst_zero: &mut [f32],
    tf: &[f32],
    cf: &[i32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
) {
    unsafe {
        crate::neon::trellis_dist_current_zero_scan_neon(
            dst_cur, dst_zero, tf, cf, scan, dc_q2, ac_q2,
        )
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn trellis_dist_current_zero_scan_avx2_wrap(
    dst_cur: &mut [f32],
    dst_zero: &mut [f32],
    tf: &[f32],
    cf: &[i32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
) {
    unsafe {
        crate::avx::trellis_dist_current_zero_scan_avx2(
            dst_cur, dst_zero, tf, cf, scan, dc_q2, ac_q2,
        )
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn trellis_round_down_scan_neon_wrap(
    cf: &mut [i32],
    tf: &[f32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    unsafe { crate::neon::trellis_round_down_scan_neon(cf, tf, scan, dc_q2, ac_q2, lambda) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn trellis_round_down_scan_avx2_wrap(
    cf: &mut [i32],
    tf: &[f32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    unsafe { crate::avx::trellis_round_down_scan_avx2(cf, tf, scan, dc_q2, ac_q2, lambda) }
}

#[inline]
pub(crate) fn trellis_dist_one(tf: &[f32], rc: usize, lev: i32, dc_q2: f32, ac_q2: f32) -> f32 {
    let dq2 = if rc == 0 { dc_q2 } else { ac_q2 };
    let e = tf[rc].abs() - lev.unsigned_abs() as f32;
    dq2 * (e * e)
}

#[inline]
pub(crate) fn trellis_dist_current_zero_scan(
    dst_cur: &mut [f32],
    dst_zero: &mut [f32],
    tf: &[f32],
    cf: &[i32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
) {
    debug_assert!(dst_cur.len() >= scan.len());
    debug_assert!(dst_zero.len() >= scan.len());
    debug_assert_eq!(scan.first().copied(), Some(0));
    let n = scan.len();
    let f = *TRELLIS_DIST_CURRENT_ZERO_SCAN.get_or_init(resolve_current_zero_scan);
    f(
        &mut dst_cur[..n],
        &mut dst_zero[..n],
        tf,
        cf,
        scan,
        dc_q2,
        ac_q2,
    );
}

#[inline]
pub(crate) fn trellis_round_down_scan(
    cf: &mut [i32],
    tf: &[f32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    debug_assert_eq!(scan.first().copied(), Some(0));
    let f = *TRELLIS_ROUND_DOWN_SCAN.get_or_init(resolve_round_down_scan);
    f(cf, tf, scan, dc_q2, ac_q2, lambda);
}

#[inline]
fn resolve_current_zero_scan() -> TrellisDistCurrentZeroScanFn {
    let mut _f: TrellisDistCurrentZeroScanFn = trellis_dist_current_zero_scan_scalar;
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        _f = trellis_dist_current_zero_scan_neon_wrap;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            _f = trellis_dist_current_zero_scan_avx2_wrap;
        }
    }
    _f
}

#[inline]
fn resolve_round_down_scan() -> TrellisRoundDownScanFn {
    let mut _f: TrellisRoundDownScanFn = trellis_round_down_scan_scalar;
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        _f = trellis_round_down_scan_neon_wrap;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            _f = trellis_round_down_scan_avx2_wrap;
        }
    }
    _f
}

pub(crate) fn trellis_dist_current_zero_scan_scalar(
    dst_cur: &mut [f32],
    dst_zero: &mut [f32],
    tf: &[f32],
    cf: &[i32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
) {
    for ((out_cur, out_zero), &rc32) in dst_cur.iter_mut().zip(dst_zero.iter_mut()).zip(scan.iter())
    {
        let rc = rc32 as usize;
        let dq2 = if rc == 0 { dc_q2 } else { ac_q2 };
        let at = tf[rc].abs();
        let e = at - cf[rc].unsigned_abs() as f32;
        *out_cur = dq2 * (e * e);
        *out_zero = dq2 * (at * at);
    }
}

pub(crate) fn trellis_round_down_scan_scalar(
    cf: &mut [i32],
    tf: &[f32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    for &rc32 in scan.iter() {
        let rc = rc32 as usize;
        let c = cf[rc];
        if c == 0 {
            continue;
        }
        let l = c.unsigned_abs();
        let cost_l =
            trellis_dist_one(tf, rc, l as i32, dc_q2, ac_q2) + rate_cost(lambda, coef_rate_bits(l));
        let cost_dn = trellis_dist_one(tf, rc, l as i32 - 1, dc_q2, ac_q2)
            + rate_cost(lambda, coef_rate_bits(l - 1));
        if cost_dn < cost_l {
            cf[rc] = if c < 0 { -(l as i32 - 1) } else { l as i32 - 1 };
        }
    }
}
