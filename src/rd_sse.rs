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
#![allow(clippy::too_many_arguments)]
use std::sync::OnceLock;

type ResidualPredFn = fn(&mut [i32], &[i32], &[i32], usize, usize, usize, usize, usize);
type ResidualDcFn = fn(&mut [i32], &[i32], usize, usize, usize, usize, usize, i32);
type SseReconFn = fn(&[i32], &[i32], &[i32], usize, usize, usize, usize, usize, i32) -> i64;
type SseReconDcFn = fn(i32, &[i32], &[i32], usize, usize, usize, usize, usize, i32) -> i64;

static RESIDUAL_PRED: OnceLock<ResidualPredFn> = OnceLock::new();
static RESIDUAL_DC: OnceLock<ResidualDcFn> = OnceLock::new();
static SSE_RECON: OnceLock<SseReconFn> = OnceLock::new();
static SSE_RECON_DC: OnceLock<SseReconDcFn> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn residual_pred_neon_wrap(
    dst: &mut [i32],
    pred: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) {
    unsafe { crate::neon::residual_pred_neon(dst, pred, src, stride, px, py, w, h) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn residual_pred_avx2_wrap(
    dst: &mut [i32],
    pred: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) {
    unsafe { crate::avx::residual_pred_avx2(dst, pred, src, stride, px, py, w, h) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn residual_dc_neon_wrap(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    unsafe { crate::neon::residual_dc_neon(dst, src, stride, px, py, w, h, dc) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn residual_dc_avx2_wrap(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    unsafe { crate::avx::residual_dc_avx2(dst, src, stride, px, py, w, h, dc) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn sse_recon_neon_wrap(
    pred: &[i32],
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    unsafe { crate::neon::sse_recon_neon(pred, resid, src, stride, px, py, w, h, maxv) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sse_recon_avx2_wrap(
    pred: &[i32],
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    unsafe { crate::avx::sse_recon_avx2(pred, resid, src, stride, px, py, w, h, maxv) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[allow(clippy::too_many_arguments)]
fn sse_recon_dc_neon_wrap(
    dc: i32,
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    unsafe { crate::neon::sse_recon_dc_neon(dc, resid, src, stride, px, py, w, h, maxv) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sse_recon_dc_avx2_wrap(
    dc: i32,
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    unsafe { crate::avx::sse_recon_dc_avx2(dc, resid, src, stride, px, py, w, h, maxv) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_pred(
    dst: &mut [i32],
    pred: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) {
    debug_assert!(dst.len() >= w * h);
    debug_assert!(pred.len() >= w * h);
    debug_assert!(px + w <= stride);
    debug_assert!((py + h - 1) * stride + px + w <= src.len());
    let f = *RESIDUAL_PRED.get_or_init(resolve_residual_pred);
    f(&mut dst[..w * h], &pred[..w * h], src, stride, px, py, w, h);
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_dc(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    debug_assert!(dst.len() >= w * h);
    debug_assert!(px + w <= stride);
    debug_assert!((py + h - 1) * stride + px + w <= src.len());
    let f = *RESIDUAL_DC.get_or_init(resolve_residual_dc);
    f(&mut dst[..w * h], src, stride, px, py, w, h, dc);
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_recon(
    pred: &[i32],
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    bd: u8,
) -> i64 {
    debug_assert!(pred.len() >= w * h);
    debug_assert!(resid.len() >= w * h);
    debug_assert!(px + w <= stride);
    debug_assert!((py + h - 1) * stride + px + w <= src.len());
    let maxv = (1i32 << bd) - 1;
    let f = *SSE_RECON.get_or_init(resolve_sse_recon);
    f(
        &pred[..w * h],
        &resid[..w * h],
        src,
        stride,
        px,
        py,
        w,
        h,
        maxv,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_recon_dc(
    dc: i32,
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    bd: u8,
) -> i64 {
    debug_assert!(resid.len() >= w * h);
    debug_assert!(px + w <= stride);
    debug_assert!((py + h - 1) * stride + px + w <= src.len());
    let maxv = (1i32 << bd) - 1;
    let f = *SSE_RECON_DC.get_or_init(resolve_sse_recon_dc);
    f(dc, &resid[..w * h], src, stride, px, py, w, h, maxv)
}

#[inline]
fn resolve_residual_pred() -> ResidualPredFn {
    let mut _f: ResidualPredFn = residual_pred_scalar;
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        _f = residual_pred_neon_wrap;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            _f = residual_pred_avx2_wrap;
        }
    }
    _f
}

#[inline]
fn resolve_residual_dc() -> ResidualDcFn {
    let mut _f: ResidualDcFn = residual_dc_scalar;
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        _f = residual_dc_neon_wrap;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            _f = residual_dc_avx2_wrap;
        }
    }
    _f
}

#[inline]
fn resolve_sse_recon() -> SseReconFn {
    let mut _f: SseReconFn = sse_recon_scalar;
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        _f = sse_recon_neon_wrap;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            _f = sse_recon_avx2_wrap;
        }
    }
    _f
}

#[inline]
fn resolve_sse_recon_dc() -> SseReconDcFn {
    let mut _f: SseReconDcFn = sse_recon_dc_scalar;
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        _f = sse_recon_dc_neon_wrap;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            _f = sse_recon_dc_avx2_wrap;
        }
    }
    _f
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_pred_scalar(
    dst: &mut [i32],
    pred: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) {
    for (ry, (drow, prow)) in dst
        .chunks_exact_mut(w)
        .zip(pred.chunks_exact(w))
        .take(h)
        .enumerate()
    {
        let srow = &src[(py + ry) * stride + px..][..w];
        for (d, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
            *d = s - p;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_dc_scalar(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    for (ry, drow) in dst.chunks_exact_mut(w).take(h).enumerate() {
        let srow = &src[(py + ry) * stride + px..][..w];
        for (d, &s) in drow.iter_mut().zip(srow.iter()) {
            *d = s - dc;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_recon_scalar(
    pred: &[i32],
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    let mut sse = 0i64;
    for (ry, (pred_row, resid_row)) in pred
        .chunks_exact(w)
        .zip(resid.chunks_exact(w))
        .take(h)
        .enumerate()
    {
        let srow = &src[(py + ry) * stride + px..][..w];
        for (&s, (&p, &e)) in srow.iter().zip(pred_row.iter().zip(resid_row.iter())) {
            let r = (p + e).clamp(0, maxv);
            let d = (s - r) as i64;
            sse += d * d;
        }
    }
    sse
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_recon_dc_scalar(
    dc: i32,
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    let mut sse = 0i64;
    for (ry, resid_row) in resid.chunks_exact(w).take(h).enumerate() {
        let srow = &src[(py + ry) * stride + px..][..w];
        for (&s, &e) in srow.iter().zip(resid_row.iter()) {
            let r = (dc + e).clamp(0, maxv);
            let d = (s - r) as i64;
            sse += d * d;
        }
    }
    sse
}
