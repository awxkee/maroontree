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
type ResidualPredFn = fn(&mut [i32], &[i32], &[u16], usize, usize, usize, usize, usize);
type ResidualDcFn = fn(&mut [i32], &[u16], usize, usize, usize, usize, usize, i32);
type ReconstructFn = fn(&mut [u16], usize, &mut [u16], usize, &[i32], &[i32], usize, usize, i32);
type SseReconFn = fn(&[i32], &[i32], &[u16], usize, usize, usize, usize, usize, i32) -> i64;
type SseU16Fn = fn(&[u16], usize, usize, usize, &[u16], usize, usize, usize, usize, usize) -> i64;
type SatdSadFn = fn(&[u16], usize, &[i32], usize, usize, usize) -> u64;
type LumaSatdFn = fn(&[u16], usize, usize, usize, usize, usize, i32, &[i32], i32, &[i32]) -> u64;
type SumI32Fn = fn(&[i32]) -> i32;
type SumU16Fn = fn(&[u16]) -> i32;
type SumU16StridedFn = fn(&[u16], usize, usize) -> i32;
type AllZeroI32Fn = fn(&[i32]) -> bool;

/// Pre-resolved low-level compute kernels used by the AV1 encoder state
/// machines. Those state machines decide what to evaluate; this table owns the
/// pixel/coefficient loops that perform the evaluation.
#[derive(Clone, Copy)]
pub(crate) struct RdDispatch {
    residual_pred: ResidualPredFn,
    residual_dc: ResidualDcFn,
    reconstruct: ReconstructFn,
    sse_recon: SseReconFn,
    sse_u16: SseU16Fn,
    satd_sad: SatdSadFn,
    luma_satd: LumaSatdFn,
    sum_i32: SumI32Fn,
    sum_u16: SumU16Fn,
    sum_u16_strided: SumU16StridedFn,
    all_zero_i32: AllZeroI32Fn,
}

impl RdDispatch {
    pub(crate) const fn scalar() -> Self {
        Self {
            residual_pred: residual_pred_scalar,
            residual_dc: residual_dc_scalar,
            reconstruct: reconstruct_scalar,
            sse_recon: sse_recon_scalar,
            sse_u16: sse_u16_scalar,
            satd_sad: satd_sad_proxy_scalar,
            luma_satd: crate::partition_rd::luma_satd_scalar,
            sum_i32: sum_i32_scalar,
            sum_u16: sum_u16_scalar,
            sum_u16_strided: sum_u16_strided_scalar,
            all_zero_i32: all_zero_i32_scalar,
        }
    }

    pub(crate) fn selected() -> Self {
        #[allow(unused_mut)]
        let mut dispatch = Self::scalar();
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            dispatch.residual_pred = residual_pred_neon_wrap;
            dispatch.residual_dc = residual_dc_neon_wrap;
            dispatch.reconstruct = reconstruct_neon_wrap;
            dispatch.sse_recon = sse_recon_neon_wrap;
            dispatch.sse_u16 = sse_u16_neon_wrap;
            dispatch.satd_sad = satd_sad_proxy_neon_wrap;
            dispatch.luma_satd = luma_satd_neon_wrap;
            dispatch.sum_i32 = sum_i32_neon_wrap;
            dispatch.sum_u16 = sum_u16_neon_wrap;
            dispatch.sum_u16_strided = sum_u16_strided_neon_wrap;
            dispatch.all_zero_i32 = all_zero_i32_neon_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            dispatch.residual_pred = residual_pred_avx2_wrap;
            dispatch.residual_dc = residual_dc_avx2_wrap;
            dispatch.reconstruct = reconstruct_avx2_wrap;
            dispatch.sse_recon = sse_recon_avx2_wrap;
            dispatch.sse_u16 = sse_u16_avx2_wrap;
            dispatch.satd_sad = satd_sad_proxy_avx2_wrap;
            dispatch.luma_satd = luma_satd_avx2_wrap;
            dispatch.sum_i32 = sum_i32_avx2_wrap;
            dispatch.sum_u16 = sum_u16_avx2_wrap;
            dispatch.sum_u16_strided = sum_u16_strided_avx2_wrap;
            dispatch.all_zero_i32 = all_zero_i32_avx2_wrap;
        }
        dispatch
    }

    #[inline]
    pub(crate) fn residual_pred(
        &self,
        dst: &mut [i32],
        pred: &[i32],
        src: &[u16],
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
        (self.residual_pred)(&mut dst[..w * h], &pred[..w * h], src, stride, px, py, w, h);
    }

    #[inline]
    pub(crate) fn residual_dc(
        &self,
        dst: &mut [i32],
        src: &[u16],
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
        (self.residual_dc)(&mut dst[..w * h], src, stride, px, py, w, h, dc);
    }

    #[inline]
    pub(crate) fn reconstruct(
        &self,
        dst: &mut [u16],
        dst_stride: usize,
        mirror: Option<(&mut [u16], usize)>,
        pred: &[i32],
        resid: &[i32],
        w: usize,
        h: usize,
        bd: u8,
    ) {
        debug_assert!(h == 0 || (h - 1) * dst_stride + w <= dst.len());
        debug_assert!(pred.len() >= w * h);
        debug_assert!(resid.is_empty() || resid.len() >= w * h);
        let resid = if resid.is_empty() {
            resid
        } else {
            &resid[..w * h]
        };
        if let Some((mirror, mirror_stride)) = mirror {
            debug_assert!(h == 0 || (h - 1) * mirror_stride + w <= mirror.len());
            (self.reconstruct)(
                dst,
                dst_stride,
                mirror,
                mirror_stride,
                &pred[..w * h],
                resid,
                w,
                h,
                (1i32 << bd) - 1,
            );
        } else {
            (self.reconstruct)(
                dst,
                dst_stride,
                &mut [],
                0,
                &pred[..w * h],
                resid,
                w,
                h,
                (1i32 << bd) - 1,
            );
        }
    }

    #[inline]
    pub(crate) fn sse_recon(
        &self,
        pred: &[i32],
        resid: &[i32],
        src: &[u16],
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
        (self.sse_recon)(
            &pred[..w * h],
            &resid[..w * h],
            src,
            stride,
            px,
            py,
            w,
            h,
            (1i32 << bd) - 1,
        )
    }

    #[inline]
    pub(crate) fn sse_u16(
        &self,
        src: &[u16],
        src_stride: usize,
        src_x: usize,
        src_y: usize,
        reference: &[u16],
        ref_stride: usize,
        ref_x: usize,
        ref_y: usize,
        w: usize,
        h: usize,
    ) -> i64 {
        debug_assert!(src_x + w <= src_stride);
        debug_assert!(ref_x + w <= ref_stride);
        debug_assert!(h == 0 || (src_y + h - 1) * src_stride + src_x + w <= src.len());
        debug_assert!(h == 0 || (ref_y + h - 1) * ref_stride + ref_x + w <= reference.len());
        (self.sse_u16)(
            src, src_stride, src_x, src_y, reference, ref_stride, ref_x, ref_y, w, h,
        )
    }

    #[inline]
    pub(crate) fn satd_sad_proxy(
        &self,
        src: &[u16],
        src_stride: usize,
        pred: &[i32],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u64 {
        debug_assert_eq!(w & 3, 0);
        debug_assert_eq!(h & 3, 0);
        debug_assert!((h - 1) * src_stride + w <= src.len());
        debug_assert!((h - 1) * pred_stride + w <= pred.len());
        (self.satd_sad)(src, src_stride, pred, pred_stride, w, h)
    }

    #[inline]
    pub(crate) fn luma_satd(
        &self,
        src: &[u16],
        stride: usize,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        bd: u8,
        pred: &[i32],
        dc: i32,
        residual: &[i32],
    ) -> u64 {
        debug_assert!(w.is_multiple_of(4) && h.is_multiple_of(4));
        debug_assert!(pred.is_empty() || pred.len() >= w * h);
        debug_assert!(residual.is_empty() || residual.len() >= w * h);
        debug_assert!((py + h - 1) * stride + px + w <= src.len());
        (self.luma_satd)(src, stride, px, py, w, h, (1 << bd) - 1, pred, dc, residual)
    }

    #[inline]
    pub(crate) fn sum_i32(&self, values: &[i32]) -> i32 {
        (self.sum_i32)(values)
    }

    #[inline]
    pub(crate) fn sum_u16(&self, values: &[u16]) -> i32 {
        (self.sum_u16)(values)
    }

    #[inline]
    pub(crate) fn sum_u16_strided(&self, values: &[u16], stride: usize, len: usize) -> i32 {
        debug_assert!(len == 0 || (len - 1) * stride < values.len());
        (self.sum_u16_strided)(values, stride, len)
    }

    #[inline]
    pub(crate) fn all_zero_i32(&self, values: &[i32]) -> bool {
        (self.all_zero_i32)(values)
    }

    /// Copy a rectangular image block into a packed scratch buffer. Row
    /// traversal lives here so callers only describe the block they need.
    #[inline]
    pub(crate) fn copy_block_u16(
        &self,
        dst: &mut [u16],
        src: &[u16],
        stride: usize,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
    ) {
        debug_assert!(dst.len() >= w * h);
        debug_assert!(px + w <= stride);
        debug_assert!(h == 0 || (py + h - 1) * stride + px + w <= src.len());
        for (row, dst) in dst[..w * h].chunks_exact_mut(w).enumerate() {
            dst.copy_from_slice(&src[(py + row) * stride + px..][..w]);
        }
    }

    /// Preserve a visible residual DC component after trellis quantization.
    /// This is shared by every transform shape instead of open-coding an
    /// integer reduction in each state-machine branch.
    #[inline]
    pub(crate) fn preserve_dc(&self, coefficient: &mut i32, residual: &[i32]) {
        debug_assert!(!residual.is_empty());
        let mean = self.sum_i32(residual) / residual.len() as i32;
        if *coefficient == 0 && mean.abs() >= 8 {
            *coefficient = mean.signum();
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn residual_pred_neon_wrap(
    dst: &mut [i32],
    pred: &[i32],
    src: &[u16],
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
    src: &[u16],
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
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    unsafe { crate::neon::residual_dc_neon(dst, src, stride, px, py, w, h, dc) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn reconstruct_neon_wrap(
    dst: &mut [u16],
    dst_stride: usize,
    mirror: &mut [u16],
    mirror_stride: usize,
    pred: &[i32],
    resid: &[i32],
    w: usize,
    h: usize,
    maxv: i32,
) {
    unsafe {
        crate::neon::reconstruct_neon(
            dst,
            dst_stride,
            mirror,
            mirror_stride,
            pred,
            resid,
            w,
            h,
            maxv,
        )
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn reconstruct_avx2_wrap(
    dst: &mut [u16],
    dst_stride: usize,
    mirror: &mut [u16],
    mirror_stride: usize,
    pred: &[i32],
    resid: &[i32],
    w: usize,
    h: usize,
    maxv: i32,
) {
    unsafe {
        crate::avx::reconstruct_avx2(
            dst,
            dst_stride,
            mirror,
            mirror_stride,
            pred,
            resid,
            w,
            h,
            maxv,
        )
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn residual_dc_avx2_wrap(
    dst: &mut [i32],
    src: &[u16],
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
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    unsafe { crate::neon::sse_recon_neon(pred, resid, src, stride, px, py, w, h, maxv) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn sse_u16_neon_wrap(
    src: &[u16],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    reference: &[u16],
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
    w: usize,
    h: usize,
) -> i64 {
    unsafe {
        crate::neon::sse_u16_neon(
            src, src_stride, src_x, src_y, reference, ref_stride, ref_x, ref_y, w, h,
        )
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sse_recon_avx2_wrap(
    pred: &[i32],
    resid: &[i32],
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    unsafe { crate::avx::sse_recon_avx2(pred, resid, src, stride, px, py, w, h, maxv) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sse_u16_avx2_wrap(
    src: &[u16],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    reference: &[u16],
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
    w: usize,
    h: usize,
) -> i64 {
    unsafe {
        crate::avx::sse_u16_avx2(
            src, src_stride, src_x, src_y, reference, ref_stride, ref_x, ref_y, w, h,
        )
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn sum_i32_neon_wrap(values: &[i32]) -> i32 {
    unsafe { crate::neon::sum_i32_neon(values) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn sum_u16_neon_wrap(values: &[u16]) -> i32 {
    unsafe { crate::neon::sum_u16_neon(values) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn sum_u16_strided_neon_wrap(values: &[u16], stride: usize, len: usize) -> i32 {
    unsafe { crate::neon::sum_u16_strided_neon(values, stride, len) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn all_zero_i32_neon_wrap(values: &[i32]) -> bool {
    unsafe { crate::neon::all_zero_i32_neon(values) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sum_i32_avx2_wrap(values: &[i32]) -> i32 {
    unsafe { crate::avx::sum_i32_avx2(values) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sum_u16_avx2_wrap(values: &[u16]) -> i32 {
    unsafe { crate::avx::sum_u16_avx2(values) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn sum_u16_strided_avx2_wrap(values: &[u16], stride: usize, len: usize) -> i32 {
    unsafe { crate::avx::sum_u16_strided_avx2(values, stride, len) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn all_zero_i32_avx2_wrap(values: &[i32]) -> bool {
    unsafe { crate::avx::all_zero_i32_avx2(values) }
}

#[inline]
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn satd_sad_proxy_neon_wrap(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    unsafe { crate::neon::satd_sad_proxy_neon(src, src_stride, pred, pred_stride, w, h) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn luma_satd_neon_wrap(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> u64 {
    unsafe { crate::neon::luma_satd_neon(src, stride, px, py, w, h, max_value, pred, dc, residual) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn satd_sad_proxy_avx2_wrap(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    unsafe { crate::avx::satd_sad_proxy_avx2(src, src_stride, pred, pred_stride, w, h) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn luma_satd_avx2_wrap(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> u64 {
    unsafe { crate::avx::luma_satd_avx2(src, stride, px, py, w, h, max_value, pred, dc, residual) }
}

pub(crate) fn satd_sad_proxy_scalar(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    #[inline]
    fn had4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
        let (e, f, g, h) = (a + c, a - c, b + d, b - d);
        [e + g, f + h, f - h, e - g]
    }
    let mut sad = 0u64;
    let mut satd = 0u64;
    for ty in (0..h).step_by(4) {
        for tx in (0..w).step_by(4) {
            let mut rows = [[0i32; 4]; 4];
            for r in 0..4 {
                let sr = &src[(ty + r) * src_stride + tx..];
                let pr = &pred[(ty + r) * pred_stride + tx..];
                let d: [i32; 4] = std::array::from_fn(|x| sr[x] as i32 - pr[x]);
                sad += d.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>();
                rows[r] = had4(d[0], d[1], d[2], d[3]);
            }
            #[allow(clippy::needless_range_loop)]
            for x in 0..4 {
                let col = had4(rows[0][x], rows[1][x], rows[2][x], rows[3][x]);
                satd += col.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>();
            }
        }
    }
    sad + (satd >> 2)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_pred_scalar(
    dst: &mut [i32],
    pred: &[i32],
    src: &[u16],
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
            *d = s as i32 - p;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_dc_scalar(
    dst: &mut [i32],
    src: &[u16],
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
            *d = s as i32 - dc;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    mirror: &mut [u16],
    mirror_stride: usize,
    pred: &[i32],
    resid: &[i32],
    w: usize,
    h: usize,
    maxv: i32,
) {
    let mirrored = !mirror.is_empty();
    for ry in 0..h {
        let dst_row = &mut dst[ry * dst_stride..][..w];
        let pred_row = &pred[ry * w..][..w];
        let resid_row = (!resid.is_empty()).then(|| &resid[ry * w..][..w]);
        for (rx, (dst, &prediction)) in dst_row.iter_mut().zip(pred_row).enumerate() {
            let residual = resid_row.map_or(0, |row| row[rx]);
            let reconstruction = (prediction + residual).clamp(0, maxv) as u16;
            *dst = reconstruction;
            if mirrored {
                mirror[ry * mirror_stride + rx] = reconstruction;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_recon_scalar(
    pred: &[i32],
    resid: &[i32],
    src: &[u16],
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
            let d = (s as i32 - r) as i64;
            sse += d * d;
        }
    }
    sse
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sse_u16_scalar(
    src: &[u16],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    reference: &[u16],
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
    w: usize,
    h: usize,
) -> i64 {
    let mut sse = 0i64;
    for row in 0..h {
        let src_row = &src[(src_y + row) * src_stride + src_x..][..w];
        let ref_row = &reference[(ref_y + row) * ref_stride + ref_x..][..w];
        for (&src, &reference) in src_row.iter().zip(ref_row) {
            let diff = i64::from(src) - i64::from(reference);
            sse += diff * diff;
        }
    }
    sse
}

pub(crate) fn sum_i32_scalar(values: &[i32]) -> i32 {
    values.iter().copied().sum()
}

pub(crate) fn sum_u16_scalar(values: &[u16]) -> i32 {
    values.iter().map(|&value| i32::from(value)).sum()
}

pub(crate) fn sum_u16_strided_scalar(values: &[u16], stride: usize, len: usize) -> i32 {
    values
        .iter()
        .step_by(stride)
        .take(len)
        .map(|&value| i32::from(value))
        .sum()
}

pub(crate) fn all_zero_i32_scalar(values: &[i32]) -> bool {
    values.iter().all(|&value| value == 0)
}

#[cfg(test)]
mod satd_tests {
    use super::*;

    #[test]
    fn residual_and_sse_simd_match_scalar_for_arbitrary_tails() {
        let dispatch = RdDispatch::selected();
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &(w, h) in &[
            (1usize, 1usize),
            (3, 5),
            (4, 7),
            (7, 4),
            (8, 8),
            (13, 9),
            (16, 32),
            (31, 17),
            (32, 32),
        ] {
            let stride = w + 9;
            let (px, py) = (3usize, 2usize);
            let image_len = (py + h + 1) * stride;
            let src: Vec<u16> = (0..image_len).map(|_| (next() % 4096) as u16).collect();
            let reference: Vec<u16> = (0..image_len).map(|_| (next() % 4096) as u16).collect();
            let pred: Vec<i32> = (0..w * h).map(|_| (next() % 4608) as i32 - 256).collect();
            let resid: Vec<i32> = (0..w * h).map(|_| (next() % 1024) as i32 - 512).collect();

            let mut want_residual = vec![0i32; w * h];
            let mut got_residual = vec![0i32; w * h];
            residual_pred_scalar(&mut want_residual, &pred, &src, stride, px, py, w, h);
            dispatch.residual_pred(&mut got_residual, &pred, &src, stride, px, py, w, h);
            assert_eq!(got_residual, want_residual, "residual {w}x{h}");

            let dc = (next() % 4096) as i32;
            residual_dc_scalar(&mut want_residual, &src, stride, px, py, w, h, dc);
            dispatch.residual_dc(&mut got_residual, &src, stride, px, py, w, h, dc);
            assert_eq!(got_residual, want_residual, "residual DC {w}x{h}");

            for &bd in &[8u8, 10, 12] {
                let max = (1 << bd) - 1;
                assert_eq!(
                    dispatch.sse_recon(&pred, &resid, &src, stride, px, py, w, h, bd),
                    sse_recon_scalar(&pred, &resid, &src, stride, px, py, w, h, max),
                    "reconstruction SSE {w}x{h} bd={bd}"
                );

                let dst_stride = w + 5;
                let mirror_stride = w + 3;
                let mut want_dst = vec![0xdead; dst_stride * h];
                let mut got_dst = want_dst.clone();
                let mut want_mirror = vec![0xbeef; mirror_stride * h];
                let mut got_mirror = want_mirror.clone();
                reconstruct_scalar(
                    &mut want_dst,
                    dst_stride,
                    &mut want_mirror,
                    mirror_stride,
                    &pred,
                    &resid,
                    w,
                    h,
                    max,
                );
                dispatch.reconstruct(
                    &mut got_dst,
                    dst_stride,
                    Some((&mut got_mirror, mirror_stride)),
                    &pred,
                    &resid,
                    w,
                    h,
                    bd,
                );
                assert_eq!(got_dst, want_dst, "reconstruction dst {w}x{h} bd={bd}");
                assert_eq!(
                    got_mirror, want_mirror,
                    "reconstruction mirror {w}x{h} bd={bd}"
                );

                let mut got_single = vec![0xdead; dst_stride * h];
                dispatch.reconstruct(&mut got_single, dst_stride, None, &pred, &resid, w, h, bd);
                assert_eq!(
                    got_single, want_dst,
                    "single reconstruction {w}x{h} bd={bd}"
                );

                let mut want_prediction = vec![0xdead; dst_stride * h];
                reconstruct_scalar(
                    &mut want_prediction,
                    dst_stride,
                    &mut [],
                    0,
                    &pred,
                    &[],
                    w,
                    h,
                    max,
                );
                let mut got_prediction = vec![0xdead; dst_stride * h];
                dispatch.reconstruct(&mut got_prediction, dst_stride, None, &pred, &[], w, h, bd);
                assert_eq!(
                    got_prediction, want_prediction,
                    "prediction-only reconstruction {w}x{h} bd={bd}"
                );
            }
            assert_eq!(
                dispatch.sse_u16(&src, stride, px, py, &reference, stride, px, py, w, h,),
                sse_u16_scalar(&src, stride, px, py, &reference, stride, px, py, w, h,),
                "u16 SSE {w}x{h}"
            );
        }
    }

    #[test]
    fn luma_partition_satd_simd_matches_static_scalar() {
        let dispatch = RdDispatch::selected();
        let mut state = 0xa076_1d64_78bd_642fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &(w, h) in &[
            (4usize, 4usize),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
        ] {
            let stride = w + 11;
            let (px, py) = (5usize, 3usize);
            let src: Vec<u16> = (0..(py + h + 1) * stride)
                .map(|_| (next() % 4096) as u16)
                .collect();
            let pred: Vec<i32> = (0..w * h).map(|_| (next() % 5120) as i32 - 512).collect();
            let residual: Vec<i32> = (0..w * h).map(|_| (next() % 1536) as i32 - 768).collect();
            let dc = (next() % 4096) as i32;
            for &bd in &[8u8, 10, 12] {
                let max_value = (1 << bd) - 1;
                for (pred, dc, residual, variant) in [
                    (&pred[..], 0, &residual[..], "pred+residual"),
                    (&pred[..], 0, &[][..], "prediction-only"),
                    (&[][..], dc, &residual[..], "dc+residual"),
                    (&[][..], dc, &[][..], "dc-only"),
                ] {
                    let want = crate::partition_rd::luma_satd_scalar(
                        &src, stride, px, py, w, h, max_value, pred, dc, residual,
                    );
                    let got =
                        dispatch.luma_satd(&src, stride, px, py, w, h, bd, pred, dc, residual);
                    assert_eq!(got, want, "{variant} {w}x{h} bd={bd}");
                }
            }
        }
    }

    #[test]
    fn reduction_and_copy_kernels_match_scalar() {
        let dispatch = RdDispatch::selected();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &len in &[0usize, 1, 3, 4, 7, 8, 15, 16, 31, 64, 127, 1024] {
            let signed: Vec<i32> = (0..len).map(|_| (next() % 4096) as i32 - 2048).collect();
            let pixels: Vec<u16> = (0..len).map(|_| (next() % 4096) as u16).collect();
            assert_eq!(dispatch.sum_i32(&signed), sum_i32_scalar(&signed));
            assert_eq!(dispatch.sum_u16(&pixels), sum_u16_scalar(&pixels));
            assert_eq!(dispatch.all_zero_i32(&signed), all_zero_i32_scalar(&signed));

            let zeros = vec![0i32; len];
            assert!(dispatch.all_zero_i32(&zeros));
            if len != 0 {
                for &position in &[0, len / 2, len - 1] {
                    let mut one = zeros.clone();
                    one[position] = if position & 1 == 0 { 1 } else { -1 };
                    assert!(!dispatch.all_zero_i32(&one), "len={len} pos={position}");
                }
            }
        }

        let image: Vec<u16> = (0..19 * 23).map(|_| (next() % 4096) as u16).collect();
        let mut packed = [0u16; 77];
        dispatch.copy_block_u16(&mut packed, &image, 23, 4, 3, 11, 7);
        for row in 0..7 {
            assert_eq!(
                &packed[row * 11..][..11],
                &image[(3 + row) * 23 + 4..][..11]
            );
        }

        for &(stride, len) in &[(1usize, 0usize), (1, 17), (2, 13), (5, 9), (17, 11)] {
            let values: Vec<u16> = (0..len.saturating_sub(1) * stride + 1)
                .map(|_| (next() % 4096) as u16)
                .collect();
            assert_eq!(
                dispatch.sum_u16_strided(&values, stride, len),
                sum_u16_strided_scalar(&values, stride, len)
            );
        }
    }

    /// The dispatched SIMD kernel must be BIT-IDENTICAL to the scalar proxy
    /// for every size/stride/value pattern (integer Hadamard is exact; the
    /// SIMD variant only reorders the separable passes, which the abs-sum
    /// cannot observe).
    #[test]
    fn satd_sad_proxy_simd_matches_scalar() {
        let dispatch = RdDispatch::selected();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &(w, h) in &[
            (4usize, 4usize),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
        ] {
            for &(src_stride, pred_stride) in &[(w, w), (w + 5, w), (97, w + 3), (w, 64)] {
                let src: Vec<u16> = (0..(h - 1) * src_stride + w + 8)
                    .map(|_| (next() % 4096) as u16)
                    .collect();
                let pred: Vec<i32> = (0..(h - 1) * pred_stride + w + 8)
                    .map(|_| (next() % 4096) as i32)
                    .collect();
                let want = satd_sad_proxy_scalar(&src, src_stride, &pred, pred_stride, w, h);
                let got = dispatch.satd_sad_proxy(&src, src_stride, &pred, pred_stride, w, h);
                assert_eq!(got, want, "{w}x{h} strides {src_stride}/{pred_stride}");
            }
        }
    }
}
