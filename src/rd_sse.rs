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
type ChromaSseFn = fn(&[u16], usize, usize, usize, usize, usize, i32, &[i32], i32, &[i32]) -> i64;
type SumI32Fn = fn(&[i32]) -> i32;
type SumU16Fn = fn(&[u16]) -> i32;
type SumU16StridedFn = fn(&[u16], usize, usize) -> i32;
type AllZeroI32Fn = fn(&[i32]) -> bool;
/// `(sum, sum of squares)` of `src - pred` over a `w`x`h` block.
type ResidualMomentsFn = fn(&[u16], usize, &[i32], usize, usize, usize) -> (i64, i64);

/// Largest block any distortion kernel sees (64x64).
const MAX_BLOCK_PIXELS: usize = 64 * 64;

/// Read-only strided view of one `w`x`h` block of a source plane — no copy.
///
/// The encoder codes frames on the 8-aligned AV1 block grid, so blocks on the
/// right/bottom frame edge cover padding columns/rows the decoder crops away.
/// `vis_w`x`vis_h` is the part that is actually displayed; every distortion
/// measured through a `SrcBlock` covers only that part, so RD decisions never
/// trade visible error for error on pixels nobody sees. Interior blocks have
/// `vis == (w, h)`.
#[derive(Clone, Copy)]
pub(crate) struct SrcBlock<'a> {
    /// Plane samples starting at the block origin.
    data: &'a [u16],
    stride: usize,
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
}

impl<'a> SrcBlock<'a> {
    /// The `w`x`h` block at `(x, y)` of `plane` (row pitch `stride`), whose
    /// displayed region is `plane_vis_w`x`plane_vis_h` (plane coordinates).
    #[inline]
    pub(crate) fn new(
        plane: &'a [u16],
        stride: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        plane_vis_w: usize,
        plane_vis_h: usize,
    ) -> Self {
        debug_assert!(w != 0 && h != 0 && w * h <= MAX_BLOCK_PIXELS);
        // A row running past `stride` would silently wrap into the next row;
        // the slice below already bounds-checks the plane itself.
        debug_assert!(
            x + w <= stride,
            "SrcBlock {w}x{h} at x={x} exceeds stride {stride}"
        );
        let start = y * stride + x;
        Self {
            data: &plane[start..start + (h - 1) * stride + w],
            stride,
            w,
            h,
            vis_w: plane_vis_w.saturating_sub(x).min(w),
            vis_h: plane_vis_h.saturating_sub(y).min(h),
        }
    }

    #[inline]
    pub(crate) fn w(&self) -> usize {
        self.w
    }

    #[inline]
    pub(crate) fn h(&self) -> usize {
        self.h
    }

    /// Displayed `(columns, rows)` of this block.
    #[inline]
    pub(crate) fn vis(&self) -> (usize, usize) {
        (self.vis_w, self.vis_h)
    }

    /// Whether part of the block lies outside the displayed frame.
    #[inline]
    pub(crate) fn is_clipped(&self) -> bool {
        self.vis_w < self.w || self.vis_h < self.h
    }

    /// Row pitch of [`Self::data`].
    #[inline]
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Samples from the block origin, `stride`-pitched (rows are `w` long).
    #[inline]
    pub(crate) fn data(&self) -> &'a [u16] {
        self.data
    }

    /// Row `y` of the block (all `w` samples, visible or not).
    #[inline]
    pub(crate) fn row(&self, y: usize) -> &'a [u16] {
        assert!(y < self.h);
        &self.data[y * self.stride..][..self.w]
    }

    /// Pack the block `w`-pitched into `out`, replacing every invisible sample
    /// with `fill(x, y)` so it contributes nothing to a difference against
    /// that value.
    fn masked_into(&self, out: &mut [u16], fill: impl Fn(usize, usize) -> i32) {
        for y in 0..self.h {
            let dst = &mut out[y * self.w..][..self.w];
            if y < self.vis_h {
                dst[..self.vis_w].copy_from_slice(&self.row(y)[..self.vis_w]);
                for (x, d) in dst.iter_mut().enumerate().skip(self.vis_w) {
                    *d = fill(x, y) as u16;
                }
            } else {
                for (x, d) in dst.iter_mut().enumerate() {
                    *d = fill(x, y) as u16;
                }
            }
        }
    }
}

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
    chroma_sse: ChromaSseFn,
    sum_i32: SumI32Fn,
    sum_u16: SumU16Fn,
    sum_u16_strided: SumU16StridedFn,
    all_zero_i32: AllZeroI32Fn,
    residual_moments: ResidualMomentsFn,
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
            chroma_sse: chroma_sse_scalar,
            sum_i32: sum_i32_scalar,
            sum_u16: sum_u16_scalar,
            sum_u16_strided: sum_u16_strided_scalar,
            all_zero_i32: all_zero_i32_scalar,
            residual_moments: residual_moments_scalar,
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
            dispatch.chroma_sse = chroma_sse_neon_wrap;
            dispatch.sum_i32 = sum_i32_neon_wrap;
            dispatch.sum_u16 = sum_u16_neon_wrap;
            dispatch.sum_u16_strided = sum_u16_strided_neon_wrap;
            dispatch.all_zero_i32 = all_zero_i32_neon_wrap;
            dispatch.residual_moments = residual_moments_neon_wrap;
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
            dispatch.chroma_sse = chroma_sse_avx2_wrap;
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

    /// `src - pred` over the WHOLE `w`x`h` block (transform input: padding
    /// included — only distortion is clipped to the visible region).
    #[inline]
    pub(crate) fn residual_pred_blk(&self, dst: &mut [i32], pred: &[i32], src: SrcBlock<'_>) {
        self.residual_pred(dst, pred, src.data, src.stride, 0, 0, src.w, src.h);
    }

    /// `src - dc` over the WHOLE `w`x`h` block (transform input).
    #[inline]
    pub(crate) fn residual_dc_blk(&self, dst: &mut [i32], src: SrcBlock<'_>, dc: i32) {
        self.residual_dc(dst, src.data, src.stride, 0, 0, src.w, src.h, dc);
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

    /// SSE of `clamp(pred + resid)` against the VISIBLE part of `src`.
    ///
    /// `pred`/`resid` are packed `src.w()`-wide. Fully visible and row-clipped
    /// blocks run the SIMD kernel; a column-clipped block (right frame edge
    /// only) takes the strided scalar path.
    #[inline]
    pub(crate) fn sse_recon(&self, pred: &[i32], resid: &[i32], src: SrcBlock<'_>, bd: u8) -> i64 {
        let (w, h) = (src.w, src.h);
        debug_assert!(pred.len() >= w * h);
        debug_assert!(resid.len() >= w * h);
        let (vw, vh) = (src.vis_w, src.vis_h);
        if vw == 0 || vh == 0 {
            return 0;
        }
        let maxv = (1i32 << bd) - 1;
        if vw == w {
            self.sse_recon_raw(pred, resid, src.data, src.stride, 0, 0, w, vh, maxv)
        } else {
            sse_recon_strided_scalar(pred, resid, w, src.data, src.stride, vw, vh, maxv)
        }
    }

    #[inline]
    fn sse_recon_raw(
        &self,
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
            maxv,
        )
    }

    /// SSE of the VISIBLE part of `src` against the same-sized rectangle of
    /// `reference` at `(ref_x, ref_y)`.
    #[inline]
    pub(crate) fn sse_u16(
        &self,
        src: SrcBlock<'_>,
        reference: &[u16],
        ref_stride: usize,
        ref_x: usize,
        ref_y: usize,
    ) -> i64 {
        let (vw, vh) = (src.vis_w, src.vis_h);
        if vw == 0 || vh == 0 {
            return 0;
        }
        self.sse_u16_raw(
            src.data, src.stride, 0, 0, reference, ref_stride, ref_x, ref_y, vw, vh,
        )
    }

    #[inline]
    fn sse_u16_raw(
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

    /// SAD + SATD/4 ranking proxy of `src - pred` over the VISIBLE part of
    /// `src`. Hadamard tiles cannot be cut, so an edge block is scored on a
    /// stack copy whose invisible pixels equal the prediction (zero residual).
    #[inline]
    pub(crate) fn satd_sad_proxy(
        &self,
        src: SrcBlock<'_>,
        pred: &[i32],
        pred_stride: usize,
    ) -> u64 {
        let (w, h) = (src.w, src.h);
        if !src.is_clipped() {
            return self.satd_sad_proxy_raw(src.data, src.stride, pred, pred_stride, w, h);
        }
        let mut masked = [0u16; MAX_BLOCK_PIXELS];
        src.masked_into(&mut masked, |x, y| pred[y * pred_stride + x]);
        self.satd_sad_proxy_raw(&masked, w, pred, pred_stride, w, h)
    }

    #[inline]
    fn satd_sad_proxy_raw(
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

    /// First two moments of the prediction residual over the VISIBLE part of
    /// `src`, plus that pixel count: `(S1, S2, n)` with `S1 = sum(src - pred)`,
    /// `S2 = sum((src - pred)^2)`. The centred energy `S2 - S1^2/n` is what
    /// remains after one DC coefficient corrects the mean offset (mode-beam
    /// sparse slot).
    #[inline]
    pub(crate) fn residual_moments(
        &self,
        src: SrcBlock<'_>,
        pred: &[i32],
        pred_stride: usize,
    ) -> (i64, i64, i64) {
        let (vw, vh) = (src.vis_w, src.vis_h);
        let n = (vw * vh) as i64;
        if n == 0 {
            return (0, 0, 0);
        }
        let (s1, s2) = if vw == src.w {
            self.residual_moments_raw(src.data, src.stride, pred, pred_stride, vw, vh)
        } else {
            residual_moments_scalar(src.data, src.stride, pred, pred_stride, vw, vh)
        };
        (s1, s2, n)
    }

    #[inline]
    fn residual_moments_raw(
        &self,
        src: &[u16],
        src_stride: usize,
        pred: &[i32],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> (i64, i64) {
        debug_assert!((h - 1) * src_stride + w <= src.len());
        debug_assert!((h - 1) * pred_stride + w <= pred.len());
        (self.residual_moments)(src, src_stride, pred, pred_stride, w, h)
    }

    /// Partition-pricing SATD of `src - clamp(pred|dc + residual)` over the
    /// VISIBLE part of `src` (edge blocks: invisible pixels are set to the
    /// reconstruction, as in [`Self::satd_sad_proxy`]).
    #[inline]
    pub(crate) fn luma_satd(
        &self,
        src: SrcBlock<'_>,
        bd: u8,
        pred: &[i32],
        dc: i32,
        residual: &[i32],
    ) -> u64 {
        let (w, h) = (src.w, src.h);
        if !src.is_clipped() {
            return self.luma_satd_raw(src.data, src.stride, 0, 0, w, h, bd, pred, dc, residual);
        }
        let maxv = (1i32 << bd) - 1;
        let mut masked = [0u16; MAX_BLOCK_PIXELS];
        src.masked_into(&mut masked, |x, y| {
            let i = y * w + x;
            let p = if pred.is_empty() { dc } else { pred[i] };
            let r = if residual.is_empty() { 0 } else { residual[i] };
            (p + r).clamp(0, maxv)
        });
        self.luma_satd_raw(&masked, w, 0, 0, w, h, bd, pred, dc, residual)
    }

    #[inline]
    fn luma_satd_raw(
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

    /// SSE of `clamp(pred|dc + residual)` against the VISIBLE part of `src`
    /// (`pred`/`residual` packed `src.w()`-wide; either may be empty).
    #[inline]
    pub(crate) fn chroma_sse(
        &self,
        src: SrcBlock<'_>,
        bd: u8,
        pred: &[i32],
        dc: i32,
        residual: &[i32],
    ) -> f32 {
        let (w, h) = (src.w, src.h);
        debug_assert!(pred.is_empty() || pred.len() >= w * h);
        debug_assert!(residual.is_empty() || residual.len() >= w * h);
        let (vw, vh) = (src.vis_w, src.vis_h);
        if vw == 0 || vh == 0 {
            return 0.0;
        }
        let maxv = (1i32 << bd) - 1;
        let sse = if vw == w {
            self.chroma_sse_raw(src.data, src.stride, 0, 0, w, vh, maxv, pred, dc, residual)
        } else {
            chroma_sse_strided_scalar(src.data, src.stride, w, vw, vh, maxv, pred, dc, residual)
        };
        sse as f32
    }

    #[inline]
    fn chroma_sse_raw(
        &self,
        src: &[u16],
        stride: usize,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        maxv: i32,
        pred: &[i32],
        dc: i32,
        residual: &[i32],
    ) -> i64 {
        debug_assert!(px + w <= stride);
        debug_assert!(h == 0 || (py + h - 1) * stride + px + w <= src.len());
        (self.chroma_sse)(src, stride, px, py, w, h, maxv, pred, dc, residual)
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
fn residual_moments_neon_wrap(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> (i64, i64) {
    unsafe { crate::neon::residual_moments_neon(src, src_stride, pred, pred_stride, w, h) }
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

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn chroma_sse_neon_wrap(
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
) -> i64 {
    unsafe {
        crate::neon::chroma_sse_neon(src, stride, px, py, w, h, max_value, pred, dc, residual)
    }
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

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn chroma_sse_avx2_wrap(
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
) -> i64 {
    unsafe { crate::avx::chroma_sse_avx2(src, stride, px, py, w, h, max_value, pred, dc, residual) }
}

pub(crate) fn residual_moments_scalar(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> (i64, i64) {
    let (mut s1, mut s2) = (0i64, 0i64);
    for y in 0..h {
        let sr = &src[y * src_stride..y * src_stride + w];
        let pr = &pred[y * pred_stride..y * pred_stride + w];
        for (&s, &p) in sr.iter().zip(pr) {
            let d = i64::from(s) - i64::from(p);
            s1 += d;
            s2 += d * d;
        }
    }
    (s1, s2)
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

/// [`sse_recon_scalar`] over a `w`x`h` sub-rectangle of `pstride`-pitched
/// `pred`/`resid` (column-clipped edge blocks).
fn sse_recon_strided_scalar(
    pred: &[i32],
    resid: &[i32],
    pstride: usize,
    src: &[u16],
    stride: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    let mut sse = 0i64;
    for y in 0..h {
        let srow = &src[y * stride..][..w];
        let prow = &pred[y * pstride..][..w];
        let rrow = &resid[y * pstride..][..w];
        for ((&s, &p), &e) in srow.iter().zip(prow).zip(rrow) {
            let d = (s as i32 - (p + e).clamp(0, maxv)) as i64;
            sse += d * d;
        }
    }
    sse
}

/// [`chroma_sse_scalar`] over a `w`x`h` sub-rectangle of `pstride`-pitched
/// `pred`/`residual` (column-clipped edge blocks).
fn chroma_sse_strided_scalar(
    src: &[u16],
    stride: usize,
    pstride: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> i64 {
    let mut sse = 0i64;
    for y in 0..h {
        let src_row = &src[y * stride..][..w];
        for (x, &source) in src_row.iter().enumerate() {
            let i = y * pstride + x;
            let prediction = if pred.is_empty() { dc } else { pred[i] };
            let reconstruction = prediction + if residual.is_empty() { 0 } else { residual[i] };
            let delta = (i32::from(source) - reconstruction.clamp(0, max_value)) as i64;
            sse += delta * delta;
        }
    }
    sse
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_sse_scalar(
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
) -> i64 {
    let mut sse = 0i64;
    for y in 0..h {
        let src_row = &src[(py + y) * stride + px..][..w];
        let pred_row = (!pred.is_empty()).then(|| &pred[y * w..][..w]);
        let residual_row = (!residual.is_empty()).then(|| &residual[y * w..][..w]);
        for (x, &source) in src_row.iter().enumerate() {
            let prediction = pred_row.map_or(dc, |row| row[x]);
            let reconstruction = prediction + residual_row.map_or(0, |row| row[x]);
            let delta = (i32::from(source) - reconstruction.clamp(0, max_value)) as i64;
            sse += delta * delta;
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
    #[test]
    fn residual_moments_dispatch_matches_scalar() {
        // Sizes 4..32 (incl. the 4-wide tail), strided planes, signed deltas.
        let stride = 37;
        let src: Vec<u16> = (0..stride * 40)
            .map(|i| ((i * 97 + 13) % 1021) as u16)
            .collect();
        let pred: Vec<i32> = (0..stride * 40)
            .map(|i| ((i * 53) % 1100) as i32 - 40)
            .collect();
        let rd = RdDispatch::selected();
        for &(w, h) in &[
            (4usize, 4usize),
            (8, 8),
            (16, 8),
            (8, 16),
            (16, 16),
            (32, 32),
            (4, 16),
        ] {
            let a = rd.residual_moments_raw(
                &src[stride * 2 + 3..],
                stride,
                &pred[stride * 2 + 3..],
                stride,
                w,
                h,
            );
            let b = residual_moments_scalar(
                &src[stride * 2 + 3..],
                stride,
                &pred[stride * 2 + 3..],
                stride,
                w,
                h,
            );
            assert_eq!(a, b, "{w}x{h}");
        }
    }

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
                    dispatch.sse_recon_raw(&pred, &resid, &src, stride, px, py, w, h, max),
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
                dispatch.sse_u16_raw(&src, stride, px, py, &reference, stride, px, py, w, h,),
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
                        dispatch.luma_satd_raw(&src, stride, px, py, w, h, bd, pred, dc, residual);
                    assert_eq!(got, want, "{variant} {w}x{h} bd={bd}");
                }
            }
        }
    }

    #[test]
    fn chroma_partition_sse_simd_matches_scalar() {
        let dispatch = RdDispatch::selected();
        let mut state = 0xe703_7ed1_a0b4_28dbu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for &(w, h) in &[
            (1usize, 1usize),
            (3, 5),
            (4, 8),
            (7, 4),
            (8, 8),
            (13, 9),
            (16, 32),
            (31, 17),
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
                    let want = chroma_sse_scalar(
                        &src, stride, px, py, w, h, max_value, pred, dc, residual,
                    );
                    let got = dispatch
                        .chroma_sse_raw(&src, stride, px, py, w, h, max_value, pred, dc, residual);
                    assert_eq!(got, want, "{variant} {w}x{h} bd={bd}");
                }
            }
        }
    }

    #[test]
    fn chroma_sse_dispatch_selects_active_arch_kernel() {
        let dispatch = RdDispatch::selected();
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        assert!(std::ptr::fn_addr_eq(
            dispatch.chroma_sse,
            chroma_sse_neon_wrap as ChromaSseFn,
        ));
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            let expected = if std::is_x86_feature_detected!("avx2") {
                chroma_sse_avx2_wrap as ChromaSseFn
            } else {
                chroma_sse_scalar as ChromaSseFn
            };
            assert!(std::ptr::fn_addr_eq(dispatch.chroma_sse, expected));
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx"),
        )))]
        assert!(std::ptr::fn_addr_eq(
            dispatch.chroma_sse,
            chroma_sse_scalar as ChromaSseFn,
        ));
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
                let got = dispatch.satd_sad_proxy_raw(&src, src_stride, &pred, pred_stride, w, h);
                assert_eq!(got, want, "{w}x{h} strides {src_stride}/{pred_stride}");
            }
        }
    }

    /// Plane of `pw`x`ph` whose displayed region is `vw`x`vh`; samples outside
    /// it are `fill`, so two planes differing only in `fill` differ only in
    /// pixels a decoder crops away.
    fn edge_plane(pw: usize, ph: usize, vw: usize, vh: usize, fill: u16, seed: u32) -> Vec<u16> {
        let mut state = seed;
        (0..pw * ph)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let (x, y) = (i % pw, i / pw);
                if x < vw && y < vh {
                    ((state >> 8) % 1024) as u16
                } else {
                    fill
                }
            })
            .collect()
    }

    #[test]
    fn src_block_clips_to_displayed_region() {
        let plane = vec![0u16; 40 * 24];
        let b = SrcBlock::new(&plane, 40, 32, 16, 8, 8, 37, 21);
        assert_eq!((b.w(), b.h()), (8, 8));
        assert_eq!(b.vis(), (5, 5));
        assert!(b.is_clipped());
        let interior = SrcBlock::new(&plane, 40, 8, 8, 8, 8, 37, 21);
        assert_eq!(interior.vis(), (8, 8));
        assert!(!interior.is_clipped());
        let rows_only = SrcBlock::new(&plane, 40, 0, 16, 16, 8, 40, 21);
        assert_eq!(rows_only.vis(), (16, 5));
    }

    #[test]
    fn clipped_distortion_ignores_invisible_pixels() {
        let dispatch = RdDispatch::selected();
        let intra = crate::intrapred::IntraPredDispatch::selected();
        let (pw, ph) = (72usize, 40usize);
        // (block x, y, w, h, displayed vw, vh): interior, row-clipped,
        // column-clipped, both.
        for &(x, y, w, h, vw, vh) in &[
            (8usize, 8usize, 16usize, 16usize, 72usize, 40usize),
            (0, 32, 16, 8, 72, 35),
            (64, 0, 8, 16, 67, 40),
            (56, 24, 16, 16, 61, 37),
            (64, 32, 8, 8, 65, 33),
        ] {
            let a = edge_plane(pw, ph, vw, vh, 0, 7);
            let b = edge_plane(pw, ph, vw, vh, 1023, 7);
            let (ba, bb) = (
                SrcBlock::new(&a, pw, x, y, w, h, vw, vh),
                SrcBlock::new(&b, pw, x, y, w, h, vw, vh),
            );
            let n = w * h;
            let pred: Vec<i32> = (0..n as i32).map(|i| (i * 37) % 900 + 40).collect();
            let resid: Vec<i32> = (0..n as i32).map(|i| (i * 13) % 61 - 30).collect();
            let (cvw, cvh) = ba.vis();

            let want: i64 = (0..cvh)
                .flat_map(|ry| (0..cvw).map(move |rx| (rx, ry)))
                .map(|(rx, ry)| {
                    let i = ry * w + rx;
                    let r = (pred[i] + resid[i]).clamp(0, 1023);
                    let d = i64::from(a[(y + ry) * pw + x + rx]) - i64::from(r);
                    d * d
                })
                .sum();
            let tag = format!("{w}x{h} at ({x},{y}) vis {cvw}x{cvh}");
            assert_eq!(
                dispatch.sse_recon(&pred, &resid, ba, 10),
                want,
                "sse_recon {tag}"
            );
            assert_eq!(
                dispatch.sse_recon(&pred, &resid, bb, 10),
                want,
                "sse_recon {tag}"
            );
            assert_eq!(
                dispatch.chroma_sse(ba, 10, &pred, 0, &resid),
                want as f32,
                "{tag}"
            );
            assert_eq!(
                dispatch.chroma_sse(bb, 10, &pred, 0, &resid),
                want as f32,
                "{tag}"
            );
            assert_eq!(
                dispatch.satd_sad_proxy(ba, &pred, w),
                dispatch.satd_sad_proxy(bb, &pred, w),
                "satd_sad_proxy {tag}"
            );
            assert_eq!(
                dispatch.luma_satd(ba, 10, &pred, 0, &resid),
                dispatch.luma_satd(bb, 10, &pred, 0, &resid),
                "luma_satd {tag}"
            );
            assert_eq!(
                dispatch.residual_moments(ba, &pred, w),
                dispatch.residual_moments(bb, &pred, w),
                "residual_moments {tag}"
            );
            assert_eq!(
                dispatch.residual_moments(ba, &pred, w).2,
                (cvw * cvh) as i64
            );
            let recon = edge_plane(pw, ph, pw, ph, 0, 99);
            assert_eq!(
                dispatch.sse_u16(ba, &recon, pw, x, y),
                dispatch.sse_u16(bb, &recon, pw, x, y),
                "sse_u16 {tag}"
            );
            let ac: Vec<i32> = (0..n as i32).map(|i| (i * 29) % 401 - 200).collect();
            assert_eq!(
                intra.cfl_best_alpha(&ac, ba, 512, 10),
                intra.cfl_best_alpha(&ac, bb, 512, 10),
                "cfl_best_alpha {tag}"
            );
            if !ba.is_clipped() {
                // Interior: identical to the unclipped raw kernels.
                assert_eq!(
                    dispatch.satd_sad_proxy(ba, &pred, w),
                    dispatch.satd_sad_proxy_raw(&a[y * pw + x..], pw, &pred, w, w, h)
                );
            }
        }
    }
}
