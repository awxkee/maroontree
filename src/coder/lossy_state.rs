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

impl<'a> LossyTile<'a> {
    /// CDFs used by DECISION-side rate estimates (RDOQ trellis, filter-intra /
    /// angle-delta costs): the frozen [`Cdfs::decision_snapshot`], so decisions
    /// are independent of SB coding order (SB-wavefront prerequisite). The
    /// `MT_AV1_LIVE_DECIDE_CDF=1` escape hatch restores the historical
    /// adaptive-cost behavior for A/B comparisons only.
    #[inline]
    fn dcdf(&self) -> &Cdfs {
        if self.dec_live { &self.cdfs } else { &self.dec_cdfs }
    }

    fn new(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3], qm: QmLevels) -> Self {
        LossyTile {
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma_with_delta_qm(q, chroma_dc_delta(q), bd, qm.u),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; w * h], vec![0; w * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: vec![0; w / 4],
            l_uv_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            dec_cdfs: Cdfs::decision_snapshot(crate::coef_q::qcat(q)),
            dec_live: false,
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    /// Monochrome tile: codes the luma plane only (`NumPlanes = 1`). Only
    /// `src[0]` is used; the chroma reconstruction and context arrays are left
    /// empty so any stray chroma access panics instead of corrupting output.
    /// Forces 8x8 luma transforms (see `prefer_16x16`/`prefer_32x32`).
    fn new_mono(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3], qm: QmLevels) -> Self {
        LossyTile {
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma(q, bd),
            w,
            h,
            cw: w,
            ss422: false,
            ss420: false,
            mono: true,
            src,
            recon: [vec![0; w * h], Vec::new(), Vec::new()],
            a_coef: [vec![0x40; w / 4], Vec::new(), Vec::new()],
            l_coef: [vec![0x40; h / 4], Vec::new(), Vec::new()],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: Vec::new(),
            l_uv_mode: Vec::new(),
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            dec_cdfs: Cdfs::decision_snapshot(crate::coef_q::qcat(q)),
            dec_live: false,
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    /// 4:2:2 tile: luma is full w x h, chroma planes are subsampled to (w/2) x h.
    /// `src[1]`/`src[2]` must already be the half-width chroma planes.
    fn new_422(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3], qm: QmLevels) -> Self {
        let cw = w / 2;
        LossyTile {
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma_with_delta_qm(q, chroma_dc_delta(q), bd, qm.u),
            w,
            h,
            cw,
            ss422: true,
            ss420: false,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; cw * h], vec![0; cw * h]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: vec![0; w / 4],
            l_uv_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            dec_cdfs: Cdfs::decision_snapshot(crate::coef_q::qcat(q)),
            dec_live: false,
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    /// 4:2:0 tile: luma is full w x h, chroma planes are subsampled to
    /// (w/2) x (h/2). `src[1]`/`src[2]` must already be the quarter-size planes.
    fn new_420(q: u8, bd: u8, w: usize, h: usize, src: &'a [Vec<i32>; 3], qm: QmLevels) -> Self {
        let (cw, ch) = (w / 2, h / 2);
        LossyTile {
            bd,
            quant: Quant::new_with_qm(q, bd, qm.y),
            cquant: Quant::new_chroma_with_delta_qm(q, chroma_dc_delta(q), bd, qm.u),
            w,
            h,
            cw,
            ss422: false,
            ss420: true,
            mono: false,
            src,
            recon: [vec![0; w * h], vec![0; cw * ch], vec![0; cw * ch]],
            a_coef: [vec![0x40; w / 4], vec![0x40; cw / 4], vec![0x40; cw / 4]],
            l_coef: [vec![0x40; h / 4], vec![0x40; ch / 4], vec![0x40; ch / 4]],
            a_part: vec![0; w / 8],
            l_part: vec![0; h / 8],
            a_skip: vec![0; w / 4],
            l_skip: vec![0; h / 4],
            a_mode: vec![0; w / 4],
            l_mode: vec![0; h / 4],
            a_uv_mode: vec![0; w / 4],
            l_uv_mode: vec![0; h / 4],
            blk4: vec![0; (w / 4) * (h / 4)],
            blk4h: vec![0; (w / 4) * (h / 4)],
            blk4v: vec![false; (w / 4) * (h / 4)],
            blk4t: vec![false; (w / 4) * (h / 4)],
            skip8: vec![true; w.div_ceil(8) * h.div_ceil(8)],
            cdef_point_marked: false,
            enc: OdEcEncoder::new(),
            cdfs: Cdfs::new(crate::coef_q::qcat(q)),
            dec_cdfs: Cdfs::decision_snapshot(crate::coef_q::qcat(q)),
            dec_live: false,
            sb_mode: SbMode::Off,
            rec: DecisionRecord::default(),
            cur: RecordCursor::default(),
            speed: Speed::Slow,
            aq: AqCtx::off(),
            wiener: None,
            lr_ref_h: crate::wiener::WIENER_TAPS_MID,
            lr_ref_v: crate::wiener::WIENER_TAPS_MID,
            frame_x0: 0,
            frame_y0: 0,
            frame_w: w,
            frame_h: h,
            base_q_idx: q,
        }
    }

    fn skip_ctx(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0 // luma: TX size == block size -> ctx 0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = (a[bx4] != 0x40 || a[bx4 + 1] != 0x40) as usize;
            let cl = (l[by4] != 0x40 || l[by4 + 1] != 0x40) as usize;
            7 + ca + cl
        }
    }

    fn dc_sign_ctx(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma = (a[bx4] >> 6) as i32 + (a[bx4 + 1] >> 6) as i32;
        let suml = (l[by4] >> 6) as i32 + (l[by4 + 1] >> 6) as i32;
        let s = suma + suml - 4;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a 16x16 transform. Luma: tx == block size -> 0.
    /// Chroma (4:4:4, chroma tx == chroma block): `7 + above_nz + left_nz` over
    /// the 4-unit (16-sample) footprint (`get_txb_skip_ctx`, ctx_offset = 7).
    fn skip_ctx_16(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
            let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
            7 + ca + cl
        }
    }

    /// dc_sign context for a 16x16 transform (4-unit footprint, baseline -8).
    fn dc_sign_ctx_16(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 8;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a luma RTX_16X8 block coded as a single transform.
    fn skip_ctx_16x8_luma(&self) -> usize {
        0
    }

    fn skip_ctx_8x16_luma(&self) -> usize {
        0
    }

    fn dc_sign_ctx_8x16_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let suma: i32 = a[bx4..bx4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// dc_sign context for a luma RTX_16X8 transform: 4 above-units (16 wide) and
    /// 2 left-units (8 tall) top-bit sums, baseline -(4+2) = -6.
    fn dc_sign_ctx_16x8_luma(&self, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[0];
        let l = &self.l_coef[0];
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:4:4 chroma txb_skip context for an RTX_16X8 block (block == transform, so
    /// `not_one_blk` is 0): `7 + ca + cl` over 4 above-units and 2 left-units.
    fn skip_ctx_16x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:4:4 chroma dc_sign context for RTX_16X8: 4 above + 2 left, baseline -6.
    fn dc_sign_ctx_16x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_8x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 2].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_8x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma: i32 = a[bx4..bx4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx_32x16_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let sa: i32 = a[bx4..bx4 + 8].iter().map(|&x| (x >> 6) as i32).sum();
        let sl: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = sa + sl - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx_16x32_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let sa: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let sl: i32 = l[by4..by4 + 8].iter().map(|&x| (x >> 6) as i32).sum();
        let s = sa + sl - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_32x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 8].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_32x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let sa: i32 = a[bx4..bx4 + 8].iter().map(|&x| (x >> 6) as i32).sum();
        let sl: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = sa + sl - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_16x32_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 8].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_16x32_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let sa: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let sl: i32 = l[by4..by4 + 8].iter().map(|&x| (x >> 6) as i32).sum();
        let s = sa + sl - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_16x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 4].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_16x16_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let sa: i32 = a[bx4..bx4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let sl: i32 = l[by4..by4 + 4].iter().map(|&x| (x >> 6) as i32).sum();
        let s = sa + sl - 8;
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx_8x4_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let sa: i32 = a[bx4..bx4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = sa + (l[by4] >> 6) as i32 - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    fn dc_sign_ctx_4x8_luma(&self, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[0], &self.l_coef[0]);
        let sl: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = (a[bx4] >> 6) as i32 + sl - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_4x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = (a[bx4] != 0x40) as usize;
        let cl = (l[by4] != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_4x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let s = (a[bx4] >> 6) as i32 + (l[by4] >> 6) as i32 - 2;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_8x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 2].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_8x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma: i32 = a[bx4..bx4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 4;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_8x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = a[bx4..bx4 + 2].iter().any(|&x| x != 0x40) as usize;
        let cl = (l[by4] != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_8x4_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma: i32 = a[bx4..bx4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let suml = (l[by4] >> 6) as i32;
        let s = suma + suml - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    fn skip_ctx_4x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let ca = (a[bx4] != 0x40) as usize;
        let cl = l[by4..by4 + 2].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    fn dc_sign_ctx_4x8_chroma(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let (a, l) = (&self.a_coef[plane], &self.l_coef[plane]);
        let suma = (a[bx4] >> 6) as i32;
        let suml: i32 = l[by4..by4 + 2].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    /// Decide whether to code the 16x16 region at (`x8`,`y8`) as a single
    /// TX_16X16 (PARTITION_NONE) vs splitting into four 8x8. This is a pure R-D
    /// proxy — the decoder follows whatever partition we signal, so the choice
    /// affects compression only, never correctness. Proxy: compare the summed
    /// absolute quantized luma levels of the one 16x16 transform (plus a small
    /// per-block overhead) against the four 8x8 transforms (each with its own
    /// overhead). Smooth regions compact into the 16x16 and win decisively.
    /// Peak-to-peak luma range of the `dim`x`dim` source block at 8-unit origin
    /// `(x8,y8)`. Smooth low-contrast blocks (small range) ring into visible
    /// low-frequency banding under large transforms, so the partitioner uses it
    /// to keep such blocks on small (8x8) transforms.
    fn block_luma_range(&self, x8: usize, y8: usize, dim: usize) -> i32 {
        let (px, py) = (x8 * 8, y8 * 8);
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for ry in 0..dim {
            let base = (py + ry) * self.w + px;
            for &s in &self.src[0][base..base + dim] {
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }
        }
        hi - lo
    }

    /// R-D proxy for coding an 8x8 luma region as one TX_8X8 (PARTITION_NONE)
    /// vs splitting into four BLOCK_4X4. Runs the real non-directional mode
    /// search (SSE + lambda*bits) for both options so the decision reflects
    /// 4x4's per-quadrant mode diversity, not just a DC estimate. Returns
    /// `true` to keep the 8x8 whole. Split is offered only for 4:2:0/4:4:4.
    /// Mode-search lambda: libaom KF rdmult q-shape (`cost::mode_lambda_q`) times
    /// the SSIMULACRA2 rdmult weight, AQ-correct via the active `self.quant`.
    #[inline]
    fn mlam(&self) -> f32 {
        mode_lambda_q(self.quant.dc_q() as f32) * self.tune_weight()
    }

    /// As [`Self::mlam`] but for chroma planes (uses `self.cquant`).
    #[inline]
    fn mlam_c(&self) -> f32 {
        mode_lambda_q(self.cquant.dc_q() as f32) * self.tune_weight()
    }

    /// libaom SSIMULACRA2 rdmult weight for this frame (1.0 when tune is off).
    #[inline]
    fn tune_weight(&self) -> f32 {
        let tune = TUNE_SSIMULACRA2.load(std::sync::atomic::Ordering::Relaxed);
        mode_lambda_weight(self.base_q_idx, tune)
    }

    fn prefer_8x8_none(&self, x8: usize, y8: usize) -> bool {
        if self.mono || self.ss422 {
            return true;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // best non-directional cost for one 8x8 (DCT_DCT only; cheap proxy)
        let mut eff8 = f32::INFINITY;
        let directional_top = self.rank_luma_directionals::<64>(modes, px, py, 8, 8, false, false);
        for &m in modes {
            if is_directional_mode(m) && !directional_top.contains(m) {
                continue;
            }
            let mut pred = [0i32; 64];
            if m == DC_PRED {
                let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    false,
                    false,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 8, 8);
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_8X8, lam);
            let rr = idct_dequant_8x8(&cf, &self.quant);
            let distortion =
                self.luma_partition_distortion(px, py, 8, 8, self.quant.ac_q() as f32, |i| {
                    pred[i] + rr[i]
                });
            let eff =
                crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, &SCAN_8X8));
            if eff < eff8 {
                eff8 = eff;
            }
        }
        // best cost for four 4x4 (DC-pred / nd; current recon, decision-only)
        let mut eff4_sum = rate_cost(mlam, 2.0f32); // PARTITION_SPLIT symbol allowance
        for (sx, sy) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)] {
            let (bx, by) = (px + sx, py + sy);
            let mut best = f32::INFINITY;
            let directional_top =
                self.rank_luma_directionals::<16>(modes, bx, by, 4, 4, false, false);
            for &m in modes {
                if is_directional_mode(m) && !directional_top.contains(m) {
                    continue;
                }
                let mut pred = [0i32; 16];
                if m == DC_PRED {
                    let d = dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    intra_predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        false,
                        false,
                        self.w,
                        self.h,
                        self.luma_filter_type(bx, by),
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 16];
                crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 4, 4);
                let (mut cf, tf) = forward_dct_quant_4x4_t(&resid, &self.quant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_4X4, lam);
                let rr = idct_dequant_4x4(&cf, &self.quant);
                let distortion =
                    self.luma_partition_distortion(bx, by, 4, 4, self.quant.ac_q() as f32, |i| {
                        pred[i] + rr[i]
                    });
                // +mode/skip signaling allowance per 4x4 sub-block
                let eff = crate::partition_rd::rd_cost(
                    distortion,
                    mlam,
                    block_rate_bits(&cf, &SCAN_4X4) + 4.0f32,
                );
                if eff < best {
                    best = eff;
                }
            }
            eff4_sum += best;
        }
        // In 4:4:4, splitting an 8x8 luma block also changes chroma from one
        // 8x8 block into four independently predicted 4x4 blocks. In 4:2:0 the
        // four sub-8x8 luma blocks share one 4x4 chroma reference block, so the
        // chroma geometry is unchanged and cancels from this comparison.
        if !self.ss420 {
            eff8 += CHROMA_PART_RD_WEIGHT * self.rd_cost_chroma_block(px, py, 8, 8, 1.0);
            for (sx, sy) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)] {
                eff4_sum +=
                    CHROMA_PART_RD_WEIGHT * self.rd_cost_chroma_block(px + sx, py + sy, 4, 4, 1.0);
            }
        }
        eff8 <= eff4_sum
    }

    /// Mean and variance of a `w`x`h` luma source region at pixel origin
    /// (px, py). libaom's partition search uses exactly these per-candidate
    /// variance features (`block_var`, `horz_block_var[2]`, `sub_block_var[4]`)
    /// to steer and prune the decision before paying for full R-D.
    fn luma_variance(&self, px: usize, py: usize, w: usize, h: usize) -> f32 {
        let mut sum = 0i64;
        let mut sqsum = 0i64;
        for ry in 0..h {
            let row = &self.src[0][(py + ry) * self.w + px..];
            for &s in &row[..w] {
                sum += s as i64;
                sqsum += (s as i64) * (s as i64);
            }
        }
        let n = (w * h) as f32;
        let mean = sum as f32 / n;
        (sqsum as f32 / n) - mean * mean
    }

    fn luma_partition_distortion(
        &self,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        qstep: f32,
        recon_at: impl Fn(usize) -> i32,
    ) -> f32 {
        crate::partition_rd::luma_satd(&self.src[0], self.w, px, py, w, h, self.bd, qstep, recon_at)
    }

    #[allow(clippy::too_many_arguments)]
    fn rd_cost_chroma_fixed<const N: usize>(
        &self,
        cx: usize,
        cy: usize,
        cw: usize,
        ch: usize,
        prdo: f32,
        scan: &[u32],
        pred: impl Fn(&[i32], usize, usize, usize, i32) -> i32,
        fwd: impl Fn(&[i32; N], &Quant) -> ([i32; N], [f32; N]),
        inv: impl Fn(&[i32; N], &Quant) -> [i32; N],
    ) -> f32 {
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        let mut total = 0.0f32;
        for plane in 1..=2 {
            let dc = pred(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            let mut resid = [0i32; N];
            for ry in 0..ch {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for x in 0..cw {
                    resid[ry * cw + x] = srow[x] - dc;
                }
            }
            let (mut cf, tf) = fwd(&resid, &self.cquant);
            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
            let rr = inv(&cf, &self.cquant);
            let distortion = crate::partition_rd::chroma_sse(
                &self.src[plane],
                self.cw,
                cx,
                cy,
                cw,
                ch,
                self.bd,
                |i| dc + rr[i],
            );
            total += crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, scan));
        }
        total
    }

    fn rd_cost_chroma_block(&self, cx: usize, cy: usize, cw: usize, ch: usize, prdo: f32) -> f32 {
        match (cw, ch) {
            (4, 4) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_4X4,
                dc_pred_4x4,
                forward_dct_quant_4x4_t,
                idct_dequant_4x4,
            ),
            (8, 4) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_8X4,
                dc_pred_8x4,
                dct8x4_t,
                idct_dequant_8x4,
            ),
            (4, 8) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_4X8,
                dc_pred_4x8,
                dct4x8_t,
                idct_dequant_4x8,
            ),
            (8, 8) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_8X8,
                dc_pred_8x8,
                forward_dct_quant_8x8_t,
                idct_dequant_8x8,
            ),
            (16, 8) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X8,
                dc_pred_16x8,
                dct16x8_t,
                idct_dequant_16x8,
            ),
            (8, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_8X16,
                dc_pred_8x16,
                forward_dct_quant_8x16_t,
                idct_dequant_8x16,
            ),
            (16, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X16,
                dc_pred_16x16,
                forward_dct_quant_16x16_t,
                idct_dequant_16x16,
            ),
            (32, 16) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_32X16,
                dc_pred_32x16,
                dct32x16_t,
                idct_dequant_32x16,
            ),
            (16, 32) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_16X32,
                dc_pred_16x32,
                forward_dct_quant_16x32_t,
                idct_dequant_16x32,
            ),
            (32, 32) => self.rd_cost_chroma_fixed(
                cx,
                cy,
                cw,
                ch,
                prdo,
                &SCAN_32X32,
                dc_pred_32x32,
                forward_dct_quant_32x32_t,
                idct_dequant_32x32,
            ),
            _ => unreachable!("unsupported derived chroma block {cw}x{ch}"),
        }
    }

    /// Sum chroma R-D for the blocks derived from a luma partition candidate.
    fn rd_cost_chroma_partition(
        &self,
        px: usize,
        py: usize,
        luma_dim: usize,
        part: Part16,
        prdo: f32,
    ) -> f32 {
        let sub_x = (self.ss420 || self.ss422) as usize;
        let sub_y = self.ss420 as usize;
        let blocks: &[(usize, usize, usize, usize)] = match part {
            Part16::None => &[(0, 0, luma_dim, luma_dim)],
            Part16::Split => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, half),
                    (half, 0, half, half),
                    (0, half, half, half),
                    (half, half, half, half),
                ]
            }
            Part16::Horz => {
                let half = luma_dim / 2;
                &[(0, 0, luma_dim, half), (0, half, luma_dim, half)]
            }
            Part16::Vert => {
                let half = luma_dim / 2;
                &[(0, 0, half, luma_dim), (half, 0, half, luma_dim)]
            }
            Part16::HorzA => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, half),
                    (half, 0, half, half),
                    (0, half, luma_dim, half),
                ]
            }
            Part16::HorzB => {
                let half = luma_dim / 2;
                &[
                    (0, 0, luma_dim, half),
                    (0, half, half, half),
                    (half, half, half, half),
                ]
            }
            Part16::VertA => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, half),
                    (0, half, half, half),
                    (half, 0, half, luma_dim),
                ]
            }
            Part16::VertB => {
                let half = luma_dim / 2;
                &[
                    (0, 0, half, luma_dim),
                    (half, 0, half, half),
                    (half, half, half, half),
                ]
            }
        };
        CHROMA_PART_RD_WEIGHT
            * blocks
                .iter()
                .map(|&(ox, oy, lw, lh)| {
                    self.rd_cost_chroma_block(
                        (px + ox) >> sub_x,
                        (py + oy) >> sub_y,
                        lw >> sub_x,
                        lh >> sub_y,
                        prdo,
                    )
                })
                .sum::<f32>()
    }

    /// Full square+rectangular partition decision for a 16x16 luma region.
    /// Evaluate legal NONE/SPLIT/HORZ/VERT and 4:2:0 A/B candidates with the
    /// same SATD + mlam*bits objective; the previous qstep/variance pre-pruning
    /// lost too many useful rectangles at medium and coarse quantization.
    fn partition_choice_16(&self, x8: usize, y8: usize) -> Part16 {
        if self.mono {
            return Part16::Split; // monochrome codes 8x8 luma blocks only
        }
        let (px, py) = (x8 * 8, y8 * 8);

        let horz_on = HORZ_ENABLED.load(std::sync::atomic::Ordering::Relaxed);

        // Anchor the perceptual R-D scale ONCE at the parent 16x16 region and use
        // it for every candidate so all costs share one lambda axis.
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let part_lam = self.mlam() * prdo;
        let rd_none = self.rd_cost_square(px, py, 16, false, false, prdo)
            + self.rd_cost_chroma_partition(px, py, 16, Part16::None, prdo);
        let mut rd_split = rate_cost(part_lam, SPLIT_SIGNAL_BITS);
        for (sx, sy) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)] {
            rd_split += self.rd_cost_square(px + sx, py + sy, 8, false, false, prdo);
        }
        rd_split += self.rd_cost_chroma_partition(px, py, 16, Part16::Split, prdo);
        let rd_horz = if !horz_on {
            f32::INFINITY
        } else {
            self.rd_cost_horz(px, py, prdo)
                + self.rd_cost_chroma_partition(px, py, 16, Part16::Horz, prdo)
        };

        // VERT mirrors HORZ, but remains forbidden in 4:2:2.
        let vert_on = !self.ss422 && VERT_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let rd_vert = if !vert_on {
            f32::INFINITY
        } else {
            self.rd_cost_vert(px, py, prdo)
                + self.rd_cost_chroma_partition(px, py, 16, Part16::Vert, prdo)
        };
        let asym_signal = rate_cost(part_lam, ASYM_PART_SIGNAL_BITS);
        // The A/B leaf emitters are currently implemented for 4:2:0.
        let (rd_horz_a, rd_horz_b, rd_vert_a, rd_vert_b) = if self.ss420 {
            (
                asym_signal
                    + rate_cost(part_lam, ASYM_DEPENDENT_RDO_BITS)
                    + self.rd_cost_square(px, py, 8, false, false, prdo)
                    + self.rd_cost_square(px + 8, py, 8, false, false, prdo)
                    + self.rd_cost_rect16_dependent(px, py + 8, false, true, false, prdo)
                    + self.rd_cost_chroma_partition(px, py, 16, Part16::HorzA, prdo),
                asym_signal
                    + self.rd_cost_rect16_leaf(px, py, false, prdo)
                    + self.rd_cost_square(px, py + 8, 8, false, false, prdo)
                    + self.rd_cost_square(px + 8, py + 8, 8, false, false, prdo)
                    + self.rd_cost_chroma_partition(px, py, 16, Part16::HorzB, prdo),
                asym_signal
                    + rate_cost(part_lam, ASYM_DEPENDENT_RDO_BITS)
                    + self.rd_cost_square(px, py, 8, false, false, prdo)
                    + self.rd_cost_square(px, py + 8, 8, false, false, prdo)
                    + self.rd_cost_rect16_dependent(px + 8, py, true, false, true, prdo)
                    + self.rd_cost_chroma_partition(px, py, 16, Part16::VertA, prdo),
                asym_signal
                    + self.rd_cost_rect16_leaf(px, py, true, prdo)
                    + self.rd_cost_square(px + 8, py, 8, false, false, prdo)
                    + self.rd_cost_square(px + 8, py + 8, 8, false, false, prdo)
                    + self.rd_cost_chroma_partition(px, py, 16, Part16::VertB, prdo),
            )
        } else {
            (f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY)
        };

        let cands = [
            (rd_none, Part16::None),
            (rd_split, Part16::Split),
            (rd_horz, Part16::Horz),
            (rd_vert, Part16::Vert),
            (rd_horz_a, Part16::HorzA),
            (rd_horz_b, Part16::HorzB),
            (rd_vert_a, Part16::VertA),
            (rd_vert_b, Part16::VertB),
        ];
        cands
            .into_iter()
            .fold((f32::INFINITY, Part16::Split), |b, c| {
                if c.0 < b.0 { c } else { b }
            })
            .1
    }

    /// Code a 16x16 region (4:4:4 only) as a single TX_16X16 block: luma +
    /// chroma DC prediction, forward DCT16 + quant, the TX_16X16 coefficient
    /// coder, and reconstruction via the exact integer inverse. Updates the
    /// 4-unit (16-sample) skip / coef neighbor-context footprint.
    /// Set the RDO effort level (0 = full, >= 1 = fast). Returns `self` for
    /// builder-style chaining at tile construction.
    fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    /// Turn on variance-based adaptive quantization for this tile. `ref_act` is
    /// the tile mean activity (see [`tile_ref_activity`]); the running
    /// `CurrentQIndex` starts at the frame base. Must be paired with a frame
    /// header that signals `delta_q_present = 1` and the matching `delta_q_res`.
    fn enable_aq(&mut self, base_q: u8, ref_act: f32, vb: &VarianceBoost) {
        self.aq = AqCtx {
            enabled: true,
            base_q,
            res_log2: AQ_RES_LOG2,
            cur_qidx: base_q as i32,
            ref_act,
            read_deltas: false,
            pending: 0,
            vb_enabled: vb.enabled,
            vb_octile: vb.octile.clamp(1, 8),
            vb_strength: vb.strength.max(0.0),
            vb_boost_only: vb.boost_only,
            dark: vb.dark,
        };
    }

    /// Begin a superblock at luma pixel `(sb_x, sb_y)`: pick the SB's quantizer
    /// from its local activity, retarget `self.quant`/`self.cquant`, and arm the
    /// `read_delta_qindex` token that the SB's first coded block will emit. No-op
    /// when AQ is disabled, so the non-AQ path is byte-identical.
    /// Emit `read_lr` for one 64x64 luma restoration unit at the start of a
    /// superblock (spec 5.11.57/58). With `LoopRestorationSize = 64` there is
    /// exactly one unit per SB whose top-left is the SB origin. When `self.wiener`
    /// is `None` nothing is emitted (RESTORE_NONE planes have no LR syntax).
    /// Emit `read_lr` for the restoration units that start in the superblock at
    /// **tile-local** pixel `(sb_x, sb_y)` (spec 5.11.57). Loop-restoration units
    /// are frame-relative, so the superblock position and unit counts are
    /// computed in frame coordinates (`frame_x0/y0`, `frame_w/h`). With
    /// `LoopRestorationSize = 64` most superblocks contain exactly one luma unit;
    /// small partial edge superblocks merge into the previous unit (the unit
    /// count uses `(frame+unit/2)/unit`) and emit zero units. Because units are
    /// >= the 64x64 superblock size and aligned to the frame grid, each unit's
    /// > top-left superblock lies in exactly one tile, so every unit is signaled
    /// > exactly once across all tiles — no tile-boundary special-casing needed.
    /// > Emits nothing when `self.wiener` is `None`.
    fn emit_lr_sb(&mut self, sb_x: usize, sb_y: usize) {
        let Some(unit) = self.wiener else {
            return;
        };
        emit_lr_sb_syms(
            &mut self.enc,
            &mut self.cdfs.wiener_restore,
            &mut self.lr_ref_v,
            &mut self.lr_ref_h,
            &unit,
            self.frame_x0,
            self.frame_y0,
            self.frame_w,
            self.frame_h,
            sb_x,
            sb_y,
        );
    }

    /// The AQ target qindex for the superblock at `(sb_x, sb_y)` — a pure
    /// function of the SOURCE pixels and frame-level AQ constants (no running
    /// state), which is what makes the whole per-SB qindex sequence
    /// precomputable by [`precompute_aq_grid`] before any coding starts.
    fn aq_sb_target(&self, sb_x: usize, sb_y: usize) -> i32 {
        let base_q = self.aq.base_q as i32;
        // Dark-structured-detail protection: an extra qindex reduction for dark SBs
        // carrying real cross-scale structure. Independent of the Variance Boost, it
        // is combined with the boost by `max` (see below), matching `av2/aq.rs`. The
        // AV1 luma plane is native-depth `i32`, normalized to 8-bit range so the
        // calibrated dark thresholds apply at every bit depth.
        let dark_scale = 1.0 / (1u32 << (self.bd - 8)) as f32;
        let dark = crate::aq_common::dark_protection(
            &self.aq.dark,
            base_q,
            &self.src[0],
            self.w,
            sb_y,
            sb_x,
            self.w,
            self.h,
            dark_scale,
        );
        if self.aq.vb_enabled {
            // Variance Boost: pick the representative 8x8 variance at the configured
            // octile and map it to a qindex delta off the frame base. ref_act (the
            // tile mean whole-SB log-variance) anchors the coarse side, matching
            // `av2/aq.rs::AqState::per_sb`.
            let mut subvars = [0f32; 64];
            let filled = aq_sb_subblock_variances(
                &self.src[0],
                self.w,
                sb_y,
                sb_x,
                self.w,
                self.h,
                &mut subvars,
            );
            if filled == 0 {
                base_q
            } else {
                let picked = crate::aq_common::sb_octile_variance(&mut subvars, self.aq.vb_octile);
                let vb_delta = crate::aq_common::variance_boost_delta(
                    picked,
                    self.aq.ref_act,
                    self.aq.vb_strength,
                    self.aq.vb_boost_only,
                );
                // protection = max(flat-region boost, dark boost); keep the coarsen
                // (positive) side of the variance boost only when neither fires.
                let flat_boost = (-vb_delta).max(0);
                let protection = flat_boost.max(dark);
                let delta = if protection > 0 {
                    -protection
                } else {
                    vb_delta
                };
                (base_q + delta).clamp(1, 255)
            }
        } else {
            let act = sb_activity(&self.src[0], self.w, sb_y, sb_x, self.w, self.h);
            let vb_delta = aq_target_qidx(base_q, act, self.aq.ref_act) - base_q;
            // Same combine as the VB path: dark protection can only deepen the
            // reduction, never override a coarsen with a smaller refine.
            let flat_boost = (-vb_delta).max(0);
            let protection = flat_boost.max(dark);
            let delta = if protection > 0 {
                -protection
            } else {
                vb_delta
            };
            (base_q + delta).clamp(1, 255)
        }
    }

    /// Serial reference for the per-SB AQ state advance: compute the target,
    /// quantize it against the running `cur_qidx` accumulator, and retarget the
    /// quantizers. Kept as the bit-exactness oracle for [`precompute_aq_grid`]
    /// (see the `aq_grid_matches_serial` unit test); the SB loop itself
    /// consumes precomputed [`AqCell`]s via [`aq_begin_sb_cell`].
    #[cfg_attr(not(test), allow(dead_code))]
    fn aq_begin_sb(&mut self, sb_x: usize, sb_y: usize) {
        if !self.aq.enabled {
            return;
        }
        let target = self.aq_sb_target(sb_x, sb_y);
        let step = 1i32 << self.aq.res_log2;
        let steps = (((target - self.aq.cur_qidx) as f32) / step as f32)
            .fast_round()
            .clamp(-(AQ_MAX_STEPS as f32), AQ_MAX_STEPS as f32) as i32;
        // The decoder applies Clip3(1,255, cur + steps*step); mirror it exactly so
        // both sides agree on the new qindex even when the clamp bites.
        let newq = (self.aq.cur_qidx + steps * step).clamp(1, 255);
        self.aq_begin_sb_cell(&AqCell {
            newq: newq as u8,
            steps,
        });
    }

    /// Precompute the whole tile's per-SB AQ sequence: a cheap serial raster
    /// pass mirroring `aq_begin_sb`'s accumulator bit-exactly (the ONLY running
    /// state is `cur_qidx`). The wavefront decides SBs out of raster order, so
    /// it reads `grid[row * sb_cols + col]` instead of advancing the
    /// accumulator; the serial emit uses `.steps` for the `read_delta_qindex`
    /// token. Empty when AQ is off.
    fn precompute_aq_grid(&self) -> Vec<AqCell> {
        if !self.aq.enabled {
            return Vec::new();
        }
        let rows = self.h.div_ceil(64);
        let cols = self.w.div_ceil(64);
        let mut grid = Vec::with_capacity(rows * cols);
        let step = 1i32 << self.aq.res_log2;
        let mut cur = self.aq.cur_qidx;
        for r in 0..rows {
            for c in 0..cols {
                let target = self.aq_sb_target(c * 64, r * 64);
                let steps = (((target - cur) as f32) / step as f32)
                    .fast_round()
                    .clamp(-(AQ_MAX_STEPS as f32), AQ_MAX_STEPS as f32)
                    as i32;
                let newq = (cur + steps * step).clamp(1, 255);
                cur = newq;
                grid.push(AqCell {
                    newq: newq as u8,
                    steps,
                });
            }
        }
        grid
    }

    /// Apply one precomputed [`AqCell`] at the start of a superblock: advance
    /// the accumulator to the cell's qindex, arm the delta-q token, and
    /// retarget the quantizers.
    fn aq_begin_sb_cell(&mut self, cell: &AqCell) {
        let newq = cell.newq as i32;
        self.aq.cur_qidx = newq;
        self.aq.pending = cell.steps;
        self.aq.read_deltas = true;
        self.quant = Quant::new_with_qm(newq as u8, self.bd, self.quant.qm_level());
        // The chroma-DC delta is a frame-level constant (DeltaQUDc, derived from
        // the frame base_q_idx and signaled once in the header). The decoder
        // forms the chroma-DC qindex as CurrentQIndex + DeltaQUDc, so apply the
        // frame-level delta to the AQ-adjusted qindex — not chroma_dc_delta(newq).
        let frame_dc_delta = chroma_dc_delta(self.aq.base_q);
        self.cquant = Quant::new_chroma_with_delta_qm(
            newq as u8,
            frame_dc_delta,
            self.bd,
            self.cquant.qm_level(),
        );
    }

    /// Emit a block's skip flag, then the `read_cdef()` / `read_delta_qindex()`
    /// per-SB tokens in spec `intra_frame_mode_info()` order. `read_cdef()`
    /// itself emits nothing here (cdef_bits is decided frame-level after the
    /// tiles are coded); it records the trace point where a replay interleaves
    /// the per-unit `cdef_idx` literal — at the SB's FIRST non-skip block, which
    /// is exactly when the decoder's `cdef_idx[r1][c1] == -1 && !skip` fires.
    fn code_skip_and_sb_tokens(&mut self, block_skip: bool, sctx: usize) {
        self.enc
            .encode_symbol(block_skip as usize, &mut self.cdfs.skip[sctx]);
        if !block_skip && !self.cdef_point_marked {
            self.cdef_point_marked = true;
            self.enc.trace_cdef_mark();
        }
        self.code_delta_q_if_armed();
    }

    /// Emit the `read_delta_qindex()` token for the first block of a superblock,
    /// if armed (spec `ReadDeltas`). Codes the magnitude with the adaptive
    /// `delta_q` CDF, the `DELTA_Q_SMALL` literal escape for magnitudes >= 3, and
    /// the equiprobable sign bit. Called immediately after the block-skip symbol,
    /// matching `intra_frame_mode_info()` ordering (the `read_cdef()` trace point
    /// precedes it; see `code_skip_and_sb_tokens`). No-op when AQ is off or
    /// already emitted for this SB.
    fn code_delta_q_if_armed(&mut self) {
        if !self.aq.read_deltas {
            return;
        }
        self.aq.read_deltas = false;
        let m = self.aq.pending.unsigned_abs() as i32;
        const DELTA_Q_SMALL: usize = 3;
        if m < DELTA_Q_SMALL as i32 {
            self.enc.encode_symbol(m as usize, &mut self.cdfs.delta_q);
        } else {
            self.enc
                .encode_symbol(DELTA_Q_SMALL, &mut self.cdfs.delta_q);
            // m = abs_bits + (1 << rem) + 1, rem = floor(log2(m-1)) >= 1.
            let v = (m - 1) as u32;
            let rem = 31 - v.leading_zeros(); // floor(log2(v))
            let abs_bits = v - (1 << rem);
            self.enc.encode_literal(rem - 1, 3); // delta_q_rem_bits (pre-increment)
            self.enc.encode_literal(abs_bits, rem); // delta_q_abs_bits
        }
        if m != 0 {
            self.enc.encode_literal((self.aq.pending < 0) as u32, 1); // sign
        }
    }

    fn perceptual_rd_scale(&self, px: usize, py: usize, dim: usize) -> f32 {
        let k = prdo_k();
        if k == 0.0 {
            return 1.0;
        }
        let refa = self.aq.ref_act;
        if refa <= 0.0 {
            return 1.0;
        }
        let bw = dim.min(self.w.saturating_sub(px));
        let bh = dim.min(self.h.saturating_sub(py));
        if bw == 0 || bh == 0 {
            return 1.0;
        }
        let yp = &self.src[0];
        let (mut sum, mut sum2) = (0i64, 0i64);
        for r in 0..bh {
            let base = (py + r) * self.w + px;
            for &c in &yp[base..base + bw] {
                let v = c as i64;
                sum += v;
                sum2 += v * v;
            }
        }
        let n = (bw * bh) as f32;
        let mean = sum as f32 / n;
        let var = (sum2 as f32 / n - mean * mean).max(0.0);
        let act = (1.0 + var).ln();
        let c = prdo_clamp();
        (k * (act - refa)).exp().clamp(1.0 / c, c)
    }
}
