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

use crate::Speed;
use crate::coder::VarianceBoost;
use crate::dct::{ApplyQmatrixFn, DctDispatch, DctFn};
use crate::idct::IdctDispatch;
use crate::intrapred::IntraPredDispatch;
use crate::kmeans::KmeansDispatch;
use crate::loopfilter::LoopFilterDispatch;
use crate::par::Pool;
use crate::rd_sse::RdDispatch;

pub(crate) struct EncodingContext<'a> {
    pub(crate) thread_pool: &'a Pool,
    pub(crate) speed: Speed,
    pub(crate) boost: VarianceBoost,
    pub(crate) idct: IdctDispatch,
    pub(crate) intrapred: IntraPredDispatch,
    pub(crate) kmeans: KmeansDispatch,
    pub(crate) loopfilter: LoopFilterDispatch,
    pub(crate) rd: RdDispatch,
    pub(crate) apply_qmatrix: ApplyQmatrixFn,

    pub(crate) dct4x4: DctFn<16>,
    pub(crate) dct4x8: DctFn<32>,
    pub(crate) dct8x4: DctFn<32>,
    pub(crate) dct4x16: DctFn<64>,
    pub(crate) dct16x4: DctFn<64>,
    pub(crate) dct8x8: DctFn<64>,
    pub(crate) dct8x16: DctFn<128>,
    pub(crate) dct16x8: DctFn<128>,
    pub(crate) dct16x16: DctFn<256>,
    pub(crate) dct16x32: DctFn<512>,
    pub(crate) dct32x16: DctFn<512>,
    pub(crate) dct32x32: DctFn<1024>,
    pub(crate) adst4x4: DctFn<16>,
    pub(crate) adstdct4x4: DctFn<16>,
    pub(crate) dctadst4x4: DctFn<16>,
    pub(crate) adst4x8: DctFn<32>,
    pub(crate) adstdct4x8: DctFn<32>,
    pub(crate) dctadst4x8: DctFn<32>,
    pub(crate) adst8x8: DctFn<64>,
    pub(crate) adstdct8x8: DctFn<64>,
    pub(crate) dctadst8x8: DctFn<64>,
    pub(crate) adst8x16: DctFn<128>,
    pub(crate) adstdct8x16: DctFn<128>,
    pub(crate) dctadst8x16: DctFn<128>,
    pub(crate) adst16x8: DctFn<128>,
    pub(crate) adstdct16x8: DctFn<128>,
    pub(crate) dctadst16x8: DctFn<128>,
    pub(crate) adst16x16: DctFn<256>,
    pub(crate) adstdct16x16: DctFn<256>,
    pub(crate) dctadst16x16: DctFn<256>,
    pub(crate) fvdct4x4: DctFn<16>,
    pub(crate) fhdct4x4: DctFn<16>,
    pub(crate) fvdct8x8: DctFn<64>,
    pub(crate) fhdct8x8: DctFn<64>,
    pub(crate) fvdct8x16: DctFn<128>,
    pub(crate) fhdct8x16: DctFn<128>,
    pub(crate) fvdct16x8: DctFn<128>,
    pub(crate) fhdct16x8: DctFn<128>,
    pub(crate) idtx4x4: DctFn<16>,
    pub(crate) idtx8x8: DctFn<64>,
    pub(crate) idtx8x16: DctFn<128>,
    pub(crate) idtx16x8: DctFn<128>,
    pub(crate) idtx16x16: DctFn<256>,
}

impl<'a> EncodingContext<'a> {
    pub(crate) fn new(thread_pool: &'a Pool, speed: Speed, boost: VarianceBoost) -> Self {
        let dct = DctDispatch::selected();
        let idct = IdctDispatch::selected();
        let intrapred = IntraPredDispatch::selected();
        let kmeans = KmeansDispatch::selected();
        let loopfilter = LoopFilterDispatch::selected();
        let rd = RdDispatch::selected();
        Self {
            thread_pool,
            speed,
            boost,
            idct,
            intrapred,
            kmeans,
            loopfilter,
            rd,
            apply_qmatrix: dct.apply_qmatrix,
            dct4x4: dct.dct4x4,
            dct4x8: dct.dct4x8,
            dct8x4: dct.dct8x4,
            dct4x16: dct.dct4x16,
            dct16x4: dct.dct16x4,
            dct8x8: dct.dct8x8,
            dct8x16: dct.dct8x16,
            dct16x8: dct.dct16x8,
            dct16x16: dct.dct16x16,
            dct16x32: dct.dct16x32,
            dct32x16: dct.dct32x16,
            dct32x32: dct.dct32x32,
            adst4x4: dct.adst4x4,
            adstdct4x4: dct.adstdct4x4,
            dctadst4x4: dct.dctadst4x4,
            adst4x8: dct.adst4x8,
            adstdct4x8: dct.adstdct4x8,
            dctadst4x8: dct.dctadst4x8,
            adst8x8: dct.adst8x8,
            adstdct8x8: dct.adstdct8x8,
            dctadst8x8: dct.dctadst8x8,
            adst8x16: dct.adst8x16,
            adstdct8x16: dct.adstdct8x16,
            dctadst8x16: dct.dctadst8x16,
            adst16x8: dct.adst16x8,
            adstdct16x8: dct.adstdct16x8,
            dctadst16x8: dct.dctadst16x8,
            adst16x16: dct.adst16x16,
            adstdct16x16: dct.adstdct16x16,
            dctadst16x16: dct.dctadst16x16,
            fvdct4x4: dct.fvdct4x4,
            fhdct4x4: dct.fhdct4x4,
            fvdct8x8: dct.fvdct8x8,
            fhdct8x8: dct.fhdct8x8,
            fvdct8x16: dct.fvdct8x16,
            fhdct8x16: dct.fhdct8x16,
            fvdct16x8: dct.fvdct16x8,
            fhdct16x8: dct.fhdct16x8,
            idtx4x4: dct.idtx4x4,
            idtx8x8: dct.idtx8x8,
            idtx8x16: dct.idtx8x16,
            idtx16x8: dct.idtx16x8,
            idtx16x16: dct.idtx16x16,
        }
    }

    #[inline]
    pub(crate) fn dct_dispatch(&self) -> DctDispatch {
        DctDispatch {
            apply_qmatrix: self.apply_qmatrix,
            dct4x4: self.dct4x4,
            dct4x8: self.dct4x8,
            dct8x4: self.dct8x4,
            dct4x16: self.dct4x16,
            dct16x4: self.dct16x4,
            dct8x8: self.dct8x8,
            dct8x16: self.dct8x16,
            dct16x8: self.dct16x8,
            dct16x16: self.dct16x16,
            dct16x32: self.dct16x32,
            dct32x16: self.dct32x16,
            dct32x32: self.dct32x32,
            adst4x4: self.adst4x4,
            adstdct4x4: self.adstdct4x4,
            dctadst4x4: self.dctadst4x4,
            adst4x8: self.adst4x8,
            adstdct4x8: self.adstdct4x8,
            dctadst4x8: self.dctadst4x8,
            adst8x8: self.adst8x8,
            adstdct8x8: self.adstdct8x8,
            dctadst8x8: self.dctadst8x8,
            adst8x16: self.adst8x16,
            adstdct8x16: self.adstdct8x16,
            dctadst8x16: self.dctadst8x16,
            adst16x8: self.adst16x8,
            adstdct16x8: self.adstdct16x8,
            dctadst16x8: self.dctadst16x8,
            adst16x16: self.adst16x16,
            adstdct16x16: self.adstdct16x16,
            dctadst16x16: self.dctadst16x16,
            fvdct4x4: self.fvdct4x4,
            fhdct4x4: self.fhdct4x4,
            fvdct8x8: self.fvdct8x8,
            fhdct8x8: self.fhdct8x8,
            fvdct8x16: self.fvdct8x16,
            fhdct8x16: self.fhdct8x16,
            fvdct16x8: self.fvdct16x8,
            fhdct16x8: self.fhdct16x8,
            idtx4x4: self.idtx4x4,
            idtx8x8: self.idtx8x8,
            idtx8x16: self.idtx8x16,
            idtx16x8: self.idtx16x8,
            idtx16x16: self.idtx16x16,
        }
    }

    #[inline]
    pub(crate) fn idct_dispatch(&self) -> IdctDispatch {
        self.idct
    }
}
