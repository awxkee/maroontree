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

use crate::Speed;
use crate::coder::{
    VarianceBoost, align8, assemble_frame_obus, encode_lossless_mono_frame_obus,
    encode_lossy_tilegroup, pad_to_mult8,
};
use crate::color::Cicp;
use crate::encoding_context::EncodingContext;
use crate::obu::temporal_delimiter;
use crate::par::Pool;
use crate::pixel::Pixel;

#[derive(Clone, Copy)]
enum PixelLayout<'a> {
    Monochrome { full_range: bool },
    Yuv444 { color: Option<&'a Cicp> },
    Yuv422 { color: Option<&'a Cicp> },
    Yuv420 { color: Option<&'a Cicp> },
}

impl PixelLayout<'_> {
    fn subsampling(self) -> (usize, usize) {
        match self {
            Self::Monochrome { .. } | Self::Yuv444 { .. } => (0, 0),
            Self::Yuv422 { .. } => (1, 0),
            Self::Yuv420 { .. } => (1, 1),
        }
    }

    fn is_monochrome(self) -> bool {
        matches!(self, Self::Monochrome { .. })
    }

    fn sequence_header(self, width: usize, height: usize, bit_depth: u8) -> Vec<u8> {
        match self {
            Self::Monochrome { full_range } => crate::obu::sequence_header_mono(
                width as u32,
                height as u32,
                bit_depth,
                full_range,
                true,
                true,
            ),
            Self::Yuv444 { color } => {
                let profile = if bit_depth == 12 { 2 } else { 1 };
                crate::obu::sequence_header_cicp(
                    width as u32,
                    height as u32,
                    profile,
                    bit_depth,
                    color,
                    true,
                    true,
                )
            }
            Self::Yuv422 { color } => crate::obu::sequence_header_cicp_ss(
                width as u32,
                height as u32,
                2,
                bit_depth,
                color,
                1,
                0,
                true,
                true,
            ),
            Self::Yuv420 { color } => {
                let profile = if bit_depth == 12 { 2 } else { 0 };
                crate::obu::sequence_header_cicp_ss(
                    width as u32,
                    height as u32,
                    profile,
                    bit_depth,
                    color,
                    1,
                    1,
                    true,
                    true,
                )
            }
        }
    }
}

fn pad_to_mult8_u16<T: Pixel>(
    src: &[T],
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    bit_depth: u8,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(padded_width * padded_height);
    for row in src.chunks_exact(width).take(height) {
        out.extend(row.iter().map(|&sample| sample.to_u16_clamped(bit_depth)));
        let edge = *out.last().expect("plane width must be non-zero");
        out.resize(out.len() + padded_width - width, edge);
    }
    for _ in height..padded_height {
        out.extend_from_within((height - 1) * padded_width..height * padded_width);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn encode_lossy_frame<T: Pixel>(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    planes: [&[T]; 3],
    layout: PixelLayout<'_>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Vec<u8> {
    assert!(width > 0 && height > 0, "width/height must be non-zero");
    assert_eq!(planes[0].len(), width * height, "luma plane must be w*h");
    let (base_q_idx, variance_boost) = {
        let shift = if aq && variance_boost.enabled {
            crate::coder::baseq_shift(base_q_idx)
        } else {
            0
        };
        if shift != 0 {
            let shifted = (base_q_idx as i32 + shift).clamp(1, 254) as u8;
            let mut vb = variance_boost;
            vb.base_shift = shift;
            // QM + chroma-delta laws must follow the base the DECODER sees.
            let (sx, sy) = layout.subsampling();
            let sub = sx + sy;
            let c = crate::quant::qm_chroma_level_law(shifted, sub);
            vb.qm = crate::quant::QmLevels {
                y: crate::quant::qm_level_law(shifted, sub),
                u: c,
                v: c,
            };
            (shifted, vb)
        } else {
            (base_q_idx, variance_boost)
        }
    };

    let (sub_x, sub_y) = layout.subsampling();
    let monochrome = layout.is_monochrome();
    let (padded_width, padded_height) = (align8(width), align8(height));
    // Convert/clamp and edge-pad directly into the lossy coder's u16 storage.
    let mut source = [
        pad_to_mult8_u16(
            planes[0],
            width,
            height,
            padded_width,
            padded_height,
            bit_depth,
        ),
        Vec::new(),
        Vec::new(),
    ];

    if !monochrome {
        let chroma_width = width.div_ceil(1 << sub_x);
        let chroma_height = height.div_ceil(1 << sub_y);
        let padded_chroma_width = padded_width >> sub_x;
        let padded_chroma_height = padded_height >> sub_y;
        for plane in 1..=2 {
            assert_eq!(
                planes[plane].len(),
                chroma_width * chroma_height,
                "chroma plane has invalid dimensions"
            );
            source[plane] = pad_to_mult8_u16(
                planes[plane],
                chroma_width,
                chroma_height,
                padded_chroma_width,
                padded_chroma_height,
                bit_depth,
            );
        }
    }

    let context = EncodingContext::new(pool, speed, variance_boost);
    let (tilegroup, tiling, cdef_params, restoration_params, allow_intrabc) =
        encode_lossy_tilegroup(
            base_q_idx,
            bit_depth,
            padded_width,
            padded_height,
            width,
            height,
            &source,
            sub_x,
            sub_y,
            monochrome,
            &context,
            aq,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        );

    let mut bytes = temporal_delimiter();
    bytes.extend_from_slice(&layout.sequence_header(width, height, bit_depth));
    bytes.extend_from_slice(&assemble_frame_obus(
        base_q_idx,
        variance_boost.qm,
        crate::quant::chroma_ac_delta(base_q_idx, sub_x + sub_y),
        crate::quant::chroma_dc_delta(base_q_idx, sub_x + sub_y),
        &tiling,
        &tilegroup,
        monochrome,
        aq,
        allow_intrabc,
        cdef_params.as_ref(),
        restoration_params.as_ref(),
        updating_cdf,
    ));
    bytes
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_444<T: Pixel>(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[T],
    u: &[T],
    v: &[T],
    color: Option<&Cicp>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Vec<u8> {
    encode_lossy_frame(
        base_q_idx,
        bit_depth,
        width,
        height,
        [luma, u, v],
        PixelLayout::Yuv444 { color },
        pool,
        speed,
        aq,
        variance_boost,
        cdef,
        wiener,
        updating_cdf,
        screen_content,
        intrabc,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_422<T: Pixel>(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[T],
    u: &[T],
    v: &[T],
    color: Option<&Cicp>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Vec<u8> {
    encode_lossy_frame(
        base_q_idx,
        bit_depth,
        width,
        height,
        [luma, u, v],
        PixelLayout::Yuv422 { color },
        pool,
        speed,
        aq,
        variance_boost,
        cdef,
        wiener,
        updating_cdf,
        screen_content,
        intrabc,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_420<T: Pixel>(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[T],
    u: &[T],
    v: &[T],
    color: Option<&Cicp>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Vec<u8> {
    encode_lossy_frame(
        base_q_idx,
        bit_depth,
        width,
        height,
        [luma, u, v],
        PixelLayout::Yuv420 { color },
        pool,
        speed,
        aq,
        variance_boost,
        cdef,
        wiener,
        updating_cdf,
        screen_content,
        intrabc,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_monochrome<T: Pixel>(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[T],
    full_range: bool,
    threads: usize,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Vec<u8> {
    let pool = Pool::new(threads);
    encode_lossy_frame(
        base_q_idx,
        bit_depth,
        width,
        height,
        [luma, &[], &[]],
        PixelLayout::Monochrome { full_range },
        &pool,
        speed,
        aq,
        variance_boost,
        cdef,
        wiener,
        updating_cdf,
        screen_content,
        intrabc,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossless_monochrome(
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[i16],
    full_range: bool,
    threads: usize,
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    assert!(width > 0 && height > 0, "width/height must be non-zero");
    assert_eq!(luma.len(), width * height, "luma plane must be w*h");
    let (padded_width, padded_height) = (align8(width), align8(height));
    let padded = pad_to_mult8(luma, width, height, padded_width, padded_height);

    let mut bytes = temporal_delimiter();
    bytes.extend_from_slice(&crate::obu::sequence_header_mono(
        width as u32,
        height as u32,
        bit_depth,
        full_range,
        false,
        false,
    ));
    bytes.extend_from_slice(&encode_lossless_mono_frame_obus(
        bit_depth,
        padded_width,
        padded_height,
        width,
        height,
        &padded,
        threads,
        speed,
        updating_cdf,
    ));
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static U16_CONVERSIONS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    struct DirectPixel(u16);

    impl Pixel for DirectPixel {
        fn to_i32(self) -> i32 {
            panic!("u16 padding must not widen through i32")
        }

        fn to_u16_clamped(self, bit_depth: u8) -> u16 {
            U16_CONVERSIONS.fetch_add(1, Ordering::Relaxed);
            self.0.min((1u16 << bit_depth) - 1)
        }

        fn to_f32(self) -> f32 {
            panic!("u16 padding must not convert through f32")
        }

        fn from_i32_clamped(_v: i32, _bit_depth: u8) -> Self {
            unreachable!()
        }
    }

    #[test]
    fn u16_conversion_and_padding_share_one_pass() {
        U16_CONVERSIONS.store(0, Ordering::Relaxed);
        let src = [
            DirectPixel(12),
            DirectPixel(300),
            DirectPixel(5),
            DirectPixel(40),
        ];
        let padded = pad_to_mult8_u16(&src, 2, 2, 8, 8, 8);

        assert_eq!(U16_CONVERSIONS.load(Ordering::Relaxed), src.len());
        assert_eq!(&padded[..8], &[12, 255, 255, 255, 255, 255, 255, 255]);
        assert_eq!(&padded[8..16], &[5, 40, 40, 40, 40, 40, 40, 40]);
        for row in padded[16..].chunks_exact(8) {
            assert_eq!(row, &padded[8..16]);
        }
    }
}
