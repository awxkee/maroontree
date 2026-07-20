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
use crate::obu::temporal_delimiter;
use crate::par::Pool;

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

#[allow(clippy::too_many_arguments)]
fn encode_lossy_frame(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    planes: [&[i32]; 3],
    layout: PixelLayout<'_>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Vec<u8> {
    assert!(width > 0 && height > 0, "width/height must be non-zero");
    assert_eq!(planes[0].len(), width * height, "luma plane must be w*h");

    let (sub_x, sub_y) = layout.subsampling();
    let monochrome = layout.is_monochrome();
    let (padded_width, padded_height) = (align8(width), align8(height));
    let mut source = [
        pad_to_mult8(planes[0], width, height, padded_width, padded_height),
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
            source[plane] = pad_to_mult8(
                planes[plane],
                chroma_width,
                chroma_height,
                padded_chroma_width,
                padded_chroma_height,
            );
        }
    }

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
            pool,
            speed,
            aq,
            &variance_boost,
            cdef,
            wiener,
        );

    let mut bytes = temporal_delimiter();
    bytes.extend_from_slice(&layout.sequence_header(width, height, bit_depth));
    bytes.extend_from_slice(&assemble_frame_obus(
        base_q_idx,
        variance_boost.qm,
        &tiling,
        &tilegroup,
        monochrome,
        aq,
        allow_intrabc,
        cdef_params.as_ref(),
        restoration_params.as_ref(),
    ));
    bytes
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_444(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&Cicp>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
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
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_422(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&Cicp>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
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
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_420(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[i32],
    u: &[i32],
    v: &[i32],
    color: Option<&Cicp>,
    pool: &Pool,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
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
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_monochrome(
    base_q_idx: u8,
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[i32],
    full_range: bool,
    threads: usize,
    speed: Speed,
    aq: bool,
    variance_boost: VarianceBoost,
    cdef: bool,
    wiener: bool,
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
    )
}

pub(crate) fn encode_lossless_monochrome(
    bit_depth: u8,
    width: usize,
    height: usize,
    luma: &[i16],
    full_range: bool,
    threads: usize,
    speed: Speed,
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
    ));
    bytes
}
