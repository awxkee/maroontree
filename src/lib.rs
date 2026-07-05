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
#![allow(clippy::manual_clamp, clippy::excessive_precision)]

mod av1_coder;
mod av1_coefs;
mod skip_tables;
mod tile;
mod wht;

mod av2;
mod avif;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod bitwriter;
mod cdef;
mod cdf_tables;
mod coef_q;
pub mod coeff;
mod coeffs;
mod color;
mod cost;
mod dct;
mod encoder;
mod err;
mod idct;
mod intrapred;
mod isobmff;
mod loopfilter;
mod metadata;
mod msac_enc;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod obu;
mod odec;
mod pixel;
mod quant;
mod rangecoder;
mod tables;
mod transform;
mod trellis;
mod util;
mod wiener;

pub mod av2_image {
    pub use crate::av2::simple::*;
}
pub use av2::{Av2Encoder, Av2Frame, Tuning, TxPart, av2_map_quality};
pub use avif::{
    ChromaFormat, EncodeConfig, Speed, encode_gray_alpha8, encode_gray_alpha10,
    encode_gray_alpha12, encode_gray8, encode_gray10, encode_gray12, encode_rgb8, encode_rgb10,
    encode_rgb12, encode_rgba8, encode_rgba8_with_alpha, encode_rgba10, encode_rgba10_with_alpha,
    encode_rgba12, encode_rgba12_with_alpha, encode_yuv8, encode_yuv10, encode_yuv12,
    encode_yuva8_with_alpha, encode_yuva10_with_alpha, encode_yuva12_with_alpha,
};
pub use color::{
    ChromaSamplePosition, Cicp, ColorMetadata, ItutT35, MasteringDisplay, MatrixCoefficients,
    Primaries, TransferFunction,
};
pub use encoder::{
    PlanarImage, encode_lossless, encode_lossless_gray, encode_lossless_gray_alpha,
    encode_lossless_gray_obu, encode_lossless_obu, encode_lossless_with_alpha,
};
pub use err::EncodeError;
pub use metadata::{ContentLightLevel, Metadata, Orientation};
pub use pixel::{BitDepth, Pixel};
