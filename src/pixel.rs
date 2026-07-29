/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::err::EncodeError;

pub trait Pixel: Copy + Default + PartialEq + Send + Sync + std::fmt::Debug {
    /// Promote a stored sample to the signed working type used by the transform.
    fn to_i32(self) -> i32;
    /// Convert an input sample directly to the lossy coder's unsigned storage,
    /// clipped to `[0, (1<<bit_depth)-1]`.
    fn to_u16_clamped(self, bit_depth: u8) -> u16;
    fn to_f32(self) -> f32;
    /// Clip a reconstructed signed value back into `[0, (1<<bit_depth)-1]`.
    fn from_i32_clamped(v: i32, bit_depth: u8) -> Self;
}

impl Pixel for u8 {
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline]
    fn to_u16_clamped(self, _bit_depth: u8) -> u16 {
        self as u16
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_i32_clamped(v: i32, _bit_depth: u8) -> Self {
        v.clamp(0, 255) as u8
    }
}

impl Pixel for u16 {
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline]
    fn to_u16_clamped(self, bit_depth: u8) -> u16 {
        self.min((1u16 << bit_depth) - 1)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_i32_clamped(v: i32, bit_depth: u8) -> Self {
        let max = (1i32 << bit_depth) - 1;
        v.clamp(0, max) as u16
    }
}

impl Pixel for i32 {
    #[inline]
    fn to_i32(self) -> i32 {
        self
    }
    #[inline]
    fn to_u16_clamped(self, bit_depth: u8) -> u16 {
        let max = (1i32 << bit_depth) - 1;
        self.clamp(0, max) as u16
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_i32_clamped(v: i32, bit_depth: u8) -> Self {
        let max = (1i32 << bit_depth) - 1;
        v.clamp(0, max)
    }
}

impl Pixel for f32 {
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline]
    fn to_u16_clamped(self, bit_depth: u8) -> u16 {
        let max = ((1u16 << bit_depth) - 1) as f32;
        self.clamp(0.0, max) as u16
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_i32_clamped(v: i32, bit_depth: u8) -> Self {
        let max = (1i32 << bit_depth) - 1;
        v.clamp(0, max) as f32
    }
}

/// Supported coded bit depths. AV1 profile 0/1 cover 8 and 10; 12 needs
/// profile 2. We model all three; only storage type differs (u8 vs u16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDepth {
    Eight,
    Ten,
    Twelve,
}

impl BitDepth {
    pub fn bits(self) -> u8 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
            BitDepth::Twelve => 12,
        }
    }

    pub fn from_u8(bit_depth: u8) -> Result<Self, EncodeError> {
        match bit_depth {
            8 => Ok(BitDepth::Eight),
            10 => Ok(BitDepth::Ten),
            12 => Ok(BitDepth::Twelve),
            _ => Err(EncodeError::InvalidInput),
        }
    }
}
