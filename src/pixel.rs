//! Pixel abstraction.
//!
//! A single `Pixel` trait lets the whole pipeline be generic over sample type,
//! so 8-bit (`u8`) and 10/12-bit (`u16`) share one codepath instead of the
//! duplicated highbd functions you see in C encoders. Transform/quant math
//! always promotes to `i32` (see `transform.rs`); the trait only governs
//! storage and clipping back to the valid range.

pub trait Pixel: Copy + Default + PartialEq + std::fmt::Debug {
    /// Promote a stored sample to the signed working type used by the transform.
    fn to_i32(self) -> i32;
    /// Clip a reconstructed signed value back into `[0, (1<<bit_depth)-1]`.
    fn from_i32_clamped(v: i32, bit_depth: u8) -> Self;
}

impl Pixel for u8 {
    #[inline]
    fn to_i32(self) -> i32 {
        self as i32
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
    fn from_i32_clamped(v: i32, bit_depth: u8) -> Self {
        let max = (1i32 << bit_depth) - 1;
        v.clamp(0, max) as u16
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
}
