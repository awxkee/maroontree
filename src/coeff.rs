//! Compact, self-consistent coefficient coding for a 4x4 WHT block.
//!
//! Per coefficient: an adaptive binary "is non-zero" flag; then for non-zero
//! values an adaptive *bit-length* symbol (Exp-Golomb-style prefix) plus raw
//! mantissa bits and a sign bit. Small magnitudes — the common case after
//! prediction — get short codes, so this actually compresses, while staying
//! exactly invertible.
//!
//! This is NOT AV1's coefficient syntax (no EOB / base-range / golomb level
//! maps, no scan-position contexts). It is a real, compact, decodable scheme
//! that makes the codec *work* end to end. Swapping in AV1's syntax is the
//! remaining step for on-the-wire compatibility.

use crate::rangecoder::{Cdf, RangeDecoder, RangeEncoder};

/// Max coefficient bit-length we model. 12-bit input through the WHT (scaled by
/// 4) stays well under 2^24, so 24 is safe; encode asserts this.
pub const MAX_LEN: usize = 24;

/// Adaptive models reused across an entire image (one shared instance).
pub struct CoeffCdfs {
    pub nz: Cdf,
    pub len: Cdf,
}

impl Default for CoeffCdfs {
    fn default() -> Self {
        CoeffCdfs {
            nz: Cdf::uniform(2),
            len: Cdf::uniform(MAX_LEN),
        }
    }
}

pub fn encode_block(enc: &mut RangeEncoder, coeffs: &[i32; 16], cdfs: &mut CoeffCdfs) {
    for &c in coeffs.iter() {
        let nz = (c != 0) as usize;
        enc.encode_symbol(nz, &mut cdfs.nz);
        if nz == 1 {
            let mag = c.unsigned_abs();
            let bl = (32 - mag.leading_zeros()) as usize; // >= 1
            debug_assert!(bl <= MAX_LEN, "coeff magnitude {mag} exceeds MAX_LEN");
            enc.encode_symbol(bl - 1, &mut cdfs.len);
            if bl > 1 {
                let mantissa = mag & ((1u32 << (bl - 1)) - 1);
                enc.encode_literal(mantissa, (bl - 1) as u32);
            }
            enc.encode_literal((c < 0) as u32, 1);
        }
    }
}

pub fn decode_block(dec: &mut RangeDecoder, cdfs: &mut CoeffCdfs) -> [i32; 16] {
    let mut coeffs = [0i32; 16];
    for slot in coeffs.iter_mut() {
        let nz = dec.decode_symbol(&mut cdfs.nz);
        if nz == 1 {
            let bl = dec.decode_symbol(&mut cdfs.len) as u32 + 1;
            let mantissa = if bl > 1 {
                dec.decode_literal(bl - 1)
            } else {
                0
            };
            let mag = (1u32 << (bl - 1)) | mantissa;
            let sign = dec.decode_literal(1);
            *slot = if sign == 1 { -(mag as i32) } else { mag as i32 };
        }
    }
    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn block_coeffs_roundtrip() {
        let mut rng = Rng(0xC0FFEE);
        let mut blocks = Vec::new();
        let mut enc = RangeEncoder::new();
        let mut cdfs = CoeffCdfs::default();
        for _ in 0..5000 {
            let mut b = [0i32; 16];
            for v in b.iter_mut() {
                // mostly small, occasionally large, with zeros
                let r = rng.next();
                *v = match r % 4 {
                    0 => 0,
                    1 => (r as i32 % 7) - 3,
                    2 => (r as i32 % 200) - 100,
                    _ => (r as i32 % 60000) - 30000,
                };
            }
            encode_block(&mut enc, &b, &mut cdfs);
            blocks.push(b);
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes);
        let mut cdfs_d = CoeffCdfs::default();
        for b in blocks {
            assert_eq!(decode_block(&mut dec, &mut cdfs_d), b);
        }
    }
}
