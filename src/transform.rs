#![allow(unused)]
const UNIT_QUANT_SHIFT: i32 = 2;
const UNIT_QUANT_FACTOR: i32 = 1 << UNIT_QUANT_SHIFT; // = 4

/// Forward Walsh–Hadamard. Input is a residual block; output are coefficients.
///
pub(crate) fn fwht4x4(input: &[i32; 16]) -> [i32; 16] {
    let mut mid = [0i32; 16];

    // Pass 0: operate on columns, write column-wise.
    for i in 0..4 {
        let a0 = input[i];
        let b0 = input[4 + i];
        let c0 = input[8 + i];
        let d0 = input[12 + i];

        let mut a1 = a0 + b0;
        let mut d1 = d0 - c0;
        let e1 = (a1 - d1) >> 1;
        let b1 = e1 - b0;
        let c1 = e1 - c0;
        a1 -= c1;
        d1 += b1;

        mid[i] = a1;
        mid[4 + i] = c1;
        mid[8 + i] = d1;
        mid[12 + i] = b1;
    }

    // Pass 1: operate on rows, scale by UNIT_QUANT_FACTOR.
    let mut out = [0i32; 16];
    for i in 0..4 {
        let base = i * 4;
        let a0 = mid[base];
        let b0 = mid[base + 1];
        let c0 = mid[base + 2];
        let d0 = mid[base + 3];

        let mut a1 = a0 + b0;
        let mut d1 = d0 - c0;
        let e1 = (a1 - d1) >> 1;
        let b1 = e1 - b0;
        let c1 = e1 - c0;
        a1 -= c1;
        d1 += b1;

        out[base] = a1 * UNIT_QUANT_FACTOR;
        out[base + 1] = c1 * UNIT_QUANT_FACTOR;
        out[base + 2] = d1 * UNIT_QUANT_FACTOR;
        out[base + 3] = b1 * UNIT_QUANT_FACTOR;
    }
    out
}

/// Inverse Walsh–Hadamard. Input are coefficients; output is the residual.
pub(crate) fn iwht4x4(input: &[i32; 16]) -> [i32; 16] {
    let mut mid = [0i32; 16];

    // Pass 0: rows, undo the forward scale up front.
    for i in 0..4 {
        let base = i * 4;
        let mut a1 = input[base] >> UNIT_QUANT_SHIFT;
        let mut c1 = input[base + 1] >> UNIT_QUANT_SHIFT;
        let mut d1 = input[base + 2] >> UNIT_QUANT_SHIFT;
        let b0 = input[base + 3] >> UNIT_QUANT_SHIFT;

        a1 += c1;
        d1 -= b0;
        let e1 = (a1 - d1) >> 1;
        let b1 = e1 - b0;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;

        mid[base] = a1;
        mid[base + 1] = b1;
        mid[base + 2] = c1;
        mid[base + 3] = d1;
    }

    // Pass 1: columns.
    let mut out = [0i32; 16];
    for i in 0..4 {
        let mut a1 = mid[i];
        let mut c1 = mid[4 + i];
        let mut d1 = mid[8 + i];
        let b0 = mid[12 + i];

        a1 += c1;
        d1 -= b0;
        let e1 = (a1 - d1) >> 1;
        let b1 = e1 - b0;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;

        out[i] = a1;
        out[4 + i] = b1;
        out[8 + i] = c1;
        out[12 + i] = d1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple xorshift so tests need no external rng crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn residual(&mut self, bits: u32) -> i32 {
            // residual range for `bits`-deep pixels is roughly +/- 2^bits
            let span = 1i64 << (bits + 1);
            ((self.next() % (span as u64)) as i64 - (1i64 << bits)) as i32
        }
    }

    #[test]
    fn transform_roundtrip() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for depth in [8u32, 10, 12] {
            for _ in 0..50_000 {
                let mut block = [0i32; 16];
                for v in block.iter_mut() {
                    *v = rng.residual(depth);
                }
                let coeffs = fwht4x4(&block);
                let recon = iwht4x4(&coeffs);
                assert_eq!(block, recon, "WHT not lossless at depth {depth}");
            }
        }
    }

    #[test]
    fn dc_only_block() {
        // A flat residual must produce a single DC coefficient and reconstruct.
        let block = [7i32; 16];
        let coeffs = fwht4x4(&block);
        // All AC terms should be zero for a constant block.
        for &c in &coeffs[1..] {
            assert_eq!(c, 0, "constant block leaked into AC");
        }
        assert_eq!(iwht4x4(&coeffs), block);
    }
}
