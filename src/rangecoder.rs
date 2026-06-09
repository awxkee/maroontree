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

pub(crate) const CDF_TOTAL: u32 = 1 << 15; // 32768

const TOP: u32 = 1 << 24;
const BOT: u32 = 1 << 16;

/// An adaptive CDF model. Construct with [`Cdf::uniform`] or [`Cdf::from_cumulative`].
#[derive(Clone, Debug)]
pub(crate) struct Cdf {
    /// length N+1: cumulative[0..N-1], cumulative[N-1]==32768, [N]=count
    c: Vec<u16>,
}

impl Cdf {
    pub(crate) fn nsyms(&self) -> usize {
        self.c.len() - 1
    }

    /// Uniform initial distribution over `n` symbols.
    pub(crate) fn uniform(n: usize) -> Self {
        assert!(n >= 2);
        let mut c = vec![0u16; n + 1];
        for (i, dst) in c[..n].iter_mut().enumerate() {
            // cumulative upper bound of symbol i
            *dst = (((i as u64 + 1) * CDF_TOTAL as u64) / n as u64) as u16;
        }
        c[n - 1] = CDF_TOTAL as u16;
        c[n] = 0; // adaptation count
        Cdf { c }
    }

    /// Build from explicit cumulative upper bounds (length N, last must be 32768).
    #[allow(unused)]
    pub(crate) fn from_cumulative(cum: &[u16]) -> Self {
        let n = cum.len();
        assert!(n >= 2);
        assert_eq!(
            cum[n - 1] as u32,
            CDF_TOTAL,
            "last cumulative must be 32768"
        );
        let mut c = vec![0u16; n + 1];
        c[..n].copy_from_slice(cum);
        c[n] = 0;
        Cdf { c }
    }

    #[inline]
    fn boundary(&self, k: usize) -> u32 {
        if k == 0 { 0 } else { self.c[k - 1] as u32 }
    }

    /// Coding boundary with a guaranteed +1 minimum gap per symbol.
    ///
    /// Adaptation can drive a rare symbol's raw interval to zero width, which
    /// the range engine cannot encode. AV1's real engine reserves a floor
    /// (`EC_MIN_PROB`); we get the same robustness, symmetrically on both
    /// encoder and decoder, by widening boundary `k` by `k`. This leaves the
    /// spec-exact `update` untouched and only affects how intervals are mapped.
    #[inline]
    fn eff_boundary(&self, k: usize) -> u32 {
        self.boundary(k) + k as u32
    }

    /// Total of the effective interval (raw 32768 plus one unit per gap).
    #[inline]
    fn eff_total(&self) -> u32 {
        CDF_TOTAL + self.nsyms() as u32
    }

    /// AV1 spec §8.3.2 CDF update. This part is bit-exact to the spec.
    fn update(&mut self, symbol: usize) {
        let n = self.nsyms();
        let count = self.c[n] as u32;
        let rate = 3
            + (if count > 15 { 1 } else { 0 })
            + (if count > 31 { 1 } else { 0 })
            + floor_log2(n as u32).min(2);
        for i in 0..n - 1 {
            let cur = self.c[i] as i32;
            // Spec §8.3.2: tmp starts at 0 and becomes 32768 at i == symbol,
            // so cdf[i] relaxes toward 0 for i < symbol and toward 32768 for
            // i >= symbol. This keeps the cumulative CDF monotonic increasing.
            let tmp = if i >= symbol { CDF_TOTAL as i32 } else { 0 };
            // move cdf[i] toward tmp by >> rate
            let next = if tmp < cur {
                cur - ((cur - tmp) >> rate)
            } else {
                cur + ((tmp - cur) >> rate)
            };
            self.c[i] = next as u16;
        }
        if count < 32 {
            self.c[n] = (count + 1) as u16;
        }
    }
}

#[inline]
fn floor_log2(mut x: u32) -> u32 {
    let mut r = 0;
    while x > 1 {
        x >>= 1;
        r += 1;
    }
    r
}

pub(crate) struct RangeEncoder {
    low: u32,
    range: u32,
    out: Vec<u8>,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    pub(crate) fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0xFFFF_FFFF,
            out: Vec::new(),
        }
    }

    #[inline]
    fn encode_freq(&mut self, cum: u32, freq: u32, tot: u32) {
        let r = self.range / tot;
        self.low = self.low.wrapping_add(r.wrapping_mul(cum));
        self.range = r.wrapping_mul(freq);
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) < TOP {
                // top byte settled
            } else if self.range < BOT {
                self.range = self.low.wrapping_neg() & (BOT - 1);
            } else {
                break;
            }
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    /// Encode `symbol` against `cdf`, then adapt the model.
    pub(crate) fn encode_symbol(&mut self, symbol: usize, cdf: &mut Cdf) {
        debug_assert!(symbol < cdf.nsyms());
        let lo = cdf.eff_boundary(symbol);
        let hi = cdf.eff_boundary(symbol + 1);
        self.encode_freq(lo, hi - lo, cdf.eff_total());
        cdf.update(symbol);
    }

    /// Encode `nbits` raw bits (MSB first) with a uniform model (bypass-style).
    pub(crate) fn encode_literal(&mut self, value: u32, nbits: u32) {
        for i in (0..nbits).rev() {
            let bit = (value >> i) & 1;
            // uniform 2-symbol: each bit owns half the interval
            self.encode_freq(bit * (CDF_TOTAL / 2), CDF_TOTAL / 2, CDF_TOTAL);
        }
    }

    #[allow(unused)]
    pub(crate) fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        self.out
    }
}
