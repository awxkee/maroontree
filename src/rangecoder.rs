//! Multisymbol range coder with 15-bit (Q15) CDFs.
//!
//! ## Honesty note — read this before trusting the bytes
//! The *arithmetic engine* here is a clean, self-consistent carryless range
//! coder (Subbotin form). It is NOT verified bit-exact against libaom's `od_ec`
//! carry/normalize bookkeeping. The architecture is deliberately AV1-shaped —
//! Q15 CDFs, per-symbol `encode_symbol`, and the exact AV1 *CDF adaptation* rule
//! (`update_cdf`, faithful to spec §8.3.2) — so a verified `od_ec` engine can be
//! dropped in behind this same interface without touching callers.
//!
//! What IS proven (see tests): encoder and decoder are exact inverses for
//! arbitrary symbol/CDF streams, and adaptive coding round-trips.
//!
//! CDF layout matches the AV1 spec representation: for an N-ary symbol, `cdf`
//! has length `N + 1`. `cdf[0..N-1]` are increasing cumulative frequencies with
//! `cdf[N-1] == 32768`; `cdf[N]` is the adaptation counter. Symbol `s` owns the
//! interval `[b(s), b(s+1))` where `b(0)=0`, `b(k)=cdf[k-1]`, `b(N)=32768`.

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

pub(crate) struct RangeDecoder<'a> {
    low: u32,
    range: u32,
    code: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    pub(crate) fn new(input: &'a [u8]) -> Self {
        let mut d = RangeDecoder {
            low: 0,
            range: 0xFFFF_FFFF,
            code: 0,
            input,
            pos: 0,
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next_byte() as u32;
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline]
    fn renorm(&mut self) {
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) < TOP {
            } else if self.range < BOT {
                self.range = self.low.wrapping_neg() & (BOT - 1);
            } else {
                break;
            }
            self.code = (self.code << 8) | self.next_byte() as u32;
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    pub(crate) fn decode_symbol(&mut self, cdf: &mut Cdf) -> usize {
        let tot = cdf.eff_total();
        let r = self.range / tot;
        let dv = (self.code.wrapping_sub(self.low) / r).min(tot - 1);
        // find symbol s with eff_boundary(s) <= dv < eff_boundary(s+1)
        let n = cdf.nsyms();
        let mut s = 0;
        while s < n - 1 && cdf.eff_boundary(s + 1) <= dv {
            s += 1;
        }
        let lo = cdf.eff_boundary(s);
        let hi = cdf.eff_boundary(s + 1);
        self.low = self.low.wrapping_add(r.wrapping_mul(lo));
        self.range = r.wrapping_mul(hi - lo);
        self.renorm();
        cdf.update(s);
        s
    }

    pub(crate) fn decode_literal(&mut self, nbits: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..nbits {
            let r = self.range / CDF_TOTAL;
            let dv = (self.code.wrapping_sub(self.low) / r).min(CDF_TOTAL - 1);
            let bit = if dv >= CDF_TOTAL / 2 { 1 } else { 0 };
            self.low = self.low.wrapping_add(r.wrapping_mul(bit * (CDF_TOTAL / 2)));
            self.range = r.wrapping_mul(CDF_TOTAL / 2);
            self.renorm();
            v = (v << 1) | bit;
        }
        v
    }
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
    fn literal_roundtrip() {
        let mut enc = RangeEncoder::new();
        let mut rng = Rng(1);
        let mut vals = Vec::new();
        for _ in 0..10_000 {
            let nbits = 1 + (rng.next() % 16) as u32;
            let v = (rng.next() as u32) & ((1u32 << nbits) - 1);
            vals.push((v, nbits));
            enc.encode_literal(v, nbits);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (v, nbits) in vals {
            assert_eq!(dec.decode_literal(nbits), v);
        }
    }

    #[test]
    fn adaptive_symbol_roundtrip() {
        // Skewed source so adaptation actually moves the CDF.
        let mut rng = Rng(0xDEAD_BEEF);
        let n = 5;
        let mut syms = Vec::new();
        for _ in 0..50_000 {
            let r = rng.next() % 100;
            let s = if r < 70 {
                0
            } else if r < 85 {
                1
            } else if r < 93 {
                2
            } else if r < 98 {
                3
            } else {
                4
            };
            syms.push(s as usize);
        }

        let mut enc = RangeEncoder::new();
        let mut cdf_e = Cdf::uniform(n);
        for &s in &syms {
            enc.encode_symbol(s, &mut cdf_e);
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes);
        let mut cdf_d = Cdf::uniform(n);
        for &s in &syms {
            assert_eq!(dec.decode_symbol(&mut cdf_d), s);
        }
    }

    #[test]
    fn mixed_stream_roundtrip() {
        let mut enc = RangeEncoder::new();
        let mut cdf_e = Cdf::uniform(3);
        enc.encode_symbol(2, &mut cdf_e);
        enc.encode_literal(0b1011, 4);
        enc.encode_symbol(0, &mut cdf_e);
        enc.encode_literal(0x1F, 5);
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes);
        let mut cdf_d = Cdf::uniform(3);
        assert_eq!(dec.decode_symbol(&mut cdf_d), 2);
        assert_eq!(dec.decode_literal(4), 0b1011);
        assert_eq!(dec.decode_symbol(&mut cdf_d), 0);
        assert_eq!(dec.decode_literal(5), 0x1F);
    }
}
