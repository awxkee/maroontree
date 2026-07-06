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

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
const WINDOW_SIZE: i16 = 32; // ec_window = u32
const LOTS_OF_BITS: i16 = 0x4000;

#[allow(unused)]
pub(crate) fn uniform_icdf(n: usize) -> Vec<u16> {
    assert!(n >= 2);
    let mut cdf = vec![0u16; n + 1];
    for (i, slot) in cdf.iter_mut().take(n).enumerate() {
        let cumulative = ((i + 1) as u32 * 32768) / n as u32;
        *slot = (32768 - cumulative) as u16; // inverse; entry n-1 becomes 0
    }
    cdf[n] = 0; // counter
    cdf
}

/// AV1 / dav1d CDF adaptation. `cdf` is the inverse-form array (counter last).
pub(crate) fn update_cdf(cdf: &mut [u16], val: usize) {
    let nsymbs = cdf.len();
    let count = cdf[nsymbs - 1] as u32;
    let rate = 3 + (nsymbs >> 1).min(2) as u32 + (count >> 4);
    cdf[nsymbs - 1] = (count + 1 - (count >> 5)) as u16; // saturating counter
    for (i, dst) in cdf[..nsymbs - 1].iter_mut().enumerate() {
        if (i as u32) >= val as u32 {
            *dst -= *dst >> rate;
        } else {
            *dst += (32768 - *dst) >> rate;
        }
    }
}

// ----------------------------------------------------------------------------
// Encoder
// ----------------------------------------------------------------------------

/// Encode-side inverse of the spec's `inverse_recenter(r, v)`. Maps an absolute
/// value `v` in `0..n` to its recentred code so that values near the reference
/// `r` get small codes (spec 4.10.8). This is the exact inverse used by
/// `decode_unsigned_subexp_with_ref`.
pub(crate) fn recenter_finite(n: u32, r: u32, v: u32) -> u32 {
    // Mirror libaom's recenter_finite_nonneg.
    if (r << 1) <= n {
        recenter_nonneg(r, v)
    } else {
        recenter_nonneg(n - 1 - r, n - 1 - v)
    }
}

/// `recenter_nonneg(r, v)`: zig-zag the difference `v - r` around 0.
fn recenter_nonneg(r: u32, v: u32) -> u32 {
    if v > (r << 1) {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

/// Spec `inverse_recenter(r, v)` — the decode-side inverse of `recenter_nonneg`.
pub(crate) fn inverse_recenter(r: u32, v: u32) -> u32 {
    if v > (r << 1) {
        v
    } else if (v & 1) != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// Raw `store` ops with per-superblock `mark`s; replaying through a fresh
/// encoder reproduces the exact bytes while any recorded (fl, fh, nms) holds.
#[derive(Default)]
pub(crate) struct SymbolTrace {
    ops: Vec<u64>, // fl:[24..41) fh:[8..24) nms:[0..8)
    marks: Vec<u32>,
}

impl SymbolTrace {
    #[inline]
    fn push(&mut self, fl: u32, fh: u32, nms: u32) {
        debug_assert!(fl <= 32768 && fh < 65536 && nms < 256);
        self.ops
            .push(((fl as u64) << 24) | ((fh as u64) << 8) | nms as u64);
    }

    pub(crate) fn mark(&mut self) {
        self.marks.push(self.ops.len() as u32);
    }

    pub(crate) fn sb_count(&self) -> usize {
        self.marks.len()
    }

    /// Ops of superblock `i`: from its mark up to the next mark (or the end).
    pub(crate) fn sb_ops(&self, i: usize) -> &[u64] {
        let a = self.marks[i] as usize;
        let b = self
            .marks
            .get(i + 1)
            .map_or(self.ops.len(), |&m| m as usize);
        &self.ops[a..b]
    }
}

pub(crate) struct OdEcEncoder {
    low: u32,
    rng: u16,
    cnt: i16,
    precarry: Vec<u16>,
    trace: Option<Box<SymbolTrace>>,
}

impl Default for OdEcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl OdEcEncoder {
    pub(crate) fn new() -> Self {
        OdEcEncoder {
            low: 0,
            rng: 0x8000,
            cnt: -9,
            precarry: Vec::new(),
            trace: None,
        }
    }

    /// Start recording every subsequent `store` into a fresh trace.
    pub(crate) fn begin_trace(&mut self) {
        self.trace = Some(Box::default());
    }

    /// Record a superblock boundary in the active trace (no-op otherwise).
    pub(crate) fn trace_mark(&mut self) {
        if let Some(t) = self.trace.as_mut() {
            t.mark();
        }
    }

    pub(crate) fn take_trace(&mut self) -> Option<Box<SymbolTrace>> {
        self.trace.take()
    }

    /// Re-emit previously recorded ops through this encoder.
    pub(crate) fn replay(&mut self, ops: &[u64]) {
        for &op in ops {
            self.store(
                (op >> 24) as u32,
                ((op >> 8) & 0xffff) as u32,
                (op & 0xff) as u32,
            );
        }
    }

    /// Returns (low_addend, new_range) for cumulative freqs fl >= fh in Q15.
    #[inline]
    fn lr_compute(&self, fl: u32, fh: u32, nms: u32) -> (u32, u16) {
        let r = self.rng as u32;
        let mut u = (((r >> 8) * (fl >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * nms;
        if fl >= 32768 {
            u = r;
        }
        let v =
            (((r >> 8) * (fh >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (nms - 1);
        (r - u, (u - v) as u16)
    }

    #[inline]
    fn store(&mut self, fl: u32, fh: u32, nms: u32) {
        if let Some(t) = self.trace.as_mut() {
            t.push(fl, fh, nms);
        }
        let (l, r) = self.lr_compute(fl, fh, nms);
        let mut low = l + self.low;
        let mut c = self.cnt;
        let d = r.leading_zeros() as i16; // u16 range -> 0..=16
        let mut s = c + d;
        if s >= 0 {
            c += 16;
            let mut m: u32 = (1u32 << (c as u32)) - 1;
            if s >= 8 {
                self.precarry.push((low >> (c as u32)) as u16);
                low &= m;
                c -= 8;
                m >>= 8;
            }
            self.precarry.push((low >> (c as u32)) as u16);
            s = c + d - 24;
            low &= m;
        }
        self.low = low << (d as u32);
        self.rng = ((r as u32) << (d as u32)) as u16;
        self.cnt = s;
    }

    #[allow(unused)]
    pub(crate) fn enc_rng(&self) -> u16 {
        self.rng
    }
    pub(crate) fn encode_bool(&mut self, val: bool, f: u16) {
        // equivalent to symbol(val, [f, 0]) with nms = 2 - val
        let s = val as u32;
        let cdf = [f as u32, 0u32];
        let nms = 2 - s;
        let fl = if s > 0 { cdf[(s - 1) as usize] } else { 32768 };
        let fh = cdf[s as usize];
        self.store(fl, fh, nms);
    }

    /// Encode `bits` raw bits of `value`, MSB first, with flat probability.
    #[allow(unused)]
    pub(crate) fn encode_literal(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.encode_bool((value >> i) & 1 == 1, 16384);
        }
    }

    /// Encode symbol `s` against an inverse-form `cdf` (NOT adapted).
    pub(crate) fn encode_symbol_noupdate(&mut self, s: usize, cdf: &[u16]) {
        let nms = (cdf.len() - s) as u32;
        let fl = if s > 0 { cdf[s - 1] as u32 } else { 32768 };
        let fh = cdf[s] as u32;
        self.store(fl, fh, nms);
    }

    /// Encode symbol `s`, then adapt `cdf` (dav1d-compatible).
    pub(crate) fn encode_symbol(&mut self, s: usize, cdf: &mut [u16]) {
        self.encode_symbol_noupdate(s, cdf);
        update_cdf(cdf, s);
    }

    /// `ns(n)` — non-symmetric flat coding of a value in `0..n` (spec 4.10.7).
    /// Uses `floor(log2(n))` or that+1 bits so the code is as short as possible.
    pub(crate) fn encode_ns(&mut self, v: u32, n: u32) {
        if n <= 1 {
            return;
        }
        let w = (32 - (n - 1).leading_zeros()).max(1); // ceil(log2(n)) bits max
        let m = (1u32 << w) - n;
        if v < m {
            // (w-1)-bit literal
            self.encode_literal(v, w - 1);
        } else {
            // w-bit literal of (v + m), MSB first
            let coded = v + m;
            // emit high w-1 bits then the low bit (matches spec NS read order)
            self.encode_literal(coded >> 1, w - 1);
            self.encode_bool((coded & 1) == 1, 16384);
        }
    }

    /// `decode_subexp(numSyms, k)` inverse: encode `v` in `0..num_syms` with the
    /// sub-exponential scheme (spec 4.10.6). `k` is the initial exponent; Wiener
    /// taps use per-tap `k` from `Wiener_Taps_K`, not a fixed 3.
    pub(crate) fn encode_subexp(&mut self, v: u32, num_syms: u32, k: u32) {
        let mut i = 0u32;
        let mut mk = 0u32;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                // final: ns(numSyms - mk) of (v - mk)
                self.encode_ns(v - mk, num_syms - mk);
                return;
            } else if v < mk + a {
                // subexp_more_bits = 0, then b2-bit literal of (v - mk)
                self.encode_bool(false, 16384);
                self.encode_literal(v - mk, b2);
                return;
            } else {
                // subexp_more_bits = 1, advance
                self.encode_bool(true, 16384);
                i += 1;
                mk += a;
            }
        }
    }

    /// `decode_unsigned_subexp_with_ref(mx, k, r)` inverse. Encodes `v` in
    /// `0..mx` relative to reference `r` (spec 4.10.8).
    pub(crate) fn encode_unsigned_subexp_with_ref(&mut self, v: u32, mx: u32, k: u32, r: u32) {
        // The spec recentres v around r so small deltas are cheap, then codes
        // the recentred value with encode_subexp over `mx` symbols.
        let recentered = recenter_finite(mx, r, v);
        self.encode_subexp(recentered, mx, k);
    }

    /// `decode_signed_subexp_with_ref(low, high, k, r)` inverse. Encodes `v` in
    /// `[low, high)` relative to reference `r`.
    pub(crate) fn encode_signed_subexp_with_ref(
        &mut self,
        v: i32,
        low: i32,
        high: i32,
        k: u32,
        r: i32,
    ) {
        let x = (v - low) as u32;
        let mx = (high - low) as u32;
        let rr = (r - low) as u32;
        self.encode_unsigned_subexp_with_ref(x, mx, k, rr);
    }

    /// Flush and return the coded bytes.
    pub(crate) fn done(mut self) -> Vec<u8> {
        let l = self.low;
        let mut c = self.cnt;
        let mut s = 10i16;
        let m: u32 = 0x3FFF;
        let mut e: u32 = ((l + m) & !m) | (m + 1);
        s += c;
        if s > 0 {
            let mut n: u32 = (1u32 << ((c + 16) as u32)) - 1;
            loop {
                self.precarry.push((e >> ((c + 16) as u32)) as u16);
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        // Carry propagation from the precarry buffer into output bytes.
        let mut carry = 0u32;
        let mut offs = self.precarry.len();
        let mut out = vec![0u8; offs];
        while offs > 0 {
            offs -= 1;
            carry += self.precarry[offs] as u32;
            out[offs] = carry as u8;
            carry >>= 8;
        }
        out
    }
}

#[allow(unused)]
pub(crate) struct OdEcDecoder<'a> {
    buf: &'a [u8],
    bptr: usize,
    dif: u32,
    rng: u16,
    cnt: i16,
}

#[allow(unused)]
impl<'a> OdEcDecoder<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        let mut r = OdEcDecoder {
            buf,
            bptr: 0,
            dif: (1u32 << (WINDOW_SIZE - 1)) - 1,
            rng: 0x8000,
            cnt: -15,
        };
        r.refill();
        r
    }

    fn refill(&mut self) {
        let mut s = WINDOW_SIZE - 9 - (self.cnt + 15);
        while s >= 0 && self.bptr < self.buf.len() {
            self.dif ^= (self.buf[self.bptr] as u32) << (s as u32);
            self.cnt += 8;
            s -= 8;
            self.bptr += 1;
        }
        if self.bptr >= self.buf.len() {
            self.cnt = LOTS_OF_BITS;
        }
    }

    fn normalize(&mut self, dif: u32, rng: u32) {
        let d = rng.leading_zeros() as i16 - 16; // rng <= 0xFFFF -> 0..=16
        self.cnt -= d;
        self.dif = ((dif + 1) << (d as u32)) - 1;
        self.rng = (rng << (d as u32)) as u16;
        if self.cnt < 0 {
            self.refill();
        }
    }

    #[allow(unused)]
    pub(crate) fn rng_dbg(&self) -> u16 {
        self.rng
    }
    pub(crate) fn decode_bool(&mut self, f: u16) -> bool {
        let r = self.rng as u32;
        let v = (((r >> 8) * (f as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
        let vw = v << ((WINDOW_SIZE - 16) as u32);
        let (dif, rng, ret) = if self.dif >= vw {
            (self.dif - vw, r - v, false)
        } else {
            (self.dif, v, true)
        };
        self.normalize(dif, rng);
        ret
    }

    pub(crate) fn decode_literal(&mut self, bits: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..bits {
            v = (v << 1) | self.decode_bool(16384) as u32;
        }
        v
    }

    /// `ns(n)` decode (spec 4.10.7).
    pub(crate) fn decode_ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = (32 - (n - 1).leading_zeros()).max(1);
        let m = (1u32 << w) - n;
        let v = self.decode_literal(w - 1);
        if v < m {
            v
        } else {
            let extra = self.decode_bool(16384) as u32;
            (v << 1) - m + extra
        }
    }

    /// `decode_subexp(numSyms, k)` (spec 4.10.6).
    pub(crate) fn decode_subexp(&mut self, num_syms: u32, k: u32) -> u32 {
        let mut i = 0u32;
        let mut mk = 0u32;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                return self.decode_ns(num_syms - mk) + mk;
            } else if self.decode_bool(16384) {
                i += 1;
                mk += a;
            } else {
                return self.decode_literal(b2) + mk;
            }
        }
    }

    pub(crate) fn decode_unsigned_subexp_with_ref(&mut self, mx: u32, k: u32, r: u32) -> u32 {
        let v = self.decode_subexp(mx, k);
        if (r << 1) <= mx {
            inverse_recenter(r, v)
        } else {
            mx - 1 - inverse_recenter(mx - 1 - r, v)
        }
    }

    pub(crate) fn decode_signed_subexp_with_ref(
        &mut self,
        low: i32,
        high: i32,
        k: u32,
        r: i32,
    ) -> i32 {
        let x = self.decode_unsigned_subexp_with_ref((high - low) as u32, k, (r - low) as u32);
        x as i32 + low
    }

    pub(crate) fn decode_symbol_noupdate(&mut self, cdf: &[u16]) -> usize {
        let r = self.rng as u32;
        let n = cdf.len() as u32 - 1;
        let c = self.dif >> ((WINDOW_SIZE - 16) as u32);
        let mut ret = 0usize;
        let mut u = r;
        let mut v = ((r >> 8) * (cdf[0] as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB * n;
        while c < v {
            u = v;
            ret += 1;
            v = ((r >> 8) * (cdf[ret] as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            v += EC_MIN_PROB * (n - ret as u32);
        }
        let new_dif = self.dif - (v << ((WINDOW_SIZE - 16) as u32));
        self.normalize(new_dif, u - v);
        ret
    }

    pub(crate) fn decode_symbol(&mut self, cdf: &mut [u16]) -> usize {
        let s = self.decode_symbol_noupdate(cdf);
        update_cdf(cdf, s);
        s
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
    fn bool_roundtrip_reference_case() {
        // Mirrors rav1e's own booleans() test.
        let mut w = OdEcEncoder::new();
        for &(v, f) in &[
            (false, 1u16),
            (true, 2),
            (false, 3),
            (true, 1),
            (true, 2),
            (false, 3),
        ] {
            w.encode_bool(v, f);
        }
        let b = w.done();
        let mut r = OdEcDecoder::new(&b);
        for &(v, f) in &[
            (false, 1u16),
            (true, 2),
            (false, 3),
            (true, 1),
            (true, 2),
            (false, 3),
        ] {
            assert_eq!(r.decode_bool(f), v);
        }
    }

    #[test]
    fn subexp_with_ref_roundtrip() {
        // Wiener tap ranges: (min, max, mid, k) per coded tap, both axes.
        let ranges = [
            (-5i32, 11i32, 3i32, 1u32),
            (-23, 9, -7, 2),
            (-17, 47, 15, 3),
        ];
        let mut rng = Rng(123);
        let mut enc = OdEcEncoder::new();
        let mut cases = Vec::new();
        for _ in 0..20_000 {
            let idx = (rng.next() % 3) as usize;
            let (lo, hi, mid, k) = ranges[idx];
            let span = (hi - lo) as u64;
            let v = lo + (rng.next() % span) as i32;
            let r = mid;
            enc.encode_signed_subexp_with_ref(v, lo, hi, k, r);
            cases.push((v, lo, hi, k, r));
        }
        let bytes = enc.done();
        let mut dec = OdEcDecoder::new(&bytes);
        for (v, lo, hi, k, r) in cases {
            assert_eq!(dec.decode_signed_subexp_with_ref(lo, hi, k, r), v);
        }
    }

    #[test]
    fn literal_roundtrip() {
        let mut rng = Rng(7);
        let mut enc = OdEcEncoder::new();
        let mut vals = Vec::new();
        for _ in 0..20_000 {
            let bits = 1 + (rng.next() % 16) as u32;
            let v = (rng.next() as u32) & ((1u32 << bits) - 1);
            vals.push((v, bits));
            enc.encode_literal(v, bits);
        }
        let bytes = enc.done();
        let mut dec = OdEcDecoder::new(&bytes);
        for (v, bits) in vals {
            assert_eq!(dec.decode_literal(bits), v);
        }
    }

    #[test]
    fn fixed_cdf_symbol_roundtrip() {
        let cdf = uniform_icdf(5);
        let mut rng = Rng(99);
        let syms: Vec<usize> = (0..20_000).map(|_| (rng.next() % 5) as usize).collect();
        let mut enc = OdEcEncoder::new();
        for &s in &syms {
            enc.encode_symbol_noupdate(s, &cdf);
        }
        let bytes = enc.done();
        let mut dec = OdEcDecoder::new(&bytes);
        for &s in &syms {
            assert_eq!(dec.decode_symbol_noupdate(&cdf), s);
        }
    }

    #[test]
    fn adaptive_symbol_roundtrip() {
        // Skewed source so adaptation moves the CDF; enc/dec must stay in sync.
        let mut rng = Rng(0xBADC0DE);
        let n = 8;
        let syms: Vec<usize> = (0..60_000)
            .map(|_| {
                let r = rng.next() % 100;
                if r < 60 {
                    0
                } else if r < 80 {
                    1
                } else {
                    (2 + r % (n as u64 - 2)) as usize
                }
            })
            .collect();

        let mut enc = OdEcEncoder::new();
        let mut cdf_e = uniform_icdf(n);
        for &s in &syms {
            enc.encode_symbol(s, &mut cdf_e);
        }
        let bytes = enc.done();

        let mut dec = OdEcDecoder::new(&bytes);
        let mut cdf_d = uniform_icdf(n);
        for &s in &syms {
            assert_eq!(dec.decode_symbol(&mut cdf_d), s);
        }
        // Both sides must have evolved the CDF identically.
        assert_eq!(cdf_e, cdf_d);
    }

    #[test]
    fn mixed_stream_roundtrip() {
        let mut enc = OdEcEncoder::new();
        let mut cdf_e = uniform_icdf(4);
        enc.encode_symbol(2, &mut cdf_e);
        enc.encode_bool(true, 10000);
        enc.encode_literal(0b1101, 4);
        enc.encode_symbol(0, &mut cdf_e);
        enc.encode_literal(0x2A, 6);
        let bytes = enc.done();

        let mut dec = OdEcDecoder::new(&bytes);
        let mut cdf_d = uniform_icdf(4);
        assert_eq!(dec.decode_symbol(&mut cdf_d), 2);
        assert!(dec.decode_bool(10000));
        assert_eq!(dec.decode_literal(4), 0b1101);
        assert_eq!(dec.decode_symbol(&mut cdf_d), 0);
        assert_eq!(dec.decode_literal(6), 0x2A);
    }
}
