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

type EcWin = u32;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

pub(crate) struct Writer {
    rng: u16,
    cnt: i16,
    low: EcWin,
    precarry: Vec<u16>,
    updating_cdf: bool,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub(crate) fn new() -> Self {
        Writer {
            rng: 0x8000,
            cnt: -9,
            low: 0,
            precarry: Vec::new(),
            updating_cdf: true,
        }
    }

    pub(crate) fn with_updating_cdf(mut self, updating_cdf: bool) -> Self {
        self.updating_cdf = updating_cdf;
        self
    }

    #[inline]
    fn lr_compute(&self, fl: u16, fh: u16, nms: u16) -> (EcWin, u16) {
        let r = self.rng as u32;
        let mut u = (((r >> 8) * (fl as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB * nms as u32;
        if fl >= 32768 {
            u = r;
        }
        let v = (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB * (nms - 1) as u32;
        (r - u, (u - v) as u16)
    }

    #[inline]
    fn store(&mut self, fl: u16, fh: u16, nms: u16) {
        let (l, r) = self.lr_compute(fl, fh, nms);
        let mut low = l + self.low;
        let mut c = self.cnt;
        let d = r.leading_zeros() as i16; // r is u16 -> leading_zeros in 0..=15
        let mut s = c + d;
        if s >= 0 {
            c += 16;
            let mut m: EcWin = (1 << c) - 1;
            if s >= 8 {
                self.precarry.push((low >> c) as u16);
                low &= m;
                c -= 8;
                m >>= 8;
            }
            self.precarry.push((low >> c) as u16);
            s = c + d - 24;
            low &= m;
        }
        self.low = low << d;
        self.rng = r << d;
        self.cnt = s;
    }

    /// Encode symbol `s` against an inverse-cdf of length `n` (n = #symbols).
    pub(crate) fn symbol(&mut self, s: u32, cdf: &[u16]) {
        let s = s as usize;
        let n = cdf.len();
        let nms = n - s;
        let fl = if s > 0 { cdf[s - 1] } else { 32768 };
        let fh = cdf[s];
        debug_assert!(
            (fh >> EC_PROB_SHIFT) <= (fl >> EC_PROB_SHIFT),
            "cdf not decreasing"
        );
        self.store(fl, fh, nms as u16);
    }

    /// Encode a symbol and update its persistent AV1 inverse CDF.
    #[inline]
    pub(crate) fn symbol_adapt(&mut self, s: usize, cdf: &mut [u16]) {
        self.symbol(s as u32, cdf);
        if self.updating_cdf {
            crate::odec::update_cdf(cdf, s);
        }
    }

    /// Encode a single bool with probability-of-true `f` (Q15).
    pub(crate) fn bool(&mut self, val: bool, f: u16) {
        // symbol() over inverse-cdf [f, 0]: symbol 0 = false occupies [f,32768)?
        // rav1e: self.symbol(val as u32, &[f, 0]). cdf=[f,0], n=2.
        self.symbol(val as u32, &[f, 0]);
    }

    /// Equiprobable bit.
    pub(crate) fn bit(&mut self, b: u16) {
        self.bool(b == 1, 16384);
    }

    /// Raw literal, MSB first, equiprobable.
    pub(crate) fn literal(&mut self, bits: u8, v: u32) {
        for i in (0..bits).rev() {
            self.bit(((v >> i) & 1) as u16);
        }
    }

    pub(crate) fn write_golomb(&mut self, level: u32) {
        let x = level + 1;
        let length = 32 - x.leading_zeros();
        for _ in 0..length - 1 {
            self.bit(0);
        }
        for i in (0..length).rev() {
            self.bit(((x >> i) & 1) as u16);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        let l = self.low;
        let mut c = self.cnt;
        let mut s = 10i16;
        let m: EcWin = 0x3FFF;
        let mut e: EcWin = ((l + m) & !m) | (m + 1);
        s += c;
        if s > 0 {
            let mut n: EcWin = (1 << (c + 16)) - 1;
            loop {
                self.precarry.push((e >> (c + 16)) as u16);
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        let mut carry = 0u16;
        let mut offs = self.precarry.len();
        let mut out = vec![0u8; offs];
        while offs > 0 {
            offs -= 1;
            carry += self.precarry[offs];
            out[offs] = carry as u8;
            carry >>= 8;
        }
        out
    }
}
