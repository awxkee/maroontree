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

#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8, // bits filled in `cur`, 0..8
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Write `n` bits of `value`, most-significant bit first (AV1 `f(n)`).
    pub(crate) fn f(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Single flag bit.
    pub(crate) fn flag(&mut self, b: bool) {
        self.f(b as u32, 1);
    }

    /// Byte-align by padding with zero bits.
    pub(crate) fn byte_align(&mut self) {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// AV1 `trailing_bits()`: a single `1` bit then zero-pad to byte alignment.
    /// Terminates an OBU payload.
    pub(crate) fn trailing_bits(&mut self) {
        self.f(1, 1);
        self.byte_align();
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.byte_align();
        self.bytes
    }
}

/// Encode an unsigned LEB128 value (AV1 `leb128()`), little-endian, 7 bits/byte.
pub(crate) fn leb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
    out
}

/// Decode a LEB128 value from the front of `buf`.
/// Returns `(value, bytes_consumed)`.
pub(crate) fn read_leb128(buf: &[u8]) -> (u64, usize) {
    let mut value = 0u64;
    let mut i = 0;
    loop {
        let byte = buf[i];
        value |= ((byte & 0x7f) as u64) << (7 * i);
        i += 1;
        if byte & 0x80 == 0 {
            break;
        }
    }
    (value, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_pack_msb_first() {
        let mut w = BitWriter::new();
        w.f(0b101, 3);
        w.f(0b01, 2);
        w.f(0b111, 3);
        // 101 01 111 = 0b10101111
        assert_eq!(w.into_bytes(), vec![0b1010_1111]);
    }

    #[test]
    fn byte_align_pads_zero() {
        let mut w = BitWriter::new();
        w.f(0b1, 1);
        assert_eq!(w.into_bytes(), vec![0b1000_0000]);
    }

    #[test]
    fn leb128_known_values() {
        assert_eq!(leb128(0), vec![0x00]);
        assert_eq!(leb128(127), vec![0x7f]);
        assert_eq!(leb128(128), vec![0x80, 0x01]);
        assert_eq!(leb128(300), vec![0xac, 0x02]);
    }

    #[test]
    fn leb128_read_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 16384, 1_000_000, 1 << 40] {
            let enc = leb128(v);
            let (got, used) = read_leb128(&enc);
            assert_eq!(got, v);
            assert_eq!(used, enc.len());
        }
    }
}
