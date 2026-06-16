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
pub(crate) static MIN_PROB: [[u16; 8]; 8] = [
    [63, 65535, 65535, 65535, 65535, 65535, 65535, 65535],
    [47, 87, 65535, 65535, 65535, 65535, 65535, 65535],
    [31, 63, 95, 65535, 65535, 65535, 65535, 65535],
    [31, 55, 79, 103, 65535, 65535, 65535, 65535],
    [23, 47, 63, 87, 111, 65535, 65535, 65535],
    [23, 39, 55, 79, 95, 111, 65535, 65535],
    [15, 31, 47, 63, 79, 95, 111, 65535],
    // nsyms = 8 (CfL joint-sign / alpha-magnitude). The avm decoder scales with
    // av2_prob_inc_tbl[nsym-2] = row 6 for nsyms=8. Matching boundary(k) gives
    // MIN_PROB[7][k] = 127 - 8*av2_prob_inc_tbl[6][k] (the low-7-bit |127 vs >>7<<4
    // terms cancel exactly), inc_tbl[6] = {14,12,10,8,6,4,2,0} ->
    // {15,31,47,63,79,95,111, sentinel}. Verified bit-exact vs avm od_ec_prob_scale.
    [15, 31, 47, 63, 79, 95, 111, 65535],
];

/// MSB-first bit packer for the uncompressed header sections.
pub(crate) struct ByteWriter {
    bytes: Vec<u8>,
    accumulator: u64,
    pending_bits: u32,
}

impl ByteWriter {
    pub(crate) fn new() -> Self {
        ByteWriter {
            bytes: vec![],
            accumulator: 0,
            pending_bits: 0,
        }
    }

    /// Append a single bit (only bit 0 of `bit` is used).
    pub(crate) fn write_bit(&mut self, bit: u32) {
        self.accumulator = (self.accumulator << 1) | (bit as u64 & 1);
        self.pending_bits += 1;
        if self.pending_bits == 8 {
            self.bytes.push(self.accumulator as u8);
            self.accumulator = 0;
            self.pending_bits = 0;
        }
    }

    /// Append the low `count` bits of `value`, most-significant bit first.
    pub(crate) fn write_bits(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Append an unsigned variable-length code (`uvlc`).
    pub(crate) fn write_uvlc(&mut self, value: u32) {
        let shifted = value + 1;
        let leading_zeros = 31 - shifted.leading_zeros();
        for _ in 0..leading_zeros {
            self.write_bit(0);
        }
        self.write_bits(shifted, leading_zeros + 1);
    }

    /// Append `value` coded as a non-uniform `ns(max)` element (uniform code).
    pub(crate) fn write_uniform(&mut self, value: u32, max: u32) {
        let bits = (31 - max.leading_zeros()) + 1;
        let threshold = (1u32 << bits) - max;
        if value < threshold {
            self.write_bits(value, bits - 1);
        } else {
            let widened = value + threshold;
            self.write_bits(widened >> 1, bits - 1);
            self.write_bit(widened & 1);
        }
    }

    /// Write a trailing `1` bit and pad with zeros to the next byte boundary.
    pub(crate) fn align_with_one(&mut self) {
        self.write_bit(1);
        while self.pending_bits != 0 {
            self.write_bit(0);
        }
    }

    /// Pad with zeros to the next byte boundary.
    pub(crate) fn align_with_zero(&mut self) {
        while self.pending_bits != 0 {
            self.write_bit(0);
        }
    }

    /// Consume the writer and return the packed bytes.
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// AV2 multi-symbol arithmetic coder (encoder side).
pub(crate) struct RangeEncoder {
    low: u64,
    /// Current range; exposed for encoder/decoder trace alignment during testing.
    pub(crate) range: u32,
    count: i32,
    output: Vec<u16>,
    /// Coefficient-CDF q-context = get_q_ctx(base_q_idx). Selects the default
    /// CDF band avmdec loads (0:q<=90, 1:91..140, 2:141..190, 3:>=191).
    /// Defaults to 1 so legacy q120 paths are unchanged.
    pub(crate) qc: usize,
    /// Emit CfL (chroma-from-luma) signalling for chroma-ref blocks. Set per encode
    /// from the tuning flag; false keeps the bitstream byte-identical.
    pub(crate) cfl: bool,
    /// Per-block CfL state, set just before the block's mode encode. `cfl_ctx` is the
    /// is_cfl neighbour context (0..2). `cfl_use` selects CfL (uv_mode = UV_CFL_PRED);
    /// when true, `cfl_js`/`cfl_mag_u`/`cfl_mag_v` + `cfl_ctx_u`/`cfl_ctx_v` carry the
    /// resolved joint-sign, per-plane magnitude indices and alpha-cdf contexts.
    pub(crate) cfl_ctx: usize,
    pub(crate) cfl_use: bool,
    pub(crate) cfl_js: u8,
    pub(crate) cfl_mag_u: u8,
    pub(crate) cfl_mag_v: u8,
    pub(crate) cfl_ctx_u: usize,
    pub(crate) cfl_ctx_v: usize,
}

impl RangeEncoder {
    pub(crate) fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0x8000,
            count: -9,
            output: vec![],
            qc: 1,
            cfl: false,
            cfl_ctx: 0,
            cfl_use: false,
            cfl_js: 0,
            cfl_mag_u: 0,
            cfl_mag_v: 0,
            cfl_ctx_u: 0,
            cfl_ctx_v: 0,
        }
    }

    fn normalize(&mut self, mut low: u64, range: u32) {
        let base_count = self.count;
        let shift = 16 - (32 - range.leading_zeros()) as i32;
        let mut remaining = base_count + shift;
        if remaining >= 0 {
            let mut bit_pos = base_count + 16;
            let mut mask: u64 = (1u64 << bit_pos) - 1;
            if remaining >= 8 {
                self.output.push((low >> bit_pos) as u16);
                low &= mask;
                bit_pos -= 8;
                mask >>= 8;
            }
            self.output.push((low >> bit_pos) as u16);
            remaining = bit_pos + shift - 24;
            low &= mask;
        }
        self.low = low << shift;
        self.range = range << shift;
        self.count = remaining;
    }

    /// Compute the cumulative-frequency boundary for symbol index `k`.
    fn boundary(scaled_range: u32, icdf: &[u16], min_prob: &[u16; 8], k: usize) -> u32 {
        ((scaled_range * (icdf[k] as u32 | 127).saturating_sub(min_prob[k] as u32)) >> 10) << 3
    }

    /// Encode symbol `s` against an inverse-CDF table covering `nsyms` symbols.
    pub(crate) fn encode_symbol(&mut self, icdf: &[u16], s: usize, nsyms: usize) {
        let range = self.range;
        let scaled_range = range >> 8;
        let min_prob = &MIN_PROB[nsyms - 1];
        let upper = if s == 0 {
            range
        } else {
            Self::boundary(scaled_range, icdf, min_prob, s - 1)
        };
        let lower = Self::boundary(scaled_range, icdf, min_prob, s);
        let low = self.low + (range - upper) as u64;
        self.normalize(low, upper - lower);
    }

    /// Encode an escape (last) symbol against `cdf` extended with a trailing 0,
    /// using a stack buffer instead of allocating (equivalent to with_escape()).
    pub(crate) fn encode_symbol_esc(&mut self, cdf: &[u16], s: usize, nsyms: usize) {
        let mut buf = [0u16; 16];
        let n = cdf.len();
        buf[..n].copy_from_slice(cdf);
        self.encode_symbol(&buf[..n + 1], s, nsyms);
    }

    /// Encode a single adaptive boolean with CDF `cdf` (probability of `0`).
    pub(crate) fn encode_bool(&mut self, cdf: u32, bit: u32) {
        let range = self.range;
        let split = (((range >> 8) * (((cdf >> 7) << 4) + 8)) >> 7) << 3;
        let (low, range) = if bit != 0 {
            (self.low + (range - split) as u64, split)
        } else {
            (self.low, range - split)
        };
        self.normalize(low, range);
    }

    fn normalize_bypass(&mut self, mut low: u64, range: u32, bypass_bits: i32) {
        let base_count = self.count + bypass_bits;
        let mut remaining = base_count;
        if remaining >= 0 {
            let mut bit_pos = base_count + 16;
            let mut mask: u64 = (1u64 << bit_pos) - 1;
            if remaining >= 8 {
                self.output.push((low >> bit_pos) as u16);
                low &= mask;
                bit_pos -= 8;
                mask >>= 8;
            }
            self.output.push((low >> bit_pos) as u16);
            remaining = bit_pos - 24;
            low &= mask;
        }
        self.low = low;
        self.range = range;
        self.count = remaining;
    }

    /// Encode `bit_count` equiprobable bits taken from `value`, MSB-first.
    pub(crate) fn encode_bypass(&mut self, value: u32, bit_count: u32) {
        let range = self.range;
        let low = (self.low << bit_count) + (range as u64) * (value as u64);
        self.normalize_bypass(low, range, bit_count as i32);
    }

    /// Flush the coder and return the packed tile bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let low = self.low;
        let mut count = self.count;
        let mut remaining = 10 + count;
        let mask: u64 = 0x3FFF;
        let mut end = ((low + mask) & !mask) | (mask + 1);
        if remaining > 0 {
            let mut byte_mask: u64 = (1u64 << (count + 16)) - 1;
            loop {
                self.output.push((end >> (count + 16)) as u16);
                end &= byte_mask;
                remaining -= 8;
                count -= 8;
                byte_mask >>= 8;
                if remaining <= 0 {
                    break;
                }
            }
        }
        let len = self.output.len();
        let mut bytes = vec![0u8; len];
        let mut carry: u32 = 0;
        let mut i = len;
        while i > 0 {
            i -= 1;
            let x = self.output[i] as u32 + carry;
            bytes[i] = (x & 0xff) as u8;
            carry = x >> 8;
        }
        bytes
    }
}
