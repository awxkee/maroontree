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

use crate::rangecoder::{Cdf, RangeEncoder};

/// Max coefficient bit-length we model. 12-bit input through the WHT (scaled by
/// 4) stays well under 2^24, so 24 is safe; encode asserts this.
pub(crate) const MAX_LEN: usize = 24;

/// Adaptive models reused across an entire image (one shared instance).
pub(crate) struct CoeffCdfs {
    pub(crate) nz: Cdf,
    pub(crate) len: Cdf,
}

impl Default for CoeffCdfs {
    fn default() -> Self {
        CoeffCdfs {
            nz: Cdf::uniform(2),
            len: Cdf::uniform(MAX_LEN),
        }
    }
}

#[allow(unused)]
pub(crate) fn encode_block(enc: &mut RangeEncoder, coeffs: &[i32; 16], cdfs: &mut CoeffCdfs) {
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
