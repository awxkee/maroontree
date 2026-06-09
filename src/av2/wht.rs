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

/// Inverse of avm's decoder butterfly. Given the four outputs the decoder would produce
/// `(o0,o1,o2,o3)`, returns the four inputs it consumed `(in0,in1,in2,in3)`.
#[inline]
fn inv_bfly(o0: i32, o1: i32, o2: i32, o3: i32) -> (i32, i32, i32, i32) {
    let e = (o0 + o1 + o2 - o3) >> 1;
    (o0 + o1 + o2 - e, e - o2, o3 - o2 + e - o1, e - o1)
}

/// Forward 4×4 WHT: residual (row-major, 16 samples) → coded coefficient levels
/// (row-major). The result is `butterfly²(resid)`; the decoder reconstructs the exact
/// residual via `iwht4x4(level * WHT_DEQUANT)`.
pub(crate) fn fwht4x4(resid: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, d) = inv_bfly(resid[i], resid[4 + i], resid[8 + i], resid[12 + i]);
        tmp[i] = a;
        tmp[4 + i] = b;
        tmp[8 + i] = c;
        tmp[12 + i] = d;
    }
    // inverse of the decoder's row pass
    let mut out = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, d) = inv_bfly(tmp[i * 4], tmp[i * 4 + 1], tmp[i * 4 + 2], tmp[i * 4 + 3]);
        out[i * 4] = a;
        out[i * 4 + 1] = b;
        out[i * 4 + 2] = c;
        out[i * 4 + 3] = d;
    }
    out
}
