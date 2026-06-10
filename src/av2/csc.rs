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

use crate::err::EncodeError;

// Q0.13 coefficients (value = round(f * 8192))
pub(super) const Q: i32 = 13;
pub(super) const HALF: i32 = 1 << (Q - 1); // 0.5 rounding bias

pub(super) const Y_R: i32 = 2449; // round( 0.299    * 8192)
pub(super) const Y_G: i32 = 4809; // round( 0.587    * 8192)
pub(super) const Y_B: i32 = 934; // round( 0.114    * 8192)

pub(super) const CB_R: i32 = -1382; // round(-0.168736 * 8192)
pub(super) const CB_G: i32 = -2714; // round(-0.331264 * 8192)
pub(super) const CB_B: i32 = 4096; // round( 0.5 * 8192)

pub(super) const CR_R: i32 = 4096; // round( 0.5 * 8192)
pub(super) const CR_G: i32 = -3430; // round(-0.418688 * 8192)
pub(super) const CR_B: i32 = -666; // round(-0.081312 * 8192)

const MAX_DIM: u32 = 65535;
const MIN_DIM: u32 = 1;

pub fn get_q_ctx(q: u8) -> usize {
    if q <= 90 {
        0
    } else if q <= 140 {
        1
    } else if q <= 190 {
        2
    } else {
        3
    }
}

pub(super) fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM || width > MAX_DIM || height > MAX_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}
