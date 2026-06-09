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

#![allow(unused)]
use crate::bitwriter::read_leb128;
use crate::coeff::{CoeffCdfs, decode_block};
use crate::obu::FRAME_HEADER_LEN;
use crate::pixel::Pixel;
use crate::predict::left_predictor;
use crate::rangecoder::RangeDecoder;
use crate::transform::iwht4x4;
use crate::{BitDepth, PlanarImage};

const OBU_FRAME: u8 = 6;

/// Walk the OBU stream and return the OBU_FRAME payload, if present.
fn find_frame_payload(bytes: &[u8]) -> Option<&[u8]> {
    let mut pos = 0;
    while pos < bytes.len() {
        let header = bytes[pos];
        let obu_type = (header >> 3) & 0x0f;
        let has_size = (header >> 1) & 0x01;
        let ext = (header >> 2) & 0x01;
        pos += 1;
        if ext == 1 {
            pos += 1;
        }
        if has_size == 0 {
            return None;
        }
        let (size, used) = read_leb128(&bytes[pos..]);
        pos += used;
        let payload = &bytes[pos..pos + size as usize];
        if obu_type == OBU_FRAME {
            return Some(payload);
        }
        pos += size as usize;
    }
    None
}
