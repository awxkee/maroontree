/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

//! IVF container writer for AV2. Byte layout matches AVM `common/ivfenc.c`.

/// AV2 IVF fourcc ("AV02" little-endian). From AVM `tools_common.h`.
const AV2_FOURCC: u32 = 0x3230_5641;

/// Streaming IVF muxer: 32-byte file header (frame count patched on finish),
/// then a 12-byte header + payload per frame.
pub struct IvfWriter {
    buf: Vec<u8>,
    frames: u32,
    /// Byte offset of the 4-byte frame-count field, patched in `finish`.
    count_off: usize,
}

impl IvfWriter {
    pub fn new(width: u16, height: u16, fps_num: u32, fps_den: u32) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(b"DKIF");
        buf.extend_from_slice(&0u16.to_le_bytes()); // version
        buf.extend_from_slice(&32u16.to_le_bytes()); // header size
        buf.extend_from_slice(&AV2_FOURCC.to_le_bytes());
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&fps_num.to_le_bytes()); // rate = timebase.den
        buf.extend_from_slice(&fps_den.to_le_bytes()); // scale = timebase.num
        let count_off = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes()); // frame count (patched)
        buf.extend_from_slice(&0u32.to_le_bytes()); // unused
        Self {
            buf,
            frames: 0,
            count_off,
        }
    }

    /// Append one coded frame with presentation timestamp `pts`.
    pub fn write_frame(&mut self, data: &[u8], pts: u64) {
        self.buf
            .extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.buf
            .extend_from_slice(&((pts & 0xFFFF_FFFF) as u32).to_le_bytes());
        self.buf
            .extend_from_slice(&((pts >> 32) as u32).to_le_bytes());
        self.buf.extend_from_slice(data);
        self.frames += 1;
    }

    /// Patch the frame count and return the complete IVF byte stream.
    pub fn finish(mut self) -> Vec<u8> {
        self.buf[self.count_off..self.count_off + 4].copy_from_slice(&self.frames.to_le_bytes());
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_32_bytes_and_fourcc_av02() {
        let w = IvfWriter::new(320, 240, 30, 1);
        let out = w.finish();
        assert_eq!(&out[0..4], b"DKIF");
        assert_eq!(u16::from_le_bytes([out[6], out[7]]), 32);
        assert_eq!(&out[8..12], b"AV02");
        assert_eq!(u16::from_le_bytes([out[12], out[13]]), 320);
        assert_eq!(u32::from_le_bytes([out[24], out[25], out[26], out[27]]), 0);
    }

    #[test]
    fn frame_count_patched() {
        let mut w = IvfWriter::new(16, 16, 30, 1);
        w.write_frame(&[1, 2, 3], 0);
        w.write_frame(&[4, 5], 1);
        let out = w.finish();
        assert_eq!(u32::from_le_bytes([out[24], out[25], out[26], out[27]]), 2);
        // first frame header: size=3 at offset 32
        assert_eq!(u32::from_le_bytes([out[32], out[33], out[34], out[35]]), 3);
    }
}
