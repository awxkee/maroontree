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

/// YUV chroma sampling layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Layout {
    /// 4:0:0 — luma only.
    Monochrome,
    /// 4:2:0 — chroma at half width and half height (32x32 transform).
    I420,
    /// 4:2:2 — chroma at half width, full height (32x64 transform).
    I422,
    /// 4:4:4 — chroma at full resolution (64x64 transform).
    I444,
}

impl Layout {
    /// Whether this layout carries chroma planes.
    pub(crate) fn has_chroma(self) -> bool {
        self != Layout::Monochrome
    }

    /// Sequence-header layout code (the uvlc index into the decoder's layout table).
    pub(crate) fn header_uvlc(self) -> u32 {
        match self {
            Layout::I420 => 0,
            Layout::Monochrome => 1,
            Layout::I444 => 2,
            Layout::I422 => 3,
        }
    }

    /// AVM `seq_profile_idc` for this chroma format (5-bit field). AVM does not use
    /// AV1's 0/1/2 scheme: MAIN_420_10_IP0=0 covers 4:0:0/4:2:0, MAIN_422_10_IP1=3
    /// covers 4:2:2, MAIN_444_10_IP1=4 covers 4:4:4. avmdec enforces this
    /// profile↔format consistency in av2_check_profile_interop_conformance.
    pub(crate) fn profile(self) -> u32 {
        match self {
            Layout::Monochrome | Layout::I420 => 0, // MAIN_420_10_IP0
            Layout::I422 => 3,                      // MAIN_422_10_IP1
            Layout::I444 => 4,                      // MAIN_444_10_IP1
        }
    }
}
