/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! Rate-distortion cost `J = D + λ·R`: `D` is pixel-domain SSE, `R` is bits.
//!
//! The encoder's `qstep` is AVM's QTX dequant step, which is the pixel-domain
//! quantizer step scaled by 64 (QTX ×8, transform gain ×8). Every λ here is in
//! pixel SSE per bit, so all decisions price a bit on the same scale.
//!
//! Calibration: the encoder's operational slope `-ΔSSE/ΔR` between adjacent
//! qualities is ≈2.5e-5·qstep² on kodak01 and ≈1.7e-5·qstep² on s_mascot,
//! flat across qidx 38..166, i.e. the expected λ ∝ Δ² law.

/// `(QTX step / pixel step)²`.
pub(crate) const QTX_PIXEL_SCALE_SQ: f32 = 4096.0;

/// λ (pixel SSE per bit) from an RDOQ weight in level² per bit.
#[inline]
pub(crate) fn lambda_from_level(level_lambda: f32, qstep: i32) -> f32 {
    level_lambda * (qstep as f32) * (qstep as f32) / QTX_PIXEL_SCALE_SQ
}

#[inline]
pub(crate) fn rd_cost(dist: f32, bits: f32, lambda: f32) -> f32 {
    dist + lambda * bits
}

/// TEMPORARY experiment knob (`MT_RD_<NAME>` env var, default 1.0); removed once
/// the RD study settles each scale.
pub(crate) fn knob(name: &str) -> f32 {
    knob_or(name, 1.0)
}

/// TEMPORARY, like [`knob`] with an explicit default.
pub(crate) fn knob_or(name: &str, default: f32) -> f32 {
    thread_local! {
        static CACHE: std::cell::RefCell<std::collections::HashMap<(usize, usize, u32), f32>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    CACHE.with(|c| {
        *c.borrow_mut()
            .entry((name.as_ptr() as usize, name.len(), default.to_bits()))
            .or_insert_with(|| {
                std::env::var(format!("MT_RD_{name}"))
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(default)
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_lambda_converts_qtx_to_pixel_domain() {
        // qidx 120 (qstep 1248): 0.07 level²/bit -> ~26.6 pixel SSE/bit.
        assert!((lambda_from_level(0.07, 1248) - 26.6).abs() < 0.1);
        assert_eq!(rd_cost(10.0, 2.0, 3.0), 16.0);
    }
}
