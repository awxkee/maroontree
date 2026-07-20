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

use crate::dct::{FORWARD_QM_SCALES, apply_qmatrix};
use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn apply_qmatrix_avx2(levels: &mut [i32], targets: &mut [f32], inverse_weights: &[u8]) {
    assert_eq!(levels.len(), targets.len());
    assert_eq!(levels.len(), inverse_weights.len());

    let (level_chunks, level_tail) = levels.as_chunks_mut::<8>();
    let (target_chunks, target_tail) = targets.as_chunks_mut::<8>();
    let (weight_chunks, weight_tail) = inverse_weights.as_chunks::<8>();
    for ((level, target), weights) in level_chunks
        .iter_mut()
        .zip(target_chunks)
        .zip(weight_chunks)
    {
        unsafe {
            let target_v = _mm256_loadu_ps(target.as_ptr());
            let iwt8 = _mm_loadl_epi64(weights.as_ptr().cast::<__m128i>());
            let indices = _mm256_cvtepu8_epi32(iwt8);
            let scales = _mm256_i32gather_ps::<4>(FORWARD_QM_SCALES.as_ptr(), indices);
            let weighted = _mm256_mul_ps(target_v, scales);
            _mm256_storeu_ps(target.as_mut_ptr(), weighted);

            let rounded = _mm256_cvttps_epi32(_mm256_round_ps::<0x00>(weighted));
            _mm256_storeu_si256(level.as_mut_ptr().cast::<__m256i>(), rounded);
        }
    }

    apply_qmatrix(level_tail, target_tail, weight_tail);
}
