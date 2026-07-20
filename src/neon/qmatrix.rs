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
use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn apply_qmatrix_neon(levels: &mut [i32], targets: &mut [f32], inverse_weights: &[u8]) {
    assert_eq!(levels.len(), targets.len());
    assert_eq!(levels.len(), inverse_weights.len());

    let (level_chunks, level_tail) = levels.as_chunks_mut::<4>();
    let (target_chunks, target_tail) = targets.as_chunks_mut::<4>();
    let (weight_chunks, weight_tail) = inverse_weights.as_chunks::<4>();
    for ((level, target), weights) in level_chunks
        .iter_mut()
        .zip(target_chunks)
        .zip(weight_chunks)
    {
        let scales = [
            FORWARD_QM_SCALES[weights[0] as usize],
            FORWARD_QM_SCALES[weights[1] as usize],
            FORWARD_QM_SCALES[weights[2] as usize],
            FORWARD_QM_SCALES[weights[3] as usize],
        ];
        unsafe {
            let target_v = vld1q_f32(target.as_ptr());
            let weighted = vmulq_f32(target_v, vld1q_f32(scales.as_ptr()));
            vst1q_f32(target.as_mut_ptr(), weighted);
            vst1q_s32(level.as_mut_ptr(), vcvtaq_s32_f32(weighted));
        }
    }

    apply_qmatrix(level_tail, target_tail, weight_tail);
}
