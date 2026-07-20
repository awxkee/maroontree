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
use crate::quant::{Dct, Quant};

pub(crate) fn idct_dequant_32x16(levels: &[i32; 512], q: &Quant) -> [i32; 512] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 512];
    for rc in 0..512 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 32, 16);
        let mag = (((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) >> 1) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 512];
    for row in 0..16 {
        for col in 0..32 {
            tmp[row * 32 + col] = (coeff[row + col * 16] * 181 + 128) >> 8;
        }
    }
    for row in 0..16 {
        inv_dct32_1d(&mut tmp[row * 32..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for col in 0..32 {
        inv_dct16_1d(&mut tmp[col..], 32, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) type InvDctFn<const N: usize> = fn(&[i32; N], &Quant) -> [i32; N];

fn idct_dequant_8x8_scalar_quant(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    idct_dequant_8x8_scalar(levels, &IdctDequant::new(q, 8, 8))
}

fn idct_dequant_16x16_scalar_quant(levels: &[i32; 256], q: &Quant) -> [i32; 256] {
    idct_dequant_16x16_scalar(levels, &IdctDequant::new(q, 16, 16))
}

fn iadstdct_dequant_16x16_scalar_quant(levels: &[i32; 256], q: &Quant) -> [i32; 256] {
    iadstdct_dequant_16x16_scalar(levels, &IdctDequant::new(q, 16, 16))
}

fn idctadst_dequant_16x16_scalar_quant(levels: &[i32; 256], q: &Quant) -> [i32; 256] {
    idctadst_dequant_16x16_scalar(levels, &IdctDequant::new(q, 16, 16))
}

fn iadst_dequant_16x16_scalar_quant(levels: &[i32; 256], q: &Quant) -> [i32; 256] {
    iadst_dequant_16x16_scalar(levels, &IdctDequant::new(q, 16, 16))
}

fn idct_dequant_32x32_scalar_quant(levels: &[i32; 1024], q: &Quant) -> [i32; 1024] {
    idct_dequant_32x32_scalar(levels, &IdctDequant::new(q, 32, 32))
}

macro_rules! define_inverse_dispatch {
    ($((
        $field:ident, $n:literal, $w:literal, $h:literal, $_dq:literal, $flat:literal,
        $scalar:path, $neon_wrap:ident, $neon:path, $avx_wrap:ident, $avx:path
    )),* $(,)?) => {
        $(
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            fn $neon_wrap(levels: &[i32; $n], q: &Quant) -> [i32; $n] {
                let dequant = if $flat {
                    IdctDequant::new_flat(q)
                } else {
                    IdctDequant::new(q, $w, $h)
                };
                unsafe { $neon(levels, &dequant) }
            }

            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            fn $avx_wrap(levels: &[i32; $n], q: &Quant) -> [i32; $n] {
                let dequant = if $flat {
                    IdctDequant::new_flat(q)
                } else {
                    IdctDequant::new(q, $w, $h)
                };
                unsafe { $avx(levels, &dequant) }
            }
        )*

        #[derive(Clone, Copy)]
        pub(crate) struct IdctDispatch {
            $(pub(crate) $field: InvDctFn<$n>,)*
        }

        impl IdctDispatch {
            pub(crate) fn selected() -> Self {
                Self {
                    $($field: {
                        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                        {
                            $neon_wrap
                        }
                        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                        {
                            if std::is_x86_feature_detected!("avx2") {
                                $avx_wrap
                            } else {
                                $scalar
                            }
                        }
                        #[cfg(not(any(
                            all(target_arch = "aarch64", feature = "neon"),
                            all(target_arch = "x86_64", feature = "avx")
                        )))]
                        {
                            $scalar
                        }
                    },)*
                }
            }

            pub(crate) fn scalar() -> Self {
                Self { $($field: $scalar,)* }
            }

            $(
                #[inline]
                pub(crate) fn $field(
                    &self,
                    levels: &[i32; $n],
                    q: &Quant,
                ) -> [i32; $n] {
                    (self.$field)(levels, q)
                }
            )*
        }
    };
}

define_inverse_dispatch!(
    (
        idct_dequant_4x4,
        16,
        4,
        4,
        0,
        false,
        idct_dequant_4x4,
        idct4x4_neon_dispatch,
        crate::neon::idct_dequant_4x4_neon,
        idct4x4_avx2_dispatch,
        crate::avx::idct_dequant_4x4_avx2
    ),
    (
        idct_dequant_4x8,
        32,
        4,
        8,
        0,
        false,
        idct_dequant_4x8,
        idct4x8_neon_dispatch,
        crate::neon::idct_dequant_4x8_neon,
        idct4x8_avx2_dispatch,
        crate::avx::idct_dequant_4x8_avx2
    ),
    (
        idct_dequant_8x4,
        32,
        8,
        4,
        0,
        false,
        idct_dequant_8x4,
        idct8x4_neon_dispatch,
        crate::neon::idct_dequant_8x4_neon,
        idct8x4_avx2_dispatch,
        crate::avx::idct_dequant_8x4_avx2
    ),
    (
        idct_dequant_4x16,
        64,
        4,
        16,
        0,
        false,
        idct_dequant_4x16,
        idct4x16_neon_dispatch,
        crate::neon::idct_dequant_4x16_neon,
        idct4x16_avx2_dispatch,
        crate::avx::idct_dequant_4x16_avx2
    ),
    (
        idct_dequant_16x4,
        64,
        16,
        4,
        0,
        false,
        idct_dequant_16x4,
        idct16x4_neon_dispatch,
        crate::neon::idct_dequant_16x4_neon,
        idct16x4_avx2_dispatch,
        crate::avx::idct_dequant_16x4_avx2
    ),
    (
        idct_dequant_8x8,
        64,
        8,
        8,
        0,
        false,
        idct_dequant_8x8_scalar_quant,
        idct8x8_neon_dispatch2,
        crate::neon::idct_dequant_8x8_neon,
        idct8x8_avx2_dispatch2,
        crate::avx::idct_dequant_8x8_avx2
    ),
    (
        idct_dequant_8x16,
        128,
        8,
        16,
        0,
        false,
        idct_dequant_8x16,
        idct8x16_neon_dispatch,
        crate::neon::idct_dequant_8x16_neon,
        idct8x16_avx2_dispatch,
        crate::avx::idct_dequant_8x16_avx2
    ),
    (
        idct_dequant_16x8,
        128,
        16,
        8,
        0,
        false,
        idct_dequant_16x8,
        idct16x8_neon_dispatch,
        crate::neon::idct_dequant_16x8_neon,
        idct16x8_avx2_dispatch,
        crate::avx::idct_dequant_16x8_avx2
    ),
    (
        idct_dequant_16x16,
        256,
        16,
        16,
        0,
        false,
        idct_dequant_16x16_scalar_quant,
        idct16x16_neon_dispatch2,
        crate::neon::idct_dequant_16x16_neon,
        idct16x16_avx2_dispatch2,
        crate::avx::idct_dequant_16x16_avx2
    ),
    (
        idct_dequant_16x32,
        512,
        16,
        32,
        1,
        false,
        idct_dequant_16x32,
        idct16x32_neon_dispatch,
        crate::neon::idct_dequant_16x32_neon,
        idct16x32_avx2_dispatch,
        crate::avx::idct_dequant_16x32_avx2
    ),
    (
        idct_dequant_32x16,
        512,
        32,
        16,
        1,
        false,
        idct_dequant_32x16,
        idct32x16_neon_dispatch,
        crate::neon::idct_dequant_32x16_neon,
        idct32x16_avx2_dispatch,
        crate::avx::idct_dequant_32x16_avx2
    ),
    (
        idct_dequant_32x32,
        1024,
        32,
        32,
        1,
        false,
        idct_dequant_32x32_scalar_quant,
        idct32x32_neon_dispatch2,
        crate::neon::idct_dequant_32x32_neon,
        idct32x32_avx2_dispatch2,
        crate::avx::idct_dequant_32x32_avx2
    ),
    (
        iadst_dequant_4x4,
        16,
        4,
        4,
        0,
        false,
        iadst_dequant_4x4,
        iadst4x4_neon_dispatch,
        crate::neon::iadst_dequant_4x4_neon,
        iadst4x4_avx2_dispatch,
        crate::avx::iadst_dequant_4x4_avx2
    ),
    (
        iadstdct_dequant_4x4,
        16,
        4,
        4,
        0,
        false,
        iadstdct_dequant_4x4,
        iadstdct4x4_neon_dispatch,
        crate::neon::iadstdct_dequant_4x4_neon,
        iadstdct4x4_avx2_dispatch,
        crate::avx::iadstdct_dequant_4x4_avx2
    ),
    (
        idctadst_dequant_4x4,
        16,
        4,
        4,
        0,
        false,
        idctadst_dequant_4x4,
        idctadst4x4_neon_dispatch,
        crate::neon::idctadst_dequant_4x4_neon,
        idctadst4x4_avx2_dispatch,
        crate::avx::idctadst_dequant_4x4_avx2
    ),
    (
        iadst_dequant_4x8,
        32,
        4,
        8,
        0,
        false,
        iadst_dequant_4x8,
        iadst4x8_neon_dispatch,
        crate::neon::iadst_dequant_4x8_neon,
        iadst4x8_avx2_dispatch,
        crate::avx::iadst_dequant_4x8_avx2
    ),
    (
        iadstdct_dequant_4x8,
        32,
        4,
        8,
        0,
        false,
        iadstdct_dequant_4x8,
        iadstdct4x8_neon_dispatch,
        crate::neon::iadstdct_dequant_4x8_neon,
        iadstdct4x8_avx2_dispatch,
        crate::avx::iadstdct_dequant_4x8_avx2
    ),
    (
        idctadst_dequant_4x8,
        32,
        4,
        8,
        0,
        false,
        idctadst_dequant_4x8,
        idctadst4x8_neon_dispatch,
        crate::neon::idctadst_dequant_4x8_neon,
        idctadst4x8_avx2_dispatch,
        crate::avx::idctadst_dequant_4x8_avx2
    ),
    (
        iadst_dequant_8x8,
        64,
        8,
        8,
        0,
        false,
        iadst_dequant_8x8,
        iadst8x8_neon_dispatch,
        crate::neon::iadst_dequant_8x8_neon,
        iadst8x8_avx2_dispatch,
        crate::avx::iadst_dequant_8x8_avx2
    ),
    (
        iadstdct_dequant_8x8,
        64,
        8,
        8,
        0,
        false,
        iadstdct_dequant_8x8,
        iadstdct8x8_neon_dispatch,
        crate::neon::iadstdct_dequant_8x8_neon,
        iadstdct8x8_avx2_dispatch,
        crate::avx::iadstdct_dequant_8x8_avx2
    ),
    (
        idctadst_dequant_8x8,
        64,
        8,
        8,
        0,
        false,
        idctadst_dequant_8x8,
        idctadst8x8_neon_dispatch,
        crate::neon::idctadst_dequant_8x8_neon,
        idctadst8x8_avx2_dispatch,
        crate::avx::idctadst_dequant_8x8_avx2
    ),
    (
        iadst_dequant_8x16,
        128,
        8,
        16,
        0,
        false,
        iadst_dequant_8x16,
        iadst8x16_neon_dispatch,
        crate::neon::iadst_dequant_8x16_neon,
        iadst8x16_avx2_dispatch,
        crate::avx::iadst_dequant_8x16_avx2
    ),
    (
        iadstdct_dequant_8x16,
        128,
        8,
        16,
        0,
        false,
        iadstdct_dequant_8x16,
        iadstdct8x16_neon_dispatch,
        crate::neon::iadstdct_dequant_8x16_neon,
        iadstdct8x16_avx2_dispatch,
        crate::avx::iadstdct_dequant_8x16_avx2
    ),
    (
        idctadst_dequant_8x16,
        128,
        8,
        16,
        0,
        false,
        idctadst_dequant_8x16,
        idctadst8x16_neon_dispatch,
        crate::neon::idctadst_dequant_8x16_neon,
        idctadst8x16_avx2_dispatch,
        crate::avx::idctadst_dequant_8x16_avx2
    ),
    (
        iadst_dequant_16x8,
        128,
        16,
        8,
        0,
        false,
        iadst_dequant_16x8,
        iadst16x8_neon_dispatch,
        crate::neon::iadst_dequant_16x8_neon,
        iadst16x8_avx2_dispatch,
        crate::avx::iadst_dequant_16x8_avx2
    ),
    (
        iadstdct_dequant_16x8,
        128,
        16,
        8,
        0,
        false,
        iadstdct_dequant_16x8,
        iadstdct16x8_neon_dispatch,
        crate::neon::iadstdct_dequant_16x8_neon,
        iadstdct16x8_avx2_dispatch,
        crate::avx::iadstdct_dequant_16x8_avx2
    ),
    (
        idctadst_dequant_16x8,
        128,
        16,
        8,
        0,
        false,
        idctadst_dequant_16x8,
        idctadst16x8_neon_dispatch,
        crate::neon::idctadst_dequant_16x8_neon,
        idctadst16x8_avx2_dispatch,
        crate::avx::idctadst_dequant_16x8_avx2
    ),
    (
        iadst_dequant_16x16,
        256,
        16,
        16,
        0,
        false,
        iadst_dequant_16x16_scalar_quant,
        iadst16x16_neon_dispatch2,
        crate::neon::iadst_dequant_16x16_neon,
        iadst16x16_avx2_dispatch2,
        crate::avx::iadst_dequant_16x16_avx2
    ),
    (
        iadstdct_dequant_16x16,
        256,
        16,
        16,
        0,
        false,
        iadstdct_dequant_16x16_scalar_quant,
        iadstdct16x16_neon_dispatch2,
        crate::neon::iadstdct_dequant_16x16_neon,
        iadstdct16x16_avx2_dispatch2,
        crate::avx::iadstdct_dequant_16x16_avx2
    ),
    (
        idctadst_dequant_16x16,
        256,
        16,
        16,
        0,
        false,
        idctadst_dequant_16x16_scalar_quant,
        idctadst16x16_neon_dispatch2,
        crate::neon::idctadst_dequant_16x16_neon,
        idctadst16x16_avx2_dispatch2,
        crate::avx::idctadst_dequant_16x16_avx2
    ),
    (
        ivdct_dequant_4x4,
        16,
        4,
        4,
        0,
        true,
        ivdct_dequant_4x4,
        ivdct4x4_neon_dispatch,
        crate::neon::ivdct_dequant_4x4_neon,
        ivdct4x4_avx2_dispatch,
        crate::avx::ivdct_dequant_4x4_avx2
    ),
    (
        ihdct_dequant_4x4,
        16,
        4,
        4,
        0,
        true,
        ihdct_dequant_4x4,
        ihdct4x4_neon_dispatch,
        crate::neon::ihdct_dequant_4x4_neon,
        ihdct4x4_avx2_dispatch,
        crate::avx::ihdct_dequant_4x4_avx2
    ),
    (
        ivdct_dequant_8x8,
        64,
        8,
        8,
        0,
        true,
        ivdct_dequant_8x8,
        ivdct8x8_neon_dispatch,
        crate::neon::ivdct_dequant_8x8_neon,
        ivdct8x8_avx2_dispatch,
        crate::avx::ivdct_dequant_8x8_avx2
    ),
    (
        ihdct_dequant_8x8,
        64,
        8,
        8,
        0,
        true,
        ihdct_dequant_8x8,
        ihdct8x8_neon_dispatch,
        crate::neon::ihdct_dequant_8x8_neon,
        ihdct8x8_avx2_dispatch,
        crate::avx::ihdct_dequant_8x8_avx2
    ),
    (
        ivdct_dequant_8x16,
        128,
        8,
        16,
        0,
        true,
        ivdct_dequant_8x16,
        ivdct8x16_neon_dispatch,
        crate::neon::ivdct_dequant_8x16_neon,
        ivdct8x16_avx2_dispatch,
        crate::avx::ivdct_dequant_8x16_avx2
    ),
    (
        ihdct_dequant_8x16,
        128,
        8,
        16,
        0,
        true,
        ihdct_dequant_8x16,
        ihdct8x16_neon_dispatch,
        crate::neon::ihdct_dequant_8x16_neon,
        ihdct8x16_avx2_dispatch,
        crate::avx::ihdct_dequant_8x16_avx2
    ),
    (
        ivdct_dequant_16x8,
        128,
        16,
        8,
        0,
        true,
        ivdct_dequant_16x8,
        ivdct16x8_neon_dispatch,
        crate::neon::ivdct_dequant_16x8_neon,
        ivdct16x8_avx2_dispatch,
        crate::avx::ivdct_dequant_16x8_avx2
    ),
    (
        ihdct_dequant_16x8,
        128,
        16,
        8,
        0,
        true,
        ihdct_dequant_16x8,
        ihdct16x8_neon_dispatch,
        crate::neon::ihdct_dequant_16x8_neon,
        ihdct16x8_avx2_dispatch,
        crate::avx::ihdct_dequant_16x8_avx2
    ),
    (
        iidentity_dequant_4x4,
        16,
        4,
        4,
        0,
        true,
        iidentity_dequant_4x4,
        iidtx4x4_neon_dispatch,
        crate::neon::iidentity_dequant_4x4_neon,
        iidtx4x4_avx2_dispatch,
        crate::avx::iidentity_dequant_4x4_avx2
    ),
    (
        iidentity_dequant_8x8,
        64,
        8,
        8,
        0,
        true,
        iidentity_dequant_8x8,
        iidtx8x8_neon_dispatch,
        crate::neon::iidentity_dequant_8x8_neon,
        iidtx8x8_avx2_dispatch,
        crate::avx::iidentity_dequant_8x8_avx2
    ),
    (
        iidtx_dequant_8x16,
        128,
        8,
        16,
        0,
        true,
        iidtx_dequant_8x16,
        iidtx8x16_neon_dispatch,
        crate::neon::iidtx_dequant_8x16_neon,
        iidtx8x16_avx2_dispatch,
        crate::avx::iidtx_dequant_8x16_avx2
    ),
    (
        iidtx_dequant_16x8,
        128,
        16,
        8,
        0,
        true,
        iidtx_dequant_16x8,
        iidtx16x8_neon_dispatch,
        crate::neon::iidtx_dequant_16x8_neon,
        iidtx16x8_avx2_dispatch,
        crate::avx::iidtx_dequant_16x8_avx2
    ),
    (
        iidentity_dequant_16x16,
        256,
        16,
        16,
        0,
        true,
        iidentity_dequant_16x16,
        iidtx16x16_neon_dispatch,
        crate::neon::iidentity_dequant_16x16_neon,
        iidtx16x16_avx2_dispatch,
        crate::avx::iidentity_dequant_16x16_avx2
    ),
);

#[cfg(test)]
mod inverse_dispatch_tests {
    use super::*;

    fn levels<const N: usize>(seed: u32) -> [i32; N] {
        let mut out = [0; N];
        let mut state = seed ^ N as u32;
        for (i, value) in out.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let level = ((state >> 24) as i32 & 63) - 31;
            *value = if i % 9 == 0 { 0 } else { level };
        }
        out[0] = 19;
        out
    }

    #[test]
    fn selected_inverse_dispatch_matches_scalar() {
        let selected = IdctDispatch::selected();
        let scalar = IdctDispatch::scalar();
        let quants = [
            Quant::new(8, 8),
            Quant::new(96, 8),
            Quant::new_chroma(40, 10),
        ];

        macro_rules! check {
            ($field:ident, $n:literal) => {
                for (qi, q) in quants.iter().enumerate() {
                    for seed in [0x1234_5678, 0x8765_4321, 0xa5a5_5a5a] {
                        let input = levels::<$n>(seed ^ qi as u32);
                        assert_eq!(
                            selected.$field(&input, q),
                            scalar.$field(&input, q),
                            "{} N={} q={} seed={seed:#x}",
                            stringify!($field),
                            $n,
                            qi,
                        );
                    }
                }
            };
        }

        check!(idct_dequant_4x4, 16);
        check!(idct_dequant_4x8, 32);
        check!(idct_dequant_8x4, 32);
        check!(idct_dequant_4x16, 64);
        check!(idct_dequant_16x4, 64);
        check!(idct_dequant_8x8, 64);
        check!(idct_dequant_8x16, 128);
        check!(idct_dequant_16x8, 128);
        check!(idct_dequant_16x16, 256);
        check!(idct_dequant_16x32, 512);
        check!(idct_dequant_32x16, 512);
        check!(idct_dequant_32x32, 1024);
        check!(iadst_dequant_4x4, 16);
        check!(iadstdct_dequant_4x4, 16);
        check!(idctadst_dequant_4x4, 16);
        check!(iadst_dequant_4x8, 32);
        check!(iadstdct_dequant_4x8, 32);
        check!(idctadst_dequant_4x8, 32);
        check!(iadst_dequant_8x8, 64);
        check!(iadstdct_dequant_8x8, 64);
        check!(idctadst_dequant_8x8, 64);
        check!(iadst_dequant_8x16, 128);
        check!(iadstdct_dequant_8x16, 128);
        check!(idctadst_dequant_8x16, 128);
        check!(iadst_dequant_16x8, 128);
        check!(iadstdct_dequant_16x8, 128);
        check!(idctadst_dequant_16x8, 128);
        check!(iadst_dequant_16x16, 256);
        check!(iadstdct_dequant_16x16, 256);
        check!(idctadst_dequant_16x16, 256);
        check!(ivdct_dequant_4x4, 16);
        check!(ihdct_dequant_4x4, 16);
        check!(ivdct_dequant_8x8, 64);
        check!(ihdct_dequant_8x8, 64);
        check!(ivdct_dequant_8x16, 128);
        check!(ihdct_dequant_8x16, 128);
        check!(ivdct_dequant_16x8, 128);
        check!(ihdct_dequant_16x8, 128);
        check!(iidentity_dequant_4x4, 16);
        check!(iidentity_dequant_8x8, 64);
        check!(iidtx_dequant_8x16, 128);
        check!(iidtx_dequant_16x8, 128);
        check!(iidentity_dequant_16x16, 256);
    }

    #[test]
    fn selected_inverse_dispatch_with_qmatrix_matches_scalar_at_every_bit_depth() {
        let selected = IdctDispatch::selected();
        let scalar = IdctDispatch::scalar();
        let quants = [
            Quant::new_with_qm(40, 8, 0),
            Quant::new_with_qm(96, 10, 7),
            Quant::new_with_qm(180, 12, 14),
            Quant::new_chroma_with_delta_qm(128, -7, 11, 12, 3),
        ];
        macro_rules! check {
            ($field:ident, $n:literal) => {
                for (qi, q) in quants.iter().enumerate() {
                    for seed in [0x0123_4567, 0x89ab_cdef, 0x55aa_33cc] {
                        let input = levels::<$n>(seed ^ qi as u32);
                        assert_eq!(
                            selected.$field(&input, q),
                            scalar.$field(&input, q),
                            "{} N={} qmatrix={} seed={seed:#x}",
                            stringify!($field),
                            $n,
                            qi,
                        );
                    }
                }
            };
        }
        check!(idct_dequant_8x8, 64);
        check!(idct_dequant_8x16, 128);
        check!(idct_dequant_16x8, 128);
        check!(iadstdct_dequant_8x8, 64);
        check!(idct_dequant_16x16, 256);
        check!(iadstdct_dequant_16x16, 256);
        check!(idct_dequant_32x32, 1024);
        check!(idct_dequant_16x32, 512);
        check!(idct_dequant_32x16, 512);
    }
}

pub(crate) fn idct_dequant_16x32(levels: &[i32; 512], q: &Quant) -> [i32; 512] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 512];
    for rc in 0..512 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 16, 32);
        let mag = (((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) >> 1) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // rect2 prescale + transpose: tmp[row*16+col] = (coeff[row + col*32]*181+128)>>8
    let mut tmp = [0i32; 512];
    for row in 0..32 {
        for col in 0..16 {
            tmp[row * 16 + col] = (coeff[row + col * 32] * 181 + 128) >> 8;
        }
    }
    // row transform: width-16 inv_dct16 (stride 1) over each of the 32 rows
    for row in 0..32 {
        inv_dct16_1d(&mut tmp[row * 16..], 1, rmin, rmax);
    }
    // mid shift = 1: (t + 1) >> 1, clipped to int16
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    // column transform: height-32 inv_dct32 (stride 16) over each of the 16 columns
    for col in 0..16 {
        inv_dct32_1d(&mut tmp[col..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// DCT_DCT inverse for RTX_16X4 — the 2:1 pipeline without the rect2 prescale
/// (dav1d applies `*181/256` only to 2:1 shapes). Row inv_dct16 over 4 rows,
/// mid shift 1, column inv_dct4, final `(t + 8) >> 4`.
pub(crate) fn idct_dequant_16x4(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qs = q.dequant_step(rc, 16, 4);
        let mag = ((lvl.unsigned_abs() as u64 * qs as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for row in 0..4 {
        for col in 0..16 {
            tmp[row * 16 + col] = coeff[row + col * 4];
        }
    }
    for row in 0..4 {
        inv_dct16_1d(&mut tmp[row * 16..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for col in 0..16 {
        inv_dct4_1d(&mut tmp[col..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// DCT_DCT inverse for RTX_4X16.
pub(crate) fn idct_dequant_4x16(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qs = q.dequant_step(rc, 4, 16);
        let mag = ((lvl.unsigned_abs() as u64 * qs as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for row in 0..16 {
        for col in 0..4 {
            tmp[row * 4 + col] = coeff[row + col * 16];
        }
    }
    for row in 0..16 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_dct16_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn idct_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 128];
    for rc in 0..128 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 16, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // rect2 prescale + transpose: dav1d reads coeff[y + x*sh] with sh=h=8.
    // tmp is laid out row-major width-16: tmp[row*16 + col].
    let mut tmp = [0i32; 128];
    for row in 0..8 {
        for col in 0..16 {
            tmp[row * 16 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    // row transform: width-16 inv_dct16 (stride 1) over each of the 8 rows.
    for row in 0..8 {
        inv_dct16_1d(&mut tmp[row * 16..], 1, rmin, rmax);
    }
    // mid shift = 1: (t + 1) >> 1, clipped to int16.
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    // column transform: height-8 inv_dct8 (stride 16) over each of the 16 columns.
    for col in 0..16 {
        inv_dct8_1d(&mut tmp[col..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// ADST_ADST inverse for RTX_8X16 — `idct_dequant_8x16` with both 1D passes
/// swapped for the ADST kernel. The rect2 prescale, the mid shift and the final
/// `(t + 8) >> 4` are identical, so this stays bit-exact with the decoder's
/// rectangular transform pipeline.
pub(crate) fn iadst_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 128];
    for rc in 0..128 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qs = q.dequant_step(rc, 8, 16);
        let mag = ((lvl.unsigned_abs() as u64 * qs as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 128];
    for row in 0..16 {
        for col in 0..8 {
            tmp[row * 8 + col] = (coeff[row + col * 16] * 181 + 128) >> 8;
        }
    }
    for row in 0..16 {
        inv_adst8_1d(&mut tmp[row * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for col in 0..8 {
        inv_adst16_1d(&mut tmp[col..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// ADST_ADST inverse for RTX_16X8.
pub(crate) fn iadst_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 128];
    for rc in 0..128 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qs = q.dequant_step(rc, 16, 8);
        let mag = ((lvl.unsigned_abs() as u64 * qs as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 128];
    for row in 0..8 {
        for col in 0..16 {
            tmp[row * 16 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    for row in 0..8 {
        inv_adst16_1d(&mut tmp[row * 16..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for col in 0..16 {
        inv_adst8_1d(&mut tmp[col..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// One 1-D pass kind of the rect16 inverse pipeline. `Id8`/`Id16` are dav1d's
/// identity kernels (x2 resp. `2x + ((x*1697+1024)>>11)`, no internal clamp);
/// the DCT/ADST kernels are the shared clamped 1-D inverses.
#[derive(Clone, Copy, PartialEq)]
enum RectPass {
    Dct8,
    Dct16,
    Adst8,
    Adst16,
    Id8,
    Id16,
}

#[inline]
fn id16_1d(v: i32) -> i32 {
    2 * v + ((v * 1697 + 1024) >> 11)
}

fn irect_dequant_128(
    levels: &[i32; 128],
    q: &Quant,
    w: usize,
    row_pass: RectPass,
    col_pass: RectPass,
    qm: bool,
) -> [i32; 128] {
    let h = 128 / w;
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 128];
    for rc in 0..128 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qs = if qm {
            q.dequant_step(rc, w, h)
        } else if rc == 0 {
            dc_q
        } else {
            ac_q
        };
        let mag = ((lvl.unsigned_abs() as u64 * qs as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 128];
    for row in 0..h {
        for col in 0..w {
            tmp[row * w + col] = (coeff[row + col * h] * 181 + 128) >> 8;
        }
    }
    let apply_row = |tmp: &mut [i32; 128], pass: RectPass| match pass {
        RectPass::Dct8 => {
            for row in 0..h {
                inv_dct8_1d(&mut tmp[row * w..], 1, rmin, rmax);
            }
        }
        RectPass::Dct16 => {
            for row in 0..h {
                inv_dct16_1d(&mut tmp[row * w..], 1, rmin, rmax);
            }
        }
        RectPass::Adst8 => {
            for row in 0..h {
                inv_adst8_1d(&mut tmp[row * w..], 1, rmin, rmax);
            }
        }
        RectPass::Adst16 => {
            for row in 0..h {
                inv_adst16_1d(&mut tmp[row * w..], 1, rmin, rmax);
            }
        }
        RectPass::Id8 => {
            for t in tmp.iter_mut() {
                *t *= 2;
            }
        }
        RectPass::Id16 => {
            for t in tmp.iter_mut() {
                *t = id16_1d(*t);
            }
        }
    };
    apply_row(&mut tmp, row_pass);
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    match col_pass {
        RectPass::Dct8 => {
            for col in 0..w {
                inv_dct8_1d(&mut tmp[col..], w, cmin, cmax);
            }
        }
        RectPass::Dct16 => {
            for col in 0..w {
                inv_dct16_1d(&mut tmp[col..], w, cmin, cmax);
            }
        }
        RectPass::Adst8 => {
            for col in 0..w {
                inv_adst8_1d(&mut tmp[col..], w, cmin, cmax);
            }
        }
        RectPass::Adst16 => {
            for col in 0..w {
                inv_adst16_1d(&mut tmp[col..], w, cmin, cmax);
            }
        }
        RectPass::Id8 => {
            for t in tmp.iter_mut() {
                *t *= 2;
            }
        }
        RectPass::Id16 => {
            for t in tmp.iter_mut() {
                *t = id16_1d(*t);
            }
        }
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// ADST_DCT inverse for RTX_16X8 (vertical ADST-8, horizontal DCT-16).
pub(crate) fn iadstdct_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    irect_dequant_128(levels, q, 16, RectPass::Dct16, RectPass::Adst8, true)
}

/// DCT_ADST inverse for RTX_16X8 (vertical DCT-8, horizontal ADST-16).
pub(crate) fn idctadst_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    irect_dequant_128(levels, q, 16, RectPass::Adst16, RectPass::Dct8, true)
}

/// ADST_DCT inverse for RTX_8X16 (vertical ADST-16, horizontal DCT-8).
pub(crate) fn iadstdct_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    irect_dequant_128(levels, q, 8, RectPass::Dct8, RectPass::Adst16, true)
}

/// DCT_ADST inverse for RTX_8X16 (vertical DCT-16, horizontal ADST-8).
pub(crate) fn idctadst_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    irect_dequant_128(levels, q, 8, RectPass::Adst8, RectPass::Dct16, true)
}

/// V_DCT inverse for RTX_16X8 (row identity-16, column inv DCT-8).
pub(crate) fn ivdct_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    // identity family: QM not applied (qm = false == flat dc/ac dequant)
    irect_dequant_128(levels, q, 16, RectPass::Id16, RectPass::Dct8, false)
}

/// H_DCT inverse for RTX_16X8 (row inv DCT-16, column identity-8).
pub(crate) fn ihdct_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    // identity family: QM not applied (qm = false == flat dc/ac dequant)
    irect_dequant_128(levels, q, 16, RectPass::Dct16, RectPass::Id8, false)
}

/// V_DCT inverse for RTX_8X16 (row identity-8, column inv DCT-16).
pub(crate) fn ivdct_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    // identity family: QM not applied (qm = false == flat dc/ac dequant)
    irect_dequant_128(levels, q, 8, RectPass::Id8, RectPass::Dct16, false)
}

/// H_DCT inverse for RTX_8X16 (row inv DCT-8, column identity-16).
pub(crate) fn ihdct_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    // identity family: QM not applied (qm = false == flat dc/ac dequant)
    irect_dequant_128(levels, q, 8, RectPass::Dct8, RectPass::Id16, false)
}

/// IDTX inverse for RTX_16X8 (identity both passes, no quantizer matrix).
pub(crate) fn iidtx_dequant_16x8(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    irect_dequant_128(levels, q, 16, RectPass::Id16, RectPass::Id8, false)
}

/// IDTX inverse for RTX_8X16 (identity both passes, no quantizer matrix).
pub(crate) fn iidtx_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    irect_dequant_128(levels, q, 8, RectPass::Id8, RectPass::Id16, false)
}

pub(crate) fn idct_dequant_8x16(levels: &[i32; 128], q: &Quant) -> [i32; 128] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 128];
    for rc in 0..128 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 8, 16);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // rect2 prescale + transpose: tmp[row*8+col] = (coeff[row + col*16]*181+128)>>8
    let mut tmp = [0i32; 128];
    for row in 0..16 {
        for col in 0..8 {
            tmp[row * 8 + col] = (coeff[row + col * 16] * 181 + 128) >> 8;
        }
    }
    // row transform: width-8 inv_dct8 (stride 1) over each of the 16 rows
    for row in 0..16 {
        inv_dct8_1d(&mut tmp[row * 8..], 1, rmin, rmax);
    }
    // mid shift = 1: (t + 1) >> 1, clipped to int16
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    // column transform: height-16 inv_dct16 (stride 8) over each of the 8 columns
    for col in 0..8 {
        inv_dct16_1d(&mut tmp[col..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// 8x4 inverse: coeff layout `[fx*4+fy]` (8 wide x 4 tall). Transpose of 4x8.
pub(crate) fn idct_dequant_8x4(levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 32];
    for rc in 0..32 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 8, 4);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // tmp[row*8+col] = (coeff[col*4 + row] * 181 + 128) >> 8   (is_rect2 prescale)
    let mut tmp = [0i32; 32];
    for row in 0..4 {
        for col in 0..8 {
            tmp[row * 8 + col] = (coeff[col * 4 + row] * 181 + 128) >> 8;
        }
    }
    // row transform: width-8 inv_dct8 (stride 1) over each of the 4 rows
    for row in 0..4 {
        inv_dct8_1d(&mut tmp[row * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    // column transform: height-4 inv_dct4 (stride 8) over each of the 8 columns
    for col in 0..8 {
        inv_dct4_1d(&mut tmp[col..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn idct_dequant_4x8(levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 32];
    for rc in 0..32 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 4, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // tmp[row*4+col] = (coeff[row + col*8] * 181 + 128) >> 8   (is_rect2 prescale)
    let mut tmp = [0i32; 32];
    for row in 0..8 {
        for col in 0..4 {
            tmp[row * 4 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    // row transform: width-4 inv_dct4 (stride 1) over each of the 8 rows
    for row in 0..8 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    // shift = 0 => only clip
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    // column transform: height-8 inv_dct8 (stride 4) over each of the 4 columns
    for col in 0..4 {
        inv_dct8_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn inv_dct4_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    let (in0, in1, in2, in3) = (c[0], c[s], c[2 * s], c[3 * s]);
    let t0 = ((in0 + in2) * 181 + 128) >> 8;
    let t1 = ((in0 - in2) * 181 + 128) >> 8;
    let t2 = ((in1 * 1567 - in3 * (3784 - 4096) + 2048) >> 12) - in3;
    let t3 = ((in1 * (3784 - 4096) + in3 * 1567 + 2048) >> 12) + in1;
    c[0] = clip(t0 + t3);
    c[s] = clip(t1 + t2);
    c[2 * s] = clip(t1 - t2);
    c[3 * s] = clip(t0 - t3);
}

/// dav1d's exact integer 1-D inverse DCT8 (`inv_dct8_1d_internal_c`, tx64=0).
pub(crate) fn inv_dct8_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    inv_dct4_1d(c, 2 * s, min, max); // even positions c[0],c[2s],c[4s],c[6s]
    let (in1, in3, in5, in7) = (c[s], c[3 * s], c[5 * s], c[7 * s]);
    let t4a = ((in1 * 799 - in7 * (4017 - 4096) + 2048) >> 12) - in7;
    let mut t5a = (in5 * 1703 - in3 * 1138 + 1024) >> 11;
    let mut t6a = (in5 * 1138 + in3 * 1703 + 1024) >> 11;
    let t7a = ((in1 * (4017 - 4096) + in7 * 799 + 2048) >> 12) + in1;
    let t4 = clip(t4a + t5a);
    t5a = clip(t4a - t5a);
    let t7 = clip(t7a + t6a);
    t6a = clip(t7a - t6a);
    let t5 = ((t6a - t5a) * 181 + 128) >> 8;
    let t6 = ((t6a + t5a) * 181 + 128) >> 8;
    let (t0, t1, t2, t3) = (c[0], c[2 * s], c[4 * s], c[6 * s]);
    c[0] = clip(t0 + t7);
    c[s] = clip(t1 + t6);
    c[2 * s] = clip(t2 + t5);
    c[3 * s] = clip(t3 + t4);
    c[4 * s] = clip(t3 - t4);
    c[5 * s] = clip(t2 - t5);
    c[6 * s] = clip(t1 - t6);
    c[7 * s] = clip(t0 - t7);
}

/// dav1d's exact integer 1-D inverse ADST8 (`inv_adst8_1d_internal_c`). All
/// inputs are read before any output is written, so it is safe in place.
pub(crate) fn inv_adst8_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    let (in0, in1, in2, in3) = (c[0], c[s], c[2 * s], c[3 * s]);
    let (in4, in5, in6, in7) = (c[4 * s], c[5 * s], c[6 * s], c[7 * s]);
    let t0a = (((4076 - 4096) * in7 + 401 * in0 + 2048) >> 12) + in7;
    let t1a = ((401 * in7 - (4076 - 4096) * in0 + 2048) >> 12) - in0;
    let t2a = (((3612 - 4096) * in5 + 1931 * in2 + 2048) >> 12) + in5;
    let t3a = ((1931 * in5 - (3612 - 4096) * in2 + 2048) >> 12) - in2;
    let t4a = (1299 * in3 + 1583 * in4 + 1024) >> 11;
    let t5a = (1583 * in3 - 1299 * in4 + 1024) >> 11;
    let t6a = ((1189 * in1 + (3920 - 4096) * in6 + 2048) >> 12) + in6;
    let t7a = (((3920 - 4096) * in1 - 1189 * in6 + 2048) >> 12) + in1;
    let t0 = clip(t0a + t4a);
    let t1 = clip(t1a + t5a);
    let mut t2 = clip(t2a + t6a);
    let mut t3 = clip(t3a + t7a);
    let t4 = clip(t0a - t4a);
    let t5 = clip(t1a - t5a);
    let mut t6 = clip(t2a - t6a);
    let mut t7 = clip(t3a - t7a);
    let t4a = (((3784 - 4096) * t4 + 1567 * t5 + 2048) >> 12) + t4;
    let t5a = ((1567 * t4 - (3784 - 4096) * t5 + 2048) >> 12) - t5;
    let t6a = (((3784 - 4096) * t7 - 1567 * t6 + 2048) >> 12) + t7;
    let t7a = ((1567 * t7 + (3784 - 4096) * t6 + 2048) >> 12) + t6;
    c[0] = clip(t0 + t2);
    c[7 * s] = -clip(t1 + t3);
    t2 = clip(t0 - t2);
    t3 = clip(t1 - t3);
    c[s] = -clip(t4a + t6a);
    c[6 * s] = clip(t5a + t7a);
    t6 = clip(t4a - t6a);
    t7 = clip(t5a - t7a);
    c[3 * s] = -(((t2 + t3) * 181 + 128) >> 8);
    c[4 * s] = ((t2 - t3) * 181 + 128) >> 8;
    c[2 * s] = ((t6 + t7) * 181 + 128) >> 8;
    c[5 * s] = -(((t6 - t7) * 181 + 128) >> 8);
}

#[derive(Clone, Copy)]
pub(crate) struct IdctDequant {
    pub(crate) dc_q: i32,
    pub(crate) ac_q: i32,
    pub(crate) rmin: i32,
    pub(crate) rmax: i32,
    pub(crate) cmin: i32,
    pub(crate) cmax: i32,
    pub(crate) cf_max: i32,
    /// The transform's inverse QM row, resolved once in `new`; `None` when
    /// the matrix is flat. Supersedes carrying (level, chroma, tx_w, tx_h)
    /// and re-deriving the table offset for every coefficient.
    pub(crate) qm: Option<&'static [u8]>,
}

impl IdctDequant {
    #[inline]
    fn new(q: &Quant, tx_w: usize, tx_h: usize) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
        Self {
            dc_q: q.dc_q(),
            ac_q: q.ac_q(),
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
            qm: crate::quant::qm_row(q.qm_level(), q.qm_chroma(), tx_w, tx_h),
        }
    }

    #[inline]
    fn new_flat(q: &Quant) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
        Self {
            dc_q: q.dc_q(),
            ac_q: q.ac_q(),
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
            qm: None,
        }
    }

    #[inline]
    fn step(&self, rc: usize) -> i32 {
        let base = if rc == 0 { self.dc_q } else { self.ac_q };
        let Some(table) = self.qm else {
            return base;
        };
        (base * table[rc] as i32 + 16) >> 5
    }
}

#[allow(unused)]
pub(crate) fn idct_dequant_8x8_scalar(levels: &[i32; 64], dequant: &IdctDequant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = (
        dequant.rmin,
        dequant.rmax,
        dequant.cmin,
        dequant.cmax,
        dequant.cf_max,
    );
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = dequant.step(rc);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    // tmp[y*8+x] = coeff[y + x*8]; row inv_dct8, >>1 (rnd 1) clip, col inv_dct8, >>4
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for y in 0..8 {
        inv_dct8_1d(&mut tmp[y * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for x in 0..8 {
        inv_dct8_1d(&mut tmp[x..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn ivdct_dequant_8x8(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (_rmin, _rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qd = q.dequant_step(rc, 8, 8);
        let mag = ((lvl.unsigned_abs() as u64 * qd as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for t in tmp.iter_mut() {
        *t *= 2; // row pass: identity8
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for x in 0..8 {
        inv_dct8_1d(&mut tmp[x..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn ihdct_dequant_8x8(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qd = q.dequant_step(rc, 8, 8);
        let mag = ((lvl.unsigned_abs() as u64 * qd as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for y in 0..8 {
        inv_dct8_1d(&mut tmp[y * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t *= 2; // col pass: identity8
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iadstdct_dequant_8x8(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 8, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for y in 0..8 {
        inv_dct8_1d(&mut tmp[y * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for x in 0..8 {
        inv_adst8_1d(&mut tmp[x..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn idctadst_dequant_8x8(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 8, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for y in 0..8 {
        inv_adst8_1d(&mut tmp[y * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for x in 0..8 {
        inv_dct8_1d(&mut tmp[x..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iadst_dequant_8x8(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 8, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    for y in 0..8 {
        inv_adst8_1d(&mut tmp[y * 8..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    for x in 0..8 {
        inv_adst8_1d(&mut tmp[x..], 8, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iidentity_dequant_8x8(levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (_rmin, _rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 64];
    for rc in 0..64 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 64];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y * 8 + x] = coeff[y + x * 8];
        }
    }
    // Row pass: identity (x2), no internal clamp, then the per-size (t+1)>>1 clip.
    for t in tmp.iter_mut() {
        *t *= 2;
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 1) >> 1).clamp(cmin, cmax);
    }
    // Col pass: identity (x2), then the per-size (t+8)>>4 final shift.
    for t in tmp.iter_mut() {
        *t *= 2;
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d-exact TX_16X16 IDTX inverse: dequant, transpose, identity16 per pass
/// (`2*x + ((x*1697 + 1024) >> 11)`), row shift (t+2)>>2 with the
/// intermediate clip, col shift (t+8)>>4.
pub(crate) fn iidentity_dequant_16x16(levels: &[i32; 256], q: &Quant) -> [i32; 256] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (_rmin, _rmax, cmin, cmax, cf_max) = q.clips();
    let (dc_q, ac_q) = (q.dc_q(), q.ac_q());
    let mut coeff = [0i32; 256];
    for rc in 0..256 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let qd = if rc == 0 { dc_q } else { ac_q };
        let mag = ((lvl.unsigned_abs() as u64 * qd as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 256];
    for y in 0..16 {
        for x in 0..16 {
            tmp[y * 16 + x] = coeff[y + x * 16];
        }
    }
    let id16 = |v: i32| 2 * v + ((v * 1697 + 1024) >> 11);
    for t in tmp.iter_mut() {
        *t = id16(*t);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 2) >> 2).clamp(cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = id16(*t);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn inv_dct16_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    inv_dct8_1d(c, 2 * s, min, max); // even positions c[0],c[2s],..,c[14s]
    let (in1, in3, in5, in7) = (c[s], c[3 * s], c[5 * s], c[7 * s]);
    let (in9, in11, in13, in15) = (c[9 * s], c[11 * s], c[13 * s], c[15 * s]);
    // stage 1 (odd inputs -> t8a..t15a)
    let t8a = ((in1 * 401 - in15 * (4076 - 4096) + 2048) >> 12) - in15;
    let t9a = (in9 * 1583 - in7 * 1299 + 1024) >> 11;
    let t10a = ((in5 * 1931 - in11 * (3612 - 4096) + 2048) >> 12) - in11;
    let t11a = ((in13 * (3920 - 4096) - in3 * 1189 + 2048) >> 12) + in13;
    let t12a = ((in13 * 1189 + in3 * (3920 - 4096) + 2048) >> 12) + in3;
    let t13a = ((in5 * (3612 - 4096) + in11 * 1931 + 2048) >> 12) + in5;
    let t14a = (in9 * 1299 + in7 * 1583 + 1024) >> 11;
    let t15a = ((in1 * (4076 - 4096) + in15 * 401 + 2048) >> 12) + in1;
    // stage 2 (butterflies)
    let t8 = clip(t8a + t9a);
    let t9 = clip(t8a - t9a);
    let t10 = clip(t11a - t10a);
    let t11 = clip(t11a + t10a);
    let t12 = clip(t12a + t13a);
    let t13 = clip(t12a - t13a);
    let t14 = clip(t15a - t14a);
    let t15 = clip(t15a + t14a);
    // stage 3 (rotations)
    let t9a = ((t14 * 1567 - t9 * (3784 - 4096) + 2048) >> 12) - t9;
    let t14a = ((t14 * (3784 - 4096) + t9 * 1567 + 2048) >> 12) + t14;
    let t10a = ((-(t13 * (3784 - 4096) + t10 * 1567) + 2048) >> 12) - t13;
    let t13a = ((t13 * 1567 - t10 * (3784 - 4096) + 2048) >> 12) - t10;
    // stage 4 (butterflies)
    let t8a = clip(t8 + t11);
    let t9 = clip(t9a + t10a);
    let t10 = clip(t9a - t10a);
    let t11a = clip(t8 - t11);
    let t12a = clip(t15 - t12);
    let t13 = clip(t14a - t13a);
    let t14 = clip(t14a + t13a);
    let t15a = clip(t15 + t12);
    // stage 5 (181/256 rotations)
    let t10a = ((t13 - t10) * 181 + 128) >> 8;
    let t13a = ((t13 + t10) * 181 + 128) >> 8;
    let t11 = ((t12a - t11a) * 181 + 128) >> 8;
    let t12 = ((t12a + t11a) * 181 + 128) >> 8;
    // even part (already transformed, in c at even positions)
    let (t0, t1, t2, t3) = (c[0], c[2 * s], c[4 * s], c[6 * s]);
    let (t4, t5, t6, t7) = (c[8 * s], c[10 * s], c[12 * s], c[14 * s]);
    c[0] = clip(t0 + t15a);
    c[s] = clip(t1 + t14);
    c[2 * s] = clip(t2 + t13a);
    c[3 * s] = clip(t3 + t12);
    c[4 * s] = clip(t4 + t11);
    c[5 * s] = clip(t5 + t10a);
    c[6 * s] = clip(t6 + t9);
    c[7 * s] = clip(t7 + t8a);
    c[8 * s] = clip(t7 - t8a);
    c[9 * s] = clip(t6 - t9);
    c[10 * s] = clip(t5 - t10a);
    c[11 * s] = clip(t4 - t11);
    c[12 * s] = clip(t3 - t12);
    c[13 * s] = clip(t2 - t13a);
    c[14 * s] = clip(t1 - t14);
    c[15 * s] = clip(t0 - t15a);
}

pub(crate) fn idct_dequant_16x16_scalar(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    let (rmin, rmax, cmin, cmax, cf_max) = (
        dequant.rmin,
        dequant.rmax,
        dequant.cmin,
        dequant.cmax,
        dequant.cf_max,
    );
    let mut coeff = [0i32; 256];
    for rc in 0..256 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = dequant.step(rc);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 256];
    for y in 0..16 {
        for x in 0..16 {
            tmp[y * 16 + x] = coeff[y + x * 16];
        }
    }
    for y in 0..16 {
        inv_dct16_1d(&mut tmp[y * 16..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 2) >> 2).clamp(cmin, cmax);
    }
    for x in 0..16 {
        inv_dct16_1d(&mut tmp[x..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d's exact integer 1-D inverse ADST16 (`inv_adst16_1d_internal_c`). Reads
/// all 16 inputs before writing, so safe in place.
pub(crate) fn inv_adst16_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    let (in0, in1, in2, in3) = (c[0], c[s], c[2 * s], c[3 * s]);
    let (in4, in5, in6, in7) = (c[4 * s], c[5 * s], c[6 * s], c[7 * s]);
    let (in8, in9, in10, in11) = (c[8 * s], c[9 * s], c[10 * s], c[11 * s]);
    let (in12, in13, in14, in15) = (c[12 * s], c[13 * s], c[14 * s], c[15 * s]);
    let mut t0 = ((in15 * (4091 - 4096) + in0 * 201 + 2048) >> 12) + in15;
    let mut t1 = ((in15 * 201 - in0 * (4091 - 4096) + 2048) >> 12) - in0;
    let mut t2 = ((in13 * (3973 - 4096) + in2 * 995 + 2048) >> 12) + in13;
    let mut t3 = ((in13 * 995 - in2 * (3973 - 4096) + 2048) >> 12) - in2;
    let mut t4 = ((in11 * (3703 - 4096) + in4 * 1751 + 2048) >> 12) + in11;
    let mut t5 = ((in11 * 1751 - in4 * (3703 - 4096) + 2048) >> 12) - in4;
    let mut t6 = (in9 * 1645 + in6 * 1220 + 1024) >> 11;
    let mut t7 = (in9 * 1220 - in6 * 1645 + 1024) >> 11;
    let mut t8 = ((in7 * 2751 + in8 * (3035 - 4096) + 2048) >> 12) + in8;
    let mut t9 = ((in7 * (3035 - 4096) - in8 * 2751 + 2048) >> 12) + in7;
    let mut t10 = ((in5 * 2106 + in10 * (3513 - 4096) + 2048) >> 12) + in10;
    let mut t11 = ((in5 * (3513 - 4096) - in10 * 2106 + 2048) >> 12) + in5;
    let mut t12 = ((in3 * 1380 + in12 * (3857 - 4096) + 2048) >> 12) + in12;
    let mut t13 = ((in3 * (3857 - 4096) - in12 * 1380 + 2048) >> 12) + in3;
    let mut t14 = ((in1 * 601 + in14 * (4052 - 4096) + 2048) >> 12) + in14;
    let mut t15 = ((in1 * (4052 - 4096) - in14 * 601 + 2048) >> 12) + in1;
    let t0a = clip(t0 + t8);
    let t1a = clip(t1 + t9);
    let mut t2a = clip(t2 + t10);
    let mut t3a = clip(t3 + t11);
    let mut t4a = clip(t4 + t12);
    let mut t5a = clip(t5 + t13);
    let mut t6a = clip(t6 + t14);
    let mut t7a = clip(t7 + t15);
    let mut t8a = clip(t0 - t8);
    let mut t9a = clip(t1 - t9);
    let mut t10a = clip(t2 - t10);
    let mut t11a = clip(t3 - t11);
    let mut t12a = clip(t4 - t12);
    let mut t13a = clip(t5 - t13);
    let mut t14a = clip(t6 - t14);
    let mut t15a = clip(t7 - t15);
    t8 = ((t8a * (4017 - 4096) + t9a * 799 + 2048) >> 12) + t8a;
    t9 = ((t8a * 799 - t9a * (4017 - 4096) + 2048) >> 12) - t9a;
    t10 = ((t10a * 2276 + t11a * (3406 - 4096) + 2048) >> 12) + t11a;
    t11 = ((t10a * (3406 - 4096) - t11a * 2276 + 2048) >> 12) + t10a;
    t12 = ((t13a * (4017 - 4096) - t12a * 799 + 2048) >> 12) + t13a;
    t13 = ((t13a * 799 + t12a * (4017 - 4096) + 2048) >> 12) + t12a;
    t14 = ((t15a * 2276 - t14a * (3406 - 4096) + 2048) >> 12) - t14a;
    t15 = ((t15a * (3406 - 4096) + t14a * 2276 + 2048) >> 12) + t15a;
    t0 = clip(t0a + t4a);
    t1 = clip(t1a + t5a);
    t2 = clip(t2a + t6a);
    t3 = clip(t3a + t7a);
    t4 = clip(t0a - t4a);
    t5 = clip(t1a - t5a);
    t6 = clip(t2a - t6a);
    t7 = clip(t3a - t7a);
    t8a = clip(t8 + t12);
    t9a = clip(t9 + t13);
    t10a = clip(t10 + t14);
    t11a = clip(t11 + t15);
    t12a = clip(t8 - t12);
    t13a = clip(t9 - t13);
    t14a = clip(t10 - t14);
    t15a = clip(t11 - t15);
    t4a = ((t4 * (3784 - 4096) + t5 * 1567 + 2048) >> 12) + t4;
    t5a = ((t4 * 1567 - t5 * (3784 - 4096) + 2048) >> 12) - t5;
    t6a = ((t7 * (3784 - 4096) - t6 * 1567 + 2048) >> 12) + t7;
    t7a = ((t7 * 1567 + t6 * (3784 - 4096) + 2048) >> 12) + t6;
    t12 = ((t12a * (3784 - 4096) + t13a * 1567 + 2048) >> 12) + t12a;
    t13 = ((t12a * 1567 - t13a * (3784 - 4096) + 2048) >> 12) - t13a;
    t14 = ((t15a * (3784 - 4096) - t14a * 1567 + 2048) >> 12) + t15a;
    t15 = ((t15a * 1567 + t14a * (3784 - 4096) + 2048) >> 12) + t14a;
    c[0] = clip(t0 + t2);
    c[15 * s] = -clip(t1 + t3);
    t2a = clip(t0 - t2);
    t3a = clip(t1 - t3);
    c[3 * s] = -clip(t4a + t6a);
    c[12 * s] = clip(t5a + t7a);
    t6 = clip(t4a - t6a);
    t7 = clip(t5a - t7a);
    c[s] = -clip(t8a + t10a);
    c[14 * s] = clip(t9a + t11a);
    t10 = clip(t8a - t10a);
    t11 = clip(t9a - t11a);
    c[2 * s] = clip(t12 + t14);
    c[13 * s] = -clip(t13 + t15);
    t14a = clip(t12 - t14);
    t15a = clip(t13 - t15);
    c[7 * s] = -(((t2a + t3a) * 181 + 128) >> 8);
    c[8 * s] = ((t2a - t3a) * 181 + 128) >> 8;
    c[4 * s] = ((t6 + t7) * 181 + 128) >> 8;
    c[11 * s] = -(((t6 - t7) * 181 + 128) >> 8);
    c[6 * s] = ((t10 + t11) * 181 + 128) >> 8;
    c[9 * s] = -(((t10 - t11) * 181 + 128) >> 8);
    c[5 * s] = -(((t14a + t15a) * 181 + 128) >> 8);
    c[10 * s] = ((t14a - t15a) * 181 + 128) >> 8;
}

type Inv16ScalarKernel = fn(&mut [i32], usize, i32, i32);

#[inline]
fn inv16x16_dequant_scalar(
    levels: &[i32; 256],
    dequant: &IdctDequant,
    row_pass: Inv16ScalarKernel,
    col_pass: Inv16ScalarKernel,
) -> [i32; 256] {
    let (rmin, rmax, cmin, cmax, cf_max) = (
        dequant.rmin,
        dequant.rmax,
        dequant.cmin,
        dequant.cmax,
        dequant.cf_max,
    );
    let mut coeff = [0i32; 256];
    for rc in 0..256 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = dequant.step(rc);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 256];
    for y in 0..16 {
        for x in 0..16 {
            tmp[y * 16 + x] = coeff[y + x * 16];
        }
    }
    for y in 0..16 {
        row_pass(&mut tmp[y * 16..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 2) >> 2).clamp(cmin, cmax);
    }
    for x in 0..16 {
        col_pass(&mut tmp[x..], 16, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iadstdct_dequant_16x16_scalar(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    inv16x16_dequant_scalar(levels, dequant, inv_dct16_1d, inv_adst16_1d)
}

pub(crate) fn idctadst_dequant_16x16_scalar(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    inv16x16_dequant_scalar(levels, dequant, inv_adst16_1d, inv_dct16_1d)
}

pub(crate) fn iadst_dequant_16x16_scalar(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    inv16x16_dequant_scalar(levels, dequant, inv_adst16_1d, inv_adst16_1d)
}

pub(crate) fn inv_dct32_1d(c: &mut [i32], s: usize, min: i32, max: i32) {
    let clip = |x: i32| x.clamp(min, max);
    inv_dct16_1d(c, 2 * s, min, max); // even positions c[0],c[2s],..,c[30s]

    let (in1, in3, in5, in7) = (c[s], c[3 * s], c[5 * s], c[7 * s]);
    let (in9, in11, in13, in15) = (c[9 * s], c[11 * s], c[13 * s], c[15 * s]);
    let (in17, in19, in21, in23) = (c[17 * s], c[19 * s], c[21 * s], c[23 * s]);
    let (in25, in27, in29, in31) = (c[25 * s], c[27 * s], c[29 * s], c[31 * s]);

    // stage 1
    let mut t16a = ((in1 * 201 - in31 * (4091 - 4096) + 2048) >> 12) - in31;
    let mut t17a = ((in17 * (3035 - 4096) - in15 * 2751 + 2048) >> 12) + in17;
    let mut t18a = ((in9 * 1751 - in23 * (3703 - 4096) + 2048) >> 12) - in23;
    let mut t19a = ((in25 * (3857 - 4096) - in7 * 1380 + 2048) >> 12) + in25;
    let mut t20a = ((in5 * 995 - in27 * (3973 - 4096) + 2048) >> 12) - in27;
    let mut t21a = ((in21 * (3513 - 4096) - in11 * 2106 + 2048) >> 12) + in21;
    let mut t22a = (in13 * 1220 - in19 * 1645 + 1024) >> 11;
    let mut t23a = ((in29 * (4052 - 4096) - in3 * 601 + 2048) >> 12) + in29;
    let mut t24a = ((in29 * 601 + in3 * (4052 - 4096) + 2048) >> 12) + in3;
    let mut t25a = (in13 * 1645 + in19 * 1220 + 1024) >> 11;
    let mut t26a = ((in21 * 2106 + in11 * (3513 - 4096) + 2048) >> 12) + in11;
    let mut t27a = ((in5 * (3973 - 4096) + in27 * 995 + 2048) >> 12) + in5;
    let mut t28a = ((in25 * 1380 + in7 * (3857 - 4096) + 2048) >> 12) + in7;
    let mut t29a = ((in9 * (3703 - 4096) + in23 * 1751 + 2048) >> 12) + in9;
    let mut t30a = ((in17 * 2751 + in15 * (3035 - 4096) + 2048) >> 12) + in15;
    let mut t31a = ((in1 * (4091 - 4096) + in31 * 201 + 2048) >> 12) + in1;

    // stage 2
    let mut t16 = clip(t16a + t17a);
    let mut t17 = clip(t16a - t17a);
    let mut t18 = clip(t19a - t18a);
    let mut t19 = clip(t19a + t18a);
    let mut t20 = clip(t20a + t21a);
    let mut t21 = clip(t20a - t21a);
    let mut t22 = clip(t23a - t22a);
    let mut t23 = clip(t23a + t22a);
    let mut t24 = clip(t24a + t25a);
    let mut t25 = clip(t24a - t25a);
    let mut t26 = clip(t27a - t26a);
    let mut t27 = clip(t27a + t26a);
    let mut t28 = clip(t28a + t29a);
    let mut t29 = clip(t28a - t29a);
    let mut t30 = clip(t31a - t30a);
    let mut t31 = clip(t31a + t30a);

    // stage 3
    t17a = ((t30 * 799 - t17 * (4017 - 4096) + 2048) >> 12) - t17;
    t30a = ((t30 * (4017 - 4096) + t17 * 799 + 2048) >> 12) + t30;
    t18a = ((-(t29 * (4017 - 4096) + t18 * 799) + 2048) >> 12) - t29;
    t29a = ((t29 * 799 - t18 * (4017 - 4096) + 2048) >> 12) - t18;
    t21a = (t26 * 1703 - t21 * 1138 + 1024) >> 11;
    t26a = (t26 * 1138 + t21 * 1703 + 1024) >> 11;
    t22a = (-(t25 * 1138 + t22 * 1703) + 1024) >> 11;
    t25a = (t25 * 1703 - t22 * 1138 + 1024) >> 11;

    // stage 4
    t16a = clip(t16 + t19);
    t17 = clip(t17a + t18a);
    t18 = clip(t17a - t18a);
    t19a = clip(t16 - t19);
    t20a = clip(t23 - t20);
    t21 = clip(t22a - t21a);
    t22 = clip(t22a + t21a);
    t23a = clip(t23 + t20);
    t24a = clip(t24 + t27);
    t25 = clip(t25a + t26a);
    t26 = clip(t25a - t26a);
    t27a = clip(t24 - t27);
    t28a = clip(t31 - t28);
    t29 = clip(t30a - t29a);
    t30 = clip(t30a + t29a);
    t31a = clip(t31 + t28);

    // stage 5
    t18a = ((t29 * 1567 - t18 * (3784 - 4096) + 2048) >> 12) - t18;
    t29a = ((t29 * (3784 - 4096) + t18 * 1567 + 2048) >> 12) + t29;
    t19 = ((t28a * 1567 - t19a * (3784 - 4096) + 2048) >> 12) - t19a;
    t28 = ((t28a * (3784 - 4096) + t19a * 1567 + 2048) >> 12) + t28a;
    t20 = ((-(t27a * (3784 - 4096) + t20a * 1567) + 2048) >> 12) - t27a;
    t27 = ((t27a * 1567 - t20a * (3784 - 4096) + 2048) >> 12) - t20a;
    t21a = ((-(t26 * (3784 - 4096) + t21 * 1567) + 2048) >> 12) - t26;
    t26a = ((t26 * 1567 - t21 * (3784 - 4096) + 2048) >> 12) - t21;

    // stage 6
    t16 = clip(t16a + t23a);
    t17a = clip(t17 + t22);
    t18 = clip(t18a + t21a);
    t19a = clip(t19 + t20);
    t20a = clip(t19 - t20);
    t21 = clip(t18a - t21a);
    t22a = clip(t17 - t22);
    t23 = clip(t16a - t23a);
    t24 = clip(t31a - t24a);
    t25a = clip(t30 - t25);
    t26 = clip(t29a - t26a);
    t27a = clip(t28 - t27);
    t28a = clip(t28 + t27);
    t29 = clip(t29a + t26a);
    t30a = clip(t30 + t25);
    t31 = clip(t31a + t24a);

    // stage 7 (181/256 rotations)
    t20 = ((t27a - t20a) * 181 + 128) >> 8;
    t27 = ((t27a + t20a) * 181 + 128) >> 8;
    t21a = ((t26 - t21) * 181 + 128) >> 8;
    t26a = ((t26 + t21) * 181 + 128) >> 8;
    t22 = ((t25a - t22a) * 181 + 128) >> 8;
    t25 = ((t25a + t22a) * 181 + 128) >> 8;
    t23a = ((t24 - t23) * 181 + 128) >> 8;
    t24a = ((t24 + t23) * 181 + 128) >> 8;

    // even results (in c at positions 0,2s,..,30s)
    let (t0, t1, t2, t3) = (c[0], c[2 * s], c[4 * s], c[6 * s]);
    let (t4, t5, t6, t7) = (c[8 * s], c[10 * s], c[12 * s], c[14 * s]);
    let (t8, t9, t10, t11) = (c[16 * s], c[18 * s], c[20 * s], c[22 * s]);
    let (t12, t13, t14, t15) = (c[24 * s], c[26 * s], c[28 * s], c[30 * s]);

    c[0] = clip(t0 + t31);
    c[s] = clip(t1 + t30a);
    c[2 * s] = clip(t2 + t29);
    c[3 * s] = clip(t3 + t28a);
    c[4 * s] = clip(t4 + t27);
    c[5 * s] = clip(t5 + t26a);
    c[6 * s] = clip(t6 + t25);
    c[7 * s] = clip(t7 + t24a);
    c[8 * s] = clip(t8 + t23a);
    c[9 * s] = clip(t9 + t22);
    c[10 * s] = clip(t10 + t21a);
    c[11 * s] = clip(t11 + t20);
    c[12 * s] = clip(t12 + t19a);
    c[13 * s] = clip(t13 + t18);
    c[14 * s] = clip(t14 + t17a);
    c[15 * s] = clip(t15 + t16);
    c[16 * s] = clip(t15 - t16);
    c[17 * s] = clip(t14 - t17a);
    c[18 * s] = clip(t13 - t18);
    c[19 * s] = clip(t12 - t19a);
    c[20 * s] = clip(t11 - t20);
    c[21 * s] = clip(t10 - t21a);
    c[22 * s] = clip(t9 - t22);
    c[23 * s] = clip(t8 - t23a);
    c[24 * s] = clip(t7 - t24a);
    c[25 * s] = clip(t6 - t25);
    c[26 * s] = clip(t5 - t26a);
    c[27 * s] = clip(t4 - t27);
    c[28 * s] = clip(t3 - t28a);
    c[29 * s] = clip(t2 - t29);
    c[30 * s] = clip(t1 - t30a);
    c[31 * s] = clip(t0 - t31);
}

pub(crate) fn idct_dequant_32x32_scalar(
    levels: &[i32; 1024],
    dequant: &IdctDequant,
) -> [i32; 1024] {
    let (rmin, rmax, cmin, cmax, cf_max) = (
        dequant.rmin,
        dequant.rmax,
        dequant.cmin,
        dequant.cmax,
        dequant.cf_max,
    );
    let mut coeff = [0i32; 1024];
    for rc in 0..1024 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = dequant.step(rc);
        // mask to 24 bits, then dq_shift = 1, then clamp to cf_max
        let mag = (((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) >> 1) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 1024];
    for y in 0..32 {
        for x in 0..32 {
            tmp[y * 32 + x] = coeff[y + x * 32];
        }
    }
    for y in 0..32 {
        inv_dct32_1d(&mut tmp[y * 32..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = ((*t + 2) >> 2).clamp(cmin, cmax);
    }
    for x in 0..32 {
        inv_dct32_1d(&mut tmp[x..], 32, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iidentity_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (_rmin, _rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let q = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    for t in tmp.iter_mut() {
        let i = *t;
        *t = i + ((i * 1697 + 2048) >> 12);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for t in tmp.iter_mut() {
        let i = *t;
        *t = i + ((i * 1697 + 2048) >> 12);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d's exact TX_4X4 V_DCT inverse: identity4 rows (x sqrt2 via the 1697
/// formula), the shift=0 inter-pass clamp, dct4 columns, final `(t+8)>>4`.
pub(crate) fn ivdct_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (_rmin, _rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let qd = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * qd as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    for t in tmp.iter_mut() {
        let i = *t;
        *t = i + ((i * 1697 + 2048) >> 12);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_dct4_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// dav1d's exact TX_4X4 H_DCT inverse: dct4 rows, the shift=0 inter-pass
/// clamp, identity4 columns, final `(t+8)>>4`.
pub(crate) fn ihdct_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let q = &crate::quant::FlatDct(q); // identity family: QM not applied
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let qd = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * qd as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    for row in 0..4 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for t in tmp.iter_mut() {
        let i = *t;
        *t = i + ((i * 1697 + 2048) >> 12);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn idct_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let q = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        // c[x] = coeff[y + x*4]; here rc = fx*4 + fy => place at tmp[fy*4+fx]
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    // row transform: inv_dct4 (stride 1) over each of the 4 rows; shift=0
    for row in 0..4 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    // column transform: inv_dct4 (stride 4) over each of the 4 columns
    for col in 0..4 {
        inv_dct4_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn inv_adst4_1d(c: &mut [i32], s: usize, _min: i32, _max: i32) {
    let in0 = c[0];
    let in1 = c[s];
    let in2 = c[2 * s];
    let in3 = c[3 * s];
    let o0 =
        ((1321 * in0 + (3803 - 4096) * in2 + (2482 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in2
            + in3
            + in1;
    let o1 =
        (((2482 - 4096) * in0 - 1321 * in2 - (3803 - 4096) * in3 + (3344 - 4096) * in1 + 2048)
            >> 12)
            + in0
            - in3
            + in1;
    let o2 = (209 * (in0 - in2 + in3) + 128) >> 8;
    let o3 = (((3803 - 4096) * in0 + (2482 - 4096) * in2 - 1321 * in3 - (3344 - 4096) * in1
        + 2048)
        >> 12)
        + in0
        + in2
        - in1;
    c[0] = o0;
    c[s] = o1;
    c[2 * s] = o2;
    c[3 * s] = o3;
}

pub(crate) fn iadst_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let q = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    for row in 0..4 {
        inv_adst4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_adst4_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iadstdct_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let q = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    for row in 0..4 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_adst4_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

/// Inverse DCT_ADST 4x4 (AV1 `DCT_ADST`): inv_adst on rows (first), inv_dct on
/// columns (second). Pairs with the `dctadst4x4` forward.
pub(crate) fn idctadst_dequant_4x4(levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut tmp = [0i32; 16];
    for (rc, &lvl) in levels.iter().enumerate() {
        let q = q.dequant_step(rc, 4, 4);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        let coeff = if lvl < 0 { -mag } else { mag };
        let (fx, fy) = (rc / 4, rc % 4);
        tmp[fy * 4 + fx] = coeff;
    }
    for row in 0..4 {
        inv_adst4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_dct4_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn iadstdct_dequant_4x8(levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 32];
    for rc in 0..32 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 4, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 32];
    for row in 0..8 {
        for col in 0..4 {
            tmp[row * 4 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    for row in 0..8 {
        inv_dct4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_adst8_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

pub(crate) fn idctadst_dequant_4x8(levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 32];
    for rc in 0..32 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 4, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 32];
    for row in 0..8 {
        for col in 0..4 {
            tmp[row * 4 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    for row in 0..8 {
        inv_adst4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_dct8_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

#[cfg(test)]
mod adst4_tests {
    use super::*;
    use crate::dct::adst4x4_t;
    use crate::quant::Quant;

    // Forward ADST4 then inverse ADST4 must reconstruct a residual closely
    // (exactly at high quality where quant is near-lossless), confirming the
    // dav1d-sourced inverse and the derived forward are a consistent,
    // correctly-scaled pair.
    #[test]
    fn adst4_roundtrip_highq() {
        // High quality (low base_q_idx) -> small quant step -> near-lossless.
        let q = Quant::new_chroma(8, 8);
        let residuals: [[i32; 16]; 3] = [
            [
                40, -32, 20, -12, 28, -24, 16, -8, 36, -28, 24, -4, 32, -20, 12, -16,
            ],
            [
                80, 70, -40, 30, -60, 50, 20, -10, 44, -36, 24, -16, 32, -28, 12, -4,
            ],
            [
                -50, 48, -46, 44, 42, -40, 38, -36, 34, 32, -30, 28, 26, -24, 22, -20,
            ],
        ];
        for res in &residuals {
            let (levels, _t) = adst4x4_t(res, &q);
            let rec = iadst_dequant_4x4(&levels, &q);
            // Energy-preserving check: reconstruction must track the input
            // (mean-abs error small relative to signal magnitude).
            let mut err = 0i64;
            let mut sig = 0i64;
            for i in 0..16 {
                err += (rec[i] - res[i]).unsigned_abs() as i64;
                sig += res[i].unsigned_abs() as i64;
            }
            assert!(
                err * 4 <= sig + 16,
                "ADST4 round-trip error too large: err={err} sig={sig} rec={rec:?} res={res:?}"
            );
        }
    }

    // The inverse must be deterministic and stride-correct: applying it on a
    // single nonzero DC coefficient yields a flat-ish block (no NaNs/overflow).
    #[test]
    fn adst4_dc_only_is_finite() {
        let q = Quant::new_chroma(40, 8);
        let mut levels = [0i32; 16];
        levels[0] = 12;
        let rec = iadst_dequant_4x4(&levels, &q);
        // all values within a sane residual range
        for &v in &rec {
            assert!(v.abs() < 4096, "ADST4 DC inverse out of range: {v}");
        }
    }

    // ADST_DCT and DCT_ADST 4x4 must also round-trip (forward then inverse
    // tracks the input), confirming the mixed-kernel pairs are scale-consistent.
    #[test]
    fn adstdct_dctadst_4x4_roundtrip() {
        use crate::dct::{adstdct4x4_t, dctadst4x4_t};
        let q = Quant::new_chroma(8, 8);
        let res: [i32; 16] = [
            40, -32, 20, -12, 28, -24, 16, -8, 36, -28, 24, -4, 32, -20, 12, -16,
        ];
        let check = |rec: [i32; 16], tag: &str| {
            let mut err = 0i64;
            let mut sig = 0i64;
            for i in 0..16 {
                err += (rec[i] - res[i]).unsigned_abs() as i64;
                sig += res[i].unsigned_abs() as i64;
            }
            assert!(
                err * 4 <= sig + 16,
                "{tag} 4x4 round-trip error too large: err={err} sig={sig}"
            );
        };
        let (l1, _) = adstdct4x4_t(&res, &q);
        check(iadstdct_dequant_4x4(&l1, &q), "ADST_DCT");
        let (l2, _) = dctadst4x4_t(&res, &q);
        check(idctadst_dequant_4x4(&l2, &q), "DCT_ADST");
    }
}

pub(crate) fn iadst_dequant_4x8(levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    let (rmin, rmax, cmin, cmax, cf_max) = q.clips();
    let mut coeff = [0i32; 32];
    for rc in 0..32 {
        let lvl = levels[rc];
        if lvl == 0 {
            continue;
        }
        let q = q.dequant_step(rc, 4, 8);
        let mag = ((lvl.unsigned_abs() as u64 * q as u64) & 0xff_ffff) as i32;
        let mag = mag.min(cf_max + (lvl < 0) as i32);
        coeff[rc] = if lvl < 0 { -mag } else { mag };
    }
    let mut tmp = [0i32; 32];
    for row in 0..8 {
        for col in 0..4 {
            tmp[row * 4 + col] = (coeff[row + col * 8] * 181 + 128) >> 8;
        }
    }
    for row in 0..8 {
        inv_adst4_1d(&mut tmp[row * 4..], 1, rmin, rmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t).clamp(cmin, cmax);
    }
    for col in 0..4 {
        inv_adst8_1d(&mut tmp[col..], 4, cmin, cmax);
    }
    for t in tmp.iter_mut() {
        *t = (*t + 8) >> 4;
    }
    tmp
}

#[cfg(all(
    test,
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "avx"
))]
mod avx2_itx_tests {
    use super::*;

    fn dequant() -> IdctDequant {
        IdctDequant {
            dc_q: 77,
            ac_q: 91,
            rmin: -(1 << 19),
            rmax: (1 << 19) - 1,
            cmin: -(1 << 15),
            cmax: (1 << 15) - 1,
            cf_max: (1 << 22) - 1,
            qm: None,
        }
    }

    fn dequant_8bit() -> IdctDequant {
        IdctDequant {
            dc_q: 77,
            ac_q: 91,
            rmin: i16::MIN as i32,
            rmax: i16::MAX as i32,
            cmin: i16::MIN as i32,
            cmax: i16::MAX as i32,
            cf_max: i16::MAX as i32,
            qm: None,
        }
    }

    fn fill_levels<const N: usize>() -> [i32; N] {
        let mut out = [0i32; N];
        let mut s = 0x1234_5678u32 ^ N as u32;
        for (i, v) in out.iter_mut().enumerate() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let level = ((s >> 24) as i32 % 63) - 31;
            *v = if i % 11 == 0 { 0 } else { level };
        }
        out[0] = 23;
        out
    }

    #[test]
    fn idct_8x8_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let levels = fill_levels::<64>();
        for dq in [dequant(), dequant_8bit()] {
            let scalar = idct_dequant_8x8_scalar(&levels, &dq);
            let avx2 = unsafe { crate::avx::idct_dequant_8x8_avx2(&levels, &dq) };
            assert_eq!(scalar, avx2);
        }
    }

    #[test]
    fn idct_16x16_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let levels = fill_levels::<256>();
        for dq in [dequant(), dequant_8bit()] {
            let scalar = idct_dequant_16x16_scalar(&levels, &dq);
            let avx2 = unsafe { crate::avx::idct_dequant_16x16_avx2(&levels, &dq) };
            assert_eq!(scalar, avx2);
        }
    }

    #[test]
    fn idct_32x32_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let levels = fill_levels::<1024>();
        for dq in [dequant(), dequant_8bit()] {
            let scalar = idct_dequant_32x32_scalar(&levels, &dq);
            let avx2 = unsafe { crate::avx::idct_dequant_32x32_avx2(&levels, &dq) };
            assert_eq!(scalar, avx2);
        }
    }
}

#[cfg(all(test, target_arch = "aarch64", feature = "neon"))]
mod neon_itx_tests {
    use super::*;

    fn dequant() -> IdctDequant {
        IdctDequant {
            dc_q: 77,
            ac_q: 91,
            rmin: -(1 << 19),
            rmax: (1 << 19) - 1,
            cmin: -(1 << 15),
            cmax: (1 << 15) - 1,
            cf_max: (1 << 22) - 1,
            qm: None,
        }
    }

    fn fill_levels<const N: usize>() -> [i32; N] {
        let mut out = [0i32; N];
        let mut s = 0x8765_4321u32 ^ N as u32;
        for (i, v) in out.iter_mut().enumerate() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let level = ((s >> 24) as i32 % 63) - 31;
            *v = if i % 13 == 0 { 0 } else { level };
        }
        out[0] = -19;
        out
    }

    #[test]
    fn idct_8x8_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let dq = dequant();
        let levels = fill_levels::<64>();
        let scalar = idct_dequant_8x8_scalar(&levels, &dq);
        let neon = unsafe { crate::neon::idct_dequant_8x8_neon(&levels, &dq) };
        assert_eq!(scalar, neon);
    }

    #[test]
    fn idct_16x16_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let dq = dequant();
        let levels = fill_levels::<256>();
        let scalar = idct_dequant_16x16_scalar(&levels, &dq);
        let neon = unsafe { crate::neon::idct_dequant_16x16_neon(&levels, &dq) };
        assert_eq!(scalar, neon);
    }

    #[test]
    fn idct_32x32_neon_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let dq = dequant();
        let levels = fill_levels::<1024>();
        let scalar = idct_dequant_32x32_scalar(&levels, &dq);
        let neon = unsafe { crate::neon::idct_dequant_32x32_neon(&levels, &dq) };
        assert_eq!(scalar, neon);
    }
}

#[cfg(test)]
mod vh_dct_tests {
    use super::*;
    use crate::dct::{fhdct8x8_t, fvdct8x8_t};
    use crate::quant::Quant;

    #[test]
    fn vh_dct_8x8_roundtrip_recovers_residual() {
        let q = Quant::new(40, 8);
        // Structured residuals each transform should like: a vertical edge
        // (H_DCT-friendly), a horizontal edge (V_DCT-friendly).
        let mut vert_edge = [0i32; 64];
        let mut horz_edge = [0i32; 64];
        for y in 0..8 {
            for x in 0..8 {
                vert_edge[y * 8 + x] = if x < 4 { 60 } else { -60 };
                horz_edge[y * 8 + x] = if y < 4 { 60 } else { -60 };
            }
        }
        for (name, resid, fwd, inv) in [
            (
                "hdct",
                &vert_edge,
                fhdct8x8_t as fn(&[i32; 64], &Quant) -> ([i32; 64], [f32; 64]),
                ihdct_dequant_8x8 as fn(&[i32; 64], &Quant) -> [i32; 64],
            ),
            ("vdct", &horz_edge, fvdct8x8_t, ivdct_dequant_8x8),
        ] {
            let (cf, _tf) = fwd(resid, &q);
            assert!(cf.iter().any(|&c| c != 0), "{name}: no levels coded");
            let rr = inv(&cf, &q);
            let mut err = 0i64;
            for i in 0..64 {
                err += (rr[i] - resid[i]).pow(2) as i64;
            }
            let mse = err as f64 / 64.0;
            // Quantization error only: must be well under one step^2.
            let step = q.ac_q() as f64;
            assert!(
                mse < step * step,
                "{name}: mse {mse} vs step^2 {}",
                step * step
            );
        }
    }

    // Every new rect16 forward/inverse pair must reconstruct a structured
    // residual to within quantization error — this pins the derived forward
    // scales (identity 1/8 gain, 2*sqrt2 1-D factor, hybrid compositions)
    // against the dav1d-shaped inverse pipeline.
    #[test]
    fn rect16_seven_type_roundtrips() {
        use crate::dct::{
            adstdct8x16_t, adstdct16x8_t, dctadst8x16_t, dctadst16x8_t, fhdct8x16_t, fhdct16x8_t,
            fidentity_rect_t, fvdct8x16_t, fvdct16x8_t,
        };
        let q = Quant::new(40, 8);
        // Mixed-structure residual: edges both ways plus a gradient, exercised
        // through every kernel (each must round-trip regardless of preference).
        let mut resid = [0i32; 128];
        for wsel in [16usize, 8] {
            let h = 128 / wsel;
            for y in 0..h {
                for x in 0..wsel {
                    resid[y * wsel + x] =
                        (if x < wsel / 2 { 40 } else { -40 }) + (y as i32 - h as i32 / 2) * 6;
                }
            }
            type F = fn(&[i32; 128], &Quant) -> ([i32; 128], [f32; 128]);
            type I = fn(&[i32; 128], &Quant) -> [i32; 128];
            let pairs: Vec<(&str, F, I)> = if wsel == 16 {
                vec![
                    (
                        "adstdct16x8",
                        adstdct16x8_t as F,
                        iadstdct_dequant_16x8 as I,
                    ),
                    ("dctadst16x8", dctadst16x8_t, idctadst_dequant_16x8),
                    ("vdct16x8", fvdct16x8_t, ivdct_dequant_16x8),
                    ("hdct16x8", fhdct16x8_t, ihdct_dequant_16x8),
                ]
            } else {
                vec![
                    (
                        "adstdct8x16",
                        adstdct8x16_t as F,
                        iadstdct_dequant_8x16 as I,
                    ),
                    ("dctadst8x16", dctadst8x16_t, idctadst_dequant_8x16),
                    ("vdct8x16", fvdct8x16_t, ivdct_dequant_8x16),
                    ("hdct8x16", fhdct8x16_t, ihdct_dequant_8x16),
                ]
            };
            for (name, fwd, inv) in pairs {
                let (cf, _tf) = fwd(&resid, &q);
                assert!(cf.iter().any(|&c| c != 0), "{name}: no levels coded");
                let rr = inv(&cf, &q);
                let mut err = 0i64;
                for i in 0..128 {
                    err += (rr[i] - resid[i]).pow(2) as i64;
                }
                let mse = err as f64 / 128.0;
                let step = q.ac_q() as f64;
                assert!(
                    mse < step * step,
                    "{name}: mse {mse} vs step^2 {}",
                    step * step
                );
            }
            // IDTX pair.
            let (cf, _tf) = fidentity_rect_t(&resid, &q, wsel);
            assert!(cf.iter().any(|&c| c != 0), "idtx{wsel}: no levels coded");
            let rr = if wsel == 16 {
                iidtx_dequant_16x8(&cf, &q)
            } else {
                iidtx_dequant_8x16(&cf, &q)
            };
            let mut err = 0i64;
            for i in 0..128 {
                err += (rr[i] - resid[i]).pow(2) as i64;
            }
            let mse = err as f64 / 128.0;
            let step = q.ac_q() as f64;
            assert!(mse < step * step, "idtx{wsel}: mse {mse}");
        }
    }
}
