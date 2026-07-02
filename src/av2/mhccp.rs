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

//! Multi-Hypothesis Cross-Component Prediction (MHCCP)

pub(crate) const MHCCP_NUM_PARAMS: usize = 3;
pub(crate) const MHCCP_DECIM_BITS: i32 = 16;
pub(crate) const MHCCP_MODE_NUM: usize = 3; // number of filter directions (mh_dir)

const DIV_PREC_BITS: i32 = 14;
const DIV_PREC_BITS_POW2: i32 = 8;
const DIV_SLOT_BITS: i32 = 3;
const DIV_INTR_BITS: i32 = DIV_PREC_BITS - DIV_SLOT_BITS;

/// avm `size_group_lookup[BLOCK_SIZES_ALL]` — selects the `filter_dir_cdf`
/// context group. Indexed by block-size enum. Kept here so the RD/entropy code
/// can obtain the mh_dir context without pulling in the full block-size tables.
pub(crate) static SIZE_GROUP_LOOKUP: [u8; 29] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2,
];

/// avm default `cfl_mhccp_switch_cdf` split point (probability of the "use
/// MHCCP" symbol). The entropy-side ICDF/PARA live in `cfl.rs`.
pub(crate) const _CFL_MHCCP_SWITCH_CDF: u16 = 15499;

/// avm `NON_LINEAR(V, M, BD) = (V*V + M) >> BD`. `v` is a Q0 luma value.
#[inline(always)]
pub(crate) fn non_linear(v: i32, mid: i32, bd: i32) -> i32 {
    (v.wrapping_mul(v) + mid) >> bd
}

/// avm `ilog2_32`: index of the highest set bit (`31 - clz(x|1)`).
#[inline(always)]
fn ilog2_32(x: u32) -> i32 {
    31 - (x | 1).leading_zeros() as i32
}

/// avm `avm_ceil_log2(n)`: `n < 2 -> 0`, else `msb(n-1) + 1`.
#[inline(always)]
fn ceil_log2(n: i32) -> i32 {
    if n < 2 {
        0
    } else {
        // get_msb(n-1) = 31 ^ clz(n-1)
        let m = (n - 1) as u32;
        (31 - m.leading_zeros() as i32) + 1
    }
}

/// avm `floorLog2Uint64`.
#[inline(always)]
fn floor_log2_u64(x: u64) -> i32 {
    if x == 0 {
        return 0;
    }
    63 - x.leading_zeros() as i32
}

/// avm `mul_fixed32_adapt`: overflow-safe fixed-point multiply `(a*b) >> shift`
/// with symmetric rounding, dropping bits from the wider operand first.
#[inline(always)]
pub(crate) fn mul_fixed32_adapt(a: i32, b: i32, shift: i32) -> i32 {
    let ua = a.unsigned_abs();
    let ub = b.unsigned_abs();
    let bits_a = ilog2_32(ua) + 1;
    let bits_b = ilog2_32(ub) + 1;
    let bits_limit = 29;

    let mut need = bits_a + bits_b - bits_limit;
    if need < 0 {
        need = 0;
    }
    let s1 = need >> 1;
    let s2 = need - s1;
    let adj = shift - (s1 + s2);
    let a_sh = if s1 != 0 { a >> s1 } else { a };
    let b_sh = if s2 != 0 { b >> s2 } else { b };
    let prod = a_sh.wrapping_mul(b_sh);
    if adj <= 0 {
        return prod;
    }
    let bias: u32 = if adj <= bits_limit {
        1u32 << (adj - 1)
    } else {
        0
    };
    if prod >= 0 {
        if adj <= bits_limit {
            (((prod as u32).wrapping_add(bias)) >> adj) as i32
        } else {
            0
        }
    } else if adj <= bits_limit {
        -((((-prod) as u32).wrapping_add(bias) >> adj) as i32)
    } else {
        0
    }
}

/// avm `get_division_scale_shift`: piecewise-quadratic reciprocal approximation.
/// Returns `(scale, round, shift)`; only `scale`/`shift` are used downstream.
fn division_scale_shift(denom: u32) -> (i32, i32, i32) {
    static POW2W: [i32; 8] = [214, 153, 113, 86, 67, 53, 43, 35];
    static POW2O: [i32; 8] = [4822, 5952, 6624, 6792, 6408, 5424, 3792, 1466];
    static POW2B: [i32; 8] = [12784, 12054, 11670, 11583, 11764, 12195, 12870, 13782];

    let shift = floor_log2_u64(denom as u64);
    let round = if shift == 0 {
        0
    } else {
        ((1u32 << shift) >> 1) as i32
    };

    let delta = shift - DIV_PREC_BITS;
    let norm_diff_tmp: i32 = if delta >= 0 {
        let bias: u32 = if delta > 0 { 1u32 << (delta - 1) } else { 0 };
        ((denom.wrapping_add(bias)) >> delta) as i32
    } else {
        let s = -delta;
        (denom << s) as i32
    };

    let hi = (1i32 << (DIV_PREC_BITS + 1)) - 1;
    let norm_diff_clip = norm_diff_tmp.clamp(1, hi);
    let norm_diff = norm_diff_clip & ((1 << DIV_PREC_BITS) - 1);
    let index = (norm_diff >> DIV_INTR_BITS) as usize;
    let norm_diff2 = norm_diff - POW2O[index];

    let mut scale = ((POW2W[index] * ((norm_diff2 * norm_diff2) >> DIV_PREC_BITS))
        >> DIV_PREC_BITS_POW2)
        - (norm_diff2 >> 1)
        + POW2B[index];
    scale <<= MHCCP_DECIM_BITS - DIV_PREC_BITS;
    (scale, round, shift)
}

/// avm `gauss_back_substitute`.
fn gauss_back_substitute(
    x: &mut [i32; MHCCP_NUM_PARAMS],
    c: &[[i32; MHCCP_NUM_PARAMS + 1]; MHCCP_NUM_PARAMS],
    num_eq: usize,
    col: usize,
    bits: i32,
) {
    x[num_eq - 1] = c[num_eq - 1][col];
    for i in (0..num_eq - 1).rev() {
        x[i] = c[i][col];
        for j in i + 1..num_eq {
            x[i] -= mul_fixed32_adapt(c[i][j], x[j], bits);
        }
    }
}

/// avm `gauss_elimination_mhccp`: solve `(A + reg*I) p = y` in fixed point.
/// `a` is the upper-triangular auto-correlation (as filled by the caller),
/// `y` the cross-correlation, `bd` the bit depth. Writes params into `out`.
pub(crate) fn gauss_elimination(
    a: &[[i32; MHCCP_NUM_PARAMS]; MHCCP_NUM_PARAMS],
    y: &[i32; MHCCP_NUM_PARAMS],
    bd: i32,
    out: &mut [i32; MHCCP_NUM_PARAMS],
) {
    let num_eq = MHCCP_NUM_PARAMS;
    let col_chr0 = num_eq;
    let reg = 2 << (bd - 8);
    let decim_bits = MHCCP_DECIM_BITS;

    let mut c = [[0i32; MHCCP_NUM_PARAMS + 1]; MHCCP_NUM_PARAMS];
    for i in 0..num_eq {
        for j in 0..num_eq {
            c[i][j] = if j >= i { a[i][j] } else { a[j][i] };
        }
        c[i][i] += reg;
        c[i][col_chr0] = y[i];
    }

    for i in 0..num_eq {
        let diag = {
            let d = c[i][i].unsigned_abs();
            if d < 1 { 1 } else { d }
        };
        let (scale, _round, shift) = division_scale_shift(diag);
        #[allow(clippy::needless_range_loop)]
        for j in i + 1..num_eq + 1 {
            c[i][j] = mul_fixed32_adapt(c[i][j], scale, shift);
        }
        for j in i + 1..num_eq {
            let scale_factor = c[j][i];
            #[allow(clippy::needless_range_loop)]
            for k in i + 1..num_eq + 1 {
                let delta = mul_fixed32_adapt(scale_factor, c[i][k], decim_bits);
                c[j][k] -= delta;
            }
        }
    }

    gauss_back_substitute(out, &c, num_eq, col_chr0, decim_bits);
}

/// avm `convolve`: `sum_i mul_fixed32_adapt(params[i], vector[i], DECIM)`,
/// clamped to i16. `vector` holds the three Q0 predictor terms.
#[inline]
pub(crate) fn convolve(params: &[i32; MHCCP_NUM_PARAMS], vector: &[i32; MHCCP_NUM_PARAMS]) -> i32 {
    let mut sum = 0i32;
    for i in 0..MHCCP_NUM_PARAMS {
        sum += mul_fixed32_adapt(params[i], vector[i], MHCCP_DECIM_BITS);
    }
    sum.clamp(i16::MIN as i32, i16::MAX as i32)
}

#[inline(always)]
fn clip_pixel_highbd(v: i32, bd: i32) -> i32 {
    let m = (1 << bd) - 1;
    v.clamp(0, m)
}

/// avm `CFL_BUF_LINE`; the MHCCP reference buffers use a stride of
/// `CFL_BUF_LINE * 2`.
pub(crate) const CFL_BUF_LINE: usize = 128;
/// avm reference-buffer stride (`ref_stride` / `output_stride`).
pub(crate) const MHCCP_REF_STRIDE: usize = CFL_BUF_LINE * 2;
/// avm `LINE_NUM`.
pub(crate) const LINE_NUM: usize = 1;

pub(crate) struct MhccpRefBuf {
    pub(crate) data: Vec<i32>,
    pub(crate) ref_width: usize,
    pub(crate) ref_height: usize,
    pub(crate) above_lines: usize,
    pub(crate) left_lines: usize,
    /// tx (block) width/height in chroma samples.
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) is_top_sb_boundary: bool,
}

impl MhccpRefBuf {
    pub(crate) fn new(
        ref_width: usize,
        ref_height: usize,
        above_lines: usize,
        left_lines: usize,
        width: usize,
        height: usize,
        is_top_sb_boundary: bool,
    ) -> Self {
        Self {
            data: vec![0i32; ref_height * MHCCP_REF_STRIDE],
            ref_width,
            ref_height,
            above_lines,
            left_lines,
            width,
            height,
            is_top_sb_boundary,
        }
    }
    #[inline(always)]
    pub(crate) fn set(&mut self, x: usize, y: usize, v: i32) {
        self.data[y * MHCCP_REF_STRIDE + x] = v;
    }
}

/// Derive MHCCP parameters for one plane and filter direction, reading the
/// AVM-layout luma (`l`, Q3) and chroma (`c`, Q0) reference buffers. Verbatim
/// port of `av2_mhccp_derive_multi_param_hv_c`.
pub(crate) fn derive_params(
    l: &MhccpRefBuf,
    c: &MhccpRefBuf,
    dir: u8,
    bd: i32,
) -> [i32; MHCCP_NUM_PARAMS] {
    let mid = 1 << (bd - 1);
    let above_lines = l.above_lines as i32;
    let left_lines = l.left_lines as i32;
    let ref_width = l.ref_width as i32;
    let ref_height = l.ref_height as i32;
    let ref_stride = MHCCP_REF_STRIDE as i32;

    let mut a = vec![[0i32; 3]; l.ref_width * l.ref_height];
    let mut ycb = vec![0i32; l.ref_width * l.ref_height];
    let mut count = 0usize;

    if above_lines != 0 || left_lines != 0 {
        let ld = &l.data;
        let cd = &c.data;
        for j in 1..ref_height - 1 {
            for i in 1..ref_width - 1 {
                if i >= left_lines && j >= above_lines {
                    continue;
                }
                let mut ref_h_offset = 0i32;
                if l.is_top_sb_boundary && above_lines == (LINE_NUM as i32 + 1) && j < above_lines {
                    ref_h_offset = above_lines - 1 - j;
                }
                let idx = |ii: i32, jj: i32| -> usize { (ii + jj * ref_stride) as usize };
                let center = ld[idx(i, j + ref_h_offset)] >> 3;
                let a0 = match dir {
                    1 => ld[idx(i, j + ref_h_offset - 1)] >> 3, // T
                    2 => ld[idx(i - 1, j + ref_h_offset)] >> 3, // L
                    _ => center,                                // C
                };
                a[count] = [
                    (a0 as i16) as i32,
                    (non_linear(center, mid, bd) as i16) as i32,
                    (mid as i16) as i32,
                ];
                ycb[count] = cd[idx(i, j + ref_h_offset)];
                count += 1;
            }
        }
    }

    if count == 0 {
        let mut out = [0i32; MHCCP_NUM_PARAMS];
        out[MHCCP_NUM_PARAMS - 1] = 1 << MHCCP_DECIM_BITS;
        return out;
    }

    let mut ata = [[0i32; MHCCP_NUM_PARAMS]; MHCCP_NUM_PARAMS];
    let mut ty = [0i32; MHCCP_NUM_PARAMS];
    #[allow(clippy::needless_range_loop)]
    for c0 in 0..MHCCP_NUM_PARAMS {
        #[allow(clippy::needless_range_loop)]
        for c1 in c0..MHCCP_NUM_PARAMS {
            let mut acc = 0i32;
            for r in 0..count {
                acc += a[r][c0] * a[r][c1];
            }
            ata[c0][c1] = acc;
        }
    }
    for (c, tyc) in ty.iter_mut().enumerate() {
        let mut acc = 0i32;
        for r in 0..count {
            acc += a[r][c] * ycb[r];
        }
        *tyc = acc;
    }

    let matrix_shift = (MHCCP_DECIM_BITS + 6) - 2 * bd - ceil_log2(count as i32);
    if matrix_shift > 0 {
        #[allow(clippy::needless_range_loop)]
        for c0 in 0..MHCCP_NUM_PARAMS {
            #[allow(clippy::needless_range_loop)]
            for c1 in c0..MHCCP_NUM_PARAMS {
                ata[c0][c1] <<= matrix_shift;
            }
        }
        for t in ty.iter_mut() {
            *t <<= matrix_shift;
        }
    } else if matrix_shift < 0 {
        let ms = -matrix_shift;
        #[allow(clippy::needless_range_loop)]
        for c0 in 0..MHCCP_NUM_PARAMS {
            for c1 in c0..MHCCP_NUM_PARAMS {
                ata[c0][c1] >>= ms;
            }
        }
        for t in ty.iter_mut() {
            *t >>= ms;
        }
    }

    let mut out = [0i32; MHCCP_NUM_PARAMS];
    gauss_elimination(&ata, &ty, bd, &mut out);
    out
}

/// Predict a `width x height` chroma block from the AVM-layout luma reference
/// buffer `l` using solved `params` and filter direction `dir`. Verbatim port of
/// `mhccp_predict_hv_hbd_c`: prediction reads the in-block luma at buffer offset
/// `(left_lines, above_lines)`, with `input[i - stride]` (above) and
/// `input[i - 1]` (left) taps. `have_top`/`have_left` govern edge replication.
pub(crate) fn predict_block(
    l: &MhccpRefBuf,
    params: &[i32; MHCCP_NUM_PARAMS],
    dir: u8,
    have_top: bool,
    have_left: bool,
    bd: i32,
    out: &mut [i32],
) {
    let mid = 1 << (bd - 1);
    let width = l.width;
    let height = l.height;
    let stride = MHCCP_REF_STRIDE as i32;
    // Base offset of the block top-left inside the reference buffer.
    let base = l.left_lines as i32 + l.above_lines as i32 * stride;
    let get = |off: i32| -> i32 { l.data[off as usize] };
    for j in 0..height as i32 {
        for i in 0..width as i32 {
            let p = base + i + j * stride;
            let center = get(p) >> 3;
            let a = if j - 1 < 0 && !have_top {
                get(p)
            } else {
                get(p - stride)
            } >> 3;
            let c = if i - 1 < 0 && !have_left {
                get(p)
            } else {
                get(p - 1)
            } >> 3;
            let v0 = match dir {
                1 => a,
                2 => c,
                _ => center,
            };
            let vector = [v0, non_linear(center, mid, bd), mid];
            out[(j as usize) * width + i as usize] =
                clip_pixel_highbd(convolve(params, &vector), bd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reproduce the standalone golden-vector driver deterministically so the
    // Rust port is checked against values emitted by the AVM C reference.
    struct Rng(u32);
    impl Rng {
        fn next10(&mut self) -> i32 {
            self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
            ((self.0 >> 16) & 0x3ff) as i32
        }
    }

    fn solve_config(dir: u8, count: usize, rng: &mut Rng, bd: i32) -> [i32; 3] {
        let mid = 1 << (bd - 1);
        let mut a = vec![[0i16; 3]; count];
        let mut y = vec![0i32; count];
        for r in 0..count {
            let lc = rng.next10();
            let lt = rng.next10();
            let ll = rng.next10();
            let chroma = rng.next10();
            let h_tap = match dir {
                1 => lt,
                2 => ll,
                _ => lc,
            };
            a[r] = [h_tap as i16, non_linear(lc, mid, bd) as i16, mid as i16];
            y[r] = chroma;
        }
        let mut ata = [[0i32; 3]; 3];
        let mut ty = [0i32; 3];
        for c0 in 0..3 {
            for c1 in c0..3 {
                let mut acc = 0i32;
                for r in 0..count {
                    acc += a[r][c0] as i32 * a[r][c1] as i32;
                }
                ata[c0][c1] = acc;
            }
        }
        for c in 0..3 {
            let mut acc = 0i32;
            for r in 0..count {
                acc += a[r][c] as i32 * y[r];
            }
            ty[c] = acc;
        }
        let ms = (MHCCP_DECIM_BITS + 6) - 2 * bd - ceil_log2(count as i32);
        if ms > 0 {
            for c0 in 0..3 {
                for c1 in c0..3 {
                    ata[c0][c1] <<= ms;
                }
            }
            for t in ty.iter_mut() {
                *t <<= ms;
            }
        } else if ms < 0 {
            let m = -ms;
            for c0 in 0..3 {
                for c1 in c0..3 {
                    ata[c0][c1] >>= m;
                }
            }
            for t in ty.iter_mut() {
                *t >>= m;
            }
        }
        let mut out = [0i32; 3];
        gauss_elimination(&ata, &ty, bd, &mut out);
        out
    }

    #[test]
    fn golden_vectors() {
        let bd = 10;
        let mid = 1 << (bd - 1);
        let mut rng = Rng(0x1234567);
        // Expected values captured from the AVM C reference (mhccp_ref.c).
        let expect_params: [[i32; 3]; 4] = [
            [-55821, 57209, 86247],
            [-855, 14286, 67922],
            [4834, -23298, 73277],
            [-49638, 34766, 96399],
        ];
        let expect_pred: [[i32; 6]; 4] = [
            [533, 461, 462, 465, 534, 461],
            [526, 630, 547, 597, 531, 701],
            [463, 503, 371, 573, 584, 464],
            [685, 496, 477, 477, 604, 482],
        ];
        for cfg in 0..4 {
            let dir = (cfg % 3) as u8;
            let count = 20 + cfg * 7;
            let params = solve_config(dir, count, &mut rng, bd);
            assert_eq!(params, expect_params[cfg], "params cfg {cfg}");
            for t in 0..6 {
                let c = rng.next10();
                let a = rng.next10();
                let l = rng.next10();
                let h_tap = match dir {
                    1 => a,
                    2 => l,
                    _ => c,
                };
                let vector = [h_tap, non_linear(c, mid, bd), mid];
                let pred = clip_pixel_highbd(convolve(&params, &vector), bd);
                assert_eq!(pred, expect_pred[cfg][t], "pred cfg {cfg} t {t}");
            }
        }
    }
}
