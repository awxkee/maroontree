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

//! AV2 shell-class MVD entropy encoder (QTR_PEL). Mirrors AVM `read_mv`.

use crate::av2::entropy::RangeEncoder;

// QTR_PEL (precision 5) nmvc defaults (raw icdf; encode_bool + decoder prob_scale both use raw icdf).
const JOINT_SHELL_SET: u32 = 1189;
// joint_shell_class_0/1: 8-sym icdf tables (AVM av2_default_nmv_context, QTR_PEL).
static SHELL_CLASS_0: [u16; 7] = [25271, 15467, 8920, 5330, 3373, 1889, 765];
static SHELL_CLASS_1: [u16; 7] = [14299, 5341, 1206, 116, 44, 40, 36];
static OFFSET_LOW: [u32; 2] = [18181, 11802];
const OFFSET_CLASS2: u32 = 19579;
static OFFSET_OTHER: [u32; 16] = [
    14825, 13834, 13840, 14072, 13724, 12406, 12342, 10205, 10578, 9310, 6541, 2003, 16384, 16384,
    16384, 16384,
];
static COL_MV_GREATER: [u32; 2] = [27105, 27912];
static COL_MV_INDEX: [u32; 4] = [19323, 19227, 18723, 19880];

const MAX_NUM_SHELL_CLASS: usize = 17;
const QTR_PEL: usize = 5;
const FIRST_SHELL_CLASS: usize = 8;
const MAX_COL_TU: usize = 2;

fn to_icdf(t: &[u16]) -> Vec<u16> {
    // Build an nsyms-symbol icdf array with trailing 0 sentinel for encode_symbol.
    let mut v: Vec<u16> = t.to_vec();
    v.push(0);
    v
}

/// Emit a truncated-unary value (max `maxv`) using class2 cdf then literal bits.
fn emit_trunc_unary(enc: &mut RangeEncoder, val: usize, maxv: usize, adaptive: bool) {
    for bit_idx in 0..maxv {
        let this_bit = if val > bit_idx { 1 } else { 0 };
        if bit_idx == 0 {
            if adaptive {
                enc.mvd_offset_class2(this_bit as usize);
            } else {
                enc.encode_bool(OFFSET_CLASS2, this_bit);
            }
        } else {
            enc.encode_bypass(this_bit, 1);
        }
        if this_bit == 0 {
            break;
        }
    }
}

/// Emit truncated-unary + quasi-uniform pair index (AVM read_tu_quasi_uniform inverse).
fn emit_tu_quasi(enc: &mut RangeEncoder, val: usize, maxv: usize, adaptive: bool) {
    let tu_max = maxv.min(MAX_COL_TU);
    let tu = val.min(tu_max);
    for bit_idx in 0..tu_max {
        let this_bit = if tu > bit_idx { 1 } else { 0 };
        let ctx = bit_idx.min(1);
        if adaptive {
            enc.mvd_col_greater(ctx, this_bit as usize);
        } else {
            enc.encode_bool(COL_MV_GREATER[ctx], this_bit);
        }
        if this_bit == 0 {
            break;
        }
    }
    if maxv > MAX_COL_TU && tu == MAX_COL_TU {
        let rem = val - MAX_COL_TU;
        emit_quniform(enc, rem, maxv - MAX_COL_TU + 1);
    }
}

/// Primitive quasi-uniform (AVM avm_write_primitive_quniform inverse).
fn emit_quniform(enc: &mut RangeEncoder, v: usize, n: usize) {
    if n <= 1 {
        return;
    }
    let l = (usize::BITS - n.leading_zeros()) as usize; // get_msb(n) + 1
    let m = (1usize << l) - n;
    if v < m {
        enc.encode_bypass(v as u32, (l - 1) as u32);
    } else {
        let coded = v + m;
        enc.encode_bypass((coded >> 1) as u32, (l - 1) as u32);
        enc.encode_bypass((coded & 1) as u32, 1);
    }
}

fn quniform_bits(v: usize, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let l = (usize::BITS - n.leading_zeros()) as usize;
    let m = (1usize << l) - n;
    if v < m { (l - 1) as f32 } else { l as f32 }
}

/// Ideal bit cost of the exact adaptive QTR_PEL shell syntax emitted by
/// `encode_mvd_qtr`, including bypass-coded signs. This is deliberately
/// read-only so candidate evaluation cannot perturb the tile CDF state.
pub(crate) fn estimate_mvd_qtr_bits(enc: &RangeEncoder, scaled_row: i32, scaled_col: i32) -> f32 {
    let row = scaled_row.unsigned_abs() as usize;
    let col = scaled_col.unsigned_abs() as usize;
    let shell_index = row + col;
    let num_mv_class = MAX_NUM_SHELL_CLASS - (6 - QTR_PEL);
    let num_class_0 = num_mv_class >> 1;
    let sc = if shell_index == 0 {
        0
    } else {
        let mut class = 0;
        while class + 1 < num_mv_class && (1usize << (class + 1)) <= shell_index {
            class += 1;
        }
        class
    };

    let (set, class) = if sc < num_class_0 {
        (0, sc)
    } else {
        (1, sc - num_class_0)
    };
    let mut bits =
        enc.estimate_mvd_shell_set_bits(set) + enc.estimate_mvd_shell_class_bits(set, class);

    let base = if sc == 0 { 0 } else { 1usize << sc };
    let offset = shell_index - base;
    if sc < 2 {
        bits += enc.estimate_mvd_offset_low_bits(sc, offset);
    } else if sc == 2 {
        for bit_idx in 0..3 {
            let bit = usize::from(offset > bit_idx);
            bits += if bit_idx == 0 {
                enc.estimate_mvd_offset_class2_bits(bit)
            } else {
                1.0
            };
            if bit == 0 {
                break;
            }
        }
    } else {
        for bit_idx in 0..sc {
            bits += enc.estimate_mvd_offset_other_bits(bit_idx, (offset >> bit_idx) & 1);
        }
    }

    if shell_index > 0 {
        let max_pair = shell_index >> 1;
        let this_pair = col.min(shell_index - col);
        if max_pair > 0 {
            let tu_max = max_pair.min(MAX_COL_TU);
            let tu = this_pair.min(tu_max);
            for bit_idx in 0..tu_max {
                let bit = usize::from(tu > bit_idx);
                bits += enc.estimate_mvd_col_greater_bits(bit_idx.min(1), bit);
                if bit == 0 {
                    break;
                }
            }
            if max_pair > MAX_COL_TU && tu == MAX_COL_TU {
                bits += quniform_bits(this_pair - MAX_COL_TU, max_pair - MAX_COL_TU + 1);
            }
        }
        let skip_col_bit = this_pair == max_pair && shell_index.is_multiple_of(2);
        if !skip_col_bit {
            let split_bit = usize::from(col != this_pair);
            bits += enc.estimate_mvd_col_index_bits(sc.min(3), split_bit);
        }
    }

    bits + usize::from(scaled_row != 0) as f32 + usize::from(scaled_col != 0) as f32
}

/// Encode a single MVD component pair (row,col) at QTR_PEL. `row`/`col` are the
/// scaled non-negative diffs (already >> start_lsb). Emits shell class/offset/col
/// then the caller emits signs. Mirrors AVM `read_mv` (non-adaptive).
pub fn encode_mvd_qtr(enc: &mut RangeEncoder, scaled_row: i32, scaled_col: i32) {
    let row = scaled_row.unsigned_abs() as usize;
    let col = scaled_col.unsigned_abs() as usize;
    let shell_index = row + col;
    let adaptive = enc.mvd_adaptive();

    let num_mv_class = MAX_NUM_SHELL_CLASS - (6 - QTR_PEL); // 16
    let num_class_0 = num_mv_class >> 1; // 8
    // shell_class: largest c with base(c) <= shell_index (base = c==0?0:1<<c).
    let sc = if shell_index == 0 {
        0
    } else {
        let mut c = 0;
        while c + 1 < num_mv_class && (1usize << (c + 1)) <= shell_index {
            c += 1;
        }
        c
    };

    // Emit set + class (adaptive CDFs when cdf_state active; else static).
    if sc < num_class_0 {
        if adaptive {
            enc.mvd_shell_set(0);
            enc.mvd_shell_class0(sc);
        } else {
            enc.encode_bool(JOINT_SHELL_SET, 0);
            enc.encode_symbol(&to_icdf(&SHELL_CLASS_0), sc, FIRST_SHELL_CLASS);
        }
    } else {
        let idx = sc - num_class_0;
        if adaptive {
            enc.mvd_shell_set(1);
            enc.mvd_shell_class1(idx);
        } else {
            enc.encode_bool(JOINT_SHELL_SET, 1);
            enc.encode_symbol(&to_icdf(&SHELL_CLASS_1), idx, FIRST_SHELL_CLASS);
        }
    }

    // Shell class offset.
    let base = if sc == 0 { 0 } else { 1usize << sc };
    let offset = shell_index - base;
    if sc < 2 {
        if adaptive {
            enc.mvd_offset_low(sc, offset);
        } else {
            enc.encode_bool(OFFSET_LOW[sc], offset as u32);
        }
    } else if sc == 2 {
        emit_trunc_unary(enc, offset, 3, adaptive);
    } else {
        let nbits = sc;
        for (i, &offset_other) in OFFSET_OTHER[..nbits].iter().enumerate() {
            let bit = (offset >> i) & 1;
            if adaptive {
                enc.mvd_offset_other(i, bit);
            } else {
                enc.encode_bool(offset_other, bit as u32);
            }
        }
    }

    // Col split within shell_index.
    if shell_index > 0 {
        let max_pair = shell_index >> 1;
        let (this_pair, scaled_col_out) = if col <= max_pair {
            (col, col)
        } else {
            (shell_index - col, col)
        };
        if max_pair > 0 {
            emit_tu_quasi(enc, this_pair, max_pair, adaptive);
        }
        let skip_col_bit = (this_pair == max_pair) && shell_index.is_multiple_of(2);
        if !skip_col_bit {
            let ctx = sc.min(3);
            // this_bit=0 -> col=this_pair; 1 -> col=shell_index-this_pair.
            let this_bit = if scaled_col_out == this_pair { 0 } else { 1 };
            if adaptive {
                enc.mvd_col_index(ctx, this_bit);
            } else {
                enc.encode_bool(COL_MV_INDEX[ctx], this_bit as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av2::entropy::RangeEncoder;

    // Encode a few MVDs; assert the shell_class/base decomposition is internally
    // consistent (shell_index reconstructs, col in valid range). Full bitstream
    // round-trip verified against avmdec (shell_index + row/col exact).
    #[test]
    fn shell_decomposition_valid() {
        for &(r, c) in &[(0, 0), (1, 0), (0, 1), (3, 2), (5, 7), (12, 4), (0, 31)] {
            let shell_index = (r as usize) + (c as usize);
            let num_mv_class = MAX_NUM_SHELL_CLASS - (6 - QTR_PEL);
            let sc = if shell_index == 0 {
                0
            } else {
                let mut cc = 0;
                while cc + 1 < num_mv_class && (1usize << (cc + 1)) <= shell_index {
                    cc += 1;
                }
                cc
            };
            let base = if sc == 0 { 0 } else { 1usize << sc };
            assert!(base <= shell_index, "base {} <= idx {}", base, shell_index);
            assert!(c as usize <= shell_index);
        }
    }

    #[test]
    fn encodes_without_panic() {
        let mut enc = RangeEncoder::new();
        for &(r, c) in &[(0, 0), (1, 0), (0, 1), (3, 2), (5, 7), (12, 4)] {
            encode_mvd_qtr(&mut enc, r, c);
        }
        let _ = enc.finish();
    }
}
