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
use crate::coder::*;
use crate::odec::OdEcEncoder;
use crate::tables::*;

/// Encode `read_golomb` value `v` (>=0) as unary-prefixed binary via bypass bits.
pub(crate) fn encode_golomb(enc: &mut OdEcEncoder, v: u32) {
    let x = v + 1;
    let length = 32 - x.leading_zeros();
    for _ in 0..length - 1 {
        enc.encode_bool(false, 16384); // unary zeros
    }
    for i in (0..length).rev() {
        enc.encode_bool((x >> i) & 1 == 1, 16384); // x MSB-first (MSB terminates unary)
    }
}

#[inline(always)]
pub(crate) fn get_lo_ctx_2d(
    levels: &[u8],
    x: usize,
    y: usize,
    off: &[[u32; 5]; 5],
    stride: usize,
) -> (usize, u32) {
    let g = |dx: usize, dy: usize| levels[(x + dx) * stride + (y + dy)] as u32;
    let hi_mag = g(0, 1) + g(1, 0) + g(1, 1);
    let mag = hi_mag + g(0, 2) + g(2, 0);
    let offset = off[y.min(4)][x.min(4)];
    let ctx = offset + if mag > 512 { 4 } else { (mag + 64) >> 7 };
    (ctx as usize, hi_mag)
}

#[inline]
fn eob_and_cul<const N: usize>(cf: &[i32; N], scan: &[u32]) -> Option<(usize, u32)> {
    let eob = scan.iter().rposition(|&rc| cf[rc as usize] != 0)?;
    let cul = scan[..=eob]
        .iter()
        .map(|&rc| cf[rc as usize].unsigned_abs())
        .sum();
    Some((eob, cul))
}

pub(crate) fn encode_hi_tok(enc: &mut OdEcEncoder, m: u32, br_cdf: &mut [u16]) {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0;
    for _ in 0..(COEFF_BASE_RANGE / 3) {
        let s = (total_br - coded).min(3);
        enc.encode_symbol(s as usize, br_cdf);
        coded += s;
        if s < 3 {
            break;
        }
    }
}

/// Adaptive eob==0 DC-only tail (shared by all adaptive coef coders). Codes the
/// eob_pt=0 symbol, the DC base token, optional hi_tok ladder, the dc_sign, and
/// the Golomb residual — adapting each CDF.
pub(crate) fn encode_dc_tail(
    enc: &mut OdEcEncoder,
    level: i32,
    eob_bin_cdf: &mut [u16],
    base_eob: &mut [u16],
    dc_sign: &mut [u16],
    br0: &mut [u16],
) {
    enc.encode_symbol(0, eob_bin_cdf);
    let m = level.unsigned_abs();
    let base = m.min(3);
    enc.encode_symbol(base as usize - 1, base_eob);
    if base == 3 {
        encode_hi_tok(enc, m, br0);
    }
    enc.encode_symbol((level < 0) as usize, dc_sign);
    if m >= 15 {
        encode_golomb(enc, m - 15);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_tx8_coeffs_1d(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    vertical: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
) -> u8 {
    let pl = 0usize; // luma only (chroma never signals a tx type)
    let pos_rc = |i: usize| -> usize {
        let (x, y) = (i & 7, i >> 3);
        if vertical { (x << 3) | y } else { i }
    };
    // EOB over the class ordering.
    let eob = match (0..64).rev().find(|&i| cf[pos_rc(i)] != 0) {
        Some(e) => e,
        None => {
            enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]);
            return 0x40;
        }
    };
    let cul: u32 = (0..=eob).map(|i| cf[pos_rc(i)].unsigned_abs()).sum();
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]);
    enc.encode_symbol(
        if vertical { 2 } else { 3 }, // V_DCT = 2, H_DCT = 3 (7-type intra set)
        &mut cdfs.txtp[y_mode],
    );
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_64_l1d,
            &mut cdfs.eob_base[1][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[1][pl][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_64_l1d);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    // Levels scratch: stride 16, zero-padded (1-D ctx reaches (x, y+4)).
    let mut levels = [0u8; 16 * 10];
    let lvl = |lv: &[u8], x: usize, y: usize| -> u32 { lv[x * 16 + y] as u32 };
    // 1-D lo ctx at (x, y): returns (ctx, hi_mag).
    let lo_ctx_1d = |lv: &[u8], x: usize, y: usize| -> (usize, u32) {
        let mut mag = lvl(lv, x, y + 1) + lvl(lv, x + 1, y) + lvl(lv, x, y + 2);
        let hi_mag = mag;
        mag += lvl(lv, x, y + 3) + lvl(lv, x, y + 4);
        let offset = 26 + if y > 1 { 10 } else { y * 5 };
        (
            offset
                + if mag > 512 {
                    4
                } else {
                    ((mag + 64) >> 7) as usize
                },
            hi_mag,
        )
    };
    // EOB coefficient.
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    {
        let i = eob;
        let (x, y) = (i & 7, i >> 3);
        let rc = pos_rc(i);
        let m = cf[rc].unsigned_abs();
        let eob_tok = m.min(3) - 1;
        enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][pl][ctx_e]);
        if eob_tok == 2 {
            let bc = if y != 0 { 14 } else { 7 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][pl][bc]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    for i in (1..eob).rev() {
        let (x, y) = (i & 7, i >> 3);
        let rc = pos_rc(i);
        let (ctx, hi_mag) = lo_ctx_1d(&levels, x, y);
        let m = cf[rc].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if y != 0 { 14 } else { 7 })
                + if mag > 12 {
                    6
                } else {
                    ((mag + 1) >> 1) as usize
                };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][pl][bc]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    // DC: derived (non-zero) low context for the 1-D classes.
    let (dc_ctx, dc_hi_mag) = lo_ctx_1d(&levels, 0, 0);
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][pl][dc_ctx]);
    if dc_tok == 3 {
        let mag = dc_hi_mag & 63;
        let bc = if mag > 12 {
            6
        } else {
            ((mag + 1) >> 1) as usize
        };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][pl][bc]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    // AC sign + golomb tails, in class order (mirrors the 2-D coder's tail).
    for i in 1..=eob {
        let rc = pos_rc(i);
        let c = cf[rc];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma coefficient coder for the 1-D transform classes at TX_4X4. Same class
/// geometry as `encode_tx8_coeffs_1d` (the 1-D lo/br contexts depend only on
/// class coordinates), with 4x4 specifics: size-class 0 CDFs, the separate
/// `eob_bin_16[pl=0][is_1d=1]` eob bins, the `txtp4` symbol (V_DCT=2/H_DCT=3),
/// and the eob-coefficient context thresholds `1 + (eob>2) + (eob>4)`
/// (dav1d's `1 + (eob > sw*sh*2) + (eob > sw*sh*4)` with sw*sh = 1).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_tx4_coeffs_1d(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 16],
    vertical: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
) -> u8 {
    let pl = 0usize; // luma only (chroma never signals a tx type)
    let pos_rc = |i: usize| -> usize {
        let (x, y) = (i & 3, i >> 2);
        if vertical { (x << 2) | y } else { i }
    };
    let eob = match (0..16).rev().find(|&i| cf[pos_rc(i)] != 0) {
        Some(e) => e,
        None => {
            enc.encode_symbol(1, &mut cdfs.txb_skip[0][skip_ctx]);
            return 0x40;
        }
    };
    let cul: u32 = (0..=eob).map(|i| cf[pos_rc(i)].unsigned_abs()).sum();
    enc.encode_symbol(0, &mut cdfs.txb_skip[0][skip_ctx]);
    enc.encode_symbol(
        if vertical { 2 } else { 3 }, // V_DCT = 2, H_DCT = 3 (7-type intra set)
        &mut cdfs.txtp4[y_mode],
    );
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_16_l1d,
            &mut cdfs.eob_base[0][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[0][pl][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_16_l1d);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[0][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    // Levels scratch: stride 16, zero-padded (1-D ctx reaches (x, y+4)).
    let mut levels = [0u8; 16 * 10];
    let lvl = |lv: &[u8], x: usize, y: usize| -> u32 { lv[x * 16 + y] as u32 };
    let lo_ctx_1d = |lv: &[u8], x: usize, y: usize| -> (usize, u32) {
        let mut mag = lvl(lv, x, y + 1) + lvl(lv, x + 1, y) + lvl(lv, x, y + 2);
        let hi_mag = mag;
        mag += lvl(lv, x, y + 3) + lvl(lv, x, y + 4);
        let offset = 26 + if y > 1 { 10 } else { y * 5 };
        (
            offset
                + if mag > 512 {
                    4
                } else {
                    ((mag + 64) >> 7) as usize
                },
            hi_mag,
        )
    };
    // EOB coefficient.
    let ctx_e = 1 + (eob > 2) as usize + (eob > 4) as usize;
    {
        let i = eob;
        let (x, y) = (i & 3, i >> 2);
        let rc = pos_rc(i);
        let m = cf[rc].unsigned_abs();
        let eob_tok = m.min(3) - 1;
        enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[0][pl][ctx_e]);
        if eob_tok == 2 {
            let bc = if y != 0 { 14 } else { 7 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[0][pl][bc]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    for i in (1..eob).rev() {
        let (x, y) = (i & 3, i >> 2);
        let rc = pos_rc(i);
        let (ctx, hi_mag) = lo_ctx_1d(&levels, x, y);
        let m = cf[rc].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[0][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if y != 0 { 14 } else { 7 })
                + if mag > 12 {
                    6
                } else {
                    ((mag + 1) >> 1) as usize
                };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[0][pl][bc]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    // DC: derived (non-zero) low context for the 1-D classes.
    let (dc_ctx, dc_hi_mag) = lo_ctx_1d(&levels, 0, 0);
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[0][pl][dc_ctx]);
    if dc_tok == 3 {
        let mag = dc_hi_mag & 63;
        let bc = if mag > 12 {
            6
        } else {
            ((mag + 1) >> 1) as usize
        };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[0][pl][bc]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    // AC sign + golomb tails, in class order (mirrors the 2-D coder's tail).
    for i in 1..=eob {
        let rc = pos_rc(i);
        let c = cf[rc];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma coefficient coder for the 1-D transform classes (V_DCT / H_DCT) at the
/// rect16 shapes RTX_16X8 (`w=16`) / RTX_8X16 (`w=8`), `cf[fx*h + fy]`.
/// Generalizes `encode_tx8_coeffs_1d` with dav1d's per-size class geometry:
/// class H: `x = i & (h-1)`, `y = i >> log2(h)`, `rc = i`; class V:
/// `x = i & (w-1)`, `y = i >> log2(w)`, `rc = (x << log2(h)) | y`. Shares the
/// 128-coeff (class 2) base/br/eob CDFs with the 2-D rect coder but uses the
/// separate `eob_bin_128[0][is_1d=1]` eob bins. Levels scratch stride 16;
/// slow-extent-16 shapes read across a row boundary at `y+k >= 16`, which
/// lands on an earlier (not-yet-coded, still zero) scan position — exactly
/// dav1d's layout. Returns the coef neighbor byte.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_rect_coeffs_1d(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 128],
    w: usize,
    vertical: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
) -> u8 {
    let pl = 0usize; // luma only (chroma never signals a tx type)
    let h = 128 / w;
    let (hsh, wsh) = (h.trailing_zeros() as usize, w.trailing_zeros() as usize);
    // dav1d class geometry: fast/slow decomposition + cf position per index.
    let pos_xy = |i: usize| -> (usize, usize) {
        if vertical {
            (i & (w - 1), i >> wsh)
        } else {
            (i & (h - 1), i >> hsh)
        }
    };
    let pos_rc = |i: usize| -> usize {
        if vertical {
            let (x, y) = (i & (w - 1), i >> wsh);
            (x << hsh) | y
        } else {
            i
        }
    };
    let eob = match (0..128).rev().find(|&i| cf[pos_rc(i)] != 0) {
        Some(e) => e,
        None => {
            enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]);
            return 0x40;
        }
    };
    let cul: u32 = (0..=eob).map(|i| cf[pos_rc(i)].unsigned_abs()).sum();
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]);
    enc.encode_symbol(
        if vertical { 2 } else { 3 }, // V_DCT = 2, H_DCT = 3 (7-type intra set)
        &mut cdfs.txtp[y_mode],
    );
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_128_l1d,
            &mut cdfs.eob_base[2][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[2][pl][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_128_l1d);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    // Levels scratch: stride 16, sized for the widest fast axis (16 rows) plus
    // the (x, y+4) reach and two pad rows.
    let mut levels = [0u8; 16 * 18];
    let lvl = |lv: &[u8], x: usize, y: usize| -> u32 { lv[x * 16 + y] as u32 };
    let lo_ctx_1d = |lv: &[u8], x: usize, y: usize| -> (usize, u32) {
        let mut mag = lvl(lv, x, y + 1) + lvl(lv, x + 1, y) + lvl(lv, x, y + 2);
        let hi_mag = mag;
        mag += lvl(lv, x, y + 3) + lvl(lv, x, y + 4);
        let offset = 26 + if y > 1 { 10 } else { y * 5 };
        (
            offset
                + if mag > 512 {
                    4
                } else {
                    ((mag + 64) >> 7) as usize
                },
            hi_mag,
        )
    };
    // EOB coefficient.
    let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
    {
        let i = eob;
        let (x, y) = pos_xy(i);
        let rc = pos_rc(i);
        let m = cf[rc].unsigned_abs();
        let eob_tok = m.min(3) - 1;
        enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][pl][ctx_e]);
        if eob_tok == 2 {
            let bc = if y != 0 { 14 } else { 7 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][pl][bc]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    for i in (1..eob).rev() {
        let (x, y) = pos_xy(i);
        let rc = pos_rc(i);
        let (ctx, hi_mag) = lo_ctx_1d(&levels, x, y);
        let m = cf[rc].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if y != 0 { 14 } else { 7 })
                + if mag > 12 {
                    6
                } else {
                    ((mag + 1) >> 1) as usize
                };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][pl][bc]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    // DC: derived (non-zero) low context for the 1-D classes.
    let (dc_ctx, dc_hi_mag) = lo_ctx_1d(&levels, 0, 0);
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][pl][dc_ctx]);
    if dc_tok == 3 {
        let mag = dc_hi_mag & 63;
        let bc = if mag > 12 {
            6
        } else {
            ((mag + 1) >> 1) as usize
        };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][pl][bc]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let rc = pos_rc(i);
        let c = cf[rc];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_tx8_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let pl = chroma as usize;
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_8X8) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    if !chroma {
        enc.encode_symbol(txtp, &mut cdfs.txtp[y_mode]); // luma: 1=DCT_DCT, 4=ADST_ADST
    }
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        let eb = if chroma {
            &mut cdfs.eob_bin_64_c
        } else {
            &mut cdfs.eob_bin_64_l
        };
        encode_dc_tail(
            enc,
            cf[0],
            eb,
            &mut cdfs.eob_base[1][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[1][pl][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    {
        let eb = if chroma {
            &mut cdfs.eob_bin_64_c
        } else {
            &mut cdfs.eob_bin_64_l
        };
        enc.encode_symbol(eob_bin, eb);
    }
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384); // extra eob bits, equiprobable
        }
    }
    let mut levels = [0u8; 80];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_8X8[eob] as usize;
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1; // 0,1,2
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][pl][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][pl][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X8[i] as usize;
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][pl][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][pl][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][pl][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X8[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// TX_16X16 coefficient coder (class 2). Parameterized port of
/// `encode_tx8_coeffs_adapt`: 256 coeffs in `SCAN_16X16` order, stride 16, coef
/// CDF class 2, eob_pt over 256 (`eob_bin_256`), eob ctx thresholds 256>>3=32 /
/// 256>>2=64, and the 2D coeff-base context reuses `LO_CTX_OFF` + `get_lo_ctx_2d`
/// at stride 16 (the libaom 16x16 offset table equals that 5x5 region). Used for
/// 4:4:4 luma (chroma=false) and 4:4:4 chroma (chroma=true).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_tx16_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 256],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let pl = chroma as usize;
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 0
    if !chroma {
        enc.encode_symbol(txtp, &mut cdfs.txtp16[y_mode]); // luma TX_16X16: 1=DCT_DCT, 2=ADST_ADST
    }
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        let eb = if chroma {
            &mut cdfs.eob_bin_256_c
        } else {
            &mut cdfs.eob_bin_256_l
        };
        encode_dc_tail(
            enc,
            cf[0],
            eb,
            &mut cdfs.eob_base[2][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[2][pl][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    {
        let eb = if chroma {
            &mut cdfs.eob_bin_256_c
        } else {
            &mut cdfs.eob_bin_256_l
        };
        enc.encode_symbol(eob_bin, eb);
    }
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 320]; // stride 16, neighbor reads up to (x+2)*16+(y+2)=289
    let ctx_e = 1 + (eob > 32) as usize + (eob > 64) as usize;
    let rc = SCAN_16X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1; // 0,1,2
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][pl][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][pl][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][pl][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][pl][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][pl][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// TX_32X32 coefficient coder (class 3). Mirror of `encode_tx16_coeffs_adapt`
/// for 1024 coeffs in `SCAN_32X32` order, stride 32, coef CDF class 3, eob_pt
/// over 1024 (`eob_bin_1024`), eob ctx thresholds 1024>>3=128 / 1024>>2=256.
/// Intra TX_32X32 codes NO tx-type symbol: dav1d derives `t_dim->max + intra >=
/// TX_64X64` -> DCT_DCT, so unlike the 16x16 luma path there is no `txtp`
/// symbol. The 2D coeff-base context reuses `LO_CTX_OFF` + `get_lo_ctx_2d` at
/// stride 32 (the libaom offset table saturates to 21 outside the 5x5 corner,
/// identical for every square transform). Used for 4:4:4 luma and chroma.
pub(crate) fn encode_tx32_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 1024],
    chroma: bool,
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let pl = chroma as usize;
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_32X32) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]); // all_zero = 0
    // NO txtp symbol for intra TX_32X32 (DCT_DCT implied).
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        let eb = if chroma {
            &mut cdfs.eob_bin_1024_c
        } else {
            &mut cdfs.eob_bin_1024_l
        };
        encode_dc_tail(
            enc,
            cf[0],
            eb,
            &mut cdfs.eob_base[3][pl][0],
            &mut cdfs.dc_sign[pl][dcs_ctx],
            &mut cdfs.br_tok[3][pl][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    {
        let eb = if chroma {
            &mut cdfs.eob_bin_1024_c
        } else {
            &mut cdfs.eob_bin_1024_l
        };
        enc.encode_symbol(eob_bin, eb);
    }
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][pl][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 1156]; // stride 32, neighbor reads up to (x+2)*32+(y+2)
    let ctx_e = 1 + (eob > 128) as usize + (eob > 256) as usize;
    let rc = SCAN_32X32[eob] as usize;
    let (ex, ey) = (rc >> 5, rc & 31);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1; // 0,1,2
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][pl][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][pl][bc]);
    }
    levels[ex * 32 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_32X32[i] as usize;
        let (x, y) = (rc_i >> 5, rc_i & 31);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 32);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][pl][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][pl][bc as usize]);
        }
        levels[x * 32 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][pl][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[32] as u32 + levels[33] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][pl][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[pl][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_32X32[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// DC-only tail for the eob==0 path with a raw signed level.
/// 4:2:2 chroma coefficient coder for an `RTX_4X8` block (4 wide x 8 tall, 32
/// coeffs, `cf[fx*8+fy]`). `RTX_4X8` shares coef-CDF class ctx=1 with `TX_8X8`,
/// so the base/br/eob-base/eob-hi/dc-sign/skip CDFs are reused; only the eob_pt
/// CDF (`eob_bin_32`), the scan, the lo-ctx offsets (w<h) and the eob-ctx
/// thresholds differ. Returns the dav1d coef neighbor-context byte.
pub(crate) fn encode_4x8_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 32],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_4X8) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_32_c,
            &mut cdfs.eob_base[1][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[1][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_32_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob coeff: eob-ctx thresholds use sw*sh = imin(w,8)*imin(h,8) = 1*2 = 2
    let ctx_e = 1 + (eob > 4) as usize + (eob > 8) as usize;
    let rc = SCAN_4X8[eob] as usize;
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X8[i] as usize;
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X8[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma variant.
pub(crate) fn encode_4x8_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 32],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_4X8) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    enc.encode_symbol(txtp, &mut cdfs.txtp4[y_mode]);
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_32_l,
            &mut cdfs.eob_base[1][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[1][0][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_32_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob coeff: eob-ctx thresholds use sw*sh = imin(w,8)*imin(h,8) = 1*2 = 2
    let ctx_e = 1 + (eob > 4) as usize + (eob > 8) as usize;
    let rc = SCAN_4X8[eob] as usize;
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X8[i] as usize;
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X8[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// 8x4 chroma coeff coder (8 wide x 4 tall). tx-class [1], SCAN_8X4.
pub(crate) fn encode_8x4_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 32],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_8X4) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_32_c,
            &mut cdfs.eob_base[1][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[1][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_32_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob coeff: eob-ctx thresholds use sw*sh = imin(w,8)*imin(h,8) = 1*2 = 2
    let ctx_e = 1 + (eob > 4) as usize + (eob > 8) as usize;
    let rc = SCAN_8X4[eob] as usize;
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X4[i] as usize;
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[5] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X4[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma variant.
pub(crate) fn encode_8x4_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 32],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_8X4) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    enc.encode_symbol(txtp, &mut cdfs.txtp4[y_mode]);
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_32_l,
            &mut cdfs.eob_base[1][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[1][0][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_32_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob coeff: eob-ctx thresholds use sw*sh = imin(w,8)*imin(h,8) = 1*2 = 2
    let ctx_e = 1 + (eob > 4) as usize + (eob > 8) as usize;
    let rc = SCAN_8X4[eob] as usize;
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X4[i] as usize;
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[5] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X4[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// `RTX_16X4` luma coefficient coder (PARTITION_HORZ_4 strips).
/// Coefficient class 1 (min dim 4 -> t_dim ctx 1), `eob_bin_64`, LO_CTX_OFF_WGH
/// neighbour offsets at stride 4. Mirrors `encode_16x8_*_coeffs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_16x4_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X4) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    enc.encode_symbol(txtp, &mut cdfs.txtp4[y_mode]); // min-dim-4 luma set

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_64_l,
            &mut cdfs.eob_base[1][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[1][0][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_64_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_16X4[eob] as usize;
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X4[i] as usize;
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[4 + 1] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X4[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// `RTX_16X4` chroma (both planes share CDF plane 1) coefficient coder (PARTITION_HORZ_4 strips).
/// Coefficient class 1 (min dim 4 -> t_dim ctx 1), `eob_bin_64`, LO_CTX_OFF_WGH
/// neighbour offsets at stride 4. Mirrors `encode_16x8_*_coeffs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_16x4_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X4) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_64_c,
            &mut cdfs.eob_base[1][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[1][1][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_64_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_16X4[eob] as usize;
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X4[i] as usize;
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[4 + 1] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X4[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// `RTX_4X16` luma coefficient coder (PARTITION_VERT_4 strips).
/// Coefficient class 1 (min dim 4 -> t_dim ctx 1), `eob_bin_64`, LO_CTX_OFF_WLH
/// neighbour offsets at stride 16. Mirrors `encode_16x8_*_coeffs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_4x16_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_4X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    enc.encode_symbol(txtp, &mut cdfs.txtp4[y_mode]); // min-dim-4 luma set

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_64_l,
            &mut cdfs.eob_base[1][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[1][0][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_64_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 120];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_4X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][0][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[16 + 1] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// `RTX_4X16` chroma (both planes share CDF plane 1) coefficient coder (PARTITION_VERT_4 strips).
/// Coefficient class 1 (min dim 4 -> t_dim ctx 1), `eob_bin_64`, LO_CTX_OFF_WLH
/// neighbour offsets at stride 16. Mirrors `encode_16x8_*_coeffs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_4x16_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 64],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_4X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[1][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_64_c,
            &mut cdfs.eob_base[1][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[1][1][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_64_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[1][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 120];
    let ctx_e = 1 + (eob > 8) as usize + (eob > 16) as usize;
    let rc = SCAN_4X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[1][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[1][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[1][1][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[1][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[16 + 1] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[1][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma coefficient coder for an `RTX_16X8` block (16 wide x 8 tall, 128 coeffs,
/// `cf[fx*8 + fy]`, fx in 0..16, fy in 0..8). `txsize_sqr_up[TX_16X8] = TX_16X16`
/// so it shares the coef-CDF class ctx=2 with luma TX_16X16 and RTX_8X16, but uses
/// the LUMA `eob_bin_128_l` bins, the LUMA plane index [0], the `w>h` level-offset
/// table at stride 8, eob-ctx thresholds N>>3 / N>>2 = 16 / 32, and — being luma —
/// SIGNALS the transform type (txtp_intra1 set, `min = TX_8X8`, the same CDF the
/// 8x8 luma path uses). Level position decomposition: x = rc>>3 (0..16),
/// y = rc&7 (0..8); levels[x*8 + y]. Returns the coef neighbor byte.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_16x8_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 128],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X8) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 0
    enc.encode_symbol(txtp, &mut cdfs.txtp[y_mode]); // luma signals txtp

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_128_l,
            &mut cdfs.eob_base[2][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[2][0][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_128_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 200];
    let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
    let rc = SCAN_16X8[eob] as usize;
    let (ex, ey) = (rc >> 3, rc & 7); // x in 0..16, y in 0..8
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][0][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X8[i] as usize;
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][0][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X8[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// 4:4:4 chroma coefficient coder for an `RTX_16X8` block (16 wide x 8 tall).
/// Same structure as `encode_16x8_luma_coeffs` but plane index [1] (chroma),
/// the chroma `eob_bin_128_c` bins, and NO txtp symbol (chroma infers the tx).
pub(crate) fn encode_16x8_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 128],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X8) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]);
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_128_c,
            &mut cdfs.eob_base[2][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[2][1][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_128_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 200];
    let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
    let rc = SCAN_16X8[eob] as usize;
    let (ex, ey) = (rc >> 3, rc & 7);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][1][bc]);
    }
    levels[ex * 8 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X8[i] as usize;
        let (x, y) = (rc_i >> 3, rc_i & 7);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 8);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][1][bc as usize]);
        }
        levels[x * 8 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[8] as u32 + levels[9] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X8[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// 4:2:2 chroma coefficient coder for an `RTX_8X16` block (8 wide x 16 tall,
/// 128 coeffs, `cf[fx*16+fy]`). `txsize_sqr_up[TX_8X16] = TX_16X16`, so it uses
/// the coef-CDF class ctx=2 (the same base/br/eob/skip CDFs as luma TX_16X16),
/// the chroma `eob_multi128` bins, the `w<h` level-offset table at stride 16,
/// and eob-ctx thresholds N>>3 / N>>2 = 16 / 32. Chroma infers the transform
/// type (no txtp symbol). Returns the coef neighbor byte.
pub(crate) fn encode_8x16_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 128],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_8X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 0
    // chroma infers txtp (no symbol)

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_128_c,
            &mut cdfs.eob_base[2][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[2][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_128_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 200];
    // eob coeff: 128 coeffs -> thresholds 128>>3 = 16, 128>>2 = 32
    let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
    let rc = SCAN_8X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][1][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][1][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma coeff coder for an `RTX_8X16` block (8 wide x 16 tall).
pub(crate) fn encode_8x16_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 128],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_8X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 1
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[2][skip_ctx]); // all_zero = 0
    enc.encode_symbol(txtp, &mut cdfs.txtp[y_mode]);

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_128_l,
            &mut cdfs.eob_base[2][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[2][0][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_128_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[2][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 200];
    // eob coeff: 128 coeffs -> thresholds 128>>3 = 16, 128>>2 = 32
    let ctx_e = 1 + (eob > 16) as usize + (eob > 32) as usize;
    let rc = SCAN_8X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[2][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[2][0][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_8X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[2][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[2][0][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[2][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[2][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_8X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

pub(crate) fn encode_4x4_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 16],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_4X4) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[0][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[0][skip_ctx]);

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_16_c,
            &mut cdfs.eob_base[0][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[0][1][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_16_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[0][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob-ctx thresholds: sw*sh = imin(w,8)*imin(h,8) = 1*1 = 1
    let ctx_e = 1 + (eob > 2) as usize + (eob > 4) as usize;
    let rc = SCAN_4X4[eob] as usize;
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[0][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[0][1][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X4[i] as usize;
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[0][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[0][1][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[0][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[5] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[0][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X4[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// 4x4 LUMA coefficient coder (TX_4X4, plane 0) with intra tx-type signalling.
/// Mirrors `encode_4x4_chroma_coeffs` but on the luma coef CDFs and emits the
/// `txtp4` symbol (IDTX=0/DCT_DCT=1/ADST_ADST=4) after `all_zero=0`.
pub(crate) fn encode_tx4_luma_coeffs_adapt(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 16],
    skip_ctx: usize,
    dcs_ctx: usize,
    y_mode: usize,
    txtp: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_4X4) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[0][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[0][skip_ctx]);
    enc.encode_symbol(txtp, &mut cdfs.txtp4[y_mode]); // luma TX_4X4: 0=IDTX,1=DCT_DCT,4=ADST_ADST

    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_16_l,
            &mut cdfs.eob_base[0][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[0][0][0],
        );
        return res_ctx;
    }
    // eob_pt
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_16_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[0][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 80];
    // eob-ctx thresholds: sw*sh = imin(w,8)*imin(h,8) = 1*1 = 1
    let ctx_e = 1 + (eob > 2) as usize + (eob > 4) as usize;
    let rc = SCAN_4X4[eob] as usize;
    let (ex, ey) = (rc >> 2, rc & 3);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[0][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[0][0][bc]);
    }
    levels[ex * 4 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_4X4[i] as usize;
        let (x, y) = (rc_i >> 2, rc_i & 3);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF, 4);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[0][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[0][0][bc as usize]);
        }
        levels[x * 4 + y] = level_byte(m);
    }
    // DC
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[0][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[4] as u32 + levels[5] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[0][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_4X4[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// 4:2:2 chroma coefficient coder for an `RTX_16X32` block (16 wide x 32 tall,
/// 512 coeffs, `cf[fx*32+fy]`). `txsize_sqr_up[RTX_16X32] = TX_32X32`, so the
/// base/br/eob-base/eob-hi/dc-sign/skip CDFs are coef-CDF class `ctx=3`; only
/// the eob_pt CDF (`eob_bin_512`), the scan, the w<h lo-ctx offsets and the
/// eob-ctx thresholds (512>>3=64, 512>>2=128) differ from TX_32X32.
pub(crate) fn encode_16x32_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 512],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X32) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]);
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_512_c,
            &mut cdfs.eob_base[3][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[3][1][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_512_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 640]; // stride 32, max read (15+2)*32+(31+2)=577
    let ctx_e = 1 + (eob > 64) as usize + (eob > 128) as usize;
    let rc = SCAN_16X32[eob] as usize;
    let (ex, ey) = (rc >> 5, rc & 31);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][1][bc]);
    }
    levels[ex * 32 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X32[i] as usize;
        let (x, y) = (rc_i >> 5, rc_i & 31);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 32);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][1][bc as usize]);
        }
        levels[x * 32 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[32] as u32 + levels[33] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X32[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma coeff coder for RTX_16X32 (16 wide x 32 tall).
#[cfg(any())]
pub(crate) fn encode_16x32_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 512],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_16X32) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]);
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_512_l,
            &mut cdfs.eob_base[3][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[3][0][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_512_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 640]; // stride 32, max read (15+2)*32+(31+2)=577
    let ctx_e = 1 + (eob > 64) as usize + (eob > 128) as usize;
    let rc = SCAN_16X32[eob] as usize;
    let (ex, ey) = (rc >> 5, rc & 31);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][0][bc]);
    }
    levels[ex * 32 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_16X32[i] as usize;
        let (x, y) = (rc_i >> 5, rc_i & 31);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WLH, 32);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][0][bc as usize]);
        }
        levels[x * 32 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[32] as u32 + levels[33] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_16X32[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Luma coeff coder for RTX_32X16 (32 wide x 16 tall).
#[cfg(any())]
pub(crate) fn encode_32x16_luma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 512],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_32X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]);
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_512_l,
            &mut cdfs.eob_base[3][0][0],
            &mut cdfs.dc_sign[0][dcs_ctx],
            &mut cdfs.br_tok[3][0][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_512_l);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][0][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 640]; // stride 32, max read (15+2)*32+(31+2)=577
    let ctx_e = 1 + (eob > 64) as usize + (eob > 128) as usize;
    let rc = SCAN_32X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][0][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][0][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_32X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][0][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][0][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][0][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][0][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[0][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_32X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

/// Chroma coeff coder for RTX_32X16 (32 wide x 16 tall).
#[cfg(any())]
pub(crate) fn encode_32x16_chroma_coeffs(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cf: &[i32; 512],
    skip_ctx: usize,
    dcs_ctx: usize,
) -> u8 {
    let Some((eob, cul)) = eob_and_cul(cf, &SCAN_32X16) else {
        enc.encode_symbol(1, &mut cdfs.txb_skip[3][skip_ctx]);
        return 0x40;
    };
    enc.encode_symbol(0, &mut cdfs.txb_skip[3][skip_ctx]);
    let dc_sign_bits: u8 = if cf[0] == 0 {
        1 << 6
    } else if cf[0] < 0 {
        0
    } else {
        2 << 6
    };
    let res_ctx = (cul.min(63) as u8) | dc_sign_bits;
    if eob == 0 {
        encode_dc_tail(
            enc,
            cf[0],
            &mut cdfs.eob_bin_512_c,
            &mut cdfs.eob_base[3][1][0],
            &mut cdfs.dc_sign[1][dcs_ctx],
            &mut cdfs.br_tok[3][1][0],
        );
        return res_ctx;
    }
    let eob_bin = if eob < 2 {
        eob
    } else {
        32 - (eob as u32).leading_zeros() as usize
    };
    enc.encode_symbol(eob_bin, &mut cdfs.eob_bin_512_c);
    if eob_bin > 1 {
        let nbits = eob_bin - 2;
        let hi = (eob >> nbits) & 1;
        enc.encode_symbol(hi, &mut cdfs.eob_hi[3][1][eob_bin]);
        for b in (0..nbits).rev() {
            enc.encode_bool((eob >> b) & 1 == 1, 16384);
        }
    }
    let mut levels = [0u8; 640]; // stride 32, max read (15+2)*32+(31+2)=577
    let ctx_e = 1 + (eob > 64) as usize + (eob > 128) as usize;
    let rc = SCAN_32X16[eob] as usize;
    let (ex, ey) = (rc >> 4, rc & 15);
    let m = cf[rc].unsigned_abs();
    let eob_tok = m.min(3) - 1;
    enc.encode_symbol(eob_tok as usize, &mut cdfs.eob_base[3][1][ctx_e]);
    if eob_tok == 2 {
        let bc = if (ex | ey) > 1 { 14 } else { 7 };
        encode_hi_tok(enc, m, &mut cdfs.br_tok[3][1][bc]);
    }
    levels[ex * 16 + ey] = level_byte(m);
    for i in (1..eob).rev() {
        let rc_i = SCAN_32X16[i] as usize;
        let (x, y) = (rc_i >> 4, rc_i & 15);
        let (ctx, hi_mag) = get_lo_ctx_2d(&levels, x, y, &LO_CTX_OFF_WGH, 16);
        let m = cf[rc_i].unsigned_abs();
        let tok = m.min(3);
        enc.encode_symbol(tok as usize, &mut cdfs.base_tok[3][1][ctx]);
        if tok == 3 {
            let mag = hi_mag & 63;
            let bc = (if (y | x) > 1 { 14 } else { 7 }) + if mag > 12 { 6 } else { (mag + 1) >> 1 };
            encode_hi_tok(enc, m, &mut cdfs.br_tok[3][1][bc as usize]);
        }
        levels[x * 16 + y] = level_byte(m);
    }
    let dm = cf[0].unsigned_abs();
    let dc_tok = dm.min(3);
    enc.encode_symbol(dc_tok as usize, &mut cdfs.base_tok[3][1][0]);
    if dc_tok == 3 {
        let mag = (levels[1] as u32 + levels[16] as u32 + levels[17] as u32) & 63;
        let bc = if mag > 12 { 6 } else { (mag + 1) >> 1 };
        encode_hi_tok(enc, dm, &mut cdfs.br_tok[3][1][bc as usize]);
    }
    if cf[0] != 0 {
        enc.encode_symbol((cf[0] < 0) as usize, &mut cdfs.dc_sign[1][dcs_ctx]);
        if dm >= 15 {
            encode_golomb(enc, dm - 15);
        }
    }
    for i in 1..=eob {
        let c = cf[SCAN_32X16[i] as usize];
        if c != 0 {
            enc.encode_bool(c < 0, 16384);
            if c.unsigned_abs() >= 15 {
                encode_golomb(enc, c.unsigned_abs() - 15);
            }
        }
    }
    res_ctx
}

pub(crate) fn get_partition_ctx(a: &[u8], l: &[u8], bl: usize, x8: usize, y8: usize) -> usize {
    let sh = 4 - bl;
    ((a[x8] >> sh) & 1) as usize + ((((l[y8] >> sh) & 1) as usize) << 1)
}

/// Probability (0..32768) for the binary `is_split` decision dav1d reads at a
/// frame edge when only one of have_h/have_v is set. `top` selects
/// `gather_top_partition_prob` (have_h only → split-or-horz), else
/// `gather_left_partition_prob` (have_v only → split-or-vert). Operates on the
/// 9-value default partition CDF for the relevant block level/context.
pub(crate) fn gather_split_prob_icdf(cdf: &[u16], top: bool) -> u16 {
    let v = |s: usize| cdf[s] as i32; // live CDF is already icdf form
    let out = if top {
        (v(1) - v(4)) + v(5) + (v(8) - v(7))
    } else {
        (v(0) - v(1)) + (v(2) - v(6)) + (v(7) - v(8))
    };
    out.clamp(1, 32767) as u16
}
