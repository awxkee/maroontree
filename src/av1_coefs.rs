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

use crate::av1_tables::{LO_CTX_2D, SCAN_4X4};
use crate::cdf_tables as C;
use crate::msac_enc::Writer;

const STRIDE: usize = 4;

// hi-token (br) encoding: level in 3..=15 coded as repeated 4-sym br symbols.
fn encode_hi_tok(w: &mut Writer, cdf: &[u16], level: u32) {
    let mut rem = level - 3; // 0..=12
    for stage in 0..4 {
        let s = rem.min(3);
        w.symbol(s, cdf);
        if s < 3 || stage == 3 {
            break;
        }
        rem -= 3;
    }
}

fn get_lo_ctx(levels: &[u8], base: usize, x: usize, y: usize) -> (usize, u32) {
    let g = |o: usize| levels[base + o] as u32;
    let mut mag = g(1) + g(STRIDE); // (0,1)+(1,0)
    mag += g(STRIDE + 1); // (1,1)
    let hi_mag = mag;
    mag += g(2) + g(2 * STRIDE); // (0,2)+(2,0)
    let offset = LO_CTX_2D[y.min(4)][x.min(4)] as usize;
    (
        offset
            + if mag > 512 {
                4
            } else {
                ((mag + 64) >> 7) as usize
            },
        hi_mag,
    )
}

/// Encode one 4x4 transform block. `lev` = signed coefficient levels (raster).
/// Returns res_ctx byte to store into a[]/l[].
pub fn encode_coefs(
    w: &mut Writer,
    chroma: bool,
    lev: &[i32; 16],
    skip_ctx: usize,
    dc_sign_ctx: usize,
) -> u8 {
    let ci = chroma as usize;
    // find eob (last nonzero scan index)
    let mut eob_idx: i32 = -1;
    for i in (0..16).rev() {
        if lev[SCAN_4X4[i]] != 0 {
            eob_idx = i as i32;
            break;
        }
    }
    // all_zero / txb_skip
    let all_skip = eob_idx < 0;
    w.symbol(all_skip as u32, &C::C_SKIP[skip_ctx]);
    if all_skip {
        return 0x40;
    }
    let eob = eob_idx as usize; // scan index of last nonzero

    // ---- eob position coding ----
    let eob_bin: usize = if eob < 2 {
        eob
    } else {
        1 + (31 - (eob as u32).leading_zeros()) as usize
    };
    w.symbol(eob_bin as u32, &C::EOB_BIN16[ci]);
    if eob_bin > 1 {
        let hi_bit = (eob >> (eob_bin - 2)) & 1;
        w.symbol(hi_bit as u32, &C::EOB_HI[ci][eob_bin]);
        let nlow = eob_bin - 2;
        if nlow > 0 {
            let low = (eob & ((1 << nlow) - 1)) as u32;
            w.literal(nlow as u8, low);
        }
    }

    let mut levels = [0u8; 32]; // zeroed scratch (stride 4)
    let mut dc_tok: u32;

    if eob > 0 {
        // ---- EOB coefficient ----
        let rc = SCAN_4X4[eob];
        let x = rc >> 2;
        let y = rc & 3;
        let m = lev[rc].unsigned_abs();
        let ctx = 1 + (eob > 2) as usize + (eob > 4) as usize;
        let eob_tok = m.min(3) - 1; // 0..2
        w.symbol(eob_tok, &C::EOB_BASE[ci][ctx]);
        let level_byte: u8 = if eob_tok == 2 {
            let hi_ctx = if (x | y) > 1 { 14 } else { 7 };
            let lvl = m.min(15);
            encode_hi_tok(w, &C::BR_TOK[ci][hi_ctx], lvl);
            (lvl + (3 << 6)) as u8
        } else {
            let tok = eob_tok + 1; // 1 or 2
            tok.wrapping_mul(0x41) as u8
        };
        levels[x * STRIDE + y] = level_byte;

        // ---- AC loop i = eob-1 .. 1 ----
        let mut i = eob as i32 - 1;
        while i > 0 {
            let rc_i = SCAN_4X4[i as usize];
            let x = rc_i >> 2;
            let y = rc_i & 3;
            let base = x * STRIDE + y;
            let (ctx, hi_mag) = get_lo_ctx(&levels, base, x, y);
            let yx = y | x;
            let m = lev[rc_i].unsigned_abs();
            let tok = m.min(3);
            w.symbol(tok, &C::BASE_TOK[ci][ctx]);
            if tok == 3 {
                let mag = hi_mag & 63;
                let hi_ctx = (if yx > 1 { 14 } else { 7 })
                    + (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
                let lvl = m.min(15);
                encode_hi_tok(w, &C::BR_TOK[ci][hi_ctx], lvl);
                levels[base] = (lvl + (3 << 6)) as u8;
            } else {
                levels[base] = tok.wrapping_mul(0x41) as u8;
            }
            i -= 1;
        }

        // ---- DC (i=0) ----
        let m = lev[0].unsigned_abs();
        dc_tok = m.min(3);
        w.symbol(dc_tok, &C::BASE_TOK[ci][0]);
        if dc_tok == 3 {
            let mag = (levels[1] as u32 + levels[STRIDE] as u32 + levels[STRIDE + 1] as u32) & 63;
            let hi_ctx = (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
            let lvl = m.min(15);
            encode_hi_tok(w, &C::BR_TOK[ci][hi_ctx], lvl);
            dc_tok = lvl;
        }
    } else {
        // dc-only path
        let m = lev[0].unsigned_abs();
        let tok_br = m.min(3) - 1; // 0..2
        w.symbol(tok_br, &C::EOB_BASE[ci][0]);
        dc_tok = m.min(3);
        if tok_br == 2 {
            let lvl = m.min(15);
            encode_hi_tok(w, &C::BR_TOK[ci][0], lvl);
            dc_tok = lvl;
        }
    }

    // ---- residual & sign ----
    let mut cul_level: u32;
    let dc_sign_level: u8;

    // DC sign + magnitude
    if dc_tok == 0 {
        cul_level = 0;
        dc_sign_level = 1 << 6;
    } else {
        let dc_neg = lev[0] < 0;
        w.symbol(dc_neg as u32, &C::DC_SIGN[ci][dc_sign_ctx]);
        let dc_mag = lev[0].unsigned_abs();
        if dc_tok == 15 {
            w.write_golomb(dc_mag - 15);
            cul_level = dc_mag;
        } else {
            cul_level = dc_tok;
        }
        dc_sign_level = if dc_neg { 0 } else { 2 << 6 };
    }

    // AC signs in increasing scan-index order (i = 1..=eob, nonzero only)
    for &rc in SCAN_4X4[1..=eob].iter() {
        let v = lev[rc];
        if v == 0 {
            continue;
        }
        let neg = v < 0;
        w.bit(neg as u16); // equiprobable sign
        let mag = v.unsigned_abs();
        let tok = mag.min(15);
        if tok == 15 {
            w.write_golomb(mag - 15);
        }
        cul_level += mag;
    }

    (cul_level.min(63) as u8) | dc_sign_level
}
