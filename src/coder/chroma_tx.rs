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

// Chroma transform-type selection (`Mode_To_Txfm`) and the per-size forward / inverse chroma transform wrappers.

/// AV1 chroma transform type derived from the intra mode (`Mode_To_Txfm`)
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChromaTx {
    DctDct,
    AdstAdst,
    AdstDct,
    DctAdst,
}

/// `Mode_To_Txfm` for the chroma intra modes the encoder searches. SMOOTH/PAETH
/// (ADST_ADST), SMOOTH_V (ADST_DCT) and SMOOTH_H (DCT_ADST) need no angle_delta.
/// V_PRED (ADST_DCT) and H_PRED (DCT_ADST) are directional: they additionally
/// emit a chroma `angle_delta` symbol (delta 0), and are only offered where the
/// chroma block is >= 8x8 (4:4:4), matching AV1's `use_angle_delta`.
fn chroma_tx_for_mode(mode: usize) -> ChromaTx {
    // Mirrors dav1d_txtp_from_uvmode: the decoder derives the chroma transform
    // type from the uv_mode. (At tx sizes whose square is >= TX_32X32 the spec
    // forces DCT_DCT regardless; callers at those sizes ignore this and use DCT.)
    match mode {
        // ADST_ADST: SMOOTH, PAETH, D135 (DIAG_DOWN_RIGHT)
        m if m == PAETH_PRED || m == SMOOTH_PRED || m == D135_PRED => ChromaTx::AdstAdst,
        // ADST_DCT: SMOOTH_V, V, D113 (VERT_RIGHT), D67 (VERT_LEFT)
        m if m == SMOOTH_V_PRED || m == V_PRED || m == D113_PRED || m == VERT_LEFT_PRED => {
            ChromaTx::AdstDct
        }
        // DCT_ADST: SMOOTH_H, H, D157 (HOR_DOWN), D203 (HOR_UP)
        m if m == SMOOTH_H_PRED || m == H_PRED || m == D157_PRED || m == D203_PRED => {
            ChromaTx::DctAdst
        }
        // DCT_DCT: DC, D45 (DIAG_DOWN_LEFT), and anything else.
        _ => ChromaTx::DctDct,
    }
}

/// Forward transform + trellis quant for an 8x8 chroma block under the given
/// chroma tx kind. Returns levels + unrounded targets like the other `*_t`.
fn fwd_chroma_8x8(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 64],
    q: &Quant,
) -> ([i32; 64], [f32; 64]) {
    match tx {
        ChromaTx::DctDct => dct.dct8x8_t(resid, q),
        ChromaTx::AdstAdst => dct.adst8x8_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct8x8_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst8x8_t(resid, q),
    }
}

fn inv_chroma_8x8(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 64], q: &Quant) -> [i32; 64] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_8x8(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_8x8(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_8x8(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_8x8(levels, q),
    }
}

/// Forward transform + trellis quant for a 16x16 chroma block under the given
/// chroma tx kind (mirrors `fwd_chroma_8x8` at TX_16X16).
fn fwd_chroma_16x16(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 256],
    q: &Quant,
) -> ([i32; 256], [f32; 256]) {
    match tx {
        ChromaTx::DctDct => dct.dct16x16_t(resid, q),
        ChromaTx::AdstAdst => dct.adst16x16_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct16x16_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst16x16_t(resid, q),
    }
}

fn inv_chroma_16x16(
    idct: &IdctDispatch,
    tx: ChromaTx,
    levels: &[i32; 256],
    q: &Quant,
) -> [i32; 256] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_16x16(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_16x16(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_16x16(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_16x16(levels, q),
    }
}

/// Forward transform + quant for a 16x8 / 8x16 chroma block under the given
/// chroma tx kind (the decoder derives it from the uv_mode; rect chroma at
/// 4:4:4 rect16 leaves).
fn fwd_chroma_16x8(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 128],
    q: &Quant,
) -> ([i32; 128], [f32; 128]) {
    match tx {
        ChromaTx::DctDct => dct.dct16x8_t(resid, q),
        ChromaTx::AdstAdst => dct.adst16x8_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct16x8_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst16x8_t(resid, q),
    }
}

fn inv_chroma_16x8(
    idct: &IdctDispatch,
    tx: ChromaTx,
    levels: &[i32; 128],
    q: &Quant,
) -> [i32; 128] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_16x8(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_16x8(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_16x8(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_16x8(levels, q),
    }
}

fn fwd_chroma_8x16(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 128],
    q: &Quant,
) -> ([i32; 128], [f32; 128]) {
    match tx {
        ChromaTx::DctDct => dct.dct8x16_t(resid, q),
        ChromaTx::AdstAdst => dct.adst8x16_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct8x16_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst8x16_t(resid, q),
    }
}

fn inv_chroma_8x16(
    idct: &IdctDispatch,
    tx: ChromaTx,
    levels: &[i32; 128],
    q: &Quant,
) -> [i32; 128] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_8x16(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_8x16(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_8x16(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_8x16(levels, q),
    }
}

fn fwd_chroma_4x4(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 16],
    q: &Quant,
) -> ([i32; 16], [f32; 16]) {
    match tx {
        ChromaTx::DctDct => dct.dct4x4_t(resid, q),
        ChromaTx::AdstAdst => dct.adst4x4_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct4x4_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst4x4_t(resid, q),
    }
}

fn inv_chroma_4x4(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 16], q: &Quant) -> [i32; 16] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_4x4(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_4x4(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_4x4(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_4x4(levels, q),
    }
}

fn fwd_chroma_4x8(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 32],
    q: &Quant,
) -> ([i32; 32], [f32; 32]) {
    match tx {
        ChromaTx::DctDct => dct.dct4x8_t(resid, q),
        ChromaTx::AdstAdst => dct.adst4x8_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct4x8_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst4x8_t(resid, q),
    }
}

fn fwd_chroma_8x4(
    dct: &DctDispatch,
    tx: ChromaTx,
    resid: &[i32; 32],
    q: &Quant,
) -> ([i32; 32], [f32; 32]) {
    match tx {
        ChromaTx::DctDct => dct.dct8x4_t(resid, q),
        ChromaTx::AdstAdst => dct.adst8x4_t(resid, q),
        ChromaTx::AdstDct => dct.adstdct8x4_t(resid, q),
        ChromaTx::DctAdst => dct.dctadst8x4_t(resid, q),
    }
}

fn inv_chroma_8x4(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_8x4(levels, q),
        ChromaTx::AdstAdst => crate::idct::iadst_dequant_8x4(levels, q),
        ChromaTx::AdstDct => crate::idct::iadstdct_dequant_8x4(levels, q),
        ChromaTx::DctAdst => crate::idct::idctadst_dequant_8x4(levels, q),
    }
}

fn inv_chroma_4x8(idct: &IdctDispatch, tx: ChromaTx, levels: &[i32; 32], q: &Quant) -> [i32; 32] {
    match tx {
        ChromaTx::DctDct => idct.idct_dequant_4x8(levels, q),
        ChromaTx::AdstAdst => idct.iadst_dequant_4x8(levels, q),
        ChromaTx::AdstDct => idct.iadstdct_dequant_4x8(levels, q),
        ChromaTx::DctAdst => idct.idctadst_dequant_4x8(levels, q),
    }
}
