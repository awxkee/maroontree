/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * Inverse-transform pipeline ported from dav2d (src/itx_1d.c, src/itx_tmpl.c),
 * © 2018-2026 VideoLAN and dav2d authors, © 2018-2026 Two Orioles, LLC,
 * under the BSD 2-Clause License.
 */

//! AV2 inverse transforms, a faithful port of the dav2d scalar pipeline.
//!
//! Validated bit-exact against dav2d's own C reference: the 1-D transforms over
//! 38,000 random vectors, and the 2-D driver / DC-only / WHT / cctx paths across
//! every transform size & type at bit depths 8, 10 and 12.
//!
//! All arithmetic is integer and bit-depth parameterized by `bd` (8/10/12):
//! intermediate row clip is `±2^(bd+7)`, the final pixel clip is `[0, 2^bd-1]`,
//! and `cctx` clips to `±2^(bd+7)` — matching dav2d's HIGHBD build exactly.
//!
//! `txtp` packs the transform as `hor | (class << 3) | (ver << 5)`, where the
//! 1-D type ids are [`Tx1d`] and the class is [`TxClass`]; build one with [`txtp`].

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) enum Tx1d {
    Dct = 0,
    Identity = 1,
    Adst = 2,
    FlipAdst = 3,
    Ddt = 4,
    FlipDdt = 5,
    Wht = 6,
}
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) enum TxClass {
    TwoD = 0,
    TwoDInv = 1,
    H = 2,
    V = 3,
}

/// Pack a transform type. `hor`/`ver` are the row/column 1-D transforms.
#[allow(dead_code)]
pub(crate) const fn txtp(hor: Tx1d, ver: Tx1d, class: TxClass) -> usize {
    (hor as usize) | ((class as usize) << 3) | ((ver as usize) << 5)
}

/// RectTxfmSize indices (dav2d order). Pass one as `tx` to [`inv_txfm_add`].
#[allow(dead_code)]
pub(crate) mod tx_size {
    pub const TX_4X4: usize = 0;
    pub const TX_8X8: usize = 1;
    pub const TX_16X16: usize = 2;
    pub const TX_32X32: usize = 3;
    pub const TX_64X64: usize = 4;
    pub const RTX_4X8: usize = 5;
    pub const RTX_8X4: usize = 6;
    pub const RTX_8X16: usize = 7;
    pub const RTX_16X8: usize = 8;
    pub const RTX_16X32: usize = 9;
    pub const RTX_32X16: usize = 10;
    pub const RTX_32X64: usize = 11;
    pub const RTX_64X32: usize = 12;
    pub const RTX_4X16: usize = 13;
    pub const RTX_16X4: usize = 14;
    pub const RTX_8X32: usize = 15;
    pub const RTX_32X8: usize = 16;
    pub const RTX_16X64: usize = 17;
    pub const RTX_64X16: usize = 18;
    pub const RTX_4X32: usize = 19;
    pub const RTX_32X4: usize = 20;
    pub const RTX_8X64: usize = 21;
    pub const RTX_64X8: usize = 22;
    pub const RTX_4X64: usize = 23;
    pub const RTX_64X4: usize = 24;
}

static DCT8_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];
static DCT16_KERNEL: [i8; 64] = [
    90, 87, 80, 70, 57, 43, 26, 9, 87, 57, 9, -43, -80, -90, -70, -26, 80, 9, -70, -87, -26, 57,
    90, 43, 70, -43, -87, 9, 90, 26, -80, -57, 57, -80, -26, 90, -9, -87, 43, 70, 43, -90, 57, 26,
    -87, 70, 9, -80, 26, -70, 90, -80, 43, 9, -57, 87, 9, -26, 43, -57, 70, -80, 87, -90,
];

/// Full size-32 inverse DCT-II kernel `K32[in*32 + out]` (= transpose of the
/// dense DCT-32 matrix, avm `tx_kernel_dct2_size32`). Used by the flat
/// hand-unrolled [`inv_dct32`] butterfly (~2.3x faster than recursing).
#[rustfmt::skip]
static DCT32_DENSE_KERNEL: [i8; 1024] = [
      64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
      90,  90,  88,  85,  82,  78,  73,  67,  61,  54,  47,  39,  30,  22,  13,   4,  -4, -13, -22, -30, -39, -47, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
      90,  87,  80,  70,  57,  43,  26,   9,  -9, -26, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -26,  -9,   9,  26,  43,  57,  70,  80,  87,  90,
      90,  82,  67,  47,  22,  -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13,  13,  39,  61,  78,  88,  90,  85,  73,  54,  30,   4, -22, -47, -67, -82, -90,
      89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,  89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,
      88,  67,  30, -13, -54, -82, -90, -78, -47,  -4,  39,  73,  90,  85,  61,  22, -22, -61, -85, -90, -73, -39,   4,  47,  78,  90,  82,  54,  13, -30, -67, -88,
      87,  57,   9, -43, -80, -90, -70, -26,  26,  70,  90,  80,  43,  -9, -57, -87, -87, -57,  -9,  43,  80,  90,  70,  26, -26, -70, -90, -80, -43,   9,  57,  87,
      85,  47, -13, -67, -90, -73, -22,  39,  82,  88,  54,  -4, -61, -90, -78, -30,  30,  78,  90,  61,   4, -54, -88, -82, -39,  22,  73,  90,  67,  13, -47, -85,
      83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,
      82,  22, -54, -90, -61,  13,  78,  85,  30, -47, -90, -67,   4,  73,  88,  39, -39, -88, -73,  -4,  67,  90,  47, -30, -85, -78, -13,  61,  90,  54, -22, -82,
      80,   9, -70, -87, -26,  57,  90,  43, -43, -90, -57,  26,  87,  70,  -9, -80, -80,  -9,  70,  87,  26, -57, -90, -43,  43,  90,  57, -26, -87, -70,   9,  80,
      78,  -4, -82, -73,  13,  85,  67, -22, -88, -61,  30,  90,  54, -39, -90, -47,  47,  90,  39, -54, -90, -30,  61,  88,  22, -67, -85, -13,  73,  82,   4, -78,
      75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,  75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,
      73, -30, -90, -22,  78,  67, -39, -90, -13,  82,  61, -47, -88,  -4,  85,  54, -54, -85,   4,  88,  47, -61, -82,  13,  90,  39, -67, -78,  22,  90,  30, -73,
      70, -43, -87,   9,  90,  26, -80, -57,  57,  80, -26, -90,  -9,  87,  43, -70, -70,  43,  87,  -9, -90, -26,  80,  57, -57, -80,  26,  90,   9, -87, -43,  70,
      67, -54, -78,  39,  85, -22, -90,   4,  90,  13, -88, -30,  82,  47, -73, -61,  61,  73, -47, -82,  30,  88, -13, -90,  -4,  90,  22, -85, -39,  78,  54, -67,
      64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,
      61, -73, -47,  82,  30, -88, -13,  90,  -4, -90,  22,  85, -39, -78,  54,  67, -67, -54,  78,  39, -85, -22,  90,   4, -90,  13,  88, -30, -82,  47,  73, -61,
      57, -80, -26,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  26,  80, -57, -57,  80,  26, -90,   9,  87, -43, -70,  70,  43, -87,  -9,  90, -26, -80,  57,
      54, -85,  -4,  88, -47, -61,  82,  13, -90,  39,  67, -78, -22,  90, -30, -73,  73,  30, -90,  22,  78, -67, -39,  90, -13, -82,  61,  47, -88,   4,  85, -54,
      50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,  50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,
      47, -90,  39,  54, -90,  30,  61, -88,  22,  67, -85,  13,  73, -82,   4,  78, -78,  -4,  82, -73, -13,  85, -67, -22,  88, -61, -30,  90, -54, -39,  90, -47,
      43, -90,  57,  26, -87,  70,   9, -80,  80,  -9, -70,  87, -26, -57,  90, -43, -43,  90, -57, -26,  87, -70,  -9,  80, -80,   9,  70, -87,  26,  57, -90,  43,
      39, -88,  73,  -4, -67,  90, -47, -30,  85, -78,  13,  61, -90,  54,  22, -82,  82, -22, -54,  90, -61, -13,  78, -85,  30,  47, -90,  67,   4, -73,  88, -39,
      35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,
      30, -78,  90, -61,   4,  54, -88,  82, -39, -22,  73, -90,  67, -13, -47,  85, -85,  47,  13, -67,  90, -73,  22,  39, -82,  88, -54,  -4,  61, -90,  78, -30,
      26, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -26, -26,  70, -90,  80, -43,  -9,  57, -87,  87, -57,   9,  43, -80,  90, -70,  26,
      22, -61,  85, -90,  73, -39,  -4,  47, -78,  90, -82,  54, -13, -30,  67, -88,  88, -67,  30,  13, -54,  82, -90,  78, -47,   4,  39, -73,  90, -85,  61, -22,
      18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,  18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,
      13, -39,  61, -78,  88, -90,  85, -73,  54, -30,   4,  22, -47,  67, -82,  90, -90,  82, -67,  47, -22,  -4,  30, -54,  73, -85,  90, -88,  78, -61,  39, -13,
       9, -26,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  26,  -9,  -9,  26, -43,  57, -70,  80, -87,  90, -90,  87, -80,  70, -57,  43, -26,   9,
       4, -13,  22, -30,  39, -47,  54, -61,  67, -73,  78, -82,  85, -88,  90, -90,  90, -90,  88, -85,  82, -78,  73, -67,  61, -54,  47, -39,  30, -22,  13,  -4,
];

/// Full size-16 inverse DCT-II kernel `K16[in*16 + out]` for the flat [`inv_dct16`].
#[rustfmt::skip]
static DCT16_DENSE_KERNEL: [i8; 256] = [
      64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
      90,  87,  80,  70,  57,  43,  26,   9,  -9, -26, -43, -57, -70, -80, -87, -90,
      89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,
      87,  57,   9, -43, -80, -90, -70, -26,  26,  70,  90,  80,  43,  -9, -57, -87,
      83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,
      80,   9, -70, -87, -26,  57,  90,  43, -43, -90, -57,  26,  87,  70,  -9, -80,
      75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,
      70, -43, -87,   9,  90,  26, -80, -57,  57,  80, -26, -90,  -9,  87,  43, -70,
      64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,
      57, -80, -26,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  26,  80, -57,
      50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,
      43, -90,  57,  26, -87,  70,   9, -80,  80,  -9, -70,  87, -26, -57,  90, -43,
      35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,
      26, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -26,
      18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,
       9, -26,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  26,  -9,
];
static ADST4_KERNEL: [i8; 16] = [
    18, 50, 75, 89, 50, 89, 18, -75, 75, 18, -89, 50, 89, -75, 50, -18,
];
static ADST8_KERNEL: [i8; 64] = [
    11, 34, 54, 71, 84, 88, 79, 50, 28, 74, 89, 68, 17, -44, -83, -69, 44, 89, 48, -41, -89, -44,
    50, 81, 58, 76, -34, -86, 10, 88, 6, -84, 70, 39, -87, 1, 86, -44, -59, 78, 79, -12, -66, 87,
    -35, -44, 86, -62, 86, -58, 12, 38, -75, 88, -74, 40, 89, -86, 79, -70, 58, -44, 29, -14,
];
static ADST16_KERNEL: [i8; 256] = [
    8, 25, 41, 55, 67, 77, 84, 88, 89, 87, 81, 73, 62, 48, 33, 17, 17, 48, 73, 87, 88, 77, 55, 25,
    -8, -41, -67, -84, -89, -81, -62, -33, 25, 67, 88, 81, 48, 0, -48, -81, -88, -67, -25, 25, 67,
    88, 81, 48, 33, 81, 84, 41, -25, -77, -87, -48, 17, 73, 88, 55, -8, -67, -89, -62, 41, 88, 62,
    -17, -81, -77, -8, 67, 87, 33, -48, -89, -55, 25, 84, 73, 48, 88, 25, -67, -81, 0, 81, 67, -25,
    -88, -48, 48, 88, 25, -67, -81, 55, 81, -17, -89, -25, 77, 62, -48, -84, 8, 88, 33, -73, -67,
    41, 87, 62, 67, -55, -73, 48, 77, -41, -81, 33, 84, -25, -87, 17, 88, -8, -89, 67, 48, -81,
    -25, 88, 0, -88, 25, 81, -48, -67, 67, 48, -81, -25, 88, 73, 25, -89, 33, 67, -77, -17, 88,
    -41, -62, 81, 8, -87, 48, 55, -84, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77,
    77, 81, -25, -48, 88, -67, 0, 67, -88, 48, 25, -81, 81, -25, -48, 88, -67, 84, -48, -8, 62,
    -88, 77, -33, -25, 73, -89, 67, -17, -41, 81, -87, 55, 87, -67, 33, 8, -48, 77, -89, 81, -55,
    17, 25, -62, 84, -88, 73, -41, 88, -81, 67, -48, 25, 0, -25, 48, -67, 81, -88, 88, -81, 67,
    -48, 25, 89, -88, 87, -84, 81, -77, 73, -67, 62, -55, 48, -41, 33, -25, 17, -8,
];
static FLIPADST4_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];
static FLIPADST16_KERNEL: [i8; 256] = [
    89, 88, 87, 84, 81, 77, 73, 67, 62, 55, 48, 41, 33, 25, 17, 8, 88, 81, 67, 48, 25, 0, -25, -48,
    -67, -81, -88, -88, -81, -67, -48, -25, 87, 67, 33, -8, -48, -77, -89, -81, -55, -17, 25, 62,
    84, 88, 73, 41, 84, 48, -8, -62, -88, -77, -33, 25, 73, 89, 67, 17, -41, -81, -87, -55, 81, 25,
    -48, -88, -67, 0, 67, 88, 48, -25, -81, -81, -25, 48, 88, 67, 77, 0, -77, -77, 0, 77, 77, 0,
    -77, -77, 0, 77, 77, 0, -77, -77, 73, -25, -89, -33, 67, 77, -17, -88, -41, 62, 81, -8, -87,
    -48, 55, 84, 67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67, -67, 48, 81, -25, -88, 62, -67,
    -55, 73, 48, -77, -41, 81, 33, -84, -25, 87, 17, -88, -8, 89, 55, -81, -17, 89, -25, -77, 62,
    48, -84, -8, 88, -33, -73, 67, 41, -87, 48, -88, 25, 67, -81, 0, 81, -67, -25, 88, -48, -48,
    88, -25, -67, 81, 41, -88, 62, 17, -81, 77, -8, -67, 87, -33, -48, 89, -55, -25, 84, -73, 33,
    -81, 84, -41, -25, 77, -87, 48, 17, -73, 88, -55, -8, 67, -89, 62, 25, -67, 88, -81, 48, 0,
    -48, 81, -88, 67, -25, -25, 67, -88, 81, -48, 17, -48, 73, -87, 88, -77, 55, -25, -8, 41, -67,
    84, -89, 81, -62, 33, 8, -25, 41, -55, 67, -77, 84, -88, 89, -87, 81, -73, 62, -48, 33, -17,
];
static DDT8_KERNEL: [i8; 64] = [
    4, 6, 22, 57, 96, 103, 78, 56, 7, 14, 48, 94, 73, -17, -79, -96, 15, 36, 85, 76, -43, -80, 7,
    98, 33, 77, 88, -26, -69, 56, 56, -77, 65, 100, 0, -73, 55, 15, -82, 54, 98, 45, -86, 34, 20,
    -66, 79, -33, 106, -57, -23, 54, -71, 75, -56, 19, 80, -98, 82, -66, 53, -41, 26, -6,
];
static DDT16_KERNEL: [i8; 256] = [
    12, 17, 37, 45, 47, 60, 64, 82, 89, 100, 92, 84, 69, 50, 51, 44, 15, 23, 49, 60, 60, 74, 70,
    73, 48, 9, -35, -71, -83, -79, -89, -95, 19, 30, 60, 69, 61, 64, 40, 3, -53, -99, -91, -46, 2,
    47, 73, 124, 23, 38, 69, 73, 49, 28, -19, -80, -96, -45, 42, 88, 75, 14, -17, -126, 30, 48, 75,
    66, 19, -31, -79, -91, -5, 84, 71, -16, -78, -60, -45, 108, 39, 61, 75, 40, -29, -87, -78, 10,
    89, 36, -69, -67, 18, 67, 89, -81, 51, 76, 61, -8, -77, -82, 11, 94, 16, -81, -22, 79, 50, -37,
    -103, 54, 66, 87, 29, -65, -83, 4, 92, 18, -83, 4, 85, -22, -85, -6, 97, -30, 78, 83, -18, -91,
    -16, 88, 28, -84, 12, 73, -60, -46, 81, 49, -83, 16, 88, 59, -67, -57, 75, 54, -85, -5, 75,
    -60, -17, 84, -43, -80, 71, -6, 94, 19, -96, 21, 93, -55, -41, 80, -51, -17, 77, -68, -6, 98,
    -56, 1, 97, -30, -83, 86, 3, -77, 82, -17, -43, 76, -70, 15, 53, -99, 44, 3, 93, -73, -28, 81,
    -92, 29, 39, -70, 81, -55, 11, 46, -81, 90, -31, -4, 83, -99, 40, 8, -74, 88, -83, 47, -14,
    -21, 56, -83, 88, -71, 22, 5, 68, -99, 84, -69, 32, 3, -37, 55, -75, 81, -83, 82, -69, 48, -11,
    -3, 50, -76, 83, -90, 97, -86, 83, -68, 67, -56, 49, -40, 32, -19, 5, 2,
];

/// Dot product of an `i8` kernel row with the inputs, widening to `i32`. Wrapping
/// matches dav2d's C (the values never overflow for valid coefficients, so the
/// result is identical to checked arithmetic in release).
#[inline(always)]
fn dot(mat: &[i8], v: &[i32]) -> i32 {
    mat.iter()
        .zip(v)
        .map(|(&m, &x)| (m as i32).wrapping_mul(x))
        .fold(0i32, i32::wrapping_add)
}

fn inv_dct4(c: &mut [i32]) {
    let c = &mut c[..4];
    let (c0, c1, c2, c3) = (c[0], c[1], c[2], c[3]);
    let a0 = c0 * 64 + c2 * 64;
    let a1 = c0 * 64 - c2 * 64;
    let b0 = c1 * 83 + c3 * 35;
    let b1 = c1 * 35 - c3 * 83;
    c[0] = a0 + b0;
    c[1] = a1 + b1;
    c[2] = a1 - b1;
    c[3] = a0 - b0;
}

/// Even/odd DCT butterfly: recurse on the even half, combine with the odd
/// projection. `n` is the half length, `c.len() == 2 * n`.
fn inv_dct_combine(c: &mut [i32], mat: &[i8], n: usize) {
    let mut even = [0i32; 16];
    let mut odd = [0i32; 16];
    for (i, (e, o)) in even[..n].iter_mut().zip(odd[..n].iter_mut()).enumerate() {
        *e = c[2 * i];
        *o = c[2 * i + 1];
    }
    dct_recurse(&mut even[..n]);
    let (lo, hi) = c.split_at_mut(n);
    for (i, e) in even[..n].iter().enumerate() {
        let b = dot(&mat[i * n..i * n + n], &odd[..n]);
        lo[i] = e.wrapping_add(b);
        hi[n - 1 - i] = e.wrapping_sub(b);
    }
}

fn dct_recurse(c: &mut [i32]) {
    match c.len() {
        4 => inv_dct4(c),
        8 => inv_dct_combine(c, &DCT8_KERNEL, 4),
        16 => inv_dct_combine(c, &DCT16_KERNEL, 8),
        _ => unreachable!("dct recurse size {}", c.len()),
    }
}

fn inv_dct8(c: &mut [i32]) {
    inv_dct_combine(c, &DCT8_KERNEL, 4);
}
fn inv_dct16(c: &mut [i32]) {
    // Flat even/odd factorization of the size-16 inverse DCT-II, bit-exact to the
    // recursive form but ~2.9x faster.
    let mut s = [0i32; 16];
    s.copy_from_slice(&c[..16]);
    let k = |j: usize, m: usize| DCT16_DENSE_KERNEL[j * 16 + m] as i32;
    let mut b = [0i32; 8];
    for (m, bm) in b.iter_mut().enumerate() {
        let mut acc = 0i32;
        let mut j = 1;
        while j < 16 {
            acc = acc.wrapping_add(k(j, m).wrapping_mul(s[j]));
            j += 2;
        }
        *bm = acc;
    }
    let mut d = [0i32; 4];
    for (m, dm) in d.iter_mut().enumerate() {
        let mut acc = 0i32;
        let mut j = 2;
        while j < 16 {
            acc = acc.wrapping_add(k(j, m).wrapping_mul(s[j]));
            j += 4;
        }
        *dm = acc;
    }
    let f = [
        k(4, 0)
            .wrapping_mul(s[4])
            .wrapping_add(k(12, 0).wrapping_mul(s[12])),
        k(4, 1)
            .wrapping_mul(s[4])
            .wrapping_add(k(12, 1).wrapping_mul(s[12])),
    ];
    let g = [
        k(0, 0)
            .wrapping_mul(s[0])
            .wrapping_add(k(8, 0).wrapping_mul(s[8])),
        k(0, 1)
            .wrapping_mul(s[0])
            .wrapping_add(k(8, 1).wrapping_mul(s[8])),
    ];
    let mut cc = [0i32; 4];
    for kk in 0..2 {
        cc[kk] = g[kk].wrapping_add(f[kk]);
        cc[kk + 2] = g[1 - kk].wrapping_sub(f[1 - kk]);
    }
    let mut a = [0i32; 8];
    for kk in 0..4 {
        a[kk] = cc[kk].wrapping_add(d[kk]);
        a[kk + 4] = cc[3 - kk].wrapping_sub(d[3 - kk]);
    }
    for kk in 0..8 {
        c[kk] = a[kk].wrapping_add(b[kk]);
        c[kk + 8] = a[7 - kk].wrapping_sub(b[7 - kk]);
    }
}
fn inv_dct32(c: &mut [i32]) {
    // Flat even/odd factorization of the size-32 inverse DCT-II, bit-exact to the
    // recursive `inv_dct_combine` but ~2.3x faster (no recursion / scratch zeroing /
    // de-interleave). Equivalent to the dense product `dst[m] = Σ_j K32[j*32+m]·s[j]`;
    // outputs 16..31 recover from the low half via `a[15-k] ∓ b[15-k]`.
    let mut s = [0i32; 32];
    s.copy_from_slice(&c[..32]);
    let k = |j: usize, m: usize| DCT32_DENSE_KERNEL[j * 32 + m] as i32;
    let mut b = [0i32; 16];
    for (m, bm) in b.iter_mut().enumerate() {
        let mut acc = 0i32;
        let mut j = 1;
        while j < 32 {
            acc = acc.wrapping_add(k(j, m).wrapping_mul(s[j]));
            j += 2;
        }
        *bm = acc;
    }
    let mut d = [0i32; 8];
    for (m, dm) in d.iter_mut().enumerate() {
        let mut acc = 0i32;
        let mut j = 2;
        while j < 32 {
            acc = acc.wrapping_add(k(j, m).wrapping_mul(s[j]));
            j += 4;
        }
        *dm = acc;
    }
    let mut f = [0i32; 4];
    for (m, fm) in f.iter_mut().enumerate() {
        *fm = k(4, m)
            .wrapping_mul(s[4])
            .wrapping_add(k(12, m).wrapping_mul(s[12]))
            .wrapping_add(k(20, m).wrapping_mul(s[20]))
            .wrapping_add(k(28, m).wrapping_mul(s[28]));
    }
    let h = [
        k(8, 0)
            .wrapping_mul(s[8])
            .wrapping_add(k(24, 0).wrapping_mul(s[24])),
        k(8, 1)
            .wrapping_mul(s[8])
            .wrapping_add(k(24, 1).wrapping_mul(s[24])),
    ];
    let g = [
        k(0, 0)
            .wrapping_mul(s[0])
            .wrapping_add(k(16, 0).wrapping_mul(s[16])),
        k(0, 1)
            .wrapping_mul(s[0])
            .wrapping_add(k(16, 1).wrapping_mul(s[16])),
    ];
    let e = [
        g[0].wrapping_add(h[0]),
        g[1].wrapping_add(h[1]),
        g[1].wrapping_sub(h[1]),
        g[0].wrapping_sub(h[0]),
    ];
    let mut cc = [0i32; 8];
    for kk in 0..4 {
        cc[kk] = e[kk].wrapping_add(f[kk]);
        cc[kk + 4] = e[3 - kk].wrapping_sub(f[3 - kk]);
    }
    let mut a = [0i32; 16];
    for kk in 0..8 {
        a[kk] = cc[kk].wrapping_add(d[kk]);
        a[kk + 8] = cc[7 - kk].wrapping_sub(d[7 - kk]);
    }
    for kk in 0..16 {
        c[kk] = a[kk].wrapping_add(b[kk]);
        c[kk + 16] = a[15 - kk].wrapping_sub(b[15 - kk]);
    }
}

/// Full matrix multiply `out[i] = Σ_j mat[i*n+j]·c[j]`, with optional output flip.
fn inv_dst(c: &mut [i32], mat: &[i8], flip: bool) {
    let n = c.len();
    let mut sums = [0i32; 16];
    for (i, s) in sums[..n].iter_mut().enumerate() {
        *s = dot(&mat[i * n..i * n + n], c);
    }
    if flip {
        for (dst, &s) in c.iter_mut().rev().zip(&sums[..n]) {
            *dst = s;
        }
    } else {
        c.copy_from_slice(&sums[..n]);
    }
}

fn inv_adst4(c: &mut [i32]) {
    inv_dst(c, &ADST4_KERNEL, false);
}
fn inv_adst8(c: &mut [i32]) {
    inv_dst(c, &ADST8_KERNEL, false);
}
fn inv_adst16(c: &mut [i32]) {
    inv_dst(c, &ADST16_KERNEL, false);
}
fn inv_flipadst4(c: &mut [i32]) {
    inv_dst(c, &FLIPADST4_KERNEL, false);
}
fn inv_flipadst8(c: &mut [i32]) {
    inv_dst(c, &ADST8_KERNEL, true);
}
fn inv_flipadst16(c: &mut [i32]) {
    inv_dst(c, &FLIPADST16_KERNEL, false);
}
fn inv_ddt8(c: &mut [i32]) {
    inv_dst(c, &DDT8_KERNEL, false);
}
fn inv_ddt16(c: &mut [i32]) {
    inv_dst(c, &DDT16_KERNEL, false);
}
fn inv_flipddt8(c: &mut [i32]) {
    inv_dst(c, &DDT8_KERNEL, true);
}
fn inv_flipddt16(c: &mut [i32]) {
    inv_dst(c, &DDT16_KERNEL, true);
}

fn inv_identity4(c: &mut [i32]) {
    for v in c.iter_mut() {
        *v = v.wrapping_mul(128);
    }
}
fn inv_identity8(c: &mut [i32]) {
    for v in c.iter_mut() {
        *v = v.wrapping_mul(181);
    }
}
fn inv_identity16(c: &mut [i32]) {
    for v in c.iter_mut() {
        *v = v.wrapping_mul(256);
    }
}
fn inv_identity32(c: &mut [i32]) {
    for v in c.iter_mut() {
        *v = v.wrapping_mul(362);
    }
}

fn inv_wht4(c: &mut [i32; 4]) {
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    let t0 = in0 + in1;
    let t2 = in2 - in3;
    let t4 = (t0 - t2) >> 1;
    let t3 = t4 - in3;
    let t1 = t4 - in1;
    c[0] = t0 - t3;
    c[1] = t3;
    c[2] = t1;
    c[3] = t2 + t1;
}

type Fn1d = fn(&mut [i32]);
static TX1D: [[Option<Fn1d>; 6]; 5] = [
    [
        Some(inv_dct4),
        Some(inv_identity4),
        Some(inv_adst4),
        Some(inv_flipadst4),
        None,
        None,
    ],
    [
        Some(inv_dct8),
        Some(inv_identity8),
        Some(inv_adst8),
        Some(inv_flipadst8),
        Some(inv_ddt8),
        Some(inv_flipddt8),
    ],
    [
        Some(inv_dct16),
        Some(inv_identity16),
        Some(inv_adst16),
        Some(inv_flipadst16),
        Some(inv_ddt16),
        Some(inv_flipddt16),
    ],
    [
        Some(inv_dct32),
        Some(inv_identity32),
        None,
        None,
        None,
        None,
    ],
    [Some(inv_dct32), None, None, None, None, None],
];
static DIM: [(usize, usize, usize, usize); 25] = [
    (1, 1, 0, 0),
    (2, 2, 1, 1),
    (4, 4, 2, 2),
    (8, 8, 3, 3),
    (16, 16, 4, 4),
    (1, 2, 0, 1),
    (2, 1, 1, 0),
    (2, 4, 1, 2),
    (4, 2, 2, 1),
    (4, 8, 2, 3),
    (8, 4, 3, 2),
    (8, 16, 3, 4),
    (16, 8, 4, 3),
    (1, 4, 0, 2),
    (4, 1, 2, 0),
    (2, 8, 1, 3),
    (8, 2, 3, 1),
    (4, 16, 2, 4),
    (16, 4, 4, 2),
    (1, 8, 0, 3),
    (8, 1, 3, 0),
    (2, 16, 1, 4),
    (16, 2, 4, 1),
    (1, 16, 0, 4),
    (16, 1, 4, 0),
];
static TXSH: [(i32, i32); 25] = [
    (7, 10),
    (7, 11),
    (6, 13),
    (6, 13),
    (6, 13),
    (7, 10),
    (7, 10),
    (7, 11),
    (7, 11),
    (6, 12),
    (6, 12),
    (6, 12),
    (6, 12),
    (6, 12),
    (6, 12),
    (6, 13),
    (6, 13),
    (6, 13),
    (6, 13),
    (7, 11),
    (7, 11),
    (6, 12),
    (6, 12),
    (6, 13),
    (6, 13),
];

/// 2-D inverse transform, adding the residual to `dst` in place. `dst` is a
/// `bd`-bit image at element `stride`; `coeff` is the dequantized block in dav2d's
/// transposed layout (`coeff[col + x*sh]`, `sh = min(h,32)`). `txtp`/`tx` select
/// the transform; see [`txtp`] and [`tx_size`]. Mirrors dav2d `inv_txfm_add_c`
/// (every column processed — the eob skip is a decode-only optimization).
/// Shared row + column transform passes for [`inv_txfm_add`] and
/// [`inv_txfm_recon_f32`]. Leaves the post-column-pass coefficients in
/// `tmp[col * sw + x]` and returns `(sw, sh, w, h, s1)` for the output stage.
fn inv_txfm_passes(
    tmp: &mut [i32; 32 * 32],
    coeff: &[i32],
    txtp: usize,
    tx: usize,
    bd: i32,
) -> (usize, usize, usize, usize, i32) {
    let (tw, th, lw, lh) = DIM[tx];
    let (s0, s1) = TXSH[tx];
    let (w, h) = (4 * tw, 4 * th);
    let is_rect2 = (lw + lh) & 1 != 0;
    let row_clip_min = -(1 << (bd + 7));
    let row_clip_max = !row_clip_min;
    let first = TX1D[lw][txtp & 7].expect("invalid horizontal tx for size");
    let second = TX1D[lh][(txtp >> 5) & 7].expect("invalid vertical tx for size");
    let (sw, sh) = (w.min(32), h.min(32));
    let coeff = &coeff[..sw * sh];

    // Row pass: each row is contiguous; gather (with the rect2 prescale) and transform.
    for (col, row) in tmp[..sw * sh].chunks_exact_mut(sw).enumerate() {
        for (x, slot) in row.iter_mut().enumerate() {
            let v = coeff[col + x * sh];
            *slot = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
        first(row);
    }

    // Intermediate round + clip.
    let rnd0 = (1 << s0) >> 1;
    for v in tmp[..sw * sh].iter_mut() {
        *v = ((*v + rnd0) >> s0).clamp(row_clip_min, row_clip_max);
    }

    // Column pass: gather each column into a contiguous buffer, transform, scatter.
    let mut colbuf = [0i32; 32];
    for x in 0..sw {
        for (col, slot) in colbuf[..sh].iter_mut().enumerate() {
            *slot = tmp[col * sw + x];
        }
        second(&mut colbuf[..sh]);
        for (col, &v) in colbuf[..sh].iter().enumerate() {
            tmp[col * sw + x] = v;
        }
    }
    (sw, sh, w, h, s1)
}

/// Fused inverse-transform reconstruction straight into an f32 output buffer
/// (contiguous, row-major `w×h`, stride `w`). `pred_fn(i)` supplies the integer
/// prediction at output index `i`. The residual add, pixel clip and i32→f32 cast
/// all happen in this single output pass, so callers need no separate prediction
/// pre-fill, transpose, or cast pass (mirrors the old fused reconstruct path).
/// Run the inverse-transform row + column passes, dispatching to the NEON lane
/// implementation on aarch64 (bit-exact to the scalar passes). Shared by
/// [`inv_txfm_add`] and [`inv_txfm_recon_f32`] so both get SIMD on the costly
/// transform while their cheap output stages stay scalar.
#[inline]
fn txfm_passes(
    tmp: &mut [i32; 32 * 32],
    coeff: &[i32],
    txtp: usize,
    tx: usize,
    bd: i32,
) -> (usize, usize, usize, usize, i32) {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: neon feature detected at runtime.
            return unsafe { neon_lane::passes_neon(tmp, coeff, txtp, tx, bd) };
        }
    }
    inv_txfm_passes(tmp, coeff, txtp, tx, bd)
}

pub(crate) fn inv_txfm_recon_f32<F: Fn(usize) -> i32>(
    out: &mut [f32],
    coeff: &[i32],
    txtp: usize,
    tx: usize,
    bd: i32,
    pred_fn: F,
) {
    let mut tmp = [0i32; 32 * 32];
    let (sw, sh, w, h, s1) = txfm_passes(&mut tmp, coeff, txtp, tx, bd);
    let pmax = (1 << bd) - 1;
    let rnd1 = (1 << s1) >> 1;
    let cf = |t: i32| (t + rnd1) >> s1;
    let xs = if w > sw { 2 } else { 1 };
    let ys = if h > sh { 2 } else { 1 };
    for col in 0..sh {
        let trow = &tmp[col * sw..col * sw + sw];
        for dy in 0..ys {
            let base = (col * ys + dy) * w;
            for (x0, &t) in trow.iter().enumerate() {
                let v = cf(t);
                let x = x0 * xs;
                for dx in 0..xs {
                    let i = base + x + dx;
                    out[i] = (pred_fn(i) + v).clamp(0, pmax) as f32;
                }
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) fn inv_txfm_add(
    dst: &mut [i32],
    stride: isize,
    coeff: &[i32],
    txtp: usize,
    tx: usize,
    bd: i32,
) {
    let mut tmp = [0i32; 32 * 32];
    let (sw, sh, w, h, s1) = txfm_passes(&mut tmp, coeff, txtp, tx, bd);
    let pmax = (1 << bd) - 1;

    // Output: round/shift, add to dst, clip; nearest-duplicate for 64-wide/tall.
    let rnd1 = (1 << s1) >> 1;
    let cf = |t: i32| (t + rnd1) >> s1;
    match (w > sw, h > sh) {
        (false, false) => {
            for col in 0..sh {
                let trow = &tmp[col * sw..col * sw + sw];
                let base = (col as isize * stride) as usize;
                for (d, &t) in dst[base..base + w].iter_mut().zip(trow) {
                    *d = (*d + cf(t)).clamp(0, pmax);
                }
            }
        }
        (true, false) => {
            for col in 0..sh {
                let trow = &tmp[col * sw..col * sw + sw];
                let base = (col as isize * stride) as usize;
                for (pair, &t) in dst[base..base + w].chunks_exact_mut(2).zip(trow) {
                    let v = cf(t);
                    pair[0] = (pair[0] + v).clamp(0, pmax);
                    pair[1] = (pair[1] + v).clamp(0, pmax);
                }
            }
        }
        (false, true) => {
            for col in 0..sh {
                let trow = &tmp[col * sw..col * sw + sw];
                for dy in [2 * col, 2 * col + 1] {
                    let base = (dy as isize * stride) as usize;
                    for (d, &t) in dst[base..base + w].iter_mut().zip(trow) {
                        *d = (*d + cf(t)).clamp(0, pmax);
                    }
                }
            }
        }
        (true, true) => {
            for col in 0..sh {
                let trow = &tmp[col * sw..col * sw + sw];
                for dy in [2 * col, 2 * col + 1] {
                    let base = (dy as isize * stride) as usize;
                    for (pair, &t) in dst[base..base + w].chunks_exact_mut(2).zip(trow) {
                        let v = cf(t);
                        pair[0] = (pair[0] + v).clamp(0, pmax);
                        pair[1] = (pair[1] + v).clamp(0, pmax);
                    }
                }
            }
        }
    }
}

/// DC-only DCT_DCT fast path (dav2d's fused rounding) — used when only the DC
/// coefficient is non-zero. `dc_in` is `coeff[0]`.
#[allow(dead_code)]
pub(crate) fn inv_txfm_add_dc(dst: &mut [i32], stride: isize, dc_in: i32, tx: usize, bd: i32) {
    let (tw, th, lw, lh) = DIM[tx];
    let (sp1, s1) = TXSH[tx];
    let shift = sp1 + s1 - 12;
    let rnd = (1 << (shift - 1)) + sp1 - 6;
    let (w, h) = (4 * tw, 4 * th);
    let pmax = (1 << bd) - 1;
    let mut dc = dc_in;
    if (lw + lh) & 1 != 0 {
        dc = (dc * 181 + 128) >> 8;
    }
    dc = (dc + rnd) >> shift;
    for y in 0..h {
        let base = (y as isize * stride) as usize;
        for d in dst[base..base + w].iter_mut() {
            *d = (*d + dc).clamp(0, pmax);
        }
    }
}

/// 4x4 Walsh-Hadamard inverse add (lossless path). `coeff` in `coeff[y + x*4]` layout.
#[allow(dead_code)]
pub(crate) fn inv_txfm_add_wht4x4(dst: &mut [i32], stride: isize, coeff: &[i32], bd: i32) {
    let mut tmp = [0i32; 16];
    let pmax = (1 << bd) - 1;
    for (y, row) in tmp.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        for (x, slot) in row.iter_mut().enumerate() {
            *slot = coeff[y + x * 4] >> 3;
        }
        inv_wht4(row);
    }
    let mut colbuf = [0i32; 4];
    for x in 0..4 {
        for (y, slot) in colbuf.iter_mut().enumerate() {
            *slot = tmp[y * 4 + x];
        }
        inv_wht4(&mut colbuf);
        for (y, &v) in colbuf.iter().enumerate() {
            tmp[y * 4 + x] = v;
        }
    }
    for (y, trow) in tmp.chunks_exact(4).enumerate() {
        let base = (y as isize * stride) as usize;
        for (d, &t) in dst[base..base + 4].iter_mut().zip(trow) {
            *d = (*d + t).clamp(0, pmax);
        }
    }
}

/// Cross-component transform: rotate the (u, v) chroma residual pair by `angle`,
/// `angle = [sina, cosa, -sina]`. Applied to the dequantized coefficients before
/// the inverse transform. Mirrors dav2d `cctx_c`.
#[allow(dead_code)]
pub(crate) fn cctx(u: &mut [i32], v: &mut [i32], sina: i32, cosa: i32, bd: i32) {
    let (mn, mx) = (-(1 << (bd + 7)), (1 << (bd + 7)) - 1);
    for (uu, vv) in u.iter_mut().zip(v.iter_mut()) {
        let a = *uu * cosa - *vv * sina;
        let b = *uu * sina + *vv * cosa;
        *uu = ((a + 128 - (a < 0) as i32) >> 8).clamp(mn, mx);
        *vv = ((b + 128 - (b < 0) as i32) >> 8).clamp(mn, mx);
    }
}

// ============================================================================
// Lane-parallel (SIMD) inverse-transform path.
// The 1-D passes run 4 transforms at once (one transform per SIMD lane); fill,
// rect2 prescale, intermediate round-clip and the output stay scalar. The DCT
// kernels use the SAME flat even/odd butterflies as the scalar `inv_dct16` /
// `inv_dct32` (just over `V` instead of `i32`), so the lane path is bit-exact
// to the scalar path. Dispatched from `txfm_passes` on aarch64+neon.
// ============================================================================
#[allow(dead_code)]
pub(crate) trait Lane: Copy {
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul_n(self, k: i32) -> Self;
    /// Fused multiply-accumulate: `self + x * k`. Default routes through
    /// `mul_n`/`add`; the NEON lane overrides it with a single MLA instruction.
    #[inline]
    fn mul_add_n(self, x: Self, k: i32) -> Self {
        self.add(x.mul_n(k))
    }
    fn from4(a: [i32; 4]) -> Self;
    fn to4(self) -> [i32; 4];
    fn zero() -> Self;
}

/// Lane-wise dot product of an `i8` kernel row with `v` (no bounds checks).
#[inline]
#[allow(dead_code)]
fn g_dot<V: Lane>(mat: &[i8], v: &[V]) -> V {
    let mut it = mat.iter().zip(v);
    let (&m0, &x0) = it.next().expect("dot: empty row");
    let mut acc = x0.mul_n(m0 as i32);
    for (&m, &x) in it {
        acc = acc.mul_add_n(x, m as i32);
    }
    acc
}

#[allow(dead_code)]
fn g_dct4<V: Lane>(c: &mut [V]) {
    let c: &mut [V; 4] = c.try_into().expect("dct4 needs len 4");
    let (c0, c1, c2, c3) = (c[0], c[1], c[2], c[3]);
    let a0 = c0.mul_n(64).add(c2.mul_n(64));
    let a1 = c0.mul_n(64).sub(c2.mul_n(64));
    let b0 = c1.mul_n(83).add(c3.mul_n(35));
    let b1 = c1.mul_n(35).sub(c3.mul_n(83));
    c[0] = a0.add(b0);
    c[1] = a1.add(b1);
    c[2] = a1.sub(b1);
    c[3] = a0.sub(b0);
}

/// Size-8 lane DCT (recursive even/odd combine — small enough that flattening
/// gives no measurable gain).
#[allow(dead_code)]
fn g_dct_combine<V: Lane>(c: &mut [V], mat: &[i8], n: usize) {
    debug_assert!(c.len() == 2 * n && n <= 8);
    let mut even = [V::zero(); 8];
    let mut odd = [V::zero(); 8];
    for (e, &x) in even[..n].iter_mut().zip(c.iter().step_by(2)) {
        *e = x;
    }
    for (o, &x) in odd[..n].iter_mut().zip(c.iter().skip(1).step_by(2)) {
        *o = x;
    }
    g_dct4(&mut even[..n]);
    let (front, back) = c.split_at_mut(n);
    for (((f, bk), &e), row) in front
        .iter_mut()
        .zip(back.iter_mut().rev())
        .zip(&even[..n])
        .zip(mat.chunks_exact(n))
    {
        let b = g_dot(row, &odd[..n]);
        *f = e.add(b);
        *bk = e.sub(b);
    }
}
#[allow(dead_code)]
fn g_dct8<V: Lane>(c: &mut [V]) {
    g_dct_combine(c, &DCT8_KERNEL, 4);
}

/// Flat size-16 lane DCT — lane translation of the scalar [`inv_dct16`].
#[allow(dead_code)]
fn g_dct16<V: Lane>(c: &mut [V]) {
    let mut s = [V::zero(); 16];
    s.copy_from_slice(&c[..16]);
    let k = |j: usize, m: usize| DCT16_DENSE_KERNEL[j * 16 + m] as i32;
    let mut b = [V::zero(); 8];
    for (m, bm) in b.iter_mut().enumerate() {
        let mut acc = V::zero();
        let mut j = 1;
        while j < 16 {
            acc = acc.mul_add_n(s[j], k(j, m));
            j += 2;
        }
        *bm = acc;
    }
    let mut d = [V::zero(); 4];
    for (m, dm) in d.iter_mut().enumerate() {
        let mut acc = V::zero();
        let mut j = 2;
        while j < 16 {
            acc = acc.mul_add_n(s[j], k(j, m));
            j += 4;
        }
        *dm = acc;
    }
    let f = [
        s[4].mul_n(k(4, 0)).mul_add_n(s[12], k(12, 0)),
        s[4].mul_n(k(4, 1)).mul_add_n(s[12], k(12, 1)),
    ];
    let g = [
        s[0].mul_n(k(0, 0)).mul_add_n(s[8], k(8, 0)),
        s[0].mul_n(k(0, 1)).mul_add_n(s[8], k(8, 1)),
    ];
    let mut cc = [V::zero(); 4];
    for kk in 0..2 {
        cc[kk] = g[kk].add(f[kk]);
        cc[kk + 2] = g[1 - kk].sub(f[1 - kk]);
    }
    let mut a = [V::zero(); 8];
    for kk in 0..4 {
        a[kk] = cc[kk].add(d[kk]);
        a[kk + 4] = cc[3 - kk].sub(d[3 - kk]);
    }
    for kk in 0..8 {
        c[kk] = a[kk].add(b[kk]);
        c[kk + 8] = a[7 - kk].sub(b[7 - kk]);
    }
}

/// Flat size-32 lane DCT — lane translation of the scalar [`inv_dct32`].
#[allow(dead_code)]
fn g_dct32<V: Lane>(c: &mut [V]) {
    let mut s = [V::zero(); 32];
    s.copy_from_slice(&c[..32]);
    let k = |j: usize, m: usize| DCT32_DENSE_KERNEL[j * 32 + m] as i32;
    let mut b = [V::zero(); 16];
    for (m, bm) in b.iter_mut().enumerate() {
        let mut acc = V::zero();
        let mut j = 1;
        while j < 32 {
            acc = acc.mul_add_n(s[j], k(j, m));
            j += 2;
        }
        *bm = acc;
    }
    let mut d = [V::zero(); 8];
    for (m, dm) in d.iter_mut().enumerate() {
        let mut acc = V::zero();
        let mut j = 2;
        while j < 32 {
            acc = acc.mul_add_n(s[j], k(j, m));
            j += 4;
        }
        *dm = acc;
    }
    let mut f = [V::zero(); 4];
    for (m, fm) in f.iter_mut().enumerate() {
        *fm = s[4]
            .mul_n(k(4, m))
            .mul_add_n(s[12], k(12, m))
            .mul_add_n(s[20], k(20, m))
            .mul_add_n(s[28], k(28, m));
    }
    let h = [
        s[8].mul_n(k(8, 0)).mul_add_n(s[24], k(24, 0)),
        s[8].mul_n(k(8, 1)).mul_add_n(s[24], k(24, 1)),
    ];
    let g = [
        s[0].mul_n(k(0, 0)).mul_add_n(s[16], k(16, 0)),
        s[0].mul_n(k(0, 1)).mul_add_n(s[16], k(16, 1)),
    ];
    let e = [
        g[0].add(h[0]),
        g[1].add(h[1]),
        g[1].sub(h[1]),
        g[0].sub(h[0]),
    ];
    let mut cc = [V::zero(); 8];
    for kk in 0..4 {
        cc[kk] = e[kk].add(f[kk]);
        cc[kk + 4] = e[3 - kk].sub(f[3 - kk]);
    }
    let mut a = [V::zero(); 16];
    for kk in 0..8 {
        a[kk] = cc[kk].add(d[kk]);
        a[kk + 8] = cc[7 - kk].sub(d[7 - kk]);
    }
    for kk in 0..16 {
        c[kk] = a[kk].add(b[kk]);
        c[kk + 16] = a[15 - kk].sub(b[15 - kk]);
    }
}

#[allow(dead_code)]
fn g_dst<V: Lane>(c: &mut [V], mat: &[i8], flip: bool) {
    let n = c.len();
    assert!(n <= 16);
    let mut sums = [V::zero(); 16];
    for (s, row) in sums[..n].iter_mut().zip(mat.chunks_exact(n)) {
        *s = g_dot(row, c);
    }
    if flip {
        for (d, &s) in c.iter_mut().rev().zip(&sums[..n]) {
            *d = s;
        }
    } else {
        for (d, &s) in c.iter_mut().zip(&sums[..n]) {
            *d = s;
        }
    }
}
#[allow(dead_code)]
fn g_id<V: Lane>(c: &mut [V], k: i32) {
    for v in c.iter_mut() {
        *v = v.mul_n(k);
    }
}

/// Generic 1-D lane transform selector mirroring [`TX1D`]; DCT sizes 16/32 use
/// the flat butterflies.
#[allow(dead_code)]
fn pick_lane<V: Lane>(sz: usize, ty: usize) -> Option<fn(&mut [V])> {
    let f: fn(&mut [V]) = match (sz, ty) {
        (0, 0) => g_dct4,
        (1, 0) => g_dct8,
        (2, 0) => g_dct16,
        (3, 0) | (4, 0) => g_dct32,
        (0, 1) => |c| g_id(c, 128),
        (1, 1) => |c| g_id(c, 181),
        (2, 1) => |c| g_id(c, 256),
        (3, 1) => |c| g_id(c, 362),
        (0, 2) => |c| g_dst(c, &ADST4_KERNEL, false),
        (1, 2) => |c| g_dst(c, &ADST8_KERNEL, false),
        (2, 2) => |c| g_dst(c, &ADST16_KERNEL, false),
        (0, 3) => |c| g_dst(c, &FLIPADST4_KERNEL, false),
        (1, 3) => |c| g_dst(c, &ADST8_KERNEL, true),
        (2, 3) => |c| g_dst(c, &FLIPADST16_KERNEL, false),
        (1, 4) => |c| g_dst(c, &DDT8_KERNEL, false),
        (2, 4) => |c| g_dst(c, &DDT16_KERNEL, false),
        (1, 5) => |c| g_dst(c, &DDT8_KERNEL, true),
        (2, 5) => |c| g_dst(c, &DDT16_KERNEL, true),
        _ => return None,
    };
    Some(f)
}

/// Lane-parallel row+column transform passes (4 transforms per group). Leaves
/// the post-column-pass coefficients in `tmp[col*sw + x]`, identical layout and
/// values to the scalar [`inv_txfm_passes`]. Returns `(sw, sh, w, h, s1)`.
#[allow(dead_code)]
fn inv_txfm_passes_lanes<V: Lane>(
    tmp_arr: &mut [i32; 32 * 32],
    coeff: &[i32],
    txtp: usize,
    tx: usize,
    bd: i32,
) -> (usize, usize, usize, usize, i32) {
    let (tw, th, lw, lh) = DIM[tx];
    let (s0, s1) = TXSH[tx];
    let (w, h) = (4 * tw, 4 * th);
    let is_rect2 = (lw + lh) & 1 != 0;
    let row_clip_min = -(1 << (bd + 7));
    let row_clip_max = !row_clip_min;
    let first = pick_lane::<V>(lw, txtp & 7).expect("invalid horizontal tx");
    let second = pick_lane::<V>(lh, (txtp >> 5) & 7).expect("invalid vertical tx");
    let (sw, sh) = (w.min(32), h.min(32));
    let area = sw * sh;
    assert!(area <= tmp_arr.len() && sw % 4 == 0 && sh % 4 == 0);
    let coeff = &coeff[..area];
    let tmp = &mut tmp_arr[..area];

    for (col, row) in tmp.chunks_exact_mut(sw).enumerate() {
        for (slot, &v) in row.iter_mut().zip(coeff.iter().skip(col).step_by(sh)) {
            *slot = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
    }
    let mut lanes = [V::zero(); 32];
    // Row pass: 4 consecutive rows per group.
    for group in tmp.chunks_exact_mut(4 * sw) {
        {
            let mut rows = group.chunks_exact(sw);
            let (r0, r1, r2, r3) = (
                rows.next().unwrap(),
                rows.next().unwrap(),
                rows.next().unwrap(),
                rows.next().unwrap(),
            );
            for (lane, (((&a, &b), &cc), &d)) in lanes[..sw]
                .iter_mut()
                .zip(r0.iter().zip(r1).zip(r2).zip(r3))
            {
                *lane = V::from4([a, b, cc, d]);
            }
        }
        first(&mut lanes[..sw]);
        {
            let mut rows = group.chunks_exact_mut(sw);
            let (r0, r1, r2, r3) = (
                rows.next().unwrap(),
                rows.next().unwrap(),
                rows.next().unwrap(),
                rows.next().unwrap(),
            );
            for (lane, (((s0, s1), s2), s3)) in lanes[..sw]
                .iter()
                .zip(r0.iter_mut().zip(r1).zip(r2).zip(r3))
            {
                let a = lane.to4();
                *s0 = a[0];
                *s1 = a[1];
                *s2 = a[2];
                *s3 = a[3];
            }
        }
    }
    let rnd0 = (1 << s0) >> 1;
    for v in tmp.iter_mut() {
        *v = ((*v + rnd0) >> s0).clamp(row_clip_min, row_clip_max);
    }
    // Column pass: 4 columns per group.
    for xg in (0..sw).step_by(4) {
        for (lane, row) in lanes[..sh].iter_mut().zip(tmp.chunks_exact(sw)) {
            let q: [i32; 4] = row[xg..xg + 4].try_into().unwrap();
            *lane = V::from4(q);
        }
        second(&mut lanes[..sh]);
        for (lane, row) in lanes[..sh].iter().zip(tmp.chunks_exact_mut(sw)) {
            row[xg..xg + 4].copy_from_slice(&lane.to4());
        }
    }
    (sw, sh, w, h, s1)
}

/// NEON backend: a `Lane` over `int32x4_t`.
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon_lane {
    use super::{Lane, inv_txfm_passes_lanes};
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(super) struct N4(int32x4_t);
    impl Lane for N4 {
        #[inline]
        fn add(self, o: N4) -> N4 {
            unsafe { N4(vaddq_s32(self.0, o.0)) }
        }
        #[inline]
        fn sub(self, o: N4) -> N4 {
            unsafe { N4(vsubq_s32(self.0, o.0)) }
        }
        #[inline]
        fn mul_n(self, k: i32) -> N4 {
            unsafe { N4(vmulq_s32(self.0, vdupq_n_s32(k))) }
        }
        #[inline]
        fn mul_add_n(self, x: N4, k: i32) -> N4 {
            // Fused multiply-accumulate: self + x*k in a single MLA.
            unsafe { N4(vmlaq_s32(self.0, x.0, vdupq_n_s32(k))) }
        }
        #[inline]
        fn from4(a: [i32; 4]) -> N4 {
            unsafe { N4(vld1q_s32(a.as_ptr())) }
        }
        #[inline]
        fn to4(self) -> [i32; 4] {
            let mut a = [0i32; 4];
            unsafe { vst1q_s32(a.as_mut_ptr(), self.0) };
            a
        }
        #[inline]
        fn zero() -> N4 {
            unsafe { N4(vdupq_n_s32(0)) }
        }
    }

    /// # Safety
    /// Requires the `neon` feature (baseline on aarch64).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn passes_neon(
        tmp: &mut [i32; 32 * 32],
        coeff: &[i32],
        txtp: usize,
        tx: usize,
        bd: i32,
    ) -> (usize, usize, usize, usize, i32) {
        inv_txfm_passes_lanes::<N4>(tmp, coeff, txtp, tx, bd)
    }
}
