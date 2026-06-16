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

use crate::bitwriter::{BitWriter, leb128};
use crate::color::MasteringDisplay;
use crate::metadata::ContentLightLevel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ObuType {
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
}

/// Wrap a payload as an OBU with `obu_has_size_field = 1`.
pub(crate) fn wrap_obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
    let mut header = BitWriter::new();
    header.f(0, 1); // obu_forbidden_bit
    header.f(obu_type as u32, 4); // obu_type
    header.f(0, 1); // obu_extension_flag
    header.f(1, 1); // obu_has_size_field
    header.f(0, 1); // obu_reserved_1bit
    let mut out = header.into_bytes(); // 1 byte
    out.extend_from_slice(&leb128(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

/// Temporal delimiter OBU (empty payload). This one IS complete.
pub(crate) fn temporal_delimiter() -> Vec<u8> {
    wrap_obu(ObuType::TemporalDelimiter, &[])
}

fn bits_for(v: u32) -> u8 {
    let mut b = 1;
    while (1u32 << b) <= v.saturating_sub(1) {
        b += 1;
    }
    b
}

#[allow(unused)]
pub(crate) fn frame_header_lossless() -> Vec<u8> {
    frame_header_lossless_tiled(1, 1)
}

/// Like [`frame_header_lossless`] but emits the tile-count increment bits needed
/// when the frame spans more than one 64x64 superblock (single tile is always
/// signalled; see `frame_header_lossy_multitile` for the tiling rationale).
pub(crate) fn frame_header_lossless_tiled(sb_cols: u32, sb_rows: u32) -> Vec<u8> {
    // Single tile: one `0` stop-bit per dimension that spans >1 superblock.
    let cols = if sb_cols > 1 { vec![false] } else { vec![] };
    let rows = if sb_rows > 1 { vec![false] } else { vec![] };
    frame_header_lossless_impl(&cols, &rows, 0, 0, false, false)
}

/// Multi-tile lossless frame header. `cols_incr` / `rows_incr` are the
/// `increment_tile_*_log2` bit sequences (computed by the encoder's tiling
/// planner); when `tile_cols_log2 + tile_rows_log2 > 0` the header also codes
/// `context_update_tile_id` + `tile_size_bytes_minus_1` (`TileSizeBytes = 4`),
/// exactly like the lossy multi-tile header.
pub(crate) fn frame_header_lossless_multitile(
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
) -> Vec<u8> {
    frame_header_lossless_impl(
        cols_incr,
        rows_incr,
        tile_cols_log2,
        tile_rows_log2,
        false,
        false,
    )
}

/// Like [`frame_header_lossless_multitile`] but terminated with `trailing_bits()`
/// for carriage in a standalone `OBU_FRAME_HEADER` (type 3).
pub(crate) fn frame_header_lossless_multitile_th(
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
) -> Vec<u8> {
    frame_header_lossless_impl(
        cols_incr,
        rows_incr,
        tile_cols_log2,
        tile_rows_log2,
        true,
        false,
    )
}

/// Monochrome (`NumPlanes == 1`) lossless frame header. Identical to
/// [`frame_header_lossless_multitile`] except the two chroma `DeltaQ` delta-coded
/// flags are omitted: `quantization_params()` reads them only for
/// `NumPlanes > 1`, so a monochrome frame must not code them or the decoder
/// misaligns the bitstream.
pub(crate) fn frame_header_lossless_mono_multitile(
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
) -> Vec<u8> {
    frame_header_lossless_impl(
        cols_incr,
        rows_incr,
        tile_cols_log2,
        tile_rows_log2,
        false,
        true,
    )
}

/// `trailing_bits()`-terminated monochrome lossless frame header for a
/// standalone `OBU_FRAME_HEADER` (type 3).
pub(crate) fn frame_header_lossless_mono_multitile_th(
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
) -> Vec<u8> {
    frame_header_lossless_impl(
        cols_incr,
        rows_incr,
        tile_cols_log2,
        tile_rows_log2,
        true,
        true,
    )
}

fn frame_header_lossless_impl(
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    trailing: bool,
    mono: bool,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    // (reduced_still_picture_header => frame_type=KEY_FRAME, show_frame=1,
    //  error_resilient_mode=1, frame_size_override=0, etc., none coded)
    w.flag(true); // disable_cdf_update = 1 (encoder is non-adaptive; decoder must not adapt)
    w.flag(false); // allow_screen_content_tools (seq force = SELECT)
    // (allow_screen_content_tools=0 => no force_integer_mv; FrameIsIntra)
    w.flag(false); // render_and_frame_size_different
    // (allow_intrabc not coded since screen content tools off)
    w.flag(true); // uniform_tile_spacing_flag
    for &b in cols_incr {
        w.flag(b);
    }
    for &b in rows_incr {
        w.flag(b);
    }
    if tile_cols_log2 + tile_rows_log2 > 0 {
        w.f(0, (tile_rows_log2 + tile_cols_log2) as u8); // context_update_tile_id = 0
        w.f(3, 2); // tile_size_bytes_minus_1 = 3 -> TileSizeBytes = 4
    }
    // quantization_params()
    w.f(0, 8); // base_q_idx = 0  -> lossless -> CodedLossless = 1
    w.flag(false); // DeltaQYDc delta_coded
    if !mono {
        // chroma delta-Q reads are coded only for NumPlanes > 1; monochrome
        // (NumPlanes == 1) omits both, or the decoder misaligns the bitstream.
        w.flag(false); // DeltaQUDc delta_coded
        w.flag(false); // DeltaQUAc delta_coded
    }
    w.flag(false); // using_qmatrix
    // segmentation_params()
    w.flag(false); // segmentation_enabled
    // (base_q_idx==0 => delta_q_present=0, delta_lf absent)
    // (CodedLossless => loop_filter / cdef / lr skipped; TxMode=ONLY_4X4)
    // (FrameIsIntra => reference mode / skip mode / global motion skipped)
    w.flag(false); // reduced_tx_set
    // film_grain_params_present=0 => none
    if trailing {
        w.trailing_bits(); // OBU_FRAME_HEADER (type 3): 1 bit + zero pad
    }
    // byte_alignment() before tile data (OBU_FRAME) is just zero pad via into_bytes
    w.into_bytes()
}

/// Multi-tile lossy frame header. `cols_incr` / `rows_incr` are the
/// `increment_tile_*_log2` bit sequences that walk the decoder from its derived
/// minimum up to the chosen `TileColsLog2` / `TileRowsLog2` (computed by the
/// encoder, which owns the tiling math). `tile_cols_log2` / `tile_rows_log2` are
/// those chosen values; when their sum is > 0 the header additionally codes
/// `context_update_tile_id` and `tile_size_bytes_minus_1` (here `TileSizeBytes =
/// 4`). For a single tile both vectors hold the lone `0` stop-bit and the sum is
/// 0, reproducing the previous single-tile header byte-for-byte.
pub(crate) fn frame_header_lossy_multitile(
    base_q_idx: u8,
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    mono: bool,
) -> Vec<u8> {
    frame_header_lossy_impl(
        base_q_idx,
        cols_incr,
        rows_incr,
        false,
        tile_cols_log2,
        tile_rows_log2,
        false,
        mono,
    )
}

/// Like [`frame_header_lossy_multitile`] but terminates with AV1 `trailing_bits()`
/// (a `1` bit then zero pad) instead of plain zero `byte_alignment`. Use this
/// when the header is carried in a standalone `OBU_FRAME_HEADER` (type 3): the
/// spec requires trailing_bits there, and strict parsers (ffmpeg's cbs_av1)
/// enforce it. (A combined `OBU_FRAME` uses zero byte_alignment instead.)
pub(crate) fn frame_header_lossy_multitile_th(
    base_q_idx: u8,
    cols_incr: &[bool],
    rows_incr: &[bool],
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    mono: bool,
) -> Vec<u8> {
    frame_header_lossy_impl(
        base_q_idx,
        cols_incr,
        rows_incr,
        false,
        tile_cols_log2,
        tile_rows_log2,
        true,
        mono,
    )
}

/// Emit a frame as a separate `OBU_FRAME_HEADER` (type 3) followed by an
/// `OBU_TILE_GROUP` (type 4), rather than a combined `OBU_FRAME` (type 6). This
/// is the layout libaom/rav1e use for tiled frames; ffmpeg's `av1_frame_merge`
/// bitstream filter handles it cleanly, whereas a multi-tile combined
/// `OBU_FRAME` trips its parser in some versions. `frame_header` must already be
/// terminated with `trailing_bits()` (see [`frame_header_lossy_multitile_th`]).
pub(crate) fn wrap_obu_frame_split(frame_header: &[u8], tile_group: &[u8]) -> Vec<u8> {
    let mut out = wrap_obu(ObuType::FrameHeader, frame_header);
    out.extend_from_slice(&wrap_obu(ObuType::TileGroup, tile_group));
    out
}

#[allow(clippy::too_many_arguments)]
/// Deblocking filter levels (luma, chroma) derived from the base quantizer.
/// Gentle at high quality, stronger at low quality. The encoder applies exactly
/// these levels to its reconstruction so the output matches the decoder.
pub(crate) fn loop_filter_levels(base_q_idx: u8) -> (i32, i32) {
    let q = base_q_idx as i32;
    let lvl_y = (q / 8).clamp(0, 40);
    // Chroma deblocking must not switch off ahead of luma. In 4:2:0 the chroma
    // uses 4x4 transforms (the most block boundaries per area), so leaving its
    // loop filter at 0 while luma is still filtered (q/10 hits 0 below base_q_idx
    // 10, where q/8 is still >=1) leaves chroma block-boundary steps undeblocked
    // and folds the q97 chroma curve back below q95. Floor chroma at 1 whenever
    // luma is filtered. (AV1 only signals chroma filter levels when luma is
    // nonzero, so gating on lvl_y also keeps the header consistent.)
    let lvl_uv = if lvl_y > 0 {
        (q / 10).max(1).clamp(0, 32)
    } else {
        0
    };
    (lvl_y, lvl_uv)
}

#[allow(clippy::too_many_arguments)]
fn frame_header_lossy_impl(
    base_q_idx: u8,
    cols_incr: &[bool],
    rows_incr: &[bool],
    disable_cdf_update: bool,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    trailing: bool,
    mono: bool,
) -> Vec<u8> {
    debug_assert!(base_q_idx != 0, "use frame_header_lossless() for q=0");
    let mut w = BitWriter::new();
    w.flag(disable_cdf_update); // disable_cdf_update (0 = adaptive image path, 1 = static isolated APIs)
    w.flag(false); // allow_screen_content_tools
    w.flag(false); // render_and_frame_size_different
    w.flag(true); // uniform_tile_spacing_flag
    // increment_tile_cols_log2 / increment_tile_rows_log2: walk from the
    // decoder's derived minimum up to the chosen TileColsLog2 / TileRowsLog2.
    for &b in cols_incr {
        w.flag(b);
    }
    for &b in rows_incr {
        w.flag(b);
    }
    if tile_cols_log2 + tile_rows_log2 > 0 {
        // NumTiles > 1: context_update_tile_id (TileRowsLog2+TileColsLog2 bits)
        // selects which tile's CDFs persist; irrelevant for a still frame -> 0.
        w.f(0, (tile_rows_log2 + tile_cols_log2) as u8);
        w.f(3, 2); // tile_size_bytes_minus_1 = 3 -> TileSizeBytes = 4
    }
    // quantization_params()
    w.f(base_q_idx as u32, 8); // base_q_idx (non-zero -> lossy)
    w.flag(false); // DeltaQYDc delta_coded
    if !mono {
        let uv_dc = crate::quant::chroma_dc_delta(base_q_idx);
        if uv_dc != 0 {
            w.flag(true); // DeltaQUDc delta_coded
            w.f((uv_dc & 0x7f) as u32, 7); // su(1+6): 7-bit two's complement
        } else {
            w.flag(false); // DeltaQUDc delta_coded
        }
        w.flag(false); // DeltaQUAc delta_coded (== 0)
    }
    w.flag(false); // using_qmatrix
    // segmentation_params()
    w.flag(false); // segmentation_enabled
    // delta_q_params() (base_q_idx != 0 => delta_q_present bit is coded)
    w.flag(false); // delta_q_present = 0  (=> delta_lf absent)
    // CodedLossless = 0 => loop_filter_params():
    let (lvl_y, lvl_uv) = loop_filter_levels(base_q_idx);
    w.f(lvl_y as u32, 6); // loop_filter_level[0] (luma vertical)
    w.f(lvl_y as u32, 6); // loop_filter_level[1] (luma horizontal)
    if lvl_y != 0 && !mono {
        // u/v levels coded only when (level[0] || level[1]) and NumPlanes > 1
        w.f(lvl_uv as u32, 6); // loop_filter_level[2] (U)
        w.f(lvl_uv as u32, 6); // loop_filter_level[3] (V)
    }
    w.f(0, 3); // loop_filter_sharpness = 0
    w.flag(false); // loop_filter_delta_enabled = 0
    // cdef_params(): omitted — CDEF disabled at sequence level (enable_cdef = 0).
    // lr_params(): sequence enable_restoration = 0 => skipped
    // read_tx_mode(): !CodedLossless => tx_mode_select bit
    w.flag(false); // tx_mode_select = 0 => TX_MODE_LARGEST
    // FrameIsIntra => reference/skip-mode/global-motion skipped
    w.flag(false); // reduced_tx_set = 0
    // film_grain_params_present = 0 (seq) => none
    if trailing {
        w.trailing_bits(); // OBU_FRAME_HEADER (type 3): 1 bit + zero pad
    }
    w.into_bytes()
}

/// Wrap a frame header + tile payload as a single OBU_FRAME (type 6).
pub(crate) fn wrap_obu_frame(frame_header: &[u8], tile_data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(frame_header.len() + tile_data.len());
    payload.extend_from_slice(frame_header);
    payload.extend_from_slice(tile_data);
    wrap_obu(ObuType::Frame, &payload)
}

/// Still sequence header with explicit color encoding and selectable `bit_depth`
/// (8, 10, or 12). `profile`: 1 = 4:4:4 (8/10-bit), 2 = 4:2:2 or any 12-bit,
/// 0 = 4:2:0. Writes `color_config()` with the full CICP triplet, range,
/// bit-depth flags, and (for 4:2:0) chroma sample position. 4:4:4 subsampling
/// (ss_x = ss_y = 0) is assumed.
pub(crate) fn sequence_header_cicp(
    width: u32,
    height: u32,
    profile: u32,
    bit_depth: u8,
    color: Option<&crate::color::Cicp>,
) -> Vec<u8> {
    seq_header_ss(width, height, profile, bit_depth, color, 0, 0)
}

/// Like [`sequence_header_cicp`] but with explicit chroma subsampling flags:
/// `(ss_x, ss_y)` = `(0,0)` → 4:4:4, `(1,0)` → 4:2:2, `(1,1)` → 4:2:0.
pub(crate) fn sequence_header_cicp_ss(
    width: u32,
    height: u32,
    profile: u32,
    bit_depth: u8,
    color: Option<&crate::color::Cicp>,
    ss_x: u32,
    ss_y: u32,
) -> Vec<u8> {
    seq_header_ss(width, height, profile, bit_depth, color, ss_x, ss_y)
}

/// Monochrome (`mono_chrome = 1`, `NumPlanes = 1`) sequence header — a single
/// luma plane, used to encode an AVIF alpha auxiliary image (alpha in AV1 is a
/// separate grayscale image, not a 4th channel). `mono_chrome` is only coded
/// when `seq_profile != 1`, so monochrome uses profile 0 (8/10-bit) or 2
/// (12-bit). `color_description_present_flag` is 0 (CP/TC/MC default to
/// unspecified, the norm for an alpha plane); `color_range` is still coded.
pub(crate) fn sequence_header_mono(
    width: u32,
    height: u32,
    bit_depth: u8,
    full_range: bool,
) -> Vec<u8> {
    let profile: u32 = if bit_depth == 12 { 2 } else { 0 };
    let mut w = BitWriter::new();
    w.f(profile, 3); // seq_profile
    w.flag(true); // still_picture
    w.flag(true); // reduced_still_picture_header
    w.f(0, 5); // seq_level_idx[0]

    let wbits = bits_for(width);
    let hbits = bits_for(height);
    w.f((wbits - 1) as u32, 4); // frame_width_bits_minus_1
    w.f((hbits - 1) as u32, 4); // frame_height_bits_minus_1
    w.f(width - 1, wbits); // max_frame_width_minus_1
    w.f(height - 1, hbits); // max_frame_height_minus_1

    w.flag(false); // use_128x128_superblock
    w.flag(false); // enable_filter_intra
    w.flag(false); // enable_intra_edge_filter
    w.flag(false); // enable_superres
    w.flag(false); // enable_cdef (CDEF removed)
    w.flag(false); // enable_restoration

    // color_config() — monochrome branch.
    let high_bitdepth = bit_depth > 8;
    w.flag(high_bitdepth); // high_bitdepth
    if profile == 2 && high_bitdepth {
        w.flag(bit_depth == 12); // twelve_bit
    }
    w.flag(true); // mono_chrome = 1  (coded because profile != 1)
    w.flag(false); // color_description_present_flag = 0 -> CP/TC/MC unspecified
    w.flag(full_range); // color_range
    // mono_chrome => subsampling 1,1; chroma_sample_position CSP_UNKNOWN;
    // separate_uv_delta_q = 0; none coded.

    w.flag(false); // film_grain_params_present

    w.trailing_bits();
    wrap_obu(ObuType::SequenceHeader, &w.into_bytes())
}

/// Core sequence-header writer with explicit chroma subsampling `ss_x`/`ss_y`
/// (0/1 each): (0,0) = 4:4:4, (1,0) = 4:2:2, (1,1) = 4:2:0. Subsampling is coded
/// in `color_config` only for profile 2 at 12-bit (otherwise it is implied by
/// the profile); `chroma_sample_position` is coded whenever the format is 4:2:0.
///
/// `color` is optional: when `None`, `color_description_present_flag` is 0 and
/// primaries/transfer/matrix default to *_UNSPECIFIED. With no description the
/// MC_IDENTITY path is unreachable, so `color_range` and the subsampling/CSP
/// bits are always coded; `color_range` defaults to limited (0) and
/// `chroma_sample_position` to CSP_UNKNOWN (0).
fn seq_header_ss(
    width: u32,
    height: u32,
    profile: u32,
    bit_depth: u8,
    color: Option<&crate::color::Cicp>,
    ss_x: u32,
    ss_y: u32,
) -> Vec<u8> {
    use crate::color::MatrixCoefficients;
    let mut w = BitWriter::new();
    w.f(profile, 3); // seq_profile
    w.flag(true); // still_picture
    w.flag(true); // reduced_still_picture_header

    w.f(0, 5); // seq_level_idx[0]

    let wbits = bits_for(width);
    let hbits = bits_for(height);
    w.f((wbits - 1) as u32, 4); // frame_width_bits_minus_1
    w.f((hbits - 1) as u32, 4); // frame_height_bits_minus_1
    w.f(width - 1, wbits); // max_frame_width_minus_1
    w.f(height - 1, hbits); // max_frame_height_minus_1

    w.flag(false); // use_128x128_superblock
    w.flag(false); // enable_filter_intra
    w.flag(false); // enable_intra_edge_filter
    w.flag(false); // enable_superres
    w.flag(false); // enable_cdef (CDEF removed)
    w.flag(false); // enable_restoration

    // color_config()
    let high_bitdepth = bit_depth > 8;
    w.flag(high_bitdepth); // high_bitdepth
    if profile == 2 && high_bitdepth {
        w.flag(bit_depth == 12); // twelve_bit
    }
    if profile != 1 {
        w.flag(false); // mono_chrome = 0
    }

    // The MC_IDENTITY (RGB) path forces range=1 and 4:4:4 without coding them,
    // but it requires an explicit color description — so it can never apply when
    // color info is absent.
    let is_identity = matches!(color, Some(c) if c.matrix == MatrixCoefficients::Identity);

    if let Some(c) = color {
        w.flag(true); // color_description_present_flag = 1
        w.f(c.primaries as u32, 8);
        w.f(c.transfer as u32, 8);
        w.f(c.matrix as u32, 8);
    } else {
        w.flag(false); // color_description_present_flag = 0
        // primaries/transfer/matrix implicitly *_UNSPECIFIED (not coded)
    }

    if is_identity {
        // MC_IDENTITY ⇒ AV1 forces color_range = 1 and subsampling 0,0 (4:4:4);
        // neither is coded here.
    } else {
        // color_range from the description when present, else limited (0).
        let full_range = color.map(|c| c.full_range).unwrap_or(false);
        w.flag(full_range); // color_range

        // AV1 §5.5.2: subsampling is coded explicitly only for profile 2 at
        // 12-bit; for all other profiles it is implied (0 ⇒ 4:2:0, 1 ⇒ 4:4:4,
        // 2/8-10bit ⇒ 4:2:2). chroma_sample_position is coded for 4:2:0.
        if profile == 2 && bit_depth == 12 {
            w.flag(ss_x == 1); // subsampling_x
            if ss_x == 1 {
                w.flag(ss_y == 1); // subsampling_y (only coded when x==1)
            }
        }
        if ss_x == 1 && ss_y == 1 {
            let csp = color.map(|c| c.chroma_sample_position as u32).unwrap_or(0);
            w.f(csp, 2); // chroma_sample_position (CSP_UNKNOWN=0 when absent)
        }
    }
    w.flag(false); // separate_uv_delta_q

    w.flag(false); // film_grain_params_present

    w.trailing_bits();
    wrap_obu(ObuType::SequenceHeader, &w.into_bytes())
}

/// AV1 metadata OBU type codes (Section 6.7.1).
pub(crate) mod metadata_type {
    pub(crate) const HDR_CLL: u64 = 1;
    pub(crate) const HDR_MDCV: u64 = 2;
    pub(crate) const ITUT_T35: u64 = 4;
}

/// Wrap a metadata payload as an `OBU_METADATA`. Layout: `leb128(metadata_type)`
/// then the payload bytes. Each metadata payload must already include the
/// `trailing_one_bit` + zero padding (a `0x80` byte when byte-aligned): dav1d's
/// `check_trailing_bits` and the ITU-T T.35 length scan both require it.
fn metadata_obu(metadata_type: u64, payload: &[u8]) -> Vec<u8> {
    let mut body = leb128(metadata_type);
    body.extend_from_slice(payload);
    wrap_obu(ObuType::Metadata, &body)
}

/// `HDR_CLL` metadata OBU (content light level).
pub(crate) fn metadata_hdr_cll(cll: &ContentLightLevel) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.f(cll.max_content_light_level as u32, 16);
    w.f(cll.max_pic_average_light_level as u32, 16);
    w.trailing_bits();
    metadata_obu(metadata_type::HDR_CLL, &w.into_bytes())
}

/// `HDR_MDCV` metadata OBU (mastering display color volume, ST 2086).
pub(crate) fn metadata_hdr_mdcv(m: &MasteringDisplay) -> Vec<u8> {
    let mut w = BitWriter::new();
    for (x, y) in m.primaries.iter() {
        w.f(*x as u32, 16);
        w.f(*y as u32, 16);
    }
    w.f(m.white_point.0 as u32, 16);
    w.f(m.white_point.1 as u32, 16);
    w.f(m.max_luminance, 32);
    w.f(m.min_luminance, 32);
    w.trailing_bits();
    metadata_obu(metadata_type::HDR_MDCV, &w.into_bytes())
}

/// `ITUT_T35` metadata OBU (raw user data: HDR10+, Dolby Vision RPU, etc.).
pub(crate) fn metadata_itut_t35(t35: &crate::color::ItutT35) -> Vec<u8> {
    let mut body = Vec::new();
    body.push(t35.country_code);
    if t35.country_code == 0xFF {
        body.push(t35.country_code_extension.unwrap_or(0));
    }
    body.extend_from_slice(&t35.payload);
    body.push(0x80); // trailing_one_bit + zero pad (byte-aligned payload)
    metadata_obu(metadata_type::ITUT_T35, &body)
}
#[allow(unused)]
pub(crate) fn metadata_obus(
    cll: Option<&ContentLightLevel>,
    mdcv: Option<&MasteringDisplay>,
    t35: &[crate::color::ItutT35],
) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(c) = cll {
        out.extend_from_slice(&metadata_hdr_cll(c));
    }
    if let Some(m) = mdcv {
        out.extend_from_slice(&metadata_hdr_mdcv(m));
    }
    for t in t35 {
        out.extend_from_slice(&metadata_itut_t35(t));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::*;

    /// Metadata OBUs must be `OBU_METADATA` (type 5), carry the right
    /// `metadata_type` leb128, and end with the `0x80` trailing byte dav1d's
    /// `check_trailing_bits` / T.35 length scan require.
    #[test]
    fn metadata_obus_well_formed() {
        let obu_type = |b: &[u8]| (b[0] >> 3) & 0xF;
        let cll = metadata_hdr_cll(&ContentLightLevel {
            max_content_light_level: 1000,
            max_pic_average_light_level: 400,
        });
        assert_eq!(obu_type(&cll), 5);
        assert_eq!(cll[2], metadata_type::HDR_CLL as u8); // leb128(1) == 0x01
        assert_eq!(*cll.last().unwrap(), 0x80); // trailing_one_bit
        // body = leb(type)=1 + 4 bytes (16+16) + 0x80 = 6; +1 header +1 size = 8
        assert_eq!(cll.len(), 8);

        let mdcv = metadata_hdr_mdcv(&MasteringDisplay::from_floats(
            [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
            (0.3127, 0.3290),
            1000.0,
            0.005,
        ));
        assert_eq!(obu_type(&mdcv), 5);
        assert_eq!(mdcv[2], metadata_type::HDR_MDCV as u8);
        assert_eq!(*mdcv.last().unwrap(), 0x80);
        // 3*4 + 4 + 8 = 24 bytes fields + 0x80 + leb(type) = 26 body; +2 framing = 28
        assert_eq!(mdcv.len(), 28);

        let t35 = metadata_itut_t35(&ItutT35 {
            country_code: 0xFF,
            country_code_extension: Some(0x49),
            payload: b"HDR10+".to_vec(),
        });
        assert_eq!(obu_type(&t35), 5);
        assert_eq!(t35[2], metadata_type::ITUT_T35 as u8); // leb128(4)
        assert_eq!(*t35.last().unwrap(), 0x80);
    }

    #[test]
    fn obu_wrapping_roundtrips_size() {
        let payload = vec![0xAA; 200];
        let obu = wrap_obu(ObuType::Frame, &payload);
        // first byte = header, then leb128(200) = [0xc8,0x01], then payload
        assert_eq!(obu[0] >> 3 & 0xf, ObuType::Frame as u8); // type field
        assert_eq!(&obu[1..3], &leb128(200)[..]);
        assert_eq!(obu.len(), 1 + 2 + 200);
    }

    #[test]
    fn temporal_delimiter_is_two_bytes() {
        // header byte + leb128(0)
        assert_eq!(temporal_delimiter().len(), 2);
    }
}
