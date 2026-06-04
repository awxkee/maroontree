//! OBU framing and *sketched* sequence/frame headers.
//!
//! ## Status: STRUCTURAL, NOT SPEC-COMPLETE
//! OBU framing (header byte + LEB128 size) and the field *order* below follow
//! AV1, but the sequence/frame headers are intentionally a reduced sketch:
//! many conditional fields (operating points, decoder model, timing,
//! `frame_size_override`, segmentation, etc.) are stubbed to their
//! "absent/default" path. This is enough to show the framing shape and to be
//! the place real header coding gets finished — it is NOT yet a header a
//! conformant decoder will accept. The gaps are called out inline.

use crate::bitwriter::{BitWriter, leb128};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ObuType {
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
}

/// Wrap a payload as an OBU with `obu_has_size_field = 1`.
pub fn wrap_obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
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
pub fn temporal_delimiter() -> Vec<u8> {
    wrap_obu(ObuType::TemporalDelimiter, &[])
}

/// Reduced sequence header for: still image, 8/10/12-bit, 4:4:4, no scaling.
/// NOTE: stubs out operating points / decoder model — see status banner.
pub fn sequence_header_sketch(width: u32, height: u32, bit_depth: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.f(0, 3); // seq_profile (0 = 8/10-bit 4:2:0/4:4:4 ... 12-bit really needs profile 2; stub)
    w.flag(true); // still_picture
    w.flag(true); // reduced_still_picture_header  -> collapses many fields
    // In reduced_still_picture_header mode, AV1 fixes seq_level_idx etc.
    w.f(0, 5); // seq_level_idx[0]  (stub)

    // frame_width_bits_minus_1 / frame_height_bits_minus_1 then sizes
    let wbits = bits_for(width) as u32;
    let hbits = bits_for(height) as u32;
    w.f(wbits - 1, 4);
    w.f(hbits - 1, 4);
    w.f(width - 1, wbits as u8);
    w.f(height - 1, hbits as u8);

    // ... many feature-enable flags here would be coded; we take the all-off path
    // (no superres, no filters). STUB: not all required bits emitted.

    // color_config (partial): identity matrix for RGB
    let high_bitdepth = bit_depth > 8;
    w.flag(high_bitdepth); // high_bitdepth
    if bit_depth == 12 {
        w.flag(true); // twelve_bit
    }
    w.flag(false); // mono_chrome = 0 (we have 3 planes)
    w.flag(false); // color_description_present_flag (stub -> defaults)
    // color_range, subsampling, matrix would follow; for identity RGB:
    //   matrix_coefficients = 0 (identity), subsampling_x = subsampling_y = 0 (4:4:4)
    // STUB: these conditional bits are not all emitted yet.

    wrap_obu(ObuType::SequenceHeader, &w.into_bytes())
}

fn bits_for(v: u32) -> u8 {
    let mut b = 1;
    while (1u32 << b) <= v.saturating_sub(1) {
        b += 1;
    }
    b
}

/// Length in bytes of `frame_header_lossless()` output (used by the decoder to
/// skip past the header inside an OBU_FRAME).
pub const FRAME_HEADER_LEN: usize = 3;

/// Uncompressed frame header for a reduced-still-picture, KEY_FRAME, lossless,
/// 4:4:4, single-tile image. Reverse-engineered from aom's reference FRAME OBU
/// (`10 00 00 ...`) and traced field-by-field: with `base_q_idx = 0` the frame
/// is CodedLossless, which removes loop-filter / CDEF / restoration / tx-mode
/// signaling, so only 18 bits remain. Byte-aligns to `10 00 00`.
///
/// VALID FOR SINGLE-SUPERBLOCK IMAGES (<= 64x64). Larger frames add
/// tile-increment bits in `tile_info`, not emitted here yet.
pub fn frame_header_lossless() -> Vec<u8> {
    frame_header_lossless_tiled(1, 1)
}

/// Like [`frame_header_lossless`] but emits the tile-count increment bits needed
/// when the frame spans more than one 64x64 superblock (single tile is always
/// signalled; see [`frame_header_lossy_tiled`] for the tiling rationale).
pub fn frame_header_lossless_tiled(sb_cols: u32, sb_rows: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    // (reduced_still_picture_header => frame_type=KEY_FRAME, show_frame=1,
    //  error_resilient_mode=1, frame_size_override=0, etc., none coded)
    w.flag(true); // disable_cdf_update = 1 (encoder is non-adaptive; decoder must not adapt)
    w.flag(false); // allow_screen_content_tools (seq force = SELECT)
    // (allow_screen_content_tools=0 => no force_integer_mv; FrameIsIntra)
    w.flag(false); // render_and_frame_size_different
    // (allow_intrabc not coded since screen content tools off)
    w.flag(true); // uniform_tile_spacing_flag
    if sb_cols > 1 {
        w.flag(false); // increment_tile_cols_log2 = 0
    }
    if sb_rows > 1 {
        w.flag(false); // increment_tile_rows_log2 = 0
    }
    // quantization_params()
    w.f(0, 8); // base_q_idx = 0  -> lossless -> CodedLossless = 1
    w.flag(false); // DeltaQYDc delta_coded
    w.flag(false); // DeltaQUDc delta_coded
    w.flag(false); // DeltaQUAc delta_coded
    w.flag(false); // using_qmatrix
    // segmentation_params()
    w.flag(false); // segmentation_enabled
    // (base_q_idx==0 => delta_q_present=0, delta_lf absent)
    // (CodedLossless => loop_filter / cdef / lr skipped; TxMode=ONLY_4X4)
    // (FrameIsIntra => reference mode / skip mode / global motion skipped)
    w.flag(false); // reduced_tx_set
    // film_grain_params_present=0 => none
    // byte_alignment() before tile data:
    w.into_bytes()
}

/// Uncompressed frame header for a reduced-still-picture KEY_FRAME that is
/// **lossy** (`base_q_idx != 0`). With a non-zero quantizer `CodedLossless`
/// becomes false, which re-enables the loop-filter params, the
/// `delta_q_present` bit and `tx_mode_select`. Loop-filter levels are set to 0
/// (deblocking disabled) and `tx_mode = TX_LARGEST` so the TX size is inferred
/// from the block (no per-block tx_size symbol). CDEF/restoration are skipped
/// because the sequence header disables them. Field order follows the ref
/// stream parsed from aomenc (`base_q=128, lf=0, qm=0, txfm_mode=LARGEST`).
pub fn frame_header_lossy(base_q_idx: u8) -> Vec<u8> {
    // Isolated single-block demo APIs encode with static CDFs.
    frame_header_lossy_impl(base_q_idx, 1, 1, true)
}

/// Like [`frame_header_lossy`] but emits the tile-count increment bits required
/// when the frame spans more than one 64x64 superblock. A single tile is always
/// signalled (`TileColsLog2 = TileRowsLog2 = 0`); for `sb_cols > 1` /
/// `sb_rows > 1` an `increment_tile_*_log2 = 0` bit is emitted to stop the
/// count at zero (minLog2TileCols is 0 for frames up to 4096px wide). The
/// full-image path codes with **adaptive** CDFs (`disable_cdf_update = 0`).
pub fn frame_header_lossy_tiled(base_q_idx: u8, sb_cols: u32, sb_rows: u32) -> Vec<u8> {
    frame_header_lossy_impl(base_q_idx, sb_cols, sb_rows, false)
}

fn frame_header_lossy_impl(
    base_q_idx: u8,
    sb_cols: u32,
    sb_rows: u32,
    disable_cdf_update: bool,
) -> Vec<u8> {
    debug_assert!(base_q_idx != 0, "use frame_header_lossless() for q=0");
    let mut w = BitWriter::new();
    w.flag(disable_cdf_update); // disable_cdf_update (0 = adaptive image path, 1 = static isolated APIs)
    w.flag(false); // allow_screen_content_tools
    w.flag(false); // render_and_frame_size_different
    w.flag(true); // uniform_tile_spacing_flag (single tile)
    if sb_cols > 1 {
        w.flag(false); // increment_tile_cols_log2 = 0 -> TileColsLog2 = 0
    }
    if sb_rows > 1 {
        w.flag(false); // increment_tile_rows_log2 = 0 -> TileRowsLog2 = 0
    }
    // quantization_params()
    w.f(base_q_idx as u32, 8); // base_q_idx (non-zero -> lossy)
    w.flag(false); // DeltaQYDc delta_coded
    w.flag(false); // DeltaQUDc delta_coded
    w.flag(false); // DeltaQUAc delta_coded
    w.flag(false); // using_qmatrix
    // segmentation_params()
    w.flag(false); // segmentation_enabled
    // delta_q_params() (base_q_idx != 0 => delta_q_present bit is coded)
    w.flag(false); // delta_q_present = 0  (=> delta_lf absent)
    // CodedLossless = 0 => loop_filter_params():
    w.f(0, 6); // loop_filter_level[0] = 0
    w.f(0, 6); // loop_filter_level[1] = 0
    // (both levels 0 => u/v levels not coded)
    w.f(0, 3); // loop_filter_sharpness = 0
    w.flag(false); // loop_filter_delta_enabled = 0
    // cdef_params(): sequence enable_cdef = 0 => skipped
    // lr_params(): sequence enable_restoration = 0 => skipped
    // read_tx_mode(): !CodedLossless => tx_mode_select bit
    w.flag(false); // tx_mode_select = 0 => TX_MODE_LARGEST
    // FrameIsIntra => reference/skip-mode/global-motion skipped
    w.flag(false); // reduced_tx_set = 0
    // film_grain_params_present = 0 (seq) => none
    w.into_bytes()
}

/// Wrap a frame header + tile payload as a single OBU_FRAME (type 6).
pub fn wrap_obu_frame(frame_header: &[u8], tile_data: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(frame_header.len() + tile_data.len());
    payload.extend_from_slice(frame_header);
    payload.extend_from_slice(tile_data);
    wrap_obu(ObuType::Frame, &payload)
}

/// Reduced-still-picture sequence header for profile 1 / 8-bit / 4:4:4 identity
/// RGB. Layout reverse-engineered from avifenc/aom reference headers and
/// verified bit-for-bit (see the reference payloads below). Field order follows
/// AV1 §5.5.1 `sequence_header_obu` + §5.5.2 `color_config`.
///
/// Reference payloads used for validation (aom 3.8.2, profile 1 / 8-bit / 4:4:4):
///   32x32   : 38 11 3f f6 10 10 d0 02
///   64x64   : 38 15 7f fd 84 04 34 00 80
///   128x96  : 38 19 bf df 61 01 0d 00 20
pub fn sequence_header_444_8bit(width: u32, height: u32) -> Vec<u8> {
    sequence_header_444_8bit_mc(width, height, 0)
}

/// 4:4:4 8-bit sequence header. `matrix_coefficients`: 0 = MC_IDENTITY (GBR,
/// used by the lossless path), 6 = MC_BT_601 (full-range YCbCr, used by the
/// lossy path so the decoder decorrelates the planes back to RGB on output).
pub fn sequence_header_444_8bit_mc(width: u32, height: u32, matrix_coefficients: u32) -> Vec<u8> {
    sequence_header_8bit(width, height, 1, matrix_coefficients)
}

/// Bit-depth-aware still sequence header built from a `matrix_coefficients` code
/// (0 = MC_IDENTITY / GBR, 6 = MC_BT_601 full-range YCbCr) with BT.709 primaries
/// + sRGB transfer + full range, for `profile` and `bit_depth` (8/10/12). Used by
/// the lossy paths; delegates to [`sequence_header_cicp`].
pub fn sequence_header_mc(
    width: u32,
    height: u32,
    profile: u32,
    bit_depth: u8,
    matrix_coefficients: u32,
    ss_x: u32,
    ss_y: u32,
) -> Vec<u8> {
    let cicp = crate::color::Cicp {
        color_primaries: crate::color::primaries::BT709,
        transfer_characteristics: crate::color::transfer::SRGB,
        matrix_coefficients: matrix_coefficients as u8,
        full_range: true,
        chroma_sample_position: crate::color::ChromaSamplePosition::Unknown,
    };
    seq_header_ss(width, height, profile, bit_depth, &cicp, ss_x, ss_y)
}

/// 8-bit still sequence header. `profile`: 1 = 4:4:4 (no subsampling), 2 = 4:2:2
/// (profile 2 at 8-bit forces `subsampling_x=1, subsampling_y=0`, i.e. I422, and
/// codes a `mono_chrome` bit that profile 1 omits). `matrix_coefficients`:
/// 0 = MC_IDENTITY (GBR), 6 = MC_BT_601 (full-range YCbCr).
pub fn sequence_header_8bit(
    width: u32,
    height: u32,
    profile: u32,
    matrix_coefficients: u32,
) -> Vec<u8> {
    // Preserve the historical byte layout: BT.709 primaries, sRGB transfer,
    // full range, the requested matrix.
    let cicp = crate::color::Cicp {
        color_primaries: crate::color::primaries::BT709,
        transfer_characteristics: crate::color::transfer::SRGB,
        matrix_coefficients: matrix_coefficients as u8,
        full_range: true,
        chroma_sample_position: crate::color::ChromaSamplePosition::Unknown,
    };
    sequence_header_8bit_cicp(width, height, profile, &cicp)
}

/// 8-bit still sequence header with explicit CICP signalling (thin wrapper over
/// [`sequence_header_cicp`] with `bit_depth = 8`).
pub fn sequence_header_8bit_cicp(
    width: u32,
    height: u32,
    profile: u32,
    cicp: &crate::color::Cicp,
) -> Vec<u8> {
    sequence_header_cicp(width, height, profile, 8, cicp)
}

/// Still sequence header with explicit CICP signalling and selectable
/// `bit_depth` (8, 10 or 12). `profile`: 1 = 4:4:4 (8/10-bit), 2 = 4:2:2 or any
/// 12-bit, 0 = 4:2:0. Writes `color_config()` with the `high_bitdepth` /
/// `twelve_bit` bit-depth signalling, the full primaries / transfer / matrix
/// triplet, `color_range`, profile-2 subsampling, and (for 4:2:0)
/// `chroma_sample_position`. AV1 requires `matrix_coefficients == 0` (identity)
/// to be 4:4:4 + full range (neither subsampling nor range is then coded); the
/// caller must honour that. 12-bit must use profile 2; 4:4:4 12-bit is signalled
/// via `subsampling_x = 0` when the matrix is non-identity.
pub fn sequence_header_cicp(
    width: u32,
    height: u32,
    profile: u32,
    bit_depth: u8,
    cicp: &crate::color::Cicp,
) -> Vec<u8> {
    // 4:4:4 (no subsampling) is the default for the CICP/lossless callers.
    seq_header_ss(width, height, profile, bit_depth, cicp, 0, 0)
}

/// Core sequence-header writer with explicit chroma subsampling `ss_x`/`ss_y`
/// (0/1 each): (0,0) = 4:4:4, (1,0) = 4:2:2, (1,1) = 4:2:0. Subsampling is coded
/// in `color_config` only for profile 2 at 12-bit (otherwise it is implied by
/// the profile); `chroma_sample_position` is coded whenever the format is 4:2:0.
fn seq_header_ss(
    width: u32,
    height: u32,
    profile: u32,
    bit_depth: u8,
    cicp: &crate::color::Cicp,
    ss_x: u32,
    ss_y: u32,
) -> Vec<u8> {
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
    w.flag(false); // enable_cdef
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
    w.flag(true); // color_description_present_flag = 1
    w.f(cicp.color_primaries as u32, 8);
    w.f(cicp.transfer_characteristics as u32, 8);
    w.f(cicp.matrix_coefficients as u32, 8);
    if cicp.matrix_coefficients == 0 {
        // MC_IDENTITY ⇒ AV1 forces color_range = 1 and subsampling 0,0 (4:4:4);
        // neither is coded here.
    } else {
        w.flag(cicp.full_range); // color_range
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
            w.f(cicp.chroma_sample_position as u32, 2);
        }
    }
    w.flag(false); // separate_uv_delta_q

    w.flag(false); // film_grain_params_present

    w.trailing_bits();
    wrap_obu(ObuType::SequenceHeader, &w.into_bytes())
}

/// AV1 metadata OBU type codes (Section 6.7.1).
pub mod metadata_type {
    pub const HDR_CLL: u64 = 1;
    pub const HDR_MDCV: u64 = 2;
    pub const ITUT_T35: u64 = 4;
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
pub fn metadata_hdr_cll(cll: &crate::color::ContentLightLevel) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.f(cll.max_cll as u32, 16);
    w.f(cll.max_fall as u32, 16);
    w.trailing_bits();
    metadata_obu(metadata_type::HDR_CLL, &w.into_bytes())
}

/// `HDR_MDCV` metadata OBU (mastering display colour volume, ST 2086).
pub fn metadata_hdr_mdcv(m: &crate::color::MasteringDisplay) -> Vec<u8> {
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
pub fn metadata_itut_t35(t35: &crate::color::ItutT35) -> Vec<u8> {
    let mut body = Vec::new();
    body.push(t35.country_code);
    if t35.country_code == 0xFF {
        body.push(t35.country_code_extension.unwrap_or(0));
    }
    body.extend_from_slice(&t35.payload);
    body.push(0x80); // trailing_one_bit + zero pad (byte-aligned payload)
    metadata_obu(metadata_type::ITUT_T35, &body)
}

/// Emit all in-bitstream metadata OBUs for an image, in the order they should
/// appear after the sequence header and before the frame. The ICC profile is
/// container-level and is intentionally not emitted here.
pub fn metadata_obus(meta: &crate::color::ImageMetadata) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(cll) = &meta.cll {
        out.extend_from_slice(&metadata_hdr_cll(cll));
    }
    if let Some(mdcv) = &meta.mdcv {
        out.extend_from_slice(&metadata_hdr_mdcv(mdcv));
    }
    for t35 in &meta.t35 {
        out.extend_from_slice(&metadata_itut_t35(t35));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::*;

    // MSB-first bit reader for parsing the sequence-header color_config back.
    struct Br<'a> {
        d: &'a [u8],
        p: usize,
    }
    impl<'a> Br<'a> {
        fn f(&mut self, n: usize) -> u32 {
            let mut v = 0u32;
            for _ in 0..n {
                let bit = (self.d[self.p / 8] >> (7 - (self.p % 8))) & 1;
                v = (v << 1) | bit as u32;
                self.p += 1;
            }
            v
        }
    }

    fn parse_cicp(cicp: &Cicp, profile: u32, w: u32, h: u32) -> (u8, u8, u8, Option<bool>) {
        let obu = sequence_header_8bit_cicp(w, h, profile, cicp);
        let mut i = 1; // OBU header byte
        while obu[i] & 0x80 != 0 {
            i += 1;
        }
        i += 1; // end of leb128 size
        let mut br = Br { d: &obu[i..], p: 0 };
        br.f(3); // seq_profile
        br.f(1); // still_picture
        br.f(1); // reduced_still_picture_header
        br.f(5); // seq_level_idx
        let (wb, hb) = (bits_for(w) as usize, bits_for(h) as usize);
        br.f(4);
        br.f(4);
        br.f(wb);
        br.f(hb);
        for _ in 0..6 {
            br.f(1);
        }
        br.f(1); // high_bitdepth
        if profile != 1 {
            br.f(1); // mono_chrome
        }
        assert_eq!(br.f(1), 1, "color_description_present_flag");
        let cp = br.f(8) as u8;
        let tc = br.f(8) as u8;
        let mc = br.f(8) as u8;
        let range = if mc != 0 { Some(br.f(1) == 1) } else { None };
        (cp, tc, mc, range)
    }

    #[test]
    fn cicp_roundtrips_through_seq_header() {
        for cicp in [
            Cicp::identity_rgb(),
            Cicp::srgb_ycbcr(),
            Cicp::bt709(),
            Cicp::bt2020_pq(),
            Cicp::bt2020_hlg(),
        ] {
            let (cp, tc, mc, range) = parse_cicp(&cicp, 1, 1620, 1080);
            assert_eq!(cp, cicp.color_primaries);
            assert_eq!(tc, cicp.transfer_characteristics);
            assert_eq!(mc, cicp.matrix_coefficients);
            match range {
                None => assert_eq!(
                    cicp.matrix_coefficients, 0,
                    "range omitted only for identity"
                ),
                Some(r) => assert_eq!(r, cicp.full_range),
            }
        }
    }

    /// Metadata OBUs must be `OBU_METADATA` (type 5), carry the right
    /// `metadata_type` leb128, and end with the `0x80` trailing byte dav1d's
    /// `check_trailing_bits` / T.35 length scan require.
    #[test]
    fn metadata_obus_well_formed() {
        let obu_type = |b: &[u8]| (b[0] >> 3) & 0xF;
        let cll = metadata_hdr_cll(&ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
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
    fn frame_header_lossy_bytes_dav1d_verified() {
        // dav1d 1.4.1 parses this header: base_q=16, lf=0, qm=0,
        // txfm_mode=TX_LARGEST, reduced_tx_set=0, CodedLossless=0 (lossy).
        assert_eq!(frame_header_lossy(16), vec![0x91, 0x00, 0x00, 0x00, 0x00]);
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

    #[test]
    fn seq_header_64x64_matches_validated_bytes() {
        // This exact payload was accepted by dav1d 1.4.1 (filters disabled, so
        // it differs from aom's only in the enable_filter_intra/edge bits).
        // OBU = header(0x0a) + leb128 size(0x09) + 9-byte payload.
        let obu = sequence_header_444_8bit(64, 64);
        let expected = [
            0x0a, 0x09, 0x38, 0x15, 0x7f, 0xfc, 0x04, 0x04, 0x34, 0x00, 0x80,
        ];
        assert_eq!(
            obu, expected,
            "sequence header bytes drifted from validated output"
        );
    }
}
