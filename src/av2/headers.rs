//! OBU framing and the uncompressed sequence/frame headers.

use crate::av2::entropy::ByteWriter;
use crate::av2::layout::Layout;

/// Optional guided-deblock-filter parameters (currently unused by the encoder).
#[derive(Clone, Copy)]
pub(crate) struct GuidedDeblock {
    pub(crate) qp_offset: u32,
    pub(crate) scale_minus_one: u32,
}

/// Top-level encoder configuration shared by the header builders.
pub(crate) struct Config {
    pub(crate) layout: Layout,
    pub(crate) base_q: u32,
    pub(crate) deblock: bool,
    pub(crate) delta_q: i32,
    pub(crate) tx_switchable: bool,
    pub(crate) guided_deblock: Option<GuidedDeblock>,
    /// Coded bit depth: 8, 10 or 12.
    pub(crate) bit_depth: u8,
    /// When set, emit a `coded_lossless` frame: base_q forced to 0, in-loop filters and
    /// the tx-mode bit omitted (avm forces `ONLY_4X4`). See [`frame_header`].
    pub(crate) lossless: bool,
}

/// LEB128-encode an unsigned value.
pub(crate) fn leb128(mut value: u64) -> Vec<u8> {
    let mut out = vec![];
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
    out
}

/// Wrap a payload in an OBU with a size field.
pub(crate) fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let header = obu_type << 2;
    let mut out = leb128(1 + payload.len() as u64);
    out.push(header);
    out.extend(payload);
    out
}

/// Build the sequence header OBU payload.
pub(crate) fn sequence_header(config: &Config, width: u32, height: u32) -> Vec<u8> {
    let has_chroma = config.layout.has_chroma();
    let mut b = ByteWriter::new();
    b.write_uvlc(0);
    b.write_bits(config.layout.profile(), 5); // seq_profile_idc (0=420/mono, 3=422, 4=444)
    b.write_bit(1); // single_picture_header_flag
    b.write_bits(31, 5); // seq_max_level_idx = SEQ_LEVEL_MAX (unconstrained; skips DPB/ref-frame conformance)
    b.write_uvlc(config.layout.header_uvlc());
    // color_config() bit depth: avm reads a uvlc index into {10, 8, 12}
    // (av2_get_bitdepth_from_index), so 8-bit→1, 10-bit→0, 12-bit→2.
    let bitdepth_idx = match config.bit_depth {
        10 => 0,
        12 => 2,
        _ => 1, // 8-bit
    };
    b.write_uvlc(bitdepth_idx);

    // frame_width_bits_minus_1 stores (dim-1) in (n+1) bits; dim==1 needs 1 bit
    // for the value 0, so clamp the bit count to at least 1.
    let width_bits = (32 - (width - 1).leading_zeros()).max(1);
    let height_bits = (32 - (height - 1).leading_zeros()).max(1);
    b.write_bits(width_bits - 1, 4);
    b.write_bits(height_bits - 1, 4);
    b.write_bits(width - 1, width_bits);
    b.write_bits(height - 1, height_bits);

    b.write_bit(0); // crop
    b.write_bit(0);
    b.write_bit(0); // superblock size
    if has_chroma {
        b.write_bit(0); // semi-decoupled partition
    }
    b.write_bit(0);
    b.write_bit(0); // partition
    b.write_bit(0);
    b.write_bit(0); // segmentation
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0); // dip, edge filter, mrl, cfl
    if has_chroma {
        b.write_bits(0, 2); // cfl downsample filter index
    }
    b.write_bit(0);
    b.write_bit(0); // mhccp, ibp
    b.write_bit(0);
    b.write_bit(1); // reference config
    b.write_uniform(0, 3);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0); // tx group: fsc, idtx-intra, ist0, ist1
    if has_chroma {
        b.write_bit(0); // chroma dct-only
    }
    b.write_bit(0); // reduced tx partition set
    if has_chroma {
        b.write_bit(0); // cctx
    }
    b.write_bit(0);
    b.write_bit(0); // coef: tcq, parity hiding
    if has_chroma {
        b.write_bit(0); // separate uv delta-q
    }
    b.write_bit(1); // equal ac/dc quant
    if has_chroma {
        b.write_bits(23, 5); // base uv-ac delta-q (raw 23 => delta 0)
        b.write_bit(0); // uv-ac delta-q enabled
    }
    b.write_bit(0);
    b.write_bit(0); // disable lf across tiles, cdef
    if config.guided_deblock.is_some() {
        b.write_bit(1);
        b.write_bit(0);
    } else {
        b.write_bit(0);
    }
    b.write_bit(0);
    b.write_bit(0); // restoration, ccso
    b.write_bits(0, 2);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0);
    b.align_with_one();
    b.into_bytes()
}

/// Build the key-frame header OBU payload (the tile data is appended by the caller).
pub(crate) fn frame_header(config: &Config, width: u32, height: u32) -> Vec<u8> {
    let has_chroma = config.layout.has_chroma();
    let mut b = ByteWriter::new();
    b.write_bit(1);
    b.write_uvlc(0);
    b.write_uvlc(0);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(1);

    let sb_cols = width.div_ceil(64);
    let sb_rows = height.div_ceil(64);
    b.write_bit(1);
    if sb_cols > 1 {
        b.write_bit(0);
    }
    if sb_rows > 1 {
        b.write_bit(0);
    }
    b.write_bits(if config.lossless { 0 } else { config.base_q }, 8);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0); // segmentation, qm, delta-q (off)

    if config.lossless {
        // coded_lossless: setup_loopfilter returns early, guided-deblock/CDEF/CCSO are
        // skipped, and read_tx_mode is forced to ONLY_4X4 with no bit. Only the
        // 2-bit reduced_tx_set remains before byte alignment.
        b.write_bits(0, 2); // reduced_tx_set
        b.align_with_zero();
        return b.into_bytes();
    }

    if config.deblock {
        b.write_bit(1);
        b.write_bit(1); // luma deblock levels (V, H)
        if has_chroma {
            b.write_bit(0);
            b.write_bit(0); // chroma deblock levels (U, V)
        }
        if config.delta_q != 0 {
            b.write_bit(1);
            b.write_bits(((config.delta_q + 2) as u32) & 3, 2);
            b.write_bit(0); // reuse for the second luma level
        } else {
            b.write_bit(0);
            b.write_bit(0);
        }
    } else {
        b.write_bit(0);
        b.write_bit(0);
    }

    if let Some(gdf) = config.guided_deblock {
        b.write_bit(0);
        b.write_bits(gdf.qp_offset, 2);
        b.write_bits(gdf.scale_minus_one, 2);
    }
    b.write_bit(if config.tx_switchable { 1 } else { 0 }); // txfm_mode: 1=SWITCHABLE
    b.write_bits(0, 2); // reduced_txtp_set
    b.align_with_zero();
    b.into_bytes()
}
