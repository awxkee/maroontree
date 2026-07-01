//! OBU framing and the uncompressed sequence/frame headers.

use crate::av2::entropy::ByteWriter;
use crate::av2::layout::Layout;

/// Optional guided-deblock-filter parameters (currently unused by the encoder).
#[derive(Clone, Copy)]
pub(crate) struct GuidedDeblock {
    pub(crate) qp_offset: u32,
    pub(crate) scale_minus_one: u32,
}

/// CCSO (cross-component sample offset) frame parameters. Phase 2 supports U and V
/// planes, in either band-offset-only or edge-classified mode.
#[derive(Clone)]
pub(crate) struct CcsoPlane {
    /// false = edge-classified mode, true = band-offset-only.
    pub(crate) bo_only: bool,
    pub(crate) scale_idx: u8,
    pub(crate) quant_idx: u8,
    pub(crate) ext_filter_support: u8,
    pub(crate) edge_clf: u8,
    pub(crate) max_band_log2: u8,
    /// Raw `ccso_offset` value per LUT entry. For bo_only, indexed by band
    /// (`band << 4`); for edge mode, indexed `(band << 4) + (c0 << 2) + c1`.
    pub(crate) offsets: Vec<i32>,
}

#[derive(Clone)]
pub(crate) struct CcsoConfig {
    /// Per-plane enable: index 0 = Y, 1 = U, 2 = V.
    pub(crate) enable: [bool; 3],
    /// Per-plane params (only meaningful where `enable[plane]`).
    pub(crate) planes: [Option<CcsoPlane>; 3],
}

/// Top-level encoder configuration shared by the header builders.
#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) layout: Layout,
    pub(crate) base_q: u32,
    pub(crate) deblock: bool,
    pub(crate) db_apply: (bool, bool, bool, bool),
    pub(crate) db_delta: (i32, i32, i32, i32),
    pub(crate) tx_switchable: bool,
    pub(crate) guided_deblock: Option<GuidedDeblock>,
    /// In-loop CDEF.
    pub(crate) cdef: Option<(u8, u8, u8)>,
    /// CCSO (cross-component sample offset). `Some((plane_enables, blk_size_is_sb,
    /// offsets))` when enabled. Phase 1: U plane only, band-offset-only mode.
    pub(crate) ccso: Option<CcsoConfig>,
    /// Coded bit depth: 8, 10 or 12.
    pub(crate) bit_depth: u8,
    pub(crate) lossless: bool,
    pub(crate) cfl: bool,
    pub(crate) updating_cdf: bool,
    pub(crate) aq: bool,
    pub(crate) aq_res_log2: u8,
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
    b.write_bit(config.cfl as u32); // dip, edge filter, mrl, cfl_intra (this bit = enable_cfl_intra)
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
    b.write_bit(0); // disable_loopfilters_across_tiles
    b.write_bit(config.cdef.is_some() as u32); // enable_cdef
    if config.guided_deblock.is_some() {
        b.write_bit(1);
        b.write_bit(0);
    } else {
        b.write_bit(0);
    }
    b.write_bit(0); // enable_restoration
    if config.ccso.is_some() {
        b.write_bit(1); // enable_ccso
        b.write_bit(1); // ccso_unit_matches_sb_size = 1 (CCSO unit == 64px SB)
    } else {
        b.write_bit(0); // enable_ccso
    }
    b.write_bits(0, 2);
    b.write_bit(0);
    b.write_bit(0);
    b.write_bit(0);
    b.align_with_one();
    b.into_bytes()
}

/// Build the key-frame header OBU payload (the tile data is appended by the caller).
pub(crate) fn frame_header(
    config: &Config,
    width: u32,
    height: u32,
    tiles: (usize, usize, usize),
) -> Vec<u8> {
    let has_chroma = config.layout.has_chroma();
    let mut b = ByteWriter::new();
    b.write_bit(1);
    b.write_uvlc(0);
    b.write_uvlc(0);
    b.write_bit(0);
    b.write_bit(0);
    // This bit is read by the decoder as `disable_cdf_update` (decodeframe.c:9155),
    // immediately before read_tile_info. The original encoder hard-coded it to 1
    // (static CDFs). It now carries !updating_cdf: 1 = static, 0 = adaptive.
    b.write_bit(!config.updating_cdf as u32); // disable_cdf_update

    let sb_cols = (width as usize).div_ceil(64);
    let sb_rows = (height as usize).div_ceil(64);
    let (log2c, log2r, tsb) = tiles;
    let tlog2 = |blk: usize, tgt: usize| {
        let mut k = 0;
        while (blk << k) < tgt {
            k += 1;
        }
        k
    };
    // seq_max_level_idx = SEQ_LEVEL_MAX makes the decoder's max tile width and max
    // tile area unconstrained (get_max_tile_width/area return sb_cols / sb_cols*sb_rows),
    // so it computes min_log2_cols = 0 and min_log2_tiles = 0 — NOT tile_log2(64, sb).
    // The tile_info increment loops must start from these same minima or the decoder
    // reads the wrong TileColsLog2/TileRowsLog2. Using tile_log2(64, sb) desynced every
    // tiled frame with a side > 4096px (sb_cols/sb_rows > 64); small frames matched only
    // because tile_log2(64, sb)=0 there. min_log2_tile_rows = max(min_log2_tiles-log2c,0)=0.
    let (min_lc, max_lc) = (0usize, tlog2(1, sb_cols.min(64)));
    let (min_lr, max_lr) = (0usize, tlog2(1, sb_rows.min(64)));
    b.write_bit(1); // uniform_tile_spacing_flag
    for _ in min_lc..log2c {
        b.write_bit(1);
    }
    if log2c < max_lc {
        b.write_bit(0);
    }
    for _ in min_lr..log2r {
        b.write_bit(1);
    }
    if log2r < max_lr {
        b.write_bit(0);
    }
    if log2c > 0 || log2r > 0 {
        // NB: single_picture_header_flag=1 forces enable_avg_cdf=avg_cdf_type=1 in the
        // sequence header, which makes the decoder OMIT context_update_tile_id in
        // tile_info(). Emitting it here would shift tile_size_bytes and misalign the
        // tile data. So we write only tile_size_bytes_minus_1.
        b.write_bits((tsb - 1) as u32, 2); // tile_size_bytes_minus_1
    }
    // AV2 frame_header quant.yac is 8 bits for 8-bit streams and 9 bits for
    // high-bit-depth streams. The decoder reads 8 + (seqhdr.hbd != 0) bits.
    let q_bits = 8 + u32::from(config.bit_depth > 8);
    b.write_bits(if config.lossless { 0 } else { config.base_q }, q_bits);
    b.write_bit(0);
    b.write_bit(0); // segmentation.enabled = 0, qm.enabled = 0
    // delta_q_params: present flag (gated on quant.yac != 0, true for lossy),
    // then 2-bit res_log2. The decoder reads no delta_lf here (AVM continues with
    // TCQ/parity), so this inserts exactly present(+res) with no further change.
    if config.aq && !config.lossless {
        b.write_bit(1); // delta_q_present = 1
        b.write_bits(config.aq_res_log2 as u32, 2);
    } else {
        b.write_bit(0); // delta_q_present = 0
    }

    if config.lossless {
        // coded_lossless: setup_loopfilter returns early, guided-deblock/CDEF/CCSO are
        // skipped, and read_tx_mode is forced to ONLY_4X4 with no bit. Only the
        // 2-bit reduced_tx_set remains before byte alignment.
        b.write_bits(0, 2); // reduced_tx_set
        if log2c > 0 || log2r > 0 {
            b.write_bit(0); // tile_start_and_end_present_flag = 0
        }
        b.align_with_zero();
        return b.into_bytes();
    }

    if config.deblock {
        // AV2 setup_loopfilter syntax. df_par_bits = df_par_bits_minus2 + 2 = 2
        // (the sequence header writes df_par_bits_minus2 = 0), so df_par_offset = 2
        // and each signaled delta is a 2-bit literal in [-2, 1].
        let (av, ah, au, avv) = config.db_apply;
        let (dy0, dy1, du, dv) = config.db_delta;
        const DF_PAR_BITS: u32 = 2;
        const DF_PAR_OFFSET: i32 = 1 << (DF_PAR_BITS - 1); // 2
        let write_delta = |b: &mut ByteWriter, on: bool, delta: i32| {
            if on {
                if delta != 0 {
                    b.write_bit(1);
                    b.write_bits((delta + DF_PAR_OFFSET) as u32, DF_PAR_BITS);
                } else {
                    b.write_bit(0);
                }
            }
        };
        b.write_bit(av as u32); // apply_deblocking_filter[0] (vertical edges)
        b.write_bit(ah as u32); // apply_deblocking_filter[1] (horizontal edges)
        if has_chroma && (av || ah) {
            b.write_bit(au as u32); // apply_deblocking_filter_u
            b.write_bit(avv as u32); // apply_deblocking_filter_v
        }
        // Per-direction luma deltas (delta_side mirrors delta_q). Direction 0's
        // "no delta" means 0; direction 1's "no delta" means reuse direction 0.
        write_delta(&mut b, av, dy0);
        write_delta(&mut b, ah, dy1);
        // Chroma deltas.
        write_delta(&mut b, has_chroma && (av || ah) && au, du);
        write_delta(&mut b, has_chroma && (av || ah) && avv, dv);
    } else {
        b.write_bit(0);
        b.write_bit(0);
    }

    if let Some(gdf) = config.guided_deblock {
        b.write_bit(0);
        b.write_bits(gdf.qp_offset, 2);
        b.write_bits(gdf.scale_minus_one, 2);
    }
    if let Some((y_str, uv_str, damping)) = config.cdef {
        // setup_cdef: single_picture_header forces cdef_frame_enable=1 (no bit) and
        // enable_cdef_on_skip_txfm=ADAPTIVE (one frame bit). nb_cdef_strengths=1.
        b.write_bits((damping as u32).saturating_sub(3) & 3, 2); // cdef_damping-3
        b.write_bits(0, 3); // nb_cdef_strengths - 1 (== 0)
        b.write_bit(0); // cdef_on_skip_txfm_frame_enable
        let mut wstr = |s: u8| {
            if s < 4 {
                b.write_bit(1);
                b.write_bits(s as u32, 2);
            } else {
                b.write_bit(0);
                b.write_bits(s as u32, 6); // CDEF_STRENGTH_BITS
            }
        };
        wstr(y_str);
        if has_chroma {
            wstr(uv_str);
        }
    }
    // CCSO frame params (setup_ccso). single_picture_header_flag => ccso_frame_flag
    // is implied 1 (no bit). Intra-only => no reuse bits. Each enabled plane writes
    // bo_only (1b), scale_idx (2b), then either max_band_log2 (3b) for bo_only, or
    // quant_idx (2b) + ext_filter_support (3b) + [edge_clf (1b) when quant_sz != 0] +
    // max_band_log2 (2b) for edge mode; followed by the offset LUT (unary indices).
    if let Some(cc) = &config.ccso {
        const CCSO_OFFSET: [i32; 8] = [0, 1, -1, 3, -3, 7, -7, -10];
        // quant_sz[scale][quant]: a zero entry means edge_clf is implied 0 (no bit).
        const QUANT_SZ: [[u16; 4]; 4] = [
            [16, 8, 32, 0],
            [56, 40, 64, 128],
            [48, 24, 96, 192],
            [80, 112, 160, 256],
        ];
        const EDGE_INTERVAL: [usize; 2] = [3, 2];
        let write_off = |b: &mut ByteWriter, raw: i32| {
            let idx = CCSO_OFFSET.iter().position(|&v| v == raw).unwrap_or(0);
            for i in 0..7 {
                b.write_bit((idx != i) as u32);
                if idx == i {
                    break;
                }
            }
        };
        let num_planes = if has_chroma { 3 } else { 1 };
        for plane in 0..num_planes {
            b.write_bit(cc.enable[plane] as u32);
            if !cc.enable[plane] {
                continue;
            }
            let p = cc.planes[plane]
                .as_ref()
                .expect("enabled CCSO plane must carry params");
            b.write_bit(p.bo_only as u32);
            b.write_bits(p.scale_idx as u32, 2);
            if p.bo_only {
                b.write_bits(p.max_band_log2 as u32, 3);
                let max_band = 1usize << p.max_band_log2;
                for band in 0..max_band {
                    let off = p.offsets.get(band << 4).copied().unwrap_or(0);
                    write_off(&mut b, off);
                }
            } else {
                b.write_bits(p.quant_idx as u32, 2);
                b.write_bits(p.ext_filter_support as u32, 3);
                if QUANT_SZ[p.scale_idx as usize][p.quant_idx as usize] != 0 {
                    b.write_bit(p.edge_clf as u32);
                }
                b.write_bits(p.max_band_log2 as u32, 2);
                let max_band = 1usize << p.max_band_log2;
                let ni = EDGE_INTERVAL[p.edge_clf as usize];
                for d0 in 0..ni {
                    for d1 in 0..ni {
                        for band in 0..max_band {
                            let lut = (band << 4) + (d0 << 2) + d1;
                            let off = p.offsets.get(lut).copied().unwrap_or(0);
                            write_off(&mut b, off);
                        }
                    }
                }
            }
        }
    }
    b.write_bit(if config.tx_switchable { 1 } else { 0 }); // txfm_mode: 1=SWITCHABLE
    b.write_bits(0, 2); // reduced_txtp_set
    if log2c > 0 || log2r > 0 {
        b.write_bit(0); // tile_start_and_end_present_flag = 0 (decoder reads it here,
        // folded into the frame header's own byte alignment)
    }
    b.align_with_zero();
    b.into_bytes()
}
