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

//! Video (`single_picture_header_flag=0`) sequence header. Mirrors AVM's
//! `read_sequence_*_group_tool_flags` (see docs/rd_notes.md). Low-delay,
//! translation-only toolset: every inter tool we don't use is written 0.
//! Bit-exactness is finalized against the pinned `avmdec` (milestone-3 gate).

use crate::av2::entropy::ByteWriter;
use crate::av2::headers::{Config, DISABLE_LOOPFILTERS_ACROSS_TILES, ENABLE_INTRA_EDGE_FILTER};

/// Sequence header OBU payload for a multi-frame (video) stream.
pub(crate) fn sequence_header_video(config: &Config, width: u32, height: u32) -> Vec<u8> {
    let has_chroma = config.layout.has_chroma();
    let mut b = ByteWriter::new();
    b.write_uvlc(0); // seq_header_id
    b.write_bits(config.layout.profile(), 5); // seq_profile_idc
    b.write_bit(0); // single_picture_header_flag = 0 (video)
    b.write_bits(31, 5); // seq_max_level_idx (LEVEL_BITS=5), MAX
    b.write_bit(0); // seq_tier (level>=4_0 && !single_picture)
    b.write_uvlc(config.layout.header_uvlc()); // chroma_format_idc
    let bitdepth_idx = match config.bit_depth {
        10 => 0,
        12 => 2,
        _ => 1,
    };
    b.write_uvlc(bitdepth_idx);
    // Video-only layer block (single-picture infers all of these).
    b.write_bits(0, 3); // seq_lcr_id
    b.write_bit(0); // still_picture
    b.write_bits(0, 2); // max_tlayer_id (TLAYER_BITS)
    b.write_bits(0, 3); // max_mlayer_id (MLAYER_BITS); 0 => no mlayer_cnt loop
    b.write_bit(1); // monotonic_output_order_flag

    let width_bits = (32 - (width - 1).leading_zeros()).max(1);
    let height_bits = (32 - (height - 1).leading_zeros()).max(1);
    b.write_bits(width_bits - 1, 4);
    b.write_bits(height_bits - 1, 4);
    b.write_bits(width - 1, width_bits);
    b.write_bits(height - 1, height_bits);

    // conformance window (crop) — off.
    b.write_bit(0);
    // Video-only display/decoder model presence — both off.
    b.write_bit(0); // seq_max_display_model_info_present_flag
    b.write_bit(0); // decoder_model_info_present_flag

    // partition/segment/intra groups: identical to still (single_picture only
    // affects enable_extended_sdp, gated on sdp=0). Mirror the still writer.
    b.write_bit(0); // sb size bit 0
    b.write_bit(0); // sb size bit 1 (=> BLOCK_64X64)
    if has_chroma {
        b.write_bit(0); // enable_sdp
    }
    b.write_bit(0); // enable_ext_partitions
    b.write_bit(0); // max_pb_aspect present
    b.write_bit(0); // enable_ext_seg
    b.write_bit(0); // seq_seg_info_present
    b.write_bit(0); // dip
    // Must match the actual directional predictor. Advertising this as disabled while
    // encoding filtered edges produces decoder drift which restarts at every tile and
    // therefore appears as large tile-aligned bands/rectangles.
    b.write_bit(ENABLE_INTRA_EDGE_FILTER as u32);
    b.write_bit(0); // mrl
    b.write_bit(config.cfl as u32); // cfl_intra
    if has_chroma {
        b.write_bits(0, 2); // cfl_ds_filter_index
    }
    b.write_bit(config.mhccp as u32); // enable_mhccp
    b.write_bit(0); // enable_ibp

    // --- inter group (video: full field set, all tools off) ---
    // motion-mode loop INTERINTRA..MOTION_MODES = 4 bits, all 0 (translation only).
    b.write_bits(0, 4);
    // no motion mode enabled -> seq_frame_motion_modes_present omitted;
    // no WARP_DELTA -> enable_six_param_warp_delta omitted.
    b.write_bit(0); // enable_masked_compound
    b.write_bit(0); // enable_ref_frame_mvs (0 -> reduced_ref_frame_mvs_mode omitted)
    b.write_bits(6, 4); // order_hint_bits_minus_1 (7 bits)
    b.write_bit(0); // enable_refmvbank
    b.write_bit(1); // drl-reorder: first bit 1 => DRL_REORDER_DISABLED (no 2nd bit)
    // explicit ref-frame-map block
    b.write_bit(1); // enable_explicit_ref_frame_map
    b.write_bit(0); // DPB size present (0 => ref_frames = 8 default)
    b.write_bits(0, 3); // number_of_bits_for_lt_frame_id
    b.write_uniform(0, 5); // def_max_drl_bits quniform over range 5
    b.write_bit(0); // allow_frame_max_drl_bits
    // always-present bvp drl
    b.write_uniform(0, 3); // def_max_bvp_drl_bits quniform over range 3
    b.write_bit(0); // allow_frame_max_bvp_drl_bits
    b.write_bits(0, 2); // num_same_ref_compound
    b.write_bit(0); // enable_tip (0 -> tip level/hole-fill omitted)
    b.write_bit(0); // enable_mv_traj
    b.write_bit(0); // enable_bawp
    b.write_bit(0); // enable_cwp
    b.write_bit(0); // enable_imp_msk_bld
    b.write_bit(0); // enable_lf_sub_pu
    // tip_explicit_qp gated on tip==1 && lf_sub_pu (both 0) -> omitted
    b.write_bits(0, 2); // enable_opfl_refine
    b.write_bit(0); // enable_refinemv
    // tip_refinemv gated on tip -> omitted
    b.write_bit(0); // enable_bru
    b.write_bit(0); // enable_adaptive_mvd
    b.write_bit(0); // enable_mvd_sign_derive
    b.write_bit(0); // enable_flex_mvres
    b.write_bit(0); // enable_global_motion
    b.write_bit(0); // enable_short_refresh_frame_flags

    // --- scc group ---
    // Lossless video may use palettes and IBC on key frames, so select both SCC
    // decisions per frame. Lossy video keeps SCC forced off as before.
    if config.lossless {
        b.write_bit(1); // seq_choose_screen_content_tools => SELECT
        b.write_bit(1); // seq_choose_integer_mv => SELECT
    } else {
        b.write_bit(0); // seq_choose_screen_content_tools
        b.write_bit(0); // seq_force_screen_content_tools = 0
    }

    // --- transform/quant/entropy group ---
    b.write_bit(0); // enable_fsc (0 -> enable_idtx_intra bit follows)
    b.write_bit(0); // enable_idtx_intra
    b.write_bit(0); // enable_ist
    b.write_bit(0); // enable_inter_ist
    if has_chroma {
        b.write_bit(0); // enable_chroma_dctonly
    }
    b.write_bit(0); // enable_inter_ddt (video-only field)
    b.write_bit(0); // reduced_tx_part_set
    if has_chroma {
        b.write_bit(0); // enable_cctx
    }
    b.write_bit(0); // enable_tcq (0 -> no 2nd tcq bit, parity-hiding bit follows)
    b.write_bit(0); // enable_parity_hiding
    b.write_bit(0); // enable_avg_cdf (video: 0 -> no avg_cdf_type bit)
    if has_chroma {
        b.write_bit(0); // separate uv delta-q
    }
    b.write_bit(1); // equal ac/dc quant
    if has_chroma {
        let uv_raw = (23 + config.uv_ac_delta_q).clamp(0, 31) as u32;
        b.write_bits(uv_raw, 5); // base uv-ac delta-q (raw 23 => delta 0)
        b.write_bit(0); // uv-ac delta-q enabled
    }

    // --- filter group ---
    b.write_bit(DISABLE_LOOPFILTERS_ACROSS_TILES as u32);
    b.write_bit(config.cdef.is_some() as u32); // enable_cdef
    b.write_bit(0); // enable_gdf (guided deblock)
    b.write_bit(0); // enable_restoration
    if config.ccso.is_some() {
        b.write_bit(1); // enable_ccso
        b.write_bit(1); // ccso_unit_matches_sb_size
    } else {
        b.write_bit(0);
    }
    // enable_cdef_on_skip_txfm (video-only): outer bit 0, inner bit 0 => ADAPTIVE.
    b.write_bit(0);
    b.write_bit(0);
    b.write_bits(0, 2); // df_par_bits_minus2

    b.write_bit(0); // seq_tile_info_present_flag
    b.write_bit(0); // film_grain_params_present
    b.write_bit(0); // seq_extension_present_flag

    b.align_with_one();
    b.into_bytes()
}
