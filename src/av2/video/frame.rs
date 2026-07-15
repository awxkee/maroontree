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

//! Video KEY-frame OBU (OBU_CLOSED_LOOP_KEY=4). Preamble per AVM
//! `read_uncompressed_header` KEY branch, then the shared frame-header body.

use crate::av2::entropy::ByteWriter;
use crate::av2::headers::{Config, frame_header, frame_header_body, obu};
use crate::err::EncodeError;

#[inline]
fn write_screen_content_and_intrabc(b: &mut ByteWriter, cfg: &Config, allow_intrabc: bool) {
    if cfg.lossless {
        // sequence_header_video selects both decisions per frame for lossless.
        b.write_bit(1); // allow_screen_content_tools (palettes / IBC)
        b.write_bit(0); // force_integer_mv
    }
    b.write_bit(allow_intrabc as u32);
    if allow_intrabc {
        b.write_bit(1); // allow_global_intrabc
        b.write_bit(1); // allow_local_intrabc
    }
}

/// Tile entropy bytes from a still frame OBU: `obu(4, still_header || tile)`.
/// Split point = still-header length recomputed from (config, dims).
pub(crate) fn tile_payload_of<'a>(
    still_frame_obu: &'a [u8],
    cfg: &Config,
    sw: u32,
    sh: u32,
    tiles: (usize, usize, usize),
) -> &'a [u8] {
    let mut i = 0;
    loop {
        let b = still_frame_obu[i];
        i += 1;
        if b & 0x80 == 0 {
            break;
        }
    }
    let payload = &still_frame_obu[i + 1..];
    // Strip the still frame header for the ACTUAL tile grid, leaving the
    // size-prefixed tile data (identical bytes the video header re-wraps).
    let hdr_len = frame_header(cfg, sw, sh, tiles).len();
    &payload[hdr_len..]
}

/// Video KEY-frame OBU (type 4) = preamble + shared body (video=true) + tile.
pub(crate) fn video_key_frame(
    cfg: &Config,
    sw: u32,
    sh: u32,
    tile: &[u8],
    order_hint: u64,
    tiles: (usize, usize, usize),
) -> Result<Vec<u8>, EncodeError> {
    let has_chroma = cfg.layout.has_chroma();
    let mut b = ByteWriter::new();
    // Preamble (CLK KEY, non-single-picture, single-layer, order_hint_bits=1).
    b.write_bit(1); // is_first_tile_group (send_uncompressed_header inferred 1)
    b.write_uvlc(0); // mfh_id
    b.write_uvlc(0); // seq_header_id
    b.write_bit(1); // immediate_output_picture (shown keyframe)
    b.write_bit(0); // frame_size_override_flag
    b.write_bits((order_hint & 0x7f) as u32, 7); // order_hint (7 bits)
    // setup_frame_size/render: no bits. Then frame-selected SCC and IBC.
    write_screen_content_and_intrabc(&mut b, cfg, cfg.allow_intrabc);
    frame_header_body(&mut b, cfg, sw, sh, tiles, has_chroma, true, false);

    b.align_with_zero();
    let mut payload = b.into_bytes();
    payload.extend_from_slice(tile);
    Ok(obu(4, &payload))
}

/// Inter frame (OBU_REGULAR_TILE_GROUP=7), one explicitly selected forward
/// reference and one rotating DPB refresh slot.
/// Minimal-tool seq: mvs/tip/global-motion/bawp off, order_hint_bits=1.
pub(crate) struct VideoInterFrameSpec<'a> {
    pub(crate) sw: u32,
    pub(crate) sh: u32,
    pub(crate) tile: &'a [u8],
    pub(crate) order_hint: u64,
    pub(crate) tiles: (usize, usize, usize),
    pub(crate) reference_slot: usize,
    pub(crate) refresh_slot: usize,
}

pub(crate) fn video_inter_frame(
    cfg: &Config,
    spec: VideoInterFrameSpec<'_>,
) -> Result<Vec<u8>, EncodeError> {
    let VideoInterFrameSpec {
        sw,
        sh,
        tile,
        order_hint,
        tiles,
        reference_slot,
        refresh_slot,
    } = spec;
    let has_chroma = cfg.layout.has_chroma();
    let mut b = ByteWriter::new();
    b.write_bit(1); // is_first_tile_group
    b.write_uvlc(0); // mfh_id
    b.write_uvlc(0); // seq_header_id
    b.write_bit(1); // frame_type: 1 => INTER
    b.write_bit(1); // immediate_output_picture
    b.write_bit(0); // frame_size_override_flag
    b.write_bits((order_hint & 0x7f) as u32, 7); // order_hint (7 bits)
    b.write_bit(1); // signal_primary_ref_frame
    b.write_bit(0); // cross_frame_context
    b.write_bits(7, 3); // primary_ref_frame = PRIMARY_REF_NONE (reset CDFs)
    b.write_bits(1u32 << refresh_slot, 8);
    b.write_bit(1); // explicit_ref_frame_map
    b.write_bits(1, 3); // num_total_refs = 1 (no per-block ref symbol)
    b.write_bits(reference_slot as u32, 3); // decoder DPB slot
    // allow_ref_frame_mvs is omitted because the sequence disables it.
    // Inter tiles do not use IBC; lossless still selects SCC so palette syntax in
    // the reused tile payload remains valid.
    write_screen_content_and_intrabc(&mut b, cfg, false);
    b.write_bit(1); // fr_mv_precision == QTR_PEL
    b.write_bit(0); // interp: not switchable
    b.write_bits(0, 2); // EIGHTTAP_REGULAR
    frame_header_body(&mut b, cfg, sw, sh, tiles, has_chroma, true, true);
    b.align_with_zero();
    let mut payload = b.into_bytes();
    payload.extend_from_slice(tile);
    Ok(obu(7, &payload))
}
