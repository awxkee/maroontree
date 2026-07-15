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

use super::*;

/// Resolve the requested tile grid (`Tuning::tile_cols/rows`) into (log2_cols,
/// log2_rows). Rounds each request up to a power of two and clamps to the AV2 maximum
/// log2 for the available (including a partial final) superblock count. Returns `None`
/// for a single tile.
pub(super) fn tile_grid_for(
    tile_cols: usize,
    tile_rows: usize,
    width: usize,
    height: usize,
) -> Option<(usize, usize)> {
    let log2 = |n: usize| {
        let mut k = 0;
        while (1usize << k) < n {
            k += 1;
        }
        k
    };
    let (mut lc, mut lr) = (log2(tile_cols.max(1)), log2(tile_rows.max(1)));
    // A uniform tile needs at least one complete SB on every non-final span.
    // Do not count the partial last SB when clamping the requested power-of-two
    // count: allowing tile_sbs == 0 makes the encoder and consumers disagree on
    // the number of tile payloads for sizes such as 453x604. This direct clamp
    // retains the largest valid power-of-two grid (4x8 for that example).
    let floor_log2 = |n: usize| usize::BITS as usize - 1 - n.max(1).leading_zeros() as usize;
    let full_sbs = |dim: usize| ((dim + 7) >> 6).clamp(1, 64);
    lc = lc.min(floor_log2(full_sbs(width)));
    lr = lr.min(floor_log2(full_sbs(height)));
    if lc == 0 && lr == 0 {
        return None;
    }
    Some((lc, lr))
}

/// Copy the `tw x th` sub-plane at `(x0, y0)` out of a `width`-stride plane.
pub(super) fn extract_subplane(
    p: &[f32],
    width: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<f32> {
    let mut o = vec![0f32; tw * th];
    for (dst_row, src_row) in o
        .chunks_exact_mut(tw)
        .zip(rect_rows(p, width, y0, x0, tw, th))
    {
        dst_row.copy_from_slice(src_row);
    }
    o
}

pub(super) fn pad_plane_pixels<T: Pixel>(
    src: &[T],
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); padded_width * padded_height];
    for y in 0..padded_height {
        let sy = y.min(height - 1);
        for x in 0..padded_width {
            out[y * padded_width + x] = src[sy * width + x.min(width - 1)];
        }
    }
    out
}

pub(super) fn extract_subplane_pixels<T: Pixel>(
    src: &[T],
    stride: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); width * height];
    for row in 0..height {
        out[row * width..(row + 1) * width]
            .copy_from_slice(&src[(y0 + row) * stride + x0..(y0 + row) * stride + x0 + width]);
    }
    out
}

/// Extract the entropy-coded tile from a standalone single-tile still frame.
pub(super) fn lossless_tile_payload(
    frame: &Av2Frame,
    config: &Config,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let (_, _, obu) = frame.split_obus();
    let mut leb_len = 0usize;
    while obu[leb_len] & 0x80 != 0 {
        leb_len += 1;
    }
    leb_len += 1;
    let payload = &obu[leb_len + 1..]; // OBU header follows the leading size LEB.
    let header_len = frame_header(config, width as u32, height as u32, (0, 0, 1)).len();
    payload[header_len..].to_vec()
}

/// Tile SB-start boundaries exactly as AV2 `uniform_spacing()` computes them. The base
/// size is derived from the number of full SBs, but the loop and final boundary use the
/// ceil SB count. When `tile_sb == 0`, every existing (including partial) SB receives a
/// one-SB tile until either the requested power-of-two count or the SB count is reached.
fn tile_starts(dim_px: usize, log2: usize) -> Vec<usize> {
    let tile_count = 1usize << log2;
    let mi = 2 * ((dim_px + 7) >> 3);
    let ceil_sbs = (mi + 15) >> 4;
    let full_sbs = mi >> 4;
    let tile_sbs = full_sbs >> log2;
    let extra_sbs = if tile_sbs == 0 {
        ceil_sbs
    } else {
        full_sbs - (tile_sbs << log2)
    };
    let mut starts = Vec::with_capacity(tile_count.min(ceil_sbs) + 1);
    let mut start_sb = 0usize;
    for i in 0..tile_count {
        if start_sb >= ceil_sbs {
            break;
        }
        starts.push(start_sb);
        start_sb += tile_sbs + if i < extra_sbs { 1 } else { 0 };
    }
    starts.push(ceil_sbs); // sentinel: tile k spans [starts[k], starts[k+1])
    starts
}

/// Luma tile regions in raster order: `(x0, y0, tw, th)` (all luma-pixel units, x0/y0 on
/// 64-px SB boundaries). Tile column/row boundaries are in superblock units and so are
/// identical across chroma formats.
pub(super) fn tile_specs(
    width: usize,
    height: usize,
    log2c: usize,
    log2r: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let col_starts = tile_starts(width, log2c);
    let row_starts = tile_starts(height, log2r);
    let mut specs = Vec::new();
    for tr in 0..row_starts.len() - 1 {
        let (r0, r1) = (row_starts[tr], row_starts[tr + 1]);
        let (y0, th) = (r0 * 64, (r1 * 64).min(height) - r0 * 64);
        for tc in 0..col_starts.len() - 1 {
            let (c0, c1) = (col_starts[tc], col_starts[tc + 1]);
            let (x0, tw) = (c0 * 64, (c1 * 64).min(width) - c0 * 64);
            specs.push((x0, y0, tw, th));
        }
    }
    specs
}

/// Shared inputs for the format-specific tiled encoders. The three source planes,
/// frame geometry, tile layout, header configuration and worker count always travel
/// together from the public format API into the tiled implementation.
/// Inter reference context for one tile encode. Tiles are independent only for
/// intra/entropy *within the current frame*; motion compensation reads the FULL
/// reference frame freely (an MV may point across a tile boundary, exactly as the
/// decoder reconstructs it). So the reference is kept whole and indexed at
/// frame-absolute coordinates: the core adds `(x0, y0)` to its region-local block
/// position and uses `luma_stride`/`chroma_stride` (frame strides), rather than
/// cropping the reference to the tile (which would edge-replicate real neighbor
/// content and desync the decoder for larger motion).
#[derive(Clone)]
pub(super) struct TileRefCtx {
    pub(super) planes: std::sync::Arc<Vec<Vec<f32>>>, // shared full-frame Y,U,V
    pub(super) luma_stride: usize,
    pub(super) chroma_stride: usize,
    pub(super) x0: usize, // tile luma origin (frame coords)
    pub(super) y0: usize,
}

#[derive(Clone, Copy)]
pub(super) struct TiledEncodeRequest<'a> {
    pub(super) y: &'a [f32],
    pub(super) u: &'a [f32],
    pub(super) v: &'a [f32],
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) config: &'a Config,
    pub(super) color: &'a Cicp,
    pub(super) log2_cols: usize,
    pub(super) log2_rows: usize,
    pub(super) threads: usize,
}

/// Header and display metadata needed to assemble already encoded tile payloads.
#[derive(Clone, Copy)]
pub(super) struct MultitileAssembly<'a> {
    pub(super) config: &'a Config,
    pub(super) coded_width: usize,
    pub(super) coded_height: usize,
    pub(super) display_width: usize,
    pub(super) display_height: usize,
    pub(super) color: &'a Cicp,
    pub(super) log2_cols: usize,
    pub(super) log2_rows: usize,
    pub(super) bit_depth: u8,
    pub(super) chroma_format: ChromaFormat,
}

/// Wrap already-encoded per-tile byte streams (raster order) into a single multi-tile
/// frame: frame header with the tile grid, `tsb`-byte size prefixes before every tile
/// but the last, then the TD/SEQ/FRAME OBUs. Format-agnostic — chroma signalling lives
/// in the assembly metadata.
pub(super) fn assemble_multitile(
    assembly: &MultitileAssembly<'_>,
    tiles_bytes: &[Vec<u8>],
) -> Av2Frame {
    let MultitileAssembly {
        config,
        coded_width: sig_w,
        coded_height: sig_h,
        display_width: disp_w,
        display_height: disp_h,
        color,
        log2_cols: log2c,
        log2_rows: log2r,
        bit_depth,
        chroma_format,
    } = *assembly;
    let n = tiles_bytes.len();
    // Fixed TileSizeBytes = 4 (matches the reference encoders; always sufficient).
    let tsb = 4usize;
    let mut frame = frame_header(config, sig_w as u32, sig_h as u32, (log2c, log2r, tsb));
    for (i, t) in tiles_bytes.iter().enumerate() {
        if i + 1 < n {
            let v = t.len() - 1; // - AV2_MIN_TILE_SIZE_BYTES (=1)
            for b in 0..tsb {
                frame.push(((v >> (8 * b)) & 0xff) as u8);
            }
        }
        frame.extend(t);
    }
    let mut data = vec![];
    data.extend(obu(2, &[]));
    data.extend(obu(1, &sequence_header(config, sig_w as u32, sig_h as u32)));
    data.extend(obu(4, &frame));
    Av2Frame {
        data,
        width: disp_w,
        height: disp_h,
        // Coded size = the OBU-signaled size. When it exceeds the display size (the
        // padded-tiling fallback for non-boundary-exact frames) the AVIF muxer crops
        // via a `clap` box.
        coded_width: sig_w,
        coded_height: sig_h,
        bit_depth,
        color: *color,
        chroma_format,
        tile_grid: (log2c, log2r, tsb),
        recon: Vec::new(),
        allow_intrabc: config.allow_intrabc,
        video_config: config.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{tile_grid_for, tile_specs, tile_starts};

    #[test]
    fn uniform_spacing_keeps_partial_final_superblock() {
        // 484 px -> 122 mi -> 7 full SBs plus one partial SB. AV2's tileSb==0
        // branch gives all eight existing SBs their own tile.
        assert_eq!(tile_starts(484, 3), (0usize..=8).collect::<Vec<_>>());
        assert_eq!(tile_grid_for(8, 1, 484, 64), Some((2, 0)));
    }

    #[test]
    fn uniform_spacing_matches_full_sb_remainder_rule() {
        // 4000 px -> 1000 mi -> 62 full / 63 ceil SBs. Four uniform tiles start
        // at 0,16,32,47; the partial final SB extends the last tile to 63.
        assert_eq!(tile_starts(4000, 2), vec![0, 16, 32, 47, 63]);
    }

    #[test]
    fn requested_grid_and_emitted_tile_count_stay_in_sync() {
        let grid = tile_grid_for(8, 8, 484, 484).unwrap();
        assert_eq!(grid, (2, 2));
        assert_eq!(tile_specs(484, 484, grid.0, grid.1).len(), 16);
    }

    #[test]
    fn excessive_request_clamps_to_complete_superblocks() {
        let grid = tile_grid_for(8, 8, 453, 604).unwrap();
        assert_eq!(grid, (2, 3));
        let specs = tile_specs(453, 604, grid.0, grid.1);
        assert_eq!(specs.len(), 32);
        assert!(specs.iter().all(|&(_, _, w, h)| w != 0 && h != 0));
    }
}
