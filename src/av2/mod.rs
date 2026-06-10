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
mod avif;
mod cdfs_qctx;
mod cdfx_4tx;
mod chroma422;
mod coder;
mod csc;
mod encode400;
mod encode420;
mod encode422;
mod encode444;
mod entropy;
mod headers;
mod helpers;
mod intrapred;
pub mod itx422;
mod layout;
mod leaf;
mod lossless;
mod partition;
mod proj;
mod quant;
mod tables;
mod tables_tx32;
mod wht;

use crate::av2::avif::{Av2Color, Av2Format};
use crate::av2::cdfs_qctx::{
    CHROMA_EOB_BIN_QC, CHROMA_EOB_HI_BIT_QC, CHROMA_EOB128_QC, CHROMA_EOB256_QC, CHROMA_EOB512_QC,
    CHROMA_SKIP_TX32_QC, CHROMA_SKIP_TX64_QC, CHROMA_SKIP_V_QC, SKIP_TX16_QC,
};
use crate::av2::cdfx_4tx::{TXB_SKIP_TX4_Q0, V_TXB_SKIP_TX4_Q0};
use crate::av2::chroma422::{
    ChromaNeighbors, ChromaPlanes, ChromaTxSpec, code_422_chroma_tu, recon_422_chroma,
};
use crate::av2::coder::{
    Coeff, encode_chroma_block, encode_chroma_block_rect, encode_chroma_tu4,
    encode_lossless_luma_sb, encode_luma_block_split, encode_luma_leaf_16x64,
    encode_luma_leaf_32x32, encode_luma_leaf_32x64, encode_luma_leaf_64x16, encode_luma_leaf_64x32,
    encode_luma_leaf_dc_class2,
};
use crate::av2::csc::{
    CB_B, CB_G, CB_R, CR_B, CR_G, CR_R, HALF, Q, Y_B, Y_G, Y_R, get_q_ctx, validate_dims,
};
use crate::av2::entropy::RangeEncoder;
use crate::av2::headers::{Config, frame_header, obu, sequence_header};
use crate::av2::helpers::{
    dc_pred, dc_pred_rect, get_residual, get_residual_rect, levels_to_coeffs, lossless_sb_tus,
    pad_plane, put_block, put_block_rect, sb_align, sb_tu_contexts, sb_tu_contexts_64x32,
    sb_tu_contexts_pos, sb_tu_contexts_rect, sb_tu4_chroma_skip, sb_tu4_contexts,
};
use crate::av2::itx422::reconstruct_luma;
use crate::av2::layout::Layout;
use crate::av2::leaf::{
    encode_luma_leaf_s32x32, encode_luma_leaf_v32x64, encode_luma_leaf32, encode_luma_sb,
};
use crate::av2::proj::Basis;
use crate::av2::tables::{SCAN8X32, SCAN16, SCAN16X32, SCAN32X8, SCAN32X16};
use crate::err::EncodeError;
use crate::{ChromaFormat, ColorEncoding, Pixel, PlanarImage};

// Free luma-leaf prediction/coding helpers live in `leaf`.

// Q0.13 RGB→YCbCr coefficients, dimension validation, and `get_q_ctx` live in `csc`.

/// Result of an encode: the AV2 bitstream plus the metadata needed to interpret it.
pub struct Av2Frame {
    data: Vec<u8>,
    width: usize,
    height: usize,
    /// Coded (decoder-output) dimensions = the size signaled in the OBU. Equal to
    /// width/height for lossless and SB-aligned lossy; for padded lossy they are the
    /// 64-aligned size, and the AVIF muxer adds a `clap` box cropping to width/height.
    coded_width: usize,
    coded_height: usize,
    bit_depth: u8,
    color: ColorEncoding,
    chroma_format: ChromaFormat,
}

impl Av2Frame {
    pub fn view(&self) -> &[u8] {
        self.data.as_slice()
    }
}

/// A reusable still-image encoder configured for one quality.
///
/// `Av2Encoder::new(q)` loads the bundled q120 bases and rescales them to the target
/// `base_q_idx` once (see [`proj::Bases::rescaled_to_q`]); the per-superblock encode
/// then reuses that precomputed set. Lower `base_q_idx` → finer quantizer → larger,
/// higher-quality output; higher → coarser/smaller.
pub struct Av2Encoder {
    bases: proj::Bases,
    base_q_idx: u8,
    bit_depth: u8,
}

/// Returns the AV2 mi-unit frame extents `(mc, mr)` for a native (no-pad) lossy 4:4:4
/// encode, iff both dimensions are "boundary-safe". A dimension is boundary-safe when
/// the last superblock has >8 mi in-frame: mc%16==0 || mc%16>8, where mc =
/// ALIGN_POWER_OF_TWO(W,3)>>2 (avm's mi_cols). Returns None if either dimension is
/// not boundary-safe; the encoder then falls back to padding.
fn lossy_native_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    // The mi grid is 8-px aligned, so mc/mr are always even; the right/bottom SB has
    // (m mod 16) mi in frame. Supported partial-edge residues:
    //   0,10,12,14  → whole 64X64 leaves (m%16==0, or >8 so the implied split never
    //                 triggers; ≥9 mi in frame, coded with edge-clamped TUs);
    //   6,8         → 32-family force-split leaves (32X64 / 64X32 / 32X32 corner);
    //   4           → 16-tap family: 16X64 (right) / 64X16 (bottom) single edges, and
    //                 the 16X16 corner when BOTH dims are residue 4 (DC-only luma).
    // A residue-4 edge combined with a residue-{6,8} edge would need a 16X32 / 32X16
    // corner that is not built yet, so those fall back to padding+clap. Residue 2
    // (8px edge) also still falls back.
    let ok = |m: i64| m % 16 == 0 || m % 16 >= 6 || m % 16 == 4 || m % 16 == 2;
    if !(ok(mc) && ok(mr)) {
        return None;
    }
    // residue-4 in one dim is supported when the perpendicular dim is a whole SB
    // (residue 0) or also residue 4 (→ 16X16 corner). residue-2 (8-tap) is supported
    // only as a single edge against a whole-SB perpendicular (→ 8X64 / 64X8); its
    // corners (8X8 / 8X16 / …) are not built yet, so any residue-2 paired with a
    // partial perpendicular falls back to padding.
    let perp_ok4 = |a: i64, b: i64| a % 16 != 4 || b % 16 == 0 || b % 16 == 4;
    let perp_ok2 = |a: i64, b: i64| a % 16 != 2 || b % 16 == 0;
    if !(perp_ok4(mc, mr) && perp_ok4(mr, mc) && perp_ok2(mc, mr) && perp_ok2(mr, mc)) {
        return None;
    }
    Some((mc, mr))
}

/// True when the size needs a force-split partition walk (any edge residue in
/// {6,8}); residues {0,10,12,14} tile into whole 64X64 leaves and use the fast path.
fn lossy_needs_partition(width: usize, height: usize) -> bool {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    let part = |m: i64| m % 16 == 6 || m % 16 == 8 || m % 16 == 4 || m % 16 == 2;
    part(mc) || part(mr)
}

fn native_420_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    let ok = |m: i64| m % 16 == 0 || m % 16 >= 6 || m % 16 == 4;
    if !(ok(mc) && ok(mr)) {
        return None;
    }
    // residue-4 is only supported as a single edge against a whole-SB perpendicular.
    let perp_ok4 = |a: i64, b: i64| a % 16 != 4 || b % 16 == 0;
    if !(perp_ok4(mc, mr) && perp_ok4(mr, mc)) {
        return None;
    }
    Some((mc, mr))
}

fn native_422_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    lossy_native_mi(width, height)
}

/// Quantizer context threaded through a partition pass: the q-context index, the
/// DC-neutral reconstruction offset, and the integer quant step. Bundling these three
/// (which always travel together) trims three positional args off every encode call.
#[derive(Clone, Copy)]
struct QuantCtx {
    qc: usize,
    neutral: f32,
    qstep: i32,
}

/// Immutable per-pass geometry + quant parameters for a native edge-partition walk.
/// `chroma_stride` is unused by the luma-only (4:0:0) pass.
#[derive(Clone, Copy)]
struct PartitionPass {
    luma_stride: usize,
    chroma_stride: usize,
    width: usize,
    height: usize,
    sb_rows: usize,
    sb_cols: usize,
    tmc: i64,
    tmr: i64,
    quant: QuantCtx,
}

/// Luma reconstruction (`rec`, written) and source (`src`, read) plane refs.
struct LumaPlanes<'a> {
    rec: &'a mut [f32],
    src: &'a [f32],
}

/// Chroma reconstruction + source plane refs for a 4:2:0 partition pass.
struct ChromaPlaneRefs<'a> {
    rec_u: &'a mut [f32],
    rec_v: &'a mut [f32],
    src_u: &'a [f32],
    src_v: &'a [f32],
}

/// The luma above/left neighbor context buffers a partition pass mutates as it walks
/// superblocks (DC-sign / txb-skip contexts plus the partition-context rows).
struct PartitionNeighbors<'a> {
    above: &'a mut [u8],
    left: &'a mut [u8],
    above_pctx: &'a mut [u8],
    left_pctx: &'a mut [u8],
}

/// The per-plane chroma presence-flag buffers a 4:2:0 partition pass mutates per leaf.
struct ChromaNeighborBufs<'a> {
    u_above: &'a mut [i32],
    v_above: &'a mut [i32],
    u_left: &'a mut [i32],
    v_left: &'a mut [i32],
}

impl Av2Encoder {
    /// Build an 8-bit encoder for `base_q_idx`. Honors the `BASES` env override for
    /// the source basis file, otherwise uses the embedded q120 set, then rescales.
    pub fn new(base_q_idx: u8) -> Self {
        Self::with_bit_depth(base_q_idx, 8)
    }

    /// Build an encoder for `base_q_idx` at a given coded bit depth (8, 10 or 12).
    /// The avm quantiser step is bit-depth-independent, so only the sample range,
    /// reconstruction clamp, DC-prediction neutral and the sequence-header signalling
    /// differ; the bases are unchanged.
    pub fn with_bit_depth(base_q_idx: u8, bit_depth: u8) -> Self {
        assert!(
            matches!(bit_depth, 8 | 10 | 12),
            "bit_depth must be 8, 10 or 12, got {bit_depth}"
        );
        let mut bases = match std::env::var("BASES") {
            Ok(p) => proj::load_bases(&p),
            Err(_) => proj::default_bases(),
        }
        .rescaled_to_q(base_q_idx as u32);
        bases.set_bit_depth(bit_depth);
        Av2Encoder {
            bases,
            base_q_idx,
            bit_depth,
        }
    }

    /// The quality this encoder is configured for.
    pub fn base_q_idx(&self) -> u8 {
        self.base_q_idx
    }

    fn config(&self, layout: Layout) -> Config {
        Config {
            layout,
            base_q: self.base_q_idx as u32,
            deblock: false,
            delta_q: 0,
            tx_switchable: true,
            guided_deblock: None,
            bit_depth: self.bit_depth,
            lossless: self.base_q_idx == 0,
        }
    }

    /// DC-prediction neutral value for the first block (1 << (bit_depth-1)).
    fn dc_neutral(&self) -> f32 {
        (1u32 << (self.bit_depth - 1)) as f32
    }

    /// Resolve a caller-supplied thread budget: `0` = use all available cores,
    /// `1` = serial, `N` = up to N threads. Replaces the old `SLIMAV_THREADS` env.
    fn resolve_threads(threads: usize) -> usize {
        if threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            threads
        }
    }

    // Per-format encode entry points (encode_yuv444/420/422/400, their lossless
    // variants, and encode_image_*) live in the `encode444/420/422/400` modules,
    // each adding methods to this same `impl Av2Encoder`.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        enc: RangeEncoder,
        config: &Config,
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        color: &ColorEncoding,
    ) -> Av2Frame {
        let tile = enc.finish();
        // AV2 derives its mode-info grid by rounding the frame to 4px
        // (ALIGN_POWER_OF_TWO(dim, MI_SIZE_LOG2)); superblocks are 64px (16 mi).
        // A square superblock at the right/bottom edge is force-split (no bits read)
        // only when *less than half* of it (<=32px, i.e. <=8 mi) is in-frame — see
        // is_partition_implied_at_boundary. When >32px is in-frame, every SB stays
        // PARTITION_NONE exactly as in the padded encode, so we can signal the real
        // size and let the decoder crop: the coded tile is byte-identical.
        // mi grid is 8px-aligned (avm dec_set_mb_mi); superblocks are 64px (16 mi).
        let mi_cols = ((width + 7) & !7) / 4;
        let mi_rows = ((height + 7) & !7) / 4;
        const MIB: usize = 16; // 64px superblock in 4px mode-info units
        // Lossless now codes every boundary geometry via the recursive forced-split
        // partition coder, so it always signals the real size (decoder crops to W x H).
        // Lossy doesn't clip its tx blocks at boundaries, so it pads unless SB-aligned.
        let aligned = mi_cols.is_multiple_of(MIB) && mi_rows.is_multiple_of(MIB);
        // Boundary-safe lossy 4:4:4 can also signal real W×H natively (the partial-edge
        // superblock decodes correctly with the edge-clamped entropy contexts).
        let lossy_native = !config.lossless
            && ((config.layout == Layout::I444 && lossy_native_mi(width, height).is_some())
                || (config.layout == Layout::I422 && native_422_mi(width, height).is_some())
                || (config.layout == Layout::I420 && native_420_mi(width, height).is_some())
                || (config.layout == Layout::Monochrome
                    && lossy_native_mi(width, height).is_some()));
        let exact = config.lossless || aligned || lossy_native;
        // Signaled dimensions: real size when boundary-safe, else the padded size.
        let (sw, sh) = if exact { (width, height) } else { (pw, ph) };
        let mut frame = frame_header(config, sw as u32, sh as u32);
        frame.extend(&tile);
        let mut data = vec![];
        data.extend(obu(2, &[]));
        data.extend(obu(1, &sequence_header(config, sw as u32, sh as u32)));
        data.extend(obu(4, &frame));
        Av2Frame {
            data,
            width,
            height,
            // Coded size = the OBU-signaled size (decoder output). The muxer crops to
            // width/height via `clap` when this is larger (padded lossy).
            coded_width: sw,
            coded_height: sh,
            // Coded bit depth signaled in the sequence header (8/10/12). av2C/pixi in
            // the AVIF muxer must use this.
            bit_depth: self.bit_depth,
            color: *color,
            chroma_format: match config.layout {
                Layout::Monochrome => ChromaFormat::Monochrome,
                Layout::I420 => ChromaFormat::Yuv420,
                Layout::I422 => ChromaFormat::Yuv422,
                Layout::I444 => ChromaFormat::Yuv444,
            },
        }
    }

    /// Finish wrapping a color AV1 OBU stream in an AVIF container.
    pub fn wrap_avif(
        frame: &Av2Frame,
        icc_profile: Option<Vec<u8>>,
        exif: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, EncodeError> {
        let format = Av2Format {
            bit_depth: frame.bit_depth,
            monochrome: frame.chroma_format == ChromaFormat::Monochrome,
            chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                || frame.chroma_format == ChromaFormat::Yuv420,
            chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
        };
        if let (Some(exif), Some(icc_profile)) = (exif, icc_profile.as_ref()) {
            return Ok(avif::to_avif_full(
                frame,
                &format,
                Some(icc_profile),
                Some(&exif),
            ));
        }
        if let Some(icc_profile) = icc_profile.as_ref() {
            return Ok(avif::to_avif_cicp_icc(frame, &format, icc_profile.to_vec()));
        }
        Ok(avif::to_avif(frame, &format))
    }

    /// Wrap a color frame together with a monochrome alpha auxiliary item into an
    /// AVIF (alpha = an `encode_yuv400` result, typically of the alpha plane). The
    /// alpha item is linked via `auxl` and tagged with the standard alpha `auxC` URN.
    pub fn wrap_avif_alpha(
        frame: &Av2Frame,
        alpha: &Av2Frame,
        icc_profile: Option<Vec<u8>>,
        exif: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, EncodeError> {
        let format = Av2Format {
            bit_depth: frame.bit_depth,
            monochrome: frame.chroma_format == ChromaFormat::Monochrome,
            chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                || frame.chroma_format == ChromaFormat::Yuv420,
            chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
        };
        let color = match icc_profile {
            Some(icc) => Av2Color::Both {
                cicp: frame.color,
                icc,
            },
            None => Av2Color::Cicp(frame.color),
        };
        Ok(avif::to_avif_color_alpha(
            frame,
            alpha,
            &format,
            &color,
            exif.as_deref(),
        ))
    }
}

/// Maps CLI quality 1–100 to AV2 `base_q_idx` 1–254.
/// quality 100 → q≈3 (near-lossless), quality 60 → q≈100, quality 1 → q=254.
pub fn av2_map_quality(quality: u8) -> u8 {
    debug_assert!(quality >= 1 && quality <= 100);
    ((100 - quality as u32) * 254 / 99).clamp(1, 254) as u8
}