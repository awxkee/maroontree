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
use crate::av2::cdf_state::{CdfState, update_cdf};

pub(crate) static MIN_PROB: [[u16; 8]; 8] = [
    [63, 65535, 65535, 65535, 65535, 65535, 65535, 65535],
    [47, 87, 65535, 65535, 65535, 65535, 65535, 65535],
    [31, 63, 95, 65535, 65535, 65535, 65535, 65535],
    [31, 55, 79, 103, 65535, 65535, 65535, 65535],
    [23, 47, 63, 87, 111, 65535, 65535, 65535],
    [23, 39, 55, 79, 95, 111, 65535, 65535],
    [15, 31, 47, 63, 79, 95, 111, 65535],
    // nsyms = 8 (CfL joint-sign / alpha-magnitude). The avm decoder scales with
    // av2_prob_inc_tbl[nsym-2] = row 6 for nsyms=8. Matching boundary(k) gives
    // MIN_PROB[7][k] = 127 - 8*av2_prob_inc_tbl[6][k] (the low-7-bit |127 vs >>7<<4
    // terms cancel exactly), inc_tbl[6] = {14,12,10,8,6,4,2,0} ->
    // {15,31,47,63,79,95,111, sentinel}. Verified bit-exact vs avm od_ec_prob_scale.
    [15, 31, 47, 63, 79, 95, 111, 65535],
];

/// MSB-first bit packer for the uncompressed header sections.
pub(crate) struct ByteWriter {
    bytes: Vec<u8>,
    accumulator: u64,
    pending_bits: u32,
}

impl ByteWriter {
    pub(crate) fn new() -> Self {
        ByteWriter {
            bytes: vec![],
            accumulator: 0,
            pending_bits: 0,
        }
    }

    /// Append a single bit (only bit 0 of `bit` is used).
    pub(crate) fn write_bit(&mut self, bit: u32) {
        self.accumulator = (self.accumulator << 1) | (bit as u64 & 1);
        self.pending_bits += 1;
        if self.pending_bits == 8 {
            self.bytes.push(self.accumulator as u8);
            self.accumulator = 0;
            self.pending_bits = 0;
        }
    }

    /// Append the low `count` bits of `value`, most-significant bit first.
    pub(crate) fn write_bits(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Append an unsigned variable-length code (`uvlc`).
    pub(crate) fn write_uvlc(&mut self, value: u32) {
        let shifted = value + 1;
        let leading_zeros = 31 - shifted.leading_zeros();
        for _ in 0..leading_zeros {
            self.write_bit(0);
        }
        self.write_bits(shifted, leading_zeros + 1);
    }

    /// Append `value` coded as a non-uniform `ns(max)` element (uniform code).
    pub(crate) fn write_uniform(&mut self, value: u32, max: u32) {
        let bits = (31 - max.leading_zeros()) + 1;
        let threshold = (1u32 << bits) - max;
        if value < threshold {
            self.write_bits(value, bits - 1);
        } else {
            let widened = value + threshold;
            self.write_bits(widened >> 1, bits - 1);
            self.write_bit(widened & 1);
        }
    }

    /// Write a trailing `1` bit and pad with zeros to the next byte boundary.
    pub(crate) fn align_with_one(&mut self) {
        self.write_bit(1);
        while self.pending_bits != 0 {
            self.write_bit(0);
        }
    }

    /// Pad with zeros to the next byte boundary.
    pub(crate) fn align_with_zero(&mut self) {
        while self.pending_bits != 0 {
            self.write_bit(0);
        }
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// AV2 multi-symbol arithmetic coder (encoder side).
pub(crate) struct RangeEncoder {
    low: u64,
    /// Current range; exposed for encoder/decoder trace alignment during testing.
    pub(crate) range: u32,
    count: i32,
    output: Vec<u16>,
    /// SB-relative (16x16) luma-mi coded mask, mirroring the decoder's
    /// `xd->is_mi_coded`. The interior-split leaf loop resets it per SB and marks
    /// each leaf's mi as it is coded; MHCCP reads it to resolve top-right /
    /// bottom-left reference availability in the true (VERT-then-HORZ) coding
    /// order. Left all-zero on paths that do not maintain it (whole-SB / edge
    /// walks), where MHCCP's prev-SB-row/col special cases suffice.
    pub(crate) sb_coded: [u8; 256],
    /// Coefficient-CDF q-context = get_q_ctx(base_q_idx). Selects the default
    /// CDF band avmdec loads (0:q<=90, 1:91..140, 2:141..190, 3:>=191).
    /// Defaults to 1 so legacy q120 paths are unchanged.
    pub(crate) qc: usize,
    /// Per-superblock adaptive-quantization (delta-Q).
    pub(crate) delta_q_present: bool,
    pub(crate) delta_q_signaled: i32,
    /// Set by the caller just before emitting the SB's first leaf; the mode
    /// emitter consumes it (emits the delta-Q symbol after the partition bit and
    /// clears the flag), so delta-Q is coded exactly once per SB regardless of
    /// how the SB partitions.
    pub(crate) delta_q_pending: bool,
    /// CCSO per-superblock flag state. `ccso_u_enable` mirrors the frame header's
    /// U-plane CCSO enable; when set the mode emitter writes one `blk_idc` symbol
    /// per SB (after the partition bit, before delta-Q), using the neighbor-based
    /// context. Phase 1 always filters (blk_idc = 1). `ccso_pending` is armed by the
    /// caller per SB and consumed exactly once. `ccso_grid`/`ccso_cols` track the
    /// per-SB decisions for the above/left context lookup; `ccso_sb_rc` is the
    /// current SB's (row, col).
    pub(crate) ccso_u_enable: bool,
    pub(crate) ccso_v_enable: bool,
    pub(crate) ccso_pending: bool,
    pub(crate) ccso_grid: Vec<u8>,
    pub(crate) ccso_grid_v: Vec<u8>,
    /// Final recon planes (Y,U,V as f32) captured for the video DPB; empty for still.
    pub(crate) recon: Vec<Vec<f32>>,
    pub(crate) ccso_cols: usize,
    pub(crate) ccso_sb_rc: (usize, usize),
    /// Derived CCSO U-plane band offsets (raw `ccso_offset` values) + params,
    /// produced by the post-reconstruction search and consumed when the frame
    /// header is built. `None` when CCSO is off.
    pub(crate) ccso_u_result: Option<crate::av2::ccso::PlaneResult>,
    pub(crate) ccso_v_result: Option<crate::av2::ccso::PlaneResult>,
    /// Frame-level CDEF strength `(y_str, uv_str, damping)` chosen by the
    /// post-reconstruction search, or `None` when no strength beat "off".
    /// Authoritative over the header default when the search ran.
    pub(crate) cdef_result: Option<(u8, u8, u8)>,
    /// Per-block CDEF emission state (pass 2). `cdef_nb` is nb_cdef_strengths;
    /// per-SB emission is active only when it is >= 2. `cdef_grid[r*cols+c]` is the
    /// chosen strength index for the CDEF unit (SB) at (row, col): 0 = off, 1 = the
    /// active strength. `cdef_pending`/`cdef_sb_rc` mirror the CCSO deferral so the
    /// index0 symbol is emitted once per SB at its first coded block.
    pub(crate) cdef_pending: bool,
    pub(crate) cdef_sb_rc: (usize, usize),
    pub(crate) cdef_cols: usize,
    pub(crate) cdef_grid: Vec<u8>,
    pub(crate) cdef_nb: usize,
    /// Pass-1 CDEF decision (strengths + per-SB grid) derived from the
    /// reconstruction, handed to the pass-2 emit. `None` when CDEF stays off.
    pub(crate) cdef_decided: Option<crate::av2::cdef_est::CdefDecision>,
    /// MHCCP fit cached by the chroma-mode *preset* pass (which decides + emits the
    /// mode flag before the luma residual) so the chroma *leaf* encode can reuse the
    /// identical predictor instead of re-running the 3-direction filter fit. Keyed
    /// by leaf `(y, x, w, h)`; taken (one-shot) by the matching leaf, else the leaf
    /// re-fits. `preset` and `leaf` feed the fit identical inputs, so reusing is
    /// bit-exact — and it removes the fragile "both searches agree" dependency.
    pub(crate) mhccp_cache: Option<(
        usize,
        usize,
        usize,
        usize,
        Option<crate::av2::cfl::CflChoice>,
    )>,
    /// Decision-pass outputs (Phase 3): the chosen edge filter + per-SB grid for
    /// each plane, and the SB column count, handed to the emit pass.
    pub(crate) ccso_decided_u: Option<(crate::av2::ccso::CcsoEdgeResult, Vec<u8>)>,
    pub(crate) ccso_decided_v: Option<(crate::av2::ccso::CcsoEdgeResult, Vec<u8>)>,
    pub(crate) ccso_sb_cols_out: usize,
    /// Emit CfL (chroma-from-luma) signalling for chroma-ref blocks. Set per encode
    /// from the tuning flag; false keeps the bitstream byte-identical.
    pub(crate) cfl: bool,
    /// Per-block CfL state, set just before the block's mode encode. `cfl_ctx` is the
    /// is_cfl neighbor context (0..2). `cfl_use` selects CfL (uv_mode = UV_CFL_PRED);
    /// when true, `cfl_js`/`cfl_mag_u`/`cfl_mag_v` + `cfl_ctx_u`/`cfl_ctx_v` carry the
    /// resolved joint-sign, per-plane magnitude indices and alpha-cdf contexts.
    pub(crate) cfl_ctx: usize,
    pub(crate) cfl_use: bool,
    /// y-mode index context for the next intra luma mode emit: the decoder's
    /// get_y_mode_idx_ctx = count of directional above-right/bottom-left
    /// neighbor modes (0..=2). Set per block by the encoders.
    pub(crate) y_ctx: usize,
    pub(crate) cfl_signaled: bool,
    pub(crate) cfl_js: u8,
    pub(crate) cfl_mag_u: u8,
    pub(crate) cfl_mag_v: u8,
    pub(crate) cfl_ctx_u: usize,
    pub(crate) cfl_ctx_v: usize,
    /// MHCCP block state.
    pub(crate) mhccp: bool,
    /// Chroma subsampling of the current stream, for the per-block
    pub(crate) mhccp_ssx: bool,
    /// Current coded block dimensions in 4-sample units
    pub(crate) cur_bw4: usize,
    pub(crate) cur_bh4: usize,
    pub(crate) mhccp_ssy: bool,
    /// Per-block: does this block satisfy avm `is_mhccp_allowed` (block size in
    /// [8x8, 64x64], not 4x4, within max UV tx)?
    pub(crate) mhccp_allowed: bool,
    pub(crate) mhccp_use: bool,
    /// True while coding leaves of an interior chroma-motivated square split. MHCCP in
    /// these 32x32 leaves is not yet bit-exact against the reference decoder (the implicit
    /// luma-neighbor fetch differs for an interior split node), so it is suppressed there;
    /// the split's quality gain comes from the 32x32 transform, not from MHCCP.
    pub(crate) in_interior_split: bool,
    pub(crate) mhccp_dir: u8,
    pub(crate) mhccp_size_group: u8,
    /// Chroma intra prediction mode for the current block, in the internal
    /// numbering used by the luma dispatch: 0 = DC (default), 1 = SMOOTH,
    /// 2 = SMOOTH_V, 3 = SMOOTH_H, 4 = PAETH.
    pub(crate) uv_mode: usize,
    /// Per-tile mutable CDF working copies.
    pub(crate) cdf_state: Option<Box<CdfState>>,
    /// Inter tile: emit intra_inter=0 before each block's mode-info. ctx set per block.
    pub(crate) inter_tile: bool,
    pub(crate) inter_txb: bool, // route TX32 txb_skip to inter plane
    pub(crate) intra_inter_ctx: usize,
    /// Reference count listed in this frame's header. When 2, every inter block
    /// codes one single_ref bit (rank via `ref_rank`, context via `ref_bit_ctx`).
    pub(crate) num_refs: u8,
    /// av2_get_ref_pred_context result for the current block (set per block).
    pub(crate) ref_bit_ctx: usize,
    /// Reference rank (0 or 1) the current inter block predicts from.
    pub(crate) ref_rank: usize,
    /// Discard all output: `normalize`/`normalize_bypass` become no-ops. Used by the
    /// SB-wavefront decide (Capture) pass, whose bytes are thrown away — only the
    /// serial Replay emits the real bitstream. Skips range coding + `output` growth.
    pub(crate) sink: bool,
}

impl RangeEncoder {
    pub(crate) fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0x8000,
            count: -9,
            output: vec![],
            sb_coded: [0u8; 256],
            qc: 1,
            delta_q_present: false,
            delta_q_signaled: 0,
            delta_q_pending: false,
            ccso_u_enable: false,
            ccso_v_enable: false,
            ccso_pending: false,
            ccso_grid: Vec::new(),
            recon: Vec::new(),
            ccso_grid_v: Vec::new(),
            ccso_cols: 0,
            ccso_sb_rc: (0, 0),
            ccso_u_result: None,
            ccso_v_result: None,
            cdef_result: None,
            cdef_pending: false,
            cdef_sb_rc: (0, 0),
            cdef_cols: 0,
            cdef_grid: Vec::new(),
            cdef_nb: 1,
            cdef_decided: None,
            mhccp_cache: None,
            ccso_decided_u: None,
            ccso_decided_v: None,
            ccso_sb_cols_out: 0,
            cfl: false,
            cfl_ctx: 0,
            cfl_use: false,
            y_ctx: 0,
            cfl_signaled: false,
            cfl_js: 0,
            cfl_mag_u: 0,
            cfl_mag_v: 0,
            cfl_ctx_u: 0,
            cfl_ctx_v: 0,
            mhccp: false,
            mhccp_ssx: false,
            mhccp_ssy: false,
            cur_bw4: 16,
            cur_bh4: 16,
            mhccp_allowed: false,
            mhccp_use: false,
            in_interior_split: false,
            mhccp_dir: 0,
            mhccp_size_group: 0,
            uv_mode: 0,
            cdf_state: None,
            inter_tile: false,
            inter_txb: false,
            intra_inter_ctx: 0,
            num_refs: 1,
            ref_bit_ctx: 0,
            ref_rank: 0,
            sink: false,
        }
    }

    /// Enable adaptive CDF updating for this tile.  Must be called before
    /// any `encode_symbol_mut` calls.  `qc` is the q-context index (0-3).
    pub(crate) fn enable_adaptive_cdf(&mut self, qc: usize) {
        self.cdf_state = Some(Box::new(CdfState::new(qc)));
    }

    /// Non-mutating ideal arithmetic-code length for one symbol in an inverse
    /// CDF. `nsyms` excludes the adaptation metadata stored after the CDF
    /// boundaries.
    #[inline]
    fn symbol_bits(cdf: &[u16], symbol: usize, nsyms: usize) -> f32 {
        debug_assert!(symbol < nsyms);
        let hi = if symbol == 0 {
            32768i32
        } else {
            cdf[symbol - 1] as i32
        };
        let lo = if symbol + 1 < nsyms {
            cdf[symbol] as i32
        } else {
            0
        };
        let probability = (hi - lo).max(1) as f32 / 32768.0;
        -probability.log2()
    }

    /// Estimate the syntax rate from the tile's current adaptive CDFs without
    /// advancing either the range coder or the adaptation state. The constants
    /// are neutral fallbacks for non-adaptive callers; video tiles use the live
    /// CDF state.
    pub(crate) fn estimate_intra_inter_bits(&self, val: usize) -> f32 {
        if !self.inter_tile {
            return 0.0;
        }
        self.cdf_state.as_ref().map_or(1.0, |cs| {
            Self::symbol_bits(&cs.intra_inter[self.intra_inter_ctx], val, 2)
        })
    }

    pub(crate) fn estimate_skip_txfm_bits(&self, ctx: usize, val: usize) -> f32 {
        self.cdf_state
            .as_ref()
            .map_or(1.0, |cs| Self::symbol_bits(&cs.skip_txfm_blk[ctx], val, 2))
    }

    pub(crate) fn estimate_inter_mode_bits(&self, ctx: usize, mode: usize) -> f32 {
        self.cdf_state.as_ref().map_or(3.0f32.log2(), |cs| {
            Self::symbol_bits(&cs.inter_single_mode[ctx], mode, 3)
        })
    }

    pub(crate) fn estimate_drl_bits(&self, ctx: usize, bit: usize) -> f32 {
        self.cdf_state
            .as_ref()
            .map_or(1.0, |cs| Self::symbol_bits(&cs.drl[ctx], bit, 2))
    }

    pub(crate) fn estimate_mvd_shell_set_bits(&self, bit: usize) -> f32 {
        self.cdf_state
            .as_ref()
            .map_or(1.0, |cs| Self::symbol_bits(&cs.mvd_shell_set, bit, 2))
    }

    pub(crate) fn estimate_mvd_shell_class_bits(&self, set: usize, class: usize) -> f32 {
        self.cdf_state.as_ref().map_or(3.0, |cs| {
            let cdf = if set == 0 {
                &cs.mvd_shell_class0
            } else {
                &cs.mvd_shell_class1
            };
            Self::symbol_bits(cdf, class, 8)
        })
    }

    pub(crate) fn estimate_mvd_offset_low_bits(&self, ctx: usize, bit: usize) -> f32 {
        self.cdf_state
            .as_ref()
            .map_or(1.0, |cs| Self::symbol_bits(&cs.mvd_offset_low[ctx], bit, 2))
    }

    pub(crate) fn estimate_mvd_offset_class2_bits(&self, bit: usize) -> f32 {
        self.cdf_state
            .as_ref()
            .map_or(1.0, |cs| Self::symbol_bits(&cs.mvd_offset_class2, bit, 2))
    }

    pub(crate) fn estimate_mvd_offset_other_bits(&self, idx: usize, bit: usize) -> f32 {
        self.cdf_state.as_ref().map_or(1.0, |cs| {
            Self::symbol_bits(&cs.mvd_offset_other[idx], bit, 2)
        })
    }

    pub(crate) fn estimate_mvd_col_greater_bits(&self, ctx: usize, bit: usize) -> f32 {
        self.cdf_state.as_ref().map_or(1.0, |cs| {
            Self::symbol_bits(&cs.mvd_col_greater[ctx], bit, 2)
        })
    }

    pub(crate) fn estimate_mvd_col_index_bits(&self, ctx: usize, bit: usize) -> f32 {
        self.cdf_state
            .as_ref()
            .map_or(1.0, |cs| Self::symbol_bits(&cs.mvd_col_index[ctx], bit, 2))
    }

    fn normalize(&mut self, mut low: u64, range: u32) {
        if self.sink {
            return;
        }
        let base_count = self.count;
        let shift = 16 - (32 - range.leading_zeros()) as i32;
        let mut remaining = base_count + shift;
        if remaining >= 0 {
            let mut bit_pos = base_count + 16;
            let mut mask: u64 = (1u64 << bit_pos) - 1;
            if remaining >= 8 {
                self.output.push((low >> bit_pos) as u16);
                low &= mask;
                bit_pos -= 8;
                mask >>= 8;
            }
            self.output.push((low >> bit_pos) as u16);
            remaining = bit_pos + shift - 24;
            low &= mask;
        }
        self.low = low << shift;
        self.range = range << shift;
        self.count = remaining;
    }

    /// Compute the cumulative-frequency boundary for symbol index `k`.
    fn boundary(scaled_range: u32, icdf: &[u16], min_prob: &[u16; 8], k: usize) -> u32 {
        ((scaled_range * (icdf[k] as u32 | 127).saturating_sub(min_prob[k] as u32)) >> 10) << 3
    }

    /// Encode symbol `s` against an inverse-CDF table covering `nsyms` symbols.
    /// Emit intra_inter=0 (intra) for inter tiles, then adapt CDF. No-op otherwise.
    pub(crate) fn emit_intra_inter(&mut self) {
        self.emit_intra_inter_val(0);
    }

    /// Emit intra_inter flag: 0=intra, 1=inter. No-op outside inter tiles.
    pub(crate) fn emit_intra_inter_val(&mut self, val: usize) {
        if !self.inter_tile {
            return;
        }
        let ctx = self.intra_inter_ctx;
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.intra_inter[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                val,
                1,
            );
        }
    }

    /// Emit a single-ref inter mode symbol (NEARMV=0, GLOBALMV=1, NEWMV=2).
    pub(crate) fn emit_inter_single_mode(&mut self, ctx: usize, mode: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.inter_single_mode[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                mode,
                2, // nsyms_mt = 3-symbol alphabet
            );
        }
    }

    /// Emit DRL bit (drl_cdf[0][ctx], idx 0). bit=0 selects candidate 0.
    pub(crate) fn emit_drl(&mut self, ctx: usize, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.drl[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit,
                1,
            );
        }
    }

    /// Emit the current block's single-ref rank bit when the frame lists two
    /// references (AVM read_single_ref, n_refs=2): bit=1 selects rank 0.
    /// No-op for single-reference frames.
    pub(crate) fn emit_single_ref_rank(&mut self) {
        if self.num_refs < 2 {
            return;
        }
        let bit = usize::from(self.ref_rank == 0);
        let ctx = self.ref_bit_ctx;
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.single_ref[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit,
                1,
            );
        }
    }

    /// Rate of the single-ref rank bit for `rank` under the current block's
    /// context; zero when the frame lists a single reference.
    pub(crate) fn estimate_single_ref_bits(&self, rank: usize) -> f32 {
        if self.num_refs < 2 {
            return 0.0;
        }
        let bit = usize::from(rank == 0);
        self.cdf_state.as_ref().map_or(1.0, |cs| {
            Self::symbol_bits(&cs.single_ref[self.ref_bit_ctx], bit, 2)
        })
    }

    // ---- adaptive MVD (QTR_PEL) — mirror AVM read_mv CDFs -------------------
    /// True if adaptive cdf_state is active (MVD must use adaptive path).
    pub(crate) fn mvd_adaptive(&self) -> bool {
        self.cdf_state.is_some()
    }
    pub(crate) fn mvd_shell_set(&mut self, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_shell_set,
                bit,
                1,
            );
        }
    }
    pub(crate) fn mvd_shell_class0(&mut self, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_shell_class0,
                s,
                7,
            );
        }
    }
    pub(crate) fn mvd_shell_class1(&mut self, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_shell_class1,
                s,
                7,
            );
        }
    }
    pub(crate) fn mvd_offset_low(&mut self, ctx: usize, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_offset_low[ctx],
                bit,
                1,
            );
        }
    }
    pub(crate) fn mvd_offset_class2(&mut self, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_offset_class2,
                bit,
                1,
            );
        }
    }
    pub(crate) fn mvd_offset_other(&mut self, bit_idx: usize, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_offset_other[bit_idx],
                bit,
                1,
            );
        }
    }
    pub(crate) fn mvd_col_greater(&mut self, ctx: usize, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_col_greater[ctx],
                bit,
                1,
            );
        }
    }
    pub(crate) fn mvd_col_index(&mut self, ctx: usize, bit: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.mvd_col_index[ctx],
                bit,
                1,
            );
        }
    }

    /// Emit block-level skip_txfm (inter blocks); ctx 0..5.
    pub(crate) fn emit_skip_txfm(&mut self, ctx: usize, val: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.skip_txfm_blk[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                val,
                1,
            );
        }
    }

    /// Emit inter tx_type (set 3, flat 2-way). idx 0=IDTX, 1=DCT_DCT. eob_ctx 0-2.
    pub(crate) fn emit_inter_tx_type(&mut self, eob_ctx: usize, idx: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.inter_ext_tx3[eob_ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                idx,
                1,
            );
        }
    }

    /// Emit DCT_DCT for a TX16 inter transform. AV2 signals this through the
    /// eset=2 split alphabet: bank 0 followed by symbol 3 in the 8-way bank.
    pub(crate) fn emit_inter_tx16_dct(&mut self, eob_ctx: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.inter_tx16_set[eob_ctx],
                0,
                1,
            );
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.inter_tx16_idx[eob_ctx],
                3,
                7,
            );
        } else {
            const SET: [u32; 3] = [11933, 2048, 3911];
            const IDX: [[u16; 7]; 3] = [
                [31628, 31043, 30444, 18115, 13696, 9150, 4659],
                [32710, 32507, 32181, 451, 212, 142, 60],
                [15364, 15099, 14365, 8716, 6375, 4262, 2092],
            ];
            self.encode_bool(SET[eob_ctx], 0);
            self.sym_static(&IDX[eob_ctx], 3, 7);
        }
    }

    /// Emit inter 64x64 tx do_partition (0=none, 1=split).
    pub(crate) fn emit_tx_do_partition(&mut self, val: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_do_partition;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                val,
                1,
            );
        }
    }

    /// Emit inter 64x64 4-way tx partition type (SPLIT=0), 7-sym alphabet.
    pub(crate) fn emit_tx_part_type(&mut self, ptype: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_part_type;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                ptype,
                6, // nsyms_mt for 7-sym alphabet
            );
        }
    }

    pub(crate) fn encode_symbol(&mut self, icdf: &[u16], s: usize, nsyms: usize) {
        if self.sink {
            return;
        }
        let range = self.range;
        let scaled_range = range >> 8;
        let min_prob = &MIN_PROB[nsyms - 1];
        let upper = if s == 0 {
            range
        } else {
            Self::boundary(scaled_range, icdf, min_prob, s - 1)
        };
        let lower = Self::boundary(scaled_range, icdf, min_prob, s);
        let low = self.low + (range - upper) as u64;
        self.normalize(low, upper - lower);
    }

    /// Encode an escape (last) symbol against `cdf` extended with a trailing 0,
    /// using a stack buffer instead of allocating (equivalent to with_escape()).
    pub(crate) fn encode_symbol_esc(&mut self, cdf: &[u16], s: usize, nsyms: usize) {
        if self.sink {
            return;
        }
        let mut buf = [0u16; 16];
        let n = cdf.len();
        buf[..n].copy_from_slice(cdf);
        self.encode_symbol(&buf[..n + 1], s, nsyms);
    }

    /// Encode a single adaptive boolean with CDF `cdf` (probability of `0`).
    pub(crate) fn encode_bool(&mut self, cdf: u32, bit: u32) {
        if self.sink {
            return;
        }
        let range = self.range;
        let split = (((range >> 8) * (((cdf >> 7) << 4) + 8)) >> 7) << 3;
        let (low, range) = if bit != 0 {
            (self.low + (range - split) as u64, split)
        } else {
            (self.low, range - split)
        };
        self.normalize(low, range);
    }

    fn normalize_bypass(&mut self, mut low: u64, range: u32, bypass_bits: i32) {
        if self.sink {
            return;
        }
        let base_count = self.count + bypass_bits;
        let mut remaining = base_count;
        if remaining >= 0 {
            let mut bit_pos = base_count + 16;
            let mut mask: u64 = (1u64 << bit_pos) - 1;
            if remaining >= 8 {
                self.output.push((low >> bit_pos) as u16);
                low &= mask;
                bit_pos -= 8;
                mask >>= 8;
            }
            self.output.push((low >> bit_pos) as u16);
            remaining = bit_pos - 24;
            low &= mask;
        }
        self.low = low;
        self.range = range;
        self.count = remaining;
    }

    /// Encode `bit_count` equiprobable bits taken from `value`, MSB-first.
    pub(crate) fn encode_bypass(&mut self, value: u32, bit_count: u32) {
        let range = self.range;
        let low = (self.low << bit_count) + (range as u64) * (value as u64);
        self.normalize_bypass(low, range, bit_count as i32);
    }

    /// Encode AVM's `read_uniform(n)` syntax through the arithmetic bypass path.
    pub(crate) fn encode_uniform(&mut self, value: u32, n: u32) {
        debug_assert!(n > 1 && value < n);
        let l = 32 - n.leading_zeros();
        let m = (1u32 << l) - n;
        if value < m {
            self.encode_bypass(value, l - 1);
        } else {
            let v = value + m;
            self.encode_bypass(v >> 1, l - 1);
            self.encode_bypass(v & 1, 1);
        }
    }

    /// Static palette syntax is used only by coded-lossless frames, for which
    /// AVM forces `disable_cdf_update=1`.
    pub(crate) fn sym_palette_y_mode(&mut self, val: usize) {
        self.sym_static(&[2723], val, 1);
    }

    pub(crate) fn sym_palette_y_size(&mut self, size: usize) {
        const CDF: [u16; 6] = [23989, 17673, 11991, 7865, 4845, 2365];
        self.sym_static(&CDF, size - 2, 6);
    }

    pub(crate) fn sym_palette_identity_row_off(&mut self, first_row: bool) {
        // The first row uses context 3; subsequent non-copy rows use context 0.
        let cdf = if first_row {
            [19769, 12]
        } else {
            [10253, 7017]
        };
        self.sym_static(&cdf, 0, 2);
    }

    pub(crate) fn sym_palette_y_color(&mut self, size: usize, ctx: usize, val: usize) {
        const CDF: [[[u16; 7]; 5]; 7] = [
            [
                [4628, 0, 0, 0, 0, 0, 0],
                [16384, 0, 0, 0, 0, 0, 0],
                [24186, 0, 0, 0, 0, 0, 0],
                [5355, 0, 0, 0, 0, 0, 0],
                [2339, 0, 0, 0, 0, 0, 0],
            ],
            [
                [7418, 3742, 0, 0, 0, 0, 0],
                [21405, 7495, 0, 0, 0, 0, 0],
                [25927, 4189, 0, 0, 0, 0, 0],
                [11418, 6756, 0, 0, 0, 0, 0],
                [2195, 1122, 0, 0, 0, 0, 0],
            ],
            [
                [9062, 5806, 3708, 0, 0, 0, 0],
                [22792, 10252, 5386, 0, 0, 0, 0],
                [26077, 7308, 3534, 0, 0, 0, 0],
                [13859, 8843, 4365, 0, 0, 0, 0],
                [2460, 1692, 950, 0, 0, 0, 0],
            ],
            [
                [8652, 5811, 4282, 2827, 0, 0, 0],
                [23200, 12296, 8474, 3826, 0, 0, 0],
                [27062, 7525, 4728, 2362, 0, 0, 0],
                [12663, 9786, 5744, 3857, 0, 0, 0],
                [1871, 1426, 1002, 569, 0, 0, 0],
            ],
            [
                [11944, 8541, 6842, 5309, 3502, 0, 0],
                [24627, 13779, 11169, 6586, 4192, 0, 0],
                [27516, 8428, 6318, 4330, 2143, 0, 0],
                [13249, 10073, 7181, 5796, 4345, 0, 0],
                [2385, 1878, 1521, 1115, 618, 0, 0],
            ],
            [
                [11140, 8256, 6895, 5714, 4637, 3229, 0],
                [24740, 14504, 12155, 7344, 5656, 3862, 0],
                [26279, 10526, 8307, 6374, 4418, 2258, 0],
                [10720, 8339, 5778, 4824, 4351, 3194, 0],
                [1967, 1563, 1296, 1040, 763, 463, 0],
            ],
            [
                [10297, 7685, 6784, 5875, 5114, 4018, 2865],
                [25226, 15711, 13617, 9218, 7309, 5702, 3964],
                [25186, 12331, 10040, 8146, 6253, 4189, 2136],
                [10666, 8624, 5852, 4617, 3922, 3556, 2615],
                [2244, 1881, 1612, 1375, 1142, 857, 487],
            ],
        ];
        self.sym_static(&CDF[size - 2][ctx], val, size - 1);
    }

    /// Flush the coder and return the packed tile bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let low = self.low;
        let mut count = self.count;
        let mut remaining = 10 + count;
        let mask: u64 = 0x3FFF;
        let mut end = ((low + mask) & !mask) | (mask + 1);
        if remaining > 0 {
            let mut byte_mask: u64 = (1u64 << (count + 16)) - 1;
            loop {
                self.output.push((end >> (count + 16)) as u16);
                end &= byte_mask;
                remaining -= 8;
                count -= 8;
                byte_mask >>= 8;
                if remaining <= 0 {
                    break;
                }
            }
        }
        let len = self.output.len();
        let mut bytes = vec![0u8; len];
        let mut carry: u32 = 0;
        let mut i = len;
        while i > 0 {
            i -= 1;
            let x = self.output[i] as u32 + carry;
            bytes[i] = (x & 0xff) as u8;
            carry = x >> 8;
        }
        bytes
    }
}

impl RangeEncoder {
    #[inline]
    fn sym_static(&mut self, icdf: &[u16], s: usize, nsyms: usize) {
        if s == nsyms {
            self.encode_symbol_esc(icdf, s, nsyms);
        } else {
            self.encode_symbol(icdf, s, nsyms);
        }
    }

    #[inline]
    fn sym_mut_inner(
        low: &mut u64,
        range_ref: &mut u32,
        count_ref: &mut i32,
        output: &mut Vec<u16>,
        cdf: &mut [u16],
        s: usize,
        nsyms_mt: usize,
    ) {
        let range = *range_ref;
        let scaled_range = range >> 8;
        // Detect whether the original table already carried a trailing-0 sentinel
        // (cdf_state::expand uses the same rule). If so this is an nsyms_mt-symbol
        // no-escape alphabet (nsyms_avm = nsyms_mt); otherwise the escape adds the
        // implicit sentinel (nsyms_avm = nsyms_mt + 1).
        let has_sentinel = cdf[nsyms_mt - 1] == 0;
        let nsyms_avm = if has_sentinel { nsyms_mt } else { nsyms_mt + 1 };
        // AVM's probability scaling (`od_ec_prob_scale`) selects the increment row
        // by the *coded alphabet size* `nsym` (= nsyms_avm), via
        // `av2_prob_inc_tbl[nsym - 2]`. The MIN_PROB table mirrors that table, so it
        // must also be indexed by `nsyms_avm - 2`, NOT `nsyms_mt - 1`. These agree
        // for bools and escape-coded alphabets but differ for no-escape multi-symbol
        // alphabets (e.g. the 3-ary MHCCP `mh_dir`, which carries a trailing-0
        // sentinel so nsyms_avm = nsyms_mt = 3 -> row 1, not row 2).
        let min_prob = &MIN_PROB[nsyms_avm - 2];
        // Encode against [icdf_0 .. sentinel]; boundary(k) needs index up to nsyms_mt
        // (the sentinel) for the escape symbol of an escape-coded table.
        let icdf = &cdf[..=nsyms_mt];
        let upper = if s == 0 {
            range
        } else {
            Self::boundary(scaled_range, icdf, min_prob, s - 1)
        };
        let lower = Self::boundary(scaled_range, icdf, min_prob, s);
        // normalize inline (mirrors RangeEncoder::normalize exactly)
        let new_low = *low + (range - upper) as u64;
        let new_range = upper - lower;
        let shift = 16 - (32 - new_range.leading_zeros()) as i32;
        let mut remaining = *count_ref + shift;
        let mut lo = new_low;
        if remaining >= 0 {
            let mut bit_pos = *count_ref + 16;
            let mut mask: u64 = (1u64 << bit_pos) - 1;
            if remaining >= 8 {
                output.push((lo >> bit_pos) as u16);
                lo &= mask;
                bit_pos -= 8;
                mask >>= 8;
            }
            output.push((lo >> bit_pos) as u16);
            remaining = bit_pos + shift - 24;
            lo &= mask;
        }
        *low = lo << shift;
        *range_ref = new_range << shift;
        *count_ref = remaining;
        // Adapt with AVM nsyms.
        update_cdf(cdf, s, nsyms_avm);
    }

    // ---- luma 8x8 ---------------------------------------------------------
    pub(crate) fn sym_luma8_hf(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma8_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA8_BASE_TOK_HF_QC;
            self.sym_static(&LUMA8_BASE_TOK_HF_QC[self.qc][ctx], s, 3);
        }
    }
    pub(crate) fn sym_luma8_lf(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma8_lf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                5,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA8_BASE_TOK_LF_QC;
            self.sym_static(&LUMA8_BASE_TOK_LF_QC[self.qc][ctx], s, 5);
        }
    }
    pub(crate) fn sym_luma8_eob_hf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma8_eob_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA8_EOB_TOK_HF_QC;
            self.sym_static(&LUMA8_EOB_TOK_HF_QC[self.qc][ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_luma8_eob_lf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma8_eob_lf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA8_EOB_TOK_LF_QC;
            self.sym_static(&LUMA8_EOB_TOK_LF_QC[self.qc][ctx], s, nsyms);
        }
    }
    // ---- luma 16x16 -------------------------------------------------------
    pub(crate) fn sym_luma16_hf(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma16_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA16_BASE_TOK_HF_QC;
            self.sym_static(&LUMA16_BASE_TOK_HF_QC[self.qc][ctx], s, 3);
        }
    }
    pub(crate) fn sym_luma16_lf(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma16_lf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                5,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA16_BASE_TOK_LF_QC;
            self.sym_static(&LUMA16_BASE_TOK_LF_QC[self.qc][ctx], s, 5);
        }
    }
    pub(crate) fn sym_luma16_eob_hf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma16_eob_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA16_EOB_TOK_HF_QC;
            self.sym_static(&LUMA16_EOB_TOK_HF_QC[self.qc][ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_luma16_eob_lf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma16_eob_lf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA16_EOB_TOK_LF_QC;
            self.sym_static(&LUMA16_EOB_TOK_LF_QC[self.qc][ctx], s, nsyms);
        }
    }
    // ---- luma 32x32 -------------------------------------------------------
    pub(crate) fn sym_luma32_hf(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma32_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA32_BASE_TOK_HF_QC;
            self.sym_static(&LUMA32_BASE_TOK_HF_QC[self.qc][ctx], s, 3);
        }
    }
    pub(crate) fn sym_luma32_lf(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma32_lf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                5,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA32_BASE_TOK_LF_QC;
            self.sym_static(&LUMA32_BASE_TOK_LF_QC[self.qc][ctx], s, 5);
        }
    }
    pub(crate) fn sym_luma32_eob_hf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma32_eob_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA32_EOB_TOK_HF_QC;
            self.sym_static(&LUMA32_EOB_TOK_HF_QC[self.qc][ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_luma32_eob_lf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.luma32_eob_lf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::LUMA32_EOB_TOK_LF_QC;
            self.sym_static(&LUMA32_EOB_TOK_LF_QC[self.qc][ctx], s, nsyms);
        }
    }

    pub(crate) fn sym_br_hf(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.br_hf[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::BR_TOK_HF_QC;
            self.sym_static(&BR_TOK_HF_QC[self.qc][ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_br(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.br[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::BR_TOK_QC;
            self.sym_static(&BR_TOK_QC[self.qc][ctx], s, nsyms);
        }
    }

    // ---- EOB bin (7-symbol, used via encode_eob) --------------------------
    pub(crate) fn sym_eob_bin(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = if self.inter_txb {
                &mut cs.eob_bin_inter
            } else {
                &mut cs.eob_bin
            };
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::EOB_BIN_QC;
            self.sym_static(&EOB_BIN_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_eob64_luma(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.eob64_luma;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::EOB64_LUMA_QC;
            self.sym_static(&EOB64_LUMA_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_eob128_luma(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.eob128_luma;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::EOB128_LUMA_QC;
            self.sym_static(&EOB128_LUMA_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_eob256(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.eob256;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::EOB256_QC;
            self.sym_static(&EOB256_QC[self.qc], s, nsyms);
        }
    }

    pub(crate) fn sym_eob256_inter(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.eob256_inter,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::INTER_EOB256_QC;
            self.sym_static(&INTER_EOB256_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_eob512(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.eob512;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::EOB512_QC;
            self.sym_static(&EOB512_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_chr_eob_bin(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.chr_eob_bin;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::CHROMA_EOB_BIN_QC;
            self.sym_static(&CHROMA_EOB_BIN_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_chr_eob32(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.chr_eob32;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::CHROMA_EOB32_QC;
            self.sym_static(&CHROMA_EOB32_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_chr_eob64(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.chr_eob64;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::CHROMA_EOB64_QC;
            self.sym_static(&CHROMA_EOB64_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_chr_eob128(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.chr_eob128;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::CHROMA_EOB128_QC;
            self.sym_static(&CHROMA_EOB128_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_chr_eob256(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.chr_eob256;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::CHROMA_EOB256_QC;
            self.sym_static(&CHROMA_EOB256_QC[self.qc], s, nsyms);
        }
    }
    pub(crate) fn sym_chr_eob512(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.chr_eob512;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_qctx::CHROMA_EOB512_QC;
            self.sym_static(&CHROMA_EOB512_QC[self.qc], s, nsyms);
        }
    }
    // ---- TX4 --------------------------------------------------------------
    pub(crate) fn sym_eob16_q0(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.eob16_q0[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfx_4tx::EOB16_Q0;
            self.sym_static(&EOB16_Q0[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_base_lf_tx4(&mut self, ctx: usize, pair: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_lf_tx4[ctx][pair];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                5,
            );
        } else {
            use crate::av2::cdfx_4tx::BASE_LF_TX4_Q0;
            self.sym_static(&BASE_LF_TX4_Q0[ctx][pair], s, 5);
        }
    }
    pub(crate) fn sym_base_tx4(&mut self, ctx: usize, pair: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_tx4[ctx][pair];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdfx_4tx::BASE_TX4_Q0;
            self.sym_static(&BASE_TX4_Q0[ctx][pair], s, 3);
        }
    }
    pub(crate) fn sym_base_lf_eob_tx4(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_lf_eob_tx4[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfx_4tx::BASE_LF_EOB_TX4_Q0;
            self.sym_static(&BASE_LF_EOB_TX4_Q0[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_base_eob_tx4(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_eob_tx4[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfx_4tx::BASE_EOB_TX4_Q0;
            self.sym_static(&BASE_EOB_TX4_Q0[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_br_lf_q0(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.br_lf_q0[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfx_4tx::BR_LF_Q0;
            self.sym_static(&BR_LF_Q0[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_br_q0(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.br_q0[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfx_4tx::BR_Q0;
            self.sym_static(&BR_Q0[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_base_lf_uv(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_lf_uv[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                5,
            );
        } else {
            // Lossy TX_4X4 chroma still uses the frame q-context.  Using the
            // q0-only table here makes disable_cdf_update bitstreams diverge as
            // soon as base_q_idx crosses the qc0/qc1 boundary.
            use crate::av2::cdfs_uv_qcx::BASE_LF_UV_QCX;
            self.sym_static(&BASE_LF_UV_QCX[self.qc][ctx], s, 5);
        }
    }
    pub(crate) fn sym_base_uv(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_uv[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdfs_uv_qcx::BASE_UV_QCX;
            self.sym_static(&BASE_UV_QCX[self.qc][ctx], s, 3);
        }
    }
    pub(crate) fn sym_br_uv(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.br_uv[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_uv_qcx::BR_UV_QCX;
            self.sym_static(&BR_UV_QCX[self.qc][ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_base_lf_eob_uv(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_lf_eob_uv[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_uv_qcx::BASE_LF_EOB_UV_QCX;
            self.sym_static(&BASE_LF_EOB_UV_QCX[self.qc][ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_base_eob_uv(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.base_eob_uv[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdfs_uv_qcx::BASE_EOB_UV_QCX;
            self.sym_static(&BASE_EOB_UV_QCX[self.qc][ctx], s, nsyms);
        }
    }
    // ---- intra mode -------------------------------------------------------
    pub(crate) fn sym_y_set(&mut self, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.y_set;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdf_state::Y_SET_INIT;
            self.sym_static(&Y_SET_INIT, s, 3);
        }
    }
    pub(crate) fn sym_y_idx0(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.y_idx0[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::Y_IDX0_INIT;
            self.sym_static(&Y_IDX0_INIT[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_y_idx1(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.y_idx1[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::Y_IDX1_INIT;
            self.sym_static(&Y_IDX1_INIT[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_uv_mode(&mut self, ctx: usize, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.uv_mode[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::UV_MODE_INIT;
            self.sym_static(&UV_MODE_INIT[ctx], s, nsyms);
        }
    }
    pub(crate) fn sym_delta_q(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.delta_q;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::DELTA_Q_INIT;
            self.sym_static(&DELTA_Q_INIT, s, nsyms);
        }
    }
    // ---- TX partition -----------------------------------------------------
    pub(crate) fn sym_tx_part_64(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_part_64;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::tables_tx32::TX_PART_2D_64;
            self.sym_static(&TX_PART_2D_64, s, nsyms);
        }
    }
    pub(crate) fn sym_tx_part_64x16(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_part_64x16;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::TX_PART_64X16_INIT;
            self.sym_static(&TX_PART_64X16_INIT, s, nsyms);
        }
    }
    pub(crate) fn sym_tx_part_16x64(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_part_16x64;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::TX_PART_16X64_INIT;
            self.sym_static(&TX_PART_16X64_INIT, s, nsyms);
        }
    }
    pub(crate) fn sym_tx_part_64x32(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_part_64x32;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::TX_PART_64X32_INIT;
            self.sym_static(&TX_PART_64X32_INIT, s, nsyms);
        }
    }
    pub(crate) fn sym_tx_part_32x64(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_part_32x64;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::TX_PART_32X64_INIT;
            self.sym_static(&TX_PART_32X64_INIT, s, nsyms);
        }
    }
    /// CCSO per-superblock on/off flag (adaptive 2-symbol CDF, `ccso[plane][ctx]`).
    /// Only emitted when CCSO is enabled for `plane`; mirrors AVM `write_ccso`.
    pub(crate) fn sym_ccso(&mut self, plane: usize, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.ccso[plane][ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                1,
            );
        } else {
            // CCSO requires adaptive CDFs (always enabled for lossy frames).
            unreachable!("CCSO flag emitted without adaptive CDF state");
        }
    }

    /// Emit the per-CDEF-unit "strength is index 0" flag (`s = 1` => index 0 / off).
    /// Used only for per-block CDEF (nb_cdef_strengths == 2, so index 1 = the single
    /// active strength needs no further symbol).
    pub(crate) fn sym_cdef(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.cdef[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                1,
            );
        } else {
            unreachable!("CDEF strength flag emitted without adaptive CDF state");
        }
    }

    pub(crate) fn sym_tx_short_side(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.tx_short_side[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cdf_state::TX_SHORT_SIDE_INIT;
            self.sym_static(&TX_SHORT_SIDE_INIT[ctx], s, 3);
        }
    }
    // ---- CfL -------------------------------------------------------------
    pub(crate) fn bool_cfl_is(&mut self, ctx: usize, static_cdf: u32, bit: u32) {
        self.cfl_signaled = bit != 0; // what the stream actually says
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.cfl_is[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn bool_cfl_mhccp(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.cfl_mhccp;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    /// mh_dir (MHCCP filter direction), 3-ary symbol. `ctx` is the
    /// size_group_lookup context (0..4). Static fallback uses the default CDF.
    pub(crate) fn sym_mh_dir(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.filter_dir[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                3,
            );
        } else {
            use crate::av2::cfl::FILTER_DIR_CDF;
            self.sym_static(&FILTER_DIR_CDF[ctx], s, 3);
        }
    }
    pub(crate) fn bool_cfl_index(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.cfl_index;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn sym_cfl_sign(&mut self, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.cfl_sign;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                8,
            );
        } else {
            use crate::av2::cfl::CFL_SIGN_ICDF;
            self.sym_static(&CFL_SIGN_ICDF, s, 8);
        }
    }
    pub(crate) fn sym_cfl_alpha(&mut self, ctx: usize, s: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.cfl_alpha[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                8,
            );
        } else {
            use crate::av2::cfl::CFL_ALPHA_ICDF;
            self.sym_static(&CFL_ALPHA_ICDF[ctx], s, 8);
        }
    }
    /// Adaptive 4x4 chroma txb_skip with an explicit plane selector and context
    /// index, avoiding value-collision ambiguity (the neutral 16384 probability
    /// appears at multiple contexts). `is_v` picks the V-plane table (avm
    /// v_txb_skip); otherwise the U/luma TX4 table. `static_cdf` is still used for
    /// the non-adaptive (static) fallback.
    pub(crate) fn bool_txb_skip_tx4_ctx(
        &mut self,
        static_cdf: u32,
        bit: u32,
        is_v: bool,
        ctx: usize,
    ) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = if is_v {
                &mut cs.skip_v[ctx]
            } else {
                &mut cs.skip_tx4[ctx]
            };
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn bool_txb_skip(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let (table, ctx) = cs.skip_slot_of(static_cdf as u16);
            let cdf = match table {
                0 if self.inter_txb => &mut cs.txb_skip_inter[ctx],
                0 => &mut cs.txb_skip[ctx],
                1 => &mut cs.skip_tx64[ctx],
                3 => &mut cs.skip_tx16[ctx],
                4 => &mut cs.skip_tx8[ctx],
                6 => &mut cs.skip_tx16_inter[ctx],
                5 => &mut cs.skip_tx4[ctx],
                _ => &mut cs.skip_v[ctx],
            };
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    /// Adaptive skip with an explicit table id (0=TX32, 1=TX64, 2=V). Used where
    /// the caller knows the table, avoiding value-collision ambiguity for the
    /// neutral 16384 probability that appears in multiple skip tables.
    pub(crate) fn bool_u_skip32(&mut self, ctx: usize, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = if self.inter_txb {
                &mut cs.txb_skip_inter[ctx]
            } else {
                &mut cs.txb_skip[ctx]
            };
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(
                crate::av2::cdfs_qctx::CHROMA_SKIP_TX32_QC[self.qc][ctx] as u32,
                bit,
            );
        }
    }
    pub(crate) fn bool_u_skip64(&mut self, ctx: usize, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = if self.inter_txb {
                &mut cs.skip_tx64_inter[ctx]
            } else {
                &mut cs.skip_tx64[ctx]
            };
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(
                crate::av2::cdfs_qctx::CHROMA_SKIP_TX64_QC[self.qc][ctx] as u32,
                bit,
            );
        }
    }
    pub(crate) fn bool_u_skip8_inter(&mut self, ctx: usize, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                &mut cs.skip_tx8_inter[ctx],
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(
                crate::av2::cdfs_qctx::INTER_SKIP_TX8_QC[self.qc][ctx] as u32,
                bit,
            );
        }
    }
    pub(crate) fn bool_v_skip(&mut self, ctx: usize, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.skip_v[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(
                crate::av2::cdfs_qctx::V_SKIP_TX4_QC[self.qc][ctx] as u32,
                bit,
            );
        }
    }
    pub(crate) fn bool_skip_tbl(&mut self, static_cdf: u32, bit: u32, table: u8) {
        if let Some(ref mut cs) = self.cdf_state {
            // Auto-detect the skip table (TX32 / TX64 / V) from the resolved static CDF
            // value rather than trusting the caller's `table` hint: a chroma U block coded
            // at TX_32X32 (e.g. a partial edge superblock) carries a CHROMA_SKIP_TX32
            // value and must adapt the TX32 slot (avm txs_ctx=3), not the TX64 slot. The
            // V plane is unambiguous and is selected explicitly.
            let (slot, ctx) = if table == 2 {
                (2u8, static_cdf as usize)
            } else {
                cs.skip_slot_of(static_cdf as u16)
            };
            let cdf = match slot {
                0 if self.inter_txb => &mut cs.txb_skip_inter[ctx],
                0 => &mut cs.txb_skip[ctx],
                1 => &mut cs.skip_tx64[ctx],
                3 => &mut cs.skip_tx16[ctx],
                4 => &mut cs.skip_tx8[ctx],
                6 => &mut cs.skip_tx16_inter[ctx],
                5 => &mut cs.skip_tx4[ctx],
                _ => &mut cs.skip_v[ctx],
            };
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else if table == 2 {
            use crate::av2::cdfs_qctx::V_SKIP_TX4_QC;
            self.encode_bool(V_SKIP_TX4_QC[self.qc][static_cdf as usize] as u32, bit);
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn bool_dc_sign(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let ctx = cs.dc_sign_ctx_of(static_cdf as u16);
            let cdf = &mut cs.dc_sign[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn bool_eob_extra(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.eob_extra;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn bool_do_split(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let ctx = cs.do_split_ctx_of(static_cdf as u16);
            let cdf = &mut cs.do_split[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }

    /// Adaptive `do_square_split` bool (PARTITION_SPLIT selection). Static mode is
    /// byte-identical to `encode_bool`; adaptive mode adapts the per-context working
    /// copy, matching avmdec which adapts this symbol on plane 0.
    pub(crate) fn bool_do_square_split(&mut self, static_cdf: u32, bit: u32) {
        if let Some(ref mut cs) = self.cdf_state {
            let ctx = cs.do_square_split_ctx_of(static_cdf as u16);
            let cdf = &mut cs.do_square_split[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }

    /// Adaptive rect_type partition bool. In static mode this is byte-identical
    /// to encode_bool; in adaptive mode it adapts the per-context working copy
    /// (avm rect_type_cdf), matching avmdec which adapts this symbol.
    pub(crate) fn bool_rect_type(&mut self, static_cdf: u32, bit: u32, ctx: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.rect_type[ctx];
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    /// Adaptive tx-partition / misc bool, value-keyed (each distinct static CDF
    /// value is a distinct avmdec context). Byte-identical to encode_bool in
    /// static mode; adapts the per-value working copy in adaptive mode.
    pub(crate) fn bool_txfm_part(&mut self, static_cdf: u32, bit: u32) {
        if let Some(cs) = self.cdf_state.as_deref_mut() {
            let cdf = cs.txfm_bool(static_cdf as u16);
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                bit as usize,
                1,
            );
        } else {
            self.encode_bool(static_cdf, bit);
        }
    }
    pub(crate) fn sym_intra_ext_tx16(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.intra_ext_tx16;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::INTRA_EXT_TX16_INIT;
            self.sym_static(&INTRA_EXT_TX16_INIT, s, nsyms);
        }
    }
    /// Intra ext-tx type for a native TX_8X8 leaf. Mirrors [`sym_intra_ext_tx16`]:
    /// the decoder's `intra_ext_tx_cdf[INTRA_TX_SET1][TX_8X8]` is adaptive, so this
    /// MUST adapt too. Coding it with a static cdf desyncs after the first 8x8 leaf.
    pub(crate) fn sym_intra_ext_tx8(&mut self, s: usize, nsyms: usize) {
        if let Some(ref mut cs) = self.cdf_state {
            let cdf = &mut cs.intra_ext_tx8;
            Self::sym_mut_inner(
                &mut self.low,
                &mut self.range,
                &mut self.count,
                &mut self.output,
                cdf,
                s,
                nsyms,
            );
        } else {
            use crate::av2::cdf_state::INTRA_EXT_TX8_INIT;
            self.sym_static(&INTRA_EXT_TX8_INIT, s, nsyms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RangeEncoder;

    fn assert_static_matches_initial_adaptive<F>(qc: usize, emit: F)
    where
        F: Fn(&mut RangeEncoder),
    {
        let mut static_enc = RangeEncoder::new();
        static_enc.qc = qc;

        let mut adaptive_enc = RangeEncoder::new();
        adaptive_enc.qc = qc;
        adaptive_enc.enable_adaptive_cdf(qc);

        emit(&mut static_enc);
        emit(&mut adaptive_enc);

        assert_eq!(static_enc.low, adaptive_enc.low);
        assert_eq!(static_enc.range, adaptive_enc.range);
        assert_eq!(static_enc.count, adaptive_enc.count);
        assert_eq!(static_enc.output, adaptive_enc.output);
    }

    #[test]
    fn static_tx4_chroma_uses_the_frame_q_context() {
        for qc in 0..4 {
            for ctx in 0..12 {
                for s in 0..=5 {
                    assert_static_matches_initial_adaptive(qc, |enc| enc.sym_base_lf_uv(ctx, s));
                }
                for s in 0..=3 {
                    assert_static_matches_initial_adaptive(qc, |enc| enc.sym_base_uv(ctx, s));
                }
            }

            for ctx in 0..4 {
                for s in 0..=3 {
                    assert_static_matches_initial_adaptive(qc, |enc| enc.sym_br_uv(ctx, s, 3));
                }
                for s in 0..=4 {
                    assert_static_matches_initial_adaptive(qc, |enc| {
                        enc.sym_base_lf_eob_uv(ctx, s, 4)
                    });
                }
                for s in 0..=2 {
                    assert_static_matches_initial_adaptive(qc, |enc| {
                        enc.sym_base_eob_uv(ctx, s, 2)
                    });
                }
            }
        }
    }
}
