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

/// How the tile's decision sites behave. Kept on `LossyTile` (not a call
/// parameter) so the `decode_sb` recursion needs no signature changes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum SbMode {
    /// Fused decide+emit (historical behaviour).
    #[default]
    Off,
    /// Run every search, push each winner onto the record. Emitted bytes are
    /// byte-identical to `Off` (capture is write-only side effects).
    Capture,
    /// Pop each winner from the record instead of searching. Emit runs live.
    Replay,
}

/// Transform type the luma winner was coded with (the `best_is_*` flags).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum TxSel {
    #[default]
    Dct,
    Adst,
    AdstDct,
    DctAdst,
    Idtx,
}

impl TxSel {
    /// Pack the four winner flags (mutually exclusive) into the enum.
    fn from_flags(adst: bool, idtx: bool, adstdct: bool, dctadst: bool) -> Self {
        if idtx {
            TxSel::Idtx
        } else if adst {
            TxSel::Adst
        } else if adstdct {
            TxSel::AdstDct
        } else if dctadst {
            TxSel::DctAdst
        } else {
            TxSel::Dct
        }
    }
}

/// `LumaSel.filter` sentinel: the winner did not use filter-intra.
const NO_FILTER: u8 = 0xff;

/// One coded luma block's search winner: enough to re-evaluate exactly one
/// candidate per sub-search in Replay (mode loop, filter-intra, angle-delta,
/// tx refinement), regenerating the identical coefficients and prediction
/// without visiting the losing candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LumaSel {
    mode: u8,
    delta: i8,
    /// Winning `FilterIntraMode as u8`, or [`NO_FILTER`].
    filter: u8,
    tx: TxSel,
}

/// One coded block's chroma winner: the uv prediction mode actually signaled
/// (`CFL_PRED` when chroma-from-luma won; alphas re-derived deterministically).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UvSel {
    uv: u8,
}

/// Call-order log of one tile's RD decisions. Sequences are independent
/// cursors, each aligned by its own deterministic call order.
#[derive(Default, Clone, PartialEq)]
struct DecisionRecord {
    /// Partition choice per decision site, in `decode_sb` recursion order:
    /// 32-level (`choose_rect32`), 16-level (`partition_choice_16`), 8-level
    /// (`want_split`/`choose_rect8`, `Split` = the 4x4 scaffold).
    parts: Vec<Part16>,
    /// Luma search winner per converted coded block, in call order.
    luma: Vec<LumaSel>,
    /// Chroma winner per converted coded block, in call order.
    uv: Vec<UvSel>,
    /// The winner's POST-TRELLIS luma coefficients, 1:1 with `luma` (same
    /// cursor). With these + the preloaded recon, Replay is PURE EMIT: no
    /// prediction, no transforms, no trellis — the AV2 "capture the winner,
    /// don't recompute it" design.
    luma_cf: Vec<Vec<i32>>,
    /// The winner's chroma coefficients per plane, 1:1 with `uv`. Empty vecs
    /// for chroma-less blocks (mono / chroma-less SPLIT4 units).
    uv_cf: Vec<[Vec<i32>; 2]>,
    /// CfL alphas per `uv` entry ([0,0] unless `uv == CFL_PRED`).
    cfl: Vec<[i32; 2]>,
    /// The superblock's reconstructed pixel blocks (per plane, tightly packed
    /// `h2*w2`), ONE entry per SB. Streamed with the record so the pure-emit
    /// replay copies pixels in instead of re-running prediction + inverse
    /// transforms; unconverted (DC) leaves then read correct neighbours and
    /// regenerate identical values.
    recon: Vec<[Vec<i32>; 3]>,
}

/// Cursor state for [`SbMode::Replay`]; one position per record sequence.
#[derive(Default, Clone, Copy)]
struct RecordCursor {
    parts: usize,
    luma: usize,
    uv: usize,
}

impl<'a> LossyTile<'a> {
    /// Funnel for a partition-level decision site: search / capture / replay
    /// according to `self.sb_mode`. `f` must be the complete decision (any
    /// pre-gates folded in) so a replayed value fully determines the branch.
    #[inline]
    fn part_decision(&mut self, f: impl Fn(&Self) -> Part16) -> Part16 {
        match self.sb_mode {
            SbMode::Off => f(self),
            SbMode::Capture => {
                let c = f(self);
                self.rec.parts.push(c);
                c
            }
            SbMode::Replay => {
                let c = self.rec.parts[self.cur.parts];
                self.cur.parts += 1;
                c
            }
        }
    }

    /// Replay-side pop of the current block's luma winner; `None` outside
    /// Replay. Each converted leaf pops exactly once, first thing.
    #[inline]
    fn luma_sel_replay(&mut self) -> Option<LumaSel> {
        if self.sb_mode == SbMode::Replay {
            let s = self.rec.luma[self.cur.luma];
            self.cur.luma += 1;
            Some(s)
        } else {
            None
        }
    }

    /// Capture-side push of the current block's luma winner (no-op otherwise).
    /// Must be called exactly once per converted leaf, mirroring the pop.
    #[inline]
    fn push_luma_sel(&mut self, s: LumaSel) {
        if self.sb_mode == SbMode::Capture {
            self.rec.luma.push(s);
        }
    }

    /// Replay-side pop of the current block's chroma winner; `None` outside
    /// Replay. Exactly one per converted leaf (mono pushes a DC dummy so the
    /// cursor stays aligned across formats).
    #[inline]
    fn uv_sel_replay(&mut self) -> Option<UvSel> {
        if self.sb_mode == SbMode::Replay {
            let s = self.rec.uv[self.cur.uv];
            self.cur.uv += 1;
            Some(s)
        } else {
            None
        }
    }

    /// Replay-side take of the current luma block's captured coefficients.
    /// Uses the SAME cursor position as the immediately preceding
    /// [`luma_sel_replay`] pop (call it right after). Empties the slot (the
    /// record is consumed once).
    #[inline]
    fn luma_cf_replay(&mut self) -> Option<Vec<i32>> {
        if self.sb_mode == SbMode::Replay {
            Some(std::mem::take(&mut self.rec.luma_cf[self.cur.luma - 1]))
        } else {
            None
        }
    }

    /// Capture-side push of the winner's luma coefficients (no-op otherwise);
    /// call exactly once per `push_luma_sel`, right after it.
    #[inline]
    fn push_luma_cf(&mut self, cf: &[i32]) {
        if self.sb_mode == SbMode::Capture {
            self.rec.luma_cf.push(cf.to_vec());
        }
    }

    /// Replay-side take of the current block's captured chroma coefficients
    /// and CfL alphas (cursor position of the preceding [`uv_sel_replay`]).
    #[inline]
    fn uv_cf_replay(&mut self) -> Option<([Vec<i32>; 2], [i32; 2])> {
        if self.sb_mode == SbMode::Replay {
            let i = self.cur.uv - 1;
            Some((std::mem::take(&mut self.rec.uv_cf[i]), self.rec.cfl[i]))
        } else {
            None
        }
    }

    /// Capture-side push of the winner's chroma coefficients + CfL alphas
    /// (no-op otherwise); exactly once per `push_uv_sel`, right after it.
    #[inline]
    fn push_uv_cf(&mut self, u: &[i32], v: &[i32], cfl: [i32; 2]) {
        if self.sb_mode == SbMode::Capture {
            self.rec.uv_cf.push([u.to_vec(), v.to_vec()]);
            self.rec.cfl.push(cfl);
        }
    }

    /// Capture-side push of the current block's chroma winner (no-op otherwise).
    #[inline]
    fn push_uv_sel(&mut self, s: UvSel) {
        if self.sb_mode == SbMode::Capture {
            self.rec.uv.push(s);
        }
    }
}
