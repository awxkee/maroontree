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
use crate::av2::cfl::CflChoice;
use crate::av2::coder::Coeff;

thread_local! {
    static TILE_SUBENCODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// RAII guard that marks the current thread as running a per-tile sub-encode for
/// its lifetime, restoring the previous value on drop.
pub(crate) struct TileSubencodeGuard(bool);
impl TileSubencodeGuard {
    #[inline]
    pub(crate) fn enter() -> Self {
        let prev = TILE_SUBENCODE.with(|c| c.replace(true));
        TileSubencodeGuard(prev)
    }
}
impl Drop for TileSubencodeGuard {
    #[inline]
    fn drop(&mut self) {
        let prev = self.0;
        TILE_SUBENCODE.set(prev);
    }
}
#[inline]
pub(crate) fn in_tile_subencode() -> bool {
    TILE_SUBENCODE.with(|c| c.get())
}

/// The luma tx-partition a whole-64 SB committed to (mirrors the inline `Part`
/// enum in the 4:4:4 fast path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WholePart {
    Split,
    Vert4,
    Horz4,
}

/// The chroma decision a whole-64 SB committed to.
#[derive(Clone)]
pub(crate) enum WholeChroma {
    /// Cross-component-from-luma prediction (the choice carries the alphas +
    /// per-pixel luma AC needed to reconstruct without `recy`).
    Cfl(CflChoice),
    /// A directional/DC chroma intra mode signalled via `uv_mode`.
    Mode(usize),
}

/// Per-superblock winning decision for the whole-64 interior fast path.
#[derive(Clone)]
pub(crate) struct Whole64Decision {
    pub part: WholePart,
    /// The four luma sub-transform coefficient groups for the committed part.
    pub tus: [Vec<Coeff>; 4],
    pub mode_idx: usize,
    pub angle_delta: i8,
    pub chroma: WholeChroma,
    /// The winner's 64×64 reconstruction (luma, u, v), captured at decide time so
    /// replay can `put_block` it directly instead of re-running predict+itx+add.
    /// Row-major 64×64. Feeds neighboring blocks' prediction in replay.
    pub recon_y: Vec<f32>,
    pub recon_u: Vec<f32>,
    pub recon_v: Vec<f32>,
}

/// One winning luma leaf in the split/native `ops` walk. An SB on that path
/// yields a *variable* number of these (one per search-bearing `Op::Leaf`), so a
/// `Walk` SB records a `Vec<LeafDecision>` — captured all-or-nothing (any
/// un-capturable leaf ⇒ the whole SB falls back to a re-search, keeping the
/// cursor 1 entry/SB).
///
/// Leaves are visited in `ops` order, so position is implicit (the walk
/// regenerates it). Only the size, the luma winner (single-TU coeffs + mode/tx
/// index), and the leaf's luma reconstruction are stored — enough for replay to
/// skip the luma mode/tx search and emit the same bits. Chroma is reproduced by
/// the already-cached MHCCP choice + the deterministic chroma projection, exactly
/// as the emit does, so no chroma fields are needed here.
#[derive(Clone)]
pub(crate) struct LeafDecision {
    pub bw_mi: usize,
    pub bh_mi: usize,
    /// The leaf's single luma transform-unit coefficients.
    pub tu: Vec<Coeff>,
    /// Luma intra mode (32×32 split leaf) or 16×16 tx-type index; 0 where unused.
    pub mode_idx: usize,
    /// The leaf's luma reconstruction, row-major `bw_mi*4 × bh_mi*4`, restored via
    /// `put_block(_rect)` in replay so neighbors + emit contexts match the search.
    pub recon_y: Vec<f32>,
}

/// One winning luma leaf in the 4:0:0 (monochrome) partition walk. Unlike
/// [`LeafDecision`] (single TU), a 4:0:0 leaf may carry 1 TU (the split
/// leaves) or 4 TUs (the whole-64 no-split `(16,16)` leaf, coded as 4× TX_32X32
/// via `encode_luma_sb`). Stored: the leaf shape, its transform-unit coefficient
/// groups, the luma mode index (`(16,16)`/`(8,8)`), the `(4,4)` tx16-type index
/// (else 0), and the leaf reconstruction (row-major `bw_mi*4 × bh_mi*4`) so
/// replay restores it via `put_block(_rect)` and skips the mode/tx search.
#[derive(Clone)]
pub(crate) struct Leaf400 {
    pub bw_mi: usize,
    pub bh_mi: usize,
    pub tus: Vec<Vec<Coeff>>,
    pub mode_idx: usize,
    pub tx_idx: usize,
    pub recon_y: Vec<f32>,
}

/// One 4:0:0 superblock's captured walk. `Walk` = every leaf shape was
/// replayable (`(16,16)`/`(8,8)`/`(4,4)`/`(2,2)`), so replay reuses each captured
/// winner (zero searches). `Fallback` = some leaf shape is not yet captured (the
/// native edge partitions), so replay re-searches that SB serially — byte-safe
/// because the serial replay recon entering it equals the fully-serial Off recon.
#[derive(Clone)]
pub(crate) enum Sb400 {
    Walk(Vec<Leaf400>),
    Fallback,
}

/// One winning luma leaf in the 4:2:0 partition walk. Like [`Leaf400`] (luma TUs
/// + mode/tx index + luma reconstruction) but ALSO carries the per-leaf chroma
///   decision (`CfL` for the whole-64 leaf; MHCCP for the smaller shapes; `None` =
///   deterministic DC chroma), so replay reuses the chroma winner and skips the
///   CfL/MHCCP RD search. Chroma reconstruction is re-derived deterministically
///   from the restored luma recon + this choice by re-running the emit's chroma
///   projection, so no chroma recon is stored.
#[derive(Clone)]
pub(crate) struct Leaf420 {
    pub bw_mi: usize,
    pub bh_mi: usize,
    pub tus: Vec<Vec<Coeff>>,
    pub mode_idx: usize,
    pub tx_idx: usize,
    pub recon_y: Vec<f32>,
    pub chroma: Option<CflChoice>,
}

/// One 4:2:0 superblock's captured walk (its own call-order sequence, one entry
/// per SB). `Walk` = every leaf shape was replayable, so replay reuses each
/// captured luma + chroma winner (zero searches). `Fallback` = some leaf shape is
/// not yet captured (edge/native shapes, or an inter-coded SB), so replay
/// re-searches that SB serially — byte-safe because the serial replay recon
/// entering it equals the fully-serial Off recon.
#[derive(Clone)]
pub(crate) enum Sb420 {
    Walk(Vec<Leaf420>),
    Fallback,
}

/// How a single walk leaf treats its luma mode/tx search. Passed per `Op::Leaf`
/// to the converted leaf methods (`(8,8)` split-32 + `(4,4)` tx16): `Off` =
/// search + emit inline (today's behavior); `Capture` = search, emit, and push
/// the winner into the SB's accumulator; `Replay` = restore the captured recon +
/// reuse the captured coeffs/mode, skipping the search entirely (pure emit).
pub(crate) enum LeafWalk<'a> {
    Off,
    Capture(&'a mut Vec<LeafDecision>),
    Replay(&'a LeafDecision),
}

/// One superblock's captured decision. Each SB takes exactly one branch
/// (whole-64 *or* the `ops` walk) and that branch is deterministic (it is
/// decided by `choose_luma_64x64_partition`, run in both decide and replay), so
/// a single call-order cursor stays aligned as long as **every** SB pushes
/// exactly one entry: `Whole64` at the fast-path exit, `Walk` at the walk exit.
/// No loop-top dispatch or per-SB `Fallback` sentinel is needed for that — each
/// branch pops its own variant. `Fallback` is the defensive/uncaptured case
/// (replay re-searches that SB).
#[derive(Clone)]
pub(crate) enum SbDecision {
    Whole64(Box<Whole64Decision>),
    Walk(Vec<LeafDecision>),
    Fallback,
}

/// Call-order log of per-SB decisions for one tile. `parts` is a SECOND,
/// independent call-order sequence holding the `choose_luma_64x64_partition`
/// result for each full-interior SB (in raster order) — replay pops it to skip
/// that ~35% partition search. It has its own cursor because it is populated at a
/// different point (before the whole-64-vs-walk branch) than `entries`.
#[derive(Clone, Default)]
pub(crate) struct DecisionRecord {
    entries: Vec<SbDecision>,
    parts: Vec<crate::av2::leaf::LumaPartitionDecision>,
    /// A THIRD call-order sequence: the MHCCP choice (`mhccp_decide` result) for
    /// each split-path 32×32 chroma leaf, so replay skips the MHCCP filter fit.
    mhccp: Vec<Option<CflChoice>>,
    /// 4:2:0 whole-64 luma mode-search winners (its own call-order sequence).
    luma420: Vec<Luma420>,
    /// 4:2:2 whole-64 fast-path chroma decision (`cfl_decide` result). Its own
    /// call-order sequence, popped once per whole-64 SB right after `luma420`, so
    /// replay skips the CfL RD search and reconstructs from the cached choice.
    chroma422_whole: Vec<Option<CflChoice>>,
    /// 4:0:0 partition-walk per-SB decisions (its own call-order sequence, one
    /// entry per SB): `Walk` reuses the captured leaf winners, `Fallback`
    /// re-searches. Populated by `decide_sb_400`.
    luma400: Vec<Sb400>,
    /// 4:2:0 partition-walk per-SB decisions (its own call-order sequence, one
    /// entry per SB): `Walk` reuses the captured leaf luma + chroma winners,
    /// `Fallback` re-searches. Populated by the 4:2:0 partition walk.
    luma420_walk: Vec<Sb420>,
}

impl DecisionRecord {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            parts: Vec::new(),
            mhccp: Vec::new(),
            luma420: Vec::new(),
            chroma422_whole: Vec::new(),
            luma400: Vec::new(),
            luma420_walk: Vec::new(),
        }
    }
    pub(crate) fn push(&mut self, d: SbDecision) {
        self.entries.push(d);
    }
    pub(crate) fn push_part(&mut self, p: crate::av2::leaf::LumaPartitionDecision) {
        self.parts.push(p);
    }
    pub(crate) fn push_mhccp(&mut self, m: Option<CflChoice>) {
        self.mhccp.push(m);
    }
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
    /// Concatenate another record's four call-order sequences onto this one. Used to
    /// merge per-SB records produced by a parallel wavefront decide back into a
    /// single raster-order record the serial `Replay` consumes: appending the
    /// per-SB records in raster order reproduces the exact sequence the serial
    /// Capture would have logged, so Replay stays byte-identical.
    #[allow(dead_code)]
    pub(crate) fn append(&mut self, mut other: DecisionRecord) {
        self.entries.append(&mut other.entries);
        self.parts.append(&mut other.parts);
        self.mhccp.append(&mut other.mhccp);
        self.luma420.append(&mut other.luma420);
        self.chroma422_whole.append(&mut other.chroma422_whole);
        self.luma400.append(&mut other.luma400);
        self.luma420_walk.append(&mut other.luma420_walk);
    }

    pub(crate) fn push_sb400(&mut self, s: Sb400) {
        self.luma400.push(s);
    }

    pub(crate) fn push_sb420(&mut self, s: Sb420) {
        self.luma420_walk.push(s);
    }
}

/// A read cursor over a captured record for the replay pass.
pub(crate) struct DecisionCursor<'a> {
    rec: &'a DecisionRecord,
    at: usize,
    part_at: usize,
    mhccp_at: usize,
    luma420_at: usize,
    chroma422_whole_at: usize,
    luma400_at: usize,
    luma420_walk_at: usize,
}

impl<'a> DecisionCursor<'a> {
    pub(crate) fn new(rec: &'a DecisionRecord) -> Self {
        Self {
            rec,
            at: 0,
            part_at: 0,
            mhccp_at: 0,
            luma420_at: 0,
            chroma422_whole_at: 0,
            luma400_at: 0,
            luma420_walk_at: 0,
        }
    }
    /// Next 4:0:0 walk decision in call order, or `None` if exhausted (defensive
    /// re-search). One entry is consumed per SB.
    pub(crate) fn next_sb400(&mut self) -> Option<&'a Sb400> {
        let s = self.rec.luma400.get(self.luma400_at);
        if s.is_some() {
            self.luma400_at += 1;
        }
        s
    }
    /// Next 4:2:0 walk decision in call order, or `None` if exhausted (defensive
    /// re-search). One entry is consumed per SB.
    pub(crate) fn next_sb420(&mut self) -> Option<&'a Sb420> {
        let s = self.rec.luma420_walk.get(self.luma420_walk_at);
        if s.is_some() {
            self.luma420_walk_at += 1;
        }
        s
    }
    /// Next partition decision in call order, or `None` if exhausted (defensive:
    /// a mismatch degrades to re-running `choose_luma_64x64_partition`).
    pub(crate) fn next_part(&mut self) -> Option<crate::av2::leaf::LumaPartitionDecision> {
        let p = self.rec.parts.get(self.part_at).copied();
        if p.is_some() {
            self.part_at += 1;
        }
        p
    }
    /// Next MHCCP choice in call order. Outer `Some` = a cached entry exists (use
    /// its inner `Option<CflChoice>`, skipping the fit); outer `None` = exhausted
    /// (defensive: re-run `mhccp_decide`).
    pub(crate) fn next_mhccp(&mut self) -> Option<Option<CflChoice>> {
        let m = self.rec.mhccp.get(self.mhccp_at).cloned();
        if m.is_some() {
            self.mhccp_at += 1;
        }
        m
    }
    /// Next decision in call order, or `Fallback` if the record is exhausted
    /// (defensive: a mismatch degrades to a re-search rather than a desync).
    pub(crate) fn next(&mut self) -> &'a SbDecision {
        static FALLBACK: SbDecision = SbDecision::Fallback;
        let d = self.rec.entries.get(self.at).unwrap_or(&FALLBACK);
        self.at += 1;
        d
    }
}

/// 4:2:0 whole-64 luma mode-search winner (shared record `luma420` sequence).
/// The 420 core has no `choose_luma_64x64_partition`; its dominant pass-2 cost is
/// the per-SB `encode_luma_sb` mode search, cached here so replay skips it.
#[derive(Clone)]
pub(crate) struct Luma420 {
    pub tus: [Vec<Coeff>; 4],
    pub mode_idx: usize,
    pub adelta: i8,
    /// 64×64 luma reconstruction (row-major), restored in replay for CfL/neighbors.
    pub recon_y: Vec<f32>,
}

impl DecisionRecord {
    pub(crate) fn push_luma420(&mut self, l: Luma420) {
        self.luma420.push(l);
    }
    /// Log the 4:2:2 whole-64 chroma CfL decision for this SB.
    pub(crate) fn push_chroma422_whole(&mut self, c: Option<CflChoice>) {
        self.chroma422_whole.push(c);
    }
}

impl<'a> DecisionCursor<'a> {
    /// Next 420 luma winner in call order, or `None` if exhausted (defensive
    /// re-search).
    pub(crate) fn next_luma420(&mut self) -> Option<Luma420> {
        let l = self.rec.luma420.get(self.luma420_at).cloned();
        if l.is_some() {
            self.luma420_at += 1;
        }
        l
    }
    /// Next 4:2:2 whole-64 chroma decision in call order. Outer `Some` = a cached
    /// entry exists (use its inner `Option<CflChoice>`, skipping the CfL search);
    /// outer `None` = exhausted (defensive: re-run `cfl_decide`).
    pub(crate) fn next_chroma422_whole(&mut self) -> Option<Option<CflChoice>> {
        let c = self
            .rec
            .chroma422_whole
            .get(self.chroma422_whole_at)
            .cloned();
        if c.is_some() {
            self.chroma422_whole_at += 1;
        }
        c
    }
}

/// How `encode_444_core` treats per-SB decisions.
pub(crate) enum DecideMode<'a> {
    /// Search every SB and emit inline (today's behavior; byte-identical).
    Off,
    /// Search every SB, emit inline, and log each winner into the record.
    Capture(&'a mut DecisionRecord),
    /// Replay winners from the record (re-searching only `Fallback` SBs).
    Replay(DecisionCursor<'a>),
}
