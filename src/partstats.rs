//! TEMPORARY instrumentation: how much picture area each 16x16 partition choice
//! actually wins, so the value of improving a given leaf shape can be sized
//! before any of it is built. Enabled by `MT_PARTSTATS`; prints on drop.
use crate::coder::Part16;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) static COUNTS: [AtomicU64; 8] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub(crate) fn enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MT_PARTSTATS").is_ok())
}

pub(crate) static B64: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Tally the 64x64 NONE-vs-SPLIT decision, to bound how much BLOCK_64X64 work
/// is worth: index 0 = whole-64 kept, 1 = split to four 32x32.
pub(crate) fn tally64(none: bool) {
    if !enabled() {
        return;
    }
    B64[usize::from(!none)].fetch_add(1, Ordering::Relaxed);
}

pub(crate) static MODE_COST_SUM: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
pub(crate) static MODE_COST_N: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Accumulate the real contextual kf_y symbol cost (milli-bits) split by
/// DC (index 0) vs non-DC (index 1), to see what magnitude the flat 30-bit
/// thumb is actually standing in for.
pub(crate) fn tally_mode_cost(m: usize, bits: f32) {
    if !enabled() {
        return;
    }
    let i = usize::from(m != 0);
    MODE_COST_SUM[i].fetch_add((bits * 1000.0) as u64, Ordering::Relaxed);
    MODE_COST_N[i].fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn tally16(p: Part16) {
    if !enabled() {
        return;
    }
    let i = match p {
        Part16::Intrabc => return,
        Part16::None => 0,
        Part16::Split => 1,
        Part16::Horz => 2,
        Part16::Vert => 3,
        Part16::HorzA => 4,
        Part16::HorzB => 5,
        Part16::VertA => 6,
        Part16::VertB => 7,
    };
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
}
