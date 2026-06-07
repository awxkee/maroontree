//! Bit-exact port of dav1d 1.4.1 CDEF (Constrained Directional Enhancement
//! Filter). Runs after the in-loop deblocking filter. Mirrors
//! `src/cdef_tmpl.c` (constrain / padding / cdef_filter_block / cdef_find_dir)
//! and the single-thread path of `src/cdef_apply_tmpl.c`.
//!
//! Encoder simplification that preserves bit-exactness: CDEF reads all of its
//! inputs (centre pixel, top/bottom/left/right neighbours) from *pre-CDEF*
//! (deblocked) data. dav1d achieves this with 2-line / 2x8 backups; we instead
//! filter from a full pre-CDEF copy of each plane into the output plane, so
//! every neighbour read is automatically pre-filter. The numeric result is
//! identical.

/// Sentinel for "pixel not available" — matches dav1d's INT16_MIN fill. Chosen
/// so that, read as unsigned it is huge (never lowers `min`), read as signed it
/// is very negative (never raises `max`), and `constrain(sentinel - px)` is 0.
const CDEF_UNAVAIL: i32 = -32768; // INT16_MIN

// CDEF is OFF by default: the RD strength search is expensive, and CDEF is an
// optional in-loop filter. Enable it per encoding thread before encoding. The
// flag is read on the calling thread (where the search, frame-level apply, and
// header writing all run), so it is race-free across parallel encodes/tests.
thread_local! {
    static CDEF_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Enable or disable CDEF for encodes initiated on the current thread (default off).
pub fn set_cdef_enabled(on: bool) {
    CDEF_ENABLED.with(|c| c.set(on));
}

/// Whether CDEF is enabled for the current thread.
pub fn cdef_enabled() -> bool {
    CDEF_ENABLED.with(|c| c.get())
}

pub const CDEF_HAVE_LEFT: u8 = 1 << 0;
pub const CDEF_HAVE_RIGHT: u8 = 1 << 1;
pub const CDEF_HAVE_TOP: u8 = 1 << 2;
pub const CDEF_HAVE_BOTTOM: u8 = 1 << 3;

/// `dav1d_cdef_directions[2 + 8 + 2][2]` in units of the tmp stride (12).
/// Offsets are `row * 12 + col`.
const CDEF_DIRECTIONS: [[i32; 2]; 12] = [
    [1 * 12 + 0, 2 * 12 + 0], // 6
    [1 * 12 + 0, 2 * 12 - 1], // 7
    [-1 * 12 + 1, -2 * 12 + 2], // 0
    [0 * 12 + 1, -1 * 12 + 2], // 1
    [0 * 12 + 1, 0 * 12 + 2], // 2
    [0 * 12 + 1, 1 * 12 + 2], // 3
    [1 * 12 + 1, 2 * 12 + 2], // 4
    [1 * 12 + 0, 2 * 12 + 1], // 5
    [1 * 12 + 0, 2 * 12 + 0], // 6
    [1 * 12 + 0, 2 * 12 - 1], // 7
    [-1 * 12 + 1, -2 * 12 + 2], // 0
    [0 * 12 + 1, -1 * 12 + 2], // 1
];

#[inline]
fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 {
        -v
    } else {
        v
    }
}

#[inline]
fn constrain(diff: i32, threshold: i32, shift: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let adiff = diff.abs();
    apply_sign(adiff.min((threshold - (adiff >> shift)).max(0)), diff)
}

#[inline]
fn ulog2(x: u32) -> i32 {
    31 - x.leading_zeros() as i32
}

#[inline]
fn umin(a: i32, b: i32) -> i32 {
    if (a as u32) < (b as u32) {
        a
    } else {
        b
    }
}

/// CDEF parameters as signalled in the frame header (single preset, n_bits = 0).
#[derive(Clone, Copy, Debug)]
pub struct CdefParams {
    pub damping: i32,    // 3..=6
    pub y_pri: i32,      // luma primary strength level (0..=15)
    pub y_sec: i32,      // luma secondary (0,1,2,4 after the sec==3 bump)
    pub uv_pri: i32,     // chroma primary
    pub uv_sec: i32,     // chroma secondary (0,1,2,4)
}

impl CdefParams {
    pub fn is_noop(&self) -> bool {
        self.y_pri == 0 && self.y_sec == 0 && self.uv_pri == 0 && self.uv_sec == 0
    }
}

/// dav1d `adjust_strength`: scale the luma primary strength by the block's
/// directional variance.
fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 {
        (ulog2(var >> 6)).min(12)
    } else {
        0
    };
    (strength * (4 + i) + 8) >> 4
}

/// Port of `cdef_find_dir_c`. `img` is an 8x8 block addressed via `stride`.
/// Returns (direction 0..7, variance).
fn cdef_find_dir(img: &[i32], origin: usize, stride: usize, bitdepth_min_8: i32) -> (usize, u32) {
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        let row = origin + y * stride;
        for x in 0..8usize {
            let px = (img[row + x] >> bitdepth_min_8) - 128;
            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0u32; 8];
    for n in 0..8usize {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as u32;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    const DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7usize {
        let d = DIV_TABLE[n];
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n]) as u32)
            * d;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n]) as u32)
            * d;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7]) as u32 * 105;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7]) as u32 * 105;

    for n in 0..4usize {
        let cp = n * 2 + 1;
        for m in 0..5usize {
            cost[cp] += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as u32;
        }
        cost[cp] *= 105;
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            cost[cp] += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m]) as u32)
                * d;
        }
    }

    let mut best_dir = 0usize;
    let mut best_cost = cost[0];
    for n in 1..8usize {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }
    let var = (best_cost.wrapping_sub(cost[best_dir ^ 4])) >> 10;
    (best_dir, var)
}

/// Build the 12-stride padded tmp buffer for one block, reading pre-CDEF data.
/// `tmp` has 12*12 entries; the logical origin (block pixel (0,0)) is at
/// `2 * 12 + 2`.
#[allow(clippy::too_many_arguments)]
fn padding(
    tmp: &mut [i32; 144],
    plane: &[i32],
    pw: usize,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    const TS: i32 = 12;
    let org: i32 = 2 * TS + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    // fill() the borders that don't exist with the sentinel.
    let mut fill = |tmp: &mut [i32; 144], ox: i32, oy: i32, fw: i32, fh: i32| {
        for j in 0..fh {
            for i in 0..fw {
                let idx = org + (oy + j) * TS + (ox + i);
                tmp[idx as usize] = CDEF_UNAVAIL;
            }
        }
    };
    if edges & CDEF_HAVE_TOP == 0 {
        fill(tmp, -2, -2, w as i32 + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        fill(tmp, -2, h as i32, w as i32 + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        fill(tmp, -2, y_start, 2, y_end - y_start);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        fill(tmp, w as i32, y_start, 2, y_end - y_start);
        x_end -= 2;
    }

    // Copy real pixels from the pre-CDEF plane for [x_start,x_end) x [y_start,y_end).
    for y in y_start..y_end {
        for x in x_start..x_end {
            let sx = bx as i32 + x;
            let sy = by as i32 + y;
            let v = plane[(sy as usize) * pw + sx as usize];
            let idx = org + y * TS + x;
            tmp[idx as usize] = v;
        }
    }
}

/// Port of `cdef_filter_block_c`. Filters one `w`x`h` block (w,h in {4,8}) from
/// pre-CDEF `inp` into `out`, both addressed by `pw` (plane width). `(bx,by)` is
/// the block's top-left pixel.
#[allow(clippy::too_many_arguments)]
fn cdef_filter_block(
    out: &mut [i32],
    inp: &[i32],
    pw: usize,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: usize,
    damping: i32,
    edges: u8,
    bitdepth_min_8: i32,
    maxval: i32,
) {
    const TS: i32 = 12;
    let org: i32 = 2 * TS + 2;
    let mut tmp = [0i32; 144];
    padding(&mut tmp, inp, pw, bx, by, w, h, edges);

    if pri_strength != 0 {
        let pri_tap = 4 - ((pri_strength >> bitdepth_min_8) & 1);
        let pri_shift = (damping - ulog2(pri_strength as u32)).max(0);
        if sec_strength != 0 {
            let sec_shift = damping - ulog2(sec_strength as u32);
            for y in 0..h {
                for x in 0..w {
                    let px = inp[(by + y) * pw + (bx + x)];
                    let t = (org + y as i32 * TS + x as i32) as usize;
                    let mut sum = 0i32;
                    let mut max = px;
                    let mut min = px;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2usize {
                        let off1 = CDEF_DIRECTIONS[dir + 2][k];
                        let p0 = tmp[(t as i32 + off1) as usize];
                        let p1 = tmp[(t as i32 - off1) as usize];
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = (pri_tap_k & 3) | 2;
                        min = umin(p0, min);
                        max = p0.max(max);
                        min = umin(p1, min);
                        max = p1.max(max);
                        let off2 = CDEF_DIRECTIONS[dir + 4][k];
                        let off3 = CDEF_DIRECTIONS[dir + 0][k];
                        let s0 = tmp[(t as i32 + off2) as usize];
                        let s1 = tmp[(t as i32 - off2) as usize];
                        let s2 = tmp[(t as i32 + off3) as usize];
                        let s3 = tmp[(t as i32 - off3) as usize];
                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                        min = umin(s0, min);
                        max = s0.max(max);
                        min = umin(s1, min);
                        max = s1.max(max);
                        min = umin(s2, min);
                        max = s2.max(max);
                        min = umin(s3, min);
                        max = s3.max(max);
                    }
                    let v = px + ((sum - (sum < 0) as i32 + 8) >> 4);
                    out[(by + y) * pw + (bx + x)] = v.clamp(min, max).clamp(0, maxval);
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let px = inp[(by + y) * pw + (bx + x)];
                    let t = (org + y as i32 * TS + x as i32) as usize;
                    let mut sum = 0i32;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2usize {
                        let off = CDEF_DIRECTIONS[dir + 2][k];
                        let p0 = tmp[(t as i32 + off) as usize];
                        let p1 = tmp[(t as i32 - off) as usize];
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = (pri_tap_k & 3) | 2;
                    }
                    let v = px + ((sum - (sum < 0) as i32 + 8) >> 4);
                    out[(by + y) * pw + (bx + x)] = v.clamp(0, maxval);
                }
            }
        }
    } else {
        // sec_strength only
        let sec_shift = damping - ulog2(sec_strength as u32);
        for y in 0..h {
            for x in 0..w {
                let px = inp[(by + y) * pw + (bx + x)];
                let t = (org + y as i32 * TS + x as i32) as usize;
                let mut sum = 0i32;
                for k in 0..2usize {
                    let off1 = CDEF_DIRECTIONS[dir + 4][k];
                    let off2 = CDEF_DIRECTIONS[dir + 0][k];
                    let s0 = tmp[(t as i32 + off1) as usize];
                    let s1 = tmp[(t as i32 - off1) as usize];
                    let s2 = tmp[(t as i32 + off2) as usize];
                    let s3 = tmp[(t as i32 - off2) as usize];
                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                }
                let v = px + ((sum - (sum < 0) as i32 + 8) >> 4);
                out[(by + y) * pw + (bx + x)] = v.clamp(0, maxval);
            }
        }
    }
}

/// uv direction remap per `uv_dirs` in dav1d (index by 4:2:2).
const UV_DIRS: [[usize; 8]; 2] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [7, 0, 2, 4, 5, 6, 6, 6],
];

/// Apply CDEF to one tile/frame. `planes[0..3]` are Y,U,V as i32 (mutated in
/// place). `skip8` is a per-8x8-luma-block flag (true = block had no coded
/// coefficients → not filtered). `(ss_hor, ss_ver)` is chroma subsampling.
#[allow(clippy::too_many_arguments)]
pub fn apply_cdef(
    planes: &mut [Vec<i32>; 3],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    ss_hor: usize,
    ss_ver: usize,
    mono: bool,
    skip8: &[bool],
    sb8w: usize,
    params: &CdefParams,
    bd: u8,
) {
    if params.is_noop() {
        return;
    }
    let bitdepth_min_8 = bd as i32 - 8;
    let maxval = (1i32 << bd) - 1;
    // Pre-CDEF copies (sources). Outputs are the plane buffers themselves.
let inp_y = planes[0].clone();
    let inp_u = if mono { Vec::new() } else { planes[1].clone() };
    let inp_v = if mono { Vec::new() } else { planes[2].clone() };

    let damping = params.damping + bitdepth_min_8;
    // Strength decode (already split into pri level + sec level by the caller;
    // here we just apply the bitdepth shift, mirroring cdef_apply).
    let y_pri = params.y_pri << bitdepth_min_8;
    let y_sec = params.y_sec << bitdepth_min_8;
    let uv_pri = params.uv_pri << bitdepth_min_8;
    let uv_sec = params.uv_sec << bitdepth_min_8;
    let uv_dir = if ss_hor == 1 && ss_ver == 0 {
        &UV_DIRS[1] // 4:2:2
    } else {
        &UV_DIRS[0]
    };

    let nbx = w.div_ceil(8);
    let nby = h.div_ceil(8);
    for by8 in 0..nby {
        for bx8 in 0..nbx {
            // skip map index (per-8x8 within the tile, row-major over sb8w cols)
            if skip8[by8 * sb8w + bx8] {
                continue;
            }
            let bx = bx8 * 8;
            let by = by8 * 8;
            let bw = (w - bx).min(8);
            let bh = (h - by).min(8);
            // CDEF only processes full 8x8 luma units (partial edge units at the
            // frame's right/bottom that are < 8 still get the available portion;
            // dav1d clamps via bw/bh through the block fns which require w,h in
            // {4,8}). We require 8x8 alignment of the buffer, which the encoder
            // guarantees by padding the recon to multiples of 8.
            debug_assert!(bw == 8 && bh == 8);
            let mut edges = 0u8;
            if bx8 > 0 {
                edges |= CDEF_HAVE_LEFT;
            }
            if bx + 8 < w {
                edges |= CDEF_HAVE_RIGHT;
            }
            if by8 > 0 {
                edges |= CDEF_HAVE_TOP;
            }
            if by + 8 < h {
                edges |= CDEF_HAVE_BOTTOM;
            }

            // direction (only needed if a primary strength is active)
            let (dir, var) = if y_pri != 0 || uv_pri != 0 {
                cdef_find_dir(&inp_y, by * w + bx, w, bitdepth_min_8)
            } else {
                (0, 0)
            };

            // luma
            if y_pri != 0 {
                let adj = adjust_strength(y_pri, var);
                if adj != 0 || y_sec != 0 {
                    cdef_filter_block(
                        &mut planes[0], &inp_y, w, bx, by, 8, 8, adj, y_sec, dir, damping,
                        edges, bitdepth_min_8, maxval,
                    );
                }
            } else if y_sec != 0 {
                cdef_filter_block(
                    &mut planes[0], &inp_y, w, bx, by, 8, 8, 0, y_sec, 0, damping, edges,
                    bitdepth_min_8, maxval,
                );
            }

            // chroma
            if !mono && (uv_pri != 0 || uv_sec != 0) {
                let uvdir = if uv_pri != 0 { uv_dir[dir] } else { 0 };
                let cbx = bx >> ss_hor;
                let cby = by >> ss_ver;
                let cbw = 8 >> ss_hor;
                let cbh = 8 >> ss_ver;
                let _ = (cw, ch);
                cdef_filter_block(
                    &mut planes[1], &inp_u, cw, cbx, cby, cbw, cbh, uv_pri, uv_sec, uvdir,
                    damping - 1, edges, bitdepth_min_8, maxval,
                );
                cdef_filter_block(
                    &mut planes[2], &inp_v, cw, cbx, cby, cbw, cbh, uv_pri, uv_sec, uvdir,
                    damping - 1, edges, bitdepth_min_8, maxval,
                );
            }
        }
    }
}

/// Encode a (pri, sec) strength pair into the 6-bit header value. `sec` is the
/// final strength (0,1,2,4); it is stored as a 2-bit index (4 -> 3).
pub fn encode_strength(pri: i32, sec: i32) -> u32 {
    let s_idx = if sec == 4 { 3 } else { sec };
    (((pri << 2) | s_idx) & 0x3f) as u32
}

/// Frame-level CDEF parameter heuristic, used identically by the frame-header
/// writer (to signal) and the encoder (to apply), so they stay in lock-step.
/// `base_q_idx` is the frame's base quantiser index (1..=255).
pub fn cdef_params_for(base_q_idx: u8, mono: bool) -> CdefParams {
    if mono {
        // Leave the alpha/mono path untouched (precise); signalled as zero-strength.
        return CdefParams { damping: 3, y_pri: 0, y_sec: 0, uv_pri: 0, uv_sec: 0 };
    }
    let q = base_q_idx as i32;
    // Ringing grows with quantisation, so ramp strength with q_idx. Kept modest
    // to clean grid-aligned ringing/colour points without over-smoothing.
    let y_pri = (q / 48).clamp(0, 6);
    let y_sec = if q > 140 { 2 } else { 1 };
    let uv_pri = (q / 72).clamp(0, 3);
    let uv_sec = 1;
    CdefParams {
        damping: 3,
        y_pri,
        y_sec,
        uv_pri,
        uv_sec,
    }
}

#[inline]
fn plane_sse(a: &[i32], b: &[i32]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as i64;
            d * d
        })
        .sum()
}

/// Rate-distortion CDEF strength search. The encoder has the source, so it tries
/// candidate strength presets against the deblocked reconstruction and keeps the
/// one minimising SSE-to-source — luma and chroma chosen independently (they are
/// signalled independently and the direction is shared). Damping is fixed at 3.
/// Returning all-zero strengths (a no-op) is always a candidate, so CDEF can
/// never make fidelity worse than plain deblocking.
#[allow(clippy::too_many_arguments)]
pub fn search_params(
    recon: &[Vec<i32>; 3],
    src: &[Vec<i32>; 3],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    ss_hor: usize,
    ss_ver: usize,
    mono: bool,
    skip8: &[bool],
    sb8w: usize,
    bd: u8,
) -> CdefParams {
    let damping = 3;
    let mut scratch: [Vec<i32>; 3] = recon.clone();

    // ---- luma: minimise luma SSE ----
    let mut best_y = (0i32, 0i32);
    let mut best_y_sse = plane_sse(&recon[0], &src[0]);
    const Y_PRI: [i32; 7] = [0, 1, 2, 3, 4, 6, 8];
    const SEC: [i32; 4] = [0, 1, 2, 4];
    for &y_pri in &Y_PRI {
        for &y_sec in &SEC {
            if y_pri == 0 && y_sec == 0 {
                continue;
            }
            scratch[0].copy_from_slice(&recon[0]);
            let p = CdefParams { damping, y_pri, y_sec, uv_pri: 0, uv_sec: 0 };
            apply_cdef(&mut scratch, w, h, cw, ch, ss_hor, ss_ver, mono, skip8, sb8w, &p, bd);
            let sse = plane_sse(&scratch[0], &src[0]);
            if sse < best_y_sse {
                best_y_sse = sse;
                best_y = (y_pri, y_sec);
            }
        }
    }

    // ---- chroma: minimise U+V SSE (direction needs valid luma in scratch[0]) ----
    let mut best_uv = (0i32, 0i32);
    if !mono {
        scratch[0].copy_from_slice(&recon[0]);
        let mut best_uv_sse = plane_sse(&recon[1], &src[1]) + plane_sse(&recon[2], &src[2]);
        const UV_PRI: [i32; 6] = [0, 1, 2, 3, 4, 6];
        for &uv_pri in &UV_PRI {
            for &uv_sec in &SEC {
                if uv_pri == 0 && uv_sec == 0 {
                    continue;
                }
                scratch[1].copy_from_slice(&recon[1]);
                scratch[2].copy_from_slice(&recon[2]);
                let p = CdefParams { damping, y_pri: 0, y_sec: 0, uv_pri, uv_sec };
                apply_cdef(&mut scratch, w, h, cw, ch, ss_hor, ss_ver, mono, skip8, sb8w, &p, bd);
                let sse = plane_sse(&scratch[1], &src[1]) + plane_sse(&scratch[2], &src[2]);
                if sse < best_uv_sse {
                    best_uv_sse = sse;
                    best_uv = (uv_pri, uv_sec);
                }
            }
        }
    }

    CdefParams {
        damping,
        y_pri: best_y.0,
        y_sec: best_y.1,
        uv_pri: best_uv.0,
        uv_sec: best_uv.1,
    }
}
