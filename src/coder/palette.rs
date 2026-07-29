use crate::util::dirty_log2f;

#[derive(Clone, Debug, PartialEq, Eq)]
struct LossyLumaPalette {
    colors: Vec<i32>,
    map: Vec<u8>,
    packed_map: Vec<u8>,
    width: usize,
    height: usize,
    /// Candidate family: `false` = quantile-init Lloyd, `true` = the n most
    /// frequent values used directly (aom/SVT "dominant colors"). Encoded in
    /// the replay sel as `colors.len() + 8*top` so the wavefront re-derives
    /// the exact candidate.
    top: bool,
}

fn palette_y_mode_cdfs() -> Vec<Vec<Vec<u16>>> {
    static RAW: [[u16; 3]; 7] = [
        [31676, 3419, 1261],
        [31912, 2859, 980],
        [31823, 3400, 781],
        [32030, 3561, 904],
        [32309, 7337, 1462],
        [32265, 4015, 1521],
        [32450, 7946, 129],
    ];
    RAW.iter()
        .map(|r| r.iter().map(|&v| icdf(&[v])).collect())
        .collect()
}

fn palette_y_size_cdfs() -> Vec<Vec<u16>> {
    static RAW: [[u16; 6]; 7] = [
        [7952, 13000, 18149, 21478, 25527, 29241],
        [7139, 11421, 16195, 19544, 23666, 28073],
        [7788, 12741, 17325, 20500, 24315, 28530],
        [8271, 14064, 18246, 21564, 25071, 28533],
        [12725, 19180, 21863, 24839, 27535, 30120],
        [9711, 14888, 16923, 21052, 25661, 27875],
        [14940, 20797, 21678, 24186, 27033, 28999],
    ];
    RAW.iter().map(|r| icdf(r)).collect()
}

fn palette_y_color_raw(size: usize, ctx: usize) -> &'static [u16] {
    match (size, ctx) {
        (2, 0) => &[28710],
        (2, 1) => &[16384],
        (2, 2) => &[10553],
        (2, 3) => &[27036],
        (2, 4) => &[31603],
        (3, 0) => &[27877, 30490],
        (3, 1) => &[11532, 25697],
        (3, 2) => &[6544, 30234],
        (3, 3) => &[23018, 28072],
        (3, 4) => &[31915, 32385],
        (4, 0) => &[25572, 28046, 30045],
        (4, 1) => &[9478, 21590, 27256],
        (4, 2) => &[7248, 26837, 29824],
        (4, 3) => &[19167, 24486, 28349],
        (4, 4) => &[31400, 31825, 32250],
        (5, 0) => &[24779, 26955, 28576, 30282],
        (5, 1) => &[8669, 20364, 24073, 28093],
        (5, 2) => &[4255, 27565, 29377, 31067],
        (5, 3) => &[19864, 23674, 26716, 29530],
        (5, 4) => &[31646, 31893, 32147, 32426],
        (6, 0) => &[23132, 25407, 26970, 28435, 30073],
        (6, 1) => &[7443, 17242, 20717, 24762, 27982],
        (6, 2) => &[6300, 24862, 26944, 28784, 30671],
        (6, 3) => &[18916, 22895, 25267, 27435, 29652],
        (6, 4) => &[31270, 31550, 31808, 32059, 32353],
        (7, 0) => &[23105, 25199, 26464, 27684, 28931, 30318],
        (7, 1) => &[6950, 15447, 18952, 22681, 25567, 28563],
        (7, 2) => &[7560, 23474, 25490, 27203, 28921, 30708],
        (7, 3) => &[18544, 22373, 24457, 26195, 28119, 30045],
        (7, 4) => &[31198, 31451, 31670, 31882, 32123, 32391],
        (8, 0) => &[21689, 23883, 25163, 26352, 27506, 28827, 30195],
        (8, 1) => &[6892, 15385, 17840, 21606, 24287, 26753, 29204],
        (8, 2) => &[5651, 23182, 25042, 26518, 27982, 29392, 30900],
        (8, 3) => &[19349, 22578, 24418, 25994, 27524, 29031, 30448],
        (8, 4) => &[31028, 31270, 31504, 31705, 31927, 32153, 32392],
        _ => unreachable!(),
    }
}

fn palette_y_color_cdfs() -> Vec<Vec<Vec<u16>>> {
    (2..=8)
        .map(|n| {
            (0..5)
                .map(|ctx| icdf(palette_y_color_raw(n, ctx)))
                .collect()
        })
        .collect()
}

fn palette_uv_size_cdfs() -> Vec<Vec<u16>> {
    // dav1d default pal_sz[1] (UV), 7 bsize contexts x 6-boundary CDFs.
    const RAW: [[u16; 6]; 7] = [
        [8713, 19979, 27128, 29609, 31331, 32272],
        [5839, 15573, 23581, 26947, 29848, 31700],
        [4426, 11260, 17999, 21483, 25863, 29430],
        [3228, 9464, 14993, 18089, 22523, 27420],
        [3768, 8886, 13091, 17852, 22495, 27207],
        [2464, 8451, 12861, 21632, 25525, 28555],
        [1269, 5435, 10433, 18963, 21700, 25865],
    ];
    RAW.iter().map(|r| icdf(r)).collect()
}

fn palette_uv_color_raw(size: usize, ctx: usize) -> &'static [u16] {
    // dav1d default color_map[1] (UV index map), [size-2][5 ctx].
    static S2: [[u16; 1]; 5] = [[29089], [16384], [8713], [29257], [31610]];
    static S3: [[u16; 2]; 5] = [
        [25257, 29145],
        [12287, 27293],
        [7033, 27960],
        [20145, 25405],
        [30608, 31639],
    ];
    static S4: [[u16; 3]; 5] = [
        [24210, 27175, 29903],
        [9888, 22386, 27214],
        [5901, 26053, 29293],
        [18318, 22152, 28333],
        [30459, 31136, 31926],
    ];
    static S5: [[u16; 4]; 5] = [
        [22980, 25479, 27781, 29986],
        [8413, 21408, 24859, 28874],
        [2257, 29449, 30594, 31598],
        [19189, 21202, 25915, 28620],
        [31844, 32044, 32281, 32518],
    ];
    static S6: [[u16; 5]; 5] = [
        [22217, 24567, 26637, 28683, 30548],
        [7307, 16406, 19636, 24632, 28424],
        [4441, 25064, 26879, 28942, 30919],
        [17210, 20528, 23319, 26750, 29582],
        [30674, 30953, 31396, 31735, 32207],
    ];
    static S7: [[u16; 6]; 5] = [
        [21239, 23168, 25044, 26962, 28705, 30506],
        [6545, 15012, 18004, 21817, 25503, 28701],
        [3448, 26295, 27437, 28704, 30126, 31442],
        [15889, 18323, 21704, 24698, 26976, 29690],
        [30988, 31204, 31479, 31734, 31983, 32325],
    ];
    static S8: [[u16; 7]; 5] = [
        [21442, 23288, 24758, 26246, 27649, 28980, 30563],
        [5863, 14933, 17552, 20668, 23683, 26411, 29273],
        [3415, 25810, 26877, 27990, 29223, 30394, 31618],
        [17965, 20084, 22232, 23974, 26274, 28402, 30390],
        [31190, 31329, 31516, 31679, 31825, 32026, 32322],
    ];
    match size {
        2 => &S2[ctx],
        3 => &S3[ctx],
        4 => &S4[ctx],
        5 => &S5[ctx],
        6 => &S6[ctx],
        7 => &S7[ctx],
        _ => &S8[ctx],
    }
}

fn palette_uv_color_cdfs() -> Vec<Vec<Vec<u16>>> {
    (2..=8usize)
        .map(|size| {
            (0..5)
                .map(|ctx| icdf(palette_uv_color_raw(size, ctx)))
                .collect()
        })
        .collect()
}

/// A chroma palette for one block: per-plane color lists of equal size (the
/// shared index map selects (u\[i\], v\[i\]) pairs). Pairs sorted by (u, v),
/// so the U list is non-decreasing as the U coding scheme requires.
#[derive(Clone)]
struct LossyUvPalette {
    u: Vec<i32>,
    v: Vec<i32>,
    map: Vec<u8>,
    width: usize,
    height: usize,
    /// Candidate family, mirroring [`LossyLumaPalette::top`]: `true` = the n
    /// most frequent (U, V) pairs used directly. Sel-encoded as `n + 8*top`.
    top: bool,
}

/// Exact chroma palette for a `w`x`h` chroma block at (cx, cy): succeeds iff
/// the block holds 2..=8 distinct (U, V) sample PAIRS — the palette then
/// reproduces both planes exactly. 4:4:4 scope (chroma dims == luma dims).
fn exact_uv_palette(
    src_u: &[u16],
    src_v: &[u16],
    stride: usize,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
) -> Option<LossyUvPalette> {
    // At most 8 distinct pairs by definition of the exact path, so the
    // candidate list is inline rather than a per-call allocation.
    let mut pairs = FixedList::<(i32, i32), 8>::new((0, 0));
    for (u_row, v_row) in src_u
        .chunks_exact(stride)
        .skip(cy)
        .take(h)
        .zip(src_v.chunks_exact(stride).skip(cy).take(h))
    {
        let u_slice = &u_row[cx..cx + w];
        let v_slice = &v_row[cx..cx + w];

        for (&u, &v) in u_slice.iter().zip(v_slice.iter()) {
            let p = (u as i32, v as i32);
            if !pairs.contains(&p) {
                if pairs.len() == 8 {
                    return None;
                }
                pairs.push(p);
            }
        }
    }
    if pairs.len() < 2 {
        return None;
    }
    pairs.as_mut_slice().sort_unstable();
    let mut map = vec![0u8; w * h];

    for ((u_row, v_row), out_row) in src_u
        .chunks_exact(stride)
        .skip(cy)
        .take(h)
        .zip(src_v.chunks_exact(stride).skip(cy).take(h))
        .zip(map.chunks_exact_mut(w))
    {
        let u_slice = &u_row[cx..cx + w];
        let v_slice = &v_row[cx..cx + w];

        for ((&u, &v), out) in u_slice.iter().zip(v_slice.iter()).zip(out_row.iter_mut()) {
            let p = (u as i32, v as i32);
            *out = pairs.iter().position(|&q| q == p).unwrap_or(0) as u8;
        }
    }
    Some(LossyUvPalette {
        u: pairs.iter().map(|&(u, _)| u).collect(),
        v: pairs.iter().map(|&(_, v)| v).collect(),
        map,
        width: w,
        height: h,
        top: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn lossy_uv_palette(
    kmeans: &crate::kmeans::KmeansDispatch,
    src_u: &[u16],
    src_v: &[u16],
    stride: usize,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    colors: usize,
    top: bool,
) -> Option<LossyUvPalette> {
    debug_assert!((2..=8).contains(&colors));
    // Pair histogram, lex-sorted.
    let n = w * h;
    let mut centers = LOSSY_UV_SCRATCH.with_borrow_mut(|scratch| {
        let LossyUvScratch { all, hist, idx } = scratch;
        all.clear();
        if all.len() != w * h {
            all.resize(w * h, (0i32, 0i32));
        }

        for ((u_chunk, v_chunk), out_row) in src_u
            .chunks_exact(stride)
            .skip(cy)
            .take(h)
            .zip(src_v.chunks_exact(stride).skip(cy).take(h))
            .zip(all.chunks_exact_mut(w))
        {
            for ((&u, &v), out) in u_chunk[cx..cx + w]
                .iter()
                .zip(v_chunk[cx..cx + w].iter())
                .zip(out_row.iter_mut())
            {
                *out = (u as i32, v as i32);
            }
        }
        all.sort_unstable();
        hist.clear();
        for &p in all.iter() {
            match hist.last_mut() {
                Some((q, c)) if *q == p => *c += 1,
                _ => hist.push((p, 1)),
            }
        }
        if hist.len() < 2 || colors > hist.len() {
            return None;
        }
        Some(if colors == hist.len() {
            hist.iter().map(|&(p, _)| p).collect::<Vec<(i32, i32)>>()
        } else if top {
            // Dominant-pairs family (see `top_palette_centers`): the most
            // frequent (U, V) pairs directly, tie-break (count desc, pair lex
            // asc).
            idx.clear();
            idx.extend(0..hist.len());
            idx.sort_unstable_by_key(|&i| (core::cmp::Reverse(hist[i].1), hist[i].0));
            idx[..colors].iter().map(|&i| hist[i].0).collect()
        } else {
            // Weighted-quantile init along the lex order.
            let total: u32 = hist.iter().map(|&(_, c)| c).sum();
            let mut init: Vec<(i32, i32)> = (0..colors)
                .map(|i| {
                    let target = ((2 * i + 1) * total as usize / (2 * colors)) as u32;
                    let mut acc = 0u32;
                    for &(p, c) in hist.iter() {
                        acc += c;
                        if acc > target {
                            return p;
                        }
                    }
                    hist.last().unwrap().0
                })
                .collect();
            init.dedup();
            if init.len() != colors {
                return None;
            }
            for _ in 0..8 {
                let mut su = [0i64; 8];
                let mut sv = [0i64; 8];
                let mut cnt = [0i64; 8];
                for &((u, v), c) in hist.iter() {
                    let best = init
                        .iter()
                        .enumerate()
                        .min_by_key(|&(i, &(uu, vv))| {
                            let (du, dv) = ((u - uu) as i64, (v - vv) as i64);
                            (du * du + dv * dv, i)
                        })
                        .unwrap()
                        .0;
                    su[best] += i64::from(u) * i64::from(c);
                    sv[best] += i64::from(v) * i64::from(c);
                    cnt[best] += i64::from(c);
                }
                for (((&su_i, &sv_i), &cnt_i), init_i) in su
                    .iter()
                    .zip(sv.iter())
                    .zip(cnt.iter())
                    .zip(init.iter_mut())
                {
                    if cnt_i != 0 {
                        *init_i = (
                            ((su_i + cnt_i / 2) / cnt_i) as i32,
                            ((sv_i + cnt_i / 2) / cnt_i) as i32,
                        );
                    }
                }
                init.sort_unstable();
                init.dedup();
                if init.len() != colors {
                    return None;
                }
            }
            init
        })
    })?;
    centers.sort_unstable();
    let mut map = vec![0u8; n];
    (kmeans.uv_nearest_indices)(
        src_u, src_v, stride, cx, cy, w, h, &centers, &mut map,
    );
    Some(LossyUvPalette {
        u: centers.iter().map(|&(u, _)| u).collect(),
        v: centers.iter().map(|&(_, v)| v).collect(),
        map,
        width: w,
        height: h,
        top,
    })
}

/// Re-derive the captured UV palette during wavefront replay: the capture
/// tried exact first, then lossy — so an exact palette exists iff the capture
/// used it (and then its size equals `k`); otherwise the lossy clustering is
/// deterministic and reproduces the recorded candidate from `k` alone.
#[allow(clippy::too_many_arguments)]
fn uv_palette_rederive(
    kmeans: &crate::kmeans::KmeansDispatch,
    src_u: &[u16],
    src_v: &[u16],
    stride: usize,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    sel: usize,
) -> LossyUvPalette {
    let top = sel > 8;
    let k = if top { sel - 8 } else { sel };
    if !top && let Some(up) = exact_uv_palette(src_u, src_v, stride, cx, cy, w, h) {
        debug_assert_eq!(up.u.len(), k);
        return up;
    }
    lossy_uv_palette(kmeans, src_u, src_v, stride, cx, cy, w, h, k, top)
        .expect("uv palette replay: lossy re-derivation failed")
}

#[inline]
fn palette_bsize_ctx(width: usize, height: usize) -> usize {
    (width.trailing_zeros() as usize + height.trailing_zeros() as usize - 6).min(6)
}

fn palette_cache(above: &[i32], left: &[i32], allow_above: bool) -> FixedList<i32, 16> {
    let mut cache = FixedList::new(0);
    if allow_above {
        for &color in above {
            cache.push(color);
        }
    }
    for &color in left {
        cache.push(color);
    }
    cache.as_mut_slice().sort_unstable();
    cache.dedup();
    cache
}

fn palette_color_ctx(map: &[u8], stride: usize, y: usize, x: usize, size: usize) -> (usize, usize) {
    // Specialized rewrite of the hottest palette function (runs once per map
    // entry per rate evaluation AND per emitted map; ~13% of encode self-time
    // on photographic content via the partition proxy). Only three neighbors
    // (left w=2, above w=2, above-left w=1) ever score, so the generic
    // insertion sort collapses to five equality cases; the trailing
    // "remaining colors ascending" scan collapses to a popcount rank.
    // Bit-for-bit identical to the generic version (see
    // `palette_color_ctx_matches_generic`).
    let current = map[y * stride + x] as usize;
    let (n, ctx, r0, r1, r2);
    if x > 0 && y > 0 {
        let l = map[y * stride + x - 1] as usize;
        let u = map[(y - 1) * stride + x] as usize;
        let ul = map[(y - 1) * stride + x - 1] as usize;
        if l == u {
            if l == ul {
                // one color, score 5: hash = 5*1
                n = 1;
                ctx = 4;
                (r0, r1, r2) = (l, 0, 0);
            } else {
                // scores 4 and 1: hash = 4*1 + 1*2
                n = 2;
                ctx = 3;
                (r0, r1, r2) = (l, ul, 0);
            }
        } else if l == ul {
            // scores l=3, u=2: hash = 3*1 + 2*2
            n = 2;
            ctx = 2;
            (r0, r1, r2) = (l, u, 0);
        } else if u == ul {
            // scores u=3, l=2: hash = 3*1 + 2*2
            n = 2;
            ctx = 2;
            (r0, r1, r2) = (u, l, 0);
        } else {
            // scores 2, 2, 1: ties rank by index asc: hash = 2*1 + 2*2 + 1*2
            n = 3;
            ctx = 1;
            (r0, r1, r2) = (l.min(u), l.max(u), ul);
        }
    } else if x > 0 || y > 0 {
        // Exactly one scored neighbor (left on the top row, above on the
        // first column); the first map entry (0,0) never reaches here.
        let v = if x > 0 {
            map[y * stride + x - 1] as usize
        } else {
            map[(y - 1) * stride + x] as usize
        };
        n = 1;
        ctx = 0;
        (r0, r1, r2) = (v, 0, 0);
    } else {
        debug_assert!(false, "map entry (0,0) is coded uniformly, never ctx'd");
        return (0, current);
    }
    // symbol = current's rank: scored colors first (r0..rn), then the
    // remaining colors in ascending index order.
    let symbol = if current == r0 {
        0
    } else if n >= 2 && current == r1 {
        1
    } else if n >= 3 && current == r2 {
        2
    } else {
        let mut scored_below = usize::from(r0 < current);
        if n >= 2 {
            scored_below += usize::from(r1 < current);
        }
        if n >= 3 {
            scored_below += usize::from(r2 < current);
        }
        n + current - scored_below
    };
    debug_assert!(symbol < size);
    let _ = size;
    (ctx, symbol)
}

#[cfg(test)]
fn palette_color_ctx_generic(
    map: &[u8],
    stride: usize,
    y: usize,
    x: usize,
    size: usize,
) -> (usize, usize) {
    let current = map[y * stride + x] as usize;
    let mut scores = [0u8; 8];
    if x > 0 {
        scores[map[y * stride + x - 1] as usize] += 2;
    }
    if y > 0 {
        scores[map[(y - 1) * stride + x] as usize] += 2;
    }
    if x > 0 && y > 0 {
        scores[map[(y - 1) * stride + x - 1] as usize] += 1;
    }
    let mut order = [0u8; 8];
    let mut n = 0usize;
    for (c, &score) in scores[..size].iter().enumerate() {
        if score == 0 {
            continue;
        }
        let mut i = n;
        while i > 0 && scores[order[i - 1] as usize] < score {
            order[i] = order[i - 1];
            i -= 1;
        }
        order[i] = c as u8;
        n += 1;
    }
    let ctx = if x == 0 || y == 0 {
        0
    } else {
        static MULT: [usize; 3] = [1, 2, 2];
        let mut hash = 0usize;
        for i in 0..n.min(3) {
            hash += scores[order[i] as usize] as usize * MULT[i];
        }
        9 - hash
    };
    for c in 0..size {
        if !order[..n].contains(&(c as u8)) {
            order[n] = c as u8;
            n += 1;
        }
    }
    let symbol = order[..n]
        .iter()
        .position(|&c| c as usize == current)
        .unwrap();
    (ctx, symbol)
}

struct PaletteHistogramScratch {
    seen: Box<[u64; 64]>,
    counts: Box<[u32; 4096]>,
}

impl PaletteHistogramScratch {
    fn new() -> Self {
        Self {
            seen: Box::new([0; 64]),
            counts: Box::new([0; 4096]),
        }
    }

    /// Restore the zero invariant by clearing only values touched by the last
    /// block, rather than zeroing the complete 16 KiB count table per call.
    fn clear_touched(&mut self) {
        for (word, mask) in self.seen.iter_mut().enumerate() {
            let mut bits = *mask;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                self.counts[(word << 6) | bit] = 0;
                bits &= bits - 1;
            }
            *mask = 0;
        }
    }
}

thread_local! {
    /// `block_color_histogram` is a free helper used by decision and replay
    /// paths, so it has no natural `LossyTile` owner. One lazily allocated
    /// table per worker thread avoids both a large repeated stack frame and
    /// global synchronization.
    static PALETTE_HISTOGRAM_SCRATCH: std::cell::RefCell<PaletteHistogramScratch> =
        std::cell::RefCell::new(PaletteHistogramScratch::new());
}

/// Reusable buffers for [`lossy_uv_palette`]. Like `PALETTE_HISTOGRAM_SCRATCH`
/// this is a free helper with no `LossyTile` owner, and it runs once per
/// candidate color count per block, so the pair list and its run-length
/// histogram would otherwise be two fresh allocations (plus the sort scratch)
/// on every call. Capacity is retained across calls; the contents never are.
#[derive(Default)]
struct LossyUvScratch {
    all: Vec<(i32, i32)>,
    hist: Vec<((i32, i32), u32)>,
    idx: Vec<usize>,
}

thread_local! {
    static LOSSY_UV_SCRATCH: std::cell::RefCell<LossyUvScratch> =
        std::cell::RefCell::new(LossyUvScratch::default());
}

/// Distinct-value histogram of a `w`x`h` luma block as sorted (value, count)
/// pairs, or None when the block is flat (<2) or too color-rich (>64 — the
/// libaom palette admission gate). O(N) bitset scan + count table; this is the
/// fast gate that keeps photographic blocks nearly free.
fn block_color_histogram(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) -> Option<FixedList<(i32, u32), 64>> {
    PALETTE_HISTOGRAM_SCRATCH.with_borrow_mut(|scratch| {
        debug_assert!(scratch.seen.iter().all(|&word| word == 0));
        let mut distinct = 0usize;
        for y in 0..h {
            for &s in &src[(py + y) * stride + px..][..w] {
                let v = s as usize;
                let (word, bit) = (v >> 6, v & 63);
                if scratch.seen[word] & (1u64 << bit) == 0 {
                    scratch.seen[word] |= 1u64 << bit;
                    distinct += 1;
                    if distinct > 64 {
                        scratch.clear_touched();
                        return None;
                    }
                }
                scratch.counts[v] += 1;
            }
        }
        if distinct < 2 {
            scratch.clear_touched();
            return None;
        }
        let mut hist = FixedList::new((0, 0));
        for (word, &mask) in scratch.seen.iter().enumerate() {
            let mut bits = mask;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let v = (word << 6) | bit;
                hist.push((v as i32, scratch.counts[v]));
                bits &= bits - 1;
            }
        }
        scratch.clear_touched();
        Some(hist)
    })
}

/// Deterministic weighted k-means over a color histogram: identical result to
/// per-sample k-means (duplicate samples assign identically) at O(distinct)
/// instead of O(N) per iteration. Exact shortcut: `colors == hist.len()`
/// returns the distinct values themselves (the palette reproduces the source
/// exactly). Returns sorted distinct centers, or None on center collapse.
/// aom/SVT "dominant colors" family: the `colors` most frequent sample values
/// used DIRECTLY as the palette, no Lloyd refinement. On peaked histograms
/// (screen/synthetic content) most pixels then match their palette entry
/// exactly and the residual carries only the outliers — centroids averaged by
/// Lloyd land between the peaks and everything rings. Tie-break (count desc,
/// value asc) and the ascending final sort keep it a pure function of the
/// histogram, as wavefront replay requires.
fn top_palette_centers(hist: &[(i32, u32)], colors: usize) -> Option<FixedList<i32, 8>> {
    debug_assert!((2..=8).contains(&colors));
    // == case: identical to the exact path of `palette_centers`; skip the dup.
    if colors >= hist.len() {
        return None;
    }
    let mut idx = FixedList::<usize, 64>::new(0);
    for i in 0..hist.len() {
        idx.push(i);
    }
    idx.as_mut_slice()
        .sort_unstable_by_key(|&i| (core::cmp::Reverse(hist[i].1), hist[i].0));
    let mut centers = FixedList::new(0);
    for &i in &idx[..colors] {
        centers.push(hist[i].0);
    }
    centers.as_mut_slice().sort_unstable();
    Some(centers)
}

fn palette_centers(hist: &[(i32, u32)], colors: usize) -> Option<FixedList<i32, 8>> {
    debug_assert!((2..=8).contains(&colors));
    if colors > hist.len() {
        return None;
    }
    if colors == hist.len() {
        let mut centers = FixedList::new(0);
        for &(value, _) in hist {
            centers.push(value);
        }
        return Some(centers);
    }
    let total: u32 = hist.iter().map(|&(_, c)| c).sum();
    // Weighted-quantile init == the old sorted-with-duplicates indexing.
    let mut centers = FixedList::new(0);
    for i in 0..colors {
        let target = ((2 * i + 1) * total as usize / (2 * colors)) as u32;
        let mut acc = 0u32;
        let mut center = hist.last().unwrap().0;
        for &(v, c) in hist {
            acc += c;
            if acc > target {
                center = v;
                break;
            }
        }
        centers.push(center);
    }
    centers.dedup();
    if centers.len() != colors {
        return None;
    }
    for _ in 0..8 {
        let previous = centers;
        let mut sums = [0i64; 8];
        let mut cnts = [0i64; 8];
        for &(v, c) in hist {
            let best = nearest_center_idx(&centers, v);
            sums[best] += i64::from(v) * i64::from(c);
            cnts[best] += i64::from(c);
        }
        for i in 0..colors {
            if cnts[i] != 0 {
                centers[i] = ((sums[i] + cnts[i] / 2) / cnts[i]) as i32;
            }
        }
        centers.as_mut_slice().sort_unstable();
        centers.dedup();
        if centers.len() != colors {
            return None;
        }
        // Lloyd has reached a fixed point. Further iterations are guaranteed
        // to repeat the same assignments and centers, so stopping here is
        // result-identical and avoids the common 8x redundant tail.
        if centers == previous {
            break;
        }
    }
    Some(centers)
}

/// Palette candidate for a `w`x`h` luma block with `colors` entries: centers
/// via [`palette_centers`], plus the per-sample index map. Deterministic (the
/// wavefront replay re-derives it from (position, colors) alone).
#[allow(clippy::too_many_arguments)]
fn lossy_luma_palette(
    kmeans: &KmeansDispatch,
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    sel: usize,
) -> Option<LossyLumaPalette> {
    let hist = block_color_histogram(src, stride, px, py, w, h)?;
    let top = sel > 8;
    let colors = if top { sel - 8 } else { sel };
    lossy_luma_palette_from(kmeans, &hist, src, stride, px, py, w, h, colors, top)
}

#[inline]
fn nearest_center_idx(centers: &[i32], v: i32) -> usize {
    let idx = centers.partition_point(|&c| c < v);
    if idx == 0 {
        return 0;
    }
    if idx == centers.len() {
        return idx - 1;
    }
    if v - centers[idx - 1] <= centers[idx] - v {
        idx - 1
    } else {
        idx
    }
}

#[allow(clippy::too_many_arguments)]
fn lossy_luma_palette_from(
    kmeans: &crate::kmeans::KmeansDispatch,
    hist: &[(i32, u32)],
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    colors: usize,
    top: bool,
) -> Option<LossyLumaPalette> {
    let centers = if top {
        top_palette_centers(hist, colors)?
    } else {
        palette_centers(hist, colors)?
    };
    Some(lossy_luma_palette_from_centers(
        kmeans,
        src,
        stride,
        px,
        py,
        w,
        h,
        centers.as_slice().to_vec(),
        top,
    ))
}

#[allow(clippy::too_many_arguments)]
fn lossy_luma_palette_from_centers(
    kmeans: &crate::kmeans::KmeansDispatch,
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    centers: Vec<i32>,
    top: bool,
) -> LossyLumaPalette {
    let n = w * h;
    let mut map = vec![0u8; n];
    (kmeans.luma_nearest_indices)(src, stride, px, py, w, h, &centers, &mut map);
    let mut packed_map = vec![0u8; n.div_ceil(2)];
    for (p, &idx) in map.iter().enumerate() {
        if p & 1 == 0 {
            packed_map[p / 2] = idx;
        } else {
            packed_map[p / 2] |= idx << 4;
        }
    }
    LossyLumaPalette {
        colors: centers,
        map,
        packed_map,
        width: w,
        height: h,
        top,
    }
}

impl<'a> LossyTile<'a> {
    /// Build every legal palette center set, rank it with histogram-domain
    /// prediction error plus a compact entropy/header model, then construct
    /// maps and run exact syntax/transform RD only for the finalists.
    #[allow(clippy::too_many_arguments)]
    fn rank_luma_palette_candidates<const N: usize>(
        &self,
        hist: &[(i32, u32)],
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        mlam: f32,
    ) -> Vec<(LossyLumaPalette, [i32; N])> {
        debug_assert_eq!(N, w * h);
        let mut ranked = FixedList::<(f32, usize, FixedList<i32, 8>, bool), 14>::new((
            f32::INFINITY,
            0,
            FixedList::new(0),
            false,
        ));
        // Every dominant-color candidate uses a prefix of the same
        // frequency ordering. Build that ordering once instead of sorting the
        // histogram independently for palette sizes 2 through 8.
        let mut top_order = FixedList::<usize, 64>::new(0);
        for i in 0..hist.len() {
            top_order.push(i);
        }
        top_order
            .as_mut_slice()
            .sort_unstable_by_key(|&i| (core::cmp::Reverse(hist[i].1), hist[i].0));
        for (order, (n, top)) in (2..=8).flat_map(|n| [(n, false), (n, true)]).enumerate() {
            let Some(centers) = (if top {
                if n >= hist.len() {
                    None
                } else {
                    let mut centers = FixedList::new(0);
                    for &i in &top_order[..n] {
                        centers.push(hist[i].0);
                    }
                    centers.as_mut_slice().sort_unstable();
                    Some(centers)
                }
            } else {
                palette_centers(hist, n)
            }) else {
                continue;
            };
            // Histogram-domain model: exact palette prediction SSE plus a
            // compact entropy/header estimate, without constructing fourteen
            // per-pixel maps and running the context rate coder for all of
            // them. Full syntax/RD is still used for every finalist below.
            let mut counts = [0u32; 8];
            let mut sse = 0i64;
            let mut samples = 0u32;
            for &(v, count) in hist {
                let idx = nearest_center_idx(&centers, v);
                let d = v - centers[idx];
                sse += i64::from(d * d) * i64::from(count);
                counts[idx] += count;
                samples += count;
            }
            let entropy_bits = counts[..centers.len()]
                .iter()
                .filter(|&&count| count != 0)
                .map(|&count| {
                    let p = count as f32 / samples as f32;
                    -(count as f32) * dirty_log2f(p)
                })
                .sum::<f32>();
            let model_bits = centers.len() as f32 * self.bd as f32 + entropy_bits * 0.5 + 8.0;
            ranked.push((rd_cost_i64(sse, mlam, model_bits), order, centers, top));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        ranked
            .iter()
            .take(self.speed.palette_refine_budget())
            .map(|&(_, _, centers, top)| {
                let palette = lossy_luma_palette_from_centers(
                    &self.kmeans,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                    centers.as_slice().to_vec(),
                    top,
                );
                let mut pred = [0i32; N];
                palette_pred(&mut pred, w, &palette.colors, &palette.packed_map, w, h);
                (palette, pred)
            })
            .collect()
    }

    fn palette_rate_bits(&self, px: usize, py: usize, p: &LossyLumaPalette) -> f32 {
        let (bx4, by4) = (px / 4, py / 4);
        let c = self.dcdf();
        let bctx = palette_bsize_ctx(p.width, p.height);
        // Cache-AWARE: a_palette/l_palette are now carried through the
        // wavefront capture's ctx handoff (packed 9xi32 planes, coder.rs), so
        // reading them here prices identically under -t1 and -tN. The color
        // cache credit matters most exactly where palette matters — dense
        // same-palette regions (screen content) — where the context-free
        // model overpriced every block's colors as new.
        let mctx = usize::from(!self.a_palette[bx4].is_empty())
            + usize::from(!self.l_palette[by4].is_empty());
        let mut bits = cdf_cost(&c.palette_y_mode[bctx][mctx], 1)
            + cdf_cost(&c.palette_y_size[bctx], p.colors.len() - 2);
        let cache = palette_cache(
            &self.a_palette[bx4],
            &self.l_palette[by4],
            !py.is_multiple_of(64),
        );
        let mut found = 0;
        for &cv in &cache {
            if found == p.colors.len() {
                break;
            }
            bits += 1.0; // reuse flag, p = 1/2 bypass
            found += usize::from(p.colors.binary_search(&cv).is_ok());
        }
        let mut out = FixedList::<u32, 8>::new(0);
        for &color in &p.colors {
            if cache.binary_search(&color).is_err() {
                out.push(color as u32);
            }
        }
        if let Some(&first) = out.first() {
            let _ = first;
            bits += self.bd as f32;
            if out.len() > 1 {
                let max_delta = out.array_windows::<2>().map(|w| w[1] - w[0]).max().unwrap();
                let min_bits = self.bd - 3;
                let mut nb = (32 - (max_delta - 1).leading_zeros()) as u8;
                nb = nb.max(min_bits);
                bits += 2.0;
                let mut range = (1u32 << self.bd) - out[0] - 1;
                for w in out.array_windows::<2>() {
                    let delta = w[1] - w[0];
                    bits += nb as f32;
                    range -= delta;
                    let rb = if range <= 1 {
                        0
                    } else {
                        (32 - (range - 1).leading_zeros()) as u8
                    };
                    nb = nb.min(rb);
                }
            }
        }
        // Index map: first sample uniform, rest context-coded on the diagonal.
        let ns = p.colors.len();
        bits += dirty_log2f(ns as f32);
        for diagonal in 1..p.width + p.height - 1 {
            let first_x = diagonal.min(p.width - 1);
            let last_x = diagonal.saturating_sub(p.height - 1);
            for x in (last_x..=first_x).rev() {
                let y = diagonal - x;
                let (ctx, symbol) = palette_color_ctx(&p.map, p.width, y, x, ns);
                bits += cdf_cost(&c.palette_y_color[ns - 2][ctx], symbol);
            }
        }
        bits
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_palette_mode_info(
        &mut self,
        px: usize,
        py: usize,
        width: usize,
        height: usize,
        y_mode: usize,
        has_chroma: bool,
        palette: Option<&LossyLumaPalette>,
        uv_palette: Option<&LossyUvPalette>,
    ) {
        let (bx4, by4) = (px / 4, py / 4);
        // dav1d gate: `imax(bw4, bh4) <= 16 && bw4 + bh4 >= 4` (4px units) —
        // i.e. max(w, h) <= 64 AND w + h >= 16 px. This ADMITS the H4/V4 strip
        // sizes 16x4 / 4x16 (4+1 units >= 4): the decoder reads y_pal (and
        // uv_pal when the strip has chroma) for them, so the old `< 8` gate
        // silently desynced every quad-split bitstream.
        if width.max(height) > 64 || width + height < 16 {
            let xe = (bx4 + width.div_ceil(4)).min(self.a_palette.len());
            let ye = (by4 + height.div_ceil(4)).min(self.l_palette.len());
            self.a_palette[bx4..xe].fill(Vec::new());
            self.l_palette[by4..ye].fill(Vec::new());
            // The UV state must clear too (the decoder zeroes pal_sz for
            // EVERY block, eligible or not). Leaving it stale poisoned the
            // next UV palette's color cache — invisible while palettes were
            // exact-only (too sparse to ever bracket a 4x4 run), decode-fatal
            // once lossy UV palettes made them dense (x_fractal 444 q80).
            self.a_palette_uv[bx4..xe].fill(Vec::new());
            self.l_palette_uv[by4..ye].fill(Vec::new());
            return;
        }
        let bctx = palette_bsize_ctx(width, height);
        let mctx = usize::from(!self.a_palette[bx4].is_empty())
            + usize::from(!self.l_palette[by4].is_empty());
        if y_mode == DC_PRED {
            self.enc.encode_symbol(
                usize::from(palette.is_some()),
                &mut self.cdfs.palette_y_mode[bctx][mctx],
            );
            if let Some(p) = palette {
                #[cfg(test)]
                if !self.enc.sink {
                    LOSSY_PALETTE_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                self.enc
                    .encode_symbol(p.colors.len() - 2, &mut self.cdfs.palette_y_size[bctx]);
                // AV1 excludes the above palette from the color cache at a
                // superblock-row boundary (the mode context still sees it).
                let cache = palette_cache(
                    &self.a_palette[bx4],
                    &self.l_palette[by4],
                    !py.is_multiple_of(64),
                );
                let mut found = 0;
                for &c in &cache {
                    if found == p.colors.len() {
                        break;
                    }
                    let yes = p.colors.binary_search(&c).is_ok();
                    self.enc.encode_bool(yes, 16384);
                    found += usize::from(yes);
                }
                let mut out = FixedList::<u32, 8>::new(0);
                for &color in &p.colors {
                    if cache.binary_search(&color).is_err() {
                        out.push(color as u32);
                    }
                }
                if let Some(&first) = out.first() {
                    self.enc.encode_literal(first, self.bd as u32);
                    if out.len() > 1 {
                        let max_delta =
                            out.array_windows::<2>().map(|w| w[1] - w[0]).max().unwrap();
                        let min_bits = self.bd - 3;
                        let mut bits = (32 - (max_delta - 1).leading_zeros()) as u8;
                        bits = bits.max(min_bits);
                        self.enc.encode_literal(u32::from(bits - min_bits), 2);
                        let mut range = (1u32 << self.bd) - out[0] - 1;
                        for w in out.array_windows::<2>() {
                            let delta = w[1] - w[0];
                            self.enc.encode_literal(delta - 1, bits as u32);
                            range -= delta;
                            let rb = if range <= 1 {
                                0
                            } else {
                                (32 - (range - 1).leading_zeros()) as u8
                            };
                            bits = bits.min(rb);
                        }
                    }
                }
            }
        }
        if has_chroma && !self.mono && self.a_uv_mode[bx4] as usize == DC_PRED {
            self.enc.encode_symbol(
                usize::from(uv_palette.is_some()),
                &mut self.cdfs.palette_uv_mode[usize::from(palette.is_some())],
            );
            if let Some(up) = uv_palette {
                let bctx = palette_bsize_ctx(width, height);
                self.enc
                    .encode_symbol(up.u.len() - 2, &mut self.cdfs.palette_uv_size[bctx]);
                // U plane: the Y color-coding machinery with a NON-DECREASING
                // list (deltas may be 0, increment +0) and the same cache
                // mechanism, over the U-plane neighbor palettes.
                let cache = palette_cache(
                    &self.a_palette_uv[bx4],
                    &self.l_palette_uv[by4],
                    !py.is_multiple_of(64),
                );
                // The U list may hold DUPLICATES (pairs sharing U). Each
                // cache reuse contributes exactly ONE palette slot, so the
                // "new colors" list is the remaining MULTISET — one instance
                // of every reused cache color removed, duplicates kept.
                // Filtering out all instances of cached values instead
                // starved the decoder (it read the V-plane bits as the
                // missing U colors — volcanic 444 q100 multitile desync).
                let mut rem = FixedList::<i32, 8>::new(0);
                for &color in &up.u {
                    rem.push(color);
                }
                let mut found = 0;
                for &c in &cache {
                    if found == up.u.len() {
                        break;
                    }
                    let yes = up.u.binary_search(&c).is_ok();
                    self.enc.encode_bool(yes, 16384);
                    if yes {
                        found += 1;
                        if let Ok(pos) = rem.binary_search(&c) {
                            rem.remove(pos);
                        }
                    }
                }
                let mut out = FixedList::<u32, 8>::new(0);
                for &color in &rem {
                    out.push(color as u32);
                }
                if let Some(&first) = out.first() {
                    self.enc.encode_literal(first, self.bd as u32);
                    if out.len() > 1 {
                        let max = (1u32 << self.bd) - 1;
                        let max_delta =
                            out.array_windows::<2>().map(|w| w[1] - w[0]).max().unwrap();
                        let min_bits = self.bd - 3;
                        let mut bits = if max_delta == 0 {
                            0
                        } else {
                            (32 - max_delta.leading_zeros()) as u8
                        };
                        bits = bits.max(min_bits).min(self.bd);
                        self.enc.encode_literal(u32::from(bits - min_bits), 2);
                        let mut prev = out[0];
                        for w in out.array_windows::<2>() {
                            let delta = w[1] - w[0];
                            self.enc.encode_literal(delta, bits as u32);
                            prev = (prev + delta).min(max);
                            if prev >= max {
                                // decoder fills the rest with max and stops
                                break;
                            }
                            let ulog2 = 31 - (max - prev).leading_zeros();
                            bits = bits.min(1 + ulog2 as u8);
                        }
                    }
                }
                // V plane: literal mode (leading equiprobable bool = 0, then
                // one bd-bit literal per color) — always legal; the wrapped
                // delta mode is an optimization left for later.
                self.enc.encode_bool(false, 16384);
                for &v in &up.v {
                    self.enc.encode_literal(v as u32, self.bd as u32);
                }
            }
        }
        let stored = palette.map_or_else(Vec::new, |p| p.colors.clone());
        let xe = (bx4 + width / 4).min(self.a_palette.len());
        let ye = (by4 + height / 4).min(self.l_palette.len());
        for v in &mut self.a_palette[bx4..xe] {
            v.clone_from(&stored);
        }
        for v in &mut self.l_palette[by4..ye] {
            v.clone_from(&stored);
        }
        let stored_uv = uv_palette.map_or_else(Vec::new, |p| p.u.clone());
        for v in &mut self.a_palette_uv[bx4..xe] {
            v.clone_from(&stored_uv);
        }
        for v in &mut self.l_palette_uv[by4..ye] {
            v.clone_from(&stored_uv);
        }
    }

    /// Entropy-accurate rate of the UV palette transaction. U colors are
    /// priced as ALL NEW (no cache credit): the U-plane neighbor cache
    /// (a_palette_uv/l_palette_uv) is not in the wavefront ctx handoff, so a
    /// decision model reading it would break -t1/-tN byte identity — the
    /// EMITTER still uses the real cache, so this slightly overprices dense
    /// regions (conservative). V is priced in literal mode, as emitted.
    fn palette_uv_rate_bits(&self, y_pal: bool, up: &LossyUvPalette) -> f32 {
        let c = self.dcdf();
        let bctx = palette_bsize_ctx(up.width, up.height);
        let mut bits = cdf_cost(&c.palette_uv_mode[usize::from(y_pal)], 1)
            + cdf_cost(&c.palette_uv_size[bctx], up.u.len() - 2);
        bits += self.bd as f32; // first U color literal
        if up.u.len() > 1 {
            let max = (1u32 << self.bd) - 1;
            let deltas: Vec<u32> =
                up.u.array_windows::<2>()
                    .map(|w| (w[1] - w[0]) as u32)
                    .collect();
            let max_delta = deltas.iter().copied().max().unwrap();
            let min_bits = self.bd - 3;
            let mut nb = if max_delta == 0 {
                0
            } else {
                (32 - max_delta.leading_zeros()) as u8
            };
            nb = nb.max(min_bits).min(self.bd);
            bits += 2.0;
            let mut prev = up.u[0] as u32;
            for &d in &deltas {
                bits += nb as f32;
                prev = (prev + d).min(max);
                if prev >= max {
                    break;
                }
                let ulog2 = 31 - (max - prev).leading_zeros();
                nb = nb.min(1 + ulog2 as u8);
            }
        }
        bits += 1.0 + (up.v.len() * self.bd as usize) as f32; // V literal mode
        let ns = up.u.len();
        bits += dirty_log2f(ns as f32);
        for diagonal in 1..up.width + up.height - 1 {
            let first_x = diagonal.min(up.width - 1);
            let last_x = diagonal.saturating_sub(up.height - 1);
            for x in (last_x..=first_x).rev() {
                let y = diagonal - x;
                let (ctx, symbol) = palette_color_ctx(&up.map, up.width, y, x, ns);
                bits += cdf_cost(&c.palette_uv_color[ns - 2][ctx], symbol);
            }
        }
        bits
    }

    fn emit_palette_uv_map(&mut self, p: &LossyUvPalette) {
        self.enc.encode_ns(p.map[0] as u32, p.u.len() as u32);
        for diagonal in 1..p.width + p.height - 1 {
            let first_x = diagonal.min(p.width - 1);
            let last_x = diagonal.saturating_sub(p.height - 1);
            for x in (last_x..=first_x).rev() {
                let y = diagonal - x;
                let (ctx, symbol) = palette_color_ctx(&p.map, p.width, y, x, p.u.len());
                self.enc
                    .encode_symbol(symbol, &mut self.cdfs.palette_uv_color[p.u.len() - 2][ctx]);
            }
        }
    }

    fn emit_palette_map(&mut self, p: &LossyLumaPalette) {
        self.enc.encode_ns(p.map[0] as u32, p.colors.len() as u32);
        for diagonal in 1..p.width + p.height - 1 {
            let first_x = diagonal.min(p.width - 1);
            let last_x = diagonal.saturating_sub(p.height - 1);
            for x in (last_x..=first_x).rev() {
                let y = diagonal - x;
                let (ctx, symbol) = palette_color_ctx(&p.map, p.width, y, x, p.colors.len());
                self.enc.encode_symbol(
                    symbol,
                    &mut self.cdfs.palette_y_color[p.colors.len() - 2][ctx],
                );
            }
        }
    }
}

#[cfg(test)]
mod palette_generalization_tests {
    use super::*;

    #[test]
    fn histogram_kmeans_palette_basics() {
        // 16x16 block with 4 distinct values in a 32-wide plane.
        let mut plane = vec![0u16; 32 * 32];
        for y in 0..16 {
            for x in 0..16 {
                plane[(y + 4) * 32 + x + 4] = [10, 80, 150, 220][(x / 4 + y / 4) % 4];
            }
        }
        let h = block_color_histogram(&plane, 32, 4, 4, 16, 16).expect("histogram");
        assert_eq!(h.len(), 4);
        let p = lossy_luma_palette(
            &crate::kmeans::KmeansDispatch::scalar(),
            &plane,
            32,
            4,
            4,
            16,
            16,
            4,
        )
        .expect("exact palette");
        assert_eq!(p.colors, vec![10, 80, 150, 220]);
        let mut pred = vec![0i32; 256];
        palette_pred(&mut pred, 16, &p.colors, &p.packed_map, 16, 16);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pred[y * 16 + x],
                    plane[(y + 4) * 32 + x + 4] as i32,
                    "exact recon"
                );
            }
        }
        // Lossy: 2 colors over 4 values still succeeds.
        assert!(
            lossy_luma_palette(
                &crate::kmeans::KmeansDispatch::scalar(),
                &plane,
                32,
                4,
                4,
                16,
                16,
                2,
            )
            .is_some()
        );
        // Flat block -> None; photo-like block (many distinct) -> None.
        let flat = vec![7u16; 32 * 32];
        assert!(block_color_histogram(&flat, 32, 4, 4, 16, 16).is_none());
        let noisy: Vec<u16> = (0..32 * 32).map(|i| ((i * 37) % 251) as u16).collect();
        assert!(block_color_histogram(&noisy, 32, 0, 0, 16, 16).is_none());
    }
}

#[cfg(test)]
mod ctx_tests {
    use super::*;

    #[test]
    fn nearest_center_matches_min_by_key() {
        // all sorted-distinct center lists over a small domain x all values
        fn rec(centers: &mut Vec<i32>, next: i32, max: i32) {
            if centers.len() >= 2 {
                for v in -1..=max + 1 {
                    let want = centers
                        .iter()
                        .enumerate()
                        .min_by_key(|&(i, &c)| ((v - c).abs(), i))
                        .unwrap()
                        .0;
                    assert_eq!(nearest_center_idx(centers, v), want, "{centers:?} v={v}");
                }
            }
            if centers.len() == 8 {
                return;
            }
            for c in next..=max {
                centers.push(c);
                rec(centers, c + 1, max);
                centers.pop();
            }
        }
        rec(&mut Vec::new(), 0, 9);
    }

    /// The specialized `palette_color_ctx` must match the generic
    /// (score/insertion-sort) implementation bit for bit on every reachable
    /// input: all sizes 2..=8, every neighbor color combination, both edges.
    #[test]
    fn palette_color_ctx_matches_generic() {
        for size in 2..=8usize {
            // Exhaustive 3x3 maps: 9 entries over `size` colors is too many to
            // enumerate fully at size 8, so enumerate the 4 cells that matter
            // (current + 3 neighbors) and fix the rest at 0.
            for cur in 0..size as u8 {
                for l in 0..size as u8 {
                    for u in 0..size as u8 {
                        for ul in 0..size as u8 {
                            // interior pixel (1,1) of a 2-wide map
                            let map = [ul, u, l, cur];
                            assert_eq!(
                                palette_color_ctx(&map, 2, 1, 1, size),
                                palette_color_ctx_generic(&map, 2, 1, 1, size),
                                "interior size={size} cur={cur} l={l} u={u} ul={ul}"
                            );
                            // top row (0,1): left neighbor only
                            let map2 = [l, cur, u, ul];
                            assert_eq!(
                                palette_color_ctx(&map2, 2, 0, 1, size),
                                palette_color_ctx_generic(&map2, 2, 0, 1, size),
                                "toprow size={size}"
                            );
                            // first column (1,0): above neighbor only
                            let map3 = [u, l, cur, ul];
                            assert_eq!(
                                palette_color_ctx(&map3, 2, 1, 0, size),
                                palette_color_ctx_generic(&map3, 2, 1, 0, size),
                                "firstcol size={size}"
                            );
                        }
                    }
                }
            }
        }
    }
}
