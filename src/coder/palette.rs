#[derive(Clone, Debug, PartialEq, Eq)]
struct LossyLumaPalette {
    colors: Vec<i32>,
    map: Vec<u8>,
    packed_map: Vec<u8>,
    width: usize,
    height: usize,
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
    const RAW: [[u16; 6]; 7] = [
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

#[inline]
fn palette_bsize_ctx(width: usize, height: usize) -> usize {
    (width.trailing_zeros() as usize + height.trailing_zeros() as usize - 6).min(6)
}

fn palette_cache(above: &[i32], left: &[i32], allow_above: bool) -> Vec<i32> {
    let mut cache = Vec::with_capacity(16);
    if allow_above {
        cache.extend_from_slice(above);
    }
    cache.extend_from_slice(left);
    cache.sort_unstable();
    cache.dedup();
    cache
}

fn palette_color_ctx(map: &[u8], stride: usize, y: usize, x: usize, size: usize) -> (usize, usize) {
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
    let mut ranked: Vec<usize> = (0..size).filter(|&i| scores[i] != 0).collect();
    ranked.sort_by_key(|&i| (std::cmp::Reverse(scores[i]), i));
    let ctx = if x == 0 || y == 0 {
        0
    } else {
        static MULT: [usize; 3] = [1, 2, 2];
        let hash: usize = ranked
            .iter()
            .zip(MULT)
            .map(|(&c, m)| scores[c] as usize * m)
            .sum();
        9 - hash
    };
    let mut order = ranked;
    for c in 0..size {
        if !order.contains(&c) {
            order.push(c);
        }
    }
    (ctx, order.iter().position(|&c| c == current).unwrap())
}

fn lossy_luma_palette(
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    colors: usize,
) -> Option<LossyLumaPalette> {
    debug_assert!((2..=8).contains(&colors));
    let mut samples = [0i32; 64];
    for y in 0..8 {
        samples[y * 8..y * 8 + 8].copy_from_slice(&src[(py + y) * stride + px..][..8]);
    }
    let mut sorted = samples;
    sorted.sort_unstable();
    if sorted[0] == sorted[63] {
        return None;
    }
    let mut centers: Vec<i32> = (0..colors)
        .map(|i| sorted[((2 * i + 1) * 64 / (2 * colors)).min(63)])
        .collect();
    for _ in 0..8 {
        let mut sums = vec![0i64; colors];
        let mut counts = vec![0i32; colors];
        for &s in &samples {
            let best = centers
                .iter()
                .enumerate()
                .min_by_key(|&(i, &c)| ((s - c).abs(), i))
                .unwrap()
                .0;
            sums[best] += i64::from(s);
            counts[best] += 1;
        }
        for i in 0..colors {
            if counts[i] != 0 {
                centers[i] = ((sums[i] + i64::from(counts[i] / 2)) / i64::from(counts[i])) as i32;
            }
        }
        centers.sort_unstable();
        centers.dedup();
        if centers.len() != colors {
            return None;
        }
    }
    let mut map = Vec::with_capacity(64);
    let mut packed_map = vec![0u8; 32];
    for (p, &s) in samples.iter().enumerate() {
        let idx = centers
            .iter()
            .enumerate()
            .min_by_key(|&(i, &c)| ((s - c).abs(), i))
            .unwrap()
            .0 as u8;
        map.push(idx);
        if p & 1 == 0 {
            packed_map[p / 2] = idx;
        } else {
            packed_map[p / 2] |= idx << 4;
        }
    }
    Some(LossyLumaPalette {
        colors: centers,
        map,
        packed_map,
        width: 8,
        height: 8,
    })
}

fn palette_signal_bits(p: &LossyLumaPalette, bit_depth: u8) -> f32 {
    2.0 + (p.colors.len() * bit_depth as usize) as f32
        + p.map.len() as f32 * (p.colors.len() as f32).log2()
}

impl<'a> LossyTile<'a> {
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
    ) {
        let (bx4, by4) = (px / 4, py / 4);
        if width < 8 || height < 8 || width > 64 || height > 64 {
            let xe = (bx4 + width.div_ceil(4)).min(self.a_palette.len());
            let ye = (by4 + height.div_ceil(4)).min(self.l_palette.len());
            self.a_palette[bx4..xe].fill(Vec::new());
            self.l_palette[by4..ye].fill(Vec::new());
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
                let out: Vec<u32> = p
                    .colors
                    .iter()
                    .filter(|c| cache.binary_search(c).is_err())
                    .map(|&c| c as u32)
                    .collect();
                if let Some(&first) = out.first() {
                    self.enc.encode_literal(first, self.bd as u32);
                    if out.len() > 1 {
                        let max_delta = out.windows(2).map(|w| w[1] - w[0]).max().unwrap();
                        let min_bits = self.bd - 3;
                        let mut bits = (32 - (max_delta - 1).leading_zeros()) as u8;
                        bits = bits.max(min_bits);
                        self.enc.encode_literal(u32::from(bits - min_bits), 2);
                        let mut range = (1u32 << self.bd) - out[0] - 1;
                        for w in out.windows(2) {
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
                0,
                &mut self.cdfs.palette_uv_mode[usize::from(palette.is_some())],
            );
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
