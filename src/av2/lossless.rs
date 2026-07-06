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

pub(crate) static SCAN_4X4: [u8; 16] = [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15];

/// AVM luma palette information for one coding block.  Palette colors are kept
/// sorted, so each map entry is also the index used by the decoder.
#[derive(Clone)]
pub(crate) struct LumaPalette {
    pub(crate) colors: Vec<u16>,
    pub(crate) map: Vec<u8>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

/// Boolean block-neighbor state at MI (4x4) granularity. Palette decisions use
/// direct above/left flags, while intra-BC and skip use AV2's ordered `NPos` /
/// `NPosBuf` walk. Keep one compact reusable grid per syntax element.
pub(crate) struct BlockFlagGrid {
    rows: usize,
    cols: usize,
    cells: Vec<bool>,
}

impl BlockFlagGrid {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            cells: vec![false; rows * cols],
        }
    }

    /// Direct above/left context used by the current palette search policy.
    pub(crate) fn context(&self, row: usize, col: usize) -> usize {
        usize::from(row > 0 && self.cells[(row - 1) * self.cols + col])
            + usize::from(col > 0 && self.cells[row * self.cols + col - 1])
    }

    /// Reproduce AV2's ordered `add_neighbor` walk for `NPos` / `NPosBuf`:
    /// bottom-left, above-right, direct-left, direct-above, stopping after two
    /// valid positions. `buffered_above` selects `NPosBuf`; when false, samples
    /// from the preceding superblock row are excluded as required by `NPos`.
    pub(crate) fn npos_context(
        &self,
        row: usize,
        col: usize,
        block_rows: usize,
        block_cols: usize,
        buffered_above: bool,
    ) -> usize {
        let mut count = 0usize;
        let mut neighbors = 0usize;
        let bottom = block_rows
            .checked_sub(1)
            .and_then(|offset| row.checked_add(offset));
        let right = block_cols
            .checked_sub(1)
            .and_then(|offset| col.checked_add(offset));
        let candidates = [
            (bottom, col.checked_sub(1)),
            (row.checked_sub(1), right),
            (Some(row), col.checked_sub(1)),
            (row.checked_sub(1), Some(col)),
        ];
        for (nr, nc) in candidates {
            if neighbors == 2 {
                break;
            }
            let (Some(nr), Some(nc)) = (nr, nc) else {
                continue;
            };
            if nr >= self.rows || nc >= self.cols {
                continue;
            }
            if !buffered_above && (nr >> 4) != (row >> 4) {
                continue;
            }
            count += usize::from(self.cells[nr * self.cols + nc]);
            neighbors += 1;
        }
        count
    }

    pub(crate) fn joint_mode_context(
        &self,
        row: usize,
        col: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> usize {
        let bottom_left = row
            .checked_add(block_rows.saturating_sub(1))
            .zip(col.checked_sub(1));
        let above_right = row
            .checked_sub(1)
            .zip(col.checked_add(block_cols.saturating_sub(1)));

        [bottom_left, above_right]
            .into_iter()
            .flatten()
            .filter(|&(nr, nc)| nr < self.rows && nc < self.cols)
            .map(|(nr, nc)| usize::from(self.cells[nr * self.cols + nc]))
            .sum()
    }

    pub(crate) fn set_block(
        &mut self,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        value: bool,
    ) {
        for y in row..row + rows {
            self.cells[y * self.cols + col..y * self.cols + col + cols].fill(value);
        }
    }
}

/// Palette-neighbor state. The first palette pass avoids adjacent palette
/// blocks so AVM's above/left color cache stays empty.
pub(crate) type PaletteGrid = BlockFlagGrid;

/// Convert a 4×4 transform unit's 16 raster-order levels into the entropy coder's
/// `(scan_position, level)` list (nonzero only), walked in `SCAN_4X4` order.
pub(crate) fn levels_to_coeffs_4x4(levels: &[i32; 16]) -> Vec<(usize, i32)> {
    let mut out = Vec::new();
    for (scan_pos, &raster) in SCAN_4X4.iter().enumerate() {
        let l = levels[raster as usize];
        if l != 0 {
            out.push((scan_pos, l));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::BlockFlagGrid;

    #[test]
    fn npos_uses_bottom_left_then_above_right() {
        let mut grid = BlockFlagGrid::new(32, 32);
        grid.set_block(11, 7, 1, 1, true); // bottom-left of row 8, col 8, 4x4 mi block
        grid.set_block(7, 11, 1, 1, false); // above-right
        grid.set_block(8, 7, 1, 1, true); // direct-left must not be reached
        grid.set_block(7, 8, 1, 1, true); // direct-above must not be reached

        assert_eq!(grid.npos_context(8, 8, 4, 4, true), 1);
    }

    #[test]
    fn npos_excludes_previous_superblock_row_but_nposbuf_keeps_it() {
        let mut grid = BlockFlagGrid::new(32, 32);
        grid.set_block(15, 0, 1, 2, true);

        assert_eq!(grid.npos_context(16, 0, 1, 1, false), 0);
        assert_eq!(grid.npos_context(16, 0, 1, 1, true), 2);
    }

    #[test]
    fn block_fill_and_palette_context_share_mi_grid() {
        let mut grid = BlockFlagGrid::new(8, 8);
        grid.set_block(2, 3, 2, 3, true);

        assert_eq!(grid.context(4, 4), 1);
        assert_eq!(grid.context(2, 6), 1);
        assert_eq!(grid.context(0, 0), 0);
    }

    #[test]
    fn joint_mode_context_has_no_direct_neighbor_fallback() {
        let mut grid = BlockFlagGrid::new(16, 16);
        grid.set_block(7, 3, 1, 1, true); // bottom-left
        grid.set_block(3, 7, 1, 1, true); // above-right
        grid.set_block(4, 3, 1, 1, true); // direct-left, ignored
        grid.set_block(3, 4, 1, 1, true); // direct-above, ignored

        assert_eq!(grid.joint_mode_context(4, 4, 4, 4), 2);

        grid.set_block(7, 3, 1, 1, false);
        grid.set_block(3, 7, 1, 1, false);
        assert_eq!(grid.joint_mode_context(4, 4, 4, 4), 0);
    }
}
