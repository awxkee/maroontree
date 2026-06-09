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

use crate::av2::quant::{BASE_Q, qstep};
use crate::av2::tables::SCAN;

pub(crate) struct Basis {
    pub(crate) dc: f32,
    pub(crate) _n_pix: usize,
    pub(crate) n_cf: usize,
    pub(crate) norm2: Vec<f32>,
    /// Separable 1D profiles, laid out `[freq*side + spatial]` with 32 frequencies.
    /// The dense 2D basis is `outer(hv[c], hh[a]) / dc`; storing only the 1D axes lets
    /// project/reconstruct run as two 1D passes (≈10× fewer mults, 4MB matrix → ~8KB).
    pub(crate) hv: Vec<f32>,
    pub(crate) side_v: usize,
    pub(crate) hh: Vec<f32>,
    pub(crate) side_h: usize,
    /// Quantiser rescale factor (qstep_target / qstep_measured), applied at run time
    /// instead of mutating the profiles.
    pub(crate) scale: f32,
    /// Max reconstructed sample value = (1 << bit_depth) - 1. Defaults to 8-bit (255);
    /// set per-encode from the signalled bit depth.
    pub(crate) max_val: f32,
}

const NF: usize = 32; // coded frequencies per axis

/// 8-lane dot product; autovectorises (slices are always a multiple of 8 long).
#[inline]
fn dot8(x: &[f32], y: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (a, b) in x.chunks_exact(8).zip(y.chunks_exact(8)) {
        for j in 0..8 {
            acc[j] += a[j] * b[j];
        }
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

pub(crate) struct Bases {
    pub(crate) luma: Basis,
    pub(crate) chroma420: Basis,
    /// 4:2:2 chroma: a 32-wide × 64-tall (TX_32X64) transform. avmdec codes its
    /// coefficients exactly like the 64×64 chroma (adjusted size TX_32X32, scan
    /// `default_scan_32x32`, EOB size class 6, txs entropy context TX_64X64), so only
    /// the basis differs: the vertical axis is the 64-tap profile and the horizontal
    /// axis the 32-tap profile. The overall amplitude (`dc_mix` below) carries avm's
    /// rectangular-transform normalisation and is the one value to confirm on avmdec.
    pub(crate) chroma422: Basis,
    pub(crate) chroma444: Basis,
}

impl Basis {
    /// Rescale to a different quantiser. The bases are the decoder's level-1
    /// reconstruction `qstep · T(k)` with a q-independent transform `T`, so the target
    /// quantiser is just a multiply by `f = qstep_target/qstep_measured`. We record it
    /// as a factor (applied in project/reconstruct) rather than touching the profiles.
    pub(crate) fn scale(&mut self, f: f32) {
        self.scale *= f;
    }
}

impl Bases {
    /// Rescale bases measured at `quant::BASE_Q` to an arbitrary 8-bit base_q_idx.
    /// Set the reconstruction clamp ceiling from the signalled bit depth (8/10/12).
    pub(crate) fn set_bit_depth(&mut self, bit_depth: u8) {
        let mv = ((1u32 << bit_depth) - 1) as f32;
        self.luma.max_val = mv;
        self.chroma420.max_val = mv;
        self.chroma422.max_val = mv;
        self.chroma444.max_val = mv;
    }
    pub(crate) fn rescaled_to_q(mut self, base_q_idx: u32) -> Bases {
        let f = qstep(base_q_idx) as f32 / qstep(BASE_Q) as f32;
        if f != 1.0 {
            self.luma.scale(f);
            self.chroma420.scale(f);
            self.chroma422.scale(f);
            self.chroma444.scale(f);
        }
        self
    }
}

fn need(b: &[u8], o: usize, n: usize, what: &str) {
    assert!(
        o.checked_add(n).map_or(false, |e| e <= b.len()),
        "bases file truncated/mismatched while reading {what}: need {n} bytes at offset {o}, file is {} bytes (is proj.rs in sync with the .bin?)",
        b.len()
    );
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    need(b, o, 4, "f32");
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    need(b, o, 4, "u32");
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Basis {
    /// Square separable basis. Equivalent to the rectangular builder with the same
    /// profile on both axes (`m[k][py,px] = h[c,py]·h[a,px]/dc`, c=SCAN[k]&31,
    /// a=SCAN[k]>>5).
    fn from_1d(dc: f32, side: usize, h: &[f32]) -> Basis {
        Self::from_1d_rect(dc, h, side, h, side)
    }

    /// Non-square (`side_h` wide × `side_v` tall) separable basis. The coded coefficient
    /// grid is 32×32 (avm zeroes the long axis' high frequencies and reuses the 32×32
    /// scan), so scan position k → horizontal frequency `a = SCAN[k] >> 5`, vertical
    /// frequency `c = SCAN[k] & 31`. Only the per-axis spatial profiles are stored; the
    /// dense outer product is never materialised. `norm2[k]` is computed separably,
    /// `Σ(hv[c,py]·hh[a,px]/dc)² = (Σ hv[c]²)(Σ hh[a]²)/dc²`.
    fn from_1d_rect(
        dc: f32,
        h_vert: &[f32],
        side_v: usize,
        h_horiz: &[f32],
        side_h: usize,
    ) -> Basis {
        let n_pix = side_v * side_h;
        let n_cf = 1024;
        let nv: Vec<f32> = (0..NF)
            .map(|c| {
                h_vert[c * side_v..c * side_v + side_v]
                    .iter()
                    .map(|&x| x * x)
                    .sum()
            })
            .collect();
        let nh: Vec<f32> = (0..NF)
            .map(|a| {
                h_horiz[a * side_h..a * side_h + side_h]
                    .iter()
                    .map(|&x| x * x)
                    .sum()
            })
            .collect();
        let dc2 = dc * dc;
        let mut norm2 = vec![0f32; n_cf];
        for k in 0..n_cf {
            let rc = SCAN[k] as usize;
            norm2[k] = nv[rc & 31] * nh[rc >> 5] / dc2;
        }
        Basis {
            dc,
            _n_pix: n_pix,
            n_cf,
            norm2,
            hv: h_vert.to_vec(),
            side_v,
            hh: h_horiz.to_vec(),
            side_h,
            scale: 1.0,
            max_val: 255.0,
        }
    }

    /// Project a residual block (`n_pix` samples, row-major) → integer coefficient
    /// levels (`n_cf`); |projection| < thresh is dropped. Separable: a horizontal 1D
    /// transform of each row, then a vertical 1D transform per coefficient.
    pub(crate) fn project(&self, resid: &[f32], thresh: f32) -> Vec<f32> {
        let (sv, sh) = (self.side_v, self.side_h);
        // horizontal pass → t[a*sv + py] = Σ_px resid[py,px]·hh[a,px]
        let mut t = [0f32; NF * 64];
        for py in 0..sv {
            let row = &resid[py * sh..py * sh + sh];
            for a in 0..NF {
                t[a * sv + py] = dot8(row, &self.hh[a * sh..a * sh + sh]);
            }
        }
        // vertical pass + quantise. S[c,a] = Σ_py t[a,py]·hv[c,py]; the dense projection
        // is S/dc, and pr = S/(dc·scale·norm2).
        let mut lev = vec![0f32; self.n_cf];
        let inv = 1.0 / (self.dc * self.scale);
        for k in 0..self.n_cf {
            let rc = SCAN[k] as usize;
            let (a, c) = (rc >> 5, rc & 31);
            let s = dot8(&t[a * sv..a * sv + sv], &self.hv[c * sv..c * sv + sv]);
            let pr = s * inv / self.norm2[k];
            if pr.abs() >= thresh {
                lev[k] = pr.round();
            }
        }
        lev
    }

    /// Reconstruct: clip(round(pred + scale·inverse_transform(lev))), `n_pix` samples.
    /// Separable inverse: scatter levels into the 32×32 frequency grid, synthesise
    /// horizontally per frequency row, then vertically into pixels. Empty rows skip.
    pub(crate) fn reconstruct(&self, pred: f32, lev: &[f32]) -> Vec<f32> {
        let (sv, sh) = (self.side_v, self.side_h);
        let mut s_grid = [0f32; NF * NF];
        let mut row_nz = [false; NF];
        for k in 0..self.n_cf {
            let l = lev[k];
            if l == 0.0 {
                continue;
            }
            let rc = SCAN[k] as usize;
            s_grid[(rc & 31) * NF + (rc >> 5)] = l;
            row_nz[rc & 31] = true;
        }
        // horizontal synthesis → w[c, px] = Σ_a S[c,a]·hh[a,px]
        let mut w = [0f32; NF * 64];
        for c in 0..NF {
            if !row_nz[c] {
                continue;
            }
            let wc = &mut w[c * sh..c * sh + sh];
            for a in 0..NF {
                let sca = s_grid[c * NF + a];
                if sca == 0.0 {
                    continue;
                }
                for (o, &b) in wc.iter_mut().zip(&self.hh[a * sh..a * sh + sh]) {
                    *o += sca * b;
                }
            }
        }
        // vertical synthesis → out[py,px] = pred + (scale/dc)·Σ_c w[c,px]·hv[c,py]
        let mut out = vec![pred; sv * sh];
        let g = self.scale / self.dc;
        for c in 0..NF {
            if !row_nz[c] {
                continue;
            }
            let wc = &w[c * sh..c * sh + sh];
            let hvc = &self.hv[c * sv..c * sv + sv];
            for py in 0..sv {
                let coef = hvc[py] * g;
                let orow = &mut out[py * sh..py * sh + sh];
                for (o, &b) in orow.iter_mut().zip(wc) {
                    *o += coef * b;
                }
            }
        }
        let mv = self.max_val;
        for v in out.iter_mut() {
            *v = v.round().clamp(0.0, mv);
        }
        out
    }
}

/// The 16 KB 1D bases are embedded directly in the binary, so the default encode
/// path needs no external file and cannot fail to locate it.
static EMBEDDED_BASES: &[u8] = include_bytes!("tree_av2_bases.bin");

/// Read one 1D-profile block, returning `(dc, side, profile)`. The dense square basis
/// is built by the caller via [`Basis::from_1d`]; keeping the raw profile lets us also
/// assemble the rectangular 4:2:2 basis from the 64- and 32-tap profiles already present.
fn read_raw(b: &[u8], o: &mut usize) -> (f32, usize, Vec<f32>) {
    let dc = rd_f32(b, *o);
    let side = rd_u32(b, *o + 4) as usize;
    let nfreq = rd_u32(b, *o + 8) as usize;
    *o += 12;
    assert_eq!(
        nfreq, 32,
        "expected 32 retained frequencies per dimension, got {nfreq} (proj.rs/.bin mismatch?)"
    );
    assert!(
        side == 32 || side == 64,
        "unexpected transform side {side} (proj.rs/.bin mismatch?)"
    );
    let count = nfreq * side;
    need(b, *o, count * 4, "1D basis profile");
    let mut h = vec![0f32; count];
    for v in h.iter_mut() {
        *v = rd_f32(b, *o);
        *o += 4;
    }
    (dc, side, h)
}

fn parse_bases(b: &[u8]) -> Bases {
    assert!(
        b.len() >= 8 && &b[0..4] == b"SL1D",
        "bad bases magic (expected SL1D); proj.rs and slimav_bases1d_q120.bin are out of sync"
    );
    let nblocks = rd_u32(b, 4);
    assert!(
        nblocks >= 3,
        "bases file needs luma + chroma420 + chroma444 (nblocks={nblocks})"
    );
    let mut o = 8usize;
    let (ldc, lside, lh) = read_raw(b, &mut o);
    let (c420dc, c420side, c420h) = read_raw(b, &mut o);
    let (c444dc, c444side, c444h) = read_raw(b, &mut o);
    debug_assert_eq!(
        o,
        b.len(),
        "bases file not fully consumed: {o} of {}",
        b.len()
    );
    let luma = Basis::from_1d(ldc, lside, &lh);
    let chroma420 = Basis::from_1d(c420dc, c420side, &c420h);
    let chroma444 = Basis::from_1d(c444dc, c444side, &c444h);
    // 4:2:2 = vertical 64-tap profile (from the 64×64 block) × horizontal 32-tap profile
    // (from the 32×32 block). The amplitude `dc_mix` is the geometric mean of the two
    // square DC gains, which reproduces avm's rectangular (1/√2-class) normalisation to
    // first order; verify/tune it against avmdec.
    assert_eq!(
        (c444side, c420side),
        (64, 32),
        "expected chroma444 side 64 and chroma420 side 32"
    );
    let dc_mix = (c420dc * c444dc).sqrt();
    let chroma422 = Basis::from_1d_rect(dc_mix, &c444h, 64, &c420h, 32);
    Bases {
        luma,
        chroma420,
        chroma422,
        chroma444,
    }
}

/// Bases compiled into the binary (default).
pub(crate) fn default_bases() -> Bases {
    parse_bases(EMBEDDED_BASES)
}

/// Override: load bases from an external `SL1D` file (e.g. a different quantiser).
pub(crate) fn load_bases(path: &str) -> Bases {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read bases file {path}"));
    parse_bases(&b)
}
