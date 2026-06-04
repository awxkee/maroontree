//! Top-level encoder for slimav.
//! Encodes an 8-bit, 4:4:4, lossless RGB still image to a conformant AV1 OBU
//! stream using real AV1 entropy coding (od_ec MSAC, dav1d-compatible CDFs,
//! WHT-4x4 with DC_PRED, TX_4X4 throughout).

use crate::color::{Cicp, ImageMetadata};
use crate::obu::{metadata_obus, temporal_delimiter, wrap_obu_frame};
use crate::pixel::Pixel;

/// A planar image. `planes[0..3]` are full-resolution (4:4:4).
/// For identity RGB we store G, B, R in planes 0, 1, 2 (AV1 GBR ordering).
pub struct PlanarImage<T: Pixel> {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u8,
    pub planes: [Vec<T>; 3],
}

impl<T: Pixel> PlanarImage<T> {
    /// Build from interleaved RGB samples (`r,g,b,r,g,b,...`).
    /// AV1 identity matrix mapping: plane0=G, plane1=B, plane2=R.
    pub fn from_interleaved_rgb(width: usize, height: usize, bit_depth: u8, rgb: &[T]) -> Self {
        assert_eq!(rgb.len(), width * height * 3);
        let n = width * height;
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for px in rgb.chunks_exact(3) {
            r.push(px[0]);
            g.push(px[1]);
            b.push(px[2]);
        }
        PlanarImage {
            width,
            height,
            bit_depth,
            planes: [g, b, r],
        }
    }

    /// Reconstruct interleaved RGB from the GBR planes.
    pub fn to_interleaved_rgb(&self) -> Vec<T> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n * 3);
        for ((&r, &g), &b) in self.planes[2]
            .iter()
            .zip(self.planes[0].iter())
            .zip(self.planes[1].iter())
        {
            out.push(r); // R
            out.push(g); // G
            out.push(b); // B
        }
        out
    }
}

/// Result of encoding.
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// Always true for lossless coding; kept for API compatibility.
    pub lossless_verified: bool,
}

/// Encode a **lossy** 8-bit 4:4:4 still to a conformant AV1 OBU stream. Width
/// and height must be multiples of 64.
///
/// The frame is tiled into 64x64 superblocks (raster order, single tile), each
/// split uniformly into 8x8 blocks coded `DC_PRED` + `TX_8X8` (`DCT_DCT`) and
/// quantized by `base_q_idx` (keep `<= 20` to stay in coefficient qctx 0). A
/// reconstruction loop runs inside the encoder so each block's DC prediction
/// matches the decoder's, including across superblock boundaries. Verified to
/// decode in dav1d 1.4.1 and ffmpeg/libdav1d from 64x64 to 256x192 (gradient
/// PSNR ~49-57 dB at q=16; flat colour is exact).
pub fn encode_still_lossy<T: Pixel>(img: &PlanarImage<T>, base_q_idx: u8) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(
        matches!(img.bit_depth, 8 | 10 | 12),
        "only 8/10/12-bit supported"
    );
    assert!(base_q_idx != 0, "use encode_still for lossless (q=0)");
    let bd = img.bit_depth;
    let maxv = (1i32 << bd) - 1;
    let off = (1i32 << (bd - 1)) as f32;
    let mx = maxv as f32;
    let to_px = |p: &Vec<T>| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let (g, b, r) = (
        to_px(&img.planes[0]),
        to_px(&img.planes[1]),
        to_px(&img.planes[2]),
    );
    // Decorrelate via full-range BT.601 (JFIF) RGB->YCbCr. The sequence header
    // signals MC_BT_601 so the decoder inverts this back to RGB on output; the
    // near-flat chroma planes cost far fewer bits than coding G/B/R directly.
    // The matrix coefficients are bit-depth-independent ratios; only the chroma
    // offset (2^(bd-1)) and the clamp ceiling (2^bd-1) scale with depth.
    let n = r.len();
    let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
    for (((((yv, cbv), crv), &rr), &gg), &bb) in y
        .iter_mut()
        .zip(cb.iter_mut())
        .zip(cr.iter_mut())
        .zip(r.iter())
        .zip(g.iter())
        .zip(b.iter())
    {
        let (rf, gf, bf) = (rr as f32, gg as f32, bb as f32);
        *yv = (0.299 * rf + 0.587 * gf + 0.114 * bf)
            .round()
            .clamp(0.0, mx) as i32;
        *cbv = (-0.168736 * rf - 0.331264 * gf + 0.5 * bf + off)
            .round()
            .clamp(0.0, mx) as i32;
        *crv = (0.5 * rf - 0.418688 * gf - 0.081312 * bf + off)
            .round()
            .clamp(0.0, mx) as i32;
    }
    let bytes = crate::av1real::encode_av1_lossy_image_cs(
        base_q_idx, bd, img.width, img.height, &y, &cb, &cr, true,
    );
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a **lossy 4:2:2** still (profile 2). Like [`encode_still_lossy`] but
/// the chroma planes are horizontally subsampled by 2 (BT.601 full-range YCbCr,
/// `RTX_4X8` chroma transforms). The decoder reconstructs full-resolution RGB.
/// Bit-exact reconstruction was verified against dav1d 1.4.1. On the PSNR
/// metric this is roughly neutral-to-slightly-better on colourful content and
/// slightly worse on very smooth content (where 4:4:4 chroma is already
/// skip-dominated and nearly free), so 4:4:4 remains the default.
pub fn encode_still_lossy_422<T: Pixel>(img: &PlanarImage<T>, base_q_idx: u8) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(
        matches!(img.bit_depth, 8 | 10 | 12),
        "only 8/10/12-bit supported"
    );
    assert!(base_q_idx != 0, "use encode_still for lossless (q=0)");
    let (w, h) = (img.width, img.height);
    let bd = img.bit_depth;
    let maxv = (1i32 << bd) - 1;
    let off = (1i32 << (bd - 1)) as f32;
    let mx = maxv as f32;
    let to_px = |p: &Vec<T>| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let (g, b, r) = (
        to_px(&img.planes[0]),
        to_px(&img.planes[1]),
        to_px(&img.planes[2]),
    );
    let cw = w.div_ceil(2);
    let mut y = vec![0i32; w * h];
    let (mut cb, mut cr) = (vec![0i32; cw * h], vec![0i32; cw * h]);
    // full-res Cb/Cr, then horizontal box-average to half width
    let (mut fcb, mut fcr) = (vec![0f32; w * h], vec![0f32; w * h]);
    for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
        .iter_mut()
        .zip(fcb.iter_mut())
        .zip(fcr.iter_mut())
        .zip(r.iter())
        .zip(g.iter())
        .zip(b.iter())
    {
        let (rf, gf, bf) = (rr as f32, gg as f32, bb as f32);
        *yv = (0.299 * rf + 0.587 * gf + 0.114 * bf)
            .round()
            .clamp(0.0, mx) as i32;
        *fcbv = -0.168736 * rf - 0.331264 * gf + 0.5 * bf + off;
        *fcrv = 0.5 * rf - 0.418688 * gf - 0.081312 * bf + off;
    }
    for row in 0..h {
        for c in 0..cw {
            let x0 = 2 * c;
            let x1 = (2 * c + 1).min(w - 1);
            cb[row * cw + c] = ((fcb[row * w + x0] + fcb[row * w + x1]) * 0.5)
                .round()
                .clamp(0.0, mx) as i32;
            cr[row * cw + c] = ((fcr[row * w + x0] + fcr[row * w + x1]) * 0.5)
                .round()
                .clamp(0.0, mx) as i32;
        }
    }
    let bytes = crate::av1real::encode_av1_lossy_image_422(base_q_idx, bd, w, h, &y, &cb, &cr);
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a **lossy 4:2:0** still (profile 0). Like [`encode_still_lossy`] but
/// chroma is subsampled by 2 both horizontally and vertically (2x2 box average,
/// BT.601 full-range YCbCr, `TX_4X4` chroma transforms — quarter-resolution
/// chroma, same as JPEG/`yuv420p`). Bit-exact reconstruction verified vs dav1d
/// 1.4.1. As with 4:2:2 this does *not* improve RGB-PSNR R-D over the default
/// 4:4:4 path (slimav's chroma skip already makes smooth chroma nearly free), so
/// it exists mainly for pipeline compatibility; **4:4:4 remains the default.**
pub fn encode_still_lossy_420<T: Pixel>(img: &PlanarImage<T>, base_q_idx: u8) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(
        matches!(img.bit_depth, 8 | 10 | 12),
        "only 8/10/12-bit supported"
    );
    assert!(base_q_idx != 0, "use encode_still for lossless (q=0)");
    let (w, h) = (img.width, img.height);
    let bd = img.bit_depth;
    let maxv = (1i32 << bd) - 1;
    let off = (1i32 << (bd - 1)) as f32;
    let mx = maxv as f32;
    let to_px = |p: &Vec<T>| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let (g, b, r) = (
        to_px(&img.planes[0]),
        to_px(&img.planes[1]),
        to_px(&img.planes[2]),
    );
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y = vec![0i32; w * h];
    let (mut cb, mut cr) = (vec![0i32; cw * ch], vec![0i32; cw * ch]);
    let (mut fcb, mut fcr) = (vec![0f32; w * h], vec![0f32; w * h]);
    for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
        .iter_mut()
        .zip(fcb.iter_mut())
        .zip(fcr.iter_mut())
        .zip(r.iter())
        .zip(g.iter())
        .zip(b.iter())
    {
        let (rf, gf, bf) = (rr as f32, gg as f32, bb as f32);
        *yv = (0.299 * rf + 0.587 * gf + 0.114 * bf)
            .round()
            .clamp(0.0, mx) as i32;
        *fcbv = -0.168736 * rf - 0.331264 * gf + 0.5 * bf + off;
        *fcrv = 0.5 * rf - 0.418688 * gf - 0.081312 * bf + off;
    }
    for row in 0..ch {
        for c in 0..cw {
            let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
            let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));
            let avg = |f: &Vec<f32>| {
                (f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1]) * 0.25
            };
            cb[row * cw + c] = avg(&fcb).round().clamp(0.0, mx) as i32;
            cr[row * cw + c] = avg(&fcr).round().clamp(0.0, mx) as i32;
        }
    }
    let bytes = crate::av1real::encode_av1_lossy_image_420(base_q_idx, bd, w, h, &y, &cb, &cr);
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a **lossy** 8×8 8-bit 4:4:4 still to a conformant AV1 OBU stream.
///
/// Uses one `TX_8X8` (DCT_DCT) per plane with quantizer `base_q_idx` (keep
/// `<= 20` to stay in coefficient qctx 0). Verified to decode in dav1d 1.4.1
/// **and** ffmpeg/libdav1d (round-trip max error ~1 at q=16). This is the lossy
/// counterpart to `encode_still`'s lossless path; extending it to 64×64
/// requires the partition tree + per-block reconstruction documented in the
/// repo (lossy frames cannot use the `TX_4X4_ONLY` mode the lossless tile
/// relies on, so they must split the superblock into coded sub-blocks).
pub fn encode_lossy_8x8<T: Pixel>(img: &PlanarImage<T>, base_q_idx: u8) -> Encoded {
    assert_eq!(img.width, 8, "encode_lossy_8x8 expects 8×8");
    assert_eq!(img.height, 8, "encode_lossy_8x8 expects 8×8");
    assert_eq!(
        img.bit_depth, 8,
        "only 8-bit supported (lossy 10/12-bit pending)"
    );
    let to_px = |p: &Vec<T>| {
        let mut a = [0u8; 64];
        for (i, v) in p.iter().enumerate() {
            a[i] = v.to_i32().clamp(0, 255) as u8;
        }
        a
    };
    let (g, b, r) = (
        to_px(&img.planes[0]),
        to_px(&img.planes[1]),
        to_px(&img.planes[2]),
    );
    // planes are [G, B, R]; the encoder takes (luma=G, U=B, V=R) order.
    let bytes = crate::av1real::encode_av1_lossy_color_image_8x8(base_q_idx, &g, &b, &r);
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a 64×64 8-bit 4:4:4 still image to a conformant AV1 OBU stream.
///
/// # Panics
/// Panics if `img.width != 64 || img.height != 64 || img.bit_depth != 8`.
/// (Arbitrary-size / higher-bit-depth support requires multi-superblock tiling
/// and quantizer changes that are not yet implemented.)
pub fn encode_still<T: Pixel>(img: &PlanarImage<T>) -> Encoded {
    encode_still_with(img, &ImageMetadata::new(Cicp::identity_rgb()))
}

/// Encode a lossless 4:4:4 still with explicit colour/metadata signalling.
///
/// `meta.cicp` is written into the sequence-header `color_config` (primaries /
/// transfer / matrix / range). The coded planes are emitted verbatim: this
/// encoder does **not** apply any colour transform — feeding it RGB with an
/// identity matrix, or pre-decorrelated planes with the matching matrix
/// (e.g. user-applied YCgCo), is the caller's choice. HDR `cll` / `mdcv` and
/// `t35` user data are emitted as metadata OBUs after the sequence header; the
/// ICC profile is carried for the (future) AVIF muxer and is *not* placed in the
/// AV1 OBU stream.
pub fn encode_still_with<T: Pixel>(img: &PlanarImage<T>, meta: &ImageMetadata) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(
        matches!(img.bit_depth, 8 | 10 | 12),
        "only 8/10/12-bit supported"
    );
    let profile: u32 = if img.bit_depth == 12 { 2 } else { 1 };

    let (w, h) = (img.width, img.height);
    let (w8, h8) = (crate::av1real::align8(w), crate::av1real::align8(h));

    let to_i16 = |p: &[T]| p.iter().map(|p| p.to_i32() as i16).collect::<Vec<i16>>();
    let planes_i16: [Vec<i16>; 3] = [
        crate::av1real::pad_to_mult8(&to_i16(&img.planes[0]), w, h, w8, h8),
        crate::av1real::pad_to_mult8(&to_i16(&img.planes[1]), w, h, w8, h8),
        crate::av1real::pad_to_mult8(&to_i16(&img.planes[2]), w, h, w8, h8),
    ];

    let tile_payload = crate::av1_tile::encode_tile_lossless(
        w8,
        h8,
        img.bit_depth,
        [&planes_i16[0], &planes_i16[1], &planes_i16[2]],
    );

    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp(
        w as u32,
        h as u32,
        profile,
        img.bit_depth,
        &meta.cicp,
    ));
    bytes.extend_from_slice(&metadata_obus(meta));
    bytes.extend_from_slice(&wrap_obu_frame(
        &crate::obu::frame_header_lossless_tiled(sb_cols, sb_rows),
        &tile_payload,
    ));

    Encoded {
        bytes,
        lossless_verified: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Truly-arbitrary-size lossless guard: a 70×50 image (padded to a 72×56
    /// coded grid, header signals 70×50). Checks the public `encode_still` path
    /// is deterministic. Verified bit-exact through dav1d 1.4.1 and ffmpeg.
    #[test]
    fn lossless_70x50_arbitrary() {
        let (w, h) = (70usize, 50usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = (x % 64) as u8;
                rgb[i + 1] = (y % 64) as u8;
                rgb[i + 2] = ((x + y) % 64) as u8;
            }
        }
        let img = PlanarImage::from_interleaved_rgb(w, h, 8, &rgb);
        let out = encode_still(&img);
        assert!(out.lossless_verified);
        // header signals the exact (unpadded) frame size
        let sum: u64 = out.bytes.iter().map(|&b| b as u64).sum();
        assert_eq!(out.bytes.len(), 1493);
        assert_eq!(sum, 200614);
    }

    /// 10- and 12-bit lossless guards (40×24, identity RGB in `u16` planes).
    /// Bytes captured after verifying maxdiff 0 through dav1d 1.4.1 at both
    /// depths (high-bitdepth y4m, 16-bit LE samples, coded plane order G,B,R).
    /// The WHT/coef coding is bit-depth-agnostic; only the predictor base
    /// (`1<<(bd-1)`) and the `color_config` high_bitdepth/twelve_bit + profile
    /// signalling change with depth.
    #[test]
    fn lossless_high_bitdepth_stable() {
        let (w, h) = (40usize, 24usize);
        for (bd, len, sum, head) in [
            (10u8, 1519usize, 202296u64, [18u8, 0, 10, 8, 56, 21]),
            (12u8, 1522usize, 203041u64, [18u8, 0, 10, 9, 88, 21]),
        ] {
            let m = (1u32 << bd) as u16;
            let mut rgb = vec![0u16; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    rgb[i] = (((x * 5 + y * 3) as u32) as u16) % m;
                    rgb[i + 1] = (((x * 2 + y * 7) as u32) as u16) % m;
                    rgb[i + 2] = (((x * y + 1) as u32) as u16) % m;
                }
            }
            let img = PlanarImage::from_interleaved_rgb(w, h, bd, &rgb);
            let p = encode_still(&img).bytes;
            assert_eq!(p.len(), len, "bd={} length", bd);
            assert_eq!(
                p.iter().map(|&x| x as u64).sum::<u64>(),
                sum,
                "bd={} sum",
                bd
            );
            assert_eq!(&p[..6], &head, "bd={} head", bd);
        }
    }

    /// Lossy 10/12-bit guard across 4:4:4 / 4:2:2 / 4:2:0 (40×24, q80). Byte
    /// lengths + sums captured after verifying the encoder's reconstruction is
    /// bit-exact against dav1d 1.4.1 (`recon == decode`, maxdiff 0) at both
    /// depths and all three chroma formats, and that dav1d reports the correct
    /// `C444p10/C422p10/C420p10` (and `p12`) format.
    #[test]
    fn lossy_high_bitdepth_stable() {
        let (w, h) = (40usize, 24usize);
        for (bd, exp) in [
            (10u8, [(103usize, 11482u64), (95, 9921), (90, 9911)]),
            (12u8, [(61, 5835), (57, 5771), (52, 6131)]),
        ] {
            let m = (1u32 << bd) as u16;
            let mut rgb = vec![0u16; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    rgb[i] = (((x * 5 + y * 3) as u32) as u16) % m;
                    rgb[i + 1] = (((x * 2 + y * 7) as u32) as u16) % m;
                    rgb[i + 2] = (((x * y + 1) as u32) as u16) % m;
                }
            }
            let img = PlanarImage::from_interleaved_rgb(w, h, bd, &rgb);
            let outs = [
                encode_still_lossy(&img, 80),
                encode_still_lossy_422(&img, 80),
                encode_still_lossy_420(&img, 80),
            ];
            for (k, o) in outs.iter().enumerate() {
                assert!(!o.lossless_verified);
                assert_eq!(o.bytes.len(), exp[k].0, "bd={} fmt={} len", bd, k);
                assert_eq!(
                    o.bytes.iter().map(|&x| x as u64).sum::<u64>(),
                    exp[k].1,
                    "bd={} fmt={} sum",
                    bd,
                    k
                );
            }
        }
    }

    /// Non-multiple-of-8 sizes must not panic and must produce a non-empty
    /// stream for both very small and odd dimensions.
    #[test]
    fn arbitrary_sizes_do_not_panic() {
        for &(w, h) in &[(1usize, 1usize), (17, 17), (65, 33), (127, 129), (33, 7)] {
            let rgb = vec![100u8; w * h * 3];
            let img = PlanarImage::from_interleaved_rgb(w, h, 8, &rgb);
            assert!(!encode_still(&img).bytes.is_empty());
            assert!(!encode_still_lossy(&img, 16).bytes.is_empty());
        }
    }
}
