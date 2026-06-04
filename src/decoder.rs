//! Decoder — the other half that makes the codec *work*.
//!
//! It parses our OBU framing (real: header byte + LEB128 size), locates the
//! tile-group payload, strips its inner length prefix, and range-decodes every
//! plane with the same models, predictor, and inverse transform the encoder
//! used. Because encoding is lossless, the decoded pixels equal the source
//! exactly (proved by the round-trip tests).
//!
//! NOTE: image dimensions and bit depth are passed in here. In real AV1 they
//! come from the sequence header; wiring them through our (still-sketch) header
//! is a separate, small step and is intentionally not relied on yet.

use crate::PlanarImage;
use crate::bitwriter::read_leb128;
use crate::coeff::{CoeffCdfs, decode_block};
use crate::obu::FRAME_HEADER_LEN;
use crate::pixel::Pixel;
use crate::predict::left_predictor;
use crate::rangecoder::RangeDecoder;
use crate::transform::iwht4x4;

const OBU_FRAME: u8 = 6;

/// Walk the OBU stream and return the OBU_FRAME payload, if present.
fn find_frame_payload(bytes: &[u8]) -> Option<&[u8]> {
    let mut pos = 0;
    while pos < bytes.len() {
        let header = bytes[pos];
        let obu_type = (header >> 3) & 0x0f;
        let has_size = (header >> 1) & 0x01;
        let ext = (header >> 2) & 0x01;
        pos += 1;
        if ext == 1 {
            pos += 1;
        }
        if has_size == 0 {
            return None;
        }
        let (size, used) = read_leb128(&bytes[pos..]);
        pos += used;
        let payload = &bytes[pos..pos + size as usize];
        if obu_type == OBU_FRAME {
            return Some(payload);
        }
        pos += size as usize;
    }
    None
}

/// Decode a still image previously produced by `encode_still`.
pub fn decode_still<T: Pixel>(
    bytes: &[u8],
    width: usize,
    height: usize,
    bit_depth: u8,
) -> PlanarImage<T> {
    let payload = find_frame_payload(bytes).expect("no OBU_FRAME found");
    // OBU_FRAME payload = frame_header (FRAME_HEADER_LEN bytes) || tile entropy.
    let entropy = &payload[FRAME_HEADER_LEN..];

    let mut dec = RangeDecoder::new(entropy);
    let mut cdfs = CoeffCdfs::default();

    let mut planes: [Vec<T>; 3] = [
        vec![T::default(); width * height],
        vec![T::default(); width * height],
        vec![T::default(); width * height],
    ];
    for plane in planes.iter_mut() {
        decode_plane(plane, width, height, bit_depth, &mut dec, &mut cdfs);
    }

    PlanarImage {
        width,
        height,
        bit_depth,
        planes,
    }
}

fn decode_plane<T: Pixel>(
    out: &mut [T],
    width: usize,
    height: usize,
    bit_depth: u8,
    dec: &mut RangeDecoder,
    cdfs: &mut CoeffCdfs,
) {
    let mut recon = vec![0i32; width * height];
    let bw = width.div_ceil(4);
    let bh = height.div_ceil(4);

    for by in 0..bh {
        for bx in 0..bw {
            let pred = left_predictor(&recon, width, height, bx, by, bit_depth);
            let coeffs = decode_block(dec, cdfs);
            let inv = iwht4x4(&coeffs);
            for yy in 0..4 {
                for xx in 0..4 {
                    let gx = bx * 4 + xx;
                    let gy = by * 4 + yy;
                    if gx < width && gy < height {
                        let r = pred[yy * 4 + xx] + inv[yy * 4 + xx];
                        recon[gy * width + gx] = r;
                        out[gy * width + gx] = T::from_i32_clamped(r, bit_depth);
                    }
                }
            }
        }
    }
}
