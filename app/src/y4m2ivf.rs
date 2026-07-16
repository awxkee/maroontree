// Minimal Y4M -> AV2 IVF encoder (Phase-A gate). Build: --features video.
// usage: y4m2ivf <in.y4m> <out.ivf> [qindex]
// Kept for manual testing via the commented-out call in main.rs.
#![allow(dead_code)]
use std::io::{Read, Write};

use maroontree::BitDepth;
use maroontree::{Av2Encoder, ChromaFormat, Cicp, PlanarImage, encode_ivf};

fn read_line<R: Read>(r: &mut R) -> Vec<u8> {
    let mut out = Vec::new();
    let mut b = [0u8; 1];
    while r.read_exact(&mut b).is_ok() {
        if b[0] == b'\n' {
            break;
        }
        out.push(b[0]);
    }
    out
}

pub(crate) fn convert(inp: String, outp: String, q: u8) {
    let args: Vec<String> = std::env::args().collect();

    let mut f = std::fs::File::open(inp).expect("open y4m");
    let hdr = read_line(&mut f);
    let hdr = String::from_utf8(hdr).unwrap();
    let (mut w, mut h) = (0usize, 0usize);
    let (mut fn_, mut fd) = (30u32, 1u32);
    let mut chroma = ChromaFormat::Yuv420;
    for tok in hdr.split(' ') {
        match tok.as_bytes().first() {
            Some(b'W') => w = tok[1..].parse().unwrap(),
            Some(b'H') => h = tok[1..].parse().unwrap(),
            Some(b'F') => {
                let mut it = tok[1..].split(':');
                fn_ = it.next().unwrap().parse().unwrap();
                fd = it.next().unwrap().parse().unwrap();
            }
            Some(b'C') => {
                chroma = if tok.starts_with("C444") {
                    ChromaFormat::Yuv444
                } else if tok.starts_with("C422") {
                    ChromaFormat::Yuv422
                } else if tok.starts_with("Cmono") {
                    ChromaFormat::Monochrome
                } else {
                    ChromaFormat::Yuv420
                };
            }
            _ => {}
        }
    }

    let (cw, ch) = match chroma {
        ChromaFormat::Yuv420 => (w.div_ceil(2), h.div_ceil(2)),
        ChromaFormat::Yuv422 => (w.div_ceil(2), h),
        ChromaFormat::Yuv444 => (w, h),
        ChromaFormat::Monochrome => (0, 0),
    };
    let ysz = w * h;
    let csz = cw * ch;

    let mut frames = Vec::new();
    loop {
        let marker = read_line(&mut f); // "FRAME"
        if marker.is_empty() {
            break;
        }
        let mut y = vec![0u8; ysz];
        if f.read_exact(&mut y).is_err() {
            break;
        }
        let (mut u, mut v) = (vec![0u8; csz], vec![0u8; csz]);
        if csz > 0 {
            f.read_exact(&mut u).unwrap();
            f.read_exact(&mut v).unwrap();
        }
        frames.push(PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [y, u, v, Vec::new()],
        });
    }

    // No tiling for typical (<=1080p) clips: tiny tiles add per-tile overhead,
    // disable per-block multi-reference, and break prediction across tile edges.
    // Only tile large frames, where the threading win outweighs the overhead.
    let cfg = if w * h > 1920 * 1080 {
        Av2Encoder::new(q).with_tiles(2, 2).with_threads(12)
    } else {
        Av2Encoder::new(q).with_threads(12)
    };
    let color = Cicp::srgb_ycbcr();
    // Default to a ~2-second GOP so P-frames (inter prediction) are used. The old
    // default of 0 meant ALL-INTRA — every frame a keyframe — which is ~30x larger
    // on typical content. Pass an explicit 0 as arg 5 to force all-intra.
    let default_ki = (2 * fn_ as u64 / fd.max(1) as u64).max(1);
    let ki: u64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_ki);
    let ivf = encode_ivf(cfg, chroma, 0, &frames, &color, fn_, fd, ki).expect("encode");
    let mut out = std::fs::File::create(outp.clone()).expect("create");
    out.write_all(&ivf).unwrap();
    eprintln!(
        "wrote {} frames ({} bytes, key_interval={ki}) -> {}",
        frames.len(),
        ivf.len(),
        outp
    );
}
