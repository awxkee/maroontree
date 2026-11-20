use image::{ImageBuffer, ImageFormat, Rgb};
use img_parts::png::{Png, PngChunk};
use std::fs::{self, File};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PQ_CICP: [u8; 4] = [9, 16, 0, 1];
const HLG_CICP: [u8; 4] = [9, 18, 0, 1];
const SDR_CICP: [u8; 4] = [1, 13, 0, 1];
// PQ patches with one neutral and one colored highlight.
const PATCHES: [[u16; 3]; 2] = [[33_297; 3], [50_574, 48_111, 44_340]];

struct TestDir(PathBuf);

impl TestDir {
    fn new(test: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("maroontree-hdr-{test}-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_png(path: &Path, cicp: [u8; 4]) {
    write_patches(path, cicp, PATCHES);
}

fn write_patches(path: &Path, cicp: [u8; 4], patches: [[u16; 3]; 2]) {
    let image = ImageBuffer::from_fn(32, 16, |x, _| Rgb(patches[(x / 16) as usize]));
    let mut pixels = Cursor::new(Vec::new());
    image.write_to(&mut pixels, ImageFormat::Png).unwrap();
    let mut png = Png::from_bytes(pixels.into_inner().into()).unwrap();
    png.chunks_mut()
        .insert(1, PngChunk::new(*b"cICP", cicp.to_vec().into()));
    png.encoder().write_to(File::create(path).unwrap()).unwrap();
}

fn run(input: &Path, output: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_maroontree-cli"))
        .args(["--threads", "1", "--speed", "fast"])
        .args(args)
        .arg(input)
        .arg(output)
        .output()
        .unwrap()
}

fn encode(input: &Path, output: &Path, args: &[&str]) -> Vec<u8> {
    let result = run(input, output, args);
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr),
    );
    fs::read(output).unwrap()
}

fn reject(input: &Path, output: &Path, args: &[&str], diagnostic: &str) {
    let result = run(input, output, args);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "unexpected success for {args:?}");
    assert!(
        stderr
            .to_ascii_lowercase()
            .replace('-', " ")
            .contains(diagnostic),
        "{stderr}",
    );
    assert!(!output.exists());
}

fn find_box<'a>(mut bytes: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    while bytes.len() >= 8 {
        let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert!((8..=bytes.len()).contains(&size));
        if &bytes[4..8] == kind {
            return Some(&bytes[8..size]);
        }
        bytes = &bytes[size..];
    }
    None
}

fn assert_gain_map_payload(bytes: &[u8], format: &str) {
    match format {
        "avif" => {
            let meta = find_box(bytes, b"meta").unwrap();
            let iref = find_box(&meta[4..], b"iref").unwrap();
            assert_eq!(&iref[..4], &[0; 4]);
            let dimg = find_box(&iref[4..], b"dimg").expect("missing AVIF tone-map reference");
            let id = |bytes: &[u8]| u16::from_be_bytes(bytes.try_into().unwrap());
            assert_eq!(id(&dimg[2..4]), 2);

            let container = tealdust::AvifParser::parse(bytes).unwrap();
            let tone_map = &container.items[&id(&dimg[..2])];
            assert_eq!(tone_map.item_type, tealdust::ItemType::Other(*b"tmap"));
            assert_eq!(id(&dimg[4..6]), container.primary_item_id);
            let gain = &container.items[&id(&dimg[6..8])];
            assert_eq!(gain.item_type, tealdust::ItemType::Av01);
            assert!(
                gain.iloc_extents.iter().any(|&(offset, length)| {
                    length > 0
                        && offset
                            .checked_add(length)
                            .is_some_and(|end| end <= bytes.len() as u64)
                }),
                "missing AVIF gain-map payload",
            );
        }
        "jxl" => assert!(
            !find_box(bytes, b"jhgm")
                .expect("missing JXL gain map")
                .is_empty()
        ),
        #[cfg(feature = "heic")]
        "heic" => {
            let settings = hpvcd::HeicSettings::new().with_decode_gain_map(true);
            let decoded = hpvcd::decode_heic_yuv_with_settings(bytes, &settings).unwrap();
            let map = decoded.gain_map.expect("missing HEIC gain map");
            assert_eq!(map.channels(), 1);
            assert!(map.width() > 0 && map.height() > 0);
        }
        _ => unreachable!(),
    }
}

fn restore_png(
    input: &Path,
    output: &Path,
    mode: &str,
    expected_cicp: [u8; 4],
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let restored = encode(input, output, &["--hdr-conversion", mode]);
    let png = Png::from_bytes(restored.into()).unwrap();
    let cicp = png.chunk_by_type(*b"cICP").unwrap();
    assert_eq!(cicp.contents().as_ref(), expected_cicp, "{input:?}");
    let rgb = match image::open(output).unwrap() {
        image::DynamicImage::ImageRgb16(rgb) => rgb,
        _ => panic!("expected 16-bit RGB in {output:?}"),
    };
    assert_eq!(rgb.dimensions(), (32, 16), "{input:?}");
    rgb
}

fn assert_restored_png(input: &Path, output: &Path, tolerance: u16) {
    let image = restore_png(input, output, "pq", PQ_CICP);
    for (x, y, pixel) in image.enumerate_pixels() {
        let expected = PATCHES[(x / 16) as usize];
        for (sample, expected) in pixel.0.into_iter().zip(expected) {
            assert!(
                sample.abs_diff(expected) <= tolerance,
                "({x}, {y}): {sample} vs {expected}",
            );
        }
    }
}

#[test]
fn avif_decodes_reversible_ycgco_at_its_rgb_bit_depth() {
    use maroontree::{BitDepth, ChromaFormat, Cicp, EncodeConfig, MatrixCoefficients, PlanarImage};

    let files = TestDir::new("ycgco-r");
    for (label, matrix, added_bits) in [
        ("re", MatrixCoefficients::YCgCoRe, 2),
        ("ro", MatrixCoefficients::YCgCoRo, 1),
    ] {
        for (coded_depth, full_range) in [
            (BitDepth::Eight, true),
            (BitDepth::Ten, true),
            (BitDepth::Twelve, true),
            (BitDepth::Ten, false),
            (BitDepth::Twelve, false),
        ] {
            let depth = coded_depth.bits();
            let rgb_depth = depth - added_bits;
            let (black, white) = if full_range {
                (0, (1i32 << rgb_depth) - 1)
            } else {
                // H.273 scales the RGB codes before reversible lifting.
                (16 << (rgb_depth - 8), 235 << (rgb_depth - 8))
            };
            let span = white - black;
            let chroma_bias = 1i32 << (depth - 1);
            let colors = [
                [black; 3],
                [white; 3],
                [white, black + 11, black + 41],
                [black + 51, white, black + 13],
                [black + 17, black + span * 3 / 5, white],
                [black + 3, black + 5, black + 4],
                [white - 3, black + span / 2, black + span / 3],
                [black + span / 2, white - 3, black + span / 3],
            ];
            let width = 16;
            let height = 8;
            let mut planes: [Vec<u16>; 4] = Default::default();
            for index in 0..width * height {
                let [r, g, b] = colors[(index % width) / 2];
                // H.273 reversible lifting on integer RGB.
                let co = r - b;
                let base = b + (co >> 1);
                let cg = g - base;
                let y = base + (cg >> 1);
                for (plane, value) in planes
                    .iter_mut()
                    .zip([y, cg + chroma_bias, co + chroma_bias])
                {
                    plane.push(u16::try_from(value).unwrap());
                }
            }
            let planar = PlanarImage {
                width,
                height,
                bit_depth: coded_depth,
                planes,
            };
            let config = EncodeConfig::new()
                .with_cicp(Cicp {
                    matrix,
                    full_range,
                    ..Cicp::srgb()
                })
                .with_chroma(ChromaFormat::Yuv444)
                .with_threads(1)
                .with_speed(maroontree::Speed::Fast);
            let range = if full_range { "full" } else { "limited" };
            let input = files.path(&format!("{label}-{depth}-{range}.avif"));
            let output = files.path(&format!("{label}-{depth}-{range}.png"));
            let avif = maroontree::encode_lossless(&planar, &config).unwrap();
            let meta = find_box(&avif, b"meta").unwrap();
            let iprp = find_box(&meta[4..], b"iprp").unwrap();
            let ipco = find_box(iprp, b"ipco").unwrap();
            let colr = find_box(ipco, b"colr").unwrap();
            assert_eq!(&colr[..4], b"nclx");
            assert_eq!(
                u16::from_be_bytes(colr[8..10].try_into().unwrap()),
                matrix as u16
            );
            assert_eq!(colr[10] & 0x80 != 0, full_range);
            fs::write(&input, avif).unwrap();

            let png = Png::from_bytes(encode(&input, &output, &[]).into()).unwrap();
            let cicp = png.chunk_by_type(*b"cICP").unwrap();
            assert_eq!(cicp.contents().as_ref(), SDR_CICP);
            let decoded = image::open(&output).unwrap().to_rgb16();
            assert_eq!(decoded.dimensions(), (width as u32, height as u32));
            let (png_max, to_u16, tolerance) = if coded_depth == BitDepth::Eight {
                (255, 257, 257)
            } else {
                (65_535, 1, 1)
            };
            for (x, y, pixel) in decoded.enumerate_pixels() {
                let expected = colors[(x as usize) / 2];
                for (result, component) in pixel.0.into_iter().zip(expected) {
                    let expanded =
                        (((component - black) * png_max + span / 2) / span * to_u16) as u16;
                    assert!(
                        result.abs_diff(expanded) <= tolerance,
                        "YCgCo-{label} {depth}-bit {range} [{x}, {y}] decoded {result}, expected {expanded}",
                    );
                }
            }
        }
    }
}

#[test]
fn hdr_routes_between_avif_jxl_and_heic() {
    let files = TestDir::new("formats");
    let formats = [
        "avif",
        "jxl",
        #[cfg(feature = "heic")]
        "heic",
    ];
    let hlg_patches = [[35_000; 3], [59_000, 53_000, 48_000]];
    let input = files.path("source-hlg.png");
    write_patches(&input, HLG_CICP, hlg_patches);

    let reference = restore_png(&input, &files.path("reference-pq.png"), "pq", PQ_CICP);
    let pq_patches = [reference.get_pixel(8, 8).0, reference.get_pixel(24, 8).0];

    let encode_hdr = |input: &Path, output: &Path, mode: &str| {
        encode(
            input,
            output,
            &["--lossless", "--quality", "100", "--hdr-conversion", mode],
        )
    };
    let check_png = |input: &Path, mode, cicp, patches| {
        let output = files.path("restored.png");
        let image = restore_png(input, &output, mode, cicp);
        for (x, expected) in [8, 24].into_iter().zip(patches) {
            for (sample, expected) in image.get_pixel(x, 8).0.into_iter().zip(expected) {
                assert!(
                    sample.abs_diff(expected) <= 400,
                    "{input:?}, {mode}, x={x}: {sample} vs {expected}"
                );
            }
        }
    };
    for source in formats {
        let hlg = files.path(&format!("source-hlg.{source}"));
        let pq = files.path(&format!("source-pq.{source}"));
        let map = files.path(&format!("source-map.{source}"));
        encode_hdr(&input, &hlg, "hlg");
        encode_hdr(&input, &pq, "pq");
        assert_gain_map_payload(&encode_hdr(&input, &map, "gain-map"), source);
        for target in formats.into_iter().filter(|target| *target != source) {
            let out = files.path(&format!("{source}-hlg-to-map.{target}"));
            assert_gain_map_payload(&encode_hdr(&hlg, &out, "gain-map"), target);
            check_png(&out, "hlg", HLG_CICP, hlg_patches);

            let out = files.path(&format!("{source}-pq-to-map.{target}"));
            assert_gain_map_payload(&encode_hdr(&pq, &out, "gain-map"), target);
            check_png(&out, "pq", PQ_CICP, pq_patches);

            let out = files.path(&format!("{source}-map-to-pq.{target}"));
            encode_hdr(&map, &out, "pq");
            check_png(&out, "pq", PQ_CICP, pq_patches);

            let out = files.path(&format!("{source}-map-to-hlg.{target}"));
            encode_hdr(&map, &out, "hlg");
            check_png(&out, "hlg", HLG_CICP, hlg_patches);

            let out = files.path(&format!("{source}-pq-to-hlg.{target}"));
            encode_hdr(&pq, &out, "hlg");
            check_png(&out, "hlg", HLG_CICP, hlg_patches);

            let out = files.path(&format!("{source}-hlg-to-pq.{target}"));
            encode_hdr(&hlg, &out, "pq");
            check_png(&out, "pq", PQ_CICP, pq_patches);

            let out = files.path(&format!("{source}-map-to-map.{target}"));
            let bytes = encode(&map, &out, &["--lossless", "--quality", "100"]);
            assert_gain_map_payload(&bytes, target);
            check_png(&out, "hlg", HLG_CICP, hlg_patches);
        }
    }
}

#[test]
fn pq_to_avif_gain_map_is_opt_in() {
    let files = TestDir::new("avif");
    let input = files.path("source.png");
    write_png(&input, PQ_CICP);

    let ordinary = encode(&input, &files.path("ordinary.avif"), &[]);
    let converted = encode(
        &input,
        &files.path("converted.avif"),
        &["--hdr-conversion", "gain-map"],
    );
    let lossless = encode(
        &input,
        &files.path("converted-lossless.avif"),
        &["--hdr-conversion", "gain-map", "--lossless"],
    );
    for (file, base_transfer, has_map) in [
        (&ordinary, 16, false),
        (&converted, 13, true),
        (&lossless, 13, true),
    ] {
        let ftyp = find_box(file, b"ftyp").unwrap();
        assert_eq!(
            ftyp[8..].chunks_exact(4).any(|brand| brand == b"tmap"),
            has_map
        );
        if has_map {
            assert_gain_map_payload(file, "avif");
        }
        let meta = find_box(file, b"meta").unwrap();
        let iprp = find_box(&meta[4..], b"iprp").unwrap();
        let ipco = find_box(iprp, b"ipco").unwrap();
        let colr = find_box(ipco, b"colr").unwrap();
        assert_eq!(&colr[..4], b"nclx");
        assert_eq!(
            u16::from_be_bytes(colr[6..8].try_into().unwrap()),
            base_transfer
        );
    }
}

#[test]
fn pq_round_trips_through_a_lossless_jxl_gain_map() {
    let files = TestDir::new("jxl");
    let input = files.path("source.png");
    let jxl = files.path("mapped.jxl");
    write_png(&input, PQ_CICP);

    let mapped = encode(
        &input,
        &jxl,
        &["-e", "jxl", "--lossless", "--hdr-conversion", "gain-map"],
    );
    assert_gain_map_payload(&mapped, "jxl");

    assert_restored_png(&jxl, &files.path("restored.png"), 20);
    reject(
        &jxl,
        &files.path("unsupported.jpg"),
        &["--hdr-conversion", "pq"],
        "pq",
    );
    reject(
        &jxl,
        &files.path("8-bit.avif"),
        &["--hdr-conversion", "pq", "--depth", "8"],
        "pq",
    );
}

#[cfg(feature = "heic")]
#[test]
fn pq_round_trips_through_an_apple_heic_gain_map() {
    let files = TestDir::new("heic");
    let input = files.path("source.png");
    let heic = files.path("mapped.heic");
    write_png(&input, PQ_CICP);

    let mapped = encode(
        &input,
        &heic,
        &["-e", "heic", "--lossless", "--hdr-conversion", "gain-map"],
    );
    assert_gain_map_payload(&mapped, "heic");

    // HEIC stores the auxiliary image on an 8-bit gain curve.
    assert_restored_png(&heic, &files.path("restored.png"), 300);
}

#[cfg(feature = "heic")]
#[test]
fn heic_rebuilds_an_rgb_jxl_gain_map() {
    let files = TestDir::new("rgb-map");
    let jxl = files.path("rgb-map.jxl");
    let heic = files.path("rebuilt.heic");
    let base = ImageBuffer::from_fn(32, 16, |x, _| {
        Rgb(if x < 16 {
            [40_000; 3]
        } else {
            [55_000, 48_000, 41_000]
        })
    });
    let gain = ImageBuffer::from_fn(32, 16, |x, _| {
        Rgb(if x < 16 {
            [0; 3]
        } else {
            [u16::MAX, 20_000, 8_000]
        })
    });
    let metadata = jixel::IsoGainMap::from_floats(&jixel::GainMapFloats {
        max: [2.0; 3],
        base_offset: [0.0; 3],
        alternate_offset: [0.0; 3],
        alternate_hdr_headroom: 2.0,
        ..Default::default()
    })
    .unwrap();
    let gain = jixel::GainMap::rgb16(gain.into_raw(), 16, 32, 16, metadata)
        .with_lossless(true)
        .with_color_encoding(jixel::ColorEncoding::srgb_linear())
        .with_alternate_color_encoding(jixel::ColorEncoding::bt2020_pq());
    let configuration = jixel::EncodeConfig::default()
        .with_lossless(true)
        .with_speed(jixel::Speed::Fast)
        .with_num_threads(1)
        .with_gain_map(gain);
    let bytes = jixel::encode_image_16bit(base.as_raw(), 32, 16, &configuration).unwrap();
    fs::write(&jxl, bytes).unwrap();

    let rebuilt = encode(&jxl, &heic, &["--lossless", "--quality", "100"]);
    assert_gain_map_payload(&rebuilt, "heic");
    let reference = restore_png(&jxl, &files.path("reference.png"), "pq", PQ_CICP);
    let restored = restore_png(&heic, &files.path("restored.png"), "pq", PQ_CICP);
    for ((x, y, restored), reference) in restored.enumerate_pixels().zip(reference.pixels()) {
        for (sample, expected) in restored.0.into_iter().zip(reference.0) {
            assert!(
                sample.abs_diff(expected) <= 300,
                "({x}, {y}): {sample} vs {expected}"
            );
        }
    }
}

#[test]
fn hdr_conversion_rejects_incompatible_inputs_and_outputs() {
    let files = TestDir::new("invalid");
    let sdr = files.path("source-sdr.png");
    let pq = files.path("source-pq.png");
    write_png(&sdr, SDR_CICP);
    write_png(&pq, PQ_CICP);

    for (input, name, args, diagnostic) in [
        (
            &sdr,
            "from-sdr.avif",
            &["--hdr-conversion", "gain-map"][..],
            "pq",
        ),
        (
            &sdr,
            "without-map.png",
            &["--hdr-conversion", "pq"],
            "gain map",
        ),
        (
            &pq,
            "unsupported.png",
            &["-e", "jxl", "--hdr-conversion", "gain-map"],
            "gain map",
        ),
        (
            &pq,
            "stripped.avif",
            &["--hdr-conversion", "gain-map", "-O", "av1:gain-map=false"],
            "gain map",
        ),
    ] {
        reject(input, &files.path(name), args, diagnostic);
    }
}
