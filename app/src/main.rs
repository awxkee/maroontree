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
mod y4m2ivf;

use maroontree::{
    Av2Encoder, BitDepth, ChromaFormat, Cicp, EncodeConfig, Orientation, PlanarImage, Speed,
    TxPart, av2_map_quality, encode_rgb8,
};
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

fn main() {
    // y4m2ivf::convert(
    //     "./assets/file_example_MP4_480_1_5MG.y4m".to_string(),
    //     "./file_example_MP4_480_1_5MG.ivf".to_string(),
    //     av2_map_quality(70),
    // );
    // let (w, h) = (64usize, 64usize);
    // let mut rgb = vec![0u8; w * h * 3];
    // for y in 0..h {
    //     for x in 0..w {
    //         let i = (y * w + x) * 3;
    //         rgb[i] = 0;
    //         rgb[i + 1] = 0;
    //         rgb[i + 2] = 255;
    //     }
    // }

    // let instant = Instant::now();
    // img.save("dst_rav.avif").unwrap();
    // println!("encoding time {:?}", instant.elapsed());
    let img = image::open("./assets/manhattan.png").unwrap().to_rgb8();
    let planar_rgb = PlanarImage::from_interleaved_rgb(
        img.width() as usize,
        img.height() as usize,
        BitDepth::Eight,
        &img,
    )
    .unwrap();
    for _i in 0..15 {
        let instant = Instant::now();
        let out = encode_rgb8(
            &planar_rgb,
            &EncodeConfig::new()
                .with_quality(67)
                .with_cicp(Cicp::srgb_ycbcr())
                .with_chroma(ChromaFormat::Yuv420)
                .with_speed(Speed::Fast)
                .with_threads(12)
                .with_variance_boost(true),
            // .with_cdef(true)
            // .with_wiener(true),
        )
        .unwrap();
        println!("AV1 encoding time {:?}", instant.elapsed());
        std::fs::write("./out.avif", out).unwrap();
    }
    let img = image::open("./assets/manhattan.png")
        .unwrap()
        // .resize_exact(1600, 900, FilterType::Nearest)
        .to_rgb8();
    let pimg = PlanarImage::from_interleaved_rgb(
        img.width() as usize,
        img.height() as usize,
        BitDepth::Eight,
        &img,
    )
    .unwrap();
    let av2_encoder = Av2Encoder::with_bit_depth(av2_map_quality(60), 8)
        .with_tiles(8, 8)
        .with_txpart(TxPart::ThreeWay)
        .with_rdoq_lambda(0.09)
        .with_speed(Speed::Fast)
        .with_threads(12)
        .with_cfl(true)
        .with_updating_cdf(true)
        .with_chroma_mode_search(true)
        .with_chroma_rdoq_lambda(0.05)
        .with_ccso(false)
        .with_cdef(false)
        .with_mhccp(true);
    let encoded = av2_encoder
        .encode_image_444(black_box(&pimg), &Cicp::srgb_ycbcr())
        .unwrap();
    for _i in 0..10 {
        let instant = Instant::now();
        let _encoded = av2_encoder
            .encode_image_420(black_box(&pimg), &Cicp::srgb_ycbcr())
            .unwrap();
        println!("AV2 Encoded in {}ms", instant.elapsed().as_millis());
    }
    // let out_obu = encoded.view();
    // let path = std::env::args()
    //     .nth(1)
    //     .unwrap_or_else(|| "out10.avif".into());
    // let mut f = std::fs::File::create(&path).unwrap();
    // f.write_all(&out).unwrap();
    // eprintln!("wrote {} bytes to {}", out.len(), path);
    //
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out10_av2.obu".into());
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(encoded.view()).unwrap();
    //
    let encoded_avif_av2 =
        Av2Encoder::wrap_avif(&encoded, None, None, Orientation::Normal, None).unwrap();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out10_avif.avif".into());
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&encoded_avif_av2).unwrap();
}
