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

use image::imageops::FilterType;
use maroontree::{
    Av2Encoder, BitDepth, ChromaFormat, ColorEncoding, EncodeConfig, PlanarImage, TxPart,
    av2_map_quality, encode_gray8, encode_gray10, encode_rgb8, encode_rgb10,
};
use std::io::Write;
use std::time::Instant;

fn main() {
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
    let img = image::open("./assets/spring_tree.png").unwrap().to_rgb8();
    let instant = Instant::now();
    let out = encode_rgb8(
        &PlanarImage::from_interleaved_rgb(
            img.width() as usize,
            img.height() as usize,
            BitDepth::Eight,
            &img,
        )
        .unwrap(),
        &EncodeConfig::new()
            .with_quality(60)
            .with_cicp(ColorEncoding::srgb_ycbcr())
            .with_chroma(ChromaFormat::Yuv444),
    )
    .unwrap();
    println!("encoding time {:?}", instant.elapsed());
    let img = image::open("./assets/spring_tree.png").unwrap().to_rgb8();
    let pimg = PlanarImage::from_interleaved_rgb(
        img.width() as usize,
        img.height() as usize,
        BitDepth::Eight,
        &img.to_vec(),
    )
    .unwrap();
    let instant = Instant::now();
    let av2_encoder = Av2Encoder::new(av2_map_quality(53))
        .with_tiles(512, 512)
        .with_txpart(TxPart::ThreeWay)
        .with_rdoq_lambda(0.09);
    let encoded = av2_encoder
        .encode_image_422(&pimg, &ColorEncoding::srgb_ycbcr(), 9)
        .unwrap();
    let out_obu = encoded.view();
    println!("encoding time {:?}", instant.elapsed());
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out10.avif".into());
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&out).unwrap();
    eprintln!("wrote {} bytes to {}", out.len(), path);

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out10_av2.obu".into());
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&out_obu).unwrap();

    let encoded_avif_av2 = Av2Encoder::wrap_avif(&encoded, None, None).unwrap();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out10_avif.avif".into());
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&encoded_avif_av2).unwrap();
}
