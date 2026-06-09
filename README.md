# maroontre

Still images AV1 and AV2 AVIF encoder in Rust.

## Example

```rust
fn main() {
    let img = image::open("./assets/image.png")
        .unwrap()
        .to_rgba8();
    let arr = img.to_vec();

    let out = encode_rgb8(
        &img.to_vec(),
        img.width(),
        img.height(),
        &EncodeConfig::new()
            .with_quality(60)
            .with_cicp(ColorEncoding::srgb_ycbcr())
            .with_chroma(ChromaFormat::Yuv444),
    )
        .unwrap();
    std::fs::write("output.heic", &data).expect("failed to write output");
}
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.