## maroontree-cli CLI

### CLI Encoder overrides

Set several AV1, JXL, HEIC, or JPEG parameters in one `-O` / `--override` flag, using commas
and one encoder prefix per group:

```sh
maroontree-cli -e jxl -O 'jxl:distance=3.1,alpha-distance=0,speed=mediumslow' input.png
maroontree-cli -e jxl -O 'jxl:progressive=true,patches=true' -O jxl:distance=4 input.png
maroontree-cli -e av1 -O 'av1:quality=85,adaptive-quant=false,cdef=true,qm=7' input.png
maroontree-cli -e jxl -O 'jxl:progressive-shifts=[2,1,0],grain=photon:3200,splines=true' input.png
maroontree-cli -e jxl -O 'jxl:distance=2,speed=slow,edge-splines=true,grain=auto' input.png
maroontree-cli -e heic -O 'heic:quality=85,sao=false,parallelism=grid-wpp,persistent-rice=true' input.png
maroontree-cli -e heic -O 'heic:lossless=true,lossless-ycbcr=true,chroma=420,depth=10' input.png
maroontree-cli -e jxl --list-overrides
maroontree-cli -e av1 --list-overrides
maroontree-cli -e heic --list-overrides
maroontree-cli -e jpeg -O 'jpeg:progressive=true,polish=trellis-greedy,polish-judge=both' input.png output.jpg
maroontree-cli -e jpeg --list-overrides
```

### HDR conversion

`--hdr-conversion pq` and `--hdr-conversion hlg` convert an embedded gain map or a PQ- or HLG-tagged image to BT.2020 PQ or HLG. PQ and HLG output use 10 bits or more. PNG output uses 16 bits.

`--hdr-conversion gain-map` accepts PQ and HLG images and writes an SDR base with an embedded gain map. AV1 AVIF, JPEG XL, and HEIC support all three HDR representations as inputs and outputs. PQ- and HLG-tagged PNGs are also accepted as inputs. HEIC requires the `heic` build feature.

Without `--hdr-conversion`, transcodes between the three formats preserve the source HDR representation. When a gain-map destination requires a different map, the CLI reconstructs the HDR image and generates a compatible map and SDR base. If you omit `--encoder`, the output extension selects the encoder.

```sh
maroontree-cli --hdr-conversion hlg photo-gain-map.avif photo-hlg.jxl
maroontree-cli --hdr-conversion gain-map photo-hlg.jxl photo-gain-map.heic
maroontree-cli --hdr-conversion pq photo-gain-map.heic photo-pq.avif
```

HLG conversion targets the BT.2100 1,000 cd/m² reference display. Highlights above that range are limited when converting to HLG. Gain-map conversion uses a 203 cd/m² reference white. [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en), [ITU-R BT.2408](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf)

### AVIF decoder builds

The CLI's ISOBMFF reader extracts AVIF image items for rav1d. Default builds, including ordinary debug builds, use rav1d's Rust implementation. CLI release builds in CI enable `avif-asm`. To enable the same optimizations locally, run `cargo build -p maroontree-cli --release --features avif-asm`. Install NASM for x86 targets. [rav1d build instructions](https://github.com/memorysafety/rav1d/blob/main/README.md)

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.