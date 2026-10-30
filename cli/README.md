## maroontree-cli CLI

### CLI Encoder overrides

Set several AV1, JXL, HEIC, or JPEG parameters in one `-O` / `--override` flag, using commas
and one encoder prefix per group:

```sh
maroontree-cli -e jxl -O 'jxl:distance=3.1,alpha-distance=0,speed=mediumslow' input.png
maroontree-cli -e jxl -O 'jxl:progressive=true,patches=true' -O jxl:distance=4 input.png
maroontree-cli -e av1 -O 'av1:quality=85,adaptive-quant=false,cdef=true,qm=7' input.png
maroontree-cli -e jxl -O 'jxl:progressive-shifts=[2,1,0],grain=photon:3200,splines=true' input.png
maroontree-cli -e heic -O 'heic:quality=85,sao=false,parallelism=grid-wpp,persistent-rice=true' input.png
maroontree-cli -e heic -O 'heic:lossless=true,lossless-ycbcr=true,chroma=420,depth=10' input.png
maroontree-cli -e jxl --list-overrides
maroontree-cli -e av1 --list-overrides
maroontree-cli -e heic --list-overrides
maroontree-cli -e jpeg -O 'jpeg:progressive=true,polish=trellis-greedy,polish-judge=both' input.png output.jpg
maroontree-cli -e jpeg --list-overrides
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
