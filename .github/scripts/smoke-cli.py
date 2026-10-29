"""Exercise the built CLI with a generated image; no image tooling required."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


binary = str(Path(sys.argv[1]).resolve())
subprocess.run([binary, "--version"], check=True)
with tempfile.TemporaryDirectory(prefix="maroontree-smoke-") as directory:
    root = Path(directory)
    source = root / "input.png"
    pixels = b"".join(
        b"\0" + b"".join(bytes([x * 31, y * 31, 128]) for x in range(8)) for y in range(8)
    )
    source.write_bytes(b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    for encoder, extension in [("av1", "avif"), ("av2", "avif"), ("jxl", "jxl"), ("hevc", "heic"), ("vvc", "heif")]:
        version = subprocess.check_output([binary, "-e", encoder, "--encoder-version"], text=True)
        if "disabled at build time" in version:
            continue
        for lossless in [False, True]:
            output = root / f"{encoder}-{'lossless' if lossless else 'lossy'}.{extension}"
            options = [binary, "-e", encoder, "-s", "fast", "-t", "1"]
            if lossless:
                options.append("--lossless")
            if encoder == "jxl":
                options += ["-override", "jxl:-d:31,-s:fast,-t:1,orientation=6,splines=false,dots=false"]
                if not lossless:
                    options += ["-O", "jxl:progressive-shifts=[2,1,0],grain=photon:3200"]
            elif encoder == "av1":
                options += ["--override", "av1:updating-cdf=false,speed=fast,threads=1,orientation=6,max-cll=1000,max-fall=400"]
            elif encoder == "hevc":
                options += ["-O", "heic:parallelism=single,sao=false,persistent-rice=false,implicit-rdpcm=false,orientation=6,max-cll=1000,max-fall=400"]
                if lossless:
                    options += ["-O", "hevc:lossless-ycbcr=true,chroma=420"]
            subprocess.run(options + [str(source), str(output)], check=True)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"empty output for {encoder}, lossless={lossless}")
