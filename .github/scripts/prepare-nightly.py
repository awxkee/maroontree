"""Use encoder Git default-branch heads for a nightly checkout only."""

from pathlib import Path
import re
import sys


def prepare(manifest: Path) -> None:
    source = manifest.read_text()
    for name in ("jixel", "hpvca", "garnetash", "jhm17"):
        # Preserve optional/features attributes, but remove the release version
        # constraint so a new upstream minor/major cannot fall back to crates.io.
        pattern = rf"(?m)^{name}\s*=\s*(.+)$"
        matches = list(re.finditer(pattern, source))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {name} dependency")
        value = matches[0].group(1).strip()
        git = f'git = "https://github.com/awxkee/{name}"'
        if value.startswith("{"):
            value, count = re.subn(r'(?:version|git)\s*=\s*"[^"]+"', git, value, count=1)
            if count != 1:
                raise ValueError(f"missing release version or Git source for {name}")
        elif re.fullmatch(r'"[^"]+"', value):
            value = "{ " + git + " }"
        else:
            raise ValueError(f"unsupported dependency declaration: {name}")
        source = re.sub(pattern, lambda _: f"{name} = {value}", source)
    manifest.write_text(source)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cli/Cargo.toml"))
