from __future__ import annotations

import json
from pathlib import Path

from llm.core.checksums import file_sha256

STAGE18 = Path("data/stage18")
V2_TOKENS = Path("data/v2/tokens")
ROOT = Path("data/stage20/r")
STREAMS = {
    f"b_{language}": STAGE18 / "b" / "tokens" / f"train.{language}.bin"
    for language in ("en", "ru", "kk")
}
REPLAY = {"v2": V2_TOKENS / "train.bin"}
MIXTURE = {"b_en": 0.25, "b_ru": 0.25, "b_kk": 0.25, "v2": 0.25}


def main() -> None:
    first = STAGE18 / "a" / "tokens"
    second = STAGE18 / "b" / "tokens"
    held_out = [path.name for path in first.glob("*.bin") if not path.name.startswith("train")]
    mismatched = [
        name for name in held_out if file_sha256(first / name) != file_sha256(second / name)
    ]
    if mismatched:
        raise ValueError(f"arm held-out streams differ: {mismatched}")
    tokens = ROOT / "tokens"
    tokens.mkdir(parents=True)
    for name, source in {**STREAMS, **REPLAY}.items():
        (tokens / f"train.{name}.bin").symlink_to(source.resolve())
    for name in held_out:
        (tokens / name).symlink_to((first / name).resolve())
    manifest = {
        "mixture": MIXTURE,
        "streams": {
            name: {"source": str(path), "sha256": file_sha256(path)}
            for name, path in {**STREAMS, **REPLAY}.items()
        },
        "held_out": {name: file_sha256(first / name) for name in sorted(held_out)},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"streams": sorted(STREAMS) + sorted(REPLAY), "held_out": len(held_out)}))


if __name__ == "__main__":
    main()
