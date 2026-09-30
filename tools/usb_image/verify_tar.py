#!/usr/bin/env python3
"""USB の image の tar を 1 回読み直して確かめる (WEIGHTS.md 3)。

    python3 tools/usb_image/verify_tar.py <USB>/ikea-thor_<tag>.tar

tar 全体の sha256 (`SHA256SUMS.image` と比べる) と、中の blob が 1 つ残らず名前 (digest) どおりの中身かを見る。
`manifest.json` の RepoTags・config・層の数も出す。blob が 1 つでも合わなければ exit 1。
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile


class _HashingReader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        b = self.f.read(n)
        self.h.update(b)
        return b


def main() -> int:
    path = sys.argv[1]
    bad, checked, small = [], 0, {}
    with open(path, "rb") as raw:
        reader = _HashingReader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if member.name.startswith("blobs/sha256/"):
                    h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(8 << 20), b""):
                        h.update(chunk)
                    checked += 1
                    if h.hexdigest() != member.name.rsplit("/", 1)[-1]:
                        bad.append(member.name)
                else:
                    small[member.name] = f.read()
        for chunk in iter(
            lambda: reader.read(8 << 20), b""
        ):  # tar の末尾の 0 埋めまで含めて
            pass
    legacy = json.loads(small["manifest.json"])[0]
    index = json.loads(small["index.json"])["manifests"][0]
    print(f"{reader.h.hexdigest()}  {path.rsplit('/', 1)[-1]}")
    print(f"blobs {checked}, bad {bad or 'none'}")
    print(
        f"RepoTags {legacy['RepoTags']}, config {legacy['Config'][13:25]}, layers {len(legacy['Layers'])}, "
        f"manifest {index['digest'][7:19]}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
