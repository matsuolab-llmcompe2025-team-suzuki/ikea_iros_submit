#!/usr/bin/env python3
"""fetch_blobs.sh が落とした blob から、`docker save` と同じ形の tar を組み立てる (WEIGHTS.md 3)。

    python3 tools/usb_image/assemble_tar.py <out_dir> <USB>/ikea-thor_<tag>.tar

形は 2026-09-29 に Thor で `docker load` できた `ikea-thor_gb10-test-e1dff41.tar` (Mac の `docker save`) と同じ:
`blobs/` → `blobs/sha256/` → blob (digest の順、0444) → `index.json`・`manifest.json` (0644) → `oci-layout` (0444)、
uid/gid 0・mtime 0・PAX。層は GHCR の圧縮のまま入れる (`docker load` が解凍する)。
書きながら tar 全体の sha256 を取り、`SHA256SUMS.image` の 1 行の形で出す。書き終わったら verify_tar.py で読み直す。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile


class _HashingWriter(io.RawIOBase):
    """書いた bytes の sha256 を取りながら、下の file に渡す。"""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)


def _info(
    name: str, size: int = 0, mode: int = 0o444, kind: bytes = tarfile.REGTYPE
) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.type, ti.mtime = size, mode, kind, 0
    ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, "", ""
    return ti


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    repo, tag, man_hex = open(f"{src}/image.txt").read().split()
    man_raw = open(f"{src}/blobs/{man_hex}", "rb").read()
    if hashlib.sha256(man_raw).hexdigest() != man_hex:
        sys.exit("manifest の digest が合わない")
    man = json.loads(man_raw)
    config_hex = man["config"]["digest"][7:]
    blobs = {man_hex: len(man_raw), config_hex: man["config"]["size"]}
    for layer in man["layers"]:
        blobs[layer["digest"][7:]] = layer["size"]
    for hexd, size in blobs.items():
        if os.path.getsize(f"{src}/blobs/{hexd}") != size:
            sys.exit(f"大きさが合わない: {hexd} (fetch_blobs.sh をもう一度)")

    name = f"ghcr.io/{repo}:{tag}"
    index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": f"sha256:{man_hex}",
                "size": len(man_raw),
                "annotations": {
                    "containerd.io/distribution.source.ghcr.io": repo,
                    "io.containerd.image.name": name,
                    "org.opencontainers.image.ref.name": tag,
                },
            }
        ],
    }
    legacy = [
        {
            "Config": f"blobs/sha256/{config_hex}",
            "RepoTags": [name],
            "Layers": [
                f"blobs/sha256/{layer['digest'][7:]}" for layer in man["layers"]
            ],
        }
    ]
    small = {
        "index.json": (json.dumps(index, separators=(",", ":")).encode(), 0o644),
        "manifest.json": (json.dumps(legacy, separators=(",", ":")).encode(), 0o644),
        "oci-layout": (b'{"imageLayoutVersion":"1.0.0"}', 0o444),
    }

    part = out + ".part"
    with open(part, "wb") as raw:
        hw = _HashingWriter(raw)
        with tarfile.open(fileobj=hw, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            tar.addfile(_info("blobs", mode=0o755, kind=tarfile.DIRTYPE))
            tar.addfile(_info("blobs/sha256", mode=0o755, kind=tarfile.DIRTYPE))
            for hexd in sorted(blobs):
                with open(f"{src}/blobs/{hexd}", "rb") as f:
                    tar.addfile(_info(f"blobs/sha256/{hexd}", size=blobs[hexd]), f)
            for member, (data, mode) in small.items():
                tar.addfile(_info(member, size=len(data), mode=mode), io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(part, out)
    print(f"{hw.h.hexdigest()}  {os.path.basename(out)}")
    print(
        f"{name}: {hw.n} bytes, {len(blobs)} blobs ({len(man['layers'])} layers), "
        f"config {config_hex[:12]}, manifest {man_hex[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
