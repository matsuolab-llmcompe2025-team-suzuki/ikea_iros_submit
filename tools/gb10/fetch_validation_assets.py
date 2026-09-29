#!/usr/bin/env python3
"""Fetch pinned official test code and hash-checked G1 model assets, not weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


ORGANIZER = "497f3ab93e5baa706311daebd31c7a9798258450"
WBC = "a0732b642c0333077e127a2f56ab0014c196bca4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    def git(*cmd):
        return subprocess.check_output(["git", *map(str, cmd)], text=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}).strip()

    organizer, wbc = args.root / "organizer", args.root / "wbc"
    if not organizer.exists():
        git("clone", "https://github.com/iacevaltest/iros_g1_orin_package.git", organizer)
    git("-C", organizer, "checkout", "--detach", ORGANIZER)
    if not wbc.exists():
        git("clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/NVlabs/GR00T-WholeBodyControl.git", wbc)
        git("-C", wbc, "sparse-checkout", "set", "decoupled_wbc/control")
    git("-C", wbc, "checkout", "--detach", WBC)
    records = []
    model_path = "decoupled_wbc/control/robot_model/model_data/g1"
    for name in git("-C", wbc, "ls-files", model_path).splitlines():
        pointer = subprocess.check_output(["git", "-C", str(wbc), "show", f"{WBC}:{name}"])
        if not pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            continue
        values = dict(line.split(" ", 1) for line in pointer.decode().splitlines()[1:])
        digest = values["oid"].removeprefix("sha256:")
        size = int(values["size"])
        if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid LFS hash")
        path = wbc / name
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            url = f"https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/{WBC}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"LFS integrity failure: {name}")
            path.write_bytes(data)
        records.append({"path": name, "bytes": size, "sha256": digest})
    if not records:
        raise ValueError("No G1 LFS model assets found")
    (args.root / "assets-manifest.json").write_text(json.dumps({
        "organizer_revision": ORGANIZER, "wbc_revision": WBC, "assets": records}, indent=2)+"\n")
    print(json.dumps({"assets": len(records), "bytes": sum(x["bytes"] for x in records)}))


if __name__ == "__main__":
    main()
