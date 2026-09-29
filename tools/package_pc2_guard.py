"""Bundle the read-only PC2 guard from the generated source, without dependencies."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
import tempfile


def package(root: Path, destination: Path) -> None:
    source = root / "ramen/inference/desktop/perception"
    names = ("venue_state_guard.py", "configs/pc2_g1_3_internal.json")
    with tempfile.TemporaryDirectory() as tmp:
        checksums = Path(tmp) / "SHA256SUMS"
        checksums.write_text("".join(
            f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ))
        with tarfile.open(destination, "w:gz") as archive:
            for name in names:
                archive.add(source / name, arcname=name)
            archive.add(root / "ramen/RAMEN_SOURCE.txt", arcname="RAMEN_SOURCE.txt")
            archive.add(checksums, arcname="SHA256SUMS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], args.output)
