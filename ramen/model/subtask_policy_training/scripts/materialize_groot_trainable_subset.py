#!/usr/bin/env python3
"""Materialize a recoverable expert candidate over its immutable warm start."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(base: Path, subset: Path, output: Path) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = base.expanduser().resolve()
    subset = subset.expanduser().resolve()
    output = output.expanduser().resolve()
    base_weights = base / "model.safetensors"
    subset_manifest_path = subset.with_suffix(".json")
    for required in (base / "config.json", base_weights, subset, subset_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise FileExistsError(f"refusing to replace candidate output: {output}")

    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    if subset_manifest.get("schema_version") != 1:
        raise ValueError("unsupported trainable-subset manifest")
    if subset_manifest.get("sha256") != sha256(subset):
        raise ValueError("trainable-subset hash mismatch")
    approved_prefixes = tuple(subset_manifest.get("approved_prefixes", ()))
    if not approved_prefixes:
        raise ValueError("trainable-subset manifest lacks approved prefixes")

    with safe_open(base_weights, framework="pt", device="cpu") as base_file:
        base_keys = set(base_file.keys())
        base_shapes = {key: tuple(base_file.get_slice(key).get_shape()) for key in base_keys}
        base_metadata = base_file.metadata()
    with safe_open(subset, framework="pt", device="cpu") as subset_file:
        subset_keys = set(subset_file.keys())
        if not subset_keys:
            raise ValueError("trainable subset is empty")
        unknown = sorted(subset_keys - base_keys)
        if unknown:
            raise ValueError(f"trainable subset has unknown tensors: {unknown[:10]}")
        escaped = sorted(
            key for key in subset_keys if not key.startswith(approved_prefixes)
        )
        if escaped:
            raise ValueError(f"trainable subset escaped approved prefixes: {escaped[:10]}")
        for key in subset_keys:
            if tuple(subset_file.get_slice(key).get_shape()) != base_shapes[key]:
                raise ValueError(f"trainable subset shape mismatch for {key}")

    shutil.copytree(base, output, ignore=shutil.ignore_patterns("model.safetensors"))
    target = output / "model.safetensors"
    temporary = target.with_suffix(target.suffix + ".tmp")
    tensors = {}
    with safe_open(base_weights, framework="pt", device="cpu") as base_file, safe_open(
        subset, framework="pt", device="cpu"
    ) as subset_file:
        for key in sorted(base_keys):
            source = subset_file if key in subset_keys else base_file
            tensors[key] = source.get_tensor(key)
    save_file(tensors, temporary, metadata=base_metadata)
    temporary.replace(target)

    manifest = {
        "schema_version": "team_ramen_materialized_expert_v1",
        "base": str(base),
        "base_weights_sha256": sha256(base_weights),
        "subset": str(subset),
        "subset_weights_sha256": subset_manifest["sha256"],
        "output_weights_sha256": sha256(target),
        "replaced_tensor_count": len(subset_keys),
        "replaced_parameter_count": int(subset_manifest["parameter_count"]),
        "approved_prefixes": list(approved_prefixes),
        "optimizer_state_transferred": False,
    }
    (output / "expert_materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    print(json.dumps(materialize(args.base, args.subset, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
