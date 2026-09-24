"""Restore the canonical GR00T base-model path before publishing a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--canonical-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changed = restore_base_model_paths(
        args.output_dir,
        runtime_path=args.runtime_path,
        canonical_path=args.canonical_path,
    )
    print(f"Restored portable GR00T base_model_path in {changed} config file(s)")


def restore_base_model_paths(root: Path, *, runtime_path: str, canonical_path: str) -> int:
    changed = 0
    for path in sorted(root.rglob("config.json")):
        config = read_json(path)
        if config.get("type") != "groot" or config.get("base_model_path") != runtime_path:
            continue
        config["base_model_path"] = canonical_path
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        changed += 1
    return changed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
