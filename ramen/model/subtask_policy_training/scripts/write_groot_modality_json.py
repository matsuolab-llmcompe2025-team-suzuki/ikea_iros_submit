"""Regenerate the checked GR00T REAL_G1 modality metadata for diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

STANDARD_POLICY_CAMERAS = ("head_left", "left_wrist", "right_wrist")
FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = FEATURE_ROOT / "configs" / "subtask_training.json"
MAPPING_PATH = FEATURE_ROOT / "gr00t" / "g1_full_body_mapping.py"


def load_mapping_module() -> Any:
    spec = importlib.util.spec_from_file_location("g1_full_body_mapping", MAPPING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load mapping module: {MAPPING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mapping = load_mapping_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    available_cameras = set(config["cameras"])
    camera_names = [name for name in STANDARD_POLICY_CAMERAS if name in available_cameras]

    video_keys = [f"observation.images.{name}" for name in camera_names]
    modality = _mapping.build_real_g1_relative_eef_modality_json(video_keys)

    meta_dir = args.dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out = meta_dir / "modality.json"
    out.write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
