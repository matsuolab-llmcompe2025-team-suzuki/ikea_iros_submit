"""Resolve a leakage-safe episode split into LeRobot CLI arguments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any


SPLIT_PATH = Path("meta/team_ramen_episode_split.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve leakage-safe LeRobot train/eval episodes.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    return parser.parse_args()


def resolve_split(dataset_root: Path) -> dict[str, Any]:
    split_path = dataset_root / SPLIT_PATH
    if not split_path.is_file():
        raise FileNotFoundError(f"training split manifest is missing: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "team_ramen_grouped_episode_split_v1":
        raise ValueError(f"unsupported training split schema: {payload.get('schema_version')!r}")
    expected_sha = payload.get("sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not isinstance(expected_sha, str) or expected_sha != actual_sha:
        raise ValueError("training split manifest SHA-256 mismatch")

    splits = payload.get("splits", {})
    episode_sets: dict[str, set[int]] = {}
    source_sets: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        values = splits.get(name, {}).get("episode_indices")
        if not isinstance(values, list) or not all(isinstance(value, int) and value >= 0 for value in values):
            raise ValueError(f"split {name!r} must contain non-negative episode indices")
        if len(values) != len(set(values)):
            raise ValueError(f"split {name!r} contains duplicate episode indices")
        episode_sets[name] = set(values)
        source_names = splits.get(name, {}).get("source_episode_names")
        if not isinstance(source_names, list) or not all(
            isinstance(value, str) and value for value in source_names
        ):
            raise ValueError(f"split {name!r} must contain source episode names")
        if len(source_names) != len(set(source_names)):
            raise ValueError(f"split {name!r} contains duplicate source episode names")
        source_sets[name] = set(source_names)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = episode_sets[left] & episode_sets[right]
        if overlap:
            raise ValueError(f"episode leakage between {left} and {right}: {sorted(overlap)[:10]}")
        source_overlap = source_sets[left] & source_sets[right]
        if source_overlap:
            raise ValueError(
                f"source-recording leakage between {left} and {right}: {sorted(source_overlap)[:10]}"
            )
    if not episode_sets["train"] or not episode_sets["validation"]:
        raise ValueError("training and validation splits must both be non-empty")

    train = sorted(episode_sets["train"])
    validation = sorted(episode_sets["validation"])
    selected = train + validation
    eval_split = len(validation) / len(selected)
    if math.ceil(len(selected) * eval_split) != len(validation):
        eval_split = (len(validation) - 0.5) / len(selected)
    if math.ceil(len(selected) * eval_split) != len(validation):
        raise RuntimeError("could not represent validation split for LeRobot's ceil-based splitter")

    return {
        "DATASET_EPISODES_JSON": json.dumps(selected, separators=(",", ":")),
        "DATASET_EVAL_SPLIT": f"{eval_split:.17g}",
        "TRAIN_EPISODE_COUNT": len(train),
        "VALIDATION_EPISODE_COUNT": len(validation),
        "TEST_EPISODE_COUNT": len(episode_sets["test"]),
        "SPLIT_SHA256": expected_sha,
    }


def main() -> None:
    args = parse_args()
    values = resolve_split(args.dataset_root.resolve())
    if args.format == "json":
        print(json.dumps(values, indent=2, sort_keys=True))
        return
    for key, value in values.items():
        print(f"export {key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
