#!/usr/bin/env python3
"""Build immutable, classifier-safe M3-M4 supervision from reviewed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.gr00t.insertion_expert import (
    SCHEMA_VERSION,
    build_hard_negative_supervision,
    build_finish_supervision,
    build_success_supervision,
    validate_supervision,
)


MANIFEST_SCHEMA_VERSION = "flip_table_insertion_supervision_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-sidecar", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--eef-audit", type=Path)
    parser.add_argument("--hard-negatives", type=Path)
    parser.add_argument("--minimum-success-intervals", type=int, default=30)
    parser.add_argument("--expert-role", choices=("insertion", "finish"), default="insertion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_success_intervals < 30:
        raise ValueError("the insertion expert requires at least 30 success intervals")
    progress = _load_jsonl_by_episode(args.progress_sidecar)
    trajectory_types = _load_trajectory_types(args.trajectory_manifest)
    eef_eligibility = _load_eef_eligibility(args.eef_audit)

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for episode_index, annotation in sorted(progress.items()):
        trajectory_type = trajectory_types.get(episode_index)
        if trajectory_type not in {"full_success", "recovery_success"}:
            continue
        try:
            builder = (
                build_success_supervision
                if args.expert_role == "insertion"
                else build_finish_supervision
            )
            record = builder(
                annotation,
                trajectory_type=trajectory_type,
                eef_loss_eligible=eef_eligibility.get(episode_index, True),
            ).as_dict()
            validate_supervision(record)
            records.append(record)
        except ValueError as exc:
            rejected.append({"episode_index": episode_index, "reason": str(exc)})

    if args.hard_negatives is not None:
        for item in _load_jsonl(args.hard_negatives):
            record = build_hard_negative_supervision(
                episode_index=int(item["episode_index"]),
                length=int(item["length"]),
                reason=str(item["reason"]),
                left_support_ready=item["left_support_ready"],
                right_insert_complete=item["right_insert_complete"],
            ).as_dict()
            validate_supervision(record)
            records.append(record)

    success_count = sum(
        item["trajectory_type"] in {"full_success", "recovery_success"}
        for item in records
    )
    if success_count < args.minimum_success_intervals:
        raise RuntimeError(
            "existing reviewed data is insufficient: "
            f"{success_count} insertion successes < {args.minimum_success_intervals}"
        )

    records.sort(key=lambda item: (int(item["episode_index"]), item["trajectory_type"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "supervision_schema_version": SCHEMA_VERSION,
        "dataset_revision": args.dataset_revision,
        "annotation_file": args.output.name,
        "annotation_sha256": _sha256(args.output),
        "source_files": {
            "progress_sidecar": _source(args.progress_sidecar),
            "trajectory_manifest": _source(args.trajectory_manifest),
            "eef_audit": _source(args.eef_audit),
            "hard_negatives": _source(args.hard_negatives),
        },
        "success_interval_count": success_count,
        "expert_role": args.expert_role,
        "hard_negative_count": len(records) - success_count,
        "minimum_success_intervals": args.minimum_success_intervals,
        "rejected": rejected,
        "action_teacher": "successful_M3_M4_only",
        "hard_negatives_are_action_teacher": False,
        "eef_loss_masked_independently": True,
        "policy_inputs": [],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(value)
    return records


def _load_jsonl_by_episode(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in _load_jsonl(path):
        episode_index = int(item["episode_index"])
        if episode_index in result:
            raise ValueError(f"duplicate episode {episode_index} in {path}")
        result[episode_index] = item
    return result


def _load_trajectory_types(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        records = _load_jsonl(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value.get("episodes", value) if isinstance(value, dict) else value
        if not isinstance(records, list):
            raise ValueError("trajectory manifest must contain an episodes list")
    result: dict[int, str] = {}
    for item in records:
        episode_index = int(item["episode_index"])
        trajectory_type = str(item["trajectory_type"])
        if episode_index in result:
            raise ValueError(f"duplicate trajectory episode {episode_index}")
        result[episode_index] = trajectory_type
    return result


def _load_eef_eligibility(path: Path | None) -> dict[int, bool | list[bool]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    records = (
        value.get("episodes", value.get("per_episode"))
        if isinstance(value, dict)
        else None
    )
    if not isinstance(records, list):
        raise ValueError("EEF audit must contain an episodes list")
    result: dict[int, bool | list[bool]] = {}
    for item in records:
        episode_index = int(item["episode_index"])
        eligible = item.get(
            "eef_loss_eligible",
            item.get("action_fk_residual_pass", item.get("passed")),
        )
        if not isinstance(eligible, (bool, list)):
            raise ValueError(f"EEF eligibility is missing for episode {episode_index}")
        result[episode_index] = eligible
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {"path": path.resolve().as_posix(), "sha256": _sha256(path)}


if __name__ == "__main__":
    main()
