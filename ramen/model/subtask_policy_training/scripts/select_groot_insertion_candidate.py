#!/usr/bin/env python3
"""Select an insertion expert from held-out real-data offline evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="STEP=REPORT_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_candidate(value: str) -> tuple[int, Path]:
    step_text, separator, path_text = value.partition("=")
    if not separator:
        raise ValueError("candidate must use STEP=REPORT_JSON")
    step = int(step_text)
    if step not in {500, 1000, 2000}:
        raise ValueError("insertion candidate step must be 500, 1000, or 2000")
    return step, Path(path_text)


def summarize(step: int, path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    aggregate = report.get("aggregate")
    if report.get("schema_version") != "groot_n17_offline_chunk_reset_v2" or not isinstance(
        aggregate, dict
    ):
        raise ValueError(f"unsupported offline report: {path}")
    left = aggregate.get("left_support_classifier")
    right = aggregate.get("right_insert_classifier")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError(f"offline report lacks insertion classifier metrics: {path}")
    metrics = {
        "left_support_f1": float(left["f1"]),
        "right_insert_f1": float(right["f1"]),
        "arm_p95_error_rad": float(aggregate["arm_p95_error_rad"]),
        "dex1_transition_f1": float(aggregate["dex1_transition_f1"]),
        "replan_discontinuity_p99_rad": float(
            aggregate["replan_discontinuity_p99_rad"]
        ),
        "nonfinite_values": int(aggregate["nonfinite_values"]),
        "official_limit_violations": int(aggregate["official_limit_violations"]),
    }
    if not all(
        math.isfinite(value)
        for name, value in metrics.items()
        if name not in {"nonfinite_values", "official_limit_violations"}
    ):
        raise ValueError(f"offline report contains non-finite metrics: {path}")
    failures = []
    gates = (
        (metrics["left_support_f1"] >= 0.90, "left_support_f1"),
        (metrics["right_insert_f1"] >= 0.90, "right_insert_f1"),
        (metrics["arm_p95_error_rad"] <= 0.08, "arm_p95_error_rad"),
        (metrics["dex1_transition_f1"] >= 0.85, "dex1_transition_f1"),
        (
            metrics["replan_discontinuity_p99_rad"] <= 0.05,
            "replan_discontinuity_p99_rad",
        ),
        (metrics["nonfinite_values"] == 0, "nonfinite_values"),
        (metrics["official_limit_violations"] == 0, "official_limit_violations"),
    )
    failures.extend(name for passed, name in gates if not passed)
    return {
        "step": step,
        "report": str(path.resolve()),
        "trainable_subset": report.get("trainable_subset"),
        "metrics": metrics,
        "gate_failures": failures,
        "offline_gate_passed": not failures,
    }


def select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidates) != 3 or {item["step"] for item in candidates} != {
        500,
        1000,
        2000,
    }:
        raise ValueError("selection requires exactly the 500/1000/2000 candidates")

    def rank(item: dict[str, Any]) -> tuple[Any, ...]:
        metrics = item["metrics"]
        return (
            not item["offline_gate_passed"],
            -min(metrics["left_support_f1"], metrics["right_insert_f1"]),
            -metrics["dex1_transition_f1"],
            metrics["arm_p95_error_rad"],
            metrics["replan_discontinuity_p99_rad"],
            item["step"],
        )

    ordered = sorted(candidates, key=rank)
    selected = ordered[0]
    return {
        "schema_version": "team_ramen_groot_insertion_selection_v1",
        "selection_scope": "heldout_real_data_offline_not_physical_success",
        "selected_step": selected["step"],
        "offline_gate_passed": selected["offline_gate_passed"],
        "physical_release_eligible": False,
        "physical_release_blocker": "requires insertion 5/5, nominal 5/5, randomized 18/20",
        "candidates": sorted(candidates, key=lambda item: item["step"]),
    }


def main() -> int:
    args = parse_args()
    parsed = [parse_candidate(value) for value in args.candidate]
    result = select([summarize(step, path) for step, path in parsed])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
