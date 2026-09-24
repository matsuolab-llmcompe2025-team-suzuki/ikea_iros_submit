#!/usr/bin/env python3
"""Derive a phase-plan-compatible sampler for reviewed insertion frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.gr00t.insertion_expert import (  # noqa: E402
    build_insertion_sampling_plan,
    load_jsonl_records,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-phase-plan", type=Path, required=True)
    parser.add_argument("--insertion-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dex1-transition-multiplier", type=float, default=4.0)
    args = parser.parse_args()

    base = json.loads(args.base_phase_plan.read_text(encoding="utf-8"))
    plan = build_insertion_sampling_plan(
        base,
        load_jsonl_records(args.insertion_sidecar),
        insertion_sidecar_sha256=sha256_file(args.insertion_sidecar),
        dex1_transition_multiplier=args.dex1_transition_multiplier,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
