#!/usr/bin/env python3
"""Convert explicit multiview review decisions into milestone supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.scripts.propose_flip_table_insertion_review import (
    _load_numeric_episodes,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-accepted", type=int, default=30)
    return parser.parse_args()


def _transition_mask(command: list[list[float]], radius: int = 10) -> list[bool]:
    transitions = [False] * len(command)
    for index in range(1, len(command)):
        if max(
            abs(float(command[index][side]) - float(command[index - 1][side]))
            for side in (0, 1)
        ) >= 0.02:
            for nearby in range(max(0, index - radius), min(len(command), index + radius + 1)):
                transitions[nearby] = True
    return transitions


def main() -> int:
    args = _arguments()
    if args.minimum_accepted < 1:
        raise ValueError("minimum accepted intervals must be positive")
    queue = json.loads(args.review_queue.read_text())
    if queue.get("automatic_labels_are_action_supervision") is not False:
        raise ValueError("review queue does not preserve the auto-label safety contract")
    proposals = {
        int(item["episode_index"]): item for item in queue.get("proposals", [])
    }
    decisions = json.loads(args.decisions.read_text())
    if decisions.get("schema_version") != "flip_table_insertion_review_decisions_v1":
        raise ValueError("unsupported review decision schema")
    reviewer = str(decisions.get("reviewer", "")).strip()
    reviewed_at = str(decisions.get("reviewed_at", "")).strip()
    if not reviewer or not reviewed_at:
        raise ValueError("reviewer and reviewed_at are required")
    accepted = decisions.get("accepted")
    if not isinstance(accepted, list) or len(accepted) < args.minimum_accepted:
        raise ValueError("too few accepted multiview intervals")

    numeric = _load_numeric_episodes(args.dataset_root.resolve())
    records = []
    seen: set[int] = set()
    for decision in accepted:
        episode_index = int(decision["episode_index"])
        if episode_index in seen:
            raise ValueError(f"duplicate review decision for episode {episode_index}")
        seen.add(episode_index)
        proposal = proposals.get(episode_index)
        values = numeric.get(episode_index)
        if proposal is None or values is None:
            raise ValueError(f"episode {episode_index} is absent from review inputs")
        m3 = int(decision.get("m3_frame", proposal["milestones"]["M3"]["frame"]))
        m4 = int(decision.get("m4_frame", proposal["milestones"]["M4"]["frame"]))
        length = int(proposal["length"])
        if not 0 <= m3 < m4 < length:
            raise ValueError(f"invalid reviewed M3/M4 for episode {episode_index}")
        records.append(
            {
                "episode_index": episode_index,
                "length": length,
                "milestones": {
                    "M3": {
                        "frame": m3,
                        "valid": True,
                        "source": "reviewed_phase_frames",
                    },
                    "M4": {
                        "frame": m4,
                        "valid": True,
                        "source": "reviewed_phase_frames",
                    },
                },
                "dex1_transition_mask": _transition_mask(
                    values["command"].tolist()
                ),
                "review": {
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "views": ["head_left", "left_wrist", "right_wrist"],
                    "decision": "accepted_right_reinsertion",
                    "source_queue": args.review_queue.name,
                },
            }
        )
    records.sort(key=lambda item: item["episode_index"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in records)
    )
    print(json.dumps({"accepted": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
