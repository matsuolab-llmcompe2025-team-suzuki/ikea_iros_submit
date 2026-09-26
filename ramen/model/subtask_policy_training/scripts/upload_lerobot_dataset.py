"""Upload a validated local LeRobot dataset tree to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = FEATURE_ROOT / "configs" / "subtask_training.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subtask")
    parser.add_argument("--repo-id")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--upload-large-folder", action="store_true")
    return parser.parse_args()


def resolve_repo_id(config_path: Path, subtask_override: str | None, repo_id_override: str | None) -> tuple[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    subtask = subtask_override or os.environ.get("SUBTASK") or str(config["subtask"])
    repo_id = repo_id_override or os.environ.get("DATASET_REPO_ID") or config["dataset_repo_template"].format(subtask=subtask, policy_type="act")
    return subtask, repo_id


def main() -> None:
    args = parse_args()
    subtask, repo_id = resolve_repo_id(args.config, args.subtask, args.repo_id)
    dataset = LeRobotDataset(repo_id=repo_id, root=args.root)
    dataset.push_to_hub(
        branch=args.branch,
        private=args.private,
        upload_large_folder=args.upload_large_folder,
        tags=["iros-2026", "humanoid", "ikea-assembly", subtask.replace("_", "-"), "team-ramen"],
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
