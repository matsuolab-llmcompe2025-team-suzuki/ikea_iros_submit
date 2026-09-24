"""Download source MCAP recordings for subtask data diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = FEATURE_ROOT / "configs" / "subtask_training.json"
SOURCE_REPO_TEMPLATE = "Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1"


def load_subtask(config_path: Path) -> str:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config["subtask"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subtask")
    parser.add_argument("--repo-id")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-mcaps", type=int, default=0)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subtask = args.subtask or os.environ.get("SUBTASK") or load_subtask(args.config)
    repo_id = args.repo_id or SOURCE_REPO_TEMPLATE.format(subtask=subtask)
    output_dir = args.output_dir or FEATURE_ROOT / "data" / f"{subtask}_mcap"

    api = HfApi()
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.revision,
    )
    wanted = ["segments.jsonl", "summary.json"]
    mcaps = sorted(path for path in files if path.startswith("mcap/") and path.endswith(".mcap"))
    if args.max_mcaps:
        mcaps = mcaps[: args.max_mcaps]
    wanted.extend(mcaps)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in tqdm(wanted, desc="download"):
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename=filename,
            local_dir=output_dir,
        )

    print(f"Downloaded {len(mcaps)} MCAP files from {repo_id} to {output_dir}")


if __name__ == "__main__":
    main()
