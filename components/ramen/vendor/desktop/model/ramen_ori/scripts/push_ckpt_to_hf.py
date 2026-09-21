"""ckpt 1 個を HF Hub に push する (Issue #122 の autopush。Issue #141 で学習の process から切り離した)。

学習 (`train.py`) が `background_push.BackgroundCkptPusher` 経由で別 process として起動する。
学習の process の中で upload すると、3.5 GB の ckpt を 100 Mbps で上げる約 300 s の間、学習が止まっていた。
この script は model の package も torch も import しないので、起動は軽い。

CLI:
    python model/ramen_ori/scripts/push_ckpt_to_hf.py --ckpt <path> --repo-id <org/name> --step <int> [--public]

失敗しても例外は投げず、WARN を出して exit 1 (学習は止めない)。
HF の token は環境変数 HF_TOKEN から huggingface_hub が読む。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def push_ckpt(ckpt_path: Path, repo_id: str, step: int, private: bool = True) -> bool:
    """ckpt を push する。成功なら True、失敗は WARN を出して False。"""
    from huggingface_hub import HfApi  # lazy: test で fake に差し替える

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=ckpt_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"autopush ckpt at step {step}",
        )
    except Exception as e:
        print(f"[hf_autopush] WARN: push failed ({type(e).__name__}: {e})", flush=True)
        return False
    print(f"[hf_autopush] pushed {ckpt_path.name} -> {repo_id}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ckpt 1 個を HF Hub に push する")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--public", action="store_true", help="repo を新しく作るとき public にする (既定は private)")
    args = parser.parse_args(argv)
    return 0 if push_ckpt(args.ckpt, args.repo_id, args.step, private=not args.public) else 1


if __name__ == "__main__":
    sys.exit(main())
