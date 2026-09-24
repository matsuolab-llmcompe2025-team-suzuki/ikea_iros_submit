"""Multi-repo dataset download + merge helper (Issue #122)。

`resolve_training_config.py` の DATASET_REPO_IDS (JSON list) を受け、
各 repo を HF から snapshot_download し、`merge_curated_lerobot_datasets.py` で
1 つの merged source dataset directory に束ねる。train_lerobot.sh の pre-materialize
step で multi-repo subtask (例: combined_task5_7) 時に実行される。

CLI:
    python prepare_multi_repo_source.py \\
        --repo-ids '["Team-RAMEN/IROS2026_RAMEN_HARA_task_7_optimal_v1", ...]' \\
        --output-dir outputs/merged_sources/combined_task5_7 \\
        [--revision v1] [--force] \\
        [--max-repo-concurrency 4] [--max-workers 16]

高速化:
    Default で repo 4 並列 × 各 16 file worker = 最大 64 concurrent connection、
    500Mbps 帯域を saturate できる (実測 64GB を 11.5min で pull)。sequential (1x1)
    に戻すなら --max-repo-concurrency 1。詳細は
    docs/setup/sakura_setup.md §B-4「HF dataset DL 高速化」参照。

出力:
    stdout に merged directory の絶対 path (train_lerobot.sh がそのまま
    materialize 側の --source-root に渡す)。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


SCRIPT_DIR = Path(__file__).resolve().parent
MERGE_SCRIPT = (
    SCRIPT_DIR.parents[2]  # repo root (parents[1]=model は誤り、merge script は <root>/data/ 配下)
    / "data"
    / "bitrobot_lerobot_subtask_datasets"
    / "scripts"
    / "merge_curated_lerobot_datasets.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-ids", required=True, help="JSON list of repo_ids (from DATASET_REPO_IDS)")
    p.add_argument("--output-dir", type=Path, required=True, help="merged output directory")
    p.add_argument("--revision", default=None)
    p.add_argument("--force", action="store_true", help="overwrite existing --output-dir")
    p.add_argument(
        "--merge-script",
        type=Path,
        default=MERGE_SCRIPT,
        help=f"path to merge_curated_lerobot_datasets.py (default: {MERGE_SCRIPT})",
    )
    # Issue #122: 500Mbps 帯域 saturate 用 (詳細は module docstring)
    p.add_argument(
        "--max-repo-concurrency",
        type=int,
        default=4,
        help="repo 単位の並列 pull 数 (ThreadPool worker、default 4、1 で sequential)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="各 snapshot_download 内の file worker 数 (default 16、hf_hub default は 4)",
    )
    return p.parse_args()


def download_all(
    repo_ids: list[str],
    revision: str | None,
    max_repo_concurrency: int = 4,
    max_workers: int = 16,
) -> list[Path]:
    """各 repo を並列 snapshot_download し、repo_ids と同順の local path list を返す。

    - max_repo_concurrency: repo 単位の並列度 (default 4)。1 で sequential fallback。
    - max_workers: 各 snapshot_download 内の file worker 数 (default 16)。

    合計 max_repo_concurrency × max_workers connections。default 4×16=64 で 500Mbps
    帯域を saturate 可能 (実測 64GB / 11.5min)。順序は repo_ids と一致 (merge 側の
    global episode_index 再採番で順序 sensitive のため)。
    """

    def _pull(repo_id: str) -> Path:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["README.md", "meta/**", "data/**", "videos/**"],
                max_workers=max_workers,
            )
        )

    if max_repo_concurrency <= 1:
        # sequential fallback (debug / rate-limit 回避時)
        paths = []
        for i, repo_id in enumerate(repo_ids, 1):
            print(f"[{i}/{len(repo_ids)}] snapshot_download: {repo_id}", flush=True)
            paths.append(_pull(repo_id))
        return paths

    print(
        f"[parallel] {len(repo_ids)} repos, {max_repo_concurrency} concurrent x "
        f"{max_workers} workers = up to {max_repo_concurrency * max_workers} connections",
        flush=True,
    )
    paths: list[Path | None] = [None] * len(repo_ids)
    with ThreadPoolExecutor(max_workers=max_repo_concurrency) as ex:
        futures = {ex.submit(_pull, r): i for i, r in enumerate(repo_ids)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            paths[i] = fut.result()  # 例外は raise させ、上位で fail-fast
            done += 1
            print(f"[{done}/{len(repo_ids)}] done: {repo_ids[i]}", flush=True)
    return paths  # type: ignore[return-value]


def run_merge(source_dirs: list[Path], output_dir: Path, merge_script: Path, force: bool) -> None:
    """merge_curated_lerobot_datasets.py を subprocess で呼び出し (allow-duplicate-task-indices ON)。"""
    if not merge_script.exists():
        raise FileNotFoundError(f"merge script not found: {merge_script}")
    args: list[str] = [sys.executable, str(merge_script)]
    for d in source_dirs:
        args.extend(["--source-dir", str(d)])
    args.extend(["--output-dir", str(output_dir)])
    args.append("--allow-duplicate-task-indices")  # Issue #122 Phase A で追加した flag
    if force:
        args.append("--force")
    print(f"[merge] running: {' '.join(args[:3])} ...+{len(source_dirs)} sources", flush=True)
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"merge script failed with exit code {result.returncode}")


def main() -> None:
    args = parse_args()
    repo_ids = json.loads(args.repo_ids)
    if not isinstance(repo_ids, list) or not all(isinstance(x, str) for x in repo_ids):
        raise ValueError(f"--repo-ids must be a JSON list of strings, got {args.repo_ids!r}")
    if not repo_ids:
        raise ValueError("--repo-ids must not be empty")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.force:
        # marker check: source list が同じなら再利用
        marker = output_dir / "_merged_source_repos.json"
        if marker.exists():
            existing = json.loads(marker.read_text())
            if existing == repo_ids:
                print(f"[reuse] merged dir already up-to-date: {output_dir}", flush=True)
                print(str(output_dir))  # train_lerobot.sh が拾う
                return
        raise FileExistsError(f"{output_dir} exists; pass --force to replace")

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    source_dirs = download_all(
        repo_ids,
        args.revision,
        max_repo_concurrency=args.max_repo_concurrency,
        max_workers=args.max_workers,
    )
    run_merge(source_dirs, output_dir, args.merge_script, force=args.force)

    # marker file (再利用判定用)
    marker = output_dir / "_merged_source_repos.json"
    marker.write_text(json.dumps(repo_ids, indent=2))

    print(f"[done] merged {len(repo_ids)} repos into {output_dir}", flush=True)
    print(str(output_dir))  # train_lerobot.sh の shell subst で拾う (最終行)


if __name__ == "__main__":
    main()
