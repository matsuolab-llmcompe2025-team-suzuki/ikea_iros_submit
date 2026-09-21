"""Export RAMEN-Ori episode split as JSON (Issue #122)。

RAMEN-Ori と GR00T で **同じ episode を train/val/test に割り当てる** ため、
episode-level split を JSON dump する。GR00T 側 (`materialize_lerobot_training_view.py`
--split-json) が同じ JSON を読み込み、build_grouped_episode_split を bypass する。

CLI:
    pixi run python -m model.ramen_ori.scripts.export_split_json \\
        --config-name=real_task5_7 \\
        --output outputs/split_task5_7.json

出力 JSON schema (materialize 側 load_external_split_json + resolve_training_split と互換):
    {
      "schema_version": "team_ramen_grouped_episode_split_v1",
      "seed": 42,
      "group_key": "source_episode_name",
      "num_tasks": <int>,
      "num_episodes": <int>,
      "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
      "splits": {
        "train": {"episode_indices": [...], "source_episode_names": [...]},
        "validation": {...},
        "test": {...},
      },
      "sha256": "<hex>",
      "provenance": {
        "config_name": "real_task5_7",
        "repo_ids": [...],
        "episode_lengths": [...],
      }
    }

group_key: "source_episode_name" (Issue #122 fix): task 5+7 combined dataset は
複数 curated chunk が同一 raw 収録から抽出されるため、episode_index 単位 shuffle だと
train/val leakage の可能性あり。source_episode_index を stable grouping key として使い、
同一 raw 収録 (== 同一 source_episode_name) は同 split に集約する。resolve_training_split.py
は各 split に source_episode_names list を要求する (line 50-56)、その要件を満たす。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from model.ramen_ori.data_lerobot import (
    RamenOriLerobotDataset,
    split_episode_indices,
)


def build_split_payload(
    dataset: RamenOriLerobotDataset,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    config_name: str,
    repo_ids: list[str] | None,
) -> dict:
    """Dataset の episode metadata から split payload を構築 (JSON serializable)。"""
    meta = dataset.sample_episode_metadata()
    episode_lengths = meta["episode_lengths"].tolist()
    num_episodes = len(episode_lengths)
    source_names = meta["source_episode_names"]
    train_eps, val_eps, test_eps = split_episode_indices(
        num_episodes,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        source_episode_names=source_names,
    )

    def _per_split_source_names(eps: list[int]) -> list[str]:
        return sorted({source_names[i] for i in eps})

    # task (skill) 数を meta から拾う (multi-task 検出用)
    unique_tasks = sorted(set(int(t) for t in meta["episode_skill_ids"].tolist()))
    payload = {
        "schema_version": "team_ramen_grouped_episode_split_v1",
        "seed": seed,
        "group_key": "source_episode_name",
        "num_tasks": len(unique_tasks),
        "num_episodes": num_episodes,
        "fractions": {
            "train": 1.0 - val_ratio - test_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "splits": {
            "train": {
                "episode_indices": train_eps,
                "source_episode_names": _per_split_source_names(train_eps),
            },
            "validation": {
                "episode_indices": val_eps,
                "source_episode_names": _per_split_source_names(val_eps),
            },
            "test": {
                "episode_indices": test_eps,
                "source_episode_names": _per_split_source_names(test_eps),
            },
        },
        "provenance": {
            "config_name": config_name,
            "repo_ids": repo_ids or [],
            "episode_lengths": episode_lengths,
            "unique_task_indices": unique_tasks,
            "num_source_recordings": len(set(source_names)),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(serialized).hexdigest()
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", required=True, help="Hydra config name under configs/ (e.g. real_task5_7)")
    p.add_argument("--output", type=Path, required=True, help="output JSON path")
    p.add_argument("--val-ratio", type=float, default=None, help="override cfg.val.val_ratio")
    p.add_argument("--test-ratio", type=float, default=None, help="override cfg.val.test_ratio")
    p.add_argument("--seed", type=int, default=None, help="override cfg.val.seed")
    p.add_argument("--force", action="store_true", help="overwrite existing --output")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to overwrite")

    # Hydra config compose (@hydra.main を使わずに手動 compose、CLI 引数と共存させるため)
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((Path(__file__).resolve().parents[1] / "configs").absolute()),
    ):
        cfg: DictConfig = hydra.compose(config_name=args.config_name)

    val_cfg = cfg.get("val", {})
    val_ratio = args.val_ratio if args.val_ratio is not None else float(val_cfg.get("val_ratio", 0.1))
    test_ratio = args.test_ratio if args.test_ratio is not None else float(val_cfg.get("test_ratio", 0.1))
    seed = args.seed if args.seed is not None else int(val_cfg.get("seed", 42))

    # Dataset instantiate (aug 不要 = None、少しでも memory 節約)。
    # Issue #122: split export は parquet 側 meta (episode_index / source_episode_name) しか
    # 使わないため frame cache precompute (5-10 min) は skip。以前は env `FRAME_CACHE_PRECOMPUTE=false`
    # を手動 export する必要があったが、targeted に constructor で override して user が env を
    # 意識せずに走らせられるようにする。学習経路の auto precompute default は保持。
    dataset: RamenOriLerobotDataset = hydra.utils.instantiate(
        cfg.data,
        augmentation_cfg=None,
        auto_precompute_frame_cache=False,
    )
    if not isinstance(dataset, RamenOriLerobotDataset):
        raise TypeError(
            f"config {args.config_name!r} が RamenOriLerobotDataset を指していない "
            f"(got {type(dataset).__name__})、dummy config では split export できない"
        )

    # Issue #122 refactor: repo_ids は廃止、merged_source_root を provenance に記録。
    # 旧 config 互換で repo_ids も有れば拾う (test 用 fake の場合等)。
    repo_ids: list[str] = list(cfg.data.get("repo_ids", []))
    merged_root = cfg.data.get("merged_source_root")
    if merged_root and not repo_ids:
        repo_ids = [str(merged_root)]
    payload = build_split_payload(
        dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        config_name=args.config_name,
        repo_ids=repo_ids,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[export] {payload['num_episodes']} eps "
        f"(source recordings: {payload['provenance']['num_source_recordings']}) into "
        f"train={len(payload['splits']['train']['episode_indices'])} / "
        f"val={len(payload['splits']['validation']['episode_indices'])} / "
        f"test={len(payload['splits']['test']['episode_indices'])} "
        f"(group_key={payload['group_key']}, sha256={payload['sha256'][:12]}...) → {args.output}"
    )


if __name__ == "__main__":
    main()
