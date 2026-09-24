"""Build ACT, Diffusion, or GR00T views over the shared LeRobot v3 data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download


FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = FEATURE_ROOT / "configs" / "subtask_training.json"
MARKER_PATH = Path("meta/team_ramen_training_view.json")
SPLIT_PATH = Path("meta/team_ramen_episode_split.json")
MAPPING_PATH = FEATURE_ROOT / "gr00t" / "g1_full_body_mapping.py"
PRESERVED_DATA_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
PRESERVED_EPISODE_KEYS = (
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "source_episode_index",
    "source_episode_name",
    "source_task",
    "source_start_sec",
    "source_end_sec",
    "source_frame_start_sec",
    "source_frame_end_sec",
)
VIDEO_EPISODE_SUFFIXES = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
STATS_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def load_mapping_module() -> Any:
    spec = importlib.util.spec_from_file_location("g1_full_body_mapping", MAPPING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load mapping module: {MAPPING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mapping = load_mapping_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-multi-task",
        action="store_true",
        help=(
            "Issue #122: source dataset に複数 task_index が混在する combined dataset を "
            "受け付ける (例: task 5+7 combined)。source_episode_name grouping は leakage-safe。"
        ),
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help=(
            "Issue #122: RAMEN-Ori の export_split_json.py が出力した split.json を読み込み、"
            "materialize 内の build_grouped_episode_split を bypass して同じ episode 分割を"
            "使う (apples-to-apples 比較用、GR00T ↔ RAMEN-Ori で split 一致)。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_root = args.source_root or download_source_dataset(
        args.repo_id,
        config,
        revision=args.revision,
    )
    materialize_training_view(
        source_root=source_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        config=config,
        policy_type=args.policy_type,
        force=args.force,
        allow_multi_task=args.allow_multi_task,
        split_json=args.split_json,
    )
    print(f"Wrote {args.policy_type} training view to {args.output_root}")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "subtask_training_v1":
        raise ValueError(f"unsupported training config schema: {config.get('schema_version')}")
    return config


def download_source_dataset(repo_id: str, config: dict[str, Any], *, revision: str | None = None) -> Path:
    allow_patterns = ["README.md", "meta/**", "data/**"]
    for source_key in training_video_map(config).values():
        allow_patterns.append(f"videos/{source_key}/**")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
        )
    )


def materialize_training_view(
    *,
    source_root: Path,
    output_root: Path,
    repo_id: str,
    config: dict[str, Any],
    policy_type: str,
    force: bool = False,
    allow_multi_task: bool = False,        # Issue #122: combined multi-task 対応
    split_json: Path | None = None,        # Issue #122: 外部 split JSON (apples-to-apples 用)
) -> None:
    if policy_type not in {"act", "diffusion", "flow_matching", "groot"}:
        raise ValueError(f"unsupported policy type: {policy_type!r}")
    pa, pq = require_pyarrow()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if (
        source_root == output_root
        or output_root.is_relative_to(source_root)
        or source_root.is_relative_to(output_root)
    ):
        raise ValueError("source-root and output-root must be separate, non-nested directories")
    validate_source_contract(source_root, config)
    source_fingerprint = source_dataset_fingerprint(source_root)
    expected_marker = build_marker(
        repo_id=repo_id,
        source_root=source_root,
        config=config,
        policy_type=policy_type,
        source_fingerprint=source_fingerprint,
    )
    if output_root.exists():
        marker_path = output_root / MARKER_PATH
        if not force and marker_path.exists() and _markers_equivalent(read_json(marker_path), expected_marker):
            print(f"Reusing existing training view at {output_root}")
            return
        if not force:
            raise FileExistsError(f"{output_root} exists; pass --force to replace it")

    temporary_root = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    try:
        info = read_json(source_root / "meta" / "info.json")
        source_features = copy.deepcopy(info["features"])
        source_stats = read_json(source_root / "meta" / "stats.json")
        video_map = training_video_map(config)
        view_layout = view_layout_for_policy(config, policy_type)
        if split_json is not None:
            # Issue #122: 外部 JSON (RAMEN-Ori export) を採用、grouped_split を bypass
            episode_split = load_external_split_json(
                split_json, total_episodes=int(info["total_episodes"])
            )
            print(f"[split] loaded external split from {split_json} (sha256={episode_split.get('sha256', '?')[:12]}...)")
        else:
            episode_split = build_grouped_episode_split(
                source_root, config=config, pq=pq, allow_multi_task=allow_multi_task
            )
        training_episode_indices = set(episode_split["splits"]["train"]["episode_indices"])
        temporary_root.mkdir(parents=True)
        (temporary_root / "data").mkdir()
        (temporary_root / "meta").mkdir()
        (temporary_root / "videos").mkdir()

        copy_optional_file(source_root / "README.md", temporary_root / "README.md")
        copy_required_file(
            source_root / "meta" / "tasks.parquet",
            temporary_root / "meta" / "tasks.parquet",
        )
        mapped_vector_stats = rewrite_data_parquets(
            source_root,
            temporary_root,
            config=config,
            policy_type=policy_type,
            stats_episode_indices=training_episode_indices,
            pa=pa,
            pq=pq,
        )
        rewrite_episode_metadata(
            source_root,
            temporary_root,
            config=config,
            policy_type=policy_type,
            pa=pa,
            pq=pq,
        )
        link_video_dirs(source_root, temporary_root, video_map)

        info["features"] = build_training_features(
            config=config,
            policy_type=policy_type,
            source_features=source_features,
        )
        write_json(temporary_root / "meta" / "info.json", info)
        if view_layout == "real_g1_relative_eef_relative_joints":
            write_json(
                temporary_root / "meta" / "modality.json",
                _mapping.build_real_g1_relative_eef_modality_json(list(video_map)),
            )
        write_json(
            temporary_root / "meta" / "stats.json",
            build_training_stats(
                config=config,
                policy_type=policy_type,
                source_stats=source_stats,
                source_features=source_features,
                mapped_vector_stats=mapped_vector_stats,
            ),
        )
        write_json(temporary_root / SPLIT_PATH, episode_split)
        write_json(temporary_root / MARKER_PATH, expected_marker)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    backup_root = output_root.with_name(f".{output_root.name}.backup-{os.getpid()}")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if output_root.exists():
        output_root.replace(backup_root)
    try:
        temporary_root.replace(output_root)
    except BaseException:
        if backup_root.exists() and not output_root.exists():
            backup_root.replace(output_root)
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    else:
        # Issue #129: precompute frame cache (frame_cache*/ subdir) は materialize と独立 lifecycle
        # で数十〜数百 GB。--force で backup_root を rmtree する前に、旧 output_root 側の cache
        # subdir を new output_root へ引き継ぐ (再 precompute しなくて済むように)。
        if backup_root.exists():
            for cache_dir_name in _PRESERVE_SUBDIRS_ON_FORCE:
                backup_cache = backup_root / cache_dir_name
                if backup_cache.is_dir():
                    dest_cache = output_root / cache_dir_name
                    if dest_cache.exists():
                        continue  # 新 output_root 側に既存優先 (通常無いが safety)
                    print(f"[materialize] preserving cache subdir across --force: {cache_dir_name}")
                    backup_cache.rename(dest_cache)
            shutil.rmtree(backup_root)


def rewrite_data_parquets(
    source_root: Path,
    output_root: Path,
    *,
    config: dict[str, Any],
    policy_type: str,
    stats_episode_indices: set[int] | None = None,
    pa: Any,
    pq: Any,
) -> dict[str, dict[str, Any]] | None:
    files = sorted((source_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot v3 data parquet files found under {source_root / 'data'}")

    accumulators = make_vector_stats_accumulators()
    seen_stats_episodes: set[int] = set()
    for source_path in files:
        rel_path = source_path.relative_to(source_root)
        output_path = output_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source_path)
        state_rows, action_rows = build_policy_vectors_from_table(
            table,
            config=config,
            policy_type=policy_type,
        )
        state_dim, action_dim = policy_dims(config, policy_type)
        episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        if stats_episode_indices is None:
            stats_mask = np.ones(episode_indices.shape, dtype=bool)
        else:
            stats_mask = np.isin(episode_indices, list(stats_episode_indices))
        if np.any(stats_mask):
            accumulators[config["state"]["key"]].update(
                np.asarray(state_rows, dtype=np.float32)[stats_mask]
            )
            accumulators[config["action"]["key"]].update(
                np.asarray(action_rows, dtype=np.float32)[stats_mask]
            )
            seen_stats_episodes.update(int(value) for value in np.unique(episode_indices[stats_mask]))
        arrays = [
            fixed_size_float_array(pa, state_rows, state_dim),
            fixed_size_float_array(pa, action_rows, action_dim),
        ]
        names = [config["state"]["key"], config["action"]["key"]]
        for key in PRESERVED_DATA_KEYS:
            if key not in table.schema.names:
                raise KeyError(f"{source_path} is missing required key {key!r}")
            arrays.append(table[key])
            names.append(key)
        pq.write_table(pa.Table.from_arrays(arrays, names=names), output_path, compression="snappy")
    if stats_episode_indices is not None and seen_stats_episodes != stats_episode_indices:
        missing = sorted(stats_episode_indices - seen_stats_episodes)
        extra = sorted(seen_stats_episodes - stats_episode_indices)
        raise ValueError(
            f"training statistics episode coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    return {
        key: numpy_stats_to_json(accumulator.get_statistics())
        for key, accumulator in accumulators.items()
    }


def build_policy_vectors_from_table(
    table: Any,
    *,
    config: dict[str, Any],
    policy_type: str,
) -> tuple[list[list[float]], list[list[float]]]:
    source = config["source_dataset"]
    state_keys = source["state_keys"]
    action_keys = source["action_keys"]
    view_layout = view_layout_for_policy(config, policy_type)
    required_keys = [
        state_keys["robot_q_current"],
        state_keys["hand_state"],
        action_keys["robot_q_desired"],
        action_keys["hand_cmd"],
    ]
    if view_layout == "real_g1_relative_eef_relative_joints":
        validate_source_eef_pose_format(source)
        required_keys.extend([state_keys["ee_state"], action_keys["ee_action"]])
    for key in required_keys:
        if key not in table.schema.names:
            raise KeyError(f"source table is missing required key {key!r}")

    robot_q_current = table[state_keys["robot_q_current"]].to_pylist()
    hand_state = table[state_keys["hand_state"]].to_pylist()
    robot_q_desired = table[action_keys["robot_q_desired"]].to_pylist()
    hand_cmd = table[action_keys["hand_cmd"]].to_pylist()
    if view_layout == "real_g1_relative_eef_relative_joints":
        ee_state = table[state_keys["ee_state"]].to_pylist()
        ee_action = table[action_keys["ee_action"]].to_pylist()
        state_rows: list[list[float]] = []
        action_rows: list[list[float]] = []
        for current_eef, target_eef, current, desired, hand, cmd in zip(
            ee_state,
            ee_action,
            robot_q_current,
            robot_q_desired,
            hand_state,
            hand_cmd,
            strict=True,
        ):
            state, action = _mapping.map_source_row_to_real_g1_relative_eef(
                ee_state=as_float_list(current_eef),
                ee_action=as_float_list(target_eef),
                robot_q_current=as_float_list(current),
                robot_q_desired=as_float_list(desired),
                hand_state=as_float_list(hand),
                hand_cmd=as_float_list(cmd),
            )
            state_rows.append(state)
            action_rows.append(action)
        return state_rows, action_rows

    state_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    for current, hand, desired, cmd in zip(
        robot_q_current,
        hand_state,
        robot_q_desired,
        hand_cmd,
        strict=True,
    ):
        row = {
            config["state"]["key"]: as_float_list(current) + as_float_list(hand),
            config["action"]["key"]: as_float_list(desired) + as_float_list(cmd),
        }
        if view_layout == "robot_q_upper_body_19d":
            row = _mapping.map_robot_q_row_to_upper_body(row)
        state_rows.append(row[config["state"]["key"]])
        action_rows.append(row[config["action"]["key"]])
    return state_rows, action_rows


def rewrite_episode_metadata(
    source_root: Path,
    output_root: Path,
    *,
    config: dict[str, Any],
    policy_type: str,
    pa: Any,
    pq: Any,
) -> None:
    files = sorted((source_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        episode_root = source_root / "meta" / "episodes"
        raise FileNotFoundError(f"no LeRobot v3 episode metadata files found under {episode_root}")
    video_map = training_video_map(config)
    for source_path in files:
        table = pq.read_table(source_path)
        arrays: list[Any] = []
        names: list[str] = []
        for key in PRESERVED_EPISODE_KEYS:
            if key in table.schema.names:
                arrays.append(table[key])
                names.append(key)
        for target_key, source_key in video_map.items():
            for suffix in VIDEO_EPISODE_SUFFIXES:
                source_col = f"videos/{source_key}/{suffix}"
                target_col = f"videos/{target_key}/{suffix}"
                if source_col not in table.schema.names:
                    raise KeyError(f"{source_path} is missing required video metadata {source_col!r}")
                arrays.append(table[source_col])
                names.append(target_col)
        add_episode_stats_columns(table, arrays, names, config=config, policy_type=policy_type)
        for key in ("meta/episodes/chunk_index", "meta/episodes/file_index"):
            if key in table.schema.names:
                arrays.append(table[key])
                names.append(key)

        rel_path = source_path.relative_to(source_root)
        output_path = output_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_arrays(arrays, names=names), output_path, compression="snappy")


def add_episode_stats_columns(
    table: Any,
    arrays: list[Any],
    names: list[str],
    *,
    config: dict[str, Any],
    policy_type: str,
) -> None:
    source = config["source_dataset"]
    if view_layout_for_policy(config, policy_type) != "real_g1_relative_eef_relative_joints":
        state_prefixes = [
            f"stats/{source['state_keys']['robot_q_current']}",
            f"stats/{source['state_keys']['hand_state']}",
        ]
        action_prefixes = [
            f"stats/{source['action_keys']['robot_q_desired']}",
            f"stats/{source['action_keys']['hand_cmd']}",
        ]
        state_index_map, action_index_map = stats_index_maps(config, policy_type)
        for stat_name in STATS_NAMES:
            add_combined_episode_stat(
                table,
                arrays,
                names,
                target_prefix=f"stats/{config['state']['key']}",
                stat_name=stat_name,
                source_prefixes=state_prefixes,
                index_map=state_index_map,
            )
            add_combined_episode_stat(
                table,
                arrays,
                names,
                target_prefix=f"stats/{config['action']['key']}",
                stat_name=stat_name,
                source_prefixes=action_prefixes,
                index_map=action_index_map,
            )

    for target_key, source_key in training_video_map(config).items():
        for stat_name in STATS_NAMES:
            copy_episode_column(
                table,
                arrays,
                names,
                source_col=f"stats/{source_key}/{stat_name}",
                target_col=f"stats/{target_key}/{stat_name}",
                required=False,
            )
    for key in PRESERVED_DATA_KEYS:
        for stat_name in STATS_NAMES:
            copy_episode_column(
                table,
                arrays,
                names,
                source_col=f"stats/{key}/{stat_name}",
                target_col=f"stats/{key}/{stat_name}",
                required=False,
            )


def add_combined_episode_stat(
    table: Any,
    arrays: list[Any],
    names: list[str],
    *,
    target_prefix: str,
    stat_name: str,
    source_prefixes: list[str],
    index_map: list[int | None],
) -> None:
    columns = [f"{prefix}/{stat_name}" for prefix in source_prefixes]
    if any(column not in table.schema.names for column in columns):
        return
    if stat_name == "count":
        arrays.append(table[columns[0]])
        names.append(f"{target_prefix}/{stat_name}")
        return
    combined_rows = [
        map_stat_values(concat_lists(values), index_map)
        for values in zip(*(table[column].to_pylist() for column in columns), strict=True)
    ]
    arrays.append(combined_rows)
    names.append(f"{target_prefix}/{stat_name}")


def copy_episode_column(
    table: Any,
    arrays: list[Any],
    names: list[str],
    *,
    source_col: str,
    target_col: str,
    required: bool,
) -> None:
    if source_col not in table.schema.names:
        if required:
            raise KeyError(f"episode metadata is missing required column {source_col!r}")
        return
    arrays.append(table[source_col])
    names.append(target_col)


def link_video_dirs(source_root: Path, output_root: Path, video_map: dict[str, str]) -> None:
    for target_key, source_key in video_map.items():
        source_dir = source_root / "videos" / source_key
        if not source_dir.exists():
            raise FileNotFoundError(f"source video directory missing: {source_dir}")
        target_dir = output_root / "videos" / target_key
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        relative_source = os.path.relpath(source_dir, target_dir.parent)
        target_dir.symlink_to(relative_source, target_is_directory=True)


def build_training_features(
    *,
    config: dict[str, Any],
    policy_type: str,
    source_features: dict[str, Any],
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for target_key, source_key in training_video_map(config).items():
        if source_key not in source_features:
            raise KeyError(f"source features are missing camera key {source_key!r}")
        features[target_key] = normalize_visual_feature(source_features[source_key])

    state_dim, action_dim = policy_dims(config, policy_type)
    state_names, action_names = policy_names(config, policy_type)
    features[config["state"]["key"]] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": state_names,
    }
    features[config["action"]["key"]] = {
        "dtype": "float32",
        "shape": [action_dim],
        "names": action_names,
    }
    for key in PRESERVED_DATA_KEYS:
        if key in source_features:
            features[key] = copy.deepcopy(source_features[key])
    return features


def normalize_visual_feature(feature: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(feature)
    if normalized.get("dtype") not in {"image", "video"}:
        return normalized
    shape = normalized.get("shape", [])
    if not isinstance(shape, list) or len(shape) != 3:
        return normalized

    if shape[-1] in (1, 3, 4):
        normalized["names"] = ["height", "width", "channels"]
    elif shape[0] in (1, 3, 4):
        normalized["names"] = ["channels", "height", "width"]
    else:
        normalized["names"] = ["height", "width", "channels"]
    return normalized


def build_training_stats(
    *,
    config: dict[str, Any],
    policy_type: str,
    source_stats: dict[str, Any],
    source_features: dict[str, Any],
    mapped_vector_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = config["source_dataset"]
    if mapped_vector_stats is not None:
        stats = copy.deepcopy(mapped_vector_stats)
    else:
        state_stats = concat_stat_entries(
            [
                source_stats[source["state_keys"]["robot_q_current"]],
                source_stats[source["state_keys"]["hand_state"]],
            ]
        )
        action_stats = concat_stat_entries(
            [
                source_stats[source["action_keys"]["robot_q_desired"]],
                source_stats[source["action_keys"]["hand_cmd"]],
            ]
        )
        state_index_map, action_index_map = stats_index_maps(config, policy_type)
        stats = {
            config["state"]["key"]: map_stat_entry(state_stats, state_index_map),
            config["action"]["key"]: map_stat_entry(action_stats, action_index_map),
        }
    for target_key, source_key in training_video_map(config).items():
        if source_key in source_stats:
            stats[target_key] = copy.deepcopy(source_stats[source_key])
    for key in PRESERVED_DATA_KEYS:
        if key in source_stats and key in source_features:
            stats[key] = copy.deepcopy(source_stats[key])
    return stats


def concat_stat_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for stat_name in STATS_NAMES:
        if any(stat_name not in entry for entry in entries):
            continue
        if stat_name == "count":
            combined[stat_name] = copy.deepcopy(entries[0][stat_name])
            continue
        combined[stat_name] = concat_lists(entry[stat_name] for entry in entries)
    return combined


def make_vector_stats_accumulators() -> dict[str, Any]:
    from lerobot.datasets.compute_stats import RunningQuantileStats

    return {
        "observation.state": RunningQuantileStats(),
        "action": RunningQuantileStats(),
    }


def numpy_stats_to_json(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in stats.items()
    }


def map_stat_entry(entry: dict[str, Any], index_map: list[int | None]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for stat_name, values in entry.items():
        if stat_name == "count":
            mapped[stat_name] = copy.deepcopy(values)
            continue
        mapped[stat_name] = map_stat_values(values, index_map)
    return mapped


def stats_index_maps(config: dict[str, Any], policy_type: str) -> tuple[list[int | None], list[int | None]]:
    view_layout = view_layout_for_policy(config, policy_type)
    if view_layout == "real_g1_relative_eef_relative_joints":
        raise ValueError("relative-EEF ROT6D statistics cannot be produced with an index-only mapping")
    if view_layout == "robot_q_current_robot_q_desired_38d":
        return list(range(_mapping.SOURCE_STATE_DIM)), list(range(_mapping.SOURCE_ACTION_DIM))
    if view_layout == "robot_q_upper_body_19d":
        return list(_mapping.UPPER_BODY_SOURCE_INDEX_MAP), list(_mapping.UPPER_BODY_SOURCE_INDEX_MAP)
    raise ValueError(f"unsupported policy view layout {view_layout!r}")


def map_stat_values(values: list[Any], index_map: list[int | None]) -> list[Any]:
    return [0.0 if index is None else copy.deepcopy(values[index]) for index in index_map]


def training_video_map(config: dict[str, Any]) -> dict[str, str]:
    camera_map = config["source_dataset"]["camera_map"]
    result = {
        f"observation.images.{camera_name}": camera_map[camera_name]
        for camera_name in config["cameras"]
        if camera_name != "head_right"
    }
    if tuple(result) != (
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ):
        raise ValueError(
            "training cameras must be exactly head_left, left_wrist, right_wrist in that order"
        )
    if len(set(result.values())) != len(result):
        raise ValueError(f"source camera mapping contains aliases: {result}")
    return result


def ignored_source_video_keys(source_features: dict[str, Any], video_map: dict[str, str]) -> list[str]:
    selected = set(video_map.values())
    return sorted(
        key
        for key, feature in source_features.items()
        if isinstance(feature, dict)
        and feature.get("dtype") == "video"
        and key.startswith("observation.images.")
        and key not in selected
    )


def policy_dims(config: dict[str, Any], policy_type: str) -> tuple[int, int]:
    view_layout = view_layout_for_policy(config, policy_type)
    if view_layout == "real_g1_relative_eef_relative_joints":
        return _mapping.REAL_G1_RELATIVE_EEF_STATE_DIM, _mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM
    if view_layout == "robot_q_upper_body_19d":
        return _mapping.UPPER_BODY_STATE_DIM, _mapping.UPPER_BODY_ACTION_DIM
    return _mapping.SOURCE_STATE_DIM, _mapping.SOURCE_ACTION_DIM


def policy_names(config: dict[str, Any], policy_type: str) -> tuple[list[str], list[str]]:
    view_layout = view_layout_for_policy(config, policy_type)
    if view_layout == "real_g1_relative_eef_relative_joints":
        return (
            list(_mapping.REAL_G1_RELATIVE_EEF_STATE_NAMES),
            list(_mapping.REAL_G1_RELATIVE_EEF_ACTION_NAMES),
        )
    if view_layout == "robot_q_upper_body_19d":
        return list(_mapping.UPPER_BODY_STATE_NAMES), list(_mapping.UPPER_BODY_ACTION_NAMES)
    return full_robot_q_state_names(), full_robot_q_action_names()


def control_scope_for_policy(config: dict[str, Any], policy_type: str) -> str:
    if policy_type == "groot":
        return "upper_body_relative_eef"
    value = os.environ.get("CONTROL_SCOPE", config.get("training", {}).get("control_scope", "upper_body"))
    value = str(value).strip().lower()
    if value not in {"upper_body", "full_robot_q"}:
        raise ValueError("CONTROL_SCOPE must be 'upper_body' or 'full_robot_q'")
    return value


def view_layout_for_policy(config: dict[str, Any], policy_type: str) -> str:
    if policy_type == "groot":
        return "real_g1_relative_eef_relative_joints"
    if control_scope_for_policy(config, policy_type) == "upper_body":
        return "robot_q_upper_body_19d"
    return "robot_q_current_robot_q_desired_38d"


def full_robot_q_state_names() -> list[str]:
    return [f"robot_q_current_{index}" for index in range(_mapping.SOURCE_ROBOT_Q_DIM)] + [
        "left_gripper_q",
        "right_gripper_q",
    ]


def full_robot_q_action_names() -> list[str]:
    return [f"robot_q_desired_{index}" for index in range(_mapping.SOURCE_ROBOT_Q_DIM)] + [
        "left_gripper_q_cmd",
        "right_gripper_q_cmd",
    ]


def fixed_size_float_array(pa: Any, rows: list[list[float]], size: int) -> Any:
    flat: list[float] = []
    for row in rows:
        if len(row) != size:
            raise ValueError(f"expected {size}-D vector, got {len(row)}")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("policy vector contains NaN or Inf")
        flat.extend(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), size)


def as_float_list(values: list[Any]) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"numeric feature row must be a list, got {type(values).__name__}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("source numeric feature contains NaN or Inf")
    return result


def concat_lists(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        result.extend(copy.deepcopy(value))
    return result


# marker equivalence: `source_root` は HF snapshot path (dataset hash 依存) を保持するため
# 同じ dataset 内容でも HF 側で新 revision push があると snapshot hash が変わり値差分が出る。
# fingerprint (dataset 実 content の sha256) が一致していれば marker は同等とみなす。
_MARKER_INFORMATIONAL_FIELDS = ("source_root",)

# --force による rmtree(backup_root) で消えると困る cache subdir 群 (precompute で数百 GB 生成される)。
# materialize は data/meta/videos を書き換えるが、cache subdir は独立 lifecycle なので保持したい。
_PRESERVE_SUBDIRS_ON_FORCE = ("frame_cache", "frame_cache_v2")


def _markers_equivalent(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    stored_core = {k: v for k, v in stored.items() if k not in _MARKER_INFORMATIONAL_FIELDS}
    expected_core = {k: v for k, v in expected.items() if k not in _MARKER_INFORMATIONAL_FIELDS}
    return stored_core == expected_core


def build_marker(
    *,
    repo_id: str,
    source_root: Path,
    config: dict[str, Any],
    policy_type: str,
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    video_map = training_video_map(config)
    return {
        "schema_version": "team_ramen_training_view_v4",
        "source_repo_id": repo_id,
        "source_root": source_root.as_posix(),
        "source_fingerprint_sha256": source_fingerprint or source_dataset_fingerprint(source_root),
        "policy_type": policy_type,
        "view_layout": view_layout_for_policy(config, policy_type),
        "camera_map": video_map,
        "ignored_source_video_keys": ignored_source_video_keys(
            read_json(source_root / "meta" / "info.json")["features"],
            video_map,
        ),
        "state_sources": config["source_dataset"]["state_keys"],
        "action_sources": config["source_dataset"]["action_keys"],
        "source_eef_pose_format": config["source_dataset"].get("eef_pose_format"),
        "action_semantics": action_semantics_for_policy(config, policy_type),
        "episode_split": config.get("validation", {}),
        "state_action_statistics": "training_episodes_only",
        "groot_embodiment_tag": (
            _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG if policy_type == "groot" else None
        ),
    }


def action_semantics_for_policy(config: dict[str, Any], policy_type: str) -> str:
    if policy_type == "groot":
        return (
            "absolute EEF/joint targets in REAL_G1 slots; the N1.7 processor converts "
            "EEF with inv(T_current)@T_target and arm joints with target-current per action chunk"
        )
    if view_layout_for_policy(config, policy_type) == "robot_q_upper_body_19d":
        return "upper-body absolute joint target"
    return "full robot_q absolute joint target"


def validate_source_contract(source_root: Path, config: dict[str, Any]) -> None:
    required_files = (
        source_root / "meta" / "info.json",
        source_root / "meta" / "stats.json",
        source_root / "meta" / "tasks.parquet",
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"source dataset metadata is incomplete: {missing_files}")

    info = read_json(source_root / "meta" / "info.json")
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"source dataset must be LeRobotDataset v3, got {info.get('codebase_version')!r}")
    expected_fps = float(config["fps"])
    if not math.isclose(float(info.get("fps", -1)), expected_fps, abs_tol=1.0e-9):
        raise ValueError(f"source dataset fps must be {expected_fps:g}, got {info.get('fps')!r}")
    if int(info.get("total_episodes", 0)) <= 0 or int(info.get("total_frames", 0)) <= 0:
        raise ValueError("source dataset must contain at least one episode and frame")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("source dataset info.json has no feature mapping")
    source = config["source_dataset"]
    required_numeric = {
        source["state_keys"]["ee_state"]: 12,
        source["state_keys"]["robot_q_current"]: _mapping.SOURCE_ROBOT_Q_DIM,
        source["state_keys"]["hand_state"]: 2,
        source["action_keys"]["ee_action"]: 12,
        source["action_keys"]["robot_q_desired"]: _mapping.SOURCE_ROBOT_Q_DIM,
        source["action_keys"]["hand_cmd"]: 2,
    }
    for key, dimension in required_numeric.items():
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "float32" or feature.get("shape") != [dimension]:
            raise ValueError(f"source feature {key!r} must be float32[{dimension}], got {feature!r}")

    expected_shape = [
        int(config["image"]["height"]),
        int(config["image"]["width"]),
        int(config["image"]["channels"]),
    ]
    for source_key in training_video_map(config).values():
        feature = features.get(source_key)
        video_info = feature.get("info", {}) if isinstance(feature, dict) else {}
        if (
            not isinstance(feature, dict)
            or feature.get("dtype") != "video"
            or feature.get("shape") != expected_shape
            or float(video_info.get("video.fps", -1)) != expected_fps
        ):
            raise ValueError(
                f"source camera {source_key!r} must be {expected_shape} video at {expected_fps:g} fps, "
                f"got {feature!r}"
            )
        if not (source_root / "videos" / source_key).is_dir():
            raise FileNotFoundError(f"source camera directory is missing: {source_root / 'videos' / source_key}")


def source_dataset_fingerprint(source_root: Path) -> str:
    files: list[Path] = []
    for relative in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet"):
        path = source_root / relative
        if path.is_file():
            files.append(path)
    files.extend(sorted((source_root / "meta" / "episodes").glob("chunk-*/*.parquet")))
    files.extend(sorted((source_root / "data").glob("chunk-*/*.parquet")))
    if not files:
        raise FileNotFoundError(f"source dataset has no fingerprintable files: {source_root}")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def build_grouped_episode_split(
    source_root: Path,
    *,
    config: dict[str, Any],
    pq: Any,
    allow_multi_task: bool = False,
) -> dict[str, Any]:
    """Grouped episode split (leakage-safe: source_episode_name 単位で train/val/test)。

    Issue #122: `allow_multi_task=True` で multi-task combined dataset (e.g. task 5+7 combined)
    を許容。source_episode_name grouping で leakage safety は維持されるため、multi-task でも
    split は安全 (各 source_episode_name は単一 task の chunks から来る前提)。
    """
    split_config = config.get("validation", {})
    validation_fraction = float(split_config.get("validation_fraction", 0.1))
    test_fraction = float(split_config.get("test_fraction", 0.1))
    seed = int(split_config.get("seed", 42))
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be non-negative and sum to less than one")

    groups: dict[str, list[int]] = {}
    tasks: set[tuple[str, ...]] = set()
    all_episode_indices: list[int] = []
    episode_files = sorted((source_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"source dataset has no episode metadata: {source_root}")
    # Issue #129: 単一 repo (rotate_table_base_merged_v1 等) は source_episode_name column
    # を持たず source_episode_index (int) だけ持つ場合がある。schema-tolerant に fallback。
    # model/ramen_ori/data_lerobot.py の _extract_source_episode_name と同 pattern。
    sample_cols = pq.read_schema(episode_files[0]).names
    group_col = (
        "source_episode_name" if "source_episode_name" in sample_cols
        else "source_episode_index"
    )
    read_cols = ["episode_index", "tasks", group_col]
    for path in episode_files:
        table = pq.read_table(path, columns=read_cols)
        for row in table.to_pylist():
            source_key = str(row.get(group_col, "")).strip()
            if not source_key:
                raise ValueError(f"{group_col} is required for leakage-safe splitting")
            groups.setdefault(source_key, []).append(int(row["episode_index"]))
            all_episode_indices.append(int(row["episode_index"]))
            task_value = row.get("tasks")
            tasks.add(tuple(str(value) for value in (task_value if isinstance(task_value, list) else [task_value])))
    if len(tasks) != 1 and not allow_multi_task:
        raise ValueError(
            f"one subtask dataset must contain exactly one task label, got {sorted(tasks)} "
            "(combined multi-task dataset の場合は allow_multi_task=True を渡す、Issue #122)"
        )
    total_episodes = int(read_json(source_root / "meta" / "info.json")["total_episodes"])
    if sorted(all_episode_indices) != list(range(total_episodes)):
        raise ValueError(
            "episode metadata indices must be unique and contiguous from 0 to total_episodes-1"
        )

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    validation_count = round(len(group_names) * validation_fraction)
    test_count = round(len(group_names) * test_fraction)
    if validation_fraction > 0 and validation_count == 0:
        validation_count = 1
    if test_fraction > 0 and test_count == 0:
        test_count = 1
    if validation_count + test_count >= len(group_names):
        raise ValueError("episode split leaves no training source recordings")

    split_groups = {
        "test": group_names[:test_count],
        "validation": group_names[test_count : test_count + validation_count],
        "train": group_names[test_count + validation_count :],
    }
    payload = {
        "schema_version": "team_ramen_grouped_episode_split_v1",
        "seed": seed,
        "group_key": group_col,  # source_episode_name or source_episode_index (Issue #129 fallback)
        "num_tasks": len(tasks),  # Issue #122: multi-task combined の追跡用
        "fractions": {
            "train": 1.0 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "splits": {
            name: {
                "source_episode_names": names,
                "episode_indices": sorted(
                    episode_index for source_name in names for episode_index in groups[source_name]
                ),
            }
            for name, names in split_groups.items()
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(serialized).hexdigest()
    return payload


def load_external_split_json(path: Path, *, total_episodes: int) -> dict[str, Any]:
    """外部 split JSON (RAMEN-Ori `export_split_json.py` 出力) を読み込む (Issue #122)。

    期待 schema:
        {
          "schema_version": "team_ramen_grouped_episode_split_v1" or "team_ramen_split_json_v1",
          "splits": {
            "train": {"episode_indices": [...]},
            "validation": {"episode_indices": [...]},
            "test": {"episode_indices": [...]},
          },
          "seed": int, ...
        }

    Args:
        path: split JSON path
        total_episodes: source dataset の total_episodes (整合性 check 用)

    Returns:
        materialize が使う形式 (build_grouped_episode_split の返り値と互換)
    """
    payload = read_json(path)
    splits = payload.get("splits", {})
    if not all(name in splits for name in ("train", "validation", "test")):
        raise ValueError(
            f"external split JSON must contain 'train', 'validation', 'test' splits, got {list(splits)}"
        )
    all_indices: set[int] = set()
    for name, entry in splits.items():
        indices = entry.get("episode_indices", [])
        if not isinstance(indices, list):
            raise ValueError(f"split '{name}' episode_indices must be a list")
        all_indices.update(int(i) for i in indices)
    if all_indices != set(range(total_episodes)):
        missing = set(range(total_episodes)) - all_indices
        extra = all_indices - set(range(total_episodes))
        raise ValueError(
            f"external split JSON must cover all {total_episodes} episodes exactly once; "
            f"missing={sorted(missing)[:10]}... extra={sorted(extra)[:10]}..."
        )
    return payload


def validate_source_eef_pose_format(source_config: dict[str, Any]) -> None:
    expected = {
        "ordering": "left_then_right",
        "per_hand": "xyz_euler_xyz",
        "angle_unit": "radian",
        "reference_frame": "root_link",
    }
    actual = source_config.get("eef_pose_format")
    if actual != expected:
        raise ValueError(
            "GR00T REAL_G1 mapping requires source EEF poses in left/right "
            f"root-frame XYZ + Euler-XYZ radians; got {actual!r}"
        )


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to materialize the LeRobot training view") from exc
    return pa, pq


def copy_required_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_optional_file(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
