#!/usr/bin/env python3
"""Audit the structural and numerical quality of a LeRobot v3 subtask dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq


REQUIRED_CAMERAS = (
    "observation.images.cam_0",
    "observation.images.cam_2",
    "observation.images.cam_3",
)
REQUIRED_NUMERIC_FEATURES = {
    "observation.state.ee_state": 12,
    "observation.state.robot_q_current": 36,
    "observation.state.hand_state": 2,
    "action.ee_action": 12,
    "action.robot_q_desired": 36,
    "action.hand_cmd": 2,
}
INDEX_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_OUTPUT = FEATURE_ROOT / "outputs" / "audits" / "flip_table_dataset_audit.json"
DEFAULT_SPLIT_OUTPUT = FEATURE_ROOT / "outputs" / "audits" / "flip_table_episode_split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1")
    parser.add_argument("--revision")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument(
        "--split-output",
        type=Path,
        default=DEFAULT_SPLIT_OUTPUT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--offline", action="store_true", help="Skip the Hugging Face remote inventory check")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when warnings are present")
    return parser.parse_args()


def resolve_dataset_root(
    repo_id: str,
    dataset_root: Path | None,
    revision: str | None,
) -> Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=("README.md", "meta/**", "data/**"),
        )
    ).resolve()


def read_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info_path = root / "meta" / "info.json"
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta' / 'episodes'}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes: list[dict[str, Any]] = []
    for path in episode_files:
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda row: int(row["episode_index"]))
    return info, episodes


def audit_episode_metadata(
    info: dict[str, Any], episodes: list[dict[str, Any]]
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_episodes = int(info.get("total_episodes", -1))
    expected_frames = int(info.get("total_frames", -1))
    fps = float(info.get("fps", 0.0))

    if not str(info.get("codebase_version", "")).startswith("v3"):
        errors.append(f"codebase_version is not LeRobot v3: {info.get('codebase_version')!r}")
    if not math.isfinite(fps) or fps <= 0:
        errors.append(f"dataset fps must be positive and finite, got {fps!r}")

    if expected_episodes != len(episodes):
        errors.append(f"info.total_episodes={expected_episodes}, episode rows={len(episodes)}")
    indices = [int(row["episode_index"]) for row in episodes]
    if indices != list(range(len(episodes))):
        errors.append("episode_index is not contiguous from zero")

    lengths = np.asarray([int(row["length"]) for row in episodes], dtype=np.int64)
    if np.any(lengths <= 0):
        errors.append("one or more episodes have a non-positive length")
    if int(lengths.sum()) != expected_frames:
        errors.append(f"sum(episode.length)={int(lengths.sum())}, info.total_frames={expected_frames}")

    expected_from = 0
    bad_ranges: list[int] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        length = int(row["length"])
        if start != expected_from or end - start != length:
            bad_ranges.append(episode_index)
        expected_from = end
    if bad_ranges:
        errors.append(f"invalid or non-contiguous dataset ranges in episodes {bad_ranges[:20]}")
    if expected_from != expected_frames:
        errors.append(f"last dataset_to_index={expected_from}, info.total_frames={expected_frames}")

    source_names = [str(row.get("source_episode_name", "")) for row in episodes]
    missing_source = [indices[i] for i, name in enumerate(source_names) if not name]
    if missing_source:
        errors.append(f"missing source_episode_name in episodes {missing_source[:20]}")
    source_counts = Counter(source_names)
    duplicate_sources = sorted(name for name, count in source_counts.items() if name and count > 1)

    source_intervals: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    invalid_source_intervals: list[int] = []
    for row in episodes:
        start = float(row.get("source_start_sec", math.nan))
        end = float(row.get("source_end_sec", math.nan))
        episode_index = int(row["episode_index"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            invalid_source_intervals.append(episode_index)
        source_intervals[str(row.get("source_episode_name", ""))].append((start, end, episode_index))
    if invalid_source_intervals:
        errors.append(f"invalid source time ranges in episodes {invalid_source_intervals[:20]}")

    overlaps: list[tuple[int, int]] = []
    for intervals in source_intervals.values():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:], strict=False):
            if current[0] < previous[1]:
                overlaps.append((previous[2], current[2]))
    if overlaps:
        warnings.append(f"overlapping slices from the same source episode: {overlaps[:20]}")

    invalid_task_rows = [
        int(row["episode_index"])
        for row in episodes
        if not isinstance(row.get("tasks"), list)
        or not all(isinstance(task, str) and task for task in row["tasks"])
    ]
    if invalid_task_rows:
        errors.append(f"invalid task labels in episodes {invalid_task_rows[:20]}")
    tasks = sorted(
        {
            task
            for row in episodes
            for task in (row.get("tasks") if isinstance(row.get("tasks"), list) else [])
        }
    )
    if len(tasks) != 1:
        warnings.append(f"expected one subtask but found {tasks}")
    if not any("source_episode_name" in row for row in episodes):
        errors.append("source provenance is absent; leakage-safe splitting is impossible")

    features = info.get("features", {})
    for camera in REQUIRED_CAMERAS:
        feature = features.get(camera)
        if feature is None:
            errors.append(f"missing required camera feature {camera}")
            continue
        if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
            errors.append(f"unexpected camera metadata for {camera}: {feature}")
        video_fps = float(feature.get("info", {}).get("video.fps", 0.0))
        if abs(video_fps - fps) > 1e-6:
            errors.append(f"camera fps mismatch for {camera}: {video_fps} vs dataset {fps}")
    for key, width in REQUIRED_NUMERIC_FEATURES.items():
        feature = features.get(key)
        if feature is None:
            errors.append(f"missing required numeric feature {key}")
        elif feature.get("dtype") != "float32" or feature.get("shape") != [width]:
            errors.append(
                f"unexpected numeric metadata for {key}: "
                f"dtype={feature.get('dtype')!r}, shape={feature.get('shape')!r}"
            )

    if "success" not in features and "next.done" not in features:
        warnings.append("no explicit per-episode success label; demonstration success needs visual audit")

    duration_s = lengths / fps if fps > 0 else np.full_like(lengths, np.nan, dtype=np.float64)
    summary = {
        "episodes": len(episodes),
        "frames": int(lengths.sum()),
        "fps": fps,
        "tasks": tasks,
        "unique_source_episodes": len(source_counts),
        "duplicate_source_episode_names": duplicate_sources,
        "episode_length_frames": distribution_summary(lengths.astype(np.float64)),
        "episode_duration_seconds": distribution_summary(duration_s.astype(np.float64)),
    }
    return errors, warnings, summary


def list_array_to_numpy(array: Any, expected_width: int) -> np.ndarray:
    if array.null_count:
        raise ValueError("numeric list column contains null rows")
    if hasattr(array, "offsets"):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        widths = np.diff(offsets)
        if not np.all(widths == expected_width):
            unique_widths = np.unique(widths).tolist()
            raise ValueError(f"row widths {unique_widths} do not match {expected_width}")
    else:
        list_size = getattr(array.type, "list_size", None)
        if list_size != expected_width:
            raise ValueError(f"fixed row width {list_size} does not match {expected_width}")
    if array.values.null_count:
        raise ValueError("numeric list column contains null values")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    if values.size != len(array) * expected_width:
        raise ValueError(
            f"numeric list column has {values.size} values for {len(array)}x{expected_width} rows"
        )
    return values.reshape(len(array), expected_width)


def audit_numeric_data(
    root: Path, info: dict[str, Any], episodes: list[dict[str, Any]]
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    data_root = root / "data"
    data_files = sorted(data_root.glob("chunk-*/*.parquet"))
    if not data_files:
        return [f"no parquet data files under {data_root}"], warnings, {}

    schema_names = set(pads.dataset([str(path) for path in data_files], format="parquet").schema.names)
    required_columns = set(REQUIRED_NUMERIC_FEATURES) | set(INDEX_COLUMNS)
    missing_columns = sorted(required_columns - schema_names)
    if missing_columns:
        return [f"missing required parquet columns: {missing_columns}"], warnings, {}

    feature_min = {key: np.full(width, np.inf) for key, width in REQUIRED_NUMERIC_FEATURES.items()}
    feature_max = {key: np.full(width, -np.inf) for key, width in REQUIRED_NUMERIC_FEATURES.items()}
    nonfinite = Counter()
    row_count = 0
    expected_global_index = 0
    expected_frame_by_episode: dict[int, int] = defaultdict(int)
    previous_action: dict[int, np.ndarray] = {}
    previous_timestamp: dict[int, float] = {}
    action_step_maxima: list[np.ndarray] = []
    timestamp_steps: list[np.ndarray] = []
    command_tracking_error: list[np.ndarray] = []
    episode_action_min: dict[int, np.ndarray] = {}
    episode_action_max: dict[int, np.ndarray] = {}
    bad_index_rows: list[int] = []
    bad_frame_rows: list[int] = []
    bad_episode_rows: list[int] = []

    columns = list(REQUIRED_NUMERIC_FEATURES) + list(INDEX_COLUMNS)
    for path in data_files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16_384, columns=columns):
            arrays: dict[str, np.ndarray] = {}
            for key, width in REQUIRED_NUMERIC_FEATURES.items():
                try:
                    values = list_array_to_numpy(batch.column(batch.schema.get_field_index(key)), width)
                except ValueError as exc:
                    errors.append(f"{path.name}:{key}: {exc}")
                    continue
                arrays[key] = values
                nonfinite[key] += int((~np.isfinite(values)).sum())
                finite_values = np.where(np.isfinite(values), values, np.nan)
                with np.errstate(all="ignore"):
                    feature_min[key] = np.minimum(feature_min[key], np.nanmin(finite_values, axis=0))
                    feature_max[key] = np.maximum(feature_max[key], np.nanmax(finite_values, axis=0))
            if len(arrays) != len(REQUIRED_NUMERIC_FEATURES):
                continue

            global_indices = np.asarray(batch.column(batch.schema.get_field_index("index")))
            episode_indices = np.asarray(batch.column(batch.schema.get_field_index("episode_index")))
            frame_indices = np.asarray(batch.column(batch.schema.get_field_index("frame_index")))
            timestamps = np.asarray(batch.column(batch.schema.get_field_index("timestamp")), dtype=np.float64)
            task_indices = np.asarray(batch.column(batch.schema.get_field_index("task_index")))

            for local_index, (global_index, episode_index, frame_index) in enumerate(
                zip(global_indices, episode_indices, frame_indices, strict=True)
            ):
                global_index = int(global_index)
                episode_index = int(episode_index)
                frame_index = int(frame_index)
                if global_index != expected_global_index and len(bad_index_rows) < 20:
                    bad_index_rows.append(global_index)
                expected_global_index += 1
                expected_frame = expected_frame_by_episode[episode_index]
                if frame_index != expected_frame and len(bad_frame_rows) < 20:
                    bad_frame_rows.append(global_index)
                expected_frame_by_episode[episode_index] = expected_frame + 1
                if episode_index < 0 or episode_index >= len(episodes):
                    if len(bad_episode_rows) < 20:
                        bad_episode_rows.append(global_index)

                action = arrays["action.robot_q_desired"][local_index]
                hand = arrays["action.hand_cmd"][local_index]
                full_action = np.concatenate((action, hand))
                if episode_index not in episode_action_min:
                    episode_action_min[episode_index] = full_action.copy()
                    episode_action_max[episode_index] = full_action.copy()
                else:
                    episode_action_min[episode_index] = np.minimum(episode_action_min[episode_index], full_action)
                    episode_action_max[episode_index] = np.maximum(episode_action_max[episode_index], full_action)

            same_episode = episode_indices[1:] == episode_indices[:-1]
            action = np.concatenate(
                (arrays["action.robot_q_desired"], arrays["action.hand_cmd"]), axis=1
            )
            if len(action) > 1:
                action_step_maxima.append(np.max(np.abs(np.diff(action, axis=0)[same_episode]), axis=1))
                timestamp_steps.append(np.diff(timestamps)[same_episode])
            if len(action):
                first_episode = int(episode_indices[0])
                if first_episode in previous_action and int(frame_indices[0]) > 0:
                    action_step_maxima.append(
                        np.asarray([np.max(np.abs(action[0] - previous_action[first_episode]))])
                    )
                    timestamp_steps.append(
                        np.asarray([timestamps[0] - previous_timestamp[first_episode]])
                    )
                previous_action[int(episode_indices[-1])] = action[-1].copy()
                previous_timestamp[int(episode_indices[-1])] = float(timestamps[-1])

            tracking = np.concatenate(
                (
                    arrays["action.robot_q_desired"] - arrays["observation.state.robot_q_current"],
                    arrays["action.hand_cmd"] - arrays["observation.state.hand_state"],
                ),
                axis=1,
            )
            command_tracking_error.append(np.max(np.abs(tracking), axis=1))
            if np.any(task_indices != 0):
                warnings.append("task_index contains values other than zero")
            row_count += len(batch)

    warnings = list(dict.fromkeys(warnings))
    for key, count in nonfinite.items():
        if count:
            errors.append(f"{key} contains {count} non-finite values")
    if row_count != int(info.get("total_frames", -1)):
        errors.append(f"parquet rows={row_count}, info.total_frames={info.get('total_frames')}")
    if bad_index_rows:
        errors.append(f"global index is not contiguous near rows {bad_index_rows}")
    if bad_frame_rows:
        errors.append(f"frame_index is not contiguous within episodes near rows {bad_frame_rows}")
    if bad_episode_rows:
        errors.append(f"invalid episode_index near rows {bad_episode_rows}")

    episode_spans = {
        episode_index: float(np.max(episode_action_max[episode_index] - minimum))
        for episode_index, minimum in episode_action_min.items()
    }
    nearly_static = sorted(index for index, span in episode_spans.items() if span < 0.02)
    if nearly_static:
        warnings.append(f"episodes with <0.02 maximum action span: {nearly_static[:50]}")

    step_maxima = concatenate_or_empty(action_step_maxima)
    timestamp_delta = concatenate_or_empty(timestamp_steps)
    tracking_error = concatenate_or_empty(command_tracking_error)
    expected_transitions = int(info["total_frames"]) - int(info["total_episodes"])
    if len(timestamp_delta) != expected_transitions:
        errors.append(
            f"timestamp transitions={len(timestamp_delta)}, expected={expected_transitions}"
        )
    fps = float(info.get("fps", 0.0))
    if math.isfinite(fps) and fps > 0:
        expected_dt = 1.0 / fps
        bad_dt = int(np.sum(np.abs(timestamp_delta - expected_dt) > 2e-4))
        if bad_dt:
            errors.append(
                f"{bad_dt} timestamp steps differ from {expected_dt:.9f}s by more than 0.2ms"
            )

    summary = {
        "data_files": [str(path.relative_to(root)) for path in data_files],
        "rows": row_count,
        "feature_ranges": {
            key: {"min": minimum.tolist(), "max": feature_max[key].tolist()}
            for key, minimum in feature_min.items()
        },
        "action_step_max_abs": distribution_summary(step_maxima),
        "command_tracking_max_abs": distribution_summary(tracking_error),
        "timestamp_delta_seconds": distribution_summary(timestamp_delta),
        "nearly_static_episode_indices": nearly_static,
        "episode_action_span": distribution_summary(np.asarray(list(episode_spans.values()))),
    }
    return errors, list(dict.fromkeys(warnings)), summary


def audit_local_videos(root: Path, info: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return errors, ["ffprobe is unavailable; local MP4 streams were not probed"], {"skipped": True}

    expected_fps = float(info["fps"])
    expected_shape = (640, 480)
    cameras: dict[str, Any] = {}
    for camera in REQUIRED_CAMERAS:
        paths = sorted((root / "videos" / camera).glob("chunk-*/*.mp4"))
        camera_summary = {"files": [], "total_encoded_frames": 0}
        if not paths:
            warnings.append(f"{camera} videos are not local; ffprobe stream validation was skipped")
            cameras[camera] = {"skipped": True, "reason": "not present in local snapshot"}
            continue
        for path in paths:
            process = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            relative_path = str(path.relative_to(root))
            if process.returncode != 0:
                errors.append(f"ffprobe failed for {relative_path}: {process.stderr.strip()}")
                continue
            payload = json.loads(process.stdout)
            streams = payload.get("streams", [])
            if len(streams) != 1:
                errors.append(f"expected one video stream in {relative_path}, found {len(streams)}")
                continue
            stream = streams[0]
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
            frames = int(stream.get("nb_frames", 0))
            if (width, height) != expected_shape:
                errors.append(f"unexpected resolution in {relative_path}: {width}x{height}")
            if abs(fps - expected_fps) > 1e-6:
                errors.append(f"unexpected fps in {relative_path}: {fps}")
            if frames <= 0:
                errors.append(f"no encoded frames reported in {relative_path}")
            camera_summary["total_encoded_frames"] += frames
            camera_summary["files"].append(
                {
                    "path": relative_path,
                    "codec": stream.get("codec_name"),
                    "pixel_format": stream.get("pix_fmt"),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frames": frames,
                    "duration_seconds": float(stream.get("duration", 0.0)),
                }
            )
        cameras[camera] = camera_summary
    return errors, warnings, {"cameras": cameras}


def audit_remote_inventory(
    repo_id: str,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    revision: str | None,
) -> tuple[list[str], dict[str, Any]]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    video_inventory: dict[str, Any] = {}
    video_path_template = str(
        info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    )
    errors: list[str] = []
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        prefix = f"videos/{key}/"
        actual = {path for path in files if path.startswith(prefix) and path.endswith(".mp4")}
        chunk_column = f"videos/{key}/chunk_index"
        file_column = f"videos/{key}/file_index"
        missing_metadata = [
            int(row["episode_index"])
            for row in episodes
            if chunk_column not in row or file_column not in row
        ]
        if missing_metadata:
            errors.append(
                f"video metadata for {key} is missing in episodes {missing_metadata[:20]}"
            )
            expected: set[str] = set()
        else:
            expected = {
                video_path_template.format(
                    video_key=key,
                    chunk_index=int(row[chunk_column]),
                    file_index=int(row[file_column]),
                )
                for row in episodes
            }
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"Hub is missing {len(missing)} expected {key} videos: {missing[:10]}")
        if extra:
            errors.append(f"Hub contains {len(extra)} stale {key} videos: {extra[:10]}")
        video_inventory[key] = {
            "expected_count": len(expected),
            "actual_count": len(actual),
            "missing_count": len(missing),
            "extra_count": len(extra),
        }
    return errors, {"repo_files": len(files), "videos": video_inventory}


def assign_grouped_splits(
    episodes: list[dict[str, Any]], validation_fraction: float, test_fraction: float, seed: int
) -> dict[str, Any]:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be non-negative and sum to less than one")

    groups: dict[str, list[int]] = defaultdict(list)
    for row in episodes:
        source_name = str(row.get("source_episode_name", ""))
        if not source_name:
            raise ValueError("source_episode_name is required for grouped splitting")
        groups[source_name].append(int(row["episode_index"]))

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    total_groups = len(group_names)
    test_count = round(total_groups * test_fraction)
    validation_count = round(total_groups * validation_fraction)
    if test_fraction > 0 and test_count == 0:
        test_count = 1
    if validation_fraction > 0 and validation_count == 0:
        validation_count = 1
    if test_count + validation_count >= total_groups:
        raise ValueError("grouped split leaves no training source recordings")
    split_groups = {
        "test": group_names[:test_count],
        "validation": group_names[test_count : test_count + validation_count],
        "train": group_names[test_count + validation_count :],
    }
    return {
        "schema_version": "lerobot_grouped_episode_split_v1",
        "seed": seed,
        "group_key": "source_episode_name",
        "fractions": {
            "train": 1.0 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "splits": {
            name: {
                "source_episode_names": names,
                "episode_indices": sorted(index for source in names for index in groups[source]),
            }
            for name, names in split_groups.items()
        },
    }


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def concatenate_or_empty(parts: list[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(part) for part in parts if len(part)]
    return np.concatenate(nonempty) if nonempty else np.asarray([], dtype=np.float64)


def split_digest(split: dict[str, Any]) -> str:
    payload = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    root = resolve_dataset_root(args.repo_id, args.dataset_root, args.revision)
    info, episodes = read_metadata(root)

    metadata_errors, metadata_warnings, metadata_summary = audit_episode_metadata(info, episodes)
    numeric_errors, numeric_warnings, numeric_summary = audit_numeric_data(root, info, episodes)
    video_errors, video_warnings, video_summary = audit_local_videos(root, info)
    remote_errors: list[str] = []
    remote_summary: dict[str, Any] = {"skipped": True}
    if not args.offline:
        remote_errors, remote_summary = audit_remote_inventory(
            args.repo_id,
            info,
            episodes,
            args.revision,
        )

    split = assign_grouped_splits(episodes, args.validation_fraction, args.test_fraction, args.seed)
    split["sha256"] = split_digest(split)
    errors = metadata_errors + numeric_errors + video_errors + remote_errors
    warnings = metadata_warnings + numeric_warnings + video_warnings
    report = {
        "schema_version": "lerobot_dataset_audit_v1",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "dataset_root": str(root),
        "status": "error" if errors else "warning" if warnings else "ok",
        "errors": errors,
        "warnings": warnings,
        "metadata": metadata_summary,
        "numeric_data": numeric_summary,
        "local_video_probe": video_summary,
        "remote_inventory": remote_summary,
        "split_sha256": split["sha256"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.split_output.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"status": report["status"], "errors": errors, "warnings": warnings}, indent=2))
    print(f"audit: {args.output}")
    print(f"split: {args.split_output}")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
