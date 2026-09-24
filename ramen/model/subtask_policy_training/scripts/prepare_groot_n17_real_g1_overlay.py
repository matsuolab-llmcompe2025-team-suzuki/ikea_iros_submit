"""Create a reproducible GR00T N1.7 REAL_G1 processor overlay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


MAPPING_PATH = Path(__file__).resolve().parents[1] / "gr00t" / "g1_full_body_mapping.py"
MARKER_NAME = "team_ramen_groot_overlay.json"
POLICY_VIDEO_KEYS = ["head_left", "left_wrist", "right_wrist"]


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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = resolve_model_snapshot(args.model_path, revision=args.revision)
    overlay = prepare_overlay(
        source_root=source_root,
        output_root=args.output_root,
        model_path=args.model_path,
        revision=args.revision,
        force=args.force,
    )
    print(overlay)


def resolve_model_snapshot(model_path: str, *, revision: str) -> Path:
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    print(
        f"Resolving GR00T N1.7 base model {model_path}@{revision}; the first run downloads the model once.",
        file=sys.stderr,
    )
    return Path(snapshot_download(repo_id=model_path, revision=revision)).resolve()


def prepare_overlay(
    *,
    source_root: Path,
    output_root: Path,
    model_path: str,
    revision: str,
    force: bool,
) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    model_config_path = source_root / "config.json"
    processor_path = source_root / "processor_config.json"
    statistics_path = source_root / "statistics.json"
    embodiment_path = source_root / "embodiment_id.json"
    for required_path in (
        model_config_path,
        embodiment_path,
        processor_path,
        statistics_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    model_config = read_json(model_config_path)
    processor_config = read_json(processor_path)
    statistics = read_json(statistics_path)
    embodiment_ids = read_json(embodiment_path)
    validate_real_g1_contract(model_config, processor_config, statistics, embodiment_ids)
    source_processor_sha256 = sha256_file(processor_path)
    marker = {
        "schema_version": "team_ramen_groot_n17_overlay_v1",
        "source_model_path": model_path,
        "requested_revision": revision,
        "resolved_source_root": source_root.as_posix(),
        "source_processor_sha256": source_processor_sha256,
        "embodiment_tag": _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
        "embodiment_id": _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_ID,
        "video_delta_indices": [0],
        "video_modality_keys": POLICY_VIDEO_KEYS,
        "state_dim": _mapping.REAL_G1_RELATIVE_EEF_STATE_DIM,
        "action_dim": _mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM,
        "action_configs": _mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS,
    }

    marker_path = output_root / MARKER_NAME
    if output_root.exists() and not force:
        if marker_path.is_file() and read_json(marker_path) == marker:
            return output_root
        raise FileExistsError(f"{output_root} exists with a different overlay contract; pass --force")

    temporary = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for source in source_root.iterdir():
            if source.name == "processor_config.json":
                continue
            (temporary / source.name).symlink_to(source, target_is_directory=source.is_dir())

        overlay_config = json.loads(json.dumps(processor_config))
        modality = overlay_config["processor_kwargs"]["modality_configs"][
            _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG
        ]
        modality["video"] = {
            "delta_indices": [0],
            "modality_keys": POLICY_VIDEO_KEYS,
        }
        write_json(temporary / "processor_config.json", overlay_config)
        write_json(temporary / MARKER_NAME, marker)

        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root


def validate_real_g1_contract(
    model_config: dict[str, Any],
    processor_config: dict[str, Any],
    statistics: dict[str, Any],
    embodiment_ids: dict[str, Any],
) -> None:
    expected_horizon = _mapping.GROOT_N17_NATIVE_ACTION_HORIZON
    if model_config.get("model_type") != "Gr00tN1d7":
        raise ValueError("base model is not a GR00T N1.7 checkpoint")
    if model_config.get("action_horizon") != expected_horizon:
        raise ValueError("GR00T N1.7 model action_horizon differs from the checked mapping")

    processor_kwargs = processor_config.get("processor_kwargs", {})
    if processor_kwargs.get("use_relative_action") is not True:
        raise ValueError("GR00T N1.7 base processor does not enable native relative actions")
    modality_configs = processor_kwargs.get("modality_configs", {})
    tag = _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG
    if tag not in modality_configs:
        raise KeyError(f"GR00T N1.7 base processor is missing {tag!r}")
    if embodiment_ids.get(tag) != _mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_ID:
        raise ValueError(
            f"REAL_G1 embodiment id is {embodiment_ids.get(tag)!r}, expected "
            f"{_mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_ID}"
        )
    modality = modality_configs[tag]
    expected_state_keys = list(_mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES)
    expected_action_keys = list(_mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES)
    if modality.get("state", {}).get("modality_keys") != expected_state_keys:
        raise ValueError("REAL_G1 state group order differs from the checked mapping")
    if modality.get("action", {}).get("modality_keys") != expected_action_keys:
        raise ValueError("REAL_G1 action group order differs from the checked mapping")

    action_configs = modality.get("action", {}).get("action_configs", [])
    expected_action_configs = [
        _mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS[key] for key in expected_action_keys
    ]
    if action_configs != expected_action_configs:
        raise ValueError("REAL_G1 action representations differ from the checked mapping")
    if modality.get("video") != {
        "delta_indices": [-20, 0],
        "modality_keys": ["ego_view"],
    }:
        raise ValueError("REAL_G1 source video contract differs from the pinned checkpoint")
    if modality.get("state", {}).get("delta_indices") != [0]:
        raise ValueError("REAL_G1 source state horizon differs from the pinned checkpoint")
    action_delta_indices = modality.get("action", {}).get("delta_indices")
    if action_delta_indices != list(range(expected_horizon)):
        raise ValueError("REAL_G1 source action horizon differs from the pinned checkpoint")
    if processor_kwargs.get("max_action_horizon") != expected_horizon:
        raise ValueError("GR00T N1.7 max_action_horizon differs from the checked mapping")

    tag_stats = statistics.get(tag, {})
    state_dim = grouped_stats_dim(tag_stats.get("state", {}), expected_state_keys)
    action_dim = grouped_stats_dim(tag_stats.get("action", {}), expected_action_keys)
    if state_dim != _mapping.REAL_G1_RELATIVE_EEF_STATE_DIM:
        raise ValueError(f"REAL_G1 state statistics have {state_dim} dimensions")
    if action_dim != _mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM:
        raise ValueError(f"REAL_G1 action statistics have {action_dim} dimensions")

    relative_stats = tag_stats.get("relative_action", {})
    for key in ("left_wrist_eef_9d", "right_wrist_eef_9d", "left_arm", "right_arm"):
        start, end = _mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES[key]
        expected_dim = end - start
        values = relative_stats.get(key, {}).get("mean")
        if (
            not isinstance(values, list)
            or len(values) != expected_horizon
            or any(not isinstance(row, list) or len(row) != expected_dim for row in values)
        ):
            raise ValueError(
                f"REAL_G1 relative_action statistics for {key} must be "
                f"[{expected_horizon}, {expected_dim}]"
            )


def grouped_stats_dim(stats: dict[str, Any], keys: list[str]) -> int:
    total = 0
    for key in keys:
        values = stats.get(key, {}).get("mean")
        if not isinstance(values, list):
            raise KeyError(f"missing mean statistics for {key}")
        total += len(values)
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
