#!/usr/bin/env python3
"""Audit coarse-insert ranges against GR00T statistics and G1 URDF limits."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gr00t.g1_full_body_mapping import (
    G1_SDK_JOINT_NAMES,
    REAL_G1_RELATIVE_EEF_ACTION_SLICES,
    REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
    REAL_G1_RELATIVE_EEF_STATE_SLICES,
)


def parse_args() -> argparse.Namespace:
    feature_root = Path(__file__).resolve().parents[1]
    repo_root = feature_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=feature_root / "outputs/training_views/groot_coarse_insert_sdk_ids_v2",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=feature_root
        / "outputs/groot_base_overlays/real_g1_relative_eef_3cam_dex1_v2",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=repo_root
        / "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=feature_root / "outputs/evaluations/coarse_insert_range_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    files = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {args.dataset_root / 'data'}")
    table = pq.read_table(
        files,
        columns=["observation.state", "action", "episode_index", "frame_index"],
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if state.shape[1] != 49 or action.shape[1] != 53:
        raise ValueError(f"expected state/action 49/53, got {state.shape}/{action.shape}")

    statistics = json.loads((args.base_model / "statistics.json").read_text(encoding="utf-8"))[
        REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG
    ]
    report: dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "base_model": str(args.base_model.resolve()),
        "urdf": str(args.urdf.resolve()),
        "rows": int(len(state)),
        "episodes": int(np.unique(episodes).size),
        "horizon": args.horizon,
        "state_vs_pretrained_q01_q99": {},
        "action_vs_pretrained_q01_q99": {},
        "relative_action_vs_pretrained_q01_q99": {},
    }

    for key, (start, end) in REAL_G1_RELATIVE_EEF_STATE_SLICES.items():
        values = state[:, start:end]
        base = statistics["state"][key]
        report["state_vs_pretrained_q01_q99"][key] = range_comparison(
            values,
            np.asarray(base["q01"][: end - start]),
            np.asarray(base["q99"][: end - start]),
        )

    # Absolute output groups. EEF and arms are checked after the exact relative
    # conversion below; base/navigation are deliberately loss-masked.
    for key in ("left_hand", "right_hand", "waist"):
        start, end = REAL_G1_RELATIVE_EEF_ACTION_SLICES[key]
        values = action[:, start:end]
        base = statistics["action"][key]
        report["action_vs_pretrained_q01_q99"][key] = range_comparison(
            values,
            np.asarray(base["q01"][: end - start]),
            np.asarray(base["q99"][: end - start]),
        )

    relative_groups: dict[str, list[np.ndarray]] = {
        "left_wrist_eef_9d": [],
        "right_wrist_eef_9d": [],
        "left_arm": [],
        "right_arm": [],
    }
    for horizon in range(args.horizon):
        future = np.arange(len(state)) + horizon
        valid = future < len(state)
        valid &= episodes[np.minimum(future, len(state) - 1)] == episodes
        valid &= frames[np.minimum(future, len(state) - 1)] == frames + horizon
        current_indices = np.nonzero(valid)[0]
        target_indices = current_indices + horizon
        for key in ("left_arm", "right_arm"):
            action_start, action_end = REAL_G1_RELATIVE_EEF_ACTION_SLICES[key]
            state_start, state_end = REAL_G1_RELATIVE_EEF_STATE_SLICES[key]
            relative_groups[key].append(
                action[target_indices, action_start:action_end]
                - state[current_indices, state_start:state_end]
            )
        for key in ("left_wrist_eef_9d", "right_wrist_eef_9d"):
            action_start, action_end = REAL_G1_RELATIVE_EEF_ACTION_SLICES[key]
            state_start, state_end = REAL_G1_RELATIVE_EEF_STATE_SLICES[key]
            relative_groups[key].append(
                relative_eef(
                    state[current_indices, state_start:state_end],
                    action[target_indices, action_start:action_end],
                )
            )

    for key, by_horizon in relative_groups.items():
        base = statistics["relative_action"][key]
        comparisons = []
        for horizon, values in enumerate(by_horizon):
            comparisons.append(
                range_comparison(
                    values,
                    np.asarray(base["q01"][horizon]),
                    np.asarray(base["q99"][horizon]),
                )
            )
        report["relative_action_vs_pretrained_q01_q99"][key] = {
            "outside_fraction_all_horizons": float(
                np.mean([value["outside_fraction"] for value in comparisons])
            ),
            "max_outside_fraction_one_horizon": float(
                max(value["outside_fraction"] for value in comparisons)
            ),
            "per_horizon": comparisons,
        }

    report["urdf_joint_limits"] = audit_urdf_limits(action, args.urdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


def range_comparison(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or lower.shape != (values.shape[1],) or upper.shape != lower.shape:
        raise ValueError(f"range shape mismatch: values={values.shape}, lower={lower.shape}")
    outside = (values < lower[None, :]) | (values > upper[None, :])
    return {
        "data_q01": np.quantile(values, 0.01, axis=0).tolist(),
        "data_q99": np.quantile(values, 0.99, axis=0).tolist(),
        "pretrained_q01": lower.tolist(),
        "pretrained_q99": upper.tolist(),
        "outside_fraction": float(outside.mean()),
        "outside_fraction_per_dimension": outside.mean(axis=0).tolist(),
    }


def rot6d_rows_to_matrix(values: np.ndarray) -> np.ndarray:
    row0 = values[:, :3]
    row0 = row0 / np.linalg.norm(row0, axis=1, keepdims=True)
    row1 = values[:, 3:6] - np.sum(
        row0 * values[:, 3:6], axis=1, keepdims=True
    ) * row0
    row1 = row1 / np.linalg.norm(row1, axis=1, keepdims=True)
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=1)


def relative_eef(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current_rotation = rot6d_rows_to_matrix(current[:, 3:])
    target_rotation = rot6d_rows_to_matrix(target[:, 3:])
    current_rotation_t = np.swapaxes(current_rotation, 1, 2)
    translation = np.einsum(
        "nij,nj->ni", current_rotation_t, target[:, :3] - current[:, :3]
    )
    rotation = np.einsum("nij,njk->nik", current_rotation_t, target_rotation)
    return np.concatenate((translation, rotation[:, :2, :].reshape(-1, 6)), axis=1)


def audit_urdf_limits(action: np.ndarray, urdf: Path) -> dict[str, Any]:
    root = ET.parse(urdf).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    action_indices = list(range(46, 49)) + list(range(32, 46))
    sdk_ids = list(range(12, 29))
    joints: dict[str, Any] = {}
    total_violations = 0
    for action_index, sdk_id in zip(action_indices, sdk_ids):
        name = G1_SDK_JOINT_NAMES[sdk_id]
        lower, upper = limits[name]
        values = action[:, action_index]
        violations = int(np.count_nonzero((values < lower) | (values > upper)))
        total_violations += violations
        joints[name] = {
            "sdk_id": sdk_id,
            "data_min": float(values.min()),
            "data_max": float(values.max()),
            "urdf_lower": lower,
            "urdf_upper": upper,
            "violation_count": violations,
        }
    return {"total_violation_count": total_violations, "joints": joints}


if __name__ == "__main__":
    main()
