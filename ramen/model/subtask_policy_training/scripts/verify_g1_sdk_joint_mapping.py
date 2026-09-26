#!/usr/bin/env python3
"""Verify and print the BitRobot robot_q index to Unitree G1 SDK id mapping."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


FEATURE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FEATURE_ROOT.parents[1]
BRIDGE_MAPPING_PATH = (
    REPOSITORY_ROOT
    / "inference"
    / "orin"
    / "ros2_ws"
    / "src"
    / "g1_hw_bridge"
    / "g1_hw_bridge"
    / "joint_mapping.py"
)
sys.path.insert(0, str(FEATURE_ROOT))

from gr00t import g1_full_body_mapping as training_mapping  # noqa: E402


def load_bridge_mapping() -> Any:
    spec = importlib.util.spec_from_file_location("g1_hw_bridge_joint_mapping", BRIDGE_MAPPING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load G1 bridge mapping from {BRIDGE_MAPPING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_mapping_rows() -> list[dict[str, int | str]]:
    bridge_mapping = load_bridge_mapping()
    sdk_names = tuple(training_mapping.G1_SDK_JOINT_NAMES)
    bridge_names = tuple(bridge_mapping.G1_JOINT_NAMES)
    if sdk_names != bridge_names:
        mismatches = [
            {
                "sdk_id": sdk_id,
                "training": training_name,
                "bridge": bridge_name,
            }
            for sdk_id, (training_name, bridge_name) in enumerate(
                zip(sdk_names, bridge_names, strict=False)
            )
            if training_name != bridge_name
        ]
        raise ValueError(
            "training and runtime G1 SDK joint orders differ: "
            f"training={len(sdk_names)}, bridge={len(bridge_names)}, mismatches={mismatches}"
        )
    if len(sdk_names) != 29:
        raise ValueError(f"G1 SDK mapping must contain 29 body motors, got {len(sdk_names)}")

    synthetic_robot_q = [float(index) for index in range(training_mapping.SOURCE_ROBOT_Q_DIM)]
    mapped = training_mapping.map_dataset_robot_q_to_sdk_order(synthetic_robot_q)
    expected = [float(index) for index in range(7, 36)]
    if mapped != expected:
        raise ValueError(f"robot_q root-offset mapping is invalid: {mapped}")

    return [
        {
            "sdk_id": sdk_id,
            "joint_name": joint_name,
            "dataset_robot_q_index": training_mapping.dataset_robot_q_index_from_sdk_id(sdk_id),
        }
        for sdk_id, joint_name in enumerate(sdk_names)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = verified_mapping_rows()
    if args.format == "json":
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    print("sdk_id  dataset_index  joint_name")
    for row in rows:
        print(f"{row['sdk_id']:>6}  {row['dataset_robot_q_index']:>13}  {row['joint_name']}")
    print("Verified: dataset robot_q[7 + sdk_id] matches Unitree G1 SDK motor id 0..28")


if __name__ == "__main__":
    main()
