#!/usr/bin/env python3
"""Run a GR00T checkpoint on a held-out dataset frame and visualize G1 motion."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


ACTION_GROUPS = {
    "left_eef_translation": slice(0, 3),
    "left_eef_rotation6d": slice(3, 9),
    "right_eef_translation": slice(9, 12),
    "right_eef_rotation6d": slice(12, 18),
    "left_hand": slice(18, 25),
    "right_hand": slice(25, 32),
    "left_arm": slice(32, 39),
    "right_arm": slice(39, 46),
    "waist": slice(46, 49),
    "base_height": slice(49, 50),
    "navigate": slice(50, 53),
}
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINTS)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ACTION_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS
ACTION_JOINT_SDK_IDS = tuple(range(15, 22)) + tuple(range(22, 29)) + tuple(range(12, 15))


@dataclass
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    feature_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=feature_root / "outputs/train/groot_coarse_insert_2gpu/checkpoints/020000/pretrained_model",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=feature_root / "outputs/training_views/groot_coarse_insert_sdk_ids_v2",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=feature_root
        / "outputs/training_views/groot_coarse_insert_sdk_ids_v2/meta/team_ramen_episode_split.json",
    )
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--frame-ratio", type=float, default=0.5)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=feature_root / "outputs/evaluations/groot_coarse_insert_20k",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=repo_root
        / "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf",
    )
    parser.add_argument(
        "--source-overlay",
        type=Path,
        default=feature_root / "outputs/lerobot_source_overlays/groot_relative_eef_v3",
    )
    return parser.parse_args()


def vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.asarray(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        ),
        dtype=np.float64,
    )


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def load_urdf(path: Path) -> tuple[str, list[UrdfJoint]]:
    root = ET.parse(path).getroot()
    links = {entry.attrib["name"] for entry in root.findall("link")}
    joints: list[UrdfJoint] = []
    children: set[str] = set()
    for entry in root.findall("joint"):
        origin = entry.find("origin")
        axis = entry.find("axis")
        limit = entry.find("limit")
        parent = entry.find("parent").attrib["link"]
        child = entry.find("child").attrib["link"]
        children.add(child)
        joints.append(
            UrdfJoint(
                name=entry.attrib["name"],
                kind=entry.attrib.get("type", "fixed"),
                parent=parent,
                child=child,
                xyz=vec(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0)),
                rpy=vec(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0)),
                axis=vec(axis.attrib.get("xyz") if axis is not None else None, (1, 0, 0)),
                lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
                upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            )
        )
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"URDF must have one root link, found {roots}")
    return roots[0], joints


def forward_kinematics(
    root_link: str, joints: list[UrdfJoint], positions: dict[str, float]
) -> tuple[dict[str, np.ndarray], list[tuple[str, str, str]]]:
    pending = list(joints)
    poses = {root_link: np.eye(4, dtype=np.float64)}
    edges: list[tuple[str, str, str]] = []
    while pending:
        progressed = False
        for joint in pending[:]:
            if joint.parent not in poses:
                continue
            joint_origin = transform(rpy_rotation(joint.rpy), joint.xyz)
            value = float(positions.get(joint.name, 0.0))
            if joint.kind in {"revolute", "continuous"}:
                motion = transform(rotation=axis_rotation(joint.axis, value))
            elif joint.kind == "prismatic":
                motion = transform(translation=joint.axis * value)
            else:
                motion = np.eye(4, dtype=np.float64)
            poses[joint.child] = poses[joint.parent] @ joint_origin @ motion
            edges.append((joint.parent, joint.child, joint.name))
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError(f"could not resolve URDF joints: {[joint.name for joint in pending]}")
    return poses, edges


def clamp_to_limits(values: np.ndarray, joints: list[UrdfJoint]) -> tuple[np.ndarray, list[dict[str, float]]]:
    limits = {joint.name: (joint.lower, joint.upper) for joint in joints}
    clipped = values.copy()
    violations: list[dict[str, float]] = []
    for column, name in enumerate(ACTION_JOINTS):
        lower, upper = limits.get(name, (None, None))
        for row, value in enumerate(values[:, column]):
            bounded = value
            if lower is not None:
                bounded = max(bounded, lower)
            if upper is not None:
                bounded = min(bounded, upper)
            if bounded != value:
                violations.append(
                    {"step": row, "joint": name, "value": float(value), "clipped": float(bounded)}
                )
                clipped[row, column] = bounded
    return clipped, violations


def joint_positions(action: np.ndarray) -> dict[str, float]:
    values = np.concatenate((action[32:39], action[39:46], action[46:49]))
    # The dataset's Dex1 channels are open/close actuator coordinates, not the
    # URDF finger-prismatic displacement in metres. Render only the body motor
    # targets for which the action-to-URDF mapping is exact.
    return {name: float(value) for name, value in zip(ACTION_JOINTS, values)}


def project(point: np.ndarray, view: str, origin: tuple[int, int], scale: float) -> tuple[int, int]:
    if view == "front":
        horizontal, vertical = -point[1], point[2]
    else:
        horizontal, vertical = point[0], point[2]
    return int(origin[0] + scale * horizontal), int(origin[1] - scale * vertical)


def draw_robot(
    canvas: np.ndarray,
    poses: dict[str, np.ndarray],
    edges: list[tuple[str, str, str]],
    *,
    view: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    upper = set(ACTION_JOINTS)
    for parent, child, joint_name in edges:
        if joint_name not in upper and not any(token in child for token in ("pelvis", "torso", "head")):
            continue
        p0 = project(poses[parent][:3, 3], view, origin, 360.0)
        p1 = project(poses[child][:3, 3], view, origin, 360.0)
        edge_color = color if joint_name in upper else (130, 130, 130)
        cv2.line(canvas, p0, p1, edge_color, thickness, cv2.LINE_AA)
        if joint_name in upper:
            cv2.circle(canvas, p1, 3, edge_color, -1, cv2.LINE_AA)


def make_motion_video(
    output: Path,
    images: tuple[np.ndarray, np.ndarray, np.ndarray],
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    root_link: str,
    urdf_joints: list[UrdfJoint],
    inference_seconds: float,
) -> tuple[Path, Path, Path]:
    width, height = 1200, 760
    top_height = 250
    camera_width = width // 3
    resized = [cv2.resize(image[..., ::-1], (camera_width, top_height)) for image in images]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")
    first_frame = None
    last_frame = None
    for step in range(len(predicted)):
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        for camera, image in enumerate(resized):
            canvas[:top_height, camera * camera_width : (camera + 1) * camera_width] = image
        for camera, name in enumerate(("head_left", "left_wrist", "right_wrist")):
            cv2.putText(canvas, name, (camera * camera_width + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        pred_poses, edges = forward_kinematics(root_link, urdf_joints, joint_positions(predicted[step]))
        gt_poses, _ = forward_kinematics(root_link, urdf_joints, joint_positions(ground_truth[step]))
        for view, origin in (("front", (300, 720)), ("side", (900, 720))):
            draw_robot(canvas, gt_poses, edges, view=view, origin=origin, color=(40, 170, 40), thickness=5)
            draw_robot(canvas, pred_poses, edges, view=view, origin=origin, color=(40, 40, 220), thickness=3)
            cv2.putText(canvas, view, (origin[0] - 35, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "red: GR00T prediction   green: dataset target", (25, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"chunk step {step + 1:02d}/{len(predicted)}  t={step / 30.0:.3f}s  inference={inference_seconds:.2f}s", (25, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, "17 mapped body joints: arms + waist (hands excluded)", (25, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
        for _ in range(8):
            writer.write(canvas)
        if first_frame is None:
            first_frame = canvas.copy()
        last_frame = canvas.copy()
    writer.release()
    first_path = output.with_name(output.stem + "_first.png")
    last_path = output.with_name(output.stem + "_last.png")
    cv2.imwrite(str(first_path), first_frame)
    cv2.imwrite(str(last_path), last_frame)
    return output, first_path, last_path


def make_joint_plot(output: Path, predicted: np.ndarray, ground_truth: np.ndarray) -> Path:
    rows, cols = 6, 3
    cell_w, cell_h = 460, 170
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for index, name in enumerate(ACTION_JOINTS):
        row, col = divmod(index, cols)
        x0, y0 = col * cell_w + 55, row * cell_h + 35
        plot_w, plot_h = cell_w - 80, cell_h - 60
        values = np.concatenate((predicted[:, index], ground_truth[:, index]))
        low, high = float(values.min()), float(values.max())
        margin = max(0.03, (high - low) * 0.12)
        low, high = low - margin, high + margin
        cv2.rectangle(canvas, (x0, y0), (x0 + plot_w, y0 + plot_h), (180, 180, 180), 1)
        cv2.putText(canvas, name.replace("_joint", ""), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
        for sequence, color in ((ground_truth[:, index], (40, 170, 40)), (predicted[:, index], (40, 40, 220))):
            points = []
            for step, value in enumerate(sequence):
                x = int(x0 + plot_w * step / max(1, len(sequence) - 1))
                y = int(y0 + plot_h * (high - float(value)) / max(1e-9, high - low))
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"[{low:.2f}, {high:.2f}] rad", (x0 + plot_w - 115, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red: GR00T prediction   green: dataset target", (930, rows * cell_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)
    return output


def group_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for name, group_slice in ACTION_GROUPS.items():
        error = predicted[:, group_slice] - ground_truth[:, group_slice]
        result[name] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "max_abs": float(np.abs(error).max()),
        }
    return result


def wrist_fk_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    root_link: str,
    urdf_joints: list[UrdfJoint],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        predicted_positions = []
        target_positions = []
        link_name = f"{side}_wrist_yaw_link"
        for predicted_action, target_action in zip(predicted, ground_truth):
            predicted_poses, _ = forward_kinematics(
                root_link, urdf_joints, joint_positions(predicted_action)
            )
            target_poses, _ = forward_kinematics(
                root_link, urdf_joints, joint_positions(target_action)
            )
            predicted_positions.append(predicted_poses[link_name][:3, 3])
            target_positions.append(target_poses[link_name][:3, 3])
        predicted_array = np.asarray(predicted_positions)
        target_array = np.asarray(target_positions)
        error = np.linalg.norm(predicted_array - target_array, axis=1)
        result[f"{side}_wrist"] = {
            "mean_position_error_m": float(error.mean()),
            "final_position_error_m": float(error[-1]),
            "max_position_error_m": float(error.max()),
            "predicted_displacement_m": float(
                np.linalg.norm(predicted_array[-1] - predicted_array[0])
            ),
            "target_displacement_m": float(np.linalg.norm(target_array[-1] - target_array[0])),
        }
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.frame_ratio <= 1:
        raise ValueError("--frame-ratio must be in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_overlay.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("stage: importing LeRobot and coarse-insert GR00T runtime", flush=True)
    import torch
    from inference.coarse_insert import CoarseInsertGrootRuntime, validate_checkpoint
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(42)
    np.random.seed(42)
    validate_checkpoint(args.checkpoint)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    test_episodes = [int(value) for value in split["splits"]["test"]["episode_indices"]]
    episode_index = args.episode_index if args.episode_index is not None else test_episodes[0]
    if episode_index not in test_episodes:
        raise ValueError(f"episode {episode_index} is not in the held-out test split")

    print(f"stage: loading held-out episode {episode_index}", flush=True)
    dataset = LeRobotDataset(
        "Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_1",
        root=args.dataset_root,
        episodes=[episode_index],
        video_backend="torchcodec",
        return_uint8=True,
    )
    sample_index = min(len(dataset) - 1, round((len(dataset) - 1) * args.frame_ratio))
    sample = dataset[sample_index]
    tasks_table = pq.read_table(args.dataset_root / "meta/tasks.parquet")
    task_text_columns = [name for name in tasks_table.column_names if name != "task_index"]
    if len(task_text_columns) != 1:
        raise ValueError(f"expected one task text column, found {task_text_columns}")
    task_text_column = task_text_columns[0]
    task_rows = tasks_table.to_pylist()
    task_by_index = {
        int(row["task_index"]): str(row[task_text_column]) for row in task_rows
    }
    task = task_by_index[int(sample["task_index"])]
    request = {
        "state": sample["observation.state"].numpy(),
        "head_left": sample["observation.images.head_left"].permute(1, 2, 0).numpy(),
        "left_wrist": sample["observation.images.left_wrist"].permute(1, 2, 0).numpy(),
        "right_wrist": sample["observation.images.right_wrist"].permute(1, 2, 0).numpy(),
        "task": np.asarray([task]),
    }

    print(f"stage: loading checkpoint on {args.device}", flush=True)
    # Transformers may tie Qwen's token embedding to another parameter at load
    # time. The training checkpoint contains the saved alias as an extra key;
    # non-strict loading retains all model parameters while accepting that alias.
    runtime = CoarseInsertGrootRuntime(
        args.checkpoint,
        args.device,
        args.n_action_steps,
        strict_checkpoint_loading=False,
    )
    print("stage: predicting decoded action chunk", flush=True)
    prediction = runtime.predict_canonical(
        state=request["state"],
        head_left=request["head_left"],
        left_wrist=request["left_wrist"],
        right_wrist=request["right_wrist"],
        task=task,
    )
    decoded = prediction.canonical_action
    normalized = prediction.normalized_action
    inference_seconds = float(prediction.inference_seconds or 0.0)
    # Read target actions from the raw parquet-backed table. Calling
    # ``dataset[...]`` here would unnecessarily decode all three videos again
    # for every future step and can fail on harmless sub-millisecond video
    # timestamp rounding even though action labels are available.
    ground_truth = np.stack(
        [
            np.asarray(
                dataset.get_raw_item(min(sample_index + step, len(dataset) - 1))["action"],
                dtype=np.float32,
            )
            for step in range(args.n_action_steps)
        ]
    )
    if not np.isfinite(decoded).all():
        raise RuntimeError("prediction contains non-finite values")

    stem = f"test_ep{episode_index:04d}_frame{int(sample['frame_index']):06d}"
    archive_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(
        archive_path,
        decoded_action=decoded,
        normalized_action=normalized,
        dex1_open_close_action=prediction.dex1_open_close_targets,
        sdk_motor_ids=np.asarray(prediction.sdk_motor_ids, dtype=np.int64),
        sdk_position_targets=prediction.sdk_position_targets,
        ground_truth_action=ground_truth,
        state=request["state"],
        head_left=request["head_left"],
        left_wrist=request["left_wrist"],
        right_wrist=request["right_wrist"],
        task=np.asarray(task),
        episode_index=np.asarray(episode_index),
        frame_index=np.asarray(int(sample["frame_index"])),
        timestamp=np.asarray(float(sample["timestamp"])),
        inference_seconds=np.asarray(inference_seconds),
    )

    root_link, urdf_joints = load_urdf(args.urdf)
    predicted_joints = np.concatenate((decoded[:, 32:39], decoded[:, 39:46], decoded[:, 46:49]), axis=1)
    truth_joints = np.concatenate(
        (ground_truth[:, 32:39], ground_truth[:, 39:46], ground_truth[:, 46:49]), axis=1
    )
    clipped_predicted, violations = clamp_to_limits(predicted_joints, urdf_joints)
    decoded_for_render = decoded.copy()
    decoded_for_render[:, 32:49] = clipped_predicted

    print("stage: rendering URDF forward-kinematics motion", flush=True)
    video_path, first_path, last_path = make_motion_video(
        args.output_dir / f"{stem}_urdf_motion.mp4",
        (request["head_left"], request["left_wrist"], request["right_wrist"]),
        decoded_for_render,
        ground_truth,
        root_link,
        urdf_joints,
        inference_seconds,
    )
    plot_path = make_joint_plot(args.output_dir / f"{stem}_joint_targets.png", predicted_joints, truth_joints)
    joint_error = predicted_joints - truth_joints
    hand_metrics = {}
    from gr00t.g1_full_body_mapping import hand_to_dex1

    for side, index in (("left", 18), ("right", 25)):
        predicted_hand = np.asarray(
            [hand_to_dex1(value, side=side, kind="action") for value in decoded[:, index : index + 7]]
        )
        target_hand = np.asarray(
            [
                hand_to_dex1(value, side=side, kind="action")
                for value in ground_truth[:, index : index + 7]
            ]
        )
        error = predicted_hand - target_hand
        hand_metrics[side] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "predicted_first": float(predicted_hand[0]),
            "predicted_last": float(predicted_hand[-1]),
            "target_first": float(target_hand[0]),
            "target_last": float(target_hand[-1]),
            "predicted_synergy_residual_note": "decoded from all seven hand joints",
        }
    per_joint = {
        name: {
            "mae_rad": float(np.abs(joint_error[:, index]).mean()),
            "rmse_rad": float(np.sqrt(np.square(joint_error[:, index]).mean())),
            "max_abs_rad": float(np.abs(joint_error[:, index]).max()),
        }
        for index, name in enumerate(ACTION_JOINTS)
    }
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": "test",
        "episode_index": episode_index,
        "episode_frames": len(dataset),
        "sample_index": sample_index,
        "frame_index": int(sample["frame_index"]),
        "timestamp_s": float(sample["timestamp"]),
        "task": task,
        "n_action_steps": args.n_action_steps,
        "inference_device": args.device,
        "inference_seconds": inference_seconds,
        "checkpoint_has_dex1_loss_mask": runtime.has_dex1_loss_mask,
        "runtime_urdf_clipped_joint_values": prediction.clipped_joint_values,
        "all_finite": bool(np.isfinite(decoded).all()),
        "group_metrics": group_metrics(decoded, ground_truth),
        "per_joint_metrics": per_joint,
        "joint_overall_rmse_rad": float(np.sqrt(np.square(joint_error).mean())),
        "joint_overall_mae_rad": float(np.abs(joint_error).mean()),
        "joint_overall_rmse_deg": float(np.degrees(np.sqrt(np.square(joint_error).mean()))),
        "joint_overall_mae_deg": float(np.degrees(np.abs(joint_error).mean())),
        "rendered_body_joints": [
            {"joint_name": name, "sdk_id": sdk_id}
            for name, sdk_id in zip(ACTION_JOINTS, ACTION_JOINT_SDK_IDS)
        ],
        "wrist_fk_metrics": wrist_fk_metrics(
            decoded_for_render, ground_truth, root_link, urdf_joints
        ),
        "dex1_open_close_metrics": hand_metrics,
        "materialized_base_height_target_all_zero": bool(
            np.allclose(ground_truth[:, 49], 0.0)
        ),
        "predicted_base_height_range": [
            float(decoded[:, 49].min()),
            float(decoded[:, 49].max()),
        ],
        "predicted_max_target_velocity_rad_s": float(np.abs(np.diff(predicted_joints, axis=0) * 30).max()),
        "ground_truth_max_target_velocity_rad_s": float(np.abs(np.diff(truth_joints, axis=0) * 30).max()),
        "joint_limit_violation_count": len(violations),
        "joint_limit_violations": violations,
        "artifacts": {
            "archive": str(archive_path),
            "video": str(video_path),
            "first_frame": str(first_path),
            "last_frame": str(last_path),
            "joint_plot": str(plot_path),
        },
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
