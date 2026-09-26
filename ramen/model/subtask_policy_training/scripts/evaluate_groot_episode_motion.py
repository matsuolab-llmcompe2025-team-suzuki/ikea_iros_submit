#!/usr/bin/env python3
"""Evaluate GR00T over one complete held-out coarse-insert episode.

The policy observes the three camera images and canonical state once per
16-step action chunk.  The chunks are concatenated to cover the episode.  This
is teacher-forced offline evaluation: every new chunk starts from the recorded
dataset observation, rather than from a simulated rollout of the prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from evaluate_groot_joint_motion import (
    ACTION_JOINTS,
    ACTION_JOINT_SDK_IDS,
    clamp_to_limits,
    draw_robot,
    forward_kinematics,
    joint_positions,
    load_urdf,
    wrist_fk_metrics,
)


CAMERAS = ("head_left", "left_wrist", "right_wrist")
CAMERA_FEATURES = tuple(f"observation.images.{name}" for name in CAMERAS)
BODY_ACTION_COLUMNS = tuple(range(32, 49))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    feature_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=feature_root
        / "outputs/train/groot_coarse_insert_2gpu_batch8_100k_dex1_v2"
        / "checkpoints/100000/pretrained_model",
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
        / "outputs/training_views/groot_coarse_insert_sdk_ids_v2/meta"
        / "team_ramen_episode_split.json",
    )
    parser.add_argument("--episode-index", type=int, default=28)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=feature_root
        / "outputs/evaluations/groot_coarse_insert_100k_dex1_v2/full_episode",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=repo_root
        / "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1"
        / "g1_29dof_mode_15_with_dex1_1.urdf",
    )
    parser.add_argument(
        "--source-overlay",
        type=Path,
        default=feature_root / "outputs/lerobot_source_overlays/groot_relative_eef_v3",
    )
    return parser.parse_args()


def read_episode_metadata(dataset_root: Path, episode_index: int) -> dict:
    for path in sorted((dataset_root / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if int(row["episode_index"]) == episode_index:
                return row
    raise KeyError(f"episode {episode_index} is absent from metadata")


def read_task(dataset_root: Path, task_index: int) -> str:
    table = pq.read_table(dataset_root / "meta/tasks.parquet")
    text_columns = [name for name in table.column_names if name != "task_index"]
    if len(text_columns) != 1:
        raise ValueError(f"expected one task text column, found {text_columns}")
    text_column = text_columns[0]
    for row in table.to_pylist():
        if int(row["task_index"]) == task_index:
            return str(row[text_column])
    raise KeyError(f"task_index {task_index} is absent from tasks metadata")


class EpisodeVideoReader:
    """Read monotonically increasing local frames from a LeRobot video slice."""

    def __init__(
        self,
        dataset_root: Path,
        episode_metadata: dict,
        camera_feature: str,
        fps: float,
    ) -> None:
        prefix = f"videos/{camera_feature}"
        chunk = int(episode_metadata[f"{prefix}/chunk_index"])
        file_index = int(episode_metadata[f"{prefix}/file_index"])
        self.path = (
            dataset_root
            / "videos"
            / camera_feature
            / f"chunk-{chunk:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open dataset video {self.path}")
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isclose(actual_fps, fps, rel_tol=0, abs_tol=1e-3):
            raise ValueError(f"{self.path} has fps={actual_fps}, expected {fps}")
        start_seconds = float(episode_metadata[f"{prefix}/from_timestamp"])
        self.start_frame = round(start_seconds * fps)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.next_local_index = 0
        self.last_bgr: np.ndarray | None = None

    def read(self, local_index: int) -> np.ndarray:
        if local_index < self.next_local_index - 1:
            raise ValueError("EpisodeVideoReader only supports increasing frame indices")
        while self.next_local_index <= local_index:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(
                    f"video decode failed at local frame {self.next_local_index}: {self.path}"
                )
            self.last_bgr = frame
            self.next_local_index += 1
        assert self.last_bgr is not None
        return self.last_bgr.copy()

    def close(self) -> None:
        self.capture.release()


def encode_inputs(images_bgr: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    encoded = []
    for image in images_bgr:
        ok, jpeg = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 88))
        if not ok:
            raise RuntimeError("could not JPEG-encode model input")
        encoded.append(jpeg)
    return tuple(encoded)


def decode_inputs(images_jpeg: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    images = tuple(cv2.imdecode(value, cv2.IMREAD_COLOR) for value in images_jpeg)
    if any(value is None for value in images):
        raise RuntimeError("could not JPEG-decode stored model input")
    return images


def body_joints(actions: np.ndarray) -> np.ndarray:
    return actions[:, BODY_ACTION_COLUMNS]


def draw_dex1_commands(
    canvas: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
) -> None:
    """Draw physical Dex1 commands without inventing a finger-URDF mapping."""

    panel_left, panel_top = 430, 410
    panel_right, panel_bottom = 790, 585
    cv2.rectangle(
        canvas,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (205, 205, 205),
        -1,
    )
    cv2.rectangle(
        canvas,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (150, 150, 150),
        1,
    )
    cv2.putText(
        canvas,
        "Dex1 command: 0=open, 4.5=closed",
        (panel_left + 18, panel_top + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    bar_left, bar_right = panel_left + 72, panel_right - 28
    for side_index, (side, y) in enumerate((("left", 475), ("right", 545))):
        cv2.putText(
            canvas,
            side,
            (panel_left + 18, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (bar_left, y), (bar_right, y), (90, 90, 90), 3, cv2.LINE_AA)
        for tick_value in (0.0, 2.25, 4.5):
            x = round(bar_left + (bar_right - bar_left) * tick_value / 4.5)
            cv2.line(canvas, (x, y - 6), (x, y + 6), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "0",
            (bar_left - 5, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "4.5",
            (bar_right - 12, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        predicted_value = float(np.clip(predicted[side_index], 0.0, 4.5))
        target_value = float(np.clip(target[side_index], 0.0, 4.5))
        predicted_x = round(
            bar_left + (bar_right - bar_left) * predicted_value / 4.5
        )
        target_x = round(bar_left + (bar_right - bar_left) * target_value / 4.5)
        cv2.circle(canvas, (target_x, y + 7), 6, (40, 170, 40), -1, cv2.LINE_AA)
        cv2.circle(canvas, (predicted_x, y - 7), 6, (40, 40, 220), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"P {predicted_value:.2f}  T {target_value:.2f}",
            (bar_left, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )


def make_full_episode_video(
    output: Path,
    *,
    chunk_inputs: list[tuple[np.ndarray, ...]],
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    predicted_dex1: np.ndarray,
    target_dex1: np.ndarray,
    chunk_size: int,
    fps: float,
    episode_index: int,
    root_link: str,
    urdf_joints: list,
    inference_seconds: np.ndarray,
) -> tuple[Path, Path, Path]:
    width, height = 1200, 760
    top_height = 250
    camera_width = width // 3
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    first_frame = None
    middle_frame = None
    last_frame = None
    cached_chunk = -1
    resized: list[np.ndarray] = []
    for frame_index in range(len(predicted)):
        chunk_index = frame_index // chunk_size
        chunk_step = frame_index % chunk_size
        if chunk_index != cached_chunk:
            images = decode_inputs(chunk_inputs[chunk_index])
            resized = [
                cv2.resize(image, (camera_width, top_height), interpolation=cv2.INTER_AREA)
                for image in images
            ]
            cached_chunk = chunk_index

        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        for camera, image in enumerate(resized):
            left = camera * camera_width
            canvas[:top_height, left : left + camera_width] = image
            cv2.putText(
                canvas,
                CAMERAS[camera],
                (left + 12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        pred_poses, edges = forward_kinematics(
            root_link, urdf_joints, joint_positions(predicted[frame_index])
        )
        target_poses, _ = forward_kinematics(
            root_link, urdf_joints, joint_positions(ground_truth[frame_index])
        )
        for view, origin in (("front", (300, 720)), ("side", (900, 720))):
            draw_robot(
                canvas,
                target_poses,
                edges,
                view=view,
                origin=origin,
                color=(40, 170, 40),
                thickness=5,
            )
            draw_robot(
                canvas,
                pred_poses,
                edges,
                view=view,
                origin=origin,
                color=(40, 40, 220),
                thickness=3,
            )
            cv2.putText(
                canvas,
                view,
                (origin[0] - 35, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 40, 40),
                2,
                cv2.LINE_AA,
            )

        current_error_deg = float(
            np.degrees(
                np.sqrt(
                    np.square(
                        body_joints(predicted[frame_index : frame_index + 1])
                        - body_joints(ground_truth[frame_index : frame_index + 1])
                    ).mean()
                )
            )
        )
        cv2.putText(
            canvas,
            "red: GR00T prediction   green: dataset target",
            (25, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"test episode {episode_index}  frame {frame_index + 1:04d}/{len(predicted)}"
                f"  t={frame_index / fps:6.2f}s"
            ),
            (25, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"teacher-forced chunk {chunk_index + 1:03d}/{len(chunk_inputs)}"
                f"  step {chunk_step + 1:02d}/{min(chunk_size, len(predicted) - chunk_index * chunk_size)}"
                f"  inference={inference_seconds[chunk_index]:.2f}s"
                f"  joint RMSE={current_error_deg:.2f}deg"
            ),
            (25, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "top: images observed at the beginning of the current 16-step chunk",
            (25, 378),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        draw_dex1_commands(
            canvas,
            predicted_dex1[frame_index],
            target_dex1[frame_index],
        )
        if chunk_step == 0:
            cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), (220, 160, 20), 5)
            cv2.putText(
                canvas,
                "NEW OBSERVATION / CHUNK BOUNDARY",
                (790, 378),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 120, 20),
                2,
                cv2.LINE_AA,
            )

        writer.write(canvas)
        if first_frame is None:
            first_frame = canvas.copy()
        if frame_index == len(predicted) // 2:
            middle_frame = canvas.copy()
        last_frame = canvas.copy()

    writer.release()
    assert first_frame is not None and middle_frame is not None and last_frame is not None
    first_path = output.with_name(output.stem + "_first.png")
    middle_path = output.with_name(output.stem + "_middle.png")
    last_path = output.with_name(output.stem + "_last.png")
    cv2.imwrite(str(first_path), first_frame)
    cv2.imwrite(str(middle_path), middle_frame)
    cv2.imwrite(str(last_path), last_frame)
    return first_path, middle_path, last_path


def make_dex1_plot(
    output: Path,
    predicted: np.ndarray,
    target: np.ndarray,
    fps: float,
) -> Path:
    width, height = 1400, 620
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    duration = (len(predicted) - 1) / fps
    for side_index, side in enumerate(("left", "right")):
        x0, y0 = 90, 90 + side_index * 255
        plot_w, plot_h = width - 150, 190
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + plot_w, y0 + plot_h),
            (170, 170, 170),
            1,
        )
        for value in (0.0, 2.25, 4.5):
            y = round(y0 + plot_h * (4.5 - value) / 4.5)
            cv2.line(
                canvas,
                (x0, y),
                (x0 + plot_w, y),
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{value:.2f}",
                (x0 - 55, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )
        for sequence, color in (
            (target[:, side_index], (40, 170, 40)),
            (predicted[:, side_index], (40, 40, 220)),
        ):
            points = []
            for frame_index, value in enumerate(sequence):
                x = round(x0 + plot_w * frame_index / max(1, len(sequence) - 1))
                y = round(y0 + plot_h * (4.5 - float(np.clip(value, 0.0, 4.5))) / 4.5)
                points.append((x, y))
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        mae = float(np.abs(predicted[:, side_index] - target[:, side_index]).mean())
        cv2.putText(
            canvas,
            f"{side} Dex1 open/close command   MAE={mae:.3f}",
            (x0, y0 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"0=open   4.5=closed   0-{duration:.1f}s",
            (x0 + plot_w - 300, y0 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "red: GR00T prediction   green: dataset target",
        (width - 420, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), canvas)
    return output


def make_full_episode_joint_plot(
    output: Path,
    predicted_joints: np.ndarray,
    target_joints: np.ndarray,
    fps: float,
) -> Path:
    rows, cols = 6, 3
    cell_w, cell_h = 500, 180
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    duration = (len(predicted_joints) - 1) / fps
    for index, name in enumerate(ACTION_JOINTS):
        row, col = divmod(index, cols)
        x0, y0 = col * cell_w + 58, row * cell_h + 38
        plot_w, plot_h = cell_w - 90, cell_h - 66
        values = np.concatenate((predicted_joints[:, index], target_joints[:, index]))
        low, high = float(values.min()), float(values.max())
        margin = max(0.03, (high - low) * 0.08)
        low, high = low - margin, high + margin
        cv2.rectangle(canvas, (x0, y0), (x0 + plot_w, y0 + plot_h), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            name.replace("_joint", ""),
            (x0, y0 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        for sequence, color in (
            (target_joints[:, index], (40, 170, 40)),
            (predicted_joints[:, index], (40, 40, 220)),
        ):
            points = []
            for step, value in enumerate(sequence):
                x = int(x0 + plot_w * step / max(1, len(sequence) - 1))
                y = int(y0 + plot_h * (high - float(value)) / max(1e-9, high - low))
                points.append((x, y))
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"0-{duration:.1f}s  [{low:.2f},{high:.2f}]rad",
            (x0 + plot_w - 170, y0 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "red: GR00T prediction   green: dataset target",
        (1040, rows * cell_h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), canvas)
    return output


def velocity_metrics(joints: np.ndarray, fps: float, chunk_size: int) -> dict[str, float]:
    speeds = np.abs(np.diff(joints, axis=0) * fps)
    boundary_rows = np.asarray(
        [index for index in range(len(speeds)) if (index + 1) % chunk_size == 0],
        dtype=np.int64,
    )
    within_mask = np.ones(len(speeds), dtype=bool)
    within_mask[boundary_rows] = False
    return {
        "max_all_rad_s": float(speeds.max()),
        "p99_all_rad_s": float(np.percentile(speeds, 99)),
        "max_within_chunk_rad_s": float(speeds[within_mask].max()),
        "max_chunk_boundary_rad_s": float(speeds[boundary_rows].max()),
        "p95_chunk_boundary_rad_s": float(np.percentile(speeds[boundary_rows], 95)),
    }


def main() -> None:
    args = parse_args()
    if args.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_overlay.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("stage: importing LeRobot and coarse-insert GR00T runtime", flush=True)
    import torch
    from gr00t.g1_full_body_mapping import hand_to_dex1
    from inference.coarse_insert import CoarseInsertGrootRuntime, validate_checkpoint
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(42)
    np.random.seed(42)
    validate_checkpoint(args.checkpoint)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    test_episodes = set(map(int, split["splits"]["test"]["episode_indices"]))
    if args.episode_index not in test_episodes:
        raise ValueError(f"episode {args.episode_index} is not in the held-out test split")

    metadata = read_episode_metadata(args.dataset_root, args.episode_index)
    print(
        f"stage: loading held-out episode {args.episode_index} "
        f"({metadata['length']} frames)",
        flush=True,
    )
    dataset = LeRobotDataset(
        "Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_1",
        root=args.dataset_root,
        episodes=[args.episode_index],
        video_backend="torchcodec",
        return_uint8=True,
    )
    episode_frames = len(dataset)
    if episode_frames != int(metadata["length"]):
        raise ValueError(
            f"dataset has {episode_frames} frames, metadata says {metadata['length']}"
        )
    raw_rows = [dataset.get_raw_item(index) for index in range(episode_frames)]
    states = np.stack(
        [np.asarray(row["observation.state"], dtype=np.float32) for row in raw_rows]
    )
    ground_truth = np.stack(
        [np.asarray(row["action"], dtype=np.float32) for row in raw_rows]
    )
    task_indices = {int(row["task_index"]) for row in raw_rows}
    if len(task_indices) != 1:
        raise ValueError(f"episode has multiple task indices: {sorted(task_indices)}")
    task = read_task(args.dataset_root, task_indices.pop())

    readers = tuple(
        EpisodeVideoReader(args.dataset_root, metadata, feature, args.fps)
        for feature in CAMERA_FEATURES
    )
    chunk_starts = list(range(0, episode_frames, args.n_action_steps))
    print(
        f"stage: loading checkpoint on {args.device}; "
        f"running {len(chunk_starts)} action chunks",
        flush=True,
    )
    runtime = CoarseInsertGrootRuntime(
        args.checkpoint,
        args.device,
        args.n_action_steps,
        strict_checkpoint_loading=False,
    )

    predicted = np.empty_like(ground_truth)
    normalized = np.empty_like(ground_truth)
    chunk_inputs: list[tuple[np.ndarray, ...]] = []
    inference_seconds: list[float] = []
    runtime_clip_count = 0
    try:
        for chunk_index, start in enumerate(chunk_starts):
            images_bgr = tuple(reader.read(start) for reader in readers)
            images_rgb = tuple(np.ascontiguousarray(image[..., ::-1]) for image in images_bgr)
            result = runtime.predict_canonical(
                state=states[start],
                head_left=images_rgb[0],
                left_wrist=images_rgb[1],
                right_wrist=images_rgb[2],
                task=task,
            )
            end = min(start + args.n_action_steps, episode_frames)
            count = end - start
            predicted[start:end] = result.canonical_action[:count]
            normalized[start:end] = result.normalized_action[:count]
            chunk_inputs.append(encode_inputs(images_bgr))
            inference_seconds.append(float(result.inference_seconds or 0.0))
            runtime_clip_count += int(result.clipped_joint_values)
            if (
                chunk_index == 0
                or (chunk_index + 1) % 10 == 0
                or chunk_index + 1 == len(chunk_starts)
            ):
                print(
                    f"progress: chunk {chunk_index + 1}/{len(chunk_starts)} "
                    f"frames {start}:{end} inference={inference_seconds[-1]:.3f}s",
                    flush=True,
                )
    finally:
        for reader in readers:
            reader.close()

    if not np.isfinite(predicted).all():
        raise RuntimeError("prediction contains non-finite values")

    root_link, urdf_joints = load_urdf(args.urdf)
    predicted_joints = body_joints(predicted)
    target_joints = body_joints(ground_truth)
    clipped_predicted_joints, violations = clamp_to_limits(predicted_joints, urdf_joints)
    rendered_prediction = predicted.copy()
    rendered_prediction[:, BODY_ACTION_COLUMNS] = clipped_predicted_joints
    predicted_dex1 = np.column_stack(
        (
            [hand_to_dex1(value, side="left", kind="action") for value in predicted[:, 18:25]],
            [hand_to_dex1(value, side="right", kind="action") for value in predicted[:, 25:32]],
        )
    )
    target_dex1 = np.column_stack(
        (
            [
                hand_to_dex1(value, side="left", kind="action")
                for value in ground_truth[:, 18:25]
            ],
            [
                hand_to_dex1(value, side="right", kind="action")
                for value in ground_truth[:, 25:32]
            ],
        )
    )

    stem = f"test_ep{args.episode_index:04d}_full_{episode_frames:06d}frames"
    archive_path = args.output_dir / f"{stem}.npz"
    np.savez_compressed(
        archive_path,
        decoded_action=predicted,
        normalized_action=normalized,
        ground_truth_action=ground_truth,
        state=states,
        chunk_start_frames=np.asarray(chunk_starts, dtype=np.int64),
        inference_seconds=np.asarray(inference_seconds, dtype=np.float64),
        episode_index=np.asarray(args.episode_index),
        fps=np.asarray(args.fps),
        task=np.asarray(task),
    )

    print("stage: rendering full-episode video and joint plot", flush=True)
    video_path = args.output_dir / f"{stem}_urdf_dex1_motion.mp4"
    first_path, middle_path, last_path = make_full_episode_video(
        video_path,
        chunk_inputs=chunk_inputs,
        predicted=rendered_prediction,
        ground_truth=ground_truth,
        predicted_dex1=predicted_dex1,
        target_dex1=target_dex1,
        chunk_size=args.n_action_steps,
        fps=args.fps,
        episode_index=args.episode_index,
        root_link=root_link,
        urdf_joints=urdf_joints,
        inference_seconds=np.asarray(inference_seconds),
    )
    plot_path = make_full_episode_joint_plot(
        args.output_dir / f"{stem}_joint_targets.png",
        predicted_joints,
        target_joints,
        args.fps,
    )
    dex1_plot_path = make_dex1_plot(
        args.output_dir / f"{stem}_dex1_targets.png",
        predicted_dex1,
        target_dex1,
        args.fps,
    )

    joint_error = predicted_joints - target_joints
    per_joint = {
        name: {
            "mae_rad": float(np.abs(joint_error[:, index]).mean()),
            "rmse_rad": float(np.sqrt(np.square(joint_error[:, index]).mean())),
            "rmse_deg": float(
                np.degrees(np.sqrt(np.square(joint_error[:, index]).mean()))
            ),
            "max_abs_rad": float(np.abs(joint_error[:, index]).max()),
        }
        for index, name in enumerate(ACTION_JOINTS)
    }
    hand_metrics = {}
    for side_index, side in enumerate(("left", "right")):
        predicted_hand = predicted_dex1[:, side_index]
        target_hand = target_dex1[:, side_index]
        error = predicted_hand - target_hand
        hand_metrics[side] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "max_abs": float(np.abs(error).max()),
        }

    inference_array = np.asarray(inference_seconds)
    duration_seconds = episode_frames / args.fps
    summary = {
        "evaluation_type": "teacher_forced_non_overlapping_action_chunks",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": "test",
        "episode_index": args.episode_index,
        "source_episode_name": metadata.get("source_episode_name"),
        "episode_frames": episode_frames,
        "episode_duration_seconds": duration_seconds,
        "fps": args.fps,
        "task": task,
        "chunk_size": args.n_action_steps,
        "chunk_count": len(chunk_starts),
        "inference_device": args.device,
        "inference_seconds": {
            "mean": float(inference_array.mean()),
            "median": float(np.median(inference_array)),
            "p95": float(np.percentile(inference_array, 95)),
            "max": float(inference_array.max()),
            "total": float(inference_array.sum()),
            "offline_compute_to_episode_duration_ratio": float(
                inference_array.sum() / duration_seconds
            ),
        },
        "checkpoint_has_dex1_loss_mask": runtime.has_dex1_loss_mask,
        "runtime_urdf_clipped_joint_values": runtime_clip_count,
        "all_finite": bool(np.isfinite(predicted).all()),
        "joint_overall_rmse_rad": float(np.sqrt(np.square(joint_error).mean())),
        "joint_overall_mae_rad": float(np.abs(joint_error).mean()),
        "joint_overall_rmse_deg": float(
            np.degrees(np.sqrt(np.square(joint_error).mean()))
        ),
        "joint_overall_mae_deg": float(np.degrees(np.abs(joint_error).mean())),
        "per_joint_metrics": per_joint,
        "worst_joints_by_rmse": [
            {"joint": name, **values}
            for name, values in sorted(
                per_joint.items(), key=lambda item: item[1]["rmse_rad"], reverse=True
            )[:5]
        ],
        "wrist_fk_metrics": wrist_fk_metrics(
            rendered_prediction, ground_truth, root_link, urdf_joints
        ),
        "dex1_open_close_metrics": hand_metrics,
        "predicted_velocity": velocity_metrics(
            predicted_joints, args.fps, args.n_action_steps
        ),
        "ground_truth_velocity": velocity_metrics(
            target_joints, args.fps, args.n_action_steps
        ),
        "joint_limit_violation_count": len(violations),
        "joint_limit_violations_first_100": violations[:100],
        "rendered_body_joints": [
            {"joint_name": name, "sdk_id": sdk_id}
            for name, sdk_id in zip(ACTION_JOINTS, ACTION_JOINT_SDK_IDS)
        ],
        "artifacts": {
            "archive": str(archive_path),
            "video": str(video_path),
            "first_frame": str(first_path),
            "middle_frame": str(middle_path),
            "last_frame": str(last_path),
            "joint_plot": str(plot_path),
            "dex1_plot": str(dex1_plot_path),
        },
        "interpretation_note": (
            "Each chunk starts from the recorded dataset observation. "
            "This does not measure closed-loop rollout stability or insertion success."
        ),
    }
    summary_path = args.output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
