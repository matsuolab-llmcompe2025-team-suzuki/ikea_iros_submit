#!/usr/bin/env python3
"""Propose M3-M4 intervals and render three-camera review sheets.

Numeric hand signals only narrow the review queue.  The output deliberately
uses ``source=auto_candidate`` and cannot pass the expert training gate until
a reviewer changes accepted milestones to ``reviewed_phase_frames``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("cam_0", "cam_2", "cam_3")
REVIEW_OFFSETS = (-90, -60, -30, 0, 30)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--episode-index-min", type=int, default=0)
    parser.add_argument("--episode-index-max", type=int)
    return parser.parse_args()


def _load_numeric_episodes(root: Path) -> dict[int, dict[str, np.ndarray]]:
    episodes: dict[int, dict[str, list[np.ndarray]]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "observation.state.hand_state",
                "action.hand_cmd",
            ],
        )
        episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        hand_state = np.asarray(table["observation.state.hand_state"].to_pylist())
        hand_cmd = np.asarray(table["action.hand_cmd"].to_pylist())
        for value in np.unique(episode_index):
            selected = episode_index == value
            record = episodes.setdefault(
                int(value), {"frame": [], "state": [], "command": []}
            )
            record["frame"].append(frame_index[selected])
            record["state"].append(hand_state[selected])
            record["command"].append(hand_cmd[selected])
    merged: dict[int, dict[str, np.ndarray]] = {}
    for episode_index, pieces in episodes.items():
        frame = np.concatenate(pieces["frame"])
        order = np.argsort(frame)
        merged[episode_index] = {
            "frame": frame[order],
            "state": np.concatenate(pieces["state"])[order],
            "command": np.concatenate(pieces["command"])[order],
        }
    return merged


def propose_interval(
    hand_state: np.ndarray,
    hand_command: np.ndarray,
) -> tuple[int, int] | None:
    """Return a conservative M3/M4 proposal from persistent Dex1 transitions."""

    state = np.asarray(hand_state, dtype=np.float64)
    command = np.asarray(hand_command, dtype=np.float64)
    if state.shape != command.shape or state.ndim != 2 or state.shape[1] != 2:
        raise ValueError("hand state and command must be matching [T,2] arrays")
    if len(state) < 121 or not np.isfinite(state).all() or not np.isfinite(command).all():
        return None
    candidates = []
    in_candidate_window = False
    for m4 in range(60, len(state) - 30):
        left_support = np.median(command[m4 - 45 : m4 + 15, 0]) < 3.7
        right_previously_open = np.median(command[m4 - 45 : m4 - 10, 1]) > 4.1
        right_close_command = np.median(command[m4 : m4 + 15, 1]) < 3.7
        right_close_followed = np.median(state[m4 + 5 : m4 + 30, 1]) < 4.0
        qualifies = (
            left_support
            and right_previously_open
            and right_close_command
            and right_close_followed
        )
        if qualifies and not in_candidate_window:
            candidates.append((max(0, m4 - 90), m4))
        in_candidate_window = qualifies
    # The first right-hand close is the initial grasp.  The focused M3-M4
    # failure is the later re-grasp while the left hand already supports the
    # tilted table.  The end margin above excludes shutdown/release chatter.
    return candidates[-1] if candidates else None


def _video_frame(
    path: Path,
    *,
    timestamp_s: float,
    label: str,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        label = f"MISSING {label}"
    else:
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (320, 28), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _render_sheet(
    root: Path,
    manifest_episode: dict[str, Any],
    *,
    m4_frame: int,
    output: Path,
) -> None:
    rows = []
    for offset in REVIEW_OFFSETS:
        local_frame = int(np.clip(m4_frame + offset, 0, manifest_episode["length"] - 1))
        images = []
        for camera in CAMERAS:
            file_index = manifest_episode[f"videos/observation.images.{camera}/file_index"]
            chunk_index = manifest_episode[f"videos/observation.images.{camera}/chunk_index"]
            start = float(
                manifest_episode[f"videos/observation.images.{camera}/from_timestamp"]
            )
            timestamp = start + local_frame / 30.0
            path = (
                root
                / "videos"
                / f"observation.images.{camera}"
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            images.append(
                _video_frame(
                    path,
                    timestamp_s=timestamp,
                    label=f"{camera} f={local_frame} M4{offset:+d}",
                )
            )
        rows.append(np.hstack(images))
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> int:
    args = _arguments()
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "meta/curation/manifest.json").read_text())
    episodes = {int(item["episode_index"]): item for item in manifest["episodes"]}
    numeric = _load_numeric_episodes(root)

    proposals = []
    for episode_index, values in sorted(numeric.items()):
        if episode_index < args.episode_index_min or (
            args.episode_index_max is not None and episode_index > args.episode_index_max
        ):
            continue
        metadata = episodes.get(episode_index)
        if metadata is None or metadata.get("trajectory_type") not in {
            "full_success",
            "recovery_success",
        }:
            continue
        interval = propose_interval(values["state"], values["command"])
        if interval is None:
            continue
        m3, m4 = interval
        proposal = {
            "episode_index": episode_index,
            "length": int(metadata["length"]),
            "trajectory_type": metadata["trajectory_type"],
            "source_episode_index": int(metadata["source_episode_index"]),
            "milestones": {
                "M3": {"frame": m3, "valid": True, "source": "auto_candidate"},
                "M4": {"frame": m4, "valid": True, "source": "auto_candidate"},
            },
            "review_status": "pending",
        }
        proposals.append(proposal)
        if len(proposals) <= args.limit:
            _render_sheet(
                root,
                metadata,
                m4_frame=m4,
                output=output / f"episode_{episode_index:04d}_review.jpg",
            )

    document = {
        "schema_version": "flip_table_insertion_review_queue_v1",
        "dataset_revision": root.name,
        "camera_roles": {"cam_0": "head_left", "cam_2": "left_wrist", "cam_3": "right_wrist"},
        "automatic_labels_are_action_supervision": False,
        "proposal_count": len(proposals),
        "episode_index_range": [args.episode_index_min, args.episode_index_max],
        "proposals": proposals,
    }
    (output / "review_queue.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"proposal_count": len(proposals), "rendered": min(len(proposals), args.limit)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
