"""Exact real-G1 contract for the GR00T N1.7 pick-leg checkpoint.

The checkpoint was trained with the original BitRobot 38-D state/action view:
``root pose 7 + G1 body 29 + Dex1 2``.  The physical robot adapter deliberately
executes only the fourteen arm joints and two Dex1 values.  Unitree Regular
Mode remains the sole owner of the root, waist, and legs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODEL_REPO_ID = "Team-RAMEN/groot-n1.7-pick-legs-ver2-lora"
MODEL_REVISION = "8f875028fffb1b35ebaf6a95d1575ccec86c12a1"
DATASET_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1"
DATASET_REVISION = "4a5aef3bc43470b636f5c9d4f9102a30311b79a5"
TASK_TEXT = "pick table leg"
EMBODIMENT_TAG = "new_embodiment"
LEROBOT_VERSION = "0.6.0"

MODEL_ROOT_DIM = 7
MODEL_BODY_DIM = 29
MODEL_HAND_DIM = 2
MODEL_STATE_DIM = 38
MODEL_ACTION_DIM = 38
MODEL_ACTION_HORIZON = 16
EXECUTABLE_ACTION_DIM = 16
DEX1_DATASET_OPEN_VALUE = 4.5

# robot_q = root[7] + body[29]. Body order is the Unitree 29-DoF order:
# legs[0:12], waist[12:15], left arm[15:22], right arm[22:29].
MODEL_ARM_SLICE = slice(MODEL_ROOT_DIM + 15, MODEL_ROOT_DIM + 29)
MODEL_HAND_SLICE = slice(36, 38)
MODEL_IGNORED_ROOT_LOWER_SLICE = slice(0, MODEL_ROOT_DIM + 15)

CAMERA_ROLE_TO_KEY = {
    "head_left": "observation.images.cam_0",
    "head_right": "observation.images.cam_1",
    "left_wrist": "observation.images.cam_2",
    "right_wrist": "observation.images.cam_3",
}
CAMERA_KEYS = tuple(CAMERA_ROLE_TO_KEY.values())
CAMERA_ROLES = tuple(CAMERA_ROLE_TO_KEY)

# RealDdsBackend cannot observe global translation or base height.  The source
# dataset stores xyz+wxyz; use a stationary session-local root at the training
# standing height.  Root predictions are never sent to the robot.
REAL_ROOT_PROXY_XYZ_WXYZ = np.asarray(
    (0.0, 0.0, 0.70, 1.0, 0.0, 0.0, 0.0), dtype=np.float32
)

_MODEL_DATASET_PROVENANCE = {
    "Team-RAMEN/groot-n1.7-pick-legs-ver1": (
        DATASET_REPO_ID,
        DATASET_REVISION,
    ),
    MODEL_REPO_ID: (DATASET_REPO_ID, DATASET_REVISION),
    "Team-RAMEN/groot-n1.7-task4-optimal-ver1-lora": (
        "Team-RAMEN/IROS2026_RAMEN_HARA_task_4_optimal_merged_v1",
        "c6c1eb9ff4acf3308e066d16aa201e0b4bbd5154",
    ),
}


def _finite_vector(value: Any, width: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{width}], got {result.shape}")
    return result


def compose_model_state(
    body_joint_position_rad: Any,
    dex1_opening_fraction: Any,
    *,
    root_xyz_wxyz: Any = REAL_ROOT_PROXY_XYZ_WXYZ,
) -> np.ndarray:
    """Build the checkpoint's exact 38-D input without commanding the root."""

    root = _finite_vector(root_xyz_wxyz, MODEL_ROOT_DIM, "root xyz+wxyz")
    if not math.isclose(float(np.dot(root[3:], root[3:])), 1.0, abs_tol=1.0e-4):
        raise ValueError("root quaternion must be unit length")
    body = _finite_vector(body_joint_position_rad, MODEL_BODY_DIM, "G1 body state")
    hand = _finite_vector(
        dex1_opening_fraction, MODEL_HAND_DIM, "Dex1 opening fraction"
    )
    if np.any((hand < 0.0) | (hand > 1.0)):
        raise ValueError("Dex1 opening fraction must be in [0,1]")
    result = np.concatenate((root, body, hand * DEX1_DATASET_OPEN_VALUE))
    if result.shape != (MODEL_STATE_DIM,):
        raise RuntimeError("GR00T state assembly changed dimension")
    return result.astype(np.float32)


def extract_executable_action(action_chunk: Any) -> np.ndarray:
    """Drop root/legs/waist and return absolute arms14 + Dex1-dataset2."""

    values = np.asarray(action_chunk, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != MODEL_ACTION_DIM
        or values.shape[0] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "decoded GR00T action must be finite [T,38], "
            f"got {values.shape}"
        )
    arms = values[:, MODEL_ARM_SLICE]
    hands = values[:, MODEL_HAND_SLICE]
    result = np.concatenate((arms, hands), axis=1)
    if result.shape != (values.shape[0], EXECUTABLE_ACTION_DIM):
        raise RuntimeError("GR00T executable action assembly changed dimension")
    return result


def camera_payloads(camera_jpeg: Mapping[str, bytes]) -> dict[str, bytes]:
    missing = set(CAMERA_ROLES) - set(camera_jpeg)
    if missing:
        raise ValueError(f"GR00T observation is missing camera roles: {sorted(missing)}")
    result = {
        key: bytes(camera_jpeg[role])
        for role, key in CAMERA_ROLE_TO_KEY.items()
    }
    if any(not payload for payload in result.values()):
        raise ValueError("GR00T camera JPEG payloads must be non-empty")
    return result


def _stat_dim(entry: Mapping[str, Any]) -> int:
    for key in ("mean", "min", "q01", "max", "q99"):
        value = entry.get(key)
        if isinstance(value, list):
            return len(value[-1]) if value and isinstance(value[-1], list) else len(value)
    return 0


def validate_checkpoint_metadata(
    checkpoint: str | Path,
    *,
    model_repo_id: str = MODEL_REPO_ID,
    model_revision: str = MODEL_REVISION,
    task: str = TASK_TEXT,
) -> dict[str, Any]:
    """Fail closed if the downloaded raw checkpoint contract differs."""

    root = Path(checkpoint).expanduser().resolve()
    required = (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete GR00T checkpoint: missing {missing}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "Gr00tN1d7":
        raise ValueError("checkpoint is not a raw GR00T N1.7 model")
    if int(config.get("action_horizon", 0)) != 40:
        raise ValueError("raw GR00T model action horizon must be 40")

    processor = json.loads(
        (root / "processor_config.json").read_text(encoding="utf-8")
    )
    kwargs = processor.get("processor_kwargs", {})
    modality = (kwargs.get("modality_configs") or {}).get(EMBODIMENT_TAG)
    if not isinstance(modality, dict):
        raise ValueError(f"processor lacks embodiment {EMBODIMENT_TAG!r}")
    video = modality.get("video") or {}
    state = modality.get("state") or {}
    action = modality.get("action") or {}
    language = modality.get("language") or {}
    if video.get("modality_keys") != ["cam_0", "cam_1", "cam_2", "cam_3"]:
        raise ValueError("checkpoint camera order is not cam_0..cam_3")
    if video.get("delta_indices") != [0]:
        raise ValueError("checkpoint must consume one current frame per camera")
    if state.get("modality_keys") != ["robot_q", "hand"]:
        raise ValueError("checkpoint state group order must be robot_q, hand")
    if action.get("modality_keys") != ["robot_q", "hand"]:
        raise ValueError("checkpoint action group order must be robot_q, hand")
    if action.get("delta_indices") != list(range(MODEL_ACTION_HORIZON)):
        raise ValueError("checkpoint valid decoded action horizon must be 16")
    action_configs = action.get("action_configs")
    if not isinstance(action_configs, list) or len(action_configs) != 2:
        raise ValueError("checkpoint must define two action group configs")
    for group in action_configs:
        if not isinstance(group, dict) or str(group.get("rep", "")).upper() != "ABSOLUTE":
            raise ValueError("robot_q and hand actions must both be absolute")
        if str(group.get("type", "")).upper() != "NON_EEF":
            raise ValueError("raw action groups must use NON_EEF semantics")
    if language.get("modality_keys") != ["annotation.human.task_description"]:
        raise ValueError("checkpoint language key changed")

    stats = json.loads((root / "statistics.json").read_text(encoding="utf-8"))
    embodiment_stats = stats.get(EMBODIMENT_TAG)
    if not isinstance(embodiment_stats, dict):
        raise ValueError("checkpoint statistics lack new_embodiment")
    dimensions = {
        (modality_name, key): _stat_dim(
            (embodiment_stats.get(modality_name) or {}).get(key, {})
        )
        for modality_name in ("state", "action")
        for key in ("robot_q", "hand")
    }
    expected = {
        ("state", "robot_q"): 36,
        ("state", "hand"): 2,
        ("action", "robot_q"): 36,
        ("action", "hand"): 2,
    }
    if dimensions != expected:
        raise ValueError(f"checkpoint statistics dimensions changed: {dimensions}")

    embodiment_ids = json.loads(
        (root / "embodiment_id.json").read_text(encoding="utf-8")
    )
    if int(embodiment_ids.get(EMBODIMENT_TAG, -1)) != 10:
        raise ValueError("new_embodiment ID must be 10")
    try:
        dataset_repo_id, dataset_revision = _MODEL_DATASET_PROVENANCE[
            str(model_repo_id)
        ]
    except KeyError as exc:
        raise ValueError(
            "no pinned training-dataset provenance for GR00T model "
            f"{model_repo_id!r}"
        ) from exc
    return {
        "model_repo_id": str(model_repo_id),
        "model_revision": str(model_revision),
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": dataset_revision,
        "embodiment_tag": EMBODIMENT_TAG,
        "task": str(task),
        "state_dim": MODEL_STATE_DIM,
        "decoded_action_dim": MODEL_ACTION_DIM,
        "executable_action_dim": EXECUTABLE_ACTION_DIM,
        "action_horizon": MODEL_ACTION_HORIZON,
        "camera_keys": list(CAMERA_KEYS),
        "lower_body_command_dimensions": 0,
    }
