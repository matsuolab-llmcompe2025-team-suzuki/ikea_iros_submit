"""GR00T-N1.7 inference adapter for the G1 + Dex1-1 coarse-insert task.

This module is independent of the flip-table simulator.  Its public robot-side
contract is Unitree SDK motor order (0..28), two Dex1-1 coordinates in [0, 4.5],
and two root-frame wrist poses in XYZ + Euler-XYZ radians.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gr00t.g1_full_body_mapping import (
    G1_SDK_JOINT_NAMES,
    REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES,
    REAL_G1_RELATIVE_EEF_ACTION_CONFIGS,
    REAL_G1_RELATIVE_EEF_ACTION_DIM,
    REAL_G1_RELATIVE_EEF_ACTION_SLICES,
    REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG,
    REAL_G1_RELATIVE_EEF_STATE_DIM,
    REAL_G1_RELATIVE_EEF_STATE_SLICES,
    dex1_to_hand,
    hand_to_dex1,
    source_euler_xyz_pose_to_xyz_rot6d,
)


LEROBOT_VERSION = "0.6.0"
CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
COARSE_INSERT_SDK_MOTOR_IDS = tuple(range(12, 29))
COARSE_INSERT_SDK_JOINT_NAMES = tuple(
    G1_SDK_JOINT_NAMES[index] for index in COARSE_INSERT_SDK_MOTOR_IDS
)
DEFAULT_DEPLOYMENT_URDF = (
    Path(__file__).resolve().parents[3]
    / "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf"
)


@dataclass(frozen=True)
class CoarseInsertObservation:
    """One robot observation before conversion to the REAL_G1 model layout."""

    eef_pose_xyz_euler: Sequence[float]
    sdk_joint_positions: Sequence[float]
    dex1_open_close: Sequence[float]
    head_left: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    task: str


@dataclass(frozen=True)
class CoarseInsertCommandChunk:
    """Only the outputs that the coarse-insert robot controller can execute."""

    sdk_motor_ids: tuple[int, ...]
    sdk_joint_names: tuple[str, ...]
    sdk_position_targets: np.ndarray
    dex1_open_close_targets: np.ndarray
    canonical_action: np.ndarray
    normalized_action: np.ndarray | None = None
    inference_seconds: float | None = None
    clipped_joint_values: int = 0


def build_coarse_insert_state(
    *,
    eef_pose_xyz_euler: Sequence[float],
    sdk_joint_positions: Sequence[float],
    dex1_open_close: Sequence[float],
) -> np.ndarray:
    """Build the canonical 49-D REAL_G1 state from robot-native values."""

    eef = _finite_vector(eef_pose_xyz_euler, 12, "eef_pose_xyz_euler")
    joints = _finite_vector(sdk_joint_positions, 29, "sdk_joint_positions")
    hands = _finite_vector(dex1_open_close, 2, "dex1_open_close")
    state = np.zeros(REAL_G1_RELATIVE_EEF_STATE_DIM, dtype=np.float32)

    for side, source_start in (("left", 0), ("right", 6)):
        start, end = REAL_G1_RELATIVE_EEF_STATE_SLICES[f"{side}_wrist_eef_9d"]
        state[start:end] = source_euler_xyz_pose_to_xyz_rot6d(
            eef[source_start : source_start + 6].tolist()
        )
    state[18:25] = dex1_to_hand(hands[0], side="left", kind="state")
    state[25:32] = dex1_to_hand(hands[1], side="right", kind="state")
    state[32:39] = joints[15:22]
    state[39:46] = joints[22:29]
    state[46:49] = joints[12:15]
    return state


def load_sdk_position_limits(urdf: Path) -> dict[int, tuple[float, float]]:
    """Read position limits for SDK motors 12..28 from the deployment URDF."""

    root = ET.parse(urdf).getroot()
    by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        by_name[joint.attrib["name"]] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    missing = [name for name in COARSE_INSERT_SDK_JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"URDF is missing position limits for {missing}")
    return {
        sdk_id: by_name[G1_SDK_JOINT_NAMES[sdk_id]]
        for sdk_id in COARSE_INSERT_SDK_MOTOR_IDS
    }


def decode_coarse_insert_action(
    canonical_action: np.ndarray | Sequence[Sequence[float]],
    *,
    normalized_action: np.ndarray | None = None,
    joint_limits: Mapping[int, tuple[float, float]] | None = None,
    inference_seconds: float | None = None,
) -> CoarseInsertCommandChunk:
    """Convert decoded REAL_G1 actions to executable SDK and Dex1 commands."""

    action = np.asarray(canonical_action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2 or action.shape[1] != REAL_G1_RELATIVE_EEF_ACTION_DIM:
        raise ValueError(
            "canonical_action must have shape [T,53], "
            f"got {tuple(action.shape)}"
        )
    if not np.isfinite(action).all():
        raise ValueError("canonical_action contains NaN or Inf")

    # SDK order is waist 12:15, left arm 15:22, right arm 22:29.
    sdk_targets = np.concatenate(
        (action[:, 46:49], action[:, 32:39], action[:, 39:46]), axis=1
    ).copy()
    clipped_joint_values = 0
    if joint_limits is not None:
        for column, sdk_id in enumerate(COARSE_INSERT_SDK_MOTOR_IDS):
            if sdk_id not in joint_limits:
                raise KeyError(f"joint_limits is missing SDK motor id {sdk_id}")
            lower, upper = joint_limits[sdk_id]
            before = sdk_targets[:, column].copy()
            sdk_targets[:, column] = np.clip(before, lower, upper)
            clipped_joint_values += int(np.count_nonzero(before != sdk_targets[:, column]))

    dex1_targets = np.empty((action.shape[0], 2), dtype=np.float32)
    dex1_targets[:, 0] = [
        hand_to_dex1(values, side="left", kind="action")
        for values in action[:, 18:25]
    ]
    dex1_targets[:, 1] = [
        hand_to_dex1(values, side="right", kind="action")
        for values in action[:, 25:32]
    ]

    normalized = None if normalized_action is None else np.asarray(normalized_action).copy()
    return CoarseInsertCommandChunk(
        sdk_motor_ids=COARSE_INSERT_SDK_MOTOR_IDS,
        sdk_joint_names=COARSE_INSERT_SDK_JOINT_NAMES,
        sdk_position_targets=sdk_targets,
        dex1_open_close_targets=dex1_targets,
        canonical_action=action.copy(),
        normalized_action=normalized,
        inference_seconds=inference_seconds,
        clipped_joint_values=clipped_joint_values,
    )


def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Validate the coarse-insert model-side feature contract."""

    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete GR00T checkpoint {checkpoint}: missing {missing}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "groot":
        raise ValueError(f"checkpoint policy type must be 'groot', got {config.get('type')!r}")
    if config.get("embodiment_tag") != REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG:
        raise ValueError("checkpoint does not use the REAL_G1 relative-EEF embodiment")
    if config.get("use_relative_actions") is not True:
        raise ValueError("checkpoint must use native relative actions")
    if _feature_dim(config, "input_features", "observation.state") != 49:
        raise ValueError("checkpoint observation.state must be 49-D")
    if _feature_dim(config, "output_features", "action") != 53:
        raise ValueError("checkpoint action must be 53-D")
    if {
        key for key in config.get("input_features", {}) if key.startswith("observation.images.")
    } != set(CAMERA_KEYS):
        raise ValueError("checkpoint must use head_left, left_wrist, and right_wrist cameras")
    return config


def validate_processor_contract(
    preprocessor: Any,
    postprocessor: Any,
    *,
    required_horizon: int,
) -> bool:
    """Return whether the processor masks the unavailable base/navigation slots."""

    from lerobot.policies.groot.processor_groot import (
        GrootN17ActionDecodeStep,
        GrootN17PackInputsStep,
    )

    pack = next(
        (step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep)),
        None,
    )
    decode = next(
        (step for step in postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep)),
        None,
    )
    if pack is None or decode is None:
        raise ValueError("checkpoint processors lack N1.7 pack/decode steps")
    if decode.pack_step is not pack or not decode.use_relative_action:
        raise ValueError("checkpoint relative-action decoder is not connected to its pack step")
    if pack.action_horizon < required_horizon or pack.valid_action_horizon < required_horizon:
        raise ValueError("checkpoint processor action horizon is too short")
    modality = pack.modality_config or {}
    state_keys = tuple((modality.get("state") or {}).get("modality_keys") or ())
    action_keys = tuple((modality.get("action") or {}).get("modality_keys") or ())
    if state_keys != tuple(REAL_G1_RELATIVE_EEF_STATE_SLICES):
        raise ValueError(f"REAL_G1 state group order mismatch: {state_keys}")
    if action_keys != tuple(REAL_G1_RELATIVE_EEF_ACTION_SLICES):
        raise ValueError(f"REAL_G1 action group order mismatch: {action_keys}")
    configs = (modality.get("action") or {}).get("action_configs") or []
    expected_configs = [REAL_G1_RELATIVE_EEF_ACTION_CONFIGS[key] for key in action_keys]
    if [_plain_action_config(value) for value in configs] != [
        _plain_action_config(value) for value in expected_configs
    ]:
        raise ValueError("REAL_G1 action representations differ from the training mapping")
    exclusions = tuple((modality.get("action") or {}).get("loss_excluded_indices") or ())
    return exclusions == REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES


class CoarseInsertGrootRuntime:
    """Load a LeRobot checkpoint and return robot-executable coarse-insert chunks."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda:0",
        n_action_steps: int = 16,
        *,
        joint_limits: Mapping[int, tuple[float, float]] | None = None,
        urdf: Path | None = DEFAULT_DEPLOYMENT_URDF,
        strict_checkpoint_loading: bool = False,
    ) -> None:
        version = importlib.metadata.version("lerobot")
        if version != LEROBOT_VERSION:
            raise RuntimeError(f"expected lerobot=={LEROBOT_VERSION}, found {version}")
        checkpoint = checkpoint.resolve()
        validate_checkpoint(checkpoint)

        import torch
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        self.model = GrootPolicy.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            strict=strict_checkpoint_loading,
        )
        if not 1 <= n_action_steps <= int(self.model.config.chunk_size):
            raise ValueError(
                f"n_action_steps must be in [1,{self.model.config.chunk_size}], got {n_action_steps}"
            )
        if self.model.config.action_decode_transform is not None:
            raise ValueError("coarse-insert must not use a simulator action decode transform")
        self.model.config.n_action_steps = n_action_steps
        self.model.config.device = str(self.device)
        self.model.to(self.device)
        self.model.eval()
        self.model.reset()
        self.preprocessor, self.postprocessor = make_groot_pre_post_processors_from_pretrained(
            self.model.config,
            str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "groot_n1_7_vlm_encode_v1": {"device": str(self.device)},
            },
        )
        self.has_dex1_loss_mask = validate_processor_contract(
            self.preprocessor,
            self.postprocessor,
            required_horizon=n_action_steps,
        )
        self.n_action_steps = n_action_steps
        if joint_limits is not None:
            self.joint_limits = joint_limits
        elif urdf is not None:
            self.joint_limits = load_sdk_position_limits(urdf)
        else:
            raise ValueError(
                "coarse-insert inference requires joint_limits or a deployment URDF"
            )

    def reset(self) -> None:
        self.model.reset()

    def predict(self, observation: CoarseInsertObservation) -> CoarseInsertCommandChunk:
        state = build_coarse_insert_state(
            eef_pose_xyz_euler=observation.eef_pose_xyz_euler,
            sdk_joint_positions=observation.sdk_joint_positions,
            dex1_open_close=observation.dex1_open_close,
        )
        return self.predict_canonical(
            state=state,
            head_left=observation.head_left,
            left_wrist=observation.left_wrist,
            right_wrist=observation.right_wrist,
            task=observation.task,
        )

    def predict_canonical(
        self,
        *,
        state: np.ndarray,
        head_left: np.ndarray,
        left_wrist: np.ndarray,
        right_wrist: np.ndarray,
        task: str,
    ) -> CoarseInsertCommandChunk:
        """Run a materialized canonical 49-D state, mainly for dataset evaluation."""

        canonical_state = _finite_vector(state, 49, "state")
        raw: dict[str, Any] = {
            "observation.state": self.torch.from_numpy(canonical_state),
            "task": str(task),
        }
        for key, image in zip(CAMERA_KEYS, (head_left, left_wrist, right_wrist)):
            image_array = np.asarray(image)
            if image_array.shape != (480, 640, 3) or image_array.dtype != np.uint8:
                raise ValueError(
                    f"{key} must be uint8 HWC [480,640,3], got "
                    f"{image_array.shape} {image_array.dtype}"
                )
            raw[key] = self.torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1)

        started = time.perf_counter()
        processed = self.preprocessor(raw)
        normalized = self.model.predict_action_chunk(processed)
        decoded = self.postprocessor(normalized)
        elapsed = time.perf_counter() - started
        decoded_array = decoded.detach().cpu().float().numpy()
        normalized_array = normalized.detach().cpu().float().numpy()
        if decoded_array.ndim != 3 or decoded_array.shape[0] != 1 or decoded_array.shape[2] != 53:
            raise RuntimeError(f"decoded action must have shape [1,T,53], got {decoded_array.shape}")
        decoded_array = decoded_array[0, : self.n_action_steps]
        normalized_array = normalized_array[0, : self.n_action_steps, :53]
        return decode_coarse_insert_action(
            decoded_array,
            normalized_action=normalized_array,
            joint_limits=self.joint_limits,
            inference_seconds=elapsed,
        )


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite shape ({size},), got {array.shape}")
    return array


def _feature_dim(config: dict[str, Any], group: str, key: str) -> int | None:
    shape = config.get(group, {}).get(key, {}).get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    return int(shape[-1])


def _plain_action_config(config: Mapping[str, Any]) -> tuple[str, str, str, str]:
    def plain(value: Any) -> str:
        return str(getattr(value, "value", value)).lower().replace("_", "").replace("+", "")

    return (
        plain(config.get("rep")),
        plain(config.get("type")),
        plain(config.get("format")),
        str(config.get("state_key")),
    )
