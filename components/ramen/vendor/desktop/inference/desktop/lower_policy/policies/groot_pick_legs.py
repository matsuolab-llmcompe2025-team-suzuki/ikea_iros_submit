"""GR00T pick_legs loader (Issue #125、pick_table_leg 専用の full-body embodiment)。

# 位置付け

`Team-RAMEN/groot-n1.7-pick-legs-ver1` (Isaac-GR00T native raw finetune) 専用
Policy class。task_5+7 系 (LeRobot fork format、REAL_G1_RELATIVE_EEF embodiment)
とは format / schema が完全に異なるため、既存 `Gr00tPolicy` を汚さず別 module
として実装。

# 訓練時仕様の厳密追従

- **HF repo**: `Team-RAMEN/groot-n1.7-pick-legs-ver1` (author prefix 無しの
  raw naming、Isaac-GR00T 側の launch_finetune.py 直接 finetune 由来)
- **Ckpt subdir**: `checkpoint-40000/` (HF Trainer の native output 構造、
  model.safetensors + optimizer.pt + processor_config.json 等が subdir 配下)
- **Base model**: nvidia/GR00T-N1.7-3B (backbone は task_5+7 と共通)
- **Embodiment**: `EmbodimentTag.NEW_EMBODIMENT` (embodiment_id.json では
  "unitree_g1_full_body_with_waist_height_nav_cmd" = id 25 として登録)
- **Modality config**: `g1_pick_leg_config.py` に register_modality_config で
  runtime 登録 (Isaac-GR00T の gr00t.data.embodiment_tags API 依存)。inference
  時も同 config 適用が必要。
- **State/Action**: raw whole-body joint absolute (36 + 2 hand = 38D 両方)
- **Chunk len**: 16 (delta_indices=[0..15])
- **Video**: 4 cam (cam_0/1/2/3 = head_left/head_right/wrist_left/wrist_right)
- **use_bf16**: True (config.json:model_dtype="bfloat16")

# 現状 (Skeleton)

**実装は skeleton レベル**。実 model load + forward は以下 未着:
- HF hub subdir download (hf_hub_download subfolder or snapshot_download 経由)
- Isaac-GR00T native loader vs LeRobot fork wrapper のどちら使うか判断
- register_modality_config の inference 側適用パス
- 実 GPU 検証 (12GB weights + Cosmos-Reason2-2B backbone、Sakura H100 前提)

`@pytest.mark.integration` marker で real forward 経路を skip。Skeleton の
API 契約 (validate / build_batch_dict 等) は local test で verify。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    PolicyAction,
    PolicyConfig,
    RawRobotState,
)


# ---- pick_legs 契約定数 (g1_pick_leg_config.py + HF config.json 準拠) ---- #

STATE_DIM: int = 38     # robot_q(36) + hand(2)
ACTION_DIM: int = 38    # robot_q_desired(36) + hand_cmd(2)、ABSOLUTE
CHUNK_LEN: int = 16     # g1_pick_leg_config.py:action.delta_indices=list(range(0, 16))
IMAGE_HW: tuple[int, int] = (256, 256)  # HF config.json:image_target_size (processor 内 crop 230x230)

# 4 cam layout (cam_0-3 全部、IR 除外)。task_5+7 系 (3 cam) との違い。
# subtask_training.json:source_dataset.camera_map と integer id 順に対応:
#   cam_0 = HEAD_LEFT / cam_1 = HEAD_RIGHT / cam_2 = WRIST_LEFT / cam_3 = WRIST_RIGHT
CAMERAS: tuple[CameraKey, ...] = (
    CameraKey.HEAD_LEFT,
    CameraKey.HEAD_RIGHT,
    CameraKey.WRIST_LEFT,
    CameraKey.WRIST_RIGHT,
)

# Ckpt subdir (HF repo 内での weights 配置場所、Isaac-GR00T Trainer native output)
CKPT_SUBDIR: str = "checkpoint-40000"

# subtask_training.json:subtasks.pick_leg.task
DEFAULT_LANGUAGE_PROMPT: str = "pick table leg"

# embodiment_id.json より (id 25 = full-body variant)、pick_legs は
# EmbodimentTag.NEW_EMBODIMENT として register される (実 runtime は
# g1_pick_leg_config.py の register_modality_config を先に import する必要)
EMBODIMENT_ID: int = 25
MODEL_REVISION: str = "b63d9c49482cfc8998d1a3a0e3e0ac3ceb1d620a"
EXECUTION_HORIZON: int = 8
RUNTIME_CHECKPOINT_FILES: tuple[str, ...] = (
    "config.json",
    "embodiment_id.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
)

# The source dataset represents a fully open Dex1 as 4.5 rad.  The physical
# actuator can travel to 5.4 rad for clearance, but model state/action values
# must remain in the recorded 4.5-rad coordinate while policy inference runs.
DEX1_DATASET_MAX_RAD: float = 4.5


def validate_pick_legs_config(cfg: PolicyConfig) -> None:
    """`PolicyConfig` が pick_legs の contract を満たすかを検証する。

    - cams が 4 cam layout であること (cam_0-3 全部)
    - mode が "none" のみ (pick_legs は OBB channel 非対応 = overlay も
      current 学習 run では未使用)
    - dtype が "bf16" 推奨 (config.json:model_dtype=bfloat16、use_bf16=True)
    """
    if tuple(cfg.cams) != CAMERAS:
        raise ValueError(
            f"GR00T pick_legs requires cams={tuple(c.value for c in CAMERAS)!r} "
            f"(4 cam layout per g1_pick_leg_config.py), got "
            f"{tuple(c.value for c in cfg.cams)!r}"
        )
    if cfg.mode not in ("none", "overlay"):
        raise ValueError(
            f"GR00T pick_legs supports mode='none' (or 'overlay' if enabled). "
            f"'precomputed_token' is not supported (no OBB token channel)."
        )


def build_batch_dict(obs: Observation) -> dict:
    """Observation → pick_legs pre-processor が受け取る raw dict。

    Isaac-GR00T native processor (g1_pick_leg_config.py で登録される modality
    config) が期待する key を組む。task_5+7 系との違い:
        - image key = "cam_0" .. "cam_3" (task_5+7 は "observation.images.head_left" 等)
        - state key = "robot_q" (36) + "hand" (2) の 2 部品 (task_5+7 は
          "observation.state" 一体)
        - action は inference では入れない (predict 対象)
        - language key = "annotation.human.task_description" (from modality.json)

    Note: 本 helper は torch を要さず、後段 predict() で tensor 化する
    (default env unit test は本 helper の shape/key を verify する)。

    Args:
        obs: Skill wrapper が assemble 済の Observation。state は 38D で
             build_state_from_raw で組んである前提 (robot_q_current + hand_state
             を concat)。

    Returns:
        Isaac-GR00T processor 用の dict。
    """
    if obs.state.shape != (STATE_DIM,):
        raise ValueError(
            f"Observation.state must have shape ({STATE_DIM},) for pick_legs, got "
            f"{obs.state.shape}"
        )
    if obs.language is None:
        raise ValueError(
            "pick_legs requires obs.language (natural language prompt). Skill wrapper "
            "must set it (default: DEFAULT_LANGUAGE_PROMPT = 'pick table leg')."
        )

    # State を robot_q(36) + hand(2) に分解 (modality.json:state 準拠)
    robot_q = obs.state[:36].astype(np.float32, copy=False)
    hand = obs.state[36:38].astype(np.float32, copy=False)

    batch: dict = {
        "robot_q": robot_q,
        "hand": hand,
        "annotation.human.task_description": obs.language,
    }
    for cam in CAMERAS:
        if cam not in obs.frames_bgr:
            raise KeyError(
                f"Observation.frames_bgr missing required cam {cam.value!r} "
                f"(pick_legs needs {tuple(c.value for c in CAMERAS)!r} = 4 cams)"
            )
        frame = obs.frames_bgr[cam]
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame for {cam.value!r} must be (H, W, 3) uint8 BGR, got "
                f"shape={frame.shape} dtype={frame.dtype}"
            )
        # cam_id = 0..3 の integer key (g1_pick_leg_config.py:modality_keys)
        batch[f"cam_{cam.cam_id}"] = frame

    return batch


def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
    """Raw robot state → exact pick_legs 38D state.

    task_5+7 と違って **whole body joint absolute** を直接使う。REAL_G1_RELATIVE_EEF
    のような wrist EE 変換無し、UPPER_BODY 選択も無し (下半身含む全 29 joint 使う)。

    Training source `observation.state` (38 dim) の layout:
        [0:7]    root_pose = (x, y, z, qw, qx, qy, qz) in world frame
        [7:36]   joint_positions = G1 29 joints (leg12 + waist3 + arm14)
        [36:38]  hand_state = Dex1-1 (left_gripper_q, right_gripper_q)

    実機 (Orin `real_hw_bridge_node`) では root_pose 相当の topic が publish
    されない (system.launch.py は `/joint_states` のみ)。IMU/odometry 統合前は
    root pose = zeros で padding する。root pose を pick_legs policy がどの程度
    使っているかは未検証、必要なら Orin 側に odometry publisher を追加する。

    Args:
        raw: orchestrator の RawRobotState (Orin 実データ layout)。

    Returns:
        (38,) float32 = zeros(7) + joint_positions(29) + hand_state(2)。
    """
    if raw.joint_positions.shape != (29,):
        raise ValueError(
            f"joint_positions must be (29,), got {raw.joint_positions.shape}"
        )
    if raw.hand_state.shape != (2,):
        raise ValueError(f"hand_state must be (2,), got {raw.hand_state.shape}")

    # The real bridge has no global root pose.  Match the stationary standing
    # proxy used by the proven issue-70 physical adapter; model root/lower-body
    # outputs are never sent to the robot.
    root_pose_pad = np.asarray(
        (0.0, 0.0, 0.70, 1.0, 0.0, 0.0, 0.0), dtype=np.float32
    )
    if not np.isfinite(raw.hand_state).all():
        raise ValueError("hand_state must be finite")
    hand = np.clip(
        raw.hand_state.astype(np.float32, copy=False),
        0.0,
        DEX1_DATASET_MAX_RAD,
    )
    state = np.concatenate(
        [root_pose_pad, raw.joint_positions, hand]
    ).astype(np.float32, copy=False)
    return state


class _PickLegsWorkerClient:
    """Python 3.10 DDS runtime → isolated Python 3.12 GR00T worker."""

    def __init__(self, cfg: PolicyConfig) -> None:
        from inference.desktop.upper_policy.worker_protocol import (
            receive_message,
            send_message,
        )

        repo_root = Path(__file__).resolve().parents[4]
        checkpoint = self._resolve_checkpoint(repo_root, cfg)
        model_repo_id, model_revision = self._parse_ref(cfg)
        worker_python = repo_root / "model/subtask_policy_training/.venv/bin/python"
        worker_script = (
            repo_root
            / "model/subtask_policy_training/deployment/real_groot_n17_worker.py"
        )
        if not worker_python.is_file() or not worker_script.is_file():
            raise FileNotFoundError(
                "pick-leg GR00T worker runtime is incomplete: "
                f"python={worker_python} script={worker_script}"
            )
        command = [
            str(worker_python),
            str(worker_script),
            "--checkpoint",
            str(checkpoint),
            "--device",
            str(cfg.device),
            "--model-repo-id",
            model_repo_id,
            "--model-revision",
            model_revision,
            "--task",
            DEFAULT_LANGUAGE_PROMPT,
        ]
        print(
            "[groot-pick] starting isolated low-memory Python 3.12 worker",
            file=sys.stderr,
        )
        self._process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        self._send_message = send_message
        self._receive_message = receive_message
        self._request_id = 0
        try:
            if self._process.stdout is None:
                raise RuntimeError("pick-leg worker stdout pipe is missing")
            ready = self._receive_message(self._process.stdout)
            if not isinstance(ready, dict) or ready.get("type") != "ready":
                raise RuntimeError(f"pick-leg worker did not become ready: {ready!r}")
            contract = ready.get("contract") or {}
            if int(contract.get("state_dim", -1)) != STATE_DIM:
                raise RuntimeError(f"pick-leg worker state contract changed: {contract}")
            if int(contract.get("decoded_action_dim", -1)) != ACTION_DIM:
                raise RuntimeError(f"pick-leg worker action contract changed: {contract}")
            if int(contract.get("lower_body_command_dimensions", -1)) != 0:
                raise RuntimeError("pick-leg worker may command the lower body")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _parse_ref(cfg: PolicyConfig) -> tuple[str, str]:
        """ckpt_ref を (repo_id, revision) に分解する。

        `HF-repo@revision` 形式なら分割、@revision 省略時は ver1 pinned
        MODEL_REVISION を default に使う。worker の provenance lookup は
        @sha 無しの repo_id を key にするため、ここで必ず剥がす。
        """
        repo_id = str(cfg.ckpt_ref)
        revision = MODEL_REVISION
        if "@" in repo_id:
            repo_id, revision = repo_id.rsplit("@", 1)
            if not repo_id or not revision:
                raise ValueError(
                    "pick-leg ckpt_ref must be a local directory, HF repo, or "
                    f"HF-repo@revision; got {cfg.ckpt_ref!r}"
                )
        return repo_id, revision

    @staticmethod
    def _resolve_checkpoint(repo_root: Path, cfg: PolicyConfig) -> Path:
        # checkpoint_subdir semantics (config 駆動):
        #   None            → repo root 直下 (ver2-lora の LeRobot/root layout)
        #   "checkpoint-..." → その subfolder (ver1 Isaac-GR00T Trainer native)
        # ver1 は policy_config.yaml で checkpoint_subdir=CKPT_SUBDIR を明示指定する
        # (generic groot 経路 _resolve_groot_checkpoint_root と同一の None=root 規約)。
        subdir = cfg.checkpoint_subdir
        reference = Path(str(cfg.ckpt_ref)).expanduser()
        if reference.is_dir():
            checkpoint = reference / subdir if subdir else reference
        else:
            repo_id, revision = _PickLegsWorkerClient._parse_ref(cfg)
            sealed = repo_root / ".checkpoints/groot-n1.7-pick-legs-ver1"
            if repo_id == "Team-RAMEN/groot-n1.7-pick-legs-ver1" \
                    and subdir and (sealed / subdir).is_dir():
                checkpoint = sealed / subdir
            else:
                from huggingface_hub import snapshot_download

                snapshot = Path(snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    allow_patterns=tuple(
                        f"{subdir}/{name}" if subdir else name
                        for name in RUNTIME_CHECKPOINT_FILES
                    ),
                ))
                checkpoint = snapshot / subdir if subdir else snapshot
        required = (
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
            "model.safetensors.index.json",
        )
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"incomplete pick-leg checkpoint {checkpoint}: missing={missing}"
            )
        return checkpoint.resolve()

    @staticmethod
    def _jpeg(frame: np.ndarray, role: str) -> bytes:
        import cv2

        image = np.asarray(frame)
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"{role} must be uint8 (480,640,3), got {image.shape} {image.dtype}"
            )
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise RuntimeError(f"failed to encode {role} for pick-leg GR00T")
        return encoded.tobytes()

    def predict(self, obs: Observation) -> PolicyAction:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("pick-leg worker pipes are closed")
        self._request_id += 1
        camera_role_to_key = {
            CameraKey.HEAD_LEFT: "observation.images.cam_0",
            CameraKey.HEAD_RIGHT: "observation.images.cam_1",
            CameraKey.WRIST_LEFT: "observation.images.cam_2",
            CameraKey.WRIST_RIGHT: "observation.images.cam_3",
        }
        cameras = {
            key: self._jpeg(obs.frames_bgr[cam], cam.value)
            for cam, key in camera_role_to_key.items()
        }
        self._send_message(self._process.stdin, {
            "type": "predict",
            "request_id": self._request_id,
            "state": np.asarray(obs.state, dtype=np.float32),
            "cameras": cameras,
            "task": obs.language or DEFAULT_LANGUAGE_PROMPT,
        })
        response = self._receive_message(self._process.stdout)
        if not isinstance(response, dict):
            raise RuntimeError(f"invalid pick-leg worker response: {response!r}")
        if response.get("type") == "error":
            raise RuntimeError(f"pick-leg worker failed: {response.get('error')}")
        if response.get("type") != "prediction" \
                or int(response.get("request_id", -1)) != self._request_id:
            raise RuntimeError(f"unexpected pick-leg worker response: {response!r}")
        raw = np.asarray(response.get("actions"), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or not np.isfinite(raw).all():
            raise RuntimeError(f"invalid pick-leg action shape/value: {raw.shape}")
        # 38D = root7 + body29 + hands2.  Only waist3 + arms14 + hands2 enter
        # the shared VlaSkill contract; dispatch flags still keep Regular Mode
        # as the sole lower-body/waist owner when requested.
        action = np.concatenate(
            (raw[:, 19:22], raw[:, 22:36], raw[:, 36:38]), axis=1
        )
        return PolicyAction(
            action_chunk=action,
            latency_ms=float(response.get("inference_ms", 0.0)),
            metadata={
                "mode": "none",
                "chunk_len": int(action.shape[0]),
                "action_dim": int(action.shape[1]),
                "raw_action_shape_38d": tuple(raw.shape),
                "lower_body_command_dimensions": 0,
            },
        )

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None and process.stdout is not None:
                    self._send_message(process.stdin, {"type": "close"})
                    self._receive_message(process.stdout)
            except Exception:
                pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        self._process = None


class Gr00tPolicyPickLegs:
    """pick_legs GR00T inference loader (Isaac-GR00T native raw ckpt 対応)。

    task_5+7 系 `Gr00tPolicy` とは別 class、full-body embodiment + subdir
    load + 4 cam の別 schema を扱う。

    Attributes:
        cfg: 生成時 config (immutable)。
        _model: Isaac-GR00T model instance (lazy load)。
    """

    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM
    CHUNK_LEN = CHUNK_LEN
    IMAGE_HW = IMAGE_HW
    CAMERAS = CAMERAS
    CKPT_SUBDIR = CKPT_SUBDIR
    DEFAULT_LANGUAGE_PROMPT = DEFAULT_LANGUAGE_PROMPT
    EMBODIMENT_ID = EMBODIMENT_ID
    EXECUTION_HORIZON = EXECUTION_HORIZON
    build_state_from_raw = staticmethod(build_state_from_raw)

    def __init__(
        self,
        cfg: PolicyConfig,
        _model=None,
        _worker_client: _PickLegsWorkerClient | None = None,
    ) -> None:
        validate_pick_legs_config(cfg)
        self.cfg = cfg
        self._model = _model
        self._worker_client = _worker_client

    @classmethod
    def from_ckpt(cls, cfg: PolicyConfig) -> "Gr00tPolicyPickLegs":
        """Load the sealed raw checkpoint in the isolated Python 3.12 env."""
        return cls(cfg=cfg, _worker_client=_PickLegsWorkerClient(cfg))

    def warmup(self, n_iter: int = 5) -> None:
        if self._worker_client is not None:
            dummy_obs = self._make_dummy_observation()
            for _ in range(n_iter):
                self._worker_client.predict(dummy_obs)
            return
        if self._model is None:
            raise RuntimeError(
                "Gr00tPolicyPickLegs is not loaded. Call from_ckpt() first."
            )
        dummy_obs = self._make_dummy_observation()
        for _ in range(n_iter):
            self.predict(dummy_obs)

    def predict(self, obs: Observation) -> PolicyAction:
        """1 tick observation → pick_legs 38D action chunk (16, 38)。"""
        if self._worker_client is not None:
            return self._worker_client.predict(obs)
        if self._model is None:
            raise RuntimeError(
                "Gr00tPolicyPickLegs is not loaded. Call from_ckpt() first."
            )
        # lazy: env-isolated dependencies
        import torch

        t0 = time.monotonic_ns()
        raw_batch = build_batch_dict(obs)

        # Real forward path (Isaac-GR00T processor invocation) is pending
        # implementation. See from_ckpt() docstring.
        raise NotImplementedError(
            "Gr00tPolicyPickLegs.predict real forward pending (Issue #125)。"
        )

    def close(self) -> None:
        if self._worker_client is not None:
            self._worker_client.close()
            self._worker_client = None
            return
        try:
            import torch  # lazy
        except ImportError:
            self._model = None
            return
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_dummy_observation(self) -> Observation:
        H, W = 480, 640
        return Observation(
            frames_bgr={cam: np.zeros((H, W, 3), dtype=np.uint8) for cam in CAMERAS},
            frames_bgr_prev=None,
            state=np.asarray(
                (0.0, 0.0, 0.70, 1.0, 0.0, 0.0, 0.0) + (0.0,) * 31,
                dtype=np.float32,
            ),
            skill_id=None,
            language=DEFAULT_LANGUAGE_PROMPT,
            obb_detections=None,
            timestamp_ns=time.monotonic_ns(),
        )
