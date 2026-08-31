"""Policy protocol + 共通 dataclass (Issue #125、C-axis inference integration Phase 1)。

# 訓練時仕様の厳密追従

Batch key naming / dtype / shape の唯一の真実は `model/ramen_ori/model.py` の
docstring と `model/ramen_ori/data_lerobot.py` の実装、および GR00T 側は
`model/subtask_policy_training/configs/subtask_training.json`。本 module は
inference 側契約をこれに合わせて宣言するのみで、実際の変換は各 Policy 実装
(`groot.py` / `ramen_ori.py`) が training-side helper (`derive_state_71d` 等) を
**直接 import して再利用**する方針。

# Observation の粒度

`Observation` は **raw sensor snapshot + orchestrator が既に compute した派生 signal**
を持つ (BGR uint8 frames + assembled state + skill_id + language + YOLO detections)。

- frames_bgr: cv2 native BGR uint8。Policy 内部で resize / ImageNet normalize /
  RGB 変換する (training data_lerobot.py の `_transform_item` と 1 pixel 一致)。
- state: 既に `derive_state_71d` 相当が完了した (state_dim,) tensor。組立は caller
  (orchestrator + joint_state_source) 側の責務 (tick 間 velocity 差分の buffer は
  orchestrator に置く方が clean)。
- skill_id: RAMEN-Ori 系 policy が読む条件 signal (nn.Embedding index)。
- language: GR00T 系 policy が読む条件 signal (natural language prompt string)。
- obb_detections: orchestrator perception 層が既に走らせた YOLO の結果。C-2
  (precomputed_token) / C-11 (overlay) mode の Policy が token 変換 or 描画に使う。

# Skill ↔ Policy の関係

Skill (per-skill_name wrapper、`skills/*.py`) が自分の identity (skill_id / language)
を hardcode し、`step(obs)` で Observation にそれらを埋めて Policy.predict() を叩く。
Policy 層は backbone-focused で skill_name を全く知らない。GR00T ↔ RAMEN-Ori の swap
は Skill constructor に渡す `policy` を差し替えるだけで済む構造。

# Camera key

Training-side data_lerobot.py の `default_camera_keys()` (= `observation.images.cam_0..3`)
順序を単一の真実として、inference 側 `CameraKey` enum の `cam_id` mapping を合わせる。
subtask_training.json 側は camera_map で physical name → cam_i を明示している。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

import numpy as np

from inference.desktop.perception.yolo_obb import OBBDetection


class CameraKey(str, Enum):
    """物理 camera identifier (G1 platform、task 5/7 baseline)。

    G1_WBT 4 cam layout:
        - HEAD_LEFT   = cam_0 (head stereo split L)
        - HEAD_RIGHT  = cam_1 (head stereo split R)
        - WRIST_LEFT  = cam_2 (left wrist D405 RGB mono)
        - WRIST_RIGHT = cam_3 (right wrist D405 RGB mono)

    subtask_training.json:source_dataset.camera_map と cam_id 割当を一致させる。
    """

    HEAD_LEFT = "head_left"
    HEAD_RIGHT = "head_right"
    WRIST_LEFT = "wrist_left"
    WRIST_RIGHT = "wrist_right"

    @property
    def cam_id(self) -> int:
        """training-side model.py の cam_id integer (0..num_cams-1)。"""
        return _CAM_ID_MAP[self]

    @property
    def training_key(self) -> str:
        """training data_lerobot.py の LeRobot column key。"""
        return f"observation.images.cam_{self.cam_id}"

    @property
    def is_wrist(self) -> bool:
        """`data_lerobot.py:is_wrist` と同判定 (aug 系 dispatch 用)。"""
        return self.name.startswith("WRIST_")


# training-side data_lerobot.py の `default_camera_keys()` 順序と一致すること。
# 変更する場合、training config を再生成して整合を取る必要あり。
_CAM_ID_MAP: dict[CameraKey, int] = {
    CameraKey.HEAD_LEFT: 0,
    CameraKey.HEAD_RIGHT: 1,
    CameraKey.WRIST_LEFT: 2,
    CameraKey.WRIST_RIGHT: 3,
}


# ---- G1 upper-body joint index 定数 (RAMEN-Ori / GR00T 共通) ---- #
#
# Orin `real_hw_bridge_node` が publish する JointState.position (G1_JOINT_NAMES
# 順、29 dim) 内での index を group ごとに定義。UPPER_BODY のみ policy state に
# 使う (lower body 12 joint = leg は upper-body policy の signal に不要)。
#
# 参照: g1_hw_bridge/joint_mapping.py:G1_JOINT_NAMES
#   12-14: waist   (WaistYaw, WaistRoll, WaistPitch)
#   15-21: left_arm (Shoulder P/R/Y, Elbow, Wrist R/P/Y)
#   22-28: right_arm (Shoulder P/R/Y, Elbow, Wrist R/P/Y)
G1_WAIST_SLICE: slice = slice(12, 15)
G1_LEFT_ARM_SLICE: slice = slice(15, 22)
G1_RIGHT_ARM_SLICE: slice = slice(22, 29)
G1_UPPER_BODY_JOINT_INDICES: tuple[int, ...] = (
    tuple(range(G1_WAIST_SLICE.start, G1_WAIST_SLICE.stop))
    + tuple(range(G1_LEFT_ARM_SLICE.start, G1_LEFT_ARM_SLICE.stop))
    + tuple(range(G1_RIGHT_ARM_SLICE.start, G1_RIGHT_ARM_SLICE.stop))
)  # 3 + 7 + 7 = 17 indices
G1_UPPER_BODY_JOINT_DIM: int = len(G1_UPPER_BODY_JOINT_INDICES)  # 17


@dataclass(frozen=True)
class RawRobotState:
    """orchestrator が集めた実データ形式の proprioception snapshot。

    **設計方針** (Issue #125 Phase A-1): inference 側は Orin (real_hw_bridge_node)
    から降ってくる実データ layout を保持し、training pipeline 依存 (root_pose 7 +
    joints 29 + hand 2 = 38 dim の "source_state" 形式) はここに持ち込まない。
    各 Policy の `build_state_from_raw` staticmethod がここから target layout に
    組み替える (RAMEN-Ori 71D E-δ / GR00T 49D REAL_G1_RELATIVE_EEF / pick_legs
    38D full-body)。root_pose は Orin 実機では publish されない (system.launch.py
    の real_hw_bridge_node は /joint_states のみ) ため保持しない = 必要な policy
    (pick_legs 等) は zeros で padding する。

    Attributes:
        joint_positions: (29,) float32、G1_JOINT_NAMES 順の関節位置 [rad]
            (leg 12 + waist 3 + arm 14)。Orin `real_hw_bridge_node` が publish
            する `sensor_msgs/JointState.position` そのまま。joint_state_source
            で受けたものを VlaSkill が RawRobotState に詰める。
        hand_state: (2,) float32、Dex1-1 hand gripper 状態 [left_gripper_q,
            right_gripper_q]。公式DDS stateおよびsource datasetと同じ物理
            motor-output rad（0=閉側、約5.4=開側）。
        ee_state: (12,) float32、EE pose = left(xyz+euler xyz) + right(xyz+euler
            xyz)、reference_frame="root_link" (training source dataset の
            eef_pose_format と同じ)。Desktop 側 FK (pinocchio + G1 URDF) で計算
            する予定 (Phase A-5)、それまでは zeros。
        last_action_19d: (19,) float32 or None、前 tick に VlaSkill が送った 19D
            action = waist3 + arm14 + hand2 (subtask_training.json:action.names
            順)。RAMEN-Ori 71D の tracking_err slice ([19:38] = q_desired -
            q_current) の source。first tick は None → tracking_err zeros。
        joint_positions_prev: (29,) float32 or None、前 tick の joint_positions。
            RAMEN-Ori 71D の velocity slice ([38:57] = q_current[t] -
            q_current[t-1]) の source。first tick は None → velocity zeros。
        hand_state_prev: (2,) float32 or None、前tickの実測Dex1位置 [rad]。
            joint_positions_prevと同じ時刻の値。first tickはNone。
    """

    joint_positions: np.ndarray
    hand_state: np.ndarray
    ee_state: np.ndarray
    last_action_19d: np.ndarray | None = None
    joint_positions_prev: np.ndarray | None = None
    hand_state_prev: np.ndarray | None = None


@dataclass(frozen=True)
class Observation:
    """1 tick 分の raw sensor snapshot + orchestrator 派生 signal。

    Attributes:
        frames_bgr: cam key → (H, W, 3) uint8 BGR image (cv2 native)。key は
            `CameraKey` enum。Policy 内部で resize/normalize/RGB 変換される。
            全 cam を含む必要は無く、config `cams` で指定された cam のみあれば良い。
        frames_bgr_prev: 過去 frames。RAMEN-Ori では I_{t-1}、時間入力を持つ
            Furniture-GR00TではI_{t-20}をPolicy adapterが設定する。通常GR00Tは
            使わないのでcallerがNoneを渡してよい。訓練時仕様 (data_lerobot.py:
            "images_prev、data.py が先頭 frame は zeros") に従い、first tick で
            None を渡すと Policy 側で zeros に fill する。
        state: (state_dim,) float32、既に derive_state_71d() 等で組み上げた state
            (E-δ 71D or E-β 73D、GR00T なら 49D REAL_G1_RELATIVE_EEF)。caller
            (orchestrator + joint_state_source) が state buffer を保持して組む
            責務。tick 間の velocity 差分等はここで解決済であることが前提。
        skill_id: RAMEN-Ori 系 policy 用の条件 signal (0..num_skills-1)。GR00T 系
            は無視する。Skill wrapper が自分の identity から埋める。None は
            "この observation は skill_id conditioning 対象外" (テスト用)。
        language: GR00T 系 policy 用の条件 signal (natural language prompt)。
            RAMEN-Ori 系は無視する。Skill wrapper が subtask_training.json の
            `task` field から埋める。None は "language conditioning 対象外"。
        obb_detections: orchestrator perception 層 (YoloObbPerception) が per-cam
            per-frame 推論した OBB dict (cam_key → detection list)。C-2
            (precomputed_token) / C-11 (overlay) mode の Policy が token 変換 or
            描画に使う。dict にすることで cam attribution が保持され、cam mismatch
            (head_left の det を head_right frame に描画するような bug) を silent
            に起こさない。**YOLO は head_left + head_right の 2 cam でのみ走る**
            前提 (wrist YOLO は無し)、caller は該当 cam key のみ埋めれば良い。
            None or 空 dict は "OBB signal 無し" (mode=none の Policy でも受け取り
            可、無視される)。
        timestamp_ns: wall-clock ns (`time.monotonic_ns()`)、latency 計測 + tick
            間隔監視用。
    """

    frames_bgr: dict[CameraKey, np.ndarray]
    frames_bgr_prev: dict[CameraKey, np.ndarray] | None
    state: np.ndarray
    skill_id: int | None
    language: str | None
    obb_detections: dict[CameraKey, list[OBBDetection]] | None
    timestamp_ns: int


@dataclass(frozen=True)
class PolicyAction:
    """`Policy.predict` の戻り値。

    Attributes:
        action_chunk: (chunk_len, action_dim) float32、raw action space
            (training の action GT と同 unit)。caller が全 chunk 消費するか
            頭 1 tick のみ使うかは caller 側の判断。
        latency_ms: `predict()` 内部の wall-clock 経過時間 (ms)。budget verify
            (Phase 8) で使う。
        metadata: 診断用 dict (yolo_det_count / mode / cam_id / ...)。schema は
            固定せず、Policy 実装が診断に必要な key を任意に追加する。
    """

    action_chunk: np.ndarray
    latency_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyConfig:
    """Policy 実装共通の config。

    Attributes:
        mode: **OBB signal の使い方**の識別 (YOLO 実行の有無ではない、YOLO は
            perception 層で既に走っている前提):
                - "none": OBB signal を全く使わない (dummy zeros で埋めるか
                   token を作らない)
                - "precomputed_token": obb_detections を C-2 token に変換して
                   RAMEN-Ori に流す
                - "overlay": obb_detections を frames_bgr に描画してから
                   backbone に流す (GR00T C-11 overlay + RAMEN-Ori 併用可)
            GR00T は "none" / "overlay" のみ有効、RAMEN-Ori は 3 mode 全対応。
        ckpt_ref: HF repo path (`Team-RAMEN/...@<sha>`) or local ckpt file path。
            Policy 実装が `hf://` prefix or 存在する local path を判別する。
        checkpoint_subdir: HF snapshot/local directory内のmodel directory。
            Trainerが複数stepを一つのrepoへ保存した場合だけ指定する。絶対pathと
            `..` traversalは禁止し、HF downloadもこのsubdirだけに限定する。
        device: torch device 指定 ("cuda" / "cpu")。
        dtype: model forward の precision ("fp32" / "bf16" / "fp16")。training config
            (base.yaml: fp32、Phase 2 で bf16 検討) と合わせる。
        cams: 使用する CameraKey tuple。**順序が training data pipeline の
            `camera_keys` 順と一致**すること (cam_id 割当も一致するように)。
            GR00T (3 cam = head_left + wrist L/R) / RAMEN-Ori (4 cam) で異なる。
        temporal_lambda: chunk-based policy 出力の temporal ensemble decay 係数
            (Issue #128 Phase 2、GR00T のみ現状使用)。float なら
            `weight = exp(temporal_lambda * age_in_steps)` で古い chunk を減衰、
            None なら blend を無効化 (最新 chunk[0] を tick 毎に採用)。
            issue-70 検証値 -0.1 が推奨 (age=5 で weight~0.6、境界カクカク緩和)。
            RAMEN-Ori 側は現状無視 (predict が chunk[0] を単発 dispatch)。
        replan_family: async_replanning の per-family profile (Issue #128 Phase 4)。
            issue-70 の `FAMILY_REPLANNING_PROFILES` key と一致させる。None なら
            async replan 無効化 (Phase 2 sync 動作、per-tick predict)。GR00T
            task_5+7 (relative_eef) には "groot_relative_eef_v1" 推奨。
        execution_steps: async pipeline が 1 chunk を消費する tick 数 (Phase 4)。
            issue-70 の flip_table 検証値 VALID_EXECUTION_STEPS={5, 10, 20} の
            うち default 10 (333ms cadence @ 30Hz、chunk_len=16 に対して 6 tick
            の replan 予備を確保)。replan_family が None の時は無視される。
        hydra_overrides: RAMEN-Oriの学習時model構造を再現するHydra override。
            architecture variant（G-3 fusion等）だけが指定し、通常variantは空。
    """

    mode: str
    ckpt_ref: str
    checkpoint_subdir: str | None = None
    device: str = "cuda"
    dtype: str = "fp32"
    cams: tuple[CameraKey, ...] = (
        CameraKey.HEAD_LEFT,
        CameraKey.HEAD_RIGHT,
        CameraKey.WRIST_LEFT,
        CameraKey.WRIST_RIGHT,
    )
    temporal_lambda: float | None = -0.1
    replan_family: str | None = None
    execution_steps: int = 10
    hydra_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # mode の有効値制約 (実装が誤 mode で silent fail するのを防ぐ)。
        if self.mode not in ("none", "precomputed_token", "overlay"):
            raise ValueError(
                f"mode must be one of 'none' / 'precomputed_token' / 'overlay', "
                f"got {self.mode!r}"
            )
        if self.dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"dtype must be fp32/bf16/fp16, got {self.dtype!r}")
        if self.checkpoint_subdir is not None:
            subdir = PurePosixPath(self.checkpoint_subdir)
            if (
                not self.checkpoint_subdir
                or subdir.is_absolute()
                or any(part in {"", ".", ".."} for part in subdir.parts)
            ):
                raise ValueError(
                    "checkpoint_subdir must be a non-empty relative POSIX path "
                    f"without traversal, got {self.checkpoint_subdir!r}"
                )
        # temporal_lambda: float なら finite、None なら OK
        if self.temporal_lambda is not None:
            import math as _math
            if not _math.isfinite(float(self.temporal_lambda)):
                raise ValueError(
                    f"temporal_lambda must be finite or None, "
                    f"got {self.temporal_lambda!r}"
                )
        # execution_steps: positive int
        if not isinstance(self.execution_steps, int) or self.execution_steps < 1:
            raise ValueError(
                f"execution_steps must be positive int, got {self.execution_steps!r}"
            )
        # replan_family: str (family key、実 lookup は Gr00tPolicy 側で行う) or None
        if self.replan_family is not None and not isinstance(self.replan_family, str):
            raise TypeError(
                f"replan_family must be str or None, "
                f"got {type(self.replan_family).__name__}"
            )
        if not all(isinstance(value, str) and value for value in self.hydra_overrides):
            raise ValueError("hydra_overrides must contain non-empty strings")


@runtime_checkable
class Policy(Protocol):
    """Learned policy (VLA) の共通 protocol。

    実装は:
        - `from_ckpt(cfg)` classmethod で ckpt を load
        - `warmup(n)` で JIT/cuDNN autotune を済ませる (30Hz loop 前に消化)
        - `predict(obs)` で 1 tick observation → PolicyAction を返す (stateless)
        - `close()` で GPU memory を解放 (long-running orchestrator 想定)

    Stateless 契約: tick 間で内部 buffer を持たない (前 frame image は Observation
    caller 側で保持、state velocity 差分も orchestrator 側で解決)。Policy が唯一
    持って良いのは model weight + preprocessing constant (ImageNet mean/std 等)。
    """

    @staticmethod
    def build_state_from_raw(raw: RawRobotState) -> np.ndarray:
        """Map one canonical robot snapshot to the model-specific state."""
        ...

    @classmethod
    def from_ckpt(cls, cfg: PolicyConfig) -> "Policy": ...

    def warmup(self, n_iter: int = 5) -> None: ...

    def predict(self, obs: Observation) -> PolicyAction: ...

    def close(self) -> None: ...
