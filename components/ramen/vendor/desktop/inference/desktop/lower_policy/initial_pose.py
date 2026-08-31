"""Skill 初期姿勢を skill_config.yaml から読み出す helper。

skill_config.yaml:skills.<skill_name>.initial_pose の schema:
    arm_position_rad:
      L.shoulder_pitch: <float>
      ...   # 14 joints, ARM_JOINT_ORDER 順
    dex1_opening_fraction:
      left:  <float [0,1]>
      right: <float [0,1]>
    dataset_repo_id: <str>          # provenance
    dataset_revision: <str>         # 40-hex commit SHA
    training_episode_count: <int>
    statistic: <str>
    requires_separate_hand_initialization: <bool>  # optional, default false
    hand_initialization_instruction: <str>          # optional
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


ARM_JOINT_ORDER: tuple[str, ...] = (
    "L.shoulder_pitch", "L.shoulder_roll", "L.shoulder_yaw",
    "L.elbow", "L.wrist_roll", "L.wrist_pitch", "L.wrist_yaw",
    "R.shoulder_pitch", "R.shoulder_roll", "R.shoulder_yaw",
    "R.elbow", "R.wrist_roll", "R.wrist_pitch", "R.wrist_yaw",
)
# Dataset `hand_cmd` values were normalized against 4.5 rad.  The actuator's
# 5.4-rad mechanical full-open value is used only for collision clearance and
# safe return; frame-zero dataset targets must use the recorded 4.5-rad basis.
DEX1_MAX_RAD = 4.5


@dataclass(frozen=True)
class SkillInitialPose:
    """A skill の frame-0 初期姿勢 (arm 14D + Dex1 opening fraction)。

    Attributes:
        arm_position_rad: (14,) float64、ARM_JOINT_ORDER 順、rad。
        dex1_opening_fraction: (left, right) [0, 1] fraction。opening 全開=1.0、閉=0.0。
        requires_separate_hand_initialization: True なら operator が事前に leg 等
            を hand に配置してから起動。runner が Enter gate + hand rad 送信を扱う。
        hand_initialization_instruction: operator 向け英文 (表示用)。
        dataset_repo_id / dataset_revision / training_episode_count / statistic:
            provenance (log にそのまま流す)。
    """

    arm_position_rad: np.ndarray
    dex1_opening_fraction: tuple[float, float]
    requires_separate_hand_initialization: bool
    hand_initialization_instruction: str
    dataset_repo_id: str
    dataset_revision: str
    training_episode_count: int
    statistic: str

    @property
    def dex1_target_rad(self) -> tuple[float, float]:
        """Dataset opening fraction → recorded Dex1 target rad [0, 4.5]."""
        left, right = self.dex1_opening_fraction
        return (float(left) * DEX1_MAX_RAD, float(right) * DEX1_MAX_RAD)


def load_initial_pose(skill_config_path: Path, skill_name: str) -> SkillInitialPose:
    """skill_config.yaml から skill_name の initial_pose を SkillInitialPose に組み立てる。

    Raises:
        FileNotFoundError: skill_config.yaml が無い
        KeyError: skill_name section or initial_pose が無い / 不完全
        ValueError: 14D arm shape mismatch / dex1 fraction 範囲外
    """
    with open(skill_config_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg or "skills" not in cfg:
        raise KeyError(f"skill_config.yaml missing 'skills' section: {skill_config_path}")
    skill_cfg = cfg["skills"].get(skill_name)
    if skill_cfg is None:
        raise KeyError(f"skill {skill_name!r} not registered in {skill_config_path}")
    ip_cfg = skill_cfg.get("initial_pose")
    if ip_cfg is None:
        raise KeyError(
            f"skill {skill_name!r} has no initial_pose section in {skill_config_path}. "
            f"Skills without an explicit initial_pose (e.g. move_table_base which "
            f"uses setup Stage 2) are not directly runnable via evaluate/model_evaluation/."
        )
    arm_cfg = ip_cfg.get("arm_position_rad")
    if not isinstance(arm_cfg, dict) or set(arm_cfg) != set(ARM_JOINT_ORDER):
        raise ValueError(
            f"arm_position_rad must specify exactly the 14 keys in ARM_JOINT_ORDER, "
            f"got {sorted(arm_cfg.keys()) if isinstance(arm_cfg, dict) else type(arm_cfg)}"
        )
    arm = np.asarray(
        [float(arm_cfg[name]) for name in ARM_JOINT_ORDER], dtype=np.float64
    )
    if not np.isfinite(arm).all():
        raise ValueError(f"arm_position_rad must be finite, got {arm}")

    dex1_cfg = ip_cfg.get("dex1_opening_fraction")
    if (
        not isinstance(dex1_cfg, dict)
        or "left" not in dex1_cfg
        or "right" not in dex1_cfg
    ):
        raise ValueError("dex1_opening_fraction must have 'left' and 'right' keys")
    dex1 = (float(dex1_cfg["left"]), float(dex1_cfg["right"]))
    if not all(0.0 <= v <= 1.0 for v in dex1):
        raise ValueError(f"dex1_opening_fraction must be in [0, 1], got {dex1}")

    return SkillInitialPose(
        arm_position_rad=arm,
        dex1_opening_fraction=dex1,
        requires_separate_hand_initialization=bool(
            ip_cfg.get("requires_separate_hand_initialization", False)
        ),
        hand_initialization_instruction=str(
            ip_cfg.get("hand_initialization_instruction", "")
        ),
        dataset_repo_id=str(ip_cfg.get("dataset_repo_id", "")),
        dataset_revision=str(ip_cfg.get("dataset_revision", "")),
        training_episode_count=int(ip_cfg.get("training_episode_count", 0)),
        statistic=str(ip_cfg.get("statistic", "")),
    )
