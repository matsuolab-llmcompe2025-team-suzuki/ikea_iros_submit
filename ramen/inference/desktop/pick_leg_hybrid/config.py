"""`configs/pick_leg_hybrid.yaml` の読み込み (Issue #136)。

CLAUDE.md の「Skill numerics live in YAML」規約に従い、数値は YAML 側にある。
本 module は **dict → dataclass の変換だけ**を行う。Python 側に既定の数値を
持たせない (dataclass の default はあくまで型の既定であって、運用値は YAML)。

`yaml` は default env に無い場合があるので import は `load_yaml` の中に閉じる。
dict から組む `from_dict` は yaml 非依存なので、test はそちらを使える。

YAML は必ず `yaml.safe_load` で読む。`yaml.load` は `!!python/object` タグで
任意コード実行が可能 (CLAUDE.md)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from inference.desktop.pick_leg_hybrid.boundary import BoundaryConfig
from inference.desktop.pick_leg_hybrid.interlock import (
    GraspInterlockConfig,
    UnmeasuredHandConfig,
)
from inference.desktop.pick_leg_hybrid.phase2 import EePose, Phase2Config
from inference.desktop.pick_leg_hybrid.phase3_rule import (
    Phase3RuleConfig,
    Phase3RuleStage,
    RuleStageTiming,
)
from inference.desktop.pick_leg_hybrid.vlm import VlmConfig

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "pick_leg_hybrid.yaml"
)

#: `phase2.ee_frame` に許す値。どちらの FK 定義で目標値を書いたかを表す (D-3)。
VALID_EE_FRAMES = ("g1_wrist_fk", "dataset_ee_state")


@dataclass(frozen=True)
class OverlayConfig:
    """VLM に渡す前の OBB overlay 描画設定 (D-7)。"""

    enabled: bool = True
    class_ids: tuple[int, ...] = (1, 5, 6)   # leg / hand_right / hand_left
    conf_threshold: float = 0.30
    line_thickness: int = 2


@dataclass(frozen=True)
class ReferenceConfig:
    """参照画像 (overlay 済み) の path。未設定なら None。"""

    before: Optional[str] = None
    after: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """2 枚とも設定されているか。"""
        return bool(self.before) and bool(self.after)


@dataclass(frozen=True)
class RuntimeConfig:
    """Production-orchestrator limits for the complete hybrid skill."""

    hard_timeout_sec: float
    phase1_timeout_sec: float
    phase2_timeout_sec: float

    def __post_init__(self) -> None:
        if self.hard_timeout_sec <= 0:
            raise ValueError("runtime.hard_timeout_sec must be > 0")
        if min(
            self.phase1_timeout_sec,
            self.phase2_timeout_sec,
        ) <= 0:
            raise ValueError("runtime phase timeouts must be > 0")
        if self.hard_timeout_sec <= self.phase1_timeout_sec + self.phase2_timeout_sec:
            raise ValueError(
                "runtime.hard_timeout_sec must leave time for phase 3 after "
                "the phase 1 and phase 2 deadlines"
            )


@dataclass(frozen=True)
class PickLegHybridConfig:
    """pick_leg_hybrid 全体の設定。"""

    interlock: GraspInterlockConfig
    boundary: BoundaryConfig
    vlm: VlmConfig
    overlay: OverlayConfig
    references: ReferenceConfig
    phase2: Phase2Config
    runtime: RuntimeConfig
    phase3_rule_based: Phase3RuleConfig | None = None
    # 手の実測が無い経路の縮退動作。None なら合成 state では pick を進めない。
    unmeasured_hand: UnmeasuredHandConfig | None = None

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "PickLegHybridConfig":
        """YAML 相当の dict から組む (yaml 非依存)。

        Raises:
            ValueError: 必須項目の欠落、または値が不正。
        """
        p2 = _require(raw, "phase2")
        ee_frame = str(p2.get("ee_frame", "g1_wrist_fk"))
        if ee_frame not in VALID_EE_FRAMES:
            raise ValueError(
                f"phase2.ee_frame must be one of {VALID_EE_FRAMES}, got {ee_frame!r}"
            )
        ov = raw.get("overlay") or {}
        ref = raw.get("references") or {}
        runtime = _require(raw, "runtime")
        phase3_raw = raw.get("phase3_rule_based")
        unmeasured_raw = raw.get("unmeasured_hand")
        return PickLegHybridConfig(
            interlock=GraspInterlockConfig(**(raw.get("interlock") or {})),
            boundary=BoundaryConfig(**(raw.get("boundary") or {})),
            vlm=VlmConfig(**(raw.get("vlm") or {})),
            overlay=OverlayConfig(
                enabled=bool(ov.get("enabled", True)),
                class_ids=tuple(int(c) for c in ov.get("class_ids", (1, 5, 6))),
                conf_threshold=float(ov.get("conf_threshold", 0.30)),
                line_thickness=int(ov.get("line_thickness", 2)),
            ),
            references=ReferenceConfig(
                before=ref.get("before"), after=ref.get("after")
            ),
            runtime=RuntimeConfig(
                hard_timeout_sec=float(runtime["hard_timeout_sec"]),
                phase1_timeout_sec=float(runtime["phase1_timeout_sec"]),
                phase2_timeout_sec=float(runtime["phase2_timeout_sec"]),
            ),
            phase2=Phase2Config(
                goal_left=_pose(_require(p2, "goal_left"), "phase2.goal_left"),
                goal_right=_pose(_require(p2, "goal_right"), "phase2.goal_right"),
                duration_sec=float(p2.get("duration_sec", 2.7)),
                pos_tol=float(p2.get("pos_tol", 0.02)),
                rot_tol=float(p2.get("rot_tol", 0.12)),
                hold_hand_value=float(p2.get("hold_hand_value", 2.2)),
                hold_hand_left_value=(
                    None
                    if p2.get("hold_hand_left_value") is None
                    else float(p2["hold_hand_left_value"])
                ),
                hold_hand_right_value=(
                    None
                    if p2.get("hold_hand_right_value") is None
                    else float(p2["hold_hand_right_value"])
                ),
                ee_frame=ee_frame,
            ),
            unmeasured_hand=(
                None
                if unmeasured_raw is None
                else _unmeasured_hand_config(unmeasured_raw)
            ),
            phase3_rule_based=(
                None
                if phase3_raw is None
                else _phase3_rule_config(phase3_raw)
            ),
        )


def load_yaml(path: str | Path = DEFAULT_CONFIG_PATH) -> PickLegHybridConfig:
    """YAML file から設定を読む。

    Args:
        path: YAML file の path。省略時は同梱の既定 config。

    Returns:
        `PickLegHybridConfig`。

    Raises:
        FileNotFoundError: file が無い。
        ValueError: 内容が不正。
    """
    # lazy: yaml は default env に無い場合がある
    import yaml

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"pick_leg_hybrid config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    return PickLegHybridConfig.from_dict(raw)


def _unmeasured_hand_config(raw: Any) -> UnmeasuredHandConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("unmeasured_hand must be a mapping")
    keys = (
        "phase1_command_settle_sec",
        "phase3_grasp_command_rad",
        "phase3_hand_settle_sec",
    )
    missing = [key for key in keys if key not in raw]
    unknown = sorted(set(raw) - set(keys))
    if missing or unknown:
        raise ValueError(
            f"unmeasured_hand keys: missing={missing} unknown={unknown}"
        )
    return UnmeasuredHandConfig(**{key: float(raw[key]) for key in keys})


def _require(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config is missing the {key!r} section")
    return value


def _pose(raw: Mapping[str, Any], label: str) -> EePose:
    if "pos" not in raw or "quat" not in raw:
        raise ValueError(f"{label} must have both 'pos' and 'quat'")
    return EePose.of(raw["pos"], raw["quat"])


def _phase3_rule_config(raw: Any) -> Phase3RuleConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("phase3_rule_based must be a mapping")
    timings_raw = _require(raw, "timings")
    timing_stages = tuple(Phase3RuleStage)[:-1]
    timings: dict[Phase3RuleStage, RuleStageTiming] = {}
    for stage in timing_stages:
        item = timings_raw.get(stage.value)
        if not isinstance(item, Mapping):
            raise ValueError(
                f"phase3_rule_based.timings.{stage.value} must be a mapping"
            )
        timings[stage] = RuleStageTiming(
            duration_sec=float(item["duration_sec"]),
            timeout_sec=float(item["timeout_sec"]),
        )
    regrasp = _require(raw, "right_regrasp")
    if "offset_left_local" not in regrasp or "quat" not in regrasp:
        raise ValueError(
            "phase3_rule_based.right_regrasp needs offset_left_local and quat"
        )
    return Phase3RuleConfig(
        left_present=_pose(
            _require(raw, "left_present"), "phase3_rule_based.left_present"
        ),
        right_regrasp_offset_left_local=np.asarray(
            regrasp["offset_left_local"], dtype=np.float64
        ),
        right_regrasp_quat=np.asarray(regrasp["quat"], dtype=np.float64),
        close_hand_value=float(raw["close_hand_value"]),
        open_hand_value=float(raw["open_hand_value"]),
        right_initial_hold_value=float(raw["right_initial_hold_value"]),
        grasp_preload_rad=float(raw["grasp_preload_rad"]),
        open_min=float(raw["open_min"]),
        confirm_count=int(raw["confirm_count"]),
        pos_tol=float(raw["pos_tol"]),
        rot_tol=float(raw["rot_tol"]),
        left_present_pos_tol=float(raw.get("left_present_pos_tol", raw["pos_tol"])),
        left_present_rot_tol=float(raw.get("left_present_rot_tol", raw["rot_tol"])),
        joint_tol=float(raw["joint_tol"]),
        joint_velocity_tol=float(raw["joint_velocity_tol"]),
        joint_command_tol=float(raw["joint_command_tol"]),
        hand_tol=float(raw["hand_tol"]),
        timings=timings,
        boundary_joint_guard_rad=float(raw.get("boundary_joint_guard_rad", 0.40)),
        grasp_confirm_sec=float(raw.get("grasp_confirm_sec", 0.5)),
    )
