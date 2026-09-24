"""Per-skill motion envelope (velocity / acceleration limits) for VlaSkill dispatch.

Ported from issue-70-flip-table-data-augmentation branch (Phase 3、Issue #128)。
issue-70 の `inference/desktop/upper_policy/motion_limits.py` は flip_table 用の
per-skill 定数のみ持つ module だったが、我々は **skill_config.yaml で全 skill
分を data として管理** する方針 (user 判断)。本 module は dataclass schema と
default 値のみ提供、実値は YAML から流入。

# 役割分離 (Layer 1 vs Layer 2、Phase 3 kickoff で user 承認)

- **Layer 1 (本 MotionLimits + MotionLimiter、VlaSkill.step で per-tick 30Hz)**:
  task 依存の trajectory smoothing envelope。skill 別に「chunk 内での physically
  reachable な変化量」を規定 (arm 2 rad/s とか、insert なら 1 rad/s とか)。
- **Layer 2 (既存 ArmSafetyLimits + `_rate_limited_target`、Actuator publish 250Hz)**:
  hardware absolute cap + 30Hz→250Hz interp。全 code path (VLA / SetupSkill /
  MoveToTable / smoke script) が publish 経由で通る。
- 2 段は共存 (multi-defense、Layer 1 は task envelope、Layer 2 は hardware max
  + motor-lag tracker、目的が異なる)。詳細は Issue #128 の Phase 3 kickoff メモ。

# YAML 契約

`skill_config.yaml` に以下 2 セクション:

```yaml
motion_limits_default:                    # 未指定 skill の fallback
  arm_velocity_rad_s: 2.0
  arm_acceleration_rad_s2: 10.0
  hand_velocity_rad_s: 6.0                # Dex1 physical rad [0, 5.4]
  hand_acceleration_rad_s2: 30.0

skills:
  <skill_name>:
    motion_limits:                        # 明示 override 時のみ present
      arm_velocity_rad_s: 1.5
      arm_acceleration_rad_s2: 10.0
      hand_velocity_rad_s: 6.0
      hand_acceleration_rad_s2: 30.0
```

Waist は arm と同 limit を共有 (upper-body 一括扱い)。将来 waist 独立が要れば
schema 拡張。

# 単位注意

- **arm/waist**: G1 joint physical rad (rad/s、rad/s²)
- **hand**: Dex1 physical rad (open=4.5、closed=0.0)、Layer 2 の HAND_GRIP_MAX=5.4
  で最終 clamp。**fraction (0-1) ではない** (issue-70 の
  `hand_velocity_fraction_s` は fraction 単位、我々は physical rad で統一)。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class MotionLimits:
    """Per-skill upper-body target smoothing envelope.

    Attributes:
        arm_velocity_rad_s: arm 14 joint (=waist 3 joint も同値) の max velocity
            [rad/s]。issue-70 検証値 = flip_table で 1.5 rad/s (dataset-derived)。
            範囲: 0.5-20.0 (0.5 未満だと trajectory 歪む、20.0 超は Layer 2
            の hardware cap で clamp される)。
        arm_acceleration_rad_s2: 同上 joint の max acceleration [rad/s²]。
            issue-70 flip_table 検証値 = 10.0 rad/s²。
        hand_velocity_rad_s: Dex1 grip max velocity [rad/s]、physical rad
            [0, 5.4] range 上での変化速度。6.0 で full close-to-open ~0.9s。
        hand_acceleration_rad_s2: 同 Dex1 max acceleration [rad/s²]。
    """

    arm_velocity_rad_s: float
    arm_acceleration_rad_s2: float
    hand_velocity_rad_s: float
    hand_acceleration_rad_s2: float

    def __post_init__(self) -> None:
        for name in (
            "arm_velocity_rad_s",
            "arm_acceleration_rad_s2",
            "hand_velocity_rad_s",
            "hand_acceleration_rad_s2",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be number, got {type(value).__name__}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")

    @classmethod
    def from_config_dict(cls, cfg: Mapping) -> "MotionLimits":
        """YAML dict → MotionLimits (missing key は KeyError、typo 検出)。"""
        required = (
            "arm_velocity_rad_s",
            "arm_acceleration_rad_s2",
            "hand_velocity_rad_s",
            "hand_acceleration_rad_s2",
        )
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(
                f"motion_limits config missing keys: {missing}; got keys "
                f"{sorted(cfg.keys())}"
            )
        return cls(
            arm_velocity_rad_s=float(cfg["arm_velocity_rad_s"]),
            arm_acceleration_rad_s2=float(cfg["arm_acceleration_rad_s2"]),
            hand_velocity_rad_s=float(cfg["hand_velocity_rad_s"]),
            hand_acceleration_rad_s2=float(cfg["hand_acceleration_rad_s2"]),
        )


# Module default = skill_config.yaml が完全に欠けている場合の hard fallback
# (test / dev 用途、production では YAML 経由の値を使う)。issue-70 flip_table
# より緩めた中庸値 (未 verified skill 用の保守側)。
DEFAULT_MOTION_LIMITS: MotionLimits = MotionLimits(
    arm_velocity_rad_s=2.0,
    arm_acceleration_rad_s2=10.0,
    hand_velocity_rad_s=6.0,
    hand_acceleration_rad_s2=30.0,
)


def load_motion_limits_for_skill(
    skill_config: Mapping,
    skill_name: str,
) -> MotionLimits:
    """skill_config.yaml から指定 skill の MotionLimits を解決。

    Resolution:
        1. base = `motion_limits_default` if present else `DEFAULT_MOTION_LIMITS` の
           4 field
        2. `skills.<skill_name>.motion_limits` があれば **base に merge** (partial
           override 許容、指定した key のみ上書き、未指定 key は base 値継承)
        3. 全 4 field 確定 → MotionLimits 検証済 instance

    Partial override が便利: e.g. flip_table は arm_velocity_rad_s だけ 1.5 に
    したいので `motion_limits: {arm_velocity_rad_s: 1.5}` と書けば残り 3 field
    は default から継承される。

    Args:
        skill_config: yaml.safe_load(skill_config.yaml) の結果 (top-level dict)。
        skill_name: `skills.<name>` の key。

    Returns:
        解決済 MotionLimits (frozen)。
    """
    # Step 1: base 4 field を決定 (motion_limits_default > DEFAULT_MOTION_LIMITS)
    default_cfg = skill_config.get("motion_limits_default")
    if isinstance(default_cfg, dict):
        base = MotionLimits.from_config_dict(default_cfg)
    else:
        base = DEFAULT_MOTION_LIMITS
    # Step 2: per-skill override があれば merge
    skills = skill_config.get("skills", {})
    skill_cfg = skills.get(skill_name, {}) if isinstance(skills, dict) else {}
    if not (isinstance(skill_cfg, dict) and "motion_limits" in skill_cfg):
        return base
    override = skill_cfg["motion_limits"]
    if not isinstance(override, dict):
        raise TypeError(
            f"skills.{skill_name}.motion_limits must be a mapping, "
            f"got {type(override).__name__}"
        )
    # Merge: base 4 field を dict 化して override key で上書き、typo (未知 key)
    # は from_config_dict の required check で捕まる (MotionLimits の field 群と
    # 一致必須、Python dict の setdefault 経由で余剰 key が生きて残る場合は
    # allowed unknown key を後で potentially 検出できる)
    allowed = {
        "arm_velocity_rad_s",
        "arm_acceleration_rad_s2",
        "hand_velocity_rad_s",
        "hand_acceleration_rad_s2",
    }
    unknown = set(override) - allowed
    if unknown:
        raise KeyError(
            f"skills.{skill_name}.motion_limits has unknown keys: {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )
    merged = {
        "arm_velocity_rad_s": base.arm_velocity_rad_s,
        "arm_acceleration_rad_s2": base.arm_acceleration_rad_s2,
        "hand_velocity_rad_s": base.hand_velocity_rad_s,
        "hand_acceleration_rad_s2": base.hand_acceleration_rad_s2,
    }
    merged.update(override)
    return MotionLimits.from_config_dict(merged)
