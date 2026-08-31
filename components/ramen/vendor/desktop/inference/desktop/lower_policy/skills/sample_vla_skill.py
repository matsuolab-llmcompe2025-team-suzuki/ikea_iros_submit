"""SampleVLASkill: Type B skill の reference 実装 + 実機 mock 兼用 (Issue #75)。

## 目的

Skill.step(obs) → action tensor を返す Type B pattern の骨格を示す。実 VLA model
(GR00T / ACT / diffusion 等) を drop-in する時は **step() 内の action 生成部分を
`self._model.predict(obs)` に差し替えるだけ**で済む structure にしてある。

現状の Mock 実装は「両腕を水平まで開く」固定 pose を返す。実機 verify で:
- 発火した瞬間に arm actuator (rt/lowcmd 200Hz publish) が動き出す
- G1 の両腕が水平位置に到達
- action pipeline (Skill → Orchestrator → actuator_send_fn → G1ArmActuator)
  end-to-end が実 hardware で通ることを目視確認

## Type B contract (real VLA 実装者向け)

- `_on_start(params)`: episode の状態リセット。VLA model の hidden state / history
  buffer 等がある場合はここで clear。params は Orchestrator._build_params の返り値
  (現状 空 dict、将来 task_prompt 等を注入する予定)。
- `step(obs) → np.ndarray`: 毎 tick の action 生成。obs dict は Orchestrator._build_obs
  の返り値:
    - `obs["head_rgb"]`: head camera HWC BGR (numpy) — VLA の image 入力
    - `obs["wrist_left_rgb"]`: FrameData or None — 左手 wrist D405 RGB (Issue #75 で追加、
      real VLA が両手 wrist を使う想定の観測 pipeline)
    - `obs["wrist_right_rgb"]`: FrameData or None — 右手 wrist D405 RGB
    - `obs["joint_state"]`: JointStateData or None — 実 body の 29 joint 位置 (Issue #75 で追加)
    - `obs["cleaned"]`: list[OBBDetection] — YOLO 検出結果 (task grounding に使う場合)
    - `obs["t"]`: frame timestamp (ns)
  返り値は shape `(G1_NUM_ARM_JOINTS,) = (14,)` の numpy array (joint angle [rad])、
  G1JointIndex 15..28 順 = LeftShoulderPitch..RightWristYaw。arm actuator の
  `send_action` に流れる想定。
- `_on_stop()`: teardown。VLA session 終了 / model の GPU state clear 等。

## Fixed pose の受け取り方 (Issue #81 Phase 4)

固定 pose は必ず外部注入 (constructor `fixed_pose` param、default 廃止)。production では
`SampleVLASkill.from_config(cfg, skill_name)` 経由で skill_config.yaml の
`skills.<name>.default_pose_rad` (sparse 記法) から 14-D dense array を densify して渡す。

skill_config.yaml default (move_table_base):
- LeftShoulderRoll = +1.2 rad、RightShoulderRoll = -1.2 rad、他 12 joint = 0
  (setup skill Stage 1 と同じ肩高さで、遷移時に腕が下がらない)

Regular Mode の実機で ±1.0 rad command は肩roll実測 約 ±0.94 rad まで安定追従した
(2026-07-19)。table clearanceのためtargetは±1.2 radへ上げたが、過去の±1.5 rad command
も同程度で飽和したため、実角が上がるかは必ず実機計測で確認する。G1ArmActuator の±1.5 rad
clamp は異常VLA出力への安全上限として残す。
"""

from __future__ import annotations

import numpy as np

from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1_NUM_ARM_JOINTS
from inference.desktop.lower_policy.pose_utils import (
    densify_pose,
    validate_pose_bounds,
)
from inference.desktop.lower_policy.skills.base import Skill


class SampleVLASkill(Skill):
    """Type B skill (VLA drop-in ready) の reference + Mock 実装。

    Args:
        skill_name: dispatcher の registry key と一致する必要あり (e.g., "pick_table_leg")。
            MockSkill と同 pattern で instance 毎に指定。
        fixed_pose: step で返す action tensor (shape (14,))。**required** (Issue #81
            Phase 4 で default 廃止)。production は `from_config` 経由で YAML から
            注入する。直接 constructor を呼ぶ場合も明示的な pose を渡す必要あり。

    G1_ARM_JOINT_INDICES = 15..28 の順で 14 要素:
      0:LeftShoulderPitch  1:LeftShoulderRoll   2:LeftShoulderYaw
      3:LeftElbow          4:LeftWristRoll      5:LeftWristPitch    6:LeftWristYaw
      7:RightShoulderPitch 8:RightShoulderRoll  9:RightShoulderYaw
      10:RightElbow        11:RightWristRoll    12:RightWristPitch  13:RightWristYaw
    """

    def __init__(
        self,
        skill_name: str,
        fixed_pose: np.ndarray,
    ) -> None:
        super().__init__()
        self.name = skill_name
        pose = np.asarray(fixed_pose, dtype=np.float64)
        if pose.shape != (G1_NUM_ARM_JOINTS,):
            raise ValueError(
                f"fixed_pose must have shape ({G1_NUM_ARM_JOINTS},), got {pose.shape}"
            )
        self._pose = pose.copy()

    def _on_start(self, params: dict) -> None:
        # 実 VLA 実装時: self._model.reset() 相当をここに。
        # Mock では state を持たないので no-op。
        pass

    def _on_stop(self) -> None:
        # 実 VLA 実装時: self._model.close() / session teardown 等。
        # Mock は no-op (arm actuator の publish loop は entrypoint 側 lifecycle)。
        pass

    def step(self, obs: dict) -> np.ndarray:
        """per-tick action tensor を返す。

        Mock 実装は obs を無視して固定 pose を返す。実 VLA 実装時は:

            return self._model.predict(
                head_rgb=obs["head_rgb"],
                joint_state=obs.get("joint_state"),
                task_context=obs.get("cleaned"),
            )

        に差し替える (VLA model の interface に合わせて adapt)。
        """
        return self._pose.copy()

    @classmethod
    def from_config(cls, cfg: dict, skill_name: str) -> "SampleVLASkill":
        """dict (skill_config.yaml の `skills.<skill_name>` セクション) から構築する。

        期待する構造 (Issue #81 Phase 2b):

            {"default_pose_rad": {joint: value, ...}}  # sparse 記法

        `default_pose_rad` 未指定なら 14-D ゼロ姿勢 (VLA test 用の neutral)。
        """
        if not isinstance(cfg, dict):
            raise ValueError(
                f"{skill_name} config: must be a mapping, got {type(cfg).__name__}"
            )
        raw_pose = cfg.get("default_pose_rad", {})
        pose = densify_pose(raw_pose, context=f"skill {skill_name!r}")
        # arm actuator の ±1.5 rad clamp と揃えて、YAML で無茶な値が書かれた時に
        # silent 切り詰めではなく明示 error にする (SetupStage と対称、review #82)。
        validate_pose_bounds(pose, context=f"skill {skill_name!r}")
        return cls(skill_name, fixed_pose=pose)
