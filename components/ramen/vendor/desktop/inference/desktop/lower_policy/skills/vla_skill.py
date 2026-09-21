"""VlaSkill: learned policy を Skill として wrap する per-skill_name subclass の親 (Issue #125 Phase 9a)。

# 位置付け

- Skill (rule-based walk / arm pose etc.) と兄弟の "learned skill wrapper"
- 既存 `SkillDispatchLowerPolicy` の registry に登録される (skill_name → VlaSkill
  subclass instance) → dispatcher が start/stop/step を呼ぶ
- Type B pattern: `step(obs) → arm 14D ndarray` を返す (既存 pattern 準拠)、
  waist(3) と hand(2) は内部 dispatch (constructor-injected actuator へ)

# per-skill_name identity hardcode (user-approved 設計、Issue #125)

各 skill_name (`move_table_base`, `rotate_table_base` 等) ごとに VlaSkill を
subclass し、class attribute で自分の identity を hardcode:
- `SKILL_ID`: RAMEN-Ori 用 conditioning (skill_mapping.py canonical、None = 未対応)
- `LANGUAGE`: GR00T 用 conditioning (subtask_training.json:subtasks[X].task)

Policy 側は skill_name を知らない (backbone-focused)。VlaSkill が Observation に
skill_id / language を埋めて Policy.predict() を叩く。GR00T ↔ RAMEN-Ori swap は
Skill constructor の `policy` を差し替えるだけ。

# Action dispatch (19D → 3 actuator)

Policy が predict する chunk (16, 19) の current step を per-actuator に slice:
- waist (3D)     → self._waist_actuator.send_action  [0:3]
- arms  (14D)    → return (Skill.step の戻り値、caller = dispatcher が arm_actuator に流す) [3:17]
- hand  (2D)     → self._hand_actuator.send_action  [17:19]

既定は従来どおりper-tickに再推論してchunk[0]だけを使う。Policy classが
`EXECUTION_HORIZON > 1`を明示した場合に限り、decode済みの絶対action chunkを
順番に消費する。これによりRAMEN-Oriの既存挙動を変えず、GR00Tの公式processorが
現在state基準でdecodeしたchunkを捨てずに実行できる。

# Observation build (orchestrator obs dict → policies.base.Observation)

orchestrator の `obs` dict (SampleVLASkill docstring 参照):
    obs["head_rgb"], obs["wrist_left_rgb"], obs["wrist_right_rgb"],
    obs["joint_state"], obs["cleaned"] (list[OBBDetection]), obs["t"]

を policies.base.Observation に変換する。head は packed stereo (480x1280) なら
split、raw robot state は RawRobotState に集約。obs["cleaned"] は現状 flat list
(orchestrator 側で per-cam 未分離) なので暫定的に HEAD_LEFT に全 det を割り当てる
(orchestrator 側の per-cam YOLO 対応が来たら dict[Cam, list] 直接受け取りに切替)。
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from inference.desktop.lower_policy.actuators.hand import HandActuator
from inference.desktop.lower_policy.actuators.waist import WaistActuator
from inference.desktop.lower_policy.policies.base import (
    CameraKey,
    Observation,
    PolicyAction,
    RawRobotState,
)
from inference.desktop.lower_policy.policies.ramen_ori import split_head_stereo
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.perception.frame_source import adapt_wrist_to_training_shape
from inference.desktop.lower_policy.skills.rotate_retry import (
    ReturnMode,
    RetryPhase,
    arm_return_complete,
)
from inference.desktop.perception.g1_urdf_fk import G1WristFK


# 19D action layout (subtask_training.json:action.names 順に一致):
#   [0:3]   = waist  (yaw, roll, pitch)
#   [3:10]  = left_arm  (7 joints)
#   [10:17] = right_arm (7 joints)
#   [17:19] = hand      (left_gripper_q, right_gripper_q)
ACTION_DIM_TOTAL: int = 19
WAIST_SLICE = slice(0, 3)
ARMS_SLICE = slice(3, 17)   # 14D (arm7 + arm7)
HAND_SLICE = slice(17, 19)  # 2D


class VlaSkill(Skill):
    """Learned policy を Skill として wrap する parent class。

    Subclass が per-skill_name identity (SKILL_ID / LANGUAGE) を class attribute
    で hardcode し、`name` は Skill base の慣例通り class attribute で指定。

    Attributes:
        policy: `Policy` protocol 実装 (Gr00tPolicy or RamenOriPolicy)。
        arm_actuator は Skill.step の戻り値経由で caller (dispatcher) が扱う。
        waist_actuator / hand_actuator は DI で受け取り、step 内で dispatch。
    """

    # ---- subclass override 対象 (class attribute で hardcode) ---- #
    name: str = "_vla_base"
    SKILL_ID: int | None = None
    LANGUAGE: str | None = None

    def __init__(
        self,
        policy: Any,  # Policy (Protocol) 実装、Gr00tPolicy or RamenOriPolicy
        waist_actuator: WaistActuator,
        hand_actuator: HandActuator,
        frame_buffer_maxlen: int = 2,  # 現/前 frame 保持
        fk: G1WristFK | None = None,
        dispatch_waist: bool = True,
        motion_limiter: Any = None,  # Phase 3: MotionLimiter (optional、None なら bypass)
        language_override: str | None = None,  # Issue #132: per-variant prompt swap
        skill_id_override: int | None = None,  # Issue #141: per-variant skill_id swap
        progress_monitor: Any = None,  # Issue #137: RotateProgressMonitor (optional)
        teacher_range: Any = None,  # Issue #141: TeacherJointRange (optional)
        z_ceiling: Any = None,  # Issue #137: LeftHandZCeiling (optional)
        retry_controller: Any = None,  # Issue #137: RotateRetryController (optional)
        retry_options: dict | None = None,  # return_mode / tolerance 等
    ) -> None:
        super().__init__()
        if self.SKILL_ID is None and self.LANGUAGE is None:
            raise ValueError(
                f"{type(self).__name__}: at least one of SKILL_ID or LANGUAGE must "
                f"be set as class attribute (per-skill identity hardcode)"
            )
        # Issue #132 Phase A: policy_config.yaml の language_prompt がここに渡ってきた
        # 場合に class-level LANGUAGE を上書きする。未指定 (None) なら class hard-code
        # (現状動作) をそのまま使う。空文字は None と同義にはせず即 error にする
        # (指定した意図と誤解釈される trap を避けるため、PolicyConfig 側でも同 guard)。
        if language_override is not None:
            if not isinstance(language_override, str) or not language_override:
                raise ValueError(
                    f"{type(self).__name__}: language_override must be non-empty str "
                    f"or None, got {language_override!r}"
                )
        self._language_override = language_override
        # Issue #141 (1-2 / INF-3): ckpt が学習した skill_id は class-level の値と
        # 一致しないことがある (Phase K の rotate は 0 で学習、class は 4)。slot 側で
        # 明示できるようにして、古い ckpt もそのまま評価できるようにする。
        if skill_id_override is not None:
            if isinstance(skill_id_override, bool) or not isinstance(skill_id_override, int):
                raise ValueError(
                    f"{type(self).__name__}: skill_id_override must be int or None, "
                    f"got {skill_id_override!r}"
                )
            if skill_id_override < 0:
                raise ValueError(
                    f"{type(self).__name__}: skill_id_override must be >= 0, "
                    f"got {skill_id_override}"
                )
        self._skill_id_override = skill_id_override
        self._policy = policy
        self._waist_actuator = waist_actuator
        self._hand_actuator = hand_actuator
        # False = Model waist chunk[0:3] を actuator に流さない。腰暴走で脚 balance
        # 制御が変な方向に走るリスクを避けたい skill / variant で使う (skill_config.yaml
        # の dispatch_waist: false で有効化)。Model 出力は捨てられ、arm_actuator は
        # waist target 未受信のまま hold pose を lowcmd に載せ続ける。
        self._dispatch_waist = bool(dispatch_waist)
        # G1 URDF FK: joint_positions → ee_state (12D) を root frame で計算 (Phase A-5)。
        # None 渡された場合は ee_state = zeros (dev / test 経路、Cartesian signal 無し)。
        self._fk = fk
        # Phase 3 (Issue #128): MotionLimiter (Layer 1、per-tick 30Hz smoothing)。
        # None = bypass (test / dev、Actuator 側 Layer 2 でのみ制限)。実運用では
        # entrypoint / runner が skill_config から MotionLimits 読んで注入する。
        # 詳細は inference/desktop/lower_policy/skills/motion_limiter.py の docstring。
        self._motion_limiter = motion_limiter
        # Issue #137 Phase 2: 天板の回転進捗 monitor (skills/rotate_progress.py)。
        # None = 何もしない (既定)。渡された場合も本 Phase では metadata に出すだけで
        # 動作は変えない。retry FSM (Phase 4) がこの判定を消費する。
        self._progress_monitor = progress_monitor
        # Issue #137 Phase 3: 左手先 z の片側天井 (skills/z_ceiling.py)。
        # None = 何もしない (既定)。天井値そのものは per-attempt に変わるので
        # instance ではなく `self._z_ceiling_target_m` が持つ (retry FSM が更新)。
        self._z_ceiling = z_ceiling
        self._z_ceiling_target_m: float | None = None
        # Issue #141 (1-4): 教師データの関節範囲を出た腕関節だけを内側へ戻す
        # (`skills/teacher_range.py`)。None なら補正なし = 従来動作。
        self._teacher_range = teacher_range
        # Issue #137 Phase 4: 空振り retry FSM。None = 何もしない (既定)。
        # 成功しても skill は終了させない — 遷移は上位 policy
        # (`enter_pick_table_leg`) が持っているので責務を二重化しない。
        self._retry = retry_controller
        self._retry_options = dict(retry_options or {})
        self._retry_phase: Any = None
        self._attempt_start_z_m: float | None = None
        self._hold_target_19d: np.ndarray | None = None
        # Frame buffer for images_prev (RAMEN-Ori Temporal encoder 用)
        self._frames_prev: dict[CameraKey, np.ndarray] | None = None
        # State buffer: RAMEN-Ori 71D の velocity/tracking_err slice で使う (Phase A-4)
        self._prev_joint_positions: np.ndarray | None = None  # (29,)、前 tick JointState
        self._prev_hand_state: np.ndarray | None = None       # (2,)、前 tick Dex1 rad
        self._prev_action_19d: np.ndarray | None = None       # (19,)、前 tick 送出 action[0]
        # 診断用: 直近 PolicyAction を保持 (latency / metadata 参照用)
        self.last_action: PolicyAction | None = None
        self._action_queue: list[np.ndarray] = []
        self._action_queue_next_index = 0

    def _on_start(self, params: dict) -> None:
        """episode 開始時: frame buffer + state buffer reset + policy/limiter reset。"""
        self._frames_prev = None
        self._prev_joint_positions = None
        self._prev_hand_state = None
        self._prev_action_19d = None
        self.last_action = None
        # Phase 2 (Issue #128): Gr00tPolicy は temporal ensemble state
        # (candidate buffer + step counter) を持つ。skill 遷移時に前 skill の
        # chunk が新 skill の blend に混入しないよう reset。protocol は
        # optional (Policy base に reset は必須ではない、持つ実装のみ呼ぶ)。
        reset_fn = getattr(self._policy, "reset", None)
        if callable(reset_fn):
            reset_fn()
        # Phase 3 (Issue #128): MotionLimiter は _previous_target / _previous_velocity
        # を保持する。skill 遷移時に前 skill の trajectory が残ると新 skill の初回
        # apply で意図しない velocity 制限がかかる → reset で measured 基準に戻す。
        if self._motion_limiter is not None:
            self._motion_limiter.reset()
        # Issue #137: 参照姿勢は attempt 境界で張り直す。episode 開始も境界。
        if self._progress_monitor is not None:
            self._progress_monitor.reset()
        if self._retry is not None:
            self._retry.reset()
        self._retry_phase = None
        self._attempt_start_z_m = None
        self._hold_target_19d = None
        self._action_queue.clear()
        self._action_queue_next_index = 0

    def _on_stop(self) -> None:
        """episode 終了時: buffersを消去し、deferred policyだけを解放する。"""
        self._frames_prev = None
        self._prev_joint_positions = None
        self._prev_hand_state = None
        self._prev_action_19d = None
        self._action_queue.clear()
        self._action_queue_next_index = 0
        release = getattr(self._policy, "release_after_skill", None)
        if callable(release):
            release()

    def step(self, obs: dict) -> np.ndarray | None:
        """Per-tick: orchestrator obs → Policy.predict → 19D chunk slice + dispatch。

        Args:
            obs: SampleVLASkill docstring の orchestrator obs dict。key:
                - "head_rgb": (H, W, 3) BGR uint8 (single、packed stereo なら split 済)
                - "wrist_left_rgb" / "wrist_right_rgb": (H, W, 3) BGR uint8 or None
                - "joint_state": 29-dim raw robot state or None (velocity 計算用)
                - "cleaned": list[OBBDetection] (現状 flat、暫定 HEAD_LEFT に割り当て)
                - "t": timestamp ns

        Returns:
            (14,) float64 numpy array = arm joint targets [rad]。既存
            SkillDispatchLowerPolicy が caller に返し、caller が
            arm_actuator.send_action(arr) する既存 pattern を維持。
        """
        # A deferred expert may take tens of seconds to load in _on_start().
        # Do not execute the frame captured before that load; the publisher
        # keeps the last safe target and the next loop supplies a fresh frame.
        if bool(getattr(self._policy, "loaded_during_last_reset", False)):
            self._policy.loaded_during_last_reset = False
            return None

        # 1. orchestrator obs → policies.base.Observation
        observation = self._build_observation(obs)

        # Issue #137 Phase 4: retry FSM。progress は policy を回す前に要る
        # (今 tick で attempt を打つのか、戻すのか、hold なのかを決めるため)。
        progress = None
        if self._progress_monitor is not None:
            progress = self._progress_monitor.update(
                obs.get("cleaned"), int(obs.get("t") or 0)
            )
        decision = None
        if self._retry is not None:
            decision = self._retry.update(
                t_ns=int(obs.get("t") or 0),
                progress=progress,
                return_complete=self._return_complete(obs),
            )
            self._retry_phase = decision.phase
            self._z_ceiling_target_m = decision.ceiling_m
            if decision.flush_execution_state:
                self._flush_execution_state()

        if decision is not None and decision.phase is not RetryPhase.RUNNING:
            current_step = self._non_policy_target(obs, decision)
            self.last_action = PolicyAction(
                action_chunk=current_step.reshape(1, ACTION_DIM_TOTAL).copy(),
                latency_ms=0.0,
                metadata={"retry_phase": decision.phase.value},
            )
        else:
            if self._attempt_start_z_m is None and self._z_ceiling is not None:
                self._attempt_start_z_m = self._z_ceiling.left_tool_z(
                    self._measured_19d(obs)
                )
            # 2. Policy forward / decoded absolute chunk consumption.
            # Policies without an explicit execution horizon keep the original
            # receding-horizon behaviour exactly: predict each tick, use row 0.
            execution_horizon = max(
                1, int(getattr(self._policy, "EXECUTION_HORIZON", 1))
            )
            if execution_horizon == 1:
                policy_action = self._policy.predict(observation)
                chunk = policy_action.action_chunk
                if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM_TOTAL:
                    raise ValueError(
                        f"policy action_chunk must be (chunk_len, {ACTION_DIM_TOTAL}), "
                        f"got shape {chunk.shape}"
                    )
                if chunk.shape[0] < 1:
                    raise ValueError("policy action_chunk must contain at least one step")
                current_step = chunk[0]
                self.last_action = policy_action
            else:
                replanned = not self._action_queue
                replan_action: PolicyAction | None = None
                if replanned:
                    replan_action = self._policy.predict(observation)
                    chunk = replan_action.action_chunk
                    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM_TOTAL:
                        raise ValueError(
                            f"policy action_chunk must be (chunk_len, {ACTION_DIM_TOTAL}), "
                            f"got shape {chunk.shape}"
                        )
                    queued_steps = min(execution_horizon, chunk.shape[0])
                    if queued_steps < 1:
                        raise ValueError("policy action_chunk must contain at least one step")
                    self._action_queue = [
                        chunk[index].astype(np.float32, copy=True)
                        for index in range(queued_steps)
                    ]
                    self._action_queue_next_index = 0

                current_index = self._action_queue_next_index
                current_step = self._action_queue.pop(0)
                self._action_queue_next_index += 1
                source = replan_action if replan_action is not None else self.last_action
                if source is None:  # defensive; a non-empty queue always has a source
                    raise RuntimeError("action queue has no originating PolicyAction")
                metadata = dict(source.metadata)
                metadata.update(
                    {
                        "chunk_step_index": current_index,
                        "chunk_execution_horizon": execution_horizon,
                        "chunk_reused": not replanned,
                        "replan_latency_ms": (
                            float(replan_action.latency_ms)
                            if replan_action is not None
                            else 0.0
                        ),
                    }
                )
                # Expose the actual transmitted step as chunk[0] so existing logs
                # remain truthful; the original model metadata is retained above.
                self.last_action = PolicyAction(
                    action_chunk=current_step.reshape(1, ACTION_DIM_TOTAL),
                    latency_ms=(
                        float(replan_action.latency_ms)
                        if replan_action is not None
                        else 0.0
                    ),
                    metadata=metadata,
                )

        # Phase 3 (Issue #128): Layer 1 motion smoothing = per-tick trajectory
        # envelope (velocity + acceleration limiting)。measured 19D は raw obs から
        # 組む (waist=joint_state[12:15], arms=joint_state[15:29], hand=hand_state)。
        # 初回 apply は measured 基準、以降は前 apply 出力基準 (limiter 内で保持)。
        # limiter=None なら bypass (dev/test 経路、Layer 2 のみで最終 hardware cap)。
        raw_current_step = current_step.copy()
        # Issue #137 Phase 3: MotionLimiter の **前** に z 天井をかける。limiter は
        # 速度/加速度の包絡なので、天井を後段に置くと「limiter が丸めた目標」を
        # さらに動かすことになり、速度制限が二重にかかって効きが読めなくなる。
        z_ceiling_result = None
        if self._z_ceiling is not None:
            z_ceiling_result = self._z_ceiling.apply(
                current_step, self._z_ceiling_target_m
            )
            current_step = z_ceiling_result.target_19d.astype(
                current_step.dtype, copy=False
            )
        # Issue #141 (1-4): 教師の範囲の補正も MotionLimiter の **前**。z 天井の後に
        # 置くのは、天井が左腕 7 関節を動かした結果も教師の範囲に収めるため。戻す速さは
        # この後の limiter の速度・加速度の上限に任せる。
        teacher_range_result = None
        if self._teacher_range is not None:
            teacher_range_result = self._teacher_range.apply(current_step)
            current_step = teacher_range_result.target_19d.astype(
                current_step.dtype, copy=False
            )
        if self._motion_limiter is not None:
            measured_19d = self._measured_19d(obs)
            current_step = self._motion_limiter.apply(
                target=current_step, measured=measured_19d
            ).astype(current_step.dtype, copy=False)

        # ``last_action`` is the recorder/debug contract.  It must describe
        # the target actually handed to waist/arm/Dex1 actuators, not the raw
        # model row before the safety envelope.  Preserve model/replanning
        # diagnostics while recording the limiter intervention separately.
        source_action = self.last_action
        if source_action is None:  # defensive: both paths above set it
            raise RuntimeError("policy step produced no diagnostic action")
        action_metadata = dict(source_action.metadata)
        action_metadata.update({
            "motion_limiter_applied": self._motion_limiter is not None,
            "raw_target_19d": raw_current_step.astype(float).tolist(),
            "raw_target_max_abs_delta": float(
                np.max(np.abs(raw_current_step - current_step))
            ),
            "unsafe_position_target_held": bool(
                getattr(self._motion_limiter, "last_target_held", False)
            ),
            "position_violation_indices": list(
                getattr(
                    self._motion_limiter,
                    "last_position_violation_indices",
                    (),
                )
            ),
        })
        # Issue #141 (1-4): 補正が効いた tick・関節・はみ出した量を残す (実機の記録で
        # 何回効いたかを見るため)。補正が無効な skill では key ごと出さない。
        if teacher_range_result is not None:
            action_metadata.update({
                "teacher_range_bind": bool(teacher_range_result.bind),
                "teacher_range_corrected_joints": list(
                    teacher_range_result.corrected_joints
                ),
                "teacher_range_max_excursion_rad": float(
                    teacher_range_result.max_excursion_rad
                ),
            })
        # Issue #137: 実機評価で唯一の診断材料になる signal を残す。
        # `raw_target_19d` = MotionLimiter 適用前の model 生出力。これが無いと
        # 「限界器が丸めた後の値」しか見えず、model が範囲外を要求し続けている
        # ことに気付けない (実測: insert で right_elbow が 210/605 tick 違反)。
        action_metadata["raw_target_19d"] = [
            float(v) for v in raw_current_step
        ]
        # Issue #137 Phase 2: 天板が実際に回ったかの観測。手先が教師と同等に
        # 動いていても天板が回っていない run があった (実測 0.4deg / 教師 16.4deg)
        # ため、手の動きとは独立にここを残す。本 Phase では診断のみ。
        if z_ceiling_result is not None:
            action_metadata.update({
                "z_ceiling_target_mm": (
                    None if self._z_ceiling_target_m is None
                    else float(self._z_ceiling_target_m * 1000.0)
                ),
                "z_ceiling_bind": bool(z_ceiling_result.bind),
                "z_before_mm": float(z_ceiling_result.z_before_m * 1000.0),
                "z_after_mm": float(z_ceiling_result.z_after_m * 1000.0),
                "z_ceiling_correction_mm": float(
                    z_ceiling_result.correction_m * 1000.0
                ),
                # True = DLS の解が hard_floor を割ったので補正を破棄した tick。
                # 常時 True なら hard_floor か max_correction_m の設定が噛み合って
                # いない (天井が実質効いていない)。
                "z_ceiling_floor_rejected": bool(
                    getattr(z_ceiling_result, "floor_rejected", False)
                ),
            })
        if progress is not None:
            action_metadata.update({
                "table_rotation_deg": float(progress.rotation_deg),
                "table_rotation_sustained": bool(progress.sustained),
                "table_rotation_usable": bool(progress.usable),
                "table_top_detected": bool(progress.detected),
                "table_top_stale_s": float(progress.stale_s),
            })
        if decision is not None:
            action_metadata.update({
                "retry_phase": decision.phase.value,
                "retry_attempt_index": int(decision.attempt_index),
                "retry_reason": decision.reason,
            })
        if self._motion_limiter is not None:
            # limiter は DI なので duck-typed stub も来る。診断項目は optional 扱い。
            action_metadata.update({
                "position_violation_indices": list(
                    getattr(
                        self._motion_limiter,
                        "last_position_violation_indices",
                        (),
                    )
                ),
                "position_violation_ticks": int(
                    getattr(self._motion_limiter, "position_violation_ticks", 0)
                ),
            })
        self.last_action = PolicyAction(
            action_chunk=current_step.reshape(1, ACTION_DIM_TOTAL).copy(),
            latency_ms=float(source_action.latency_ms),
            metadata=action_metadata,
        )

        # 3. waist + hand は internal dispatch、arm は return
        if self._dispatch_waist:
            self._waist_actuator.send_action(current_step[WAIST_SLICE].tolist())
        self._hand_actuator.send_action(current_step[HAND_SLICE].tolist())
        arm_positions = current_step[ARMS_SLICE].astype(np.float64, copy=True)

        # 4. Frame + state buffer 更新 (次 tick の images_prev / velocity / tracking_err 用)
        self._frames_prev = dict(observation.frames_bgr)
        # 今 tick の raw joint_positions と 19D action を保持 (次 tick が読む)
        js = obs.get("joint_state")
        if js is not None:
            pos = getattr(js, "position", None)
            if pos is not None:
                arr = np.asarray(pos, dtype=np.float32)
                if arr.shape == (29,):
                    self._prev_joint_positions = arr
        hs = obs.get("hand_state")
        if hs is not None:
            hand = np.asarray(
                getattr(hs, "position_rad", hs), dtype=np.float32
            )
            if hand.shape == (2,) and np.all(np.isfinite(hand)):
                self._prev_hand_state = hand.copy()
        self._prev_action_19d = current_step.astype(np.float32, copy=True)

        return arm_positions

    # ---- private helpers ---- #

    def _flush_execution_state(self) -> None:
        """attempt 境界で実行層の内部状態を捨てる。

        RTC / temporal ensemble / async pipeline は「前 chunk との連続性」を
        保つのが仕事なので、捨てないと retry しても停滞していた chunk を
        引き継いでしまう。`Policy.reset` が leftover / DelayEstimator / pipeline を
        まとめて落とすので、`_on_start` と同じものを呼ぶ。

        **progress monitor は reset しない**。参照を attempt ごとに張り直すと、
        1 回目で 5deg 回した分が忘れられて「全体でどれだけ回ったか」が見えなく
        なる。参照は skill 開始時の 1 回だけ立てる。
        """
        reset_fn = getattr(self._policy, "reset", None)
        if callable(reset_fn):
            reset_fn()
        if self._motion_limiter is not None:
            self._motion_limiter.reset()
        self._action_queue.clear()
        self._action_queue_next_index = 0
        self._attempt_start_z_m = None

    def _return_target_19d(self, obs: dict) -> np.ndarray:
        """戻り動作の 19D 目標。mode で 2 通り。"""
        measured = self._measured_19d(obs)
        mode = self._retry_options.get("return_mode", ReturnMode.FULL_RETURN)
        if mode is ReturnMode.FULL_RETURN:
            initial = self._retry_options.get("initial_arm_14")
            if initial is None:
                raise RuntimeError(
                    "full_return needs retry_options['initial_arm_14'] "
                    "(skill_config の initial_pose.arm_position_rad)"
                )
            target = measured.copy()
            target[ARMS_SLICE] = np.asarray(initial, dtype=np.float64)
            return target
        # lift_and_reapproach: 手先 z だけ attempt 開始高さへ戻す。
        if self._z_ceiling is None or self._attempt_start_z_m is None:
            return measured
        margin = float(self._retry_options.get("lift_margin_m", 0.0))
        return self._z_ceiling.lift_to(
            measured, self._attempt_start_z_m + margin
        ).target_19d

    def _return_complete(self, obs: dict) -> bool:
        """戻り目標に到達したか。RETURNING 以外では参照されない。"""
        if self._retry_phase is not RetryPhase.RETURNING:
            return False
        measured = self._measured_19d(obs)
        mode = self._retry_options.get("return_mode", ReturnMode.FULL_RETURN)
        if mode is ReturnMode.FULL_RETURN:
            initial = self._retry_options.get("initial_arm_14")
            if initial is None:
                return False
            return arm_return_complete(
                measured[ARMS_SLICE],
                np.asarray(initial, dtype=np.float64),
                float(self._retry_options.get("return_tolerance_rad", 0.05)),
            )
        if self._z_ceiling is None or self._attempt_start_z_m is None:
            return True
        margin = float(self._retry_options.get("lift_margin_m", 0.0))
        return self._z_ceiling.left_tool_z(measured) >= (
            self._attempt_start_z_m + margin - 0.005
        )

    def _non_policy_target(self, obs: dict, decision: Any) -> np.ndarray:
        """RETURNING / HOLDING の 19D 目標 (policy は呼ばない)。"""
        if decision.phase is RetryPhase.RETURNING:
            self._hold_target_19d = None
            return self._return_target_19d(obs)
        # HOLDING: 最初に入った tick の目標をそのまま持ち続ける。教師 segment は
        # 回り切った時点で終わっているので、その先を policy に描かせない。
        if self._hold_target_19d is None:
            self._hold_target_19d = self._measured_19d(obs)
        return self._hold_target_19d.copy()

    def _measured_19d(self, obs: dict) -> np.ndarray:
        """orchestrator obs → 19D measured state (waist3 + arms14 + hand2)。

        MotionLimiter.apply の `measured` 引数用。VlaSkill 19D 契約に一致する
        物理単位で組む: waist/arms は joint_state (G1_JOINT_NAMES 順 29-dim) の
        該当 slice、hand は dex1 hand_state (physical rad [0, 5.4])。

        欠損時の fallback:
            - joint_state が None: 前 tick buffer (`_prev_joint_positions`) 使用、
              それも None なら zeros で組む (limiter は 0 基準の trajectory 開始)
            - hand_state が None: 前 tick buffer or zeros

        Args:
            obs: orchestrator obs dict (step() が受け取るもの)。

        Returns:
            (19,) float64、[waist3, arms14, hand2] 順。
        """
        # joint_state 29-dim (leg 12 + waist 3 + arm 14)
        jp = np.zeros(29, dtype=np.float64)
        js = obs.get("joint_state")
        if js is not None:
            pos = getattr(js, "position", None)
            if pos is not None:
                arr = np.asarray(pos, dtype=np.float64)
                if arr.shape == (29,):
                    jp = arr
        elif self._prev_joint_positions is not None:
            jp = self._prev_joint_positions.astype(np.float64, copy=False)

        # hand 2-dim (Dex1 physical rad)
        hand = np.zeros(2, dtype=np.float64)
        hs = obs.get("hand_state")
        if hs is not None:
            h = np.asarray(getattr(hs, "position_rad", hs), dtype=np.float64)
            if h.shape == (2,) and np.all(np.isfinite(h)):
                hand = h
        elif self._prev_hand_state is not None:
            hand = self._prev_hand_state.astype(np.float64, copy=False)

        out = np.zeros(ACTION_DIM_TOTAL, dtype=np.float64)
        out[WAIST_SLICE] = jp[12:15]    # waist yaw/roll/pitch
        out[ARMS_SLICE]  = jp[15:29]    # L arm 7 + R arm 7
        out[HAND_SLICE]  = hand         # Dex1 L/R physical rad
        return out

    def _build_observation(self, obs: dict) -> Observation:
        """orchestrator obs dict → Observation (per-skill identity 埋め込み)。"""
        # Head: packed stereo (480x1280) なら split、単 cam なら both slot に same 画像
        head_rgb = obs.get("head_rgb")
        if head_rgb is None:
            raise ValueError("orchestrator obs missing required 'head_rgb'")
        if head_rgb.shape[:2] == (480, 1280):
            head_l, head_r = split_head_stereo(head_rgb)
        else:
            # 単 cam 想定 (dev/debug 経路)、L に入れて R は同じ (better than zeros)
            head_l = head_rgb
            head_r = head_rgb

        frames_bgr: dict[CameraKey, np.ndarray] = {
            CameraKey.HEAD_LEFT: head_l,
            CameraKey.HEAD_RIGHT: head_r,
        }
        # wrist は available なら埋める、None なら zeros (RAMEN-Ori 4 cam 用)
        wrist_l = obs.get("wrist_left_rgb")
        wrist_r = obs.get("wrist_right_rgb")
        if wrist_l is not None:
            frames_bgr[CameraKey.WRIST_LEFT] = adapt_wrist_to_training_shape(
                np.asarray(getattr(wrist_l, "rgb", wrist_l))
            )
        if wrist_r is not None:
            frames_bgr[CameraKey.WRIST_RIGHT] = adapt_wrist_to_training_shape(
                np.asarray(getattr(wrist_r, "rgb", wrist_r))
            )

        # State: joint_state 29-dim + 前 tick buffer から RawRobotState を組む
        raw_state = self._build_raw_state(
            obs.get("joint_state"), obs.get("hand_state")
        )
        # instance 経由なら通常policyとdeferred adapterの双方で同じ契約になる。
        state = self._policy.build_state_from_raw(raw_state)

        # OBB detections: 現状 orchestrator flat list、暫定 HEAD_LEFT に全 det
        # (orchestrator 側で per-cam YOLO 対応後、dict[Cam, list] 直接受け取り)
        cleaned = obs.get("cleaned")
        obb_detections: dict[CameraKey, list] | None = None
        # None = detector pipeline unavailable; [] = detector ran and found
        # nothing.  Overlay checkpoints must distinguish these states.
        if cleaned is not None:
            obb_detections = {CameraKey.HEAD_LEFT: list(cleaned)}

        # Issue #132 Phase A: language は self._language_override (policy_config.yaml
        # 経由の per-variant 指定) が最優先。未指定なら class-level LANGUAGE を使う。
        # DEFAULT_LANGUAGE_PROMPT (Gr00tPolicy 側 module 定数) には落ちない
        # (直接 policy call 用 safety net として別途保持されている)。
        language = self._language_override or self.LANGUAGE
        skill_id = (
            self.SKILL_ID if self._skill_id_override is None else self._skill_id_override
        )
        return Observation(
            frames_bgr=frames_bgr,
            frames_bgr_prev=self._frames_prev,
            state=state,
            skill_id=skill_id,
            language=language,
            obb_detections=obb_detections,
            timestamp_ns=int(obs.get("t", time.monotonic_ns())),
        )

    def _build_raw_state(
        self, joint_state: Any, hand_state: Any = None
    ) -> RawRobotState:
        """orchestrator joint_state (Orin JointStateData or None) + instance buffer
        → RawRobotState。

        Orin `real_hw_bridge_node` は `/joint_states` (29 dim = G1_JOINT_NAMES 順)
        のみ publish、hand_state (Dex1-1) と ee_state (FK) は未統合。本 method は
        利用可能な signal (joint_positions) を実データ流入し、buffer signal
        (last_action_19d / joint_positions_prev) は VlaSkill instance state から
        補う。ep 先頭 tick (buffer None) は tracking_err / velocity zeros。

        Args:
            joint_state: `JointStateData` (perception.joint_state_source) or None
                (未受信 tick 相当、joint_positions zeros で組む)。

        Returns:
            RawRobotState: 実データ準拠 layout、buffer signal も反映。
        """
        # joint_positions: JointStateData.position が (29,) の想定 (G1_JOINT_NAMES 順)。
        # None or shape mismatch は zeros に fallback (dev 経路で verify を段階化)。
        joint_positions = np.zeros(29, dtype=np.float32)
        if joint_state is not None:
            position = getattr(joint_state, "position", None)
            if position is not None:
                arr = np.asarray(position, dtype=np.float32)
                if arr.shape == (29,):
                    joint_positions = arr

        # ee_state: FK が渡されていれば joint_positions から計算、無ければ zeros。
        # Phase A-5 で g1_urdf_fk.G1WristFK を渡す形に、default は None = zeros。
        if self._fk is not None:
            ee_state = self._fk.compute_ee_state(joint_positions)
        else:
            ee_state = np.zeros(12, dtype=np.float32)

        # Official Dex1 state and the source LeRobot datasets both use the
        # physical motor-output coordinate in radians (not a [0, 1] fraction).
        hand_position_rad = np.zeros(2, dtype=np.float32)
        if hand_state is not None:
            position = getattr(hand_state, "position_rad", hand_state)
            arr = np.asarray(position, dtype=np.float32)
            if arr.shape == (2,) and np.all(np.isfinite(arr)):
                hand_position_rad = arr
        return RawRobotState(
            joint_positions=joint_positions,
            hand_state=hand_position_rad,
            ee_state=ee_state,
            last_action_19d=self._prev_action_19d,
            joint_positions_prev=self._prev_joint_positions,
            hand_state_prev=self._prev_hand_state,
        )


class MoveTableBaseVlaSkill(VlaSkill):
    """Task 7 (move_table_base) 用 VLA wrapper (Issue #125 Phase 9a)。

    identity:
        - SKILL_ID = 5 (skill_mapping.py:SKILL_MOVE_TABLE_BASE)
        - LANGUAGE = "rotate and move table base (combined 5+7)"
          (subtask_training.json:subtasks.combined_task5_7.task。GR00T では
           rotate と move を separate せず 1 prompt で学習しているため、
           inference 側 move_table_base skill でも同 prompt を使う)
    """

    name = "move_table_base"
    SKILL_ID: int = 5
    LANGUAGE: str = "rotate and move table base (combined 5+7)"


class RotateTableBaseVlaSkill(VlaSkill):
    """Task 5 (rotate_table_base) 用 VLA wrapper (Issue #125 Phase 9a、reserved)。

    identity:
        - SKILL_ID = 4 (skill_mapping.py:SKILL_ROTATE_TABLE_BASE)
        - LANGUAGE = "rotate and move table base (combined 5+7)"
          (GR00T は同 prompt、RAMEN-Ori は skill_id で区別する)

    注記: 現行 inference stack (skill_planner + orchestrator + mock) は
    rotate_table_base skill_name を保持していない (move_table_base に merge 済み、
    skill_mapping.py docstring 参照)。本 subclass は将来 inference stack を split
    した時のための reserve。現状 registry 登録は不要。
    """

    name = "rotate_table_base"
    SKILL_ID: int = 4
    LANGUAGE: str = "rotate and move table base (combined 5+7)"


class PickTableLegVlaSkill(VlaSkill):
    """pick_table_leg 用 VLA wrapper (Issue #125、pick_legs 対応)。

    identity:
        - SKILL_ID = 3 (skill_mapping.py:SKILL_PICK_TABLE_LEG)
        - LANGUAGE = "pick table leg" (subtask_training.json:pick_leg.task 準拠)

    現在対応 policy = Gr00tPolicyPickLegs (full-body embodiment、38D 直結)。
    training pipeline が違う (Isaac-GR00T native raw) ため、build_state_from_raw
    は Gr00tPolicyPickLegs の staticmethod を経由する = VlaSkill._build_observation
    の `self._policy.__class__.build_state_from_raw(raw)` pattern がそのまま機能。
    """

    name = "pick_table_leg"
    SKILL_ID: int = 3
    LANGUAGE: str = "pick table leg"


class InsertTableLegVlaSkill(VlaSkill):
    """insert_table_leg 用 VLA wrapper (Issue #125、SOTA π0.5 = 52.5% の最難関 skill)。

    identity:
        - SKILL_ID = 0 (skill_mapping.py:SKILL_INSERT_TABLE_LEG)
        - LANGUAGE = "insert table leg to table base"
          (takada training dataset `Team-RAMEN/IROS2026_RAMEN_takada_insert_table_leg_curated_optimal`:
           meta/tasks.parquet の task_index=0 に対応する task 文字列)

    現在対応 policy = Gr00tPolicy (LeRobot fork format、REAL_G1_RELATIVE_EEF)。
    ckpt 例 = groot_insert_leg_200k (takada 200k step optimal)。
    """

    name = "insert_table_leg"
    SKILL_ID: int = 0
    LANGUAGE: str = "insert table leg to table base"


class RotateLegToTightenVlaSkill(VlaSkill):
    """rotate_leg_to_tighten 用 VLA wrapper (Issue #125、全 533ep 50% dominant skill)。

    identity:
        - SKILL_ID = 2 (skill_mapping.py:SKILL_ROTATE_LEG_TO_TIGHTEN)
        - LANGUAGE = "rotate leg to tighten"
          (takada training dataset `Team-RAMEN/IROS2026_RAMEN_takada_rotate_leg_to_tighten_curated_optimal`:
           meta/tasks.parquet の task_index=3 に対応する task 文字列、実測)

    現在対応 policy = Gr00tPolicy (LeRobot fork format、REAL_G1_RELATIVE_EEF)。
    ckpt 例 = groot_rotate_leg_200k (takada 200k step optimal)。

    注記: rotate_table_base (skill_id=4) とは完全別 skill (leg の締め動作 vs
    table 全体の base 回転)。skill_mapping.py 参照。
    """

    name = "rotate_leg_to_tighten"
    SKILL_ID: int = 2
    LANGUAGE: str = "rotate leg to tighten"


class InsertAndTightenVlaSkill(VlaSkill):
    """Curated task 8 sequence: insert the leg, then rotate it to tighten.

    This checkpoint is GR00T-only and uses language conditioning from the
    dataset's ``meta/tasks.parquet``.  ``SKILL_ID=None`` deliberately prevents
    it from being mistaken for one of RAMEN-Ori's six canonical skill IDs.
    """

    name = "insert_and_tighten"
    SKILL_ID: int | None = None
    LANGUAGE: str = "insert table leg to table base and rotate leg to tighten"


class FlipTableVlaSkill(VlaSkill):
    """flip_table 用 VLA wrapper (Issue #125、組立最終工程)。

    identity:
        - SKILL_ID = 1 (skill_mapping.py:SKILL_FLIP_TABLE)
        - LANGUAGE = "flip table"
          (suzuki training dataset `Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_4`:
           meta/tasks.parquet の task_index=0 に対応する task 文字列、実測)

    現在対応 policy = Gr00tPolicy (LeRobot fork format、REAL_G1_RELATIVE_EEF)。
    ckpt 例 = groot_flip_table_n17_4 (suzuki n17_4 iteration 最新)。
    """

    name = "flip_table"
    SKILL_ID: int = 1
    LANGUAGE: str = "flip table"
