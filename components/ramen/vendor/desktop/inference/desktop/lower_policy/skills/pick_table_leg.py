"""PickTableLegSkill: pick_table_leg を CV + Motion Planning + ルールベースで解く
4-stage サブFSM (Issue #123)。

## なぜサブFSM か

dataset 全量解析 (2114 episode / 975,291 frame) の結果、pick_leg は単腕 pick では
なく **両腕持ち替えタスク**だった (98.6% の episode で右手 → 左手の持ち替えが発生、
93.9% は両手同時保持を経由、その後 99.1% で右手が離す)。

一方、持ち替え点は **骨盤基準で** x IQR 0.027m / z IQR 0.025m と締まっている
(右グリッパ基準だと x IQR 0.062m / z IQR 0.050m とむしろ緩い)。つまり
「脚に対する相対位置」ではなく「体の前の決まった場所」で受け渡している。
→ 持ち替え以降は固定目標ポーズで再現でき、CV が要るのは stage 1 だけ。

## 全 stage が「目標ポーズ + Motion Planning」

固定軌道のリプレイはしない。毎 tick 現在姿勢から目標へ Cartesian 補間 → IK。
リプレイと違い誤差が累積せず、本番キャリブレーションは YAML の数値変更だけで済む。

## 実装上の必須制約

- **両手が同じ脚を握っている間は片腕ずつしか動かさない** (`moving` で明示)。
  剛体を介した閉ループ運動連鎖になり、独立に解いた 2 つの目標が引っ張り合って
  内力を生む。力センサが無いので検知もできない。
- **release は「開く → 引く」の順**。逆だと脚を引きずる。
- 把持指令は全閉。dataset の `hand_cmd` 1.85 は遠隔操作者のグローブ開度が記録
  されただけで、真似ると空振り時に失敗を検出できない (`actuators/dex1.py` 参照)。

## 既存プランナ非改変

Dex1 の I/O は本 skill が自前で保持する (`MoveToTable` が walk actuator を直接
叩く Type A の流儀)。これにより `Orchestrator._build_obs` を変更せずに済む。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.actuators.dex1 import (
    Dex1Gripper,
    GraspState,
    GraspThresholds,
    classify_grasp,
)
from inference.desktop.lower_policy.grasp_pose import GraspPoseProvider
from inference.desktop.lower_policy.kinematics.base import ArmKinematics
from inference.desktop.lower_policy.kinematics.types import (
    NUM_ARM_JOINTS_PER_SIDE,
    EEPose,
    IKStatus,
    Side,
)
from inference.desktop.lower_policy.pose_utils import (
    NUM_ARM_JOINTS,
    POSE_ABS_LIMIT_RAD,
    arm_positions_from_joint_state,
)
from inference.desktop.lower_policy.skills.base import Skill
from inference.desktop.lower_policy.skills.motion_limiter import MotionLimiter

# 14-D 腕 pose 内での左右のスライス。pose_utils.JOINT_NAMES の順序に対応。
_SLICE: dict[Side, slice] = {
    Side.LEFT: slice(0, NUM_ARM_JOINTS_PER_SIDE),
    Side.RIGHT: slice(NUM_ARM_JOINTS_PER_SIDE, NUM_ARM_JOINTS),
}


class SkillStatus(str, Enum):
    """skill 全体の進行状態。"""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class GoalKind(str, Enum):
    """stage の目標ポーズをどう決めるか。"""

    CV = "cv"                    # GraspPoseProvider から取得 (毎回変わる)
    FIXED = "fixed"              # root_link 基準の固定値 (YAML)
    RELATIVE = "relative"        # 他方の腕の**実測** EE pose からのオフセット


@dataclass(frozen=True)
class GoalSpec:
    """目標ポーズの決め方。

    Attributes:
        kind: 解決方法。
        pose: `FIXED` のときの root_link 基準ポーズ。
        reference: `RELATIVE` のときの基準となる腕。
        offset: 並進オフセット [m] (3,)。解釈は kind によって変わる:
            - `RELATIVE`: **基準腕のローカル座標系** (脚の長手方向に沿ってずらすため)
            - `CV`: **root_link 座標系**。把持点の真上に pre-grasp を置くのに使う。
              省略可 (無指定なら把持ポーズそのもの)。
        rpy: `RELATIVE` のときの目標姿勢 (root_link 基準、絶対)。
        rpy_offset: `CV` のときの姿勢差分 [rad] (3,)。把持姿勢からの相対。

            pre-grasp で **手首を起こす** ために要る。把持姿勢 (横倒しの円筒を掴む
            ため手首を寝かせた状態) のまま真上へ 10cm 上げると、実観測の把持配置の
            26% が到達不能になる (運用 ±1.5 rad、URDF 限界でも 18% が不能)。
            デモ者も持ち上げ時に手首を 50deg -> 18deg と起こしており、「把持姿勢の
            まま上昇」は元々やっていない動作。pitch を 0.5 rad 起こすと到達率は
            100% になる。降下 stage が測地線補間で姿勢を戻すので、接近は
            「起こした手首で真上 -> 降下しながら寝かせる」になる。
    """

    kind: GoalKind
    pose: Optional[EEPose] = None
    reference: Optional[Side] = None
    offset: Optional[np.ndarray] = None
    rpy: Optional[np.ndarray] = None
    rpy_offset: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.kind is GoalKind.FIXED and self.pose is None:
            raise ValueError("goal(fixed): 'pose' is required")
        if self.kind is GoalKind.RELATIVE:
            if self.reference is None:
                raise ValueError("goal(relative): 'reference' is required")
            if self.offset is None or self.rpy is None:
                raise ValueError("goal(relative): 'offset' and 'rpy' are required")


@dataclass(frozen=True)
class ExitSpec:
    """stage の終了条件。指定した条件を **すべて** 満たしたら次 stage へ。

    Attributes:
        pos_tol_m: 目標との位置誤差がこれ以下。None なら位置は見ない。
        rot_tol_rad: 目標との姿勢誤差がこれ以下。None なら姿勢は見ない。
        grasp: この側が `HOLDING` になること。
        release: この側が `OPEN` になること。
    """

    pos_tol_m: Optional[float] = None
    rot_tol_rad: Optional[float] = None
    grasp: Optional[Side] = None
    release: Optional[Side] = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (self.pos_tol_m, self.rot_tol_rad, self.grasp, self.release)
        )


@dataclass(frozen=True)
class StageSpec:
    """1 stage の定義。

    Attributes:
        name: stage 名 (log / test 用)。
        goal: 目標ポーズの決め方。
        moving: この stage で動かす腕。ここに無い腕は **直前の指令値を保持**する。
            両手保持区間で片腕ずつしか動かさないための唯一の仕組み。
        duration_sec: Cartesian 補間の公称所要時間。
        timeout_sec: これを超えて終了条件を満たさなければ abort。
        grip: stage 開始時に送るグリッパ指令 (side → 位置)。None の側は現状維持。
        exit: 終了条件。
        retry_from: 把持に空振りしたときに戻る stage 名。None なら直前 stage。
            右手の把持失敗は「脚を再検出して接近し直す」= stage 1 に戻るのが
            正しいが、左手の持ち替え失敗で stage 1 に戻ると **右手が保持中の脚を
            置いたまま再検出することになり破綻する**。戻り先は stage ごとに
            明示できる必要がある。
    """

    name: str
    goal: Optional[GoalSpec]
    moving: tuple[Side, ...]
    duration_sec: float
    timeout_sec: float
    grip: dict[Side, float] = field(default_factory=dict)
    exit: ExitSpec = field(default_factory=ExitSpec)
    retry_from: Optional[str] = None

    def __post_init__(self) -> None:
        if self.duration_sec <= 0:
            raise ValueError(f"stage {self.name!r}: duration_sec must be > 0")
        if self.timeout_sec < self.duration_sec:
            raise ValueError(
                f"stage {self.name!r}: timeout_sec ({self.timeout_sec}) must be "
                f">= duration_sec ({self.duration_sec})"
            )
        if self.goal is None and self.moving:
            raise ValueError(
                f"stage {self.name!r}: moving arms specified but goal is None"
            )
        if self.exit.is_empty():
            raise ValueError(f"stage {self.name!r}: exit condition must not be empty")


class PickTableLegSkill(Skill):
    """4-stage サブFSM。Type B (`step` が 14-D 腕関節指令を返す)。

    Args:
        stages: 実行順の stage 定義。
        kinematics: `ArmKinematics` 実装 (FK/IK)。
        gripper: `Dex1Gripper` 実装。skill が自前で保持する。
        grasp_provider: stage の `GoalKind.CV` を解決する。
        grasp_thresholds: 把持判定しきい値。
        max_grasp_retries: 空振り (`EMPTY_CLOSED`) 検出時に stage 1 をやり直す回数。
        time_fn: 時刻取得 (test で fake 時計を注入)。
        skill_name: dispatcher registry の key。
    """

    name = "pick_table_leg"

    def __init__(
        self,
        stages: Sequence[StageSpec],
        *,
        kinematics: ArmKinematics,
        gripper: Dex1Gripper,
        grasp_provider: GraspPoseProvider,
        grasp_thresholds: Optional[GraspThresholds] = None,
        max_grasp_retries: int = 2,
        time_fn: Callable[[], float] = time.monotonic,
        skill_name: Optional[str] = None,
        motion_limiter: Optional[MotionLimiter] = None,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("stages must be a non-empty sequence")
        self._stages: tuple[StageSpec, ...] = tuple(stages)
        self._retry_index: tuple[int, ...] = _resolve_retry_targets(self._stages)
        self._kin = kinematics
        self._gripper = gripper
        self._grasp_provider = grasp_provider
        self._th = grasp_thresholds or GraspThresholds()
        self._max_grasp_retries = int(max_grasp_retries)
        self._time_fn = time_fn
        self._motion_limiter = motion_limiter
        if skill_name is not None:
            self.name = skill_name

        # ---- runtime state (start でリセット) ----
        self._status: SkillStatus = SkillStatus.IDLE
        self._stage_index: int = 0
        self._stage_started_at: Optional[float] = None
        self._start_pose: dict[Side, EEPose] = {}
        self._goal_pose: dict[Side, EEPose] = {}
        self._target: Optional[np.ndarray] = None   # 直前 tick の 14-D 指令
        self._retries: int = 0
        self._failure_reason: Optional[str] = None
        self._issued_grip: dict[Side, float] = {}

    # ------------------------------------------------------------ lifecycle

    def _on_start(self, params: dict) -> None:
        self._status = SkillStatus.RUNNING
        self._stage_index = 0
        self._stage_started_at = None
        self._start_pose = {}
        self._goal_pose = {}
        self._target = None
        self._retries = 0
        self._failure_reason = None
        self._issued_grip = {}
        if self._motion_limiter is not None:
            self._motion_limiter.reset()

    def _on_stop(self) -> None:
        self._stage_started_at = None
        self._gripper.close()

    # ------------------------------------------------------------ properties

    @property
    def status(self) -> SkillStatus:
        return self._status

    @property
    def stage_name(self) -> Optional[str]:
        if self._stage_index >= len(self._stages):
            return None
        return self._stages[self._stage_index].name

    @property
    def stage_index(self) -> int:
        return self._stage_index

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    @property
    def retries(self) -> int:
        return self._retries

    @property
    def stages(self) -> tuple[StageSpec, ...]:
        return self._stages

    @property
    def grasp_provider(self) -> GraspPoseProvider:
        """Expose the read-only perception adapter for preflight diagnostics."""
        return self._grasp_provider

    @property
    def max_dwell_sec(self) -> float:
        """全 stage の timeout 合計 + retry 分。Orchestrator の fail-safe が読む。"""
        total = sum(s.timeout_sec for s in self._stages)
        return total + self._stages[0].timeout_sec * self._max_grasp_retries

    @property
    def is_complete(self) -> bool:
        """Expose successful FSM completion to the modern Orchestrator."""
        return self._status is SkillStatus.DONE

    # ------------------------------------------------------------ step

    def step(self, obs: dict) -> Optional[np.ndarray]:
        """1 tick 分の 14-D 腕関節指令。

        `DONE` / `FAILED` 後は直前指令を保持し続ける (actuator が hold)。
        joint_state 未受信の間は `None` を返す (caller は forward しない = 現状維持)。
        """
        if self._status in (SkillStatus.DONE, SkillStatus.FAILED):
            return None if self._target is None else self._target.copy()

        measured = self._measured_arm_positions(obs)
        if measured is None:
            # joint_state 未受信。まだ何も指令しない (0 を送ると腕が落ちる)。
            return None
        if self._target is None:
            self._target = measured.copy()

        stage = self._stages[self._stage_index]
        now = self._time_fn()
        if self._stage_started_at is None:
            if not self._enter_stage(stage, measured, obs, now):
                # 目標未解決 (脚が未検出等)。現状保持で次 tick を待つ。
                return self._target.copy()

        # --- 終了条件 / 失敗判定 ---
        outcome = self._evaluate_stage(stage, measured, now)
        if outcome is _StageOutcome.FAILED:
            return self._target.copy()
        if outcome is _StageOutcome.ADVANCE:
            return self._target.copy()

        # --- 動かす腕の目標を Cartesian 補間 → IK ---
        elapsed = now - float(self._stage_started_at)
        s = min(1.0, elapsed / stage.duration_sec)
        target = self._target.copy()
        for side in stage.moving:
            waypoint = self._start_pose[side].interpolate(self._goal_pose[side], s)
            res = self._kin.ik(waypoint, seed=measured[_SLICE[side]], side=side)
            if not res.ok:
                self._fail(
                    f"stage {stage.name!r}: IK {res.status.value} for {side.value} arm "
                    f"(pos_err={res.position_error:.4f} rot_err={res.rotation_error:.4f})"
                )
                return self._target.copy()
            if float(np.abs(res.q).max()) > POSE_ABS_LIMIT_RAD:
                # IK が status OK でも安全枠を超える解を返す実装がありうる。
                # 黙って actuator の clamp に任せると「届いていないのに進む」ので
                # ここで明示的に失敗させる。
                self._fail(
                    f"stage {stage.name!r}: IK solution exceeds "
                    f"±{POSE_ABS_LIMIT_RAD} rad for {side.value} arm"
                )
                return self._target.copy()
            target[_SLICE[side]] = res.q
        self._target = (
            target
            if self._motion_limiter is None
            else self._motion_limiter.apply(target=target, measured=measured)
        )
        return self._target.copy()

    # ------------------------------------------------------------ internals

    def _measured_arm_positions(self, obs: dict) -> Optional[np.ndarray]:
        js = obs.get("joint_state")
        if js is None:
            return None
        return arm_positions_from_joint_state(
            js.name, js.position, context=f"skill {self.name!r}"
        )

    def _enter_stage(
        self, stage: StageSpec, measured: np.ndarray, obs: dict, now: float
    ) -> bool:
        """stage 開始処理。目標が解決できなければ `False` (次 tick で再試行)。"""
        goal_by_side: dict[Side, EEPose] = {}
        for side in stage.moving:
            goal = self._resolve_goal(stage, side, measured, obs)
            if goal is None:
                return False
            goal_by_side[side] = goal

        self._stage_started_at = now
        self._start_pose = {
            side: self._kin.fk(measured[_SLICE[side]], side) for side in stage.moving
        }
        self._goal_pose = goal_by_side
        # グリッパ指令は stage 開始時に 1 回だけ (値が変わったときのみ送る)。
        for side, position in stage.grip.items():
            if self._issued_grip.get(side) != position:
                self._gripper.command(side, position)
                self._issued_grip[side] = position
        return True

    def _resolve_goal(
        self, stage: StageSpec, side: Side, measured: np.ndarray, obs: dict
    ) -> Optional[EEPose]:
        spec = stage.goal
        assert spec is not None  # StageSpec.__post_init__ が保証
        if spec.kind is GoalKind.FIXED:
            return spec.pose
        if spec.kind is GoalKind.CV:
            pose = self._grasp_provider.grasp_pose(obs)
            if pose is None:
                return pose
            if spec.offset is None and spec.rpy_offset is None:
                return pose
            # pre-grasp: 把持点の真上へ、手首を起こした姿勢で退避する。
            # 位置は root_link 基準、姿勢は把持姿勢からの差分。
            position = pose.position if spec.offset is None else pose.position + spec.offset
            rpy = pose.rpy if spec.rpy_offset is None else pose.rpy + spec.rpy_offset
            return EEPose(position=position, rpy=rpy)
        # RELATIVE: 基準腕の **実測** pose から解決する。公称値ではなく実測を使う
        # ことで、前 stage が多少ずれて終わっても左手の目標が追従する。
        assert spec.reference is not None and spec.offset is not None
        ref_pose = self._kin.fk(measured[_SLICE[spec.reference]], spec.reference)
        position = ref_pose.position + ref_pose.matrix @ spec.offset
        assert spec.rpy is not None
        return EEPose(position=position, rpy=spec.rpy)

    def _evaluate_stage(
        self, stage: StageSpec, measured: np.ndarray, now: float
    ) -> "_StageOutcome":
        started = float(self._stage_started_at)  # type: ignore[arg-type]
        elapsed = now - started

        # 空振り検出 → stage やり直し (retry 上限まで)。
        if stage.exit.grasp is not None:
            state = self._grasp_state(stage.exit.grasp)
            if state is GraspState.EMPTY_CLOSED:
                if self._retries >= self._max_grasp_retries:
                    self._fail(
                        f"stage {stage.name!r}: grasp failed (empty close) "
                        f"after {self._retries} retries"
                    )
                    return _StageOutcome.FAILED
                self._retries += 1
                self._retry_stage()
                return _StageOutcome.ADVANCE

        if self._stage_satisfied(stage, measured):
            self._advance_stage()
            return _StageOutcome.ADVANCE

        if elapsed > stage.timeout_sec:
            self._fail(f"stage {stage.name!r}: timeout after {elapsed:.2f}s")
            return _StageOutcome.FAILED
        return _StageOutcome.CONTINUE

    def _stage_satisfied(self, stage: StageSpec, measured: np.ndarray) -> bool:
        ex = stage.exit
        if ex.pos_tol_m is not None or ex.rot_tol_rad is not None:
            for side in stage.moving:
                actual = self._kin.fk(measured[_SLICE[side]], side)
                goal = self._goal_pose[side]
                if (
                    ex.pos_tol_m is not None
                    and actual.position_error(goal) > ex.pos_tol_m
                ):
                    return False
                if (
                    ex.rot_tol_rad is not None
                    and actual.rotation_error(goal) > ex.rot_tol_rad
                ):
                    return False
        if ex.grasp is not None and self._grasp_state(ex.grasp) is not GraspState.HOLDING:
            return False
        if ex.release is not None and self._grasp_state(ex.release) is not GraspState.OPEN:
            return False
        return True

    def _grasp_state(self, side: Side) -> Optional[GraspState]:
        state = self._gripper.read(side)
        command = self._gripper.last_command(side)
        if state is None or command is None:
            return None
        return classify_grasp(command, state, self._th)

    def _advance_stage(self) -> None:
        self._stage_index += 1
        self._stage_started_at = None
        if self._stage_index >= len(self._stages):
            self._status = SkillStatus.DONE

    def _retry_stage(self) -> None:
        """把持の空振りから復帰する。`retry_from` で指定された stage へ巻き戻す。

        グリッパ指令は全て再送対象に戻す (巻き戻し先の stage で開き直させるため)。
        """
        self._stage_index = self._retry_index[self._stage_index]
        self._stage_started_at = None
        self._issued_grip.clear()

    def _fail(self, reason: str) -> None:
        self._status = SkillStatus.FAILED
        self._failure_reason = reason

    # ------------------------------------------------------------ config

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        *,
        kinematics: ArmKinematics,
        gripper: Dex1Gripper,
        grasp_provider: GraspPoseProvider,
        time_fn: Callable[[], float] = time.monotonic,
        skill_name: Optional[str] = None,
        motion_limiter: Optional[MotionLimiter] = None,
    ) -> "PickTableLegSkill":
        """`skill_config.yaml` の `skills.pick_table_leg` section から構築する。

        数値は全て YAML 側にあり、Python 側に定数を持たない (CLAUDE.md 方針)。
        """
        if not isinstance(cfg, dict) or "stages" not in cfg:
            raise ValueError("pick_table_leg config: 'stages' is required")
        raw_stages = cfg["stages"]
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("pick_table_leg config: 'stages' must be a non-empty list")

        stages = [_stage_from_config(i, entry) for i, entry in enumerate(raw_stages)]
        return cls(
            stages,
            kinematics=kinematics,
            gripper=gripper,
            grasp_provider=grasp_provider,
            grasp_thresholds=GraspThresholds.from_config(cfg.get("grasp_thresholds")),
            max_grasp_retries=int(cfg.get("max_grasp_retries", 2)),
            time_fn=time_fn,
            skill_name=skill_name,
            motion_limiter=motion_limiter,
        )


class _StageOutcome(str, Enum):
    CONTINUE = "continue"
    ADVANCE = "advance"
    FAILED = "failed"


def _resolve_retry_targets(stages: tuple[StageSpec, ...]) -> tuple[int, ...]:
    """各 stage の `retry_from` を index に解決する。

    未指定なら直前 stage (先頭 stage は自分自身)。前方の stage へは戻れない
    (無限ループになるため construction 時に弾く)。
    """
    by_name: dict[str, int] = {}
    for i, s in enumerate(stages):
        if s.name in by_name:
            raise ValueError(f"duplicate stage name {s.name!r}")
        by_name[s.name] = i

    out: list[int] = []
    for i, s in enumerate(stages):
        if s.retry_from is None:
            out.append(max(0, i - 1))
            continue
        if s.retry_from not in by_name:
            raise ValueError(
                f"stage {s.name!r}: retry_from {s.retry_from!r} is not a stage name "
                f"(valid: {sorted(by_name)})"
            )
        target = by_name[s.retry_from]
        if target > i:
            raise ValueError(
                f"stage {s.name!r}: retry_from {s.retry_from!r} points forward "
                f"(index {target} > {i})"
            )
        out.append(target)
    return tuple(out)


def _side_from_config(value: object, context: str) -> Side:
    try:
        return Side(str(value))
    except ValueError as exc:
        raise ValueError(
            f"{context}: invalid side {value!r} (valid: 'left', 'right')"
        ) from exc


def _stage_from_config(index: int, entry: object) -> StageSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"pick_table_leg config: stage[{index}] must be a mapping")
    name = str(entry.get("name") or f"stage_{index}")
    ctx = f"pick_table_leg stage {name!r}"

    for key in ("duration_sec", "timeout_sec"):
        if key not in entry:
            raise ValueError(f"{ctx}: {key!r} is required")

    moving = tuple(
        _side_from_config(v, ctx) for v in entry.get("moving", ())
    )
    goal = _goal_from_config(entry.get("goal"), ctx)
    grip = {
        _side_from_config(k, ctx): float(v)
        for k, v in (entry.get("grip") or {}).items()
    }
    exit_cfg = entry.get("exit") or {}
    if not isinstance(exit_cfg, dict):
        raise ValueError(f"{ctx}: 'exit' must be a mapping")
    unknown = set(exit_cfg) - {"pos_tol_m", "rot_tol_rad", "grasp", "release"}
    if unknown:
        raise ValueError(f"{ctx}: unknown exit key(s) {sorted(unknown)}")
    exit_spec = ExitSpec(
        pos_tol_m=(
            float(exit_cfg["pos_tol_m"]) if "pos_tol_m" in exit_cfg else None
        ),
        rot_tol_rad=(
            float(exit_cfg["rot_tol_rad"]) if "rot_tol_rad" in exit_cfg else None
        ),
        grasp=(
            _side_from_config(exit_cfg["grasp"], ctx) if "grasp" in exit_cfg else None
        ),
        release=(
            _side_from_config(exit_cfg["release"], ctx)
            if "release" in exit_cfg
            else None
        ),
    )
    retry_from = entry.get("retry_from")
    return StageSpec(
        name=name,
        goal=goal,
        moving=moving,
        duration_sec=float(entry["duration_sec"]),
        timeout_sec=float(entry["timeout_sec"]),
        grip=grip,
        exit=exit_spec,
        retry_from=None if retry_from is None else str(retry_from),
    )


def _goal_from_config(cfg: object, ctx: str) -> Optional[GoalSpec]:
    if cfg is None:
        return None
    if not isinstance(cfg, dict) or "kind" not in cfg:
        raise ValueError(f"{ctx}: 'goal' must be a mapping with 'kind'")
    try:
        kind = GoalKind(str(cfg["kind"]))
    except ValueError as exc:
        raise ValueError(
            f"{ctx}: invalid goal kind {cfg['kind']!r} "
            f"(valid: {[k.value for k in GoalKind]})"
        ) from exc

    if kind is GoalKind.FIXED:
        if "pose" not in cfg:
            raise ValueError(f"{ctx}: goal(fixed) requires 'pose' (6 values)")
        return GoalSpec(kind=kind, pose=EEPose.from_vec6(cfg["pose"]))
    if kind is GoalKind.CV:
        unknown = set(cfg) - {"kind", "offset", "rpy_offset"}
        if unknown:
            raise ValueError(f"{ctx}: goal(cv) unknown key(s) {sorted(unknown)}")
        parsed: dict[str, np.ndarray] = {}
        for key in ("offset", "rpy_offset"):
            if key not in cfg:
                continue
            v = np.asarray(cfg[key], dtype=np.float64).reshape(-1)
            if v.shape != (3,):
                raise ValueError(f"{ctx}: goal(cv) {key!r} must have 3 values")
            parsed[key] = v
        return GoalSpec(kind=kind, **parsed)

    for key in ("reference", "offset", "rpy"):
        if key not in cfg:
            raise ValueError(f"{ctx}: goal(relative) requires {key!r}")
    offset = np.asarray(cfg["offset"], dtype=np.float64).reshape(-1)
    rpy = np.asarray(cfg["rpy"], dtype=np.float64).reshape(-1)
    if offset.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"{ctx}: goal(relative) 'offset'/'rpy' must have 3 values")
    return GoalSpec(
        kind=kind,
        reference=_side_from_config(cfg["reference"], ctx),
        offset=offset,
        rpy=rpy,
    )
