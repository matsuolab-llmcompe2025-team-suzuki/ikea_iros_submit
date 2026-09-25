"""実 G1 で Orchestrator を回す実行 entry point (runtime env required)。

# 前提

- **Runtime env**: Python 3.10 + `unitree_sdk2py` + `cyclonedds` (rclpy は使わない、
  Issue #58 参照)
- **G1 が walking FSM state (立ち姿勢) に既に入っている** こと。Damp からの startup は
  operator が Unitree 標準リモコン等で事前に済ませておく (scope 外)。
- **ROS2 camera driver** が `--topic` に指定した CompressedImage を publish 中
- ハーネス / E-stop / clearance 確保 (安全確認)

# 起動 flow (init → run → shutdown)

    init:
        skill_config.yaml load →
        G1SDKWalkActuator init (= SDK ChannelFactory init) → YoloObbPerception load →
        DetectionStream init → Skill registry (initial=SetupSkill、
            move_to_table=MoveToTable、move_table_base=SampleVLASkill、他=MockSkill) →
        JointStateSource subscribe (Issue #75) →
        G1ArmActuator init (start is deferred until all preflight gates pass) →
        Ros2FrameSource subscribe (packed stereo) →
        YOLO には左眼、VLA には左右眼を渡す →
        Orchestrator init (initial_skill="setup"、log_sink 開く、
            actuator_send_fn=arm.send_action)
    run:
        orch.run_live(source, hz=30) を main thread で loop、以下を繰り返し:
            source.get() → tick → YOLO → cleaner → state → fire → dispatch
        初回frame待機 / frame freshness を監視。setup skill の Type B
        step() が per-tick に arm 姿勢を送り、SetupSkill.max_dwell_sec
        (= stage dwell 合計) 経過で auto-transition → move_to_table (walk) →
        max_dwell_sec 経過で auto-transition → move_table_base (VLA / mock)。
    shutdown (Ctrl+C / SIGINT / LiveSourceSafetyError / Phase stop boundary):
        actuator.set_velocity(0,0,0,duration=1) でwalking FSMを維持 →
        arm_sdk weight 1→0 controlled release → arm_actuator.stop() →
        DDS camera reader close → log 閉じ
        (姿勢/FSM命令は送らず、Dampはoperatorが明示的に実施 = scope外)

# Usage

    pixi run -e runtime python -m inference.desktop.entrypoint \\
        --interface eth0 \\
        --topic /head/camera/color/image_raw/compressed \\
        --log outputs/orch_run.jsonl

    Skill 姿勢定義 (vx / max_dwell_sec / default_pose_rad 等) は
    `inference/desktop/lower_policy/configs/skill_config.yaml` から読み込む。
    別の YAML を指定したい時は `--skill-config <path>` で override。

# 現状の Skill 実装状況

    setup:             SetupSkill     (Type B、YAML 定義の stage を per-tick で流す
                       腕 pre-motion。Issue #81 で追加、initial skill)
    move_to_table:     MoveToTable    (実 SDK 経由の walk、Type A)
    move_table_base:   SampleVLASkill (Type B、両腕水平 fixed pose を返す。
                       real VLA 実装時に step() を model.predict(obs) に差し替え)
    pick_table_leg / insert_table_leg / rotate_leg_to_tighten / flip_table:
                       MockSkill      (no-op、log のみ)

VLA / GR00T 系の実 model 差し込みは別 Epic。SampleVLASkill は drop-in template。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO

import yaml

# JSONL log 自動生成先の default dir (旧 resume module から migrate)。
_DEFAULT_LOG_DIR: Path = Path("outputs") / "orch_logs"

HEAD_CAMERA_STEREO_VIEW = "packed"
HEAD_PERCEPTION_VIEW = "left"

# GPU に置く model の既定数 (--gpu-models で上書き)。「今 + 次」で切替が隠れる。
DEFAULT_RESIDENT_MODELS = 2
PHASE1_11_POLICY_VARIANT = "ramen_ori_default"
PHASE1_11_SETUP_SECONDS = 4.0

# Policy workers are subprocess-backed for GR00T. Register every successfully
# constructed policy immediately so a later model-load/runtime failure cannot
# leave an orphan worker behind. The physical entrypoint is single-run, hence
# one process-local registry is sufficient and keeps cleanup available to the
# top-level exception boundary as well as the normal shutdown path.
_OPEN_POLICY_RESOURCES: list[object] = []


def _register_policy_resource(policy: object) -> object:
    _OPEN_POLICY_RESOURCES.append(policy)
    return policy


def _close_policy_resources() -> list[str]:
    failures: list[str] = []
    seen: set[int] = set()
    while _OPEN_POLICY_RESOURCES:
        policy = _OPEN_POLICY_RESOURCES.pop()
        identity = id(policy)
        if identity in seen:
            continue
        seen.add(identity)
        close = getattr(policy, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as exc:
            failures.append(f"{type(policy).__name__}: {exc}")
    return failures


PHASE1_11_WALK_VX_M_S = 0.185
PHASE1_11_WALK_SECONDS = 1.0
PHASE1_POST_WALK_SETTLE_SECONDS = 1.0
PHASE1_11_VLA_SECONDS = 30.0
PHASE1_13_HAND_VELOCITY_RAD_S = 1.0

# skill_config.yaml の default path (Issue #81 Phase 2b、review comment #82 fix)。
# CWD 相対だと pixi task を repo root 外から呼んだ時に silent に "not found" になるので、
# entrypoint module (この __file__) 相対で resolve する。
_DEFAULT_SKILL_CONFIG: Path = (
    Path(__file__).resolve().parent / "lower_policy" / "configs" / "skill_config.yaml"
)
_DEFAULT_PICK_LEG_HYBRID_CONFIG: Path = (
    Path(__file__).resolve().parent
    / "pick_leg_hybrid"
    / "configs"
    / "pick_leg_hybrid.yaml"
)
PRODUCTION_HYBRID_PICK_VARIANT = "groot_pick_legs_v1"


def _default_log_path() -> Path:
    """--log 未指定時の自動生成 path。

    outputs/orch_logs/orch_<yyyy-mm-ddTHH-MM-SS>.jsonl 形式。1 run 1 file、
    同秒起動でも既存 file を上書きしない前提で seconds 精度は十分。
    """
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    return _DEFAULT_LOG_DIR / f"orch_{ts}.jsonl"


def _default_record_dir() -> Path:
    """--record-dir 未指定時の出力先 (outputs/production_runs/<時刻>/)。"""
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    return _DEFAULT_LOG_DIR.parent / "production_runs" / ts


# ---- 実行時にしか必要ない heavy import は main() 内で lazy に ----
# (SDK / cyclonedds は runtime env にしか無く、module top で import すると
#  test collection や `python -m inference.desktop.entrypoint --help` すら
#  ImportError になる。lazy import で `--help` は main env でも動くように)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"value must be a positive finite number, got {value!r}"
        )
    return parsed


def _start_hand_during_registry_build(
    *, phase1_profile: str | None, phase3_active: bool
) -> bool:
    """Whether legacy setup may start Dex1 before an operator gate.

    Every Phase 3 path, including ``--phase3-full`` where ``args.stage`` is
    intentionally None, must defer the publisher until the final safety gate.
    """

    return phase1_profile != "1.13" and not phase3_active


def _require_real_waist_and_hand(
    args: argparse.Namespace,
    label: str,
    *,
    need_waist: bool = True,
    need_hand: bool = True,
) -> None:
    """Phase 3 は実 waist / 実 hand を要求する (#128 の段階的 smoke ガード)。

    例外は `--synthetic-hand-state` のときだけ。会場のリグは hand state を配信せず
    (`boundary/states.py`)、実 state が無いまま `prime_hold` するとグリッパが
    実姿勢と食い違って飛ぶので、**実ハンドを使わない代わりに state を合成する**
    という運用に倒す。その場合 hand actuator は mock 固定 = グリッパには
    一切指令を出さないので、安全側に外れる。

    `--action-sink boundary` はこの要求ごと外れる。腕・手を `(T,25)` で運営 WBC に
    渡すので SDK 直の実 actuator は要らない (併用は `_validate_phase3_config` が
    禁止している)。**腰は policy の指令を使わず常に実測値**: 運営 adapter は
    torso 列 `[22:25]` を読まず、IK は腰を実測値に固定する (`BOUNDARY_WAIST_NOTE`)。
    **会場でグリッパを動かせるのはこの経路だけ**なので、ここを塞ぐと
    stage 1-4 が丸ごと起動できなくなる。

    `need_waist` / `need_hand` は **選んだ Stage 範囲**で決まる (Issue #148)。
    腰を使うのは脚 Stage 1..4 だけ、手を使うのは Stage 1..4 と flip の Stage 5 だけ
    なので、Stage 0 だけ・Stage 5 だけの run では要求しないどころか**禁止する**
    (要らない所有権を取ると、その関節が Regular Mode から外れたまま残る)。
    """

    if getattr(args, "action_sink", "sdk") == "boundary":
        # hand 指令は共有 mock actuator の `.latest` 経由で (T,25) の手の列に載る。
        # `--synthetic-hand-state` が無いと `_build_hand_actuator` が skill ごとに
        # **別 instance** を返すため、`.latest` が boundary sink に届かず
        # **グリッパ指令が黙って落ちる** (エラーは出ない)。ここで要求しておく。
        #
        # 実 actuator 側の可否は見なくてよい: boundary では `--use-real-waist` も
        # `--use-real-hand` も `_validate_phase3_config` が既に禁止している。
        if need_hand and not getattr(args, "synthetic_hand_state", False):
            raise ValueError(
                f"{label} with --action-sink boundary requires"
                " --synthetic-hand-state (otherwise each skill gets its own mock"
                " hand actuator and the (T,25) hand columns never see the"
                " policy's gripper commands)"
            )
        return

    if need_waist and not args.use_real_waist:
        raise ValueError(f"{label} requires --use-real-waist")
    if not need_waist and args.use_real_waist:
        raise ValueError(
            f"{label} keeps waist/legs under Regular Mode;"
            " --use-real-waist is forbidden"
        )
    if not need_hand:
        if args.use_real_hand:
            raise ValueError(f"{label} owns arms only; --use-real-hand is forbidden")
        return
    if args.use_real_hand:
        return
    if getattr(args, "synthetic_hand_state", False):
        return
    raise ValueError(
        f"{label} requires --use-real-hand"
        " (or --synthetic-hand-state if the rig does not publish Dex1 state)"
    )


#: policy slot → `--policy-variant-*` の属性名。
_POLICY_SLOT_ATTRS: dict[str, str] = {
    "move": "policy_variant",
    "rotate_table": "policy_variant_rotate_table_base",
    "pick": "policy_variant_pick",
    "insert": "policy_variant_insert",
    "rotate_leg": "policy_variant_rotate_leg",
    "flip": "policy_variant_flip",
}
#: policy slot → skill 名 (`policy_config.yaml:default_variant_by_skill` の key)。
#: move_table_base は Phase 3 の stage 列に無いので既定を持たない。
_POLICY_SLOT_SKILLS: dict[str, str] = {
    "rotate_table": "rotate_table_base",
    "pick": "pick_table_leg",
    "insert": "insert_table_leg",
    "rotate_leg": "rotate_leg_to_tighten",
    "flip": "flip_table",
}


def _return_arms_and_open_hand(
    *,
    lowered_pose: "np.ndarray",
    arm_actuator: object,
    joint_state_source: object,
    hand_actuator: Optional[object] = None,
    dex1_state_source: Optional[object] = None,
    lowering_settings: Optional[dict] = None,
    hand_open_rad: Optional[float] = None,
    hz: float = 30.0,
    timeout_s: float = 60.0,
    recorder: Optional[object] = None,
    time_fn=time.monotonic,
    sleep_fn=time.sleep,
    measured_convergence_checker=None,
    open_only: bool = False,
) -> None:
    """run の終わりに手を開き、腕を下ろしてから解放する (Issue #152)。

    正常終了・Ctrl+C・安全停止のどれでも同じ処理を通す。arm_sdk の weight は 1 の
    ままここを走り、終わってから `_safe_shutdown` が weight を 0 へ落とす。
    途中でもう一度 Ctrl+C を押したら中断して即解放へ進む (危ないときは E-stop)。
    """
    # lazy: runtime env only (numpy 以外は SDK 依存の module を引き込むため)
    import numpy as np

    from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
        CollisionAwareArmPreMotionSkill,
    )
    from inference.desktop.lower_policy.skills.hand_ramp import (
        build_hand_open_ramp,
        resolve_hand_opening_rad,
    )

    def _event(payload: dict) -> None:
        if recorder is not None:
            recorder.write_event(payload)  # type: ignore[attr-defined]

    print(
        "[return] opening Dex1 only; no arm motion was commanded"
        if open_only else "[return] opening Dex1 and lowering the arms",
        file=sys.stderr,
    )
    _event({"event": "return_started", "monotonic_ns": time.monotonic_ns()})

    # 1) 手を開く (掴んだままだと次の stage の初期状態を作れない)
    if hand_actuator is not None and dex1_state_source is not None:
        state = dex1_state_source.get()  # type: ignore[attr-defined]
        measured_hand = getattr(state, "position_rad", None)
        if measured_hand is not None:
            opening = resolve_hand_opening_rad({}) if hand_open_rad is None else hand_open_rad
            for target in build_hand_open_ramp(
                np.asarray(measured_hand, dtype=np.float64), open_rad=opening, command_hz=hz
            ):
                hand_actuator.send_action(target.tolist())  # type: ignore[attr-defined]
                # The venue boundary publishes arm/waist/hand atomically.  A
                # hand-only mock update is not physical until an arm row is
                # emitted, so re-emit the held arm target on every ramp step.
                read_last = getattr(arm_actuator, "read_last_published_targets", None)
                if callable(read_last):
                    arm_actuator.send_action(read_last()[0])  # type: ignore[attr-defined]
                sleep_fn(1.0 / hz)

    if open_only:
        _event({
            "event": "return_skipped_no_arm_motion",
            "monotonic_ns": time.monotonic_ns(),
        })
        return

    # 2) 起動時の退避を逆にたどって下ろす
    skill = CollisionAwareArmPreMotionSkill(
        tuple(np.asarray(lowered_pose, dtype=np.float64).tolist()),
        skill_name="return_to_lowered",
        waypoint_profile="reverse_clearance",
        time_fn=time_fn,
        measured_convergence_checker=measured_convergence_checker,
        **(lowering_settings or {}),
    )
    skill.start({})
    started = time_fn()
    while not skill.is_complete:
        if time_fn() - started > timeout_s:
            print(
                f"[return] lowering did not finish within {timeout_s:g}s; "
                "releasing from the current pose",
                file=sys.stderr,
            )
            _event({"event": "return_timeout", "monotonic_ns": time.monotonic_ns()})
            return
        joint_state = joint_state_source.get()  # type: ignore[attr-defined]
        command = skill.step({"joint_state": joint_state})
        # 終了時の退避は安全のための手順なので、時間切れでも今の姿勢から解放する。
        stopped = skill.failure_reason or skill.timeout_reason
        if stopped is not None:
            print(f"[return] lowering failed: {stopped}", file=sys.stderr)
            _event(
                {
                    "event": "return_failed",
                    "reason": stopped,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            return
        arm_actuator.send_action(command.tolist())  # type: ignore[attr-defined]
        sleep_fn(1.0 / hz)
    print("[return] arms are lowered; controlled release follows", file=sys.stderr)
    _event({"event": "return_complete", "monotonic_ns": time.monotonic_ns()})


def _stop_base_before_return(actuator: object) -> None:
    """return の前に歩行速度を 0 にする。例外は出さない (後始末を止めないため)。"""
    try:
        actuator.set_velocity(0.0, 0.0, 0.0, duration=1.0)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        print(f"[return] base stop before lowering failed: {e!r}", file=sys.stderr)


def _return_before_release(
    *,
    skill_config: dict,
    arm_actuator: object,
    joint_state_source: object,
    hand_actuator: Optional[object] = None,
    dex1_state_source: Optional[object] = None,
    recorder: Optional[object] = None,
    measured_convergence_checker=None,
    open_only: bool = False,
) -> None:
    """解放の直前に必ず通る後始末 (Issue #152)。何があっても例外を出さない。

    2 回目の Ctrl+C は「戻すのをやめて即解放」の意味に取る。危ないときは E-stop。
    """
    lowered_pose = _walk_lowered_pose(skill_config)
    if lowered_pose is None:
        print(
            "[return] skills.walk_lowered_pose is not configured; releasing "
            "from the current pose",
            file=sys.stderr,
        )
        return
    try:
        from inference.desktop.lower_policy.skills.hand_ramp import resolve_hand_opening_rad

        _return_arms_and_open_hand(
            lowered_pose=lowered_pose,
            arm_actuator=arm_actuator,
            joint_state_source=joint_state_source,
            hand_actuator=hand_actuator,
            dex1_state_source=dex1_state_source,
            lowering_settings=_walk_lowering_settings(skill_config),
            hand_open_rad=resolve_hand_opening_rad(skill_config),
            recorder=recorder,
            measured_convergence_checker=measured_convergence_checker,
            open_only=open_only,
        )
    except KeyboardInterrupt:
        print("[return] interrupted; releasing from the current pose", file=sys.stderr)
    except Exception as return_error:  # noqa: BLE001
        print(
            f"[return] failed: {return_error!r}; releasing from the current pose",
            file=sys.stderr,
        )


def _walk_lowered_pose(skill_config: dict) -> "np.ndarray | None":
    """歩く前に腕を下ろす先 (`skills.walk_lowered_pose.pose_rad`)。

    未設定なら None を返し、従来どおり「範囲の外なら止める」になる (Issue #152)。
    """
    from inference.desktop.lower_policy.pose_utils import densify_pose

    section = (skill_config.get("skills") or {}).get("walk_lowered_pose")
    if section is None:
        return None
    return densify_pose(section.get("pose_rad"), context="walk_lowered_pose")


def _walk_latch_check(skill_config: dict, override: Optional[str] = None) -> str:
    """歩く前に下ろした腕を保持する前の確かめ方 (`skills.walk_lowered_pose.latch_check`)。

    ``override`` (起動の ``--walk-lowering-check``) があればそちらを使う (接続テスト用)。
    """
    if override is not None:
        return override
    section = (skill_config.get("skills") or {}).get("walk_lowered_pose") or {}
    return str(section.get("latch_check", "joint"))


def _walk_lowering_settings(skill_config: dict) -> dict:
    """腕を下ろすときの速度・収束の値 (`arm_pre_motion` を流用する)。"""
    settings = dict(skill_config.get("arm_pre_motion") or {})
    settings.pop("boundary_lift_rad", None)
    return settings


def required_policy_slots(args: argparse.Namespace) -> tuple[str, ...]:
    """この run が読む policy slot (Issue #148)。

    検証 (足りない / 使わない) と既定の充填が同じ規則を使う。stage を選んでいない
    run は空を返すので、rule-based pick のように variant を持たない経路では
    何も埋まらない。
    """

    if getattr(args, "phase3_full", False):
        stages = set(range(args.phase3_start_stage, args.phase3_end_stage + 1))
    elif args.stage is not None:
        stages = {args.stage}
    else:
        return ()
    return (
        (("pick", "insert", "rotate_leg") if stages & {1, 2, 3, 4} else ())
        + (("rotate_table",) if stages & {2, 3, 4} else ())
        + (("flip",) if 5 in stages else ())
    )


def apply_default_policy_variants(
    args: argparse.Namespace, *, config_path: Path
) -> dict[str, str]:
    """CLI が渡さなかった slot を `policy_config.yaml` の既定で埋める (Issue #148)。

    埋めるのはこの run が要る slot だけ。stage 0 に variant が載ったり、stage 1 に
    rotate_table_base が載ったりすると `_validate_phase3_config` が弾くので、
    「全部埋めてから捨てる」ではなく最初から要るものだけを埋める。

    Returns:
        slot 名 → "config" / "cli" (起動ログと記録の metadata 用)。
    """

    from inference.desktop.lower_policy.policies.config_loader import (
        load_default_variant_by_skill,
    )

    required = required_policy_slots(args)
    if not required:
        return {}
    base = load_default_variant_by_skill(config_path)
    # `--policy-variant-set` は variant_sets の組み合わせを既定に重ねる (Issue #159)。
    # slot ごとの `--policy-variant-*` はこの後も優先する。
    variant_set = getattr(args, "policy_variant_set", None)
    defaults = (
        load_default_variant_by_skill(config_path, variant_set=variant_set)
        if variant_set
        else base
    )
    unknown = sorted(set(defaults) - set(_POLICY_SLOT_SKILLS.values()))
    if unknown:
        raise ValueError(
            f"{config_path}: default_variant_by_skill has unknown skills {unknown} "
            f"(valid: {sorted(_POLICY_SLOT_SKILLS.values())})"
        )
    source: dict[str, str] = {}
    for slot in required:
        attr = _POLICY_SLOT_ATTRS[slot]
        if getattr(args, attr) is not None:
            source[slot] = "cli"
            continue
        variant = defaults.get(_POLICY_SLOT_SKILLS[slot])
        if variant is None:
            raise ValueError(
                f"{config_path}: default_variant_by_skill is missing "
                f"{_POLICY_SLOT_SKILLS[slot]!r}; pass --{attr.replace('_', '-')} "
                "explicitly or add the default"
            )
        setattr(args, attr, variant)
        skill = _POLICY_SLOT_SKILLS[slot]
        source[slot] = (
            f"set:{variant_set}" if variant_set and variant != base.get(skill) else "config"
        )
    return source


def resolve_production_pick_mode(args: argparse.Namespace) -> bool:
    """Default Stage 1--4 SDK and boundary runs to the hybrid pick pipeline.

    The production pick is the complete VLM -> GR00T first grasp -> IK/MP ->
    rule-based handover pipeline.  Its terminal state is the insert model's
    measured frame-zero pose.  Previously, forgetting ``--pick-leg-hybrid``
    silently selected the old learned-only path.

    Returns True when this function installed the production default.  An
    explicit ``--no-pick-leg-hybrid`` remains available for comparison tests.
    """

    if getattr(args, "phase3_full", False):
        selected = set(range(args.phase3_start_stage, args.phase3_end_stage + 1))
    elif getattr(args, "stage", None) is not None:
        selected = {args.stage}
    else:
        selected = set()
    requested = getattr(args, "pick_leg_hybrid", None)
    if requested is not None:
        return False
    needs_pick = bool(selected.intersection({1, 2, 3, 4}))
    # The production pick algorithm must not change merely because the venue
    # transport is selected.  The official rig omits Dex1 feedback, so the
    # boundary lane uses the required command echo plus visual/VLM checks; it
    # must not silently fall back to the obsolete learned-only pick expert.
    enabled = needs_pick
    args.pick_leg_hybrid = enabled
    return enabled


def _validate_phase3_config(args: argparse.Namespace) -> None:
    """Reject unsafe or resource-wasting Phase 3 CLI combinations."""

    if (
        getattr(args, "boundary_lane", "pose") != "pose"
        and getattr(args, "action_sink", "sdk") != "boundary"
    ):
        raise ValueError("--boundary-lane joint requires --action-sink boundary")
    if getattr(args, "wrist_roll_clamp", "on") != "on" and not _boundary_taskspace_arrival(
        args
    ):
        # 効かない組み合わせを弾く (接続テストで「外したつもり」を作らない)
        raise ValueError(
            "--wrist-roll-clamp off applies only to --action-sink boundary on the pose lane "
            "(the joint lane never goes through the organizer IK)"
        )
    if getattr(args, "action_sink", "sdk") == "boundary":
        # boundary は腕・手を (T,25) で運営 WBC に渡す (腰は実測値のまま)。SDK 直の
        # 実 actuator を併用すると同じ関節を二重に動かす。
        if args.use_real_hand:
            raise ValueError(
                "--action-sink boundary and --use-real-hand are mutually exclusive"
                " (the gripper is driven through the (T,25) hand columns instead)"
            )
        if args.use_real_waist:
            raise ValueError(
                "--action-sink boundary and --use-real-waist are mutually exclusive"
                " (the organizer's adapter holds the waist at its measured angle)"
            )

    if getattr(args, "synthetic_hand_state", False) and args.use_real_hand:
        # 合成 state で実グリッパを駆動すると prime_hold が推測値から始まる。
        # 実 state があるリグなら --use-real-hand だけでよい。
        raise ValueError(
            "--synthetic-hand-state and --use-real-hand are mutually exclusive"
        )

    stage = args.stage
    full = bool(getattr(args, "phase3_full", False))
    rule_pick = bool(getattr(args, "rule_based_pick_table_leg", False))
    if rule_pick:
        if getattr(args, "action_sink", "sdk") == "boundary":
            raise ValueError(
                "--rule-based-pick-table-leg still owns a lab-only Dex1 DDS "
                "publisher and cannot be combined with the official boundary lane"
            )
        if stage is not None or full:
            raise ValueError(
                "--rule-based-pick-table-leg is mutually exclusive with "
                "--stage/--phase3-full"
            )
        if any(
            getattr(args, name) is not None
            for name in (
                "policy_variant",
                "policy_variant_pick",
                "policy_variant_insert",
                "policy_variant_rotate_leg",
                "policy_variant_flip",
                "policy_variant_rotate_table_base",
            )
        ):
            raise ValueError(
                "--rule-based-pick-table-leg must not load learned policy variants"
            )
        if args.use_real_waist or args.use_real_hand:
            raise ValueError(
                "rule-based pick owns arms and its dedicated Dex1 path only; "
                "--use-real-waist/--use-real-hand are forbidden"
            )
        if args.no_wrist_cameras:
            raise ValueError(
                "rule-based pick requires both wrist cameras for monitoring"
            )
        return
    if getattr(args, "rule_based_pick_calibration", None) is not None:
        raise ValueError(
            "--rule-based-pick-calibration requires --rule-based-pick-table-leg"
        )
    if stage is not None and full:
        raise ValueError("--stage and --phase3-full are mutually exclusive")
    if stage is None and not full:
        if args.actuate:
            raise ValueError(
                "--actuate is only supported with --stage or --phase3-full"
            )
        return
    if args.no_wrist_cameras:
        raise ValueError("Phase 3 requires both wrist-camera observations")

    variants = {
        slot: getattr(args, attr) for slot, attr in _POLICY_SLOT_ATTRS.items()
    }
    required = required_policy_slots(args)
    hybrid_pick = bool(getattr(args, "pick_leg_hybrid", False))
    if hybrid_pick:
        selected_stages = (
            set(range(args.phase3_start_stage, args.phase3_end_stage + 1))
            if full
            else ({stage} if stage is not None else set())
        )
        if not selected_stages.intersection({1, 2, 3, 4}):
            raise ValueError("--pick-leg-hybrid requires a selected Stage in 1..4")
        if args.no_wrist_cameras:
            raise ValueError("--pick-leg-hybrid requires both wrist cameras")
        boundary_hand = (
            getattr(args, "action_sink", "sdk") == "boundary"
            and getattr(args, "synthetic_hand_state", False)
        )
        if not args.use_real_hand and not boundary_hand:
            raise ValueError(
                "--pick-leg-hybrid requires --use-real-hand on the SDK lane or "
                "--synthetic-hand-state on the boundary lane"
            )
        if variants["pick"] is None:
            raise ValueError(
                "--pick-leg-hybrid requires --policy-variant-pick for its "
                "Phase 1 GR00T expert"
            )
        if variants["pick"] != PRODUCTION_HYBRID_PICK_VARIANT:
            raise ValueError(
                "production --pick-leg-hybrid is pinned to the physically "
                f"validated variant {PRODUCTION_HYBRID_PICK_VARIANT!r}; got "
                f"{variants['pick']!r}"
            )
        if getattr(args, "pick_leg_phase3_executor", "rule_based") != "rule_based":
            raise ValueError(
                "production --pick-leg-hybrid requires "
                "--pick-leg-phase3-executor rule_based because the VLA Phase 3 "
                "path has no finite completion contract"
            )
    if full:
        if args.phase3_start_stage > args.phase3_end_stage:
            raise ValueError("--phase3-start-stage must be <= --phase3-end-stage")
        selected_stages = set(range(args.phase3_start_stage, args.phase3_end_stage + 1))
        uses_leg_models = bool(selected_stages.intersection({1, 2, 3, 4}))
        uses_flip = 5 in selected_stages
        missing = [name for name in required if variants[name] is None]
        if missing:
            raise ValueError(
                "Phase 3 full run is missing policy variants: " + ",".join(missing)
            )
        if variants["move"] is not None:
            raise ValueError("Phase 3 full run must not load move_table_base")
        unused = [
            name
            for name in ("rotate_table", "pick", "insert", "rotate_leg", "flip")
            if name not in required and variants[name] is not None
        ]
        if unused:
            raise ValueError(
                "Phase 3 full run includes unused policy variants: " + ",".join(unused)
            )
        _require_real_waist_and_hand(
            args,
            f"Phase 3 full run (stages {args.phase3_start_stage}"
            f"..{args.phase3_end_stage})",
            need_waist=uses_leg_models,
            need_hand=uses_leg_models or uses_flip,
        )
        return
    if stage == 0:
        if any(value is not None for value in variants.values()):
            raise ValueError(
                "Phase 3 stage 0 is walk + arm pre-motion only; policy variants "
                "must not be loaded"
            )
        if args.use_real_waist or args.use_real_hand:
            raise ValueError("Phase 3 stage 0 owns arms only")
        return

    if stage in {1, 2, 3, 4}:
        missing = [name for name in required if variants[name] is None]
        if missing:
            raise ValueError(
                f"Phase 3 stage {stage} is missing policy variants: "
                + ",".join(missing)
            )
        unused = [
            name
            for name in ("move", "rotate_table", "pick", "insert", "rotate_leg", "flip")
            if name not in required and variants[name] is not None
        ]
        if unused:
            raise ValueError(
                f"Phase 3 stage {stage} includes unused policies: " + ",".join(unused)
            )
        _require_real_waist_and_hand(args, f"Phase 3 stage {stage}")
        return

    if variants["flip"] is None:
        raise ValueError("Phase 3 stage 5 requires --policy-variant-flip")
    if any(
        variants[name] is not None
        for name in ("move", "rotate_table", "pick", "insert", "rotate_leg")
    ):
        raise ValueError("Phase 3 stage 5 must load only the flip policy")
    if args.use_real_waist:
        raise ValueError(
            "Phase 3 stage 5 keeps waist/legs under Regular Mode; "
            "--use-real-waist is forbidden"
        )
    if getattr(args, "action_sink", "sdk") == "boundary":
        # boundary は手も `(T,25)` の hand 列で運営 WBC に渡すので SDK 直の実 hand は
        # 使わない (併用は上の `_validate_phase3_config` が禁止している)。
        # stage 1-4 と同じ理由で `--synthetic-hand-state` を要求する: 無いと
        # `_build_hand_actuator` が skill ごとに別 instance を返し、`.latest` が
        # boundary sink に届かず **グリッパ指令が黙って落ちる**。
        #
        # ここに例外が無いと `--stage 5 --action-sink boundary` が成立しない
        # (boundary は --use-real-hand を禁止、stage 5 はそれを要求する)。
        # flip は **グリッパが最も効く skill** で、会場で flip だけやり直す手段が
        # 塞がる (`--phase3-full` 経由は `_require_real_waist_and_hand` の例外を
        # 通るので動く。単独 stage だけが落ちていた)。
        if not getattr(args, "synthetic_hand_state", False):
            raise ValueError(
                "Phase 3 stage 5 with --action-sink boundary requires"
                " --synthetic-hand-state (otherwise each skill gets its own mock"
                " hand actuator and the (T,25) hand columns never see the"
                " policy's gripper commands)"
            )
    elif not args.use_real_hand:
        raise ValueError("Phase 3 stage 5 requires --use-real-hand")


def _validate_phase1_11_config(args: argparse.Namespace, skills: dict) -> None:
    """Fail closed unless the requested Phase 1.11 arm-only contract is exact."""

    if args.policy_variant != PHASE1_11_POLICY_VARIANT:
        raise ValueError(
            f"--phase1-11-arm-only requires --policy-variant {PHASE1_11_POLICY_VARIANT}"
        )
    if args.use_real_waist or args.use_real_hand:
        raise ValueError(
            "Phase 1.11 is arm-only; --use-real-waist/--use-real-hand are forbidden"
        )
    if args.no_wrist_cameras:
        raise ValueError("Phase 1.11 requires both wrist-camera observations")
    if any(
        value is not None
        for value in (
            args.policy_variant_pick,
            args.policy_variant_insert,
            args.policy_variant_rotate_leg,
            args.policy_variant_flip,
            args.policy_variant_rotate_table_base,
        )
    ):
        raise ValueError("Phase 1.11 permits only the move_table_base policy variant")

    try:
        if not skills["setup"]["stages"]:
            raise ValueError("Phase 1.11 setup must provide the final policy pose")
        walk_vx = float(skills["move_to_table"]["vx"])
        walk_seconds = float(skills["move_to_table"]["max_dwell_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Phase 1.11 skill config is incomplete") from exc
    expected = (
        ("walk vx", walk_vx, PHASE1_11_WALK_VX_M_S),
        ("walk dwell", walk_seconds, PHASE1_11_WALK_SECONDS),
    )
    for label, actual, wanted in expected:
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Phase 1.11 {label} mismatch: actual={actual:g}, expected={wanted:g}"
            )


def _validate_phase1_12_config(args: argparse.Namespace, skills: dict) -> None:
    """Fail closed unless the requested Phase 1.12 arm+waist contract is exact."""

    if args.policy_variant != PHASE1_11_POLICY_VARIANT:
        raise ValueError(
            "--phase1-12-waist-real requires "
            f"--policy-variant {PHASE1_11_POLICY_VARIANT}"
        )
    if not args.use_real_waist:
        raise ValueError("Phase 1.12 requires the real waist actuator")
    if args.use_real_hand:
        raise ValueError("Phase 1.12 keeps Dex1 Mock; --use-real-hand is forbidden")
    if args.no_wrist_cameras:
        raise ValueError("Phase 1.12 requires both wrist-camera observations")
    if any(
        value is not None
        for value in (
            args.policy_variant_pick,
            args.policy_variant_insert,
            args.policy_variant_rotate_leg,
            args.policy_variant_flip,
            args.policy_variant_rotate_table_base,
        )
    ):
        raise ValueError("Phase 1.12 permits only the move_table_base policy variant")

    try:
        if not skills["setup"]["stages"]:
            raise ValueError("Phase 1.12 setup must provide the final policy pose")
        walk_vx = float(skills["move_to_table"]["vx"])
        walk_seconds = float(skills["move_to_table"]["max_dwell_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Phase 1.12 skill config is incomplete") from exc
    expected = (
        ("walk vx", walk_vx, PHASE1_11_WALK_VX_M_S),
        ("walk dwell", walk_seconds, PHASE1_11_WALK_SECONDS),
    )
    for label, actual, wanted in expected:
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Phase 1.12 {label} mismatch: actual={actual:g}, expected={wanted:g}"
            )


def _validate_phase1_13_config(args: argparse.Namespace, skills: dict) -> None:
    """Fail closed unless arms, waist and Dex1 are the only real outputs."""

    if args.policy_variant != PHASE1_11_POLICY_VARIANT:
        raise ValueError(
            "--phase1-13-hand-real requires "
            f"--policy-variant {PHASE1_11_POLICY_VARIANT}"
        )
    if not args.use_real_waist or not args.use_real_hand:
        raise ValueError("Phase 1.13 requires real waist and real Dex1 actuators")
    if args.no_wrist_cameras:
        raise ValueError("Phase 1.13 requires both wrist-camera observations")
    if any(
        value is not None
        for value in (
            args.policy_variant_pick,
            args.policy_variant_insert,
            args.policy_variant_rotate_leg,
            args.policy_variant_flip,
            args.policy_variant_rotate_table_base,
        )
    ):
        raise ValueError("Phase 1.13 permits only the move_table_base policy variant")

    try:
        if not skills["setup"]["stages"]:
            raise ValueError("Phase 1.13 setup must provide the final policy pose")
        walk_vx = float(skills["move_to_table"]["vx"])
        walk_seconds = float(skills["move_to_table"]["max_dwell_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Phase 1.13 skill config is incomplete") from exc
    expected = (
        ("walk vx", walk_vx, PHASE1_11_WALK_VX_M_S),
        ("walk dwell", walk_seconds, PHASE1_11_WALK_SECONDS),
    )
    for label, actual, wanted in expected:
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Phase 1.13 {label} mismatch: actual={actual:g}, expected={wanted:g}"
            )


def _require_strict_regular(
    actuator: object, *, phase_label: str = "Phase 1.11 arm-only"
) -> object:
    """Read-only Regular-mode gate immediately before any physical command."""

    status = actuator.get_loco_status()  # type: ignore[attr-defined]
    actual = (status.fsm_id, status.fsm_mode)
    if actual != (501, 0):
        raise RuntimeError(
            "G1 is not in strict Regular Mode immediately before actuation: "
            f"actual={actual}, expected=(501, 0)"
        )
    print(
        f"[preflight] strict Regular=(501,0); {phase_label} ownership gate passed",
        file=sys.stderr,
    )
    return status


def _wait_for_phase1_11_sources(
    *,
    head_source: object,
    wrist_left_source: object,
    wrist_right_source: object,
    joint_state_source: object,
    dex1_state_source: object,
    timeout_s: float,
    phase_label: str = "Phase 1.11",
) -> None:
    """Wait for all read-only observations before arm_sdk can be started."""

    getters = {
        "head": head_source.get,
        "wrist_left": wrist_left_source.get,
        "wrist_right": wrist_right_source.get,
        "joint_state": joint_state_source.get,
        "dex1_state": dex1_state_source.get,
    }
    ready: set[str] = set()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for name, getter in getters.items():
            if name not in ready and getter() is not None:
                ready.add(name)
        if len(ready) == len(getters):
            print(
                "[preflight] head L/R + wrist L/R + joint + Dex1 observations ready; "
                "NO command sent",
                file=sys.stderr,
            )
            return
        time.sleep(0.01)
    missing = sorted(set(getters) - ready)
    raise RuntimeError(
        f"{phase_label} observation preflight timed out after {timeout_s:g}s; "
        f"missing={','.join(missing)}; no robot command was sent"
    )


def _validate_rule_pick_geometry_snapshot(
    *,
    head_frame: object,
    joint_state: object,
    perception: object,
    grasp_provider: object,
) -> object:
    """Run one command-free detector -> calibrated geometry -> IK preflight.

    The generic source preflight only proves that camera bytes arrive. A
    configuration can still detect legs while projecting every OBB to an
    implausibly short or unreachable object. Catch that before the operator
    prompt and before any arm/Dex1 publisher is started.
    """

    import numpy as np

    rgb = np.asarray(getattr(head_frame, "rgb", None))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(
            "rule-based pick geometry preflight received an invalid head frame; "
            "no robot command was sent"
        )
    width = int(rgb.shape[1])
    if width < 2 or width % 2 != 0:
        raise RuntimeError(
            "rule-based pick geometry preflight requires packed head stereo with "
            f"an even width, got shape={rgb.shape}; no robot command was sent"
        )
    # The detector and camera calibration are both defined for head-left.
    detections = perception.predict(rgb[:, : width // 2])
    leg_detections = [
        det
        for det in detections
        if getattr(det, "class_name", None)
        == getattr(grasp_provider, "leg_class", "leg")
        and float(getattr(det, "confidence", 0.0))
        >= float(getattr(grasp_provider, "min_confidence", 0.5))
    ]
    pose = grasp_provider.grasp_pose(
        {"cleaned": detections, "joint_state": joint_state}
    )
    if pose is None:
        raise RuntimeError(
            "rule-based pick geometry preflight found no safe reachable grasp: "
            f"eligible_leg_detections={len(leg_detections)} "
            f"last_reject_reason={getattr(grasp_provider, 'last_reject_reason', None)} "
            f"last_leg_length_m={getattr(grasp_provider, 'last_leg_length', None)}; "
            "verify the physical camera calibration, support plane, complete-leg "
            "visibility, and robot/table placement; no robot command was sent"
        )
    position = np.asarray(getattr(pose, "position", None), dtype=np.float64)
    rpy = np.asarray(getattr(pose, "rpy", None), dtype=np.float64)
    if (
        position.shape != (3,)
        or rpy.shape != (3,)
        or not (np.all(np.isfinite(position)) and np.all(np.isfinite(rpy)))
    ):
        raise RuntimeError(
            "rule-based pick geometry preflight produced a malformed grasp pose; "
            "no robot command was sent"
        )
    print(
        "[preflight] rule-based perception geometry ready; NO command sent "
        f"eligible_legs={len(leg_detections)} "
        f"grasp_xyz={np.round(position, 4).tolist()} "
        f"grasp_rpy={np.round(rpy, 4).tolist()} "
        f"leg_length_m={getattr(grasp_provider, 'last_leg_length', None)} "
        f"grasp_fraction={getattr(grasp_provider, 'last_grasp_fraction', None)}",
        file=sys.stderr,
    )
    return pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--interface",
        type=str,
        default="eth0",
        help="DDS network interface (ip a で確認、実機は eth0/enp2s0 等、sim は lo)",
    )
    p.add_argument(
        "--topic",
        type=str,
        default="/head/camera/color/image_raw/compressed",
        help="ROS2 CompressedImage topic 名 (Orin 側の camera driver が publish するもの)",
    )
    p.add_argument(
        "--skill-config",
        type=Path,
        default=_DEFAULT_SKILL_CONFIG,
        help=(
            "Skill 姿勢定義 YAML path (Issue #81 Phase 2b)。setup skill の stage 定義や"
            " move_to_table の vx / max_dwell_sec、move_table_base の default_pose_rad 等の"
            " skill config を集約。数値変更は YAML 編集で完結し、code / CLI arg を触らない。"
            " default は entrypoint module 相対で resolve (CWD 依存無し)"
        ),
    )
    p.add_argument(
        "--hz",
        type=_positive_float,
        default=30.0,
        help="Orchestrator tick rate [Hz] (default 30 = camera fps に合わせる)",
    )
    p.add_argument(
        "--camera-startup-timeout",
        type=_positive_float,
        default=10.0,
        help="最初のcamera frameを待つ上限秒数 (default 10)",
    )
    p.add_argument(
        "--frame-timeout",
        type=_positive_float,
        default=1.0,
        help="歩行中にcamera timestamp更新を待つ上限秒数 (default 1)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            "JSONL log path (省略時は outputs/orch_logs/orch_<timestamp>.jsonl に"
            " 自動生成、resume 経路が参照する既定 log)。"
            " 完全に disable したい場合は --no-log"
        ),
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help=(
            "JSONL log 出力を完全 disable (test / debug 用)。実運用では resume 経路が"
            " 参照する log を残すため基本 default 有効のまま使う"
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="YOLO device (cuda / cpu / None は auto)",
    )
    # Issue #125 Phase 9b: VLA policy 選択 (未指定なら既存 SampleVLASkill fallback)
    p.add_argument(
        "--policy-variant",
        type=str,
        default=None,
        help=(
            "move_table_base に使う learned policy variant 名 (Issue #125)。"
            " policy_config.yaml の policies section から選ぶ (例: ramen_ori_default)。"
            " 未指定なら既存 SampleVLASkill (pose mock)。groot_overlay は rotate_table_base"
            " 専用 (Issue #132 Phase B、prompt 齟齬回避)、move_table_base では選ばない。"
            " 指定時は VlaSkill (MoveTableBaseVlaSkill) を registry に置き、Waist/Hand"
            " actuator は現状 Mock (real SDK は body 到着後 Issue #63 A-2)。"
        ),
    )
    p.add_argument(
        "--policy-config",
        type=Path,
        default=Path("inference/desktop/lower_policy/configs/policy_config.yaml"),
        help="policy variant registry YAML (Issue #125 Phase 7)。default は inference/desktop 下。",
    )
    p.add_argument(
        "--gpu-models",
        default="1",
        help=(
            "GPU に置く learned model の数 (Issue #141 束 1-12)。1 (既定) = 実行中の"
            " 1 個だけ、2 = 実行中 + 次の 1 個を裏で先読み、all = stage の全部。"
            " 先読みは 1 本の thread で順番に行う。実機で tick の周期と GPU の使用量を"
            " 測ってから上げること"
        ),
    )
    # Issue #125 Phase B-1/B-3: VLA actuator の real 化 flag (opt-in で mock fallback)。
    p.add_argument(
        "--use-real-waist",
        action="store_true",
        help=(
            "waist actuator を G1WaistActuator (shared G1ArmActuator wrap) に置換 (Phase B-1)。"
            " 未指定 = MockWaistActuator (backward compat、dev / smoke)。"
            " 実 G1 で waist joint も physical に動かしたい時に指定。"
        ),
    )
    p.add_argument(
        "--use-real-hand",
        action="store_true",
        help=(
            "hand actuator を G1HandActuator (Dex1-1 SDK) に置換 (Phase B-3 skeleton)。"
            " Dex1-1 SDK が third_party/ に未 install だと start() で ImportError。"
            " body 到着後 (Issue #63 A-2) + SDK install 後に有効化する想定。"
            " 未指定 = MockHandActuator (default、SDK 依存無し)。"
        ),
    )
    # Issue #125: pick_table_leg VLA 選択 (未指定なら Issue #123 rule-based)
    p.add_argument(
        "--policy-variant-pick",
        type=str,
        default=None,
        help=(
            "pick_table_leg に使う learned policy variant 名 (Issue #125)。"
            " 現状 groot_pick_legs_v1 のみ実装 (Isaac-GR00T native raw、full-body"
            " embodiment)。未指定なら Issue #123 の CV + IK ルールベース実装。"
        ),
    )
    p.add_argument(
        "--pick-leg-hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Stage 1-4 の pick_table_leg を、VLM境界付きGR00T → G1 IK MP → "
            "rule-based持ち替え → insert初期姿勢へ置換する。SDK/boundaryのStage 1-4"
            "では既定ON。--no-pick-leg-hybridは比較診断専用"
        ),
    )
    p.add_argument(
        "--pick-leg-hybrid-config",
        type=Path,
        default=_DEFAULT_PICK_LEG_HYBRID_CONFIG,
    )
    p.add_argument(
        "--pick-leg-phase3-executor",
        choices=("vla", "rule_based"),
        default="rule_based",
        help=("本番Stageでは完了判定を持つrule_basedのみ許可する。vlaは単体評価用"),
    )
    p.add_argument(
        "--pick-leg-vlm-endpoint",
        default=None,
        help="OpenAI-compatible chat-completions URL (default=hybrid YAML)",
    )
    p.add_argument(
        "--pick-leg-vlm-model",
        default=None,
        help="served VLM model id (default=hybrid YAML)",
    )
    p.add_argument(
        "--spawn-vlm-server",
        action="store_true",
        help=(
            "会場 (提出 image / RunPod) 用: hybrid pick を使う run で、VLM server "
            "(pick_leg_hybrid/run_venue_vlm_server.sh) をこの run の子 process として起動し、"
            "読み込みを時間制限なしで待ってから慣らしを 1 回送る。run の終わりに止める。"
            "未指定なら外で立てた server を使う (開発機の run_local_vlm_server.sh)"
        ),
    )
    # Issue #125: insert_table_leg VLA 選択 (未指定なら MockSkill fallback)
    p.add_argument(
        "--policy-variant-insert",
        type=str,
        default=None,
        help=(
            "insert_table_leg に使う learned policy variant 名 (Issue #125)。"
            " 例: groot_insert_leg_200k (takada 200k、LeRobot GR00T)。未指定なら"
            " MockSkill fallback。"
        ),
    )
    # Issue #125: rotate_leg_to_tighten VLA 選択 (未指定なら MockSkill fallback)
    p.add_argument(
        "--policy-variant-rotate-leg",
        type=str,
        default=None,
        help=(
            "rotate_leg_to_tighten に使う learned policy variant 名 (Issue #125)。"
            " 例: groot_rotate_leg_200k (takada 200k、LeRobot GR00T)。未指定なら"
            " MockSkill fallback。"
        ),
    )
    # Issue #128 Phase 3: rotate_table_base VLA 選択 (未指定なら MockSkill fallback)。
    #   move_table_base とは別の VLA skill (task_index=5 分割、skill_config.yaml で
    #   initial_pose を保持)。RAMEN-Ori は task 5+7 combined で学習済、skill_id
    #   conditioning で区別。
    p.add_argument(
        "--policy-variant-rotate-table-base",
        type=str,
        default=None,
        help=(
            "rotate_table_base に使う learned policy variant 名 (Issue #128)。"
            " 例: ramen_ori_default (task 5+7 combined、skill_id=1 conditioning)。"
            " 未指定なら MockSkill fallback。"
        ),
    )
    # Issue #125: flip_table VLA 選択 (未指定なら MockSkill fallback)
    p.add_argument(
        "--policy-variant-flip",
        type=str,
        default=None,
        help=(
            "flip_table に使う learned policy variant 名 (Issue #125)。"
            " 例: groot_flip_table_n17_2_baseline (suzuki baseline 20k、"
            "Furniture-GR00T)。未指定なら"
            " MockSkill fallback。"
        ),
    )
    p.add_argument(
        "--policy-variant-set",
        type=str,
        default=None,
        help=(
            "policy_config.yaml の variant_sets にある組み合わせの名前 (Issue #159)。"
            " 書かれた skill だけ default_variant_by_skill を差し替える"
            " (例: ramen_ori = RAMEN-Ori が slot を持つ skill を全部 RAMEN-Ori に)。"
            " slot ごとの --policy-variant-* が優先する"
        ),
    )
    p.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=(
            "tick ごとの記録 (states/actions/events/metadata) の出力先 (Issue #141 D1)。"
            " 未指定なら outputs/production_runs/<時刻>/。本番の run も評価と同じ形で"
            " 残し、同じ集計 script で見られるようにする"
        ),
    )
    p.add_argument(
        "--no-record",
        action="store_true",
        help="tick ごとの記録を取らない (既定は取る。wandb への upload は本番では行わない)",
    )
    p.add_argument(
        "--setup-hold-sec",
        type=float,
        default=None,
        help=(
            "頭の手順の最後に開始姿勢で保持する秒数 (Issue #141 D2)。stage の境界の"
            " Enter を無くした代わりの準備時間。指定しなければ skill_config.yaml の"
            " hold_pose.hold_sec (大会経路と同じ値、Issue #159 T9)。評価側の既定は 0"
        ),
    )
    p.add_argument(
        "--joint-source",
        choices=("lowstate", "joint_states", "boundary"),
        default="lowstate",
        help=(
            "関節角の取り方 (Issue #141 INF-18)。lowstate (既定) = Desktop が"
            " rt/lowstate を直接受ける (新しい値が 500 Hz)、joint_states = Orin の"
            " bridge 経由 (同じ値の送り直しで新しい値は 11〜13 Hz)"
        ),
    )
    p.add_argument(
        "--head-source",
        choices=("ros2", "zmq"),
        default="ros2",
        help=(
            "head camera の取り方。ros2 (既定) = lab 構成で --topic の"
            " CompressedImage を受ける、zmq = 大会会場構成で運営 bridge"
            " (real_orin_cameras.py) の ZeroMQ を直接受ける。ROS2 へ bridge すると"
            " JPEG を二度通って学習時と画が変わるため直接受ける"
        ),
    )
    p.add_argument(
        "--zmq-endpoint",
        type=str,
        default=None,
        help=(
            "--head-source zmq のときの運営 camera bridge endpoint。省略時は"
            " IROS_ORIN_HOST または G1_IMAGE_SERVER_IP から tcp://HOST:5555 を生成"
        ),
    )
    p.add_argument(
        "--action-sink",
        choices=("sdk", "boundary"),
        default="sdk",
        help=(
            "action の出口。sdk (既定) = rt/arm_sdk へ直接 publish (lab の既存経路)。"
            " boundary = (T,25) を :5556 に publish して運営 wbc_adapter に流す"
            " (会場でグリッパを動かせる唯一の経路)。boundary のとき rt/arm_sdk の"
            " publisher、Unitree SDK、DDS actuator は立てず、歩行も action の"
            " navigation 3D として同じ公式境界へ送る"
        ),
    )
    p.add_argument(
        "--boundary-lane",
        choices=("pose", "joint"),
        default="pose",
        help=(
            "--action-sink boundary の送り方。pose (既定) = (T,25) の手先の姿勢 (運営 IK が"
            " 関節角に戻す)。joint = (T,22) の腕の関節角をそのまま (運営が 2026-09-25 に追加した"
            " joint lane、運営 IK を通らない。PC2 の adapter が joint lane 入りの版であること)。"
            " joint のとき「腕が着いたか」は関節角で比べる"
        ),
    )
    p.add_argument(
        "--walk-lowering-check",
        choices=("joint", "converged"),
        default=None,
        help=(
            "Stage 0 で下ろした腕を歩行中に保持する前の確かめ方 (既定は skill_config.yaml の"
            " skills.walk_lowered_pose.latch_check = joint)。joint = 関節ごとの歩行の範囲を外れて"
            " いれば歩かずに止める。converged = 下ろす動きが収束していれば記録して保持する。"
            " 接続テストで決めるための上書き"
        ),
    )
    p.add_argument(
        "--wrist-roll-clamp",
        choices=("on", "off"),
        default="on",
        help=(
            "pose lane で手首 roll を運営 IK の古い上限 (±0.9) に寄せるか。on (既定) はどの版の"
            " 運営 IK にも安全。off は URDF の範囲だけ (運営 IK の a1af470 = interface package"
            " 609f61d 以降で上限が無くなった。学習データは 0.9 を超える手首 roll を使う)。"
            " 接続テストで PC2 の版を見て決める"
        ),
    )
    p.add_argument(
        "--boundary-port",
        type=int,
        default=5556,
        help="--action-sink boundary のとき bind する port (運営 adapter が dial-in する)",
    )
    p.add_argument(
        "--boundary-host",
        type=str,
        default="*",
        help=(
            "--action-sink boundary のとき bind する interface (既定 * = 全て)。"
            " WBC_RUNBOOK の想定は「PC2 側の client が :5556 を bind、adapter が"
            " localhost に dial-in」。本 entrypoint を Thor で動かす場合は bind 先が"
            " Thor になるので、adapter (wbc_driver.py) を --actions-host <Thor> で"
            " 起動する (adapter は参加者が起動する、RUNBOOK §4 / Issue #161)"
        ),
    )
    p.add_argument(
        "--boundary-state-host",
        type=str,
        default=None,
        help=(
            "運営公式 robot-state SUB endpoint (:5557) の host。省略時は"
            " --zmq-endpoint、IROS_ORIN_HOST、G1_IMAGE_SERVER_IP の順に解決"
        ),
    )
    p.add_argument(
        "--boundary-state-port",
        type=int,
        default=5557,
        help="運営公式 robot-state SUB endpoint の port",
    )
    p.add_argument(
        "--synthetic-hand-state",
        action="store_true",
        help=(
            "Dex1 の hand state を合成する (会場のリグは配信しないため)。"
            " 開始 skill の frame-0 値で seed し、以降は hand 指令をエコーする。"
            " --stage の --use-real-hand 必須を免除する代わりに、hand actuator は"
            " SDK laneではmock固定。boundary laneでは実測の代わりに指令echoを"
            "観測へ戻しつつ、同じ指令を公式(T,25) hand列へ送る"
        ),
    )
    p.add_argument(
        "--joint-state-topic",
        type=str,
        default="/joint_states",
        help=(
            "JointState topic 名 (Issue #65 real_hw_bridge_node publish)。"
            " --joint-source=joint_states のときだけ使う"
        ),
    )
    p.add_argument(
        "--wrist-left-topic",
        type=str,
        default="/wrist_left/camera/color/image_raw/compressed",
        help=(
            "Wrist left camera topic 名 (Issue #75 Orin bringup enable_wrist_cameras=true 時)。"
            " obs['wrist_left_rgb'] に latest snapshot を乗せる (real VLA drop-in で使う想定)"
        ),
    )
    p.add_argument(
        "--wrist-right-topic",
        type=str,
        default="/wrist_right/camera/color/image_raw/compressed",
        help="Wrist right camera topic 名 (同上、obs['wrist_right_rgb'])",
    )
    p.add_argument(
        "--no-wrist-cameras",
        action="store_true",
        help=(
            "Wrist camera source を無効化 (Orin で enable_wrist_cameras=false or"
            " camera-only smoke で使う)。obs に wrist_{left,right}_rgb field 追加されない"
        ),
    )
    p.add_argument(
        "--phase1-11-arm-only",
        action="store_true",
        help=(
            "Issue #128 Phase 1.11 safety profile: exact 4s setup + 1s/0.185m/s "
            "walk, real arms only, Mock waist/hand, stop after 30s move_table_base "
            "or immediately on pick transition. Requires all observations and Enter."
        ),
    )
    p.add_argument(
        "--phase1-12-waist-real",
        action="store_true",
        help=(
            "Issue #128 Phase 1.12 safety profile: Phase 1.11と同じ固定flow/停止境界で、"
            "real arms + real waist、Mock Dex1を検証する。--use-real-waistは暗黙に有効。"
        ),
    )
    p.add_argument(
        "--phase1-13-hand-real",
        action="store_true",
        help=(
            "Issue #128 Phase 1.13 safety profile: Phase 1.12と同じ固定flow/停止境界で、"
            "real arms + real waist + slew-limited real Dex1を検証する。"
        ),
    )
    # Issue #128 Phase 3: stage-based orchestration。--stage=0..5 で
    # 「その stage 内で dispatch する skill 列」を選択する。stage 別の
    # 実機評価手順は skill 別 handoff MD (docs/handoff/*.md) を参照。
    p.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4, 5],
        help=(
            "Phase 3 stage 指定。0=準備 (歩く+腕+初期姿勢移動)、1-4=各 leg round"
            " (rotate_table_base → pick → insert → rotate_leg_to_tighten)、5=flip。"
            " 指定時は Phase 3 transition graph に切替、initial_skill は stage の"
            " 先頭 skill に設定される。未指定なら legacy 経路 (DEFAULT_TRANSITIONS)。"
        ),
    )
    p.add_argument(
        "--phase3-full",
        action="store_true",
        help=(
            "Run Phase 3 stages in one physical-control process. arm_sdk/Dex1 "
            "ownership is retained across stage boundaries; only policy experts "
            "are swapped."
        ),
    )
    # Keep these defaults as None until after parsing. Otherwise there is no way
    # to distinguish an omitted range from a user who accidentally supplied a
    # range without --phase3-full (which used to fall through to the legacy,
    # potentially actuating path).
    p.add_argument("--phase3-start-stage", type=int, choices=range(6), default=None)
    p.add_argument("--phase3-end-stage", type=int, choices=range(6), default=None)
    p.add_argument(
        "--actuate",
        action="store_true",
        help=(
            "Enable physical commands for --stage Phase 3 runs. Without this flag, "
            "models and all live observations are validated read-only and the "
            "process exits before any walk/arm/waist/Dex1 command."
        ),
    )
    p.add_argument(
        "--rule-based-pick-table-leg",
        action="store_true",
        help=(
            "Run only the Issue #123 CV + IK rule-based pick_table_leg skill. "
            "No walking, learned policy, waist command, or lower-body command is used."
        ),
    )
    p.add_argument(
        "--rule-based-pick-calibration",
        type=Path,
        default=None,
        help=(
            "Verified physical head-camera calibration YAML for rule-based pick. "
            "Required with --rule-based-pick-table-leg --actuate; the provisional "
            "URDF values in skill_config.yaml are read-only diagnostics only."
        ),
    )
    return p.parse_args()


def normalize_phase3_range_args(args: argparse.Namespace) -> None:
    """Validate explicit Phase 3 ranges and install the documented defaults.

    A range is meaningful only for ``--phase3-full``. Fail closed instead of
    silently entering the legacy execution path when an operator forgets that
    flag. Downstream code continues to receive integer 0/5 defaults.
    """

    start = getattr(args, "phase3_start_stage", None)
    end = getattr(args, "phase3_end_stage", None)
    full = bool(getattr(args, "phase3_full", False))
    if not full and (start is not None or end is not None):
        raise ValueError(
            "--phase3-start-stage/--phase3-end-stage require --phase3-full"
        )
    args.phase3_start_stage = 0 if start is None else start
    args.phase3_end_stage = 5 if end is None else end


def _endpoint_host(endpoint: str) -> str:
    """``tcp://HOST:PORT`` の HOST。"""
    if not endpoint.startswith("tcp://"):
        raise ValueError(f"ZMQ endpoint must be tcp://HOST:PORT, got {endpoint!r}")
    host = endpoint.removeprefix("tcp://").rsplit(":", 1)[0]
    if not host:
        raise ValueError(f"cannot read the host of {endpoint!r}")
    return host


def configure_official_endpoints(args: argparse.Namespace) -> None:
    """Resolve venue endpoints without baking robot-specific addresses in code.

    優先順位 (Codex #10): 明示した CLI (``--boundary-state-host`` /
    ``--zmq-endpoint``) > 環境変数 (``IROS_ORIN_HOST`` > ``G1_IMAGE_SERVER_IP``)。
    以前は ``--zmq-endpoint`` を明示しても state 側だけ環境変数の host を使い、
    camera と state が別のロボットに繋がり得た。camera (`:5555`) と state
    (`:5557`) は同じ Orin が配るので、最後に同じ host であることを確かめる。
    """

    if getattr(args, "action_sink", "sdk") == "boundary":
        args.head_source = "zmq"
        args.joint_source = "boundary"
    if getattr(args, "head_source", "ros2") != "zmq":
        return

    orin_env = os.environ.get("IROS_ORIN_HOST") or None
    image_env = os.environ.get("G1_IMAGE_SERVER_IP") or None
    explicit_state_host = getattr(args, "boundary_state_host", None)
    endpoint = getattr(args, "zmq_endpoint", None)
    # An explicit endpoint (or state host, from which the camera endpoint is
    # derived) selects the Orin.  Conflicting leftover environment variables
    # matter only when they are the source of that selection.
    if (
        endpoint is None
        and not explicit_state_host
        and orin_env
        and image_env
        and orin_env != image_env
    ):
        raise ValueError(
            f"IROS_ORIN_HOST={orin_env!r} and G1_IMAGE_SERVER_IP={image_env!r} "
            "disagree; unset one or pass --zmq-endpoint/--boundary-state-host"
        )
    env_host = orin_env or image_env

    if endpoint is None:
        host = explicit_state_host or env_host
        if not host:
            raise ValueError(
                "ZMQ/boundary mode requires --zmq-endpoint plus "
                "--boundary-state-host, or IROS_ORIN_HOST/G1_IMAGE_SERVER_IP"
            )
        endpoint = f"tcp://{host}:5555"
        args.zmq_endpoint = endpoint
    camera_host = _endpoint_host(endpoint)

    if getattr(args, "action_sink", "sdk") == "boundary":
        state_host = explicit_state_host or camera_host
        if state_host != camera_host:
            raise ValueError(
                f"camera endpoint host {camera_host!r} and state host "
                f"{state_host!r} differ; both are served by the same Orin"
            )
        args.boundary_state_host = state_host
    print(
        f"[init] organizer endpoints: cameras={endpoint}"
        + (
            f" state=tcp://{args.boundary_state_host}:"
            f"{getattr(args, 'boundary_state_port', 5557)}"
            if getattr(args, "action_sink", "sdk") == "boundary"
            else ""
        ),
        file=sys.stderr,
    )


def _selected_policy_start_gate_stage(args: argparse.Namespace) -> Optional[int]:
    """Return the independently selected Stage that needs an operator gate.

    A multi-Stage continuous run intentionally has no gates between Stages.  A
    one-Stage ``--phase3-full --phase3-start-stage N --phase3-end-stage N`` run is
    nevertheless an independently selected Stage and receives the same gate as
    ``--stage N``.  Stage 0 never starts a policy; it only walks and reaches the
    first pick pose.
    """

    stage = getattr(args, "stage", None)
    if stage in {1, 2, 3, 4, 5}:
        return int(stage)
    if bool(getattr(args, "phase3_full", False)):
        start = getattr(args, "phase3_start_stage", None)
        end = getattr(args, "phase3_end_stage", None)
        if start == end and start in {1, 2, 3, 4, 5}:
            return int(start)
    return None


def _insert_policy_start_gate_transition(
    transitions: dict[str, list[str]], *, first_policy: str
) -> str:
    """Insert a finite operator gate between head hold and the first policy."""

    hold_name = f"hold_pose_for_{first_policy}"
    gate_name = f"operator_gate_for_{first_policy}"
    expected = [first_policy]
    if transitions.get(hold_name) != expected:
        raise RuntimeError(
            f"cannot install policy-start gate: expected {hold_name} -> "
            f"{expected}, got {transitions.get(hold_name)!r}"
        )
    transitions[hold_name] = [gate_name]
    transitions[gate_name] = [first_policy]
    return gate_name


def load_verified_rule_pick_calibration(path: Path) -> dict:
    """Load a fail-closed physical camera/plane override."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("rule-based pick calibration must be a YAML mapping")
    allowed = {
        "calibration_verified",
        "calibration_id",
        "camera",
        "extrinsic",
        "support_plane",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"rule-based pick calibration has unknown keys: {sorted(unknown)}"
        )
    if payload.get("calibration_verified") is not True:
        raise ValueError("rule-based pick calibration_verified must be exactly true")
    calibration_id = payload.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError("rule-based pick calibration_id must be a non-empty string")
    for field in ("camera", "extrinsic", "support_plane"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"rule-based pick calibration requires mapping {field!r}")
    return {field: payload[field] for field in ("camera", "extrinsic", "support_plane")}


def build_pick_table_leg(
    skills_section: dict,
    gripper: object,
    *,
    time_fn: object = None,
    control_hz: float = 30.0,
    motion_limits: object = None,
    obb_grasp_override: Optional[dict] = None,
) -> object:
    """Build the Issue #123 rule-based pick skill without importing DDS.

    Keeping the DDS gripper injectable makes the complete shipped wiring
    testable without a physical robot and guarantees that the grasp provider
    and the skill share one kinematics instance.
    """
    import time as _time
    import numpy as np

    from inference.desktop.lower_policy.kinematics.g1_arm import G1ArmKinematics
    from inference.desktop.lower_policy.kinematics.types import Side
    from inference.desktop.lower_policy.motion_limits import DEFAULT_MOTION_LIMITS
    from inference.desktop.lower_policy.obb_grasp import ObbGraspPoseProvider
    from inference.desktop.lower_policy.skills.motion_limiter import MotionLimiter
    from inference.desktop.lower_policy.skills.pick_table_leg import PickTableLegSkill

    if "pick_table_leg" not in skills_section:
        raise ValueError("skill config invalid: 'skills.pick_table_leg' missing")
    cfg = skills_section["pick_table_leg"]
    if not isinstance(cfg, dict):
        raise ValueError(
            "skill config invalid: 'skills.pick_table_leg' must be a mapping"
        )
    kinematics = G1ArmKinematics()
    limits = DEFAULT_MOTION_LIMITS if motion_limits is None else motion_limits
    joint_limits = np.concatenate(
        (kinematics.joint_limits(Side.LEFT), kinematics.joint_limits(Side.RIGHT)),
        axis=0,
    )
    limiter = MotionLimiter(
        dim=14,
        arm_slice=slice(0, 14),
        hand_slice=slice(14, 14),
        arm_velocity=limits.arm_velocity_rad_s,
        arm_acceleration=limits.arm_acceleration_rad_s2,
        hand_velocity=limits.hand_velocity_rad_s,
        hand_acceleration=limits.hand_acceleration_rad_s2,
        control_hz=control_hz,
        lower=joint_limits[:, 0],
        upper=joint_limits[:, 1],
        reject_position_slice=slice(0, 14),
    )
    obb_grasp_cfg = dict(cfg.get("obb_grasp") or {})
    if obb_grasp_override is not None:
        obb_grasp_cfg.update(obb_grasp_override)
    return PickTableLegSkill.from_config(
        cfg,
        kinematics=kinematics,
        gripper=gripper,
        grasp_provider=ObbGraspPoseProvider.from_config(
            obb_grasp_cfg, kinematics=kinematics
        ),
        time_fn=_time.monotonic if time_fn is None else time_fn,
        motion_limiter=limiter,
    )


def resolve_stage_selection(
    args: argparse.Namespace,
) -> tuple[Optional[int], tuple[int, ...]]:
    """`--stage` / `--phase3-full` から「最初の stage」と「回る stage の列」を出す。

    main() の複数の場所 (記録の metadata、頭の手順、先読みの列、遷移表、学習した skill の
    集合) が同じ値を使うので、1 か所で決める。stage を指定しない run (rule-based pick 等)
    では `(None, ())` を返す。
    """
    if args.phase3_full:
        return args.phase3_start_stage, tuple(
            range(args.phase3_start_stage, args.phase3_end_stage + 1)
        )
    if args.stage is None:
        return None, ()
    return args.stage, (args.stage,)


#: boundary 経路の腰の扱い (Codex #4、2026-09-23 決定)。
BOUNDARY_WAIST_NOTE = (
    "waist = measured: the organizer's adapter ignores torso RPY [22:25] and its "
    "IK holds the waist at the measured angle, so policy waist commands are not "
    "executed on this lane; EE targets are FK(measured waist, commanded arms)"
)
#: joint lane の腰の扱い (運営 CONTRACT "The joint lane": 胴の列は無い)。
BOUNDARY_JOINT_WAIST_NOTE = (
    "waist = measured: the joint lane has no torso column and the organizer's "
    "adapter holds the waist at its measured angle; arm joint angles are sent "
    "as commanded (no FK, no organizer IK)"
)


def _gate_arrival_check(target_arm, task_space_checker, tolerance_rad: float):
    """policy 開始の gate が「開始姿勢に届いたか」を見る関数を作る。

    ``task_space_checker`` は `_boundary_convergence_checker` の返り値 (pose lane は手先で比べる
    関数、sdk・joint lane は None)。None なら関節角で比べる (準備動作の許容と同じ)。
    """
    import numpy as np

    from inference.desktop.lower_policy.initial_pose import ARM_JOINT_ORDER
    from inference.desktop.lower_policy.skills.operator_gate import joint_space_arrival

    target = np.asarray(target_arm, dtype=np.float64).reshape(-1).copy()

    def _check(measured_arm):
        if task_space_checker is not None:
            # 速さは gate では見ない (腕は保持中)。位置と向きだけ
            ok, detail = task_space_checker(target, measured_arm, np.zeros(14))
            return bool(ok), detail
        return joint_space_arrival(measured_arm, target, tolerance_rad, ARM_JOINT_ORDER)

    return _check


def _boundary_taskspace_arrival(args: argparse.Namespace) -> bool:
    """会場で「腕が着いたか」を手先の姿勢で比べるか。

    pose lane では運営 IK が 7 自由度の余りを選ぶので関節角は一致しない → 手先で比べる。
    joint lane は送った関節角がそのまま目標なので、関節角で比べる (運営の到達の目安も関節角)。
    """
    return (
        getattr(args, "action_sink", "sdk") == "boundary"
        and getattr(args, "boundary_lane", "pose") == "pose"
    )

#: Dex1 を持つ Stage (Stage 0 は歩行と最初の pick 姿勢だけで手を持たない)。
HAND_OWNING_STAGES = frozenset({1, 2, 3, 4, 5})


def resolve_include_hand_in_head(
    args: argparse.Namespace, selected_stages: Sequence[int]
) -> bool:
    """頭の手順と model 境界に Dex1 の手順を入れるか。

    **実際に hand 指令の経路があるか**で決める。SDK 経路は `--use-real-hand`、
    boundary 経路は `(T,25)` の hand 列 (合成 state は必須) で、どちらも run の
    どこかに手を持つ Stage があれば全 Stage の手順に手が入る。以前は boundary だけ
    「開始 Stage が 1..5」で判定しており、`--phase3-start-stage 0` の通し実行で
    後続 Stage の手の全開・初期幅・境界の手順が黙って省かれていた (Codex #3)。
    """
    if args.use_real_hand:
        return True
    return bool(
        getattr(args, "action_sink", "sdk") == "boundary"
        and getattr(args, "synthetic_hand_state", False)
        and HAND_OWNING_STAGES.intersection(selected_stages)
    )


def main() -> None:
    args = parse_args()
    try:
        normalize_phase3_range_args(args)
    except ValueError as exc:
        sys.exit(f"Phase 3 configuration rejected: {exc}")
    try:
        configure_official_endpoints(args)
    except ValueError as exc:
        sys.exit(f"Official boundary configuration rejected: {exc}")
    if args.action_sink == "boundary" and not args.synthetic_hand_state:
        # 無いと Dex1StateSource (rt/dex1 の DDS 購読) が立ち、hand 列は
        # 閉 (+1) 固定の fallback になる。会場には Dex1 state が無い。
        sys.exit(
            "Official boundary configuration rejected: --action-sink boundary "
            "requires --synthetic-hand-state (the contract has no Dex1 state and "
            "the boundary lane must not subscribe to DDS)"
        )
    pick_mode_defaulted = resolve_production_pick_mode(args)
    rule_based_pick_active = args.rule_based_pick_table_leg
    phase3_active = args.stage is not None or args.phase3_full or rule_based_pick_active
    # stage の選び方はここで 1 回だけ決める。記録の metadata (7 節) や先読みの列 (8 節)
    # など、後の複数の場所から読むため、分岐の中で作らない。
    first_phase3_stage, selected_stages = resolve_stage_selection(args)
    include_hand_in_head = resolve_include_hand_in_head(args, selected_stages)
    # --gpu-models: "all" は None (= stage の全部を GPU に置く)、それ以外は 1 以上の int。
    if str(args.gpu_models).lower() == "all":
        gpu_models: Optional[int] = None
    else:
        try:
            gpu_models = int(args.gpu_models)
        except ValueError:
            sys.exit(
                f"--gpu-models must be a positive int or 'all', got {args.gpu_models!r}"
            )
        if gpu_models < 1:
            sys.exit(f"--gpu-models must be >= 1, got {gpu_models}")

    if (
        sum(
            bool(value)
            for value in (
                args.phase1_11_arm_only,
                args.phase1_12_waist_real,
                args.phase1_13_hand_real,
            )
        )
        > 1
    ):
        sys.exit("Phase 1.11/1.12/1.13 safety profiles are mutually exclusive")
    # Issue #128 Phase 3: --stage は Phase 1 profile と両立しない
    if phase3_active and any(
        (args.phase1_11_arm_only, args.phase1_12_waist_real, args.phase1_13_hand_real)
    ):
        sys.exit("Phase 3 is mutually exclusive with Phase 1 profiles")
    try:
        # 先に既定で埋めてから検査する。検査は「その stage が要る slot が揃って
        # いるか」を見るので、埋める側も同じ `required_policy_slots` を使う。
        variant_source = apply_default_policy_variants(
            args, config_path=args.policy_config
        )
        _validate_phase3_config(args)
    except (ValueError, OSError) as exc:
        sys.exit(f"Phase 3 configuration rejected: {exc}")
    if pick_mode_defaulted:
        print(
            "[init] production pick mode defaulted to hybrid: "
            "VLM -> groot_pick_legs_v1 -> IK/MP -> rule-based handover -> "
            "insert frame-zero",
            file=sys.stderr,
        )
    if variant_source:
        print(
            "[init] policy variants: "
            + " ".join(
                f"{slot}={getattr(args, _POLICY_SLOT_ATTRS[slot])} ({origin})"
                for slot, origin in variant_source.items()
            ),
            file=sys.stderr,
        )
    phase1_profile: str | None = None
    if args.phase1_11_arm_only:
        phase1_profile = "1.11"
    elif args.phase1_12_waist_real:
        phase1_profile = "1.12"
        # The dedicated profile is the authority for enabling waist dispatch;
        # keeping this implicit prevents an operator from forgetting the flag.
        args.use_real_waist = True
    elif args.phase1_13_hand_real:
        phase1_profile = "1.13"
        args.use_real_waist = True
        args.use_real_hand = True

    # ---- lazy import (runtime env 以外では ImportError にせず --help を通す) ----
    # main() の中の会場用 closure (_send_to_boundary / _measured_boundary_arms /
    # 到達判定) が np を使う。module の先頭では import していないので、無いと
    # `--action-sink boundary --actuate` が最初の関節読み取りで NameError になる (Issue #164)。
    import numpy as np
    from inference.desktop.lower_policy.actuators.dex1_dds import Dex1DdsGripper
    from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1ArmActuator
    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator
    from inference.desktop.lower_policy.actuators.boundary_sink import (
        BoundaryArmActuator,
        BoundaryWalkActuator,
    )
    from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
    from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
        CollisionAwareArmPreMotionSkill,
        MeasuredArmWalkHoldSkill,
        PostWalkArmSettleSkill,
        validate_lowered_walk_pose,
    )
    from inference.desktop.lower_policy.skills.mock import MockSkill
    from inference.desktop.lower_policy.skills.move_to_table import MoveToTable
    from inference.desktop.lower_policy.skills.sample_vla_skill import SampleVLASkill
    from inference.desktop import assembly
    from inference.desktop.lower_policy.skills.setup_skill import SetupSkill
    from inference.desktop.orchestrator import (
        DEFAULT_ENTER_CHECK,
        DEFAULT_TRANSITIONS,
        STAGE_HEAD_SKILL,
        STAGE_SKILL_SEQUENCES,
        TIMEOUT_ACTIONS,
        LiveSourceSafetyError,
        Orchestrator,
        build_stage_skill_sequence,
        build_stage_transitions,
        model_transition_pairs_for_stage,
        stage_enter_check,
        validate_stage_model_transition_chain,
    )
    from inference.desktop.perception.cleaner import (
        load_cleanup_config,
        resolve_median_match_params,
    )
    from inference.desktop.perception.policy_filter import (
        PolicyDetectionFilter,
        load_policy_filter_config,
    )
    from inference.desktop.perception.stream import DetectionStream

    if not args.skill_config.exists():
        sys.exit(f"skill config not found: {args.skill_config}")

    print(f"[init] interface={args.interface}", file=sys.stderr)
    print(
        f"[init] topic={args.topic} head_view={HEAD_CAMERA_STEREO_VIEW} "
        f"perception_view={HEAD_PERCEPTION_VIEW} "
        f"hz={args.hz} skill_config={args.skill_config}",
        file=sys.stderr,
    )

    # 0) Skill config YAML 読み込み (Issue #81 Phase 2b)。各 skill の from_config
    #    にセクション dict を渡すため entrypoint で 1 回だけ safe_load する。
    skill_cfg_raw = yaml.safe_load(args.skill_config.read_text(encoding="utf-8"))
    if not isinstance(skill_cfg_raw, dict) or "skills" not in skill_cfg_raw:
        sys.exit(
            f"skill config invalid: 'skills' section missing in {args.skill_config}"
        )
    from inference.desktop.lower_policy.initial_pose import (
        apply_policy_variant_profile,
    )

    skill_cfg_raw = apply_policy_variant_profile(
        skill_cfg_raw, "flip_table", args.policy_variant_flip
    )
    from inference.desktop.lower_policy.skills.hand_ramp import resolve_hand_opening_rad

    opening = resolve_hand_opening_rad(skill_cfg_raw, action_sink=args.action_sink)
    skill_cfg_raw["hand_pre_motion"] = {
        **(skill_cfg_raw.get("hand_pre_motion") or {}), "open_rad": opening
    }
    print(f"[init] Dex1 preparation/release opening={opening:g}rad ({args.action_sink})", file=sys.stderr)
    skills_section: dict = skill_cfg_raw["skills"]

    from inference.desktop.lower_policy.initial_pose import initial_pose_from_config

    def _effective_initial_pose(skill_name: str):
        """Read frame-zero from the already variant-profiled configuration."""

        return initial_pose_from_config(
            skill_cfg_raw, skill_name, source=str(args.skill_config)
        )
    if args.setup_hold_sec is None:
        # 正本は YAML。大会経路の driver も同じ値を読む (Issue #159 T9)
        from inference.desktop import assembly as _hold_assembly

        args.setup_hold_sec = _hold_assembly.hold_sec_from_config(skill_cfg_raw)
    hybrid_pick_cfg = None
    hybrid_pick_runtime: Optional[dict[str, object]] = None
    if args.pick_leg_hybrid:
        # Do the network/model/reference-image check before constructing any
        # actuator.  A missing VLM can therefore never acquire arm_sdk/Dex1.
        from inference.desktop.pick_leg_hybrid.real_skill import (
            load_reference_images,
            probe_vlm_endpoint,
        )

        hybrid_pick_cfg, references = load_reference_images(
            args.pick_leg_hybrid_config,
            endpoint_override=args.pick_leg_vlm_endpoint,
            model_override=args.pick_leg_vlm_model,
        )
        if args.spawn_vlm_server:
            # 会場: VLM をこの run の子 process として立てる。止めるのは終了時の
            # _close_policy_resources (例外・Ctrl-C でも通る)。
            from inference.desktop.pick_leg_hybrid.real_skill import (
                build_vlm_self_check_images,
            )
            from inference.desktop.pick_leg_hybrid.vlm_server import (
                VenueVlmServer,
                check_venue_endpoint,
                warm_up_vlm,
            )

            try:
                check_venue_endpoint(hybrid_pick_cfg.vlm.endpoint)
            except ValueError as exc:
                sys.exit(str(exc))
            vlm_log_dir = args.log.parent if args.log is not None else _DEFAULT_LOG_DIR
            vlm_server = VenueVlmServer(
                vlm_log_dir / f"vlm_{time.strftime('%Y-%m-%dT%H-%M-%S')}.log"
            )
            _register_policy_resource(vlm_server)
            vlm_server.start()
            vlm_server.wait_until_ready()
            warm_up_vlm(
                hybrid_pick_cfg,
                build_vlm_self_check_images(hybrid_pick_cfg, references),
            )
        endpoint = probe_vlm_endpoint(hybrid_pick_cfg, references)
        hybrid_pick_runtime = {
            "config": str(args.pick_leg_hybrid_config.resolve()),
            "phase3_executor": args.pick_leg_phase3_executor,
            "reference_count": len(references),
            **endpoint,
        }
        print(
            "[hybrid] production VLM/GR00T/MP preflight passed: "
            f"model={endpoint['model']} references={len(references)} "
            f"vlm_latency={endpoint.get('multimodal_latency_sec', float('nan')):.2f}s "
            "phase3=rule_based; pick waist/legs=Regular",
            file=sys.stderr,
        )
    if args.phase1_11_arm_only:
        try:
            _validate_phase1_11_config(args, skills_section)
        except ValueError as exc:
            sys.exit(f"Phase 1.11 configuration rejected: {exc}")
        print(
            "[phase1.11] flow=lowered measured-arm hold -> 1s walk (0.185m) -> "
            "1s zero-velocity settle -> collision-aware arm loop -> "
            "30s move_table_base VLA -> controlled stop; "
            "waist/hand=Mock",
            file=sys.stderr,
        )
    elif args.phase1_12_waist_real:
        try:
            _validate_phase1_12_config(args, skills_section)
        except ValueError as exc:
            sys.exit(f"Phase 1.12 configuration rejected: {exc}")
        print(
            "[phase1.12] flow=lowered measured-arm hold -> 1s walk (0.185m) -> "
            "1s zero-velocity settle -> collision-aware arm loop -> "
            "30s move_table_base VLA -> controlled stop; "
            "arms/waist=Real, hand=Mock",
            file=sys.stderr,
        )
    elif args.phase1_13_hand_real:
        try:
            _validate_phase1_13_config(args, skills_section)
        except ValueError as exc:
            sys.exit(f"Phase 1.13 configuration rejected: {exc}")
        print(
            "[phase1.13] flow=lowered measured-arm hold -> 1s walk (0.185m) -> "
            "1s zero-velocity settle -> collision-aware arm loop -> "
            "30s move_table_base VLA -> controlled stop; "
            "arms/waist/Dex1=Real",
            file=sys.stderr,
        )

    # 1) 実 G1 Actuator (walking FSM に既に入ってる前提)。
    #    G1SDKWalkActuator.__init__ 内で ChannelFactoryInitialize(0, interface) が
    #    走り、SDK singleton の DomainParticipant が確保される。以降 Ros2FrameSource
    #    もこの singleton を流用して subscribe する。
    if args.action_sink == "boundary":
        # The organizer contract is the *only* robot interface at the venue.
        # Do not initialize ChannelFactory/LocoClient/rt/arm_sdk in this lane.
        args.head_source = "zmq"
        args.joint_source = "boundary"
        actuator = BoundaryWalkActuator()
        arm_actuator = BoundaryArmActuator()
        print(
            "[init] official boundary-only I/O: cameras=:5555 state=:5557 "
            "actions=:5556; Unitree DDS/SDK is not initialized",
            file=sys.stderr,
        )
    else:
        actuator = G1SDKWalkActuator(interface=args.interface)

    # 1b) G1ArmActuator (Issue #75) を早期 init。VlaSkill (下記 Phase 9b) が
    #     G1WaistActuator (Phase B-1、shared arm_actuator wrap) を要求するため。
    #     __init__ で rt/lowstate を wait (5s timeout)、start() は Orchestrator
    #     配置直前 (下記 step 5b) に呼ぶ。
        arm_actuator = G1ArmActuator()
    if args.action_sink != "boundary":
        print(
            "[init] arm_actuator initialized (rt/lowstate synced、start() deferred)",
            file=sys.stderr,
        )

    # 2) Perception layer。重み・conf・imgsz は policy_config.yaml の `yolo` から
    #    (評価経路と同じ設定・学習の焼き込みと同じ重み、Issue #141 INF-10)。
    perception = assembly.build_yolo_perception(args.policy_config, device=args.device)
    # 検出は 2 つに分ける: 上位 Planner は cleaner (誤検知を消す)、policy は遅れの無い
    # filter (学習の画像は「同じ frame の生の検出」で描かれている)。Issue #141 D3。
    cleanup_config = load_cleanup_config()
    cleaner = DetectionStream(cleanup_config)
    policy_filter = PolicyDetectionFilter(load_policy_filter_config())

    # 3) Skill registry。全 skill の config は skill_config.yaml から読む (Phase 2b)。
    #    - setup:            Type B、SetupSkill (Issue #81、initial skill、腕 pre-motion)
    #    - move_to_table:    Type A、SDK walk (vx / max_dwell_sec は YAML)
    #    - move_table_base:  Type B、SampleVLASkill (default_pose_rad は YAML)。real VLA
    #      drop-in で SampleVLASkill を Gr00tSkill(model) 等に差し替える想定。
    #    - 他 4 skill:       MockSkill (VLA 未実装、log のみ)
    for required in ("setup", "move_to_table", "move_table_base"):
        if required not in skills_section:
            sys.exit(f"skill config invalid: 'skills.{required}' missing")

    # Issue #141 (1-8): 部品の組み立ては `inference/desktop/assembly.py` に集約し、
    # 評価経路 (evaluate/model_evaluation/runners/run_skill.py) と同じ関数を通す。
    # 手首 FK は offset ごとに 1 個だけ作って使い回す (URDF parse ~100 ms/offset、INF-9)。
    fk_factory = assembly.FkFactory()
    # 会場の到達判定は、運営 IK が作った姿勢を運営 IK と同じ運動学で見る (Issue #164)。
    # publish (BoundaryActionSink の既定) と同じ URDF。関節角は運営 IK が選ぶので、
    # 学習用の mode_15 で比べると同じ手首位置でも約 5 mm の差が出る。
    _organizer_fk: Optional[Any] = None

    def _boundary_convergence_checker(skill_name: str):
        """Verify organizer IK execution in task space, not one arbitrary IK q.

        wrist_yaw_link 原点 (tool offset なし) で比べるので ``skill_name`` に依らない。
        """

        nonlocal _organizer_fk
        if not _boundary_taskspace_arrival(args):
            # sdk 経路と joint lane は関節角で比べる (既定の判定、許容は skill_config)
            return None
        if _organizer_fk is None:
            from inference.desktop.perception.g1_urdf_fk import (
                ORGANIZER_IK_URDF_PATH,
                G1WristFK,
            )

            _organizer_fk = G1WristFK.from_urdf(ORGANIZER_IK_URDF_PATH)
        fk = _organizer_fk

        def _check(goal, _measured, measured_velocity):
            if joint_state_source is None:
                return False, "boundary_state=missing"
            snapshot = joint_state_source.get()
            if snapshot is None:
                return False, "boundary_state=missing"
            actual = np.asarray(snapshot.position, dtype=np.float64)
            if actual.shape != (29,):
                return False, f"boundary_state_shape={actual.shape}"
            target = actual.copy()
            target[15:22] = np.asarray(goal)[:7]
            target[22:29] = np.asarray(goal)[7:]
            lp_t, lr_t, rp_t, rr_t = fk.compute_ee_transforms(target)
            lp_a, lr_a, rp_a, rr_a = fk.compute_ee_transforms(actual)
            position_error = max(
                float(np.linalg.norm(lp_t - lp_a)),
                float(np.linalg.norm(rp_t - rp_a)),
            )

            def _rotation_error(target_r, actual_r):
                cosine = (float(np.trace(target_r.T @ actual_r)) - 1.0) * 0.5
                return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

            rotation_error = max(
                _rotation_error(lr_t, lr_a), _rotation_error(rr_t, rr_a)
            )
            speed = float(np.max(np.abs(measured_velocity)))
            ok = position_error <= 0.03 and rotation_error <= 0.10 and speed <= 0.08
            return (
                ok,
                f"ee_pos_error={position_error:.4f}m "
                f"ee_rot_error={rotation_error:.4f}rad speed={speed:.3f}rad/s",
            )

        return _check

    # Issue #125 Phase 9b: --policy-variant 指定時は move_table_base skill を
    # VlaSkill (MoveTableBaseVlaSkill) に差し替え。未指定なら既存 SampleVLASkill
    # (pose mock) fallback (backward compat)。
    # Phase B-1/B-3: actuator 選択 helper (real / mock 両対応)。
    def _build_waist_actuator():
        """--use-real-waist ならshared arm_actuator を wrap した G1WaistActuator、
        未指定なら MockWaistActuator (dev / smoke)。"""
        if args.use_real_waist:
            from inference.desktop.lower_policy.actuators.waist import G1WaistActuator

            return G1WaistActuator(arm_actuator)
        from inference.desktop.lower_policy.actuators.waist import MockWaistActuator

        return MockWaistActuator()

    _real_hand_instance: Optional[object] = None  # Phase B-3: shutdown で stop()
    # --synthetic-hand-state のときだけ使う共有 mock (指令エコーの供給元)
    _shared_mock_hand_instance: Optional[object] = None

    def _build_hand_actuator():
        """--use-real-hand なら G1HandActuator (SDK 依存)、未指定なら MockHandActuator。"""
        nonlocal _real_hand_instance
        if args.use_real_hand:
            from inference.desktop.lower_policy.actuators.hand import G1HandActuator

            # 全 skill で 1 instance を共有 (SDK publisher 衝突回避)
            if _real_hand_instance is None:
                _real_hand_instance = G1HandActuator(
                    velocity_limit_rad_s=(
                        PHASE1_13_HAND_VELOCITY_RAD_S
                        if phase1_profile == "1.13"
                        else None
                    )
                )
                if _start_hand_during_registry_build(
                    phase1_profile=phase1_profile,
                    phase3_active=phase3_active,
                ):
                    _real_hand_instance.start()  # backward-compatible standard path
                    print(
                        "[init] hand actuator = G1HandActuator (Dex1-1 SDK real)",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "[init] Dex1 actuator initialized; publisher start deferred "
                        "until the final physical safety gate",
                        file=sys.stderr,
                    )
            return _real_hand_instance
        from inference.desktop.lower_policy.actuators.hand import MockHandActuator

        if not args.synthetic_hand_state:
            return MockHandActuator()
        # 合成 state は「直近の hand 指令」をエコーするので、skill をまたいで
        # 1 instance を共有する (skill ごとに作り直すと指令履歴が切れる)。
        nonlocal _shared_mock_hand_instance
        if _shared_mock_hand_instance is None:
            _shared_mock_hand_instance = MockHandActuator()
        return _shared_mock_hand_instance

    def _build_vla_skill_from_variant(
        variant, skill_name, VlaSkillClass, *, extra_skill_kwargs=None
    ):
        """slot → VlaSkill (assembly が組み立て、policy を shutdown 対象に登録する)。"""
        built = assembly.build_vla_skill(
            skill_name=skill_name,
            vla_skill_cls=VlaSkillClass,
            variant=variant,
            skill_config=skill_cfg_raw,
            waist_actuator=_build_waist_actuator(),
            hand_actuator=_build_hand_actuator(),
            fk_factory=fk_factory,
            # 先読み (ModelResidency) が読み込みと解放を握るので、stage 経路では
            # 必ず DeferredPolicy に包む。--gpu-models=all のときは起動時に全部読む。
            deferred=phase3_active and gpu_models is not None,
            extra_skill_kwargs=extra_skill_kwargs,
        )
        _register_policy_resource(built.policy)
        print(
            f"[init] {skill_name}: waist={'Real' if args.use_real_waist else 'Mock'}"
            f"{'/dispatch_off' if not built.dispatch_waist else ''}、"
            f"hand={'Real' if args.use_real_hand else 'Mock'}",
            file=sys.stderr,
        )
        return built

    def _build_vla_skill(
        variant_name, skill_name, VlaSkillClass, *, extra_skill_kwargs=None
    ):
        """CLI arg → VlaSkill instance (未指定なら MockSkill fallback)。"""
        if variant_name is None:
            return MockSkill(skill_name)
        from inference.desktop.lower_policy.policies.config_loader import (
            load_policy_variant as _load_variant,
        )

        if not args.policy_config.exists():
            sys.exit(f"policy config not found: {args.policy_config}")
        variant = _load_variant(args.policy_config, variant_name)
        return _build_vla_skill_from_variant(
            variant,
            skill_name,
            VlaSkillClass,
            extra_skill_kwargs=extra_skill_kwargs,
        ).skill

    if args.policy_variant is not None:
        from inference.desktop.lower_policy.policies.config_loader import (
            load_policy_variant,
            resolve_policy_class,
        )
        from inference.desktop.lower_policy.skills.vla_skill import (
            MoveTableBaseVlaSkill,
        )

        if not args.policy_config.exists():
            sys.exit(f"policy config not found: {args.policy_config}")
        variant = load_policy_variant(args.policy_config, args.policy_variant)
        built = _build_vla_skill_from_variant(
            variant, "move_table_base", MoveTableBaseVlaSkill
        )
        move_table_base_skill = built.skill
    else:
        move_table_base_skill = SampleVLASkill.from_config(
            skills_section["move_table_base"], skill_name="move_table_base"
        )
        print(
            "[init] move_table_base = SampleVLASkill (pose mock、--policy-variant 未指定)",
            file=sys.stderr,
        )

    # Issue #125: --policy-variant-<skill> 指定時は該当 mock を VlaSkill に差替
    # VlaSkill subclass は lazy import (mock 経路では load 不要)
    if any(
        v is not None
        for v in (
            args.policy_variant_pick,
            args.policy_variant_insert,
            args.policy_variant_rotate_leg,
            args.policy_variant_flip,
            args.policy_variant_rotate_table_base,
        )
    ):
        from inference.desktop.lower_policy.skills.vla_skill import (
            FlipTableVlaSkill,
            InsertTableLegVlaSkill,
            PickTableLegVlaSkill,
            RotateLegToTightenVlaSkill,
            RotateTableBaseVlaSkill,
        )
    else:
        PickTableLegVlaSkill = None  # type: ignore[assignment]
        InsertTableLegVlaSkill = None  # type: ignore[assignment]
        RotateLegToTightenVlaSkill = None  # type: ignore[assignment]
        FlipTableVlaSkill = None  # type: ignore[assignment]
        RotateTableBaseVlaSkill = None  # type: ignore[assignment]

    rule_based_gripper = None
    if rule_based_pick_active:
        from inference.desktop.lower_policy.motion_limits import (
            load_motion_limits_for_skill,
        )

        pick_leg_cfg = skills_section.get("pick_table_leg")
        if not isinstance(pick_leg_cfg, dict):
            sys.exit("skill config invalid: 'skills.pick_table_leg' must be a mapping")
        calibration_override = None
        if args.rule_based_pick_calibration is not None:
            try:
                calibration_override = load_verified_rule_pick_calibration(
                    args.rule_based_pick_calibration
                )
            except (OSError, ValueError, yaml.YAMLError) as exc:
                sys.exit(f"rule-based pick calibration rejected: {exc}")
        elif args.actuate:
            sys.exit(
                "rule-based pick actuation blocked: a verified physical head-camera "
                "calibration is required via --rule-based-pick-calibration; the "
                "skill_config.yaml D435 transform is provisional"
            )
        # Issue #123 の rule-based 経路だけが専用 Dex1 DDS と解析 IK を所有する。
        # learned VLA が選ばれた場合は既存の共通 hand actuator 経路を維持する。
        rule_based_gripper = Dex1DdsGripper.from_config(pick_leg_cfg.get("dex1"))
        pick_table_leg_skill = build_pick_table_leg(
            skills_section,
            rule_based_gripper,
            control_hz=args.hz,
            motion_limits=load_motion_limits_for_skill(skill_cfg_raw, "pick_table_leg"),
            obb_grasp_override=calibration_override,
        )
        print(
            "[init] pick_table_leg=rule_based stages="
            f"{[stage.name for stage in pick_table_leg_skill.stages]}",
            file=sys.stderr,
        )
    elif args.policy_variant_pick is not None:
        pick_skill_class = PickTableLegVlaSkill
        pick_extra_kwargs = None
        if args.pick_leg_hybrid:
            from inference.desktop.pick_leg_hybrid.real_skill import (
                RealPickLegHybridVlaSkill,
            )

            insert_initial = _effective_initial_pose("insert_table_leg")
            # The hybrid never dispatches waist.  Apply this before assembly so
            # both its MotionLimiter and returned BuiltSkill contract agree.
            skills_section["pick_table_leg"]["dispatch_waist"] = False
            pick_skill_class = RealPickLegHybridVlaSkill
            pick_extra_kwargs = {
                "hybrid_config_path": args.pick_leg_hybrid_config,
                "hybrid_vlm_endpoint": args.pick_leg_vlm_endpoint,
                "hybrid_vlm_model": args.pick_leg_vlm_model,
                "phase3_executor": args.pick_leg_phase3_executor,
                "next_initial_arm_target": insert_initial.arm_position_rad,
                "next_initial_hand_target": insert_initial.dex1_target_rad,
                "boundary_taskspace_arrival": _boundary_taskspace_arrival(args),
            }
            print(
                "[hybrid] Stage 1-4 pick_table_leg replaced with "
                "VLM -> GR00T -> G1 IK MP -> rule-based handover -> "
                "insert initial pose",
                file=sys.stderr,
            )
        pick_table_leg_skill = _build_vla_skill(
            args.policy_variant_pick,
            "pick_table_leg",
            pick_skill_class,
            extra_skill_kwargs=pick_extra_kwargs,
        )
    else:
        pick_table_leg_skill = MockSkill("pick_table_leg")
    insert_table_leg_skill = _build_vla_skill(
        args.policy_variant_insert, "insert_table_leg", InsertTableLegVlaSkill
    )
    rotate_leg_to_tighten_skill = _build_vla_skill(
        args.policy_variant_rotate_leg,
        "rotate_leg_to_tighten",
        RotateLegToTightenVlaSkill,
    )
    flip_table_skill = _build_vla_skill(
        args.policy_variant_flip, "flip_table", FlipTableVlaSkill
    )
    # Issue #128 Phase 3: rotate_table_base (task 5、move_table_base から split した新 skill)
    rotate_table_base_skill = _build_vla_skill(
        args.policy_variant_rotate_table_base,
        "rotate_table_base",
        RotateTableBaseVlaSkill,
    )

    configured_setup = SetupSkill.from_config(skills_section["setup"])
    skill_registry = {
        "setup": configured_setup,
        "move_to_table": MoveToTable.from_config(
            skills_section["move_to_table"], actuator
        ),
        "move_table_base": move_table_base_skill,
        "rotate_table_base": rotate_table_base_skill,
        "pick_table_leg": pick_table_leg_skill,
        "insert_table_leg": insert_table_leg_skill,
        "rotate_leg_to_tighten": rotate_leg_to_tighten_skill,
        "flip_table": flip_table_skill,
    }
    phase_transitions = None
    phase_enter_check = None
    policy_start_gate_stage: Optional[int] = None
    if rule_based_pick_active:
        rule_initial_pose = _effective_initial_pose("pick_table_leg")
        skill_registry["rule_pick_pre_motion"] = CollisionAwareArmPreMotionSkill(
            rule_initial_pose.arm_position_rad,
            skill_name="rule_pick_pre_motion",
            velocity_limit_rad_s=0.5,
            acceleration_limit_rad_s2=1.0,
            measured_tolerance_rad=0.10,
            stage_timeout_s=15.0,
        )
        phase_transitions = {
            "rule_pick_pre_motion": ["pick_table_leg"],
            "pick_table_leg": [],
        }
        # Completion, not a perception predicate, owns this transition.  This
        # prevents a visible leg from starting Cartesian pick motion before the
        # collision-aware arm path has reached the dataset frame-zero pose.
        phase_enter_check = {"pick_table_leg": lambda _dets, _state: False}
        print(
            "[init] rule-based pick startup=collision-aware arm pre-motion -> "
            "dataset frame-zero -> automatic 8-stage FSM",
            file=sys.stderr,
        )
    elif phase1_profile is not None:
        # Phase smoke profiles walk with the measured lowered pose.  The old
        # evaluator's collision-aware loop is run only after the base stops.
        skill_registry["setup"] = MeasuredArmWalkHoldSkill(
            dwell_sec=0.5,
            lowered_pose_rad=_walk_lowered_pose(skill_cfg_raw),
            lowering_settings=_walk_lowering_settings(skill_cfg_raw),
        )
        skill_registry["post_walk_settle"] = PostWalkArmSettleSkill(
            actuator,
            minimum_settle_sec=PHASE1_POST_WALK_SETTLE_SECONDS,
        )
        skill_registry["arm_pre_motion"] = CollisionAwareArmPreMotionSkill(
            configured_setup.stages[-1].pose_rad,
            velocity_limit_rad_s=0.5,
            acceleration_limit_rad_s2=1.0,
            measured_tolerance_rad=0.10,
            stage_timeout_s=15.0,
        )
        phase_transitions = {
            key: list(value) for key, value in DEFAULT_TRANSITIONS.items()
        }
        phase_transitions["move_to_table"] = ["post_walk_settle"]
        phase_transitions["post_walk_settle"] = ["arm_pre_motion"]
        phase_transitions["arm_pre_motion"] = ["move_table_base"]
        phase_enter_check = dict(DEFAULT_ENTER_CHECK)
        phase_enter_check["post_walk_settle"] = lambda _dets, _state: False
        phase_enter_check["arm_pre_motion"] = lambda _dets, _state: False
    elif phase3_active and not rule_based_pick_active:
        # Issue #128 Phase 3: stage-based orchestration。skill_config.yaml から
        # rotate_table_base / flip_table の initial_pose を読み、CollisionAware
        # ArmPreMotionSkill の final_pose に渡す。Phase 1 profile と同じ
        # measured lowered walk + post_walk_settle も再利用 (安全境界維持)。
        model_transition_skips = (
            frozenset({("pick_table_leg", "insert_table_leg")})
            if args.pick_leg_hybrid
            else frozenset()
        )

        def _stage_sequence(stage: int) -> list[str]:
            return build_stage_skill_sequence(
                stage,
                is_start_stage=(stage == first_phase3_stage),
                include_hand=include_hand_in_head,
                skip_model_transition_pairs=model_transition_skips,
            )

        # 歩行を含む stage 0 のみ MeasuredArmWalkHoldSkill + PostWalkArmSettleSkill
        # を挟む。continuous runでもStage 0は腕を下げたまま終える。
        if 0 in selected_stages:
            skill_registry["setup"] = MeasuredArmWalkHoldSkill(
                dwell_sec=0.5,
                lowered_pose_rad=_walk_lowered_pose(skill_cfg_raw),
                lowering_settings=_walk_lowering_settings(skill_cfg_raw),
                # 会場は運営 IK の解を task-space で確かめる (頭の手順と同じ判定)
                measured_convergence_checker=_boundary_convergence_checker(
                    STAGE_HEAD_SKILL[0]
                ),
                latch_check=_walk_latch_check(skill_cfg_raw, args.walk_lowering_check),
            )
            print(
                "[init] walk latch check = "
                f"{_walk_latch_check(skill_cfg_raw, args.walk_lowering_check)} "
                f"({'cli' if args.walk_lowering_check else 'skill_config.yaml'})",
                file=sys.stderr,
            )
            skill_registry["post_walk_settle"] = PostWalkArmSettleSkill(
                actuator,
                minimum_settle_sec=PHASE1_POST_WALK_SETTLE_SECONDS,
            )

        # 頭の手順 (手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持)。
        # 評価経路と同じ関数で作る (Issue #141 D2 / D7-1)。
        for stage in selected_stages:
            head_skill = STAGE_HEAD_SKILL.get(stage)
            if head_skill is None or (stage != first_phase3_stage and stage != 0):
                continue
            if any(
                name in skill_registry
                for name in _stage_sequence(stage)
                if name.startswith(
                    ("wait_for_go_live_", "hand_", "arm_pre_motion_for_", "hold_pose_for_")
                )
            ):
                continue  # 同じ skill の頭の手順は stage 間で使い回す
            for head in assembly.build_head_procedure(
                skill_config=skill_cfg_raw,
                skill_name=head_skill,
                initial_pose=_effective_initial_pose(head_skill),
                hand_actuator=_build_hand_actuator(),
                hold_sec=args.setup_hold_sec,
                include_hand=include_hand_in_head,
                measured_convergence_checker=_boundary_convergence_checker(head_skill),
            ):
                skill_registry[head.name] = head

        # Insert a finite, measured transition before every subsequent model.
        # The hybrid pick already performs and verifies pick->insert internally,
        # so that one boundary is represented exactly once rather than moving
        # the load-bearing arms out through the clearance route a second time.
        for stage in selected_stages:
            for previous_skill, next_skill in model_transition_pairs_for_stage(
                stage,
                is_start_stage=(stage == first_phase3_stage),
                skip_model_transition_pairs=model_transition_skips,
            ):
                for transition_skill in assembly.build_model_transition_procedure(
                    skill_config=skill_cfg_raw,
                    previous_skill=previous_skill,
                    next_skill=next_skill,
                    initial_pose=_effective_initial_pose(next_skill),
                    hand_actuator=_build_hand_actuator(),
                    hold_sec=args.setup_hold_sec,
                    include_hand=include_hand_in_head,
                    published_arm_target_provider=lambda: (
                        arm_actuator.read_last_published_targets()[0]
                    ),
                    measured_convergence_checker=_boundary_convergence_checker(
                        next_skill
                    ),
                ):
                    existing = skill_registry.get(transition_skill.name)
                    if existing is None:
                        skill_registry[transition_skill.name] = transition_skill
                    elif type(existing) is not type(transition_skill):
                        raise RuntimeError(
                            "model transition registry collision: "
                            f"{transition_skill.name}"
                        )

        # 自分で運ぶ境界 (hybrid pick -> insert) も model の載せ替えは HOLD の中で
        # 行う (Codex #5)。腕・手は動かさず、前 skill の最後の姿勢を保持するだけ。
        from inference.desktop.lower_policy.skills.hold_pose import HoldPoseSkill
        from inference.desktop.orchestrator import self_bridged_transition_skill_names

        for stage in selected_stages:
            for pair in model_transition_pairs_for_stage(
                stage,
                is_start_stage=(stage == first_phase3_stage),
                skip_model_transition_pairs=frozenset(),
            ):
                if pair not in model_transition_skips:
                    continue
                for hold_name in self_bridged_transition_skill_names(*pair):
                    existing = skill_registry.get(hold_name)
                    if existing is None:
                        skill_registry[hold_name] = HoldPoseSkill(
                            args.setup_hold_sec, name=hold_name
                        )
                    elif not isinstance(existing, HoldPoseSkill):
                        raise RuntimeError(
                            f"model transition registry collision: {hold_name}"
                        )

        # Treat the generated transition graph as a safety contract, not merely
        # a naming convention.  Every learned-policy boundary must contain the
        # collision-aware arm move, the measured Dex1 move (when enabled), and
        # the stable hold before the next model can be dispatched.  Validate the
        # complete chain and the concrete registry before any real publisher is
        # started.
        finite_transition_prefixes = (
            "arm_transition_",
            "hand_transition_",
            "hold_transition_",
        )
        for stage in selected_stages:
            sequence = _stage_sequence(stage)
            validate_stage_model_transition_chain(
                stage,
                sequence,
                is_start_stage=(stage == first_phase3_stage),
                include_hand=include_hand_in_head,
                self_bridged_pairs=model_transition_skips,
            )
            missing_transition_skills = [
                name
                for name in sequence
                if name.startswith(finite_transition_prefixes)
                and name not in skill_registry
            ]
            if missing_transition_skills:
                raise RuntimeError(
                    "model transition skills are not registered; refusing to "
                    f"start actuation: {missing_transition_skills}"
                )

        # Stage の skill 列 → transition graph
        phase_transitions = (
            None
            if args.phase3_full
            else build_stage_transitions(
                args.stage,
                is_start_stage=True,
                include_hand=include_hand_in_head,
                skip_model_transition_pairs=model_transition_skips,
            )
        )
        # enter_check: stage の run は YOLO (上位 policy) で次の skill へ進まない。
        # 進むのは完了と時間切れだけ。YOLO の検出は overlay 用に policy へ渡し続ける
        # (orchestrator.stage_enter_check)。
        phase_enter_check = stage_enter_check()

        # A separately selected Stage starts only after its frame-zero arm/hand pose
        # has converged and the operator explicitly confirms it.  Keep Stage 0
        # unchanged (walk + move to the first pick pose only), and do not insert
        # gates at model boundaries inside a Stage.  The gate is itself a normal
        # ticking Skill, so arm commands, boundary packets, camera freshness checks,
        # and Dex1 holding continue while stdin is waiting.
        policy_start_gate_stage = _selected_policy_start_gate_stage(args)
        if policy_start_gate_stage is not None:
            from inference.desktop.lower_policy.skills.operator_gate import (
                OperatorConfirmationHoldSkill,
            )

            first_policy = STAGE_HEAD_SKILL[policy_start_gate_stage]
            gate_name = f"operator_gate_for_{first_policy}"
            if phase_transitions is not None:
                _insert_policy_start_gate_transition(
                    phase_transitions, first_policy=first_policy
                )
            # 問いを出す時点で開始姿勢に届いたかを、準備動作と同じ判定で見る (pose lane は
            # 手先、sdk・joint lane は関節角)。準備動作が時間切れで進んだときに「届いた」と
            # 出さないため (GB10 の Stage 5 で見つけた、2026-09-25)。
            skill_registry[gate_name] = OperatorConfirmationHoldSkill(
                name=gate_name,
                next_skill_name=first_policy,
                arrival_check=_gate_arrival_check(
                    _effective_initial_pose(first_policy).arm_position_rad,
                    _boundary_convergence_checker(first_policy),
                    float(
                        (skill_cfg_raw.get("arm_pre_motion") or {}).get(
                            "measured_tolerance_rad", 0.10
                        )
                    ),
                ),
            )
            print(
                f"[init] Stage {policy_start_gate_stage} policy-start gate: "
                "initial pose -> "
                f"Enter -> {first_policy}",
                file=sys.stderr,
            )
        print(
            (
                f"[init] Phase 3 continuous stages="
                f"{args.phase3_start_stage}..{args.phase3_end_stage}"
                if args.phase3_full
                else f"[init] Phase 3 stage={args.stage} sequence="
                f"{STAGE_SKILL_SEQUENCES[args.stage]}"
            ),
            file=sys.stderr,
        )

    dispatcher = SkillDispatchLowerPolicy(skill_registry)

    if phase3_active and not rule_based_pick_active:
        print(f"[init] Phase 3 gpu_models={args.gpu_models}", file=sys.stderr)
        if gpu_models is None:
            print(
                "[init] WARNING: --gpu-models=all keeps every configured expert "
                "resident; use only when GPU/host-memory headroom has been verified",
                file=sys.stderr,
            )

    stage_hard_timeouts: dict[str, float] = {}
    stage_timeout_actions: dict[str, str] = {}
    if rule_based_pick_active:
        # The rule skill has per-stage measured-pose and gripper timeouts.  A
        # learned-policy duration derived from the dataset must not truncate
        # its eight-stage state machine.
        stage_hard_timeouts = {}
        stage_timeout_actions = {}
    elif phase3_active:
        timeout_skill_names = (
            {
                name
                for stage in range(args.phase3_start_stage, args.phase3_end_stage + 1)
                for name in STAGE_SKILL_SEQUENCES[stage]
            }
            if args.phase3_full
            else set(STAGE_SKILL_SEQUENCES[args.stage])
        )
        for skill_name in timeout_skill_names:
            section = skills_section.get(skill_name, {})
            if "max_seconds_hard" not in section:
                continue
            timeout_s = float(section["max_seconds_hard"])
            if timeout_s <= 0:
                raise ValueError(f"skills.{skill_name}.max_seconds_hard must be > 0")
            stage_hard_timeouts[skill_name] = timeout_s
            action = section.get("on_timeout", "advance")
            if action not in TIMEOUT_ACTIONS:
                raise ValueError(
                    f"skills.{skill_name}.on_timeout must be one of {TIMEOUT_ACTIONS}, "
                    f"got {action!r}"
                )
            stage_timeout_actions[skill_name] = action
        # The hybrid is a finite multi-controller state machine, not the old
        # learned pick expert.  Its validated wall-clock budget therefore owns
        # the pick timeout.  What happens on timeout stays the YAML
        # ``skills.pick_table_leg.on_timeout`` (advance): 次へ進む道は完了と
        # 時間切れだけで、止めるのは人。insert へは腕の遷移 (collision-aware) を
        # 通って入る。
        if args.pick_leg_hybrid:
            assert hybrid_pick_cfg is not None
            stage_hard_timeouts["pick_table_leg"] = (
                hybrid_pick_cfg.runtime.hard_timeout_sec
            )
            print(
                "[hybrid] pick_table_leg timeout="
                f"{hybrid_pick_cfg.runtime.hard_timeout_sec:g}s "
                f"action={stage_timeout_actions.get('pick_table_leg', 'advance')}",
                file=sys.stderr,
            )
        print(
            f"[init] Phase 3 hard timeouts={stage_hard_timeouts} "
            f"actions={stage_timeout_actions}",
            file=sys.stderr,
        )

    actuator_send_fn = arm_actuator.send_action

    # 6) initial_skill 決定。--stage 指定時は stage の先頭 skill、未指定なら setup。
    # (Resume 経路は Issue #128 Phase 3 完了に伴い削除、--stage=N で再実行が代替)
    if phase3_active:
        from inference.desktop.orchestrator import STAGE_SKILL_SEQUENCES

        first_stage = args.phase3_start_stage if args.phase3_full else args.stage
        initial_skill = (
            "rule_pick_pre_motion"
            if rule_based_pick_active
            else _stage_sequence(first_stage)[0]
        )
    else:
        initial_skill = "setup"

    # 6a) Log sink (JSONL append)。default 有効化: --no-log opt-out 以外は --log
    # 明示指定 or 自動生成 path に log を残す (debug / 事後解析用)。
    log_sink: Optional[TextIO] = None
    if not args.no_log:
        log_path = args.log if args.log is not None else _default_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_sink = log_path.open("w")
        print(f"[init] log_sink -> {log_path}", file=sys.stderr)

    source: Optional[object] = None
    wrist_left_source: Optional[object] = None
    wrist_right_source: Optional[object] = None
    joint_state_source: Optional[object] = None
    dex1_state_source: Optional[object] = None
    _boundary_sink: Optional[object] = None
    arm_actuator_started = False
    # finally が読むので try の外で初期化する (センサや記録の構築で落ちても
    # _safe_shutdown まで到達させる)。
    recorder: Optional[Any] = None
    single_stage_residency: Optional[Any] = None
    shutdown_hand_actuator: Optional[object] = None
    return_checker = None
    try:
        # 7) Frame source (ROS2 CompressedImage を cyclonedds direct で subscribe)。
        #    Ros2FrameSource の init 内で SDK ChannelFactory 経由で listener を register。
        #    cyclonedds 側が独自 thread で callback を drain するので、rclpy spin 相当の
        #    外部 thread は不要。topology β 前提 (Desktop 頭脳 / Orin publish)。
        # Source は packed stereo のまま保持する。Orchestrator が YOLO には
        # 学習時と同じ左眼だけを渡し、VLA observation には左右眼を渡す。
        #
        # 7b) 手首カメラは `--no-wrist-cameras` で外せる。D405 の color imager 出力は
        #     848x480 単一 sensor なので stereo split 不要 (packed = 全幅そのまま)。
        # 関節角と Dex1 の state も同じところで作る (Issue #141 1-8、評価経路と同じ関数)。
        # 会場のリグは Dex1 state を配信しない (boundary/states.py)。その場合だけ
        # 「開始 skill の frame-0 値で seed → 以降 hand 指令をエコー」の合成に倒す。
        # 定数 (全開) ではいけない: insert 系は脚を掴んだ状態が frame 0 のため。
        synthetic_hand_initial_rad = None
        if args.synthetic_hand_state:
            # The orchestrator's first node is normally a finite scaffold
            # (`wait_for_go_live_*`, `hand_open_*`, or `setup`), none of which
            # owns dataset frame-zero metadata.  Seed from the learned skill
            # whose pose that scaffold is preparing.
            if rule_based_pick_active:
                synthetic_hand_pose_skill = "pick_table_leg"
            elif first_phase3_stage is not None:
                synthetic_hand_pose_skill = STAGE_HEAD_SKILL[first_phase3_stage]
            else:
                raise ValueError(
                    "--synthetic-hand-state requires a selected Stage or "
                    "rule-based pick"
                )
            synthetic_hand_initial_rad = _effective_initial_pose(
                synthetic_hand_pose_skill
            ).dex1_target_rad
            print(
                "[init] Dex1 state = synthetic "
                f"(seed from {synthetic_hand_pose_skill} frame-0: "
                f"{synthetic_hand_initial_rad}); "
                + (
                    "commands are echoed to state and published through the "
                    "official (T,25) hand columns"
                    if args.action_sink == "boundary"
                    else "hand actuator stays mock; NO gripper command is published"
                ),
                file=sys.stderr,
            )

        sensors = assembly.build_sensor_sources(
            head_topic=args.topic,
            wrist_left_topic=None if args.no_wrist_cameras else args.wrist_left_topic,
            wrist_right_topic=None if args.no_wrist_cameras else args.wrist_right_topic,
            joint_source=args.joint_source,
            joint_state_topic=args.joint_state_topic,
            head_source=args.head_source,
            zmq_endpoint=args.zmq_endpoint,
            boundary_state_host=args.boundary_state_host,
            boundary_state_port=args.boundary_state_port,
            synthetic_hand_initial_rad=synthetic_hand_initial_rad,
        )
        if synthetic_hand_initial_rad is not None:
            # skill 共有 mock を繋いで、以降は policy の hand 指令をそのまま state に返す。
            sensors.dex1.bind_command_source(_build_hand_actuator())

        def _log_boundary_taskspace(record: dict) -> None:
            """publish した raw (T,25) を JSONL で残す。

            WBC_RUNBOOK §5: "a gripper that never closes during a pick stage is
            visible straight from your own published data, no video needed"。
            """
            if log_sink is None:
                return
            log_sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_sink.flush()

        # action の出口を boundary に差し替える。orchestrator が actuator_send_fn に
        # 渡すのは腕 14-D だけなので、腰と手は mock actuator の latest から組み直す。
        if args.action_sink == "boundary":
            from inference.desktop.lower_policy.actuators.boundary_sink import (
                BoundaryActionSink,
                assemble_action19,
            )

            # FK は渡さない = 運営 IK と同じ運動学 (ORGANIZER_IK_URDF_PATH、Issue #164)。
            _boundary_sink = BoundaryActionSink(
                lane=args.boundary_lane,
                clamp_wrist_roll=args.wrist_roll_clamp == "on",
                port=args.boundary_port,
                host=args.boundary_host,
                log_fn=_log_boundary_taskspace,
                # 送信時刻を運営 adapter の時計 (PC2) に揃える (Issue #161、C2-01)。
                # 差は PC2 のカメラ配信に入っている送信時刻から推定する。
                sender_clock_offset_fn=getattr(
                    sensors.head, "sender_clock_offset_s", None
                ),
            )
            # WBC_RUNBOOK §1/§4 Step 5 の想定は「PC2 の client が :5556 を bind」。
            # Thor で動かすと bind 先が変わるので、運営側の設定が要ることを明示する。
            print(
                "[boundary] :5556 is bound on this host. Start the organizer adapter "
                "(wbc_driver.py) with --actions-host pointing at this host.",
                file=sys.stderr,
            )
            _boundary_hand = _build_hand_actuator()
            print(
                "[boundary] "
                + (
                    BOUNDARY_JOINT_WAIST_NOTE
                    if args.boundary_lane == "joint"
                    else BOUNDARY_WAIST_NOTE
                ),
                file=sys.stderr,
            )

            def _send_to_boundary(arms14) -> bool:
                """1 row を publish する。publish したら True。"""
                joint = sensors.joint.get()
                received_ns = getattr(joint, "received_monotonic_ns", None)
                if (
                    joint is None
                    or received_ns is None
                    or not (0 <= time.monotonic_ns() - int(received_ns) <= 500_000_000)
                ):
                    # 関節が取れない tick は publish しない (壊れた EE を流さない)。
                    return False
                body_q29 = np.asarray(joint.position, dtype=np.float64)[:29]
                if body_q29.shape != (29,) or not np.all(np.isfinite(body_q29)):
                    return False
                action19 = assemble_action19(
                    # 腰は常に実測値 (BOUNDARY_WAIST_NOTE)。
                    None,
                    arms14,
                    getattr(_boundary_hand, "latest", None),
                    measured_waist3=body_q29[12:15],
                    fallback_hand2=(
                        synthetic_hand_initial_rad
                        if synthetic_hand_initial_rad is not None
                        else (0.0, 0.0)
                    ),
                )
                _boundary_sink.send_action(
                    action19,
                    body_q29,
                    navigate_cmd=actuator.latest,
                )
                return True

            # 送信窓口は arm_actuator.send_action の 1 本だけ。orchestrator が
            # _send_to_boundary を直に呼ぶと「最後に送った target」が更新されず、
            # 歩行の再送・model 遷移・終了処理が起動時の姿勢を送っていた。
            actuator_send_fn = arm_actuator.send_action

            def _measured_boundary_arms() -> Optional[np.ndarray]:
                snapshot = sensors.joint.get()
                if snapshot is None:
                    return None
                received_ns = getattr(snapshot, "received_monotonic_ns", None)
                if received_ns is None or not (0 <= time.monotonic_ns() - int(received_ns) <= 500_000_000):
                    return None
                body = np.asarray(snapshot.position, dtype=np.float64)
                if body.shape != (29,) or not np.all(np.isfinite(body[15:29])):
                    return None
                return np.concatenate([body[15:22], body[22:29]])

            arm_actuator.configure(
                sender=_send_to_boundary,
                state_getter=_measured_boundary_arms,
            )
            actuator.set_publish_callback(
                lambda: arm_actuator.send_action(
                    arm_actuator.read_last_published_targets()[0]
                )
            )
        source = sensors.head
        wrist_left_source = sensors.wrist_left
        wrist_right_source = sensors.wrist_right
        joint_state_source = sensors.joint
        dex1_state_source = sensors.dex1
        print(
            f"[init] joint_source={args.joint_source}"
            + (
                f" topic={args.joint_state_topic}"
                if args.joint_source == "joint_states"
                else ""
            ),
            file=sys.stderr,
        )
        print(
            "[init] Dex1 left/right state topics subscribed (read-only)",
            file=sys.stderr,
        )
        if not args.no_wrist_cameras:
            print(
                f"[init] wrist_left={args.wrist_left_topic}, "
                f"wrist_right={args.wrist_right_topic}",
                file=sys.stderr,
            )

        # Issue #141 D1 #4: 本番の run も評価と同じ形で記録する (wandb への upload は
        # 本番では行わない = API key が要らない)。
        if not args.no_record:
            from inference.desktop.recording import RunRecorder, record_policy_tick

            record_dir = args.record_dir or _default_record_dir()
            recorder = RunRecorder(record_dir)
            recorder.__enter__()
            recorder.write_metadata(
                {
                    "run": "production",
                    "stage_start": first_phase3_stage,
                    "stage_end": args.phase3_end_stage if args.phase3_full else None,
                    "gpu_models": args.gpu_models,
                    "joint_source": args.joint_source,
                    # Issue #140 の重心 fallback は env で上書きできるので、実際に効いた
                    # 値を残す (A/B の run を後から取り違えないため)。
                    "median_match": dict(
                        zip(
                            ("max_centroid_dist", "ambiguity_ratio"),
                            resolve_median_match_params(
                                cleanup_config.get("median_filter", {})
                            ),
                        )
                    ),
                    "setup_hold_sec": args.setup_hold_sec,
                    "hz": args.hz,
                    "actuate": args.actuate,
                    "use_real_waist": args.use_real_waist,
                    "use_real_hand": args.use_real_hand,
                    "pick_leg_hybrid": hybrid_pick_runtime,
                    "policy_start_operator_gate_stage": policy_start_gate_stage,
                    # どの ckpt で走ったか + その値が CLI か config の既定か
                    # (Issue #148)。後から run を取り違えないため。
                    "policy_variants": {
                        slot: getattr(args, attr)
                        for slot, attr in _POLICY_SLOT_ATTRS.items()
                        if getattr(args, attr) is not None
                    },
                    "policy_variant_source": variant_source,
                    "started_at": datetime.now().isoformat(),
                }
            )
            print(f"[init] recording -> {record_dir}", file=sys.stderr)

        learned_skill_names = frozenset(
            name
            for stage in (selected_stages if phase3_active else ())
            for name in STAGE_SKILL_SEQUENCES[stage]
        )
        probe_state: dict = {}

        def _make_tick_hook(residency):
            """tick ごとの hook: 記録 + GPU に置く model の入れ替え。"""
            if recorder is None and residency is None:
                return None

            def _hook(result, obs, skill) -> None:
                if residency is not None:
                    residency.on_skill_started(getattr(skill, "name", None))
                if recorder is not None:
                    record_policy_tick(
                        recorder,
                        obs,
                        skill,
                        learned_skills=learned_skill_names,
                        arm_actuator=arm_actuator,
                        probe_state=probe_state,
                    )

            return _hook

        def _build_residency(sequence: list[str]):
            """stage の skill 列から先読みの管理を作る (Issue #141 束 1-12 / D7-2)。

            GPU に置く model の数を `--gpu-models` に保ち、範囲外を解放する。
            model を持たない skill (頭の手順など) は無視される。
            """
            if not phase3_active or rule_based_pick_active:
                return None
            policies = {}
            for name in sequence:
                policy = getattr(skill_registry.get(name), "_policy", None)
                if policy is not None and hasattr(policy, "prepare"):
                    policies[name] = policy
            if not policies:
                return None
            from inference.desktop.lower_policy.policies.residency import ModelResidency

            # 既定は「今の skill + 次の 1 つ」。切替を隠すのにこれで足りる
            # (読み込み 約 8 秒 < 各 skill の 21〜58 秒)。**全部載せる必要は無い。**
            #
            # 以前の既定は len(policies) = 全部だったが、residency の帳簿がずれて
            # 実際には 1 つしか載っていなかったため表面化していなかった
            # (VlaSkill._on_stop が residency を通さず解放していた、2026-09-21)。
            # そのズレを直した今、既定を全部のままにすると 53D×4 + pick で
            # 約 26 GiB を本当に載せに行く。機体によっては入らない。
            # 増やしたいときは --gpu-models で明示する。
            resident = DEFAULT_RESIDENT_MODELS if gpu_models is None else gpu_models
            print(
                f"[init] gpu models resident={resident} of {len(policies)} "
                f"({', '.join(policies)})",
                file=sys.stderr,
            )
            manager = ModelResidency(sequence, policies, resident=resident)

            # model の境界で起きる順序:
            #   前の model を保持したまま次の frame-zero へ腕・手を寄せる
            #   → HOLD に入る → そこで初めて前を解放し次を読む → 次の skill
            # HOLD は arm_sdk / Dex1 の target を生かしたまま待ち、読み込みが
            # 失敗・未完了のままでは完了しない (Issue #152)。
            for index, name in enumerate(sequence):
                if not name.startswith("hold_transition_"):
                    continue
                next_model = next(
                    (
                        candidate
                        for candidate in sequence[index + 1 :]
                        if candidate in policies
                    ),
                    None,
                )
                hold_skill = skill_registry.get(name)
                setter = getattr(hold_skill, "set_completion_gate", None)
                if next_model is None or not callable(setter):
                    raise RuntimeError(
                        f"model transition hold {name!r} has no loadable next policy"
                    )
                setter(
                    lambda model=next_model, owner=manager: owner.is_ready(model),
                    error_fn=(
                        lambda model=next_model, owner=manager: owner.error_for(model)
                    ),
                    description=f"model {next_model!r} loads",
                    # 読み込み中は時間制限なしで待つ (失敗したら error_fn で止める)。
                    timeout_s=None,
                )
            return manager

        single_stage_residency = (
            None
            if args.phase3_full
            else _build_residency(
                build_stage_skill_sequence(
                    first_phase3_stage,
                    is_start_stage=True,
                    include_hand=include_hand_in_head,
                    skip_model_transition_pairs=(
                        model_transition_skips
                        if phase3_active and not rule_based_pick_active
                        else frozenset()
                    ),
                )
            )
            if first_phase3_stage is not None
            else None
        )

        # 8) Orchestrator (Issue #64 auto-transition, Issue #75 arm/joint_state/wrist、
        #    Issue #81 Phase 3: setup skill が initial、SetupSkill.max_dwell_sec 経過で
        #    move_to_table へ chain auto-transition)
        orch = Orchestrator(
            perception,
            cleaner,
            dispatcher,
            initial_skill=initial_skill,
            joint_state_source=joint_state_source,
            dex1_state_source=dex1_state_source,
            wrist_left_source=wrist_left_source,
            wrist_right_source=wrist_right_source,
            head_perception_view=HEAD_PERCEPTION_VIEW,
            transitions=(
                phase_transitions
                if rule_based_pick_active
                else build_stage_transitions(
                    first_phase3_stage,
                    is_start_stage=True,
                    include_hand=include_hand_in_head,
                    skip_model_transition_pairs=model_transition_skips,
                )
                if args.phase3_full
                else phase_transitions
            ),
            enter_check=phase_enter_check,
            actuator_send_fn=actuator_send_fn,
            log_sink=log_sink,
            hard_timeout_by_skill=stage_hard_timeouts,
            timeout_action_by_skill=stage_timeout_actions,
            policy_filter=policy_filter,
            on_tick=_make_tick_hook(single_stage_residency),
        )

        # 8b) --stage 指定時: n_legs_completed を stage 番号から init。
        # stage 1 → n=0 (1本目)、stage 2 → n=1 (2本目)、... stage 5 → n=4 (flip)。
        # 未指定 (通常起動) は default 0 のまま。enter_conditions.py の
        # enter_pick_table_leg / enter_move_table_base / enter_flip_table が
        # 正しい branch を選ぶために必要。
        if first_phase3_stage is not None and first_phase3_stage >= 1:
            orch.state.n_legs_completed = first_phase3_stage - 1
            print(
                f"[init] stage={first_phase3_stage}: n_legs_completed initialized to "
                f"{orch.state.n_legs_completed}",
                file=sys.stderr,
            )

        # Phase 1/3 must prove that every learned-policy input is live before
        # arm_sdk ownership is acquired. The confirmation may take arbitrary
        # time, so strict Regular is checked *after* Enter as the final gate.
        if phase1_profile is not None or phase3_active:
            assert wrist_left_source is not None
            assert wrist_right_source is not None
            _wait_for_phase1_11_sources(
                head_source=source,
                wrist_left_source=wrist_left_source,
                wrist_right_source=wrist_right_source,
                joint_state_source=joint_state_source,
                dex1_state_source=dex1_state_source,
                timeout_s=args.camera_startup_timeout,
                phase_label=(
                    f"Phase {phase1_profile}"
                    if phase1_profile is not None
                    else "Phase 3 continuous"
                    if args.phase3_full
                    else "rule-based pick_table_leg"
                    if rule_based_pick_active
                    else f"Phase 3 stage {args.stage}"
                ),
            )

        if rule_based_pick_active:
            head_snapshot = source.get()
            joint_snapshot = joint_state_source.get()
            if head_snapshot is None or joint_snapshot is None:
                raise RuntimeError(
                    "rule-based pick observations disappeared after source preflight; "
                    "no robot command was sent"
                )
            _validate_rule_pick_geometry_snapshot(
                head_frame=head_snapshot,
                joint_state=joint_snapshot,
                perception=perception,
                grasp_provider=pick_table_leg_skill.grasp_provider,
            )

        if phase3_active:
            if not args.actuate:
                if rule_based_pick_active:
                    print(
                        "[preflight] rule-based pick config/4-camera/joint/Dex1 "
                        "validation passed; NO command sent (--actuate absent)",
                        file=sys.stderr,
                    )
                    return
                stages = (
                    range(args.phase3_start_stage, args.phase3_end_stage + 1)
                    if args.phase3_full
                    else (args.stage,)
                )
                validated: set[int] = set()
                for stage in stages:
                    for skill_name in STAGE_SKILL_SEQUENCES[stage]:
                        policy_resource = getattr(
                            skill_registry.get(skill_name), "_policy", None
                        )
                        if policy_resource is None or id(policy_resource) in validated:
                            continue
                        validate = getattr(
                            policy_resource, "validate_load_and_release", None
                        )
                        if callable(validate):
                            validate()
                            validated.add(id(policy_resource))
                print(
                    "[preflight] Phase 3 model/config/4-camera/"
                    "joint/Dex1 validation passed; NO command sent (--actuate absent)",
                    file=sys.stderr,
                )
                return
            # Load only the first learned expert before arm_sdk ownership.  Any
            # later expert is loaded after its predecessor is closed, so Phase
            # 3 never requires multiple 13 GiB GR00T checkpoints on the GPU.
            if not rule_based_pick_active:
                if single_stage_residency is not None:
                    single_stage_residency.prime()
            if first_phase3_stage == 0:
                walk_arm_pose = validate_lowered_walk_pose(
                    arm_actuator.read_arm_positions()
                )
                print(
                    "[preflight] lowered walk-arm envelope passed "
                    f"(shoulder_pitch={walk_arm_pose[[0, 7]].tolist()}, "
                    f"shoulder_roll={walk_arm_pose[[1, 8]].tolist()})",
                    file=sys.stderr,
                )
            ownership = (
                "arms + dedicated Dex1; waist/legs remain Regular-owned"
                if rule_based_pick_active
                else "continuous arms + waist + Dex1; Stage 0 holds measured waist/Dex1"
                if args.phase3_full
                else "arms only"
                if first_phase3_stage == 0
                else "arms + waist + Dex1"
                if first_phase3_stage in {1, 2, 3, 4}
                else "arms + Dex1; waist/legs remain Regular-owned"
            )
            if policy_start_gate_stage is not None:
                start_description = (
                    f"Stage {policy_start_gate_stage} collision-aware pre-motion; "
                    "the policy will wait for a second Enter after the initial "
                    "arm/hand pose is reached"
                )
            else:
                start_description = (
                    "collision-aware pre-motion then rule-based pick_table_leg"
                    if rule_based_pick_active
                    else f"Phase 3 stage {first_phase3_stage}"
                )
            input(
                "Harness / E-stop / workspace clearance confirmed. Enter starts "
                f"{start_description} ({ownership}); Ctrl+C cancels: "
            )
            if args.action_sink == "boundary":
                print(
                    "[preflight] official boundary state is live; robot mode and "
                    "whole-body ownership are enforced by the organizer adapter",
                    file=sys.stderr,
                )
            else:
                _require_strict_regular(
                    actuator,
                    phase_label=(
                        f"rule-based pick_table_leg ({ownership})"
                        if rule_based_pick_active
                        else f"Phase 3 stage {first_phase3_stage} ({ownership})"
                    ),
                )
            if args.use_real_hand:
                assert _real_hand_instance is not None
                hand_snapshot = dex1_state_source.get()
                if hand_snapshot is None:
                    raise RuntimeError(
                        "Phase 3 lost Dex1 state at final gate; no command sent"
                    )
                _real_hand_instance.prime_hold(hand_snapshot.position_rad)
                _real_hand_instance.start()
                print(
                    "[preflight] Dex1 publisher started from measured hold pose",
                    file=sys.stderr,
                )

        if phase1_profile is not None:
            walk_arm_pose = validate_lowered_walk_pose(
                arm_actuator.read_arm_positions()
            )
            print(
                "[preflight] lowered walk-arm envelope passed "
                f"(shoulder_pitch={walk_arm_pose[[0, 7]].tolist()}, "
                f"shoulder_roll={walk_arm_pose[[1, 8]].tolist()})",
                file=sys.stderr,
            )
            control_scope = {
                "1.11": "arms only",
                "1.12": "arms + waist",
                "1.13": "arms + waist + Dex1",
            }[phase1_profile]
            input(
                "Harness / E-stop / 1.5m clearance confirmed. Enter starts "
                f"Phase {phase1_profile} ({control_scope}); "
                "Ctrl+C cancels: "
            )
            _require_strict_regular(
                actuator,
                phase_label=(
                    "Phase 1.11 arm-only"
                    if phase1_profile == "1.11"
                    else "Phase 1.12 arms+waist"
                    if phase1_profile == "1.12"
                    else "Phase 1.13 arms+waist+Dex1"
                ),
            )

            if phase1_profile == "1.13":
                assert _real_hand_instance is not None
                hand_snapshot = dex1_state_source.get()
                if hand_snapshot is None:
                    raise RuntimeError(
                        "Phase 1.13 lost Dex1 state at final gate; no command sent"
                    )
                _real_hand_instance.prime_hold(hand_snapshot.position_rad)
                _real_hand_instance.start()
                print(
                    "[preflight] Dex1 publisher started from measured hold pose "
                    f"with {PHASE1_13_HAND_VELOCITY_RAD_S:g}rad/s slew limit",
                    file=sys.stderr,
                )

        # start() is deliberately after model/source construction and the
        # Phase 1.11 final gate.  No arm command can be published during a slow
        # model load, missing-camera failure, or operator confirmation wait.
        if args.action_sink == "boundary":
            # rt/arm_sdk の publisher を **立てない**。腕は (T,25) 経由で運営 WBC が
            # 動かすので、こちらから publish すると同じ関節を二重に動かす。
            arm_actuator.start()
            arm_actuator_started = True
            print(
                "[init] action sink = official boundary; command hold enabled; "
                "rt/arm_sdk publisher NOT created",
                file=sys.stderr,
            )
        else:
            arm_actuator.start()
            arm_actuator_started = True
            print(
                "[init] arm actuator started (250Hz rt/arm_sdk; "
                + (
                    "measured waist held, Phase 1.11 waist target=Mock)"
                    if phase1_profile == "1.11"
                    else "Phase 1.12 waist target=Real, hand target=Mock)"
                    if phase1_profile == "1.12"
                    else "Phase 1.13 waist/hand targets=Real)"
                    if phase1_profile == "1.13"
                    else "rule-based pick arm/Dex1 ownership)"
                    if rule_based_pick_active
                    else "Phase 3 continuous upper-body ownership)"
                    if args.phase3_full
                    else "standard profile)"
                ),
                file=sys.stderr,
            )

        # Default SIGINT handling raises KeyboardInterrupt.  Catching it below and
        # using this single finally block avoids duplicate stop commands.
        print("[run] starting orchestrator tick loop", file=sys.stderr)
        if args.phase3_full:
            for stage in range(args.phase3_start_stage, args.phase3_end_stage + 1):
                if stage == 5:
                    # Stages 1..4 may have dispatched a waist target.  Stage 5
                    # explicitly leaves waist/legs to Regular Mode, so stale
                    # targets must not survive the stage boundary.
                    arm_actuator.clear_waist_action()
                    print(
                        "[boundary] Stage 5 waist target cleared; "
                        "waist/legs remain Regular-owned",
                        file=sys.stderr,
                    )
                if stage != args.phase3_start_stage:
                    # Stop the previous Skill (which releases only its model),
                    # while arm/Dex1 publishers continue holding their latest
                    # targets.  Load the next expert before operator approval.
                    dispatcher.stop()
                    _wait_for_phase1_11_sources(
                        head_source=source,
                        wrist_left_source=wrist_left_source,
                        wrist_right_source=wrist_right_source,
                        joint_state_source=joint_state_source,
                        dex1_state_source=dex1_state_source,
                        timeout_s=args.camera_startup_timeout,
                        phase_label=f"Phase 3 stage {stage} boundary",
                    )
                    # Issue #141 (D2 #5、ユーザー決定 2026-09-14): stage の境界の Enter は
                    # 無くし、そのまま次の stage へ進む。途中から始めたいときは
                    # --phase3-start-stage で起動する。起動時の Enter は残る。
                    print(
                        f"[boundary] stage {stage - 1} finished; continuing to "
                        f"stage {stage} without an operator gate",
                        file=sys.stderr,
                    )

                stage_sequence = build_stage_skill_sequence(
                    stage,
                    is_start_stage=(stage == args.phase3_start_stage),
                    include_hand=include_hand_in_head,
                    skip_model_transition_pairs=model_transition_skips,
                )
                residency = _build_residency(stage_sequence)
                if residency is not None:
                    if residency.first_model_waits_for_switch_hold:
                        # 次の初期姿勢へ動いてから HOLD の中で読む (Codex #9)。
                        # 読み込みの間も制御 loop が保持と鮮度監視を続ける。
                        print(
                            f"[residency] stage {stage}: first model loads inside "
                            "its transition HOLD",
                            file=sys.stderr,
                        )
                    else:
                        # stage の最初の model は run_live の前に読む (1 tick 目が
                        # 読み込みを待たないように)。残りは裏で先読みする。
                        residency.prime()
                stage_transitions = build_stage_transitions(
                    stage,
                    is_start_stage=(stage == args.phase3_start_stage),
                    include_hand=include_hand_in_head,
                    skip_model_transition_pairs=model_transition_skips,
                )
                if policy_start_gate_stage == stage:
                    _insert_policy_start_gate_transition(
                        stage_transitions,
                        first_policy=STAGE_HEAD_SKILL[stage],
                    )
                stage_orch = Orchestrator(
                    perception,
                    cleaner,
                    dispatcher,
                    initial_skill=build_stage_skill_sequence(
                        stage,
                        is_start_stage=(stage == args.phase3_start_stage),
                        include_hand=include_hand_in_head,
                        skip_model_transition_pairs=model_transition_skips,
                    )[0],
                    joint_state_source=joint_state_source,
                    dex1_state_source=dex1_state_source,
                    wrist_left_source=wrist_left_source,
                    wrist_right_source=wrist_right_source,
                    head_perception_view=HEAD_PERCEPTION_VIEW,
                    transitions=stage_transitions,
                    enter_check=phase_enter_check,
                    actuator_send_fn=actuator_send_fn,
                    log_sink=log_sink,
                    hard_timeout_by_skill=stage_hard_timeouts,
                    timeout_action_by_skill=stage_timeout_actions,
                    policy_filter=policy_filter,
                    on_tick=_make_tick_hook(residency),
                )
                if stage >= 1:
                    stage_orch.state.n_legs_completed = stage - 1
                print(
                    f"[run] Phase 3 continuous stage {stage} started; "
                    "arm_sdk/Dex1 ownership retained",
                    file=sys.stderr,
                )
                stage_orch.run_live(
                    source,
                    hz=args.hz,
                    startup_timeout=args.camera_startup_timeout,
                    frame_timeout=args.frame_timeout,
                )
                if residency is not None:
                    residency.close()
                print(
                    f"[run] Phase 3 continuous stage {stage} boundary reached; "
                    "holding current arm/Dex1 targets",
                    file=sys.stderr,
                )
        else:
            orch.run_live(
                source,
                hz=args.hz,
                startup_timeout=args.camera_startup_timeout,
                frame_timeout=args.frame_timeout,
                stop_after_skill=(
                    "move_table_base" if phase1_profile is not None else None
                ),
                stop_after_s=(
                    PHASE1_11_VLA_SECONDS if phase1_profile is not None else None
                ),
                stop_on_skills=(
                    frozenset({"pick_table_leg"})
                    if phase1_profile is not None
                    else frozenset()
                ),
            )
    except LiveSourceSafetyError as e:
        print(f"[safety-stop] {e}", file=sys.stderr)
        if phase3_active and args.actuate and arm_actuator_started:
            # Do not turn a model-boundary failure into an immediate arm drop. Both
            # actuator publisher threads still own their last safe targets at
            # this point.  Keep them alive until the operator has secured the
            # table leg / arms, then let the one common finally block perform
            # the controlled arm_sdk release.
            if recorder is not None:
                recorder.write_event(
                    {
                        "event": "phase3_safety_hold",
                        "reason": str(e),
                        "instruction": "operator_ack_before_controlled_release",
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
            print(
                "[Phase 3 HOLD] Last safe arm and Dex1 targets remain active. "
                "Use E-stop immediately for dangerous motion. Otherwise the "
                "common return (open Dex1 -> lower the arms) runs before the "
                "controlled release.",
                file=sys.stderr,
            )
        raise SystemExit(2)
    except KeyboardInterrupt:
        pass
    finally:
        # 正常終了・Ctrl+C・安全停止のどれでも、手を開いて腕を下ろしてから解放する
        # (Issue #152)。weight はまだ 1 なので、ここは指令が効く。
        #
        # ロボットの停止・解放は **model / 録画の後始末より先に、独立した finally で**
        # 必ず通す (Codex #7)。以前は model 解放 (`queue.join()` は無期限) と録画の
        # 終了が解放の前にあり、そこで止まる・例外が出ると解放に届かなかった。
        try:
            shutdown_hand_actuator = (
                _shared_mock_hand_instance
                if args.action_sink == "boundary"
                else _real_hand_instance
            )
            if phase3_active and args.actuate and arm_actuator_started:
                # 腕を下ろす前に歩行を止める。boundary では毎 row が
                # `navigate_cmd=actuator.latest` を運ぶので、歩行中に落ちると
                # return の間 (最大 60s) 歩き続けていた。
                _stop_base_before_return(actuator)
                return_checker = None
                if args.action_sink == "boundary":
                    try:
                        return_checker = _boundary_convergence_checker(
                            STAGE_HEAD_SKILL[first_phase3_stage]
                            if first_phase3_stage is not None
                            else "pick_table_leg"
                        )
                    except Exception as checker_error:  # noqa: BLE001
                        print(
                            "[return] task-space checker unavailable: "
                            f"{checker_error!r}",
                            file=sys.stderr,
                        )
                _return_before_release(
                    skill_config=skill_cfg_raw,
                    arm_actuator=arm_actuator,
                    joint_state_source=joint_state_source,
                    hand_actuator=shutdown_hand_actuator,
                    dex1_state_source=dex1_state_source,
                    recorder=recorder,
                    measured_convergence_checker=return_checker,
                    open_only=(
                        args.action_sink == "boundary"
                        and not arm_actuator.has_motion_target
                    ),
                )
        finally:
            try:
                _release_robot_outputs(
                    actuator,
                    arm_actuator=arm_actuator,
                    arm_actuator_started=arm_actuator_started,
                    hand_actuator=shutdown_hand_actuator,
                )
            finally:
                if single_stage_residency is not None:
                    _bounded_cleanup(
                        "model residency close",
                        single_stage_residency.close,
                        timeout_s=SHUTDOWN_CLEANUP_TIMEOUT_S,
                    )
                if recorder is not None:
                    _bounded_cleanup(
                        "recorder close",
                        lambda: recorder.__exit__(None, None, None),
                        timeout_s=SHUTDOWN_CLEANUP_TIMEOUT_S,
                    )
                _close_host_resources(
                    log_sink,
                    source=source,
                    extra_sources=[
                        joint_state_source,
                        dex1_state_source,
                        *(
                            [rule_based_gripper]
                            if rule_based_gripper is not None
                            else []
                        ),
                        *(
                            [wrist_left_source, wrist_right_source]
                            if not args.no_wrist_cameras
                            else []
                        ),
                        *([_boundary_sink] if _boundary_sink is not None else []),
                    ],
                )


#: 解放の後の後始末 (model 解放・録画の終了・policy worker の停止) 1 件の上限 [s]。
#: 超えたら待たずに次へ進む (daemon thread に残す。process 終了で消える)。
SHUTDOWN_CLEANUP_TIMEOUT_S = 30.0


def _bounded_cleanup(label: str, fn, *, timeout_s: float) -> bool:
    """後始末 1 件を時間制限付きで走らせる。例外も時間切れも外へ出さない。

    Returns:
        時間内に例外なく終わったら True。
    """
    outcome: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - 後始末は止めない
            outcome["error"] = exc

    import threading

    worker = threading.Thread(target=_target, name=f"cleanup:{label}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        print(
            f"[shutdown] {label} did not finish within {timeout_s:g}s; continuing",
            file=sys.stderr,
        )
        return False
    if "error" in outcome:
        print(f"[shutdown] {label} failed: {outcome['error']!r}", file=sys.stderr)
        return False
    return True


def _release_robot_outputs(
    actuator: object,
    *,
    arm_actuator: Optional[object] = None,
    arm_actuator_started: bool = False,
    hand_actuator: Optional[object] = None,
) -> None:
    """walk zero → controlled arm release → arm/hand publisher 停止。例外は出さない。

    `_safe_shutdown` の前半。model / 録画の後始末とは独立に必ず通す (Codex #7)。
    """
    # If this entrypoint owns an arm actuator but never started it, the model /
    # camera / confirmation preflight failed before *any* physical command.
    # Preserve that guarantee: even a zero-velocity RPC is not sent in that
    # state.  Legacy callers without an arm actuator retain the conservative
    # stop command.
    if arm_actuator is None or arm_actuator_started:
        # Stop base motion first.  The arm publisher remains alive and holds its
        # last target during this RPC, so this does not create an arm unload gap.
        try:
            actuator.set_velocity(  # type: ignore[attr-defined]
                0.0, 0.0, 0.0, duration=1.0
            )
        except Exception as e:
            print(f"[shutdown] actuator stop failed: {e}", file=sys.stderr)
    if arm_actuator is not None and arm_actuator_started:
        if hasattr(arm_actuator, "controlled_release"):
            try:
                print(
                    "[shutdown] arm_sdk controlled release started (weight 1 -> 0)",
                    file=sys.stderr,
                )
                arm_actuator.controlled_release(duration_s=2.0)  # type: ignore[attr-defined]
                print(
                    "[shutdown] arm_sdk controlled release complete (weight=0)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[shutdown] arm controlled release failed: {e}", file=sys.stderr)
        try:
            arm_actuator.stop()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] arm actuator stop failed: {e}", file=sys.stderr)
    if hand_actuator is not None and hasattr(hand_actuator, "stop"):
        try:
            hand_actuator.stop()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] hand actuator stop failed: {e}", file=sys.stderr)


def _close_host_resources(
    log_sink: Optional[TextIO],
    *,
    source: Optional[object] = None,
    extra_sources: Optional[list] = None,
) -> None:
    """policy worker・reader・log を閉じる。各件に時間制限。例外は出さない。"""
    def _close_policies() -> None:
        for failure in _close_policy_resources():
            print(f"[shutdown] policy close failed: {failure}", file=sys.stderr)

    _bounded_cleanup(
        "policy close", _close_policies, timeout_s=SHUTDOWN_CLEANUP_TIMEOUT_S
    )
    if source is not None:
        try:
            source.close()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] camera reader close failed: {e}", file=sys.stderr)
    for extra in extra_sources or []:
        if extra is None:
            continue
        try:
            extra.close()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] extra source close failed: {e}", file=sys.stderr)
    if log_sink is not None:
        try:
            log_sink.close()
        except Exception:
            pass


def _safe_shutdown(
    actuator: object,
    log_sink: Optional[TextIO],
    *,
    source: Optional[object] = None,
    arm_actuator: Optional[object] = None,
    arm_actuator_started: bool = False,
    hand_actuator: Optional[object] = None,
    extra_sources: Optional[list] = None,
) -> None:
    """walk zero → controlled arm release → readers/log close. Idempotent.

    順序: walk zeroを即送信 → arm publishを継続したままweightを1→0へramp →
    arm/hand publisher停止 → source/log close。walk zeroを先にするためCtrl+Cが
    walk区間に入っても余分に2秒歩かず、armはrelease完了まで現在targetを保持する。
    handはPhase B-3のreal化時だけ有効 (Mockならno-op)。

    damp() は呼ばない: 常立ち前提 (Issue #64) では motor unload = 転倒リスク。立ち姿勢の
    release (Damp) は operator が Unitree リモコン等で明示的に実施する (scope 外)。
    HighStand も通常停止には不要なので送らない。

    Camera reader は process 終了へ任せず明示 close する。CycloneDDS listener の
    native receive thread を interpreter teardown 前に停止するため。JointStateSource
    は SDK ChannelFactory singleton 経由なので process 終了時にまとめて解放される。
    """
    _release_robot_outputs(
        actuator,
        arm_actuator=arm_actuator,
        arm_actuator_started=arm_actuator_started,
        hand_actuator=hand_actuator,
    )
    _close_host_resources(log_sink, source=source, extra_sources=extra_sources)


def _run_cli_and_exit() -> None:
    """Finalize Python-owned resources, then bypass native DDS destructors."""

    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            exit_code = 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        for failure in _close_policy_resources():
            print(f"[shutdown] policy close failed: {failure}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
    # CycloneDDS owns native recvMC threads which can survive closed Python
    # readers. main() has already completed controlled robot release, stopped
    # policy workers, and closed all sources, so native module finalizers add no
    # useful cleanup and may block the next Phase 3 child indefinitely.
    os._exit(exit_code)


if __name__ == "__main__":
    _run_cli_and_exit()
