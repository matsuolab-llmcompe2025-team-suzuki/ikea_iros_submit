"""評価と本番が共有する、推論の部品の組み立て (Issue #141 束 1-8)。

# なぜ要るか

実機評価は `evaluate/model_evaluation/runners/run_skill.py`、競技は
`inference/desktop/entrypoint.py` が動かす。部品のクラスは共有しているのに、
**組み立てだけを別々に書いていた**ため、09-09 の実機評価では次の差が残っていた。

| | 評価経路 | 本番経路 | 監査 |
|---|---|---|---|
| prompt | 渡していない (class 既定) | slot の `language_prompt` | INF-8 |
| 手首 FK の tool offset | 既定値 (3 dataset 平均) | skill ごとの実測値 | INF-9 |
| YOLO の重み | CLI 引数の local path | — | INF-10 |
| 位置の限界 | URDF − 0.03 rad で 19D hold | 渡していない (±100 rad) | INF-1 |
| warmup | GR00T / ACT / DP だけ | 無し | INF-13 |

同じ設定から同じ部品を作る関数をここに集め、両方の経路がこれを呼ぶ。

# ここに入らないもの

- actuator の生成と物理の安全ゲート (経路ごとに手順が違うので各 CLI に残す)
- tick と loop (`orchestrator.py`)、stage の列 (`entrypoint.py`)、CLI の引数
- 評価だけの手順 (操作者の Enter、収録、preview、wandb への upload)

# 失敗の形

設定の不足・不整合はすべて例外 (`ValueError` / `FileNotFoundError`) で、部品を
作る時点で止まる。`sys.exit` はここでは呼ばない (評価からも import する library
なので、test から呼べる形にする)。CLI 側が捕まえて今までと同じ文言で終了する。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from inference.desktop.lower_policy.initial_pose import initial_pose_from_config
from inference.desktop.lower_policy.policies.config_loader import (
    VariantEntry,
    load_yolo_settings,
    resolve_policy_class,
)

# カメラが止まったと判断する受信からの経過 [s] (評価経路と同じ値、4 台とも同じ扱い)。
CAMERA_STALE_TIMEOUT_S = 0.50
HEAD_VIEWS = ("packed", "left", "right")

# 19D 目標を丸ごと保持する位置の限界の余裕 [rad] (URDF の限界の内側)。
# 09-09 の評価経路と同じ値。教師の範囲の補正 (`skills/teacher_range.py`) が先に効くので、
# ここまで届くのは補正でも止まらなかったときだけ = 最後の安全網。
POSITION_MARGIN_RAD = 0.03
# MotionLimiter の dt の基準。tick は 30 Hz (D4)。
CONTROL_HZ = 30.0


@dataclass(frozen=True)
class SensorSources:
    """観測を作るのに要る source 一式。"""

    joint: Any
    dex1: Any
    head: Any
    wrist_left: Optional[Any]
    wrist_right: Optional[Any]

    def close(self) -> None:
        """全部の source を閉じる (閉じられるものだけ)。"""
        for source in (
            self.joint,
            self.dex1,
            self.head,
            self.wrist_left,
            self.wrist_right,
        ):
            close = getattr(source, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True)
class BuiltSkill:
    """`build_vla_skill` の結果。

    Attributes:
        skill: VlaSkill instance。
        policy: skill に入れた policy (DeferredPolicy のこともある)。呼び出し側が
            shutdown の対象として持つため返す。
        dispatch_waist: 腰を流すか (記録の metadata 用)。
    """

    skill: Any
    policy: Any
    dispatch_waist: bool


def skill_dispatch_waist(skill_config: dict, skill_name: str) -> bool:
    """`skills.<name>.dispatch_waist` を読む。未記載は error (fail-closed)。"""
    skills = skill_config.get("skills") or {}
    skill = skills.get(skill_name)
    if not isinstance(skill, dict) or "dispatch_waist" not in skill:
        raise ValueError(f"skills.{skill_name}.dispatch_waist must be explicit")
    value = skill["dispatch_waist"]
    if not isinstance(value, bool):
        raise ValueError(
            f"skills.{skill_name}.dispatch_waist must be boolean, got {value!r}"
        )
    return value


class FkFactory:
    """手首 FK を offset ごとに 1 個だけ作って使い回す (INF-9)。

    `skill_config.yaml` の `skills.<name>.wrist_tool_offset` が指定された skill は
    その offset の FK、未指定の skill は既定 (3 dataset 平均) の FK を使う。URDF の
    parse は 1 offset あたり ~100 ms かかるので、同じ offset の skill では作り直さない。

    Args:
        urdf_path: G1 の URDF。None なら `g1_urdf_fk.DEFAULT_URDF_PATH`。
            存在しなければ FK 無し (`for_skill` が None を返す = ee_state は 0)。
    """

    def __init__(self, urdf_path: Optional[Path] = None) -> None:
        from inference.desktop.perception.g1_urdf_fk import DEFAULT_URDF_PATH

        path = Path(urdf_path) if urdf_path is not None else Path(DEFAULT_URDF_PATH)
        self._urdf_path: Optional[Path] = path if path.exists() else None
        if self._urdf_path is None:
            print(
                f"[assembly] WARNING: G1 URDF not found ({path}). "
                "ee_state will be zeros — spatial signal disabled.",
                file=sys.stderr,
            )
        self._cache: dict[Any, Any] = {}

    @property
    def urdf_path(self) -> Optional[Path]:
        return self._urdf_path

    def for_skill(self, skill_config: dict, skill_name: str) -> Optional[Any]:
        """skill に対応する FK を返す (URDF が無ければ None)。"""
        if self._urdf_path is None:
            return None
        offset = _load_skill_wrist_tool_offset(skill_config, skill_name)
        key = (
            None if offset is None else (tuple(offset["left"]), tuple(offset["right"]))
        )
        if key in self._cache:
            return self._cache[key]
        from inference.desktop.perception.g1_urdf_fk import G1WristFK

        if offset is None:
            fk = G1WristFK.from_urdf(str(self._urdf_path))
        else:
            fk = G1WristFK.from_urdf(
                str(self._urdf_path),
                left_tool_offset=np.asarray(offset["left"], dtype=np.float64),
                right_tool_offset=np.asarray(offset["right"], dtype=np.float64),
            )
            print(
                f"[assembly] skill={skill_name} wrist_tool_offset: "
                f"left={offset['left']} right={offset['right']}",
                file=sys.stderr,
            )
        self._cache[key] = fk
        return fk


def _load_skill_wrist_tool_offset(
    skill_config: dict, skill_name: str
) -> Optional[dict]:
    """`skills.<name>.wrist_tool_offset` を読む。未指定なら None、形が違えば error。"""
    section = (skill_config.get("skills") or {}).get(skill_name)
    if not isinstance(section, dict):
        return None
    override = section.get("wrist_tool_offset")
    if override is None:
        return None
    if not isinstance(override, dict):
        raise ValueError(
            f"skills.{skill_name}.wrist_tool_offset must be a mapping with "
            f"'left' and 'right', got {type(override).__name__}"
        )
    try:
        left = [float(v) for v in override["left"]]
        right = [float(v) for v in override["right"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"skills.{skill_name}.wrist_tool_offset must have 'left'/'right' as "
            f"3-element float lists, got {override!r} ({exc})"
        ) from exc
    if len(left) != 3 or len(right) != 3:
        raise ValueError(
            f"skills.{skill_name}.wrist_tool_offset.left / .right must be length 3, "
            f"got len(left)={len(left)} len(right)={len(right)}"
        )
    return {"left": left, "right": right}


def load_policy(
    variant: VariantEntry, *, deferred: bool = False, label: Optional[str] = None
) -> Any:
    """slot から policy を作る。読み込んだ直後に必ず warmup して reset する (INF-13)。

    Args:
        variant: `load_policy_variant` の結果。
        deferred: True なら `DeferredPolicy` に包む (GPU に 1 model ずつ載せる経路)。
            遅れて読むときも同じ関数を通るので、warmup は必ず付いてくる。
        label: log 用の名前 (既定は variant 名)。

    Returns:
        policy (`deferred=True` なら DeferredPolicy)。
    """
    label = label or variant.name
    policy_cls = resolve_policy_class(variant.policy_type)
    if deferred:
        from inference.desktop.lower_policy.policies.deferred import DeferredPolicy

        return DeferredPolicy(
            policy_cls,
            label=label,
            loader=lambda: _load_and_warmup(policy_cls, variant.policy_config, label),
        )
    return _load_and_warmup(policy_cls, variant.policy_config, label)


def _load_and_warmup(policy_cls: type, policy_config: Any, label: str) -> Any:
    """ckpt を読んで 1 回だけ空の予測を走らせ、内部の状態を戻す。

    最初の tick で初期化 (CUDA の kernel、lazy な weight、chunk の buffer) が走ると、
    その tick だけ時間が伸びて開始姿勢から動き出すまでに間が空く。開始の Enter の
    前に済ませるため、読み込みと warmup を 1 つの関数にまとめる。
    """
    print(f"[assembly] loading policy: {label}", file=sys.stderr)
    policy = policy_cls.from_ckpt(policy_config)
    policy.warmup(n_iter=1)
    # reset は policy の約束の任意の部分 (Gr00tPolicyPickLegs は持たない)。
    # VlaSkill._on_start と同じで、持っているものだけ呼ぶ。
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    print(f"[assembly] policy ready: {label}", file=sys.stderr)
    return policy


def build_yolo_perception(
    policy_config_path: Path | str, *, device: Optional[str] = None
) -> Any:
    """`policy_config.yaml` の `yolo` から YOLO-OBB を作る (INF-10)。

    重みは `repo@revision` で指定し、HF の cache に取る。学習の overlay を焼いた
    重み・conf・imgsz と揃える (どれかが違うと検出そのものが変わる)。
    """
    from inference.desktop.perception.yolo_obb import (
        YoloObbPerception,
        resolve_yolo_ckpt_ref,
    )

    settings = load_yolo_settings(policy_config_path)
    weight_path = resolve_yolo_ckpt_ref(settings.ckpt_ref)
    print(
        f"[assembly] YOLO: {settings.ckpt_ref} conf={settings.conf} "
        f"imgsz={settings.imgsz}",
        file=sys.stderr,
    )
    perception = YoloObbPerception(
        weight_path, conf=settings.conf, imgsz=settings.imgsz, device=device
    )
    _warmup_yolo(perception, imgsz=settings.imgsz)
    return perception


def _warmup_yolo(perception: Any, *, imgsz: int) -> None:
    """ダミー画像で 1 回だけ推論を回し、CUDA の初期化を tick loop の外に出す。

    初回の predict は kernel の compile / graph capture で ~1.6s かかる (Thor 実測)。
    これを最初の tick でやると、その tick の間にカメラの受信時刻が古くなり、
    `camera_stale_roles` の許容 (CAMERA_STALE_TIMEOUT_S = 0.5s) を超えて
    `LiveSourceSafetyError` で止まる。policy 側の `_load_and_warmup` と同じ理由で、
    走り出す前に済ませる。

    失敗しても致命ではない (次の tick で普通に走る) のでログだけ残して続ける。
    """
    import time as _time

    import numpy as _np

    try:
        dummy = _np.zeros((imgsz, imgsz, 3), dtype=_np.uint8)
        t0 = _time.monotonic()
        perception.predict(dummy)
        print(
            f"[assembly] YOLO warmup: {(_time.monotonic() - t0) * 1e3:.0f} ms",
            file=sys.stderr,
        )
    except Exception as e:  # noqa: BLE001 - warmup は best-effort
        print(f"[assembly] YOLO warmup skipped: {e!r}", file=sys.stderr)


def build_sensor_sources(
    *,
    head_topic: str,
    wrist_left_topic: Optional[str] = None,
    wrist_right_topic: Optional[str] = None,
    joint_source: str = "lowstate",
    joint_state_topic: str = "/joint_states",
    head_source: str = "ros2",
    zmq_endpoint: str = "tcp://192.168.123.164:5555",
    synthetic_hand_initial_rad: Optional[tuple[float, float]] = None,
) -> SensorSources:
    """観測に要る source を作る。`ChannelFactoryInitialize` 済みが前提。

    Args:
        head_topic: head camera の topic (packed stereo)。`head_source="ros2"` のとき使う。
        wrist_left_topic / wrist_right_topic: None なら手首カメラ無しで組む。
        joint_source: 関節角の取り方 (INF-18)。
            - "lowstate" (既定): Desktop が `rt/lowstate` を直接受ける。新しい値が 500 Hz で来る
            - "joint_states": Orin の bridge 経由。同じ値の送り直しで新しい値は 11〜13 Hz
        joint_state_topic: `joint_source="joint_states"` のときの topic。
        head_source: head camera の取り方。
            - "ros2" (既定): lab 構成。Orin が ROS2 CompressedImage を publish
            - "zmq": 大会会場構成。運営 bridge (`real_orin_cameras.py`) の ZeroMQ を
              直接受ける。ROS2 へ bridge すると JPEG を二度通り学習時と画が変わるため
        zmq_endpoint: `head_source="zmq"` のときの endpoint。
        synthetic_hand_initial_rad: 指定すると Dex1 の state を **合成**する。
            会場のリグは hand state を配信しないため (boundary/states.py が
            "HAND STATE IS USUALLY ABSENT ... synthesize whatever your model expects"
            と明記)、実 topic を待つ代わりに「開始 skill の frame-0 値で seed し、
            以降は hand 指令をエコーする」source を使う。`None` (既定) なら従来どおり
            `rt/dex1/*` を購読する。
    """
    from inference.desktop.perception.frame_source import Ros2FrameSource

    if joint_source == "lowstate":
        from inference.desktop.perception.lowstate_joint_source import (
            LowStateJointSource,
        )

        joint: Any = LowStateJointSource()
    elif joint_source == "joint_states":
        from inference.desktop.perception.joint_state_source import JointStateSource

        joint = JointStateSource(topic=joint_state_topic)
    else:
        raise ValueError(
            f"joint_source must be 'lowstate' or 'joint_states', got {joint_source!r}"
        )
    if head_source == "zmq":
        from inference.desktop.perception.frame_source import ZmqFrameSource

        # 会場の bridge は 1 endpoint に全カメラを載せる (ego_view / ego_view_left /
        # ego_view_right / left_wrist / right_wrist)。head は左右を hconcat した
        # packed、手首は単一キーをそのまま使う。
        head: Any = ZmqFrameSource(zmq_endpoint, stereo_view="packed")
        wrist_left: Any = (
            None
            if wrist_left_topic is None
            else ZmqFrameSource(
                zmq_endpoint, stereo_view="single", image_key="left_wrist"
            )
        )
        wrist_right: Any = (
            None
            if wrist_right_topic is None
            else ZmqFrameSource(
                zmq_endpoint, stereo_view="single", image_key="right_wrist"
            )
        )
    elif head_source == "ros2":
        head = Ros2FrameSource(topic=head_topic, stereo_view="packed")
        wrist_left = (
            None
            if wrist_left_topic is None
            else Ros2FrameSource(topic=wrist_left_topic)
        )
        wrist_right = (
            None
            if wrist_right_topic is None
            else Ros2FrameSource(topic=wrist_right_topic)
        )
    else:
        raise ValueError(f"head_source must be 'ros2' or 'zmq', got {head_source!r}")
    if synthetic_hand_initial_rad is None:
        from inference.desktop.perception.dex1_state_source import Dex1StateSource

        dex1: Any = Dex1StateSource()
    else:
        from inference.desktop.perception.dex1_state_source import (
            SyntheticDex1StateSource,
        )

        dex1 = SyntheticDex1StateSource(synthetic_hand_initial_rad)
    return SensorSources(
        joint=joint,
        dex1=dex1,
        head=head,
        wrist_left=wrist_left,
        wrist_right=wrist_right,
    )


def _build_motion_limiter(
    skill_config: dict, skill_name: str, *, dispatch_waist: bool
) -> Any:
    """速度・加速度の包絡 + 位置の限界 (URDF − margin) の 19D limiter を作る。

    位置の限界を出た目標は 19D まるごと捨てて前の目標を保持する。これは最後の
    安全網で、通常は教師の範囲の補正 (`teacher_range.py`) が先に効く。
    """
    from inference.desktop.lower_policy.actuators.g1_arm_sdk import (
        G1_ARM_POSITION_LOWER_RAD,
        G1_ARM_POSITION_UPPER_RAD,
        G1_WAIST_POSITION_LOWER_RAD,
        G1_WAIST_POSITION_UPPER_RAD,
    )
    from inference.desktop.lower_policy.actuators.hand import (
        HAND_GRIP_MAX,
        HAND_GRIP_MIN,
    )
    from inference.desktop.lower_policy.motion_limits import (
        load_motion_limits_for_skill,
    )
    from inference.desktop.lower_policy.skills.motion_limiter import MotionLimiter
    from inference.desktop.lower_policy.skills.vla_skill import (
        ACTION_DIM_TOTAL,
        ARMS_SLICE,
        HAND_SLICE,
        WAIST_SLICE,
    )

    limits = load_motion_limits_for_skill(skill_config, skill_name)
    lower = np.concatenate(
        (
            G1_WAIST_POSITION_LOWER_RAD + POSITION_MARGIN_RAD,
            G1_ARM_POSITION_LOWER_RAD + POSITION_MARGIN_RAD,
            np.full(2, HAND_GRIP_MIN, dtype=np.float64),
        )
    )
    upper = np.concatenate(
        (
            G1_WAIST_POSITION_UPPER_RAD - POSITION_MARGIN_RAD,
            G1_ARM_POSITION_UPPER_RAD - POSITION_MARGIN_RAD,
            np.full(2, HAND_GRIP_MAX, dtype=np.float64),
        )
    )
    return MotionLimiter(
        dim=ACTION_DIM_TOTAL,
        arm_slice=slice(WAIST_SLICE.start, ARMS_SLICE.stop),
        hand_slice=HAND_SLICE,
        arm_velocity=limits.arm_velocity_rad_s,
        arm_acceleration=limits.arm_acceleration_rad_s2,
        hand_velocity=limits.hand_velocity_rad_s,
        hand_acceleration=limits.hand_acceleration_rad_s2,
        control_hz=CONTROL_HZ,
        lower=lower,
        upper=upper,
        reject_position_slice=(
            slice(WAIST_SLICE.start, ARMS_SLICE.stop) if dispatch_waist else ARMS_SLICE
        ),
    )


def build_vla_skill(
    *,
    skill_name: str,
    vla_skill_cls: type,
    variant: VariantEntry,
    skill_config: dict,
    waist_actuator: Any,
    hand_actuator: Any,
    fk_factory: FkFactory,
    deferred: bool = False,
    policy: Optional[Any] = None,
) -> BuiltSkill:
    """slot と skill_config から VlaSkill を 1 つ組み立てる。

    入るもの: policy (warmup 済) / MotionLimiter (速度・加速度 + 位置の限界) /
    手首 FK (skill ごとの offset) / 教師の範囲の補正 / 天板の回転の monitor /
    prompt / skill_id。

    Args:
        skill_name: `skill_config.yaml` の `skills.<name>` の key。
        vla_skill_cls: `skills/vla_skill.py` の subclass。
        variant: `load_policy_variant` の結果 (slot)。
        skill_config: `yaml.safe_load(skill_config.yaml)` の結果。
        waist_actuator / hand_actuator: 呼び出し側が作った actuator。
        fk_factory: 手首 FK の生成器 (offset ごとに使い回す)。
        deferred: policy を skill の開始まで読まない (GPU に 1 model ずつ)。
        policy: 既に `load_policy` で読んだ policy。None ならここで読む。評価経路は
            腕の所有権を取る前に model を読む (初回 forward の遅れを先に消化する) ため、
            読み込みだけ先に済ませて渡す。
    """
    from inference.desktop.lower_policy.skills.rotate_progress import (
        load_progress_monitor_for_skill,
    )
    from inference.desktop.lower_policy.skills.rotate_retry import (
        load_retry_controller_for_skill,
    )
    from inference.desktop.lower_policy.skills.teacher_range import (
        load_teacher_joint_range_for_skill,
    )
    from inference.desktop.lower_policy.skills.z_ceiling import LeftHandZCeiling

    dispatch_waist = skill_dispatch_waist(skill_config, skill_name)
    if policy is None:
        policy = load_policy(
            variant, deferred=deferred, label=f"{skill_name}:{variant.name}"
        )
    teacher_range = load_teacher_joint_range_for_skill(skill_config, skill_name)
    progress_monitor = load_progress_monitor_for_skill(skill_config, skill_name)
    fk = fk_factory.for_skill(skill_config, skill_name)

    # Issue #137 Phase 3/4: 空振りの retry と左手先 z の片側天井。`retry` の節が
    # 無ければ両方 None = 従来どおり (天井も作らない)。
    retry_controller, retry_options = load_retry_controller_for_skill(
        skill_config, skill_name
    )
    z_ceiling = None
    if retry_controller is not None:
        deepest = min(
            (r for r in retry_controller.rungs if r is not None), default=None
        )
        if deepest is None:
            raise ValueError(
                f"skills.{skill_name}.retry.rungs_mm に天井が無い。"
                "天井の無い retry は下げる先が無い"
            )
        if fk is None:
            raise ValueError(
                f"skills.{skill_name}.retry は手首の FK が要るが、FK を作れていない"
            )
        z_ceiling = LeftHandZCeiling(fk, hard_floor_m=deepest)
        retry_options["initial_arm_14"] = initial_pose_from_config(
            skill_config, skill_name
        ).arm_position_rad.tolist()
        print(
            f"[assembly] {skill_name}: retry rungs={retry_controller.rungs} "
            f"mode={retry_options['return_mode'].value}",
            file=sys.stderr,
        )

    skill = vla_skill_cls(
        policy=policy,
        waist_actuator=waist_actuator,
        hand_actuator=hand_actuator,
        fk=fk,
        dispatch_waist=dispatch_waist,
        motion_limiter=_build_motion_limiter(
            skill_config, skill_name, dispatch_waist=dispatch_waist
        ),
        language_override=variant.policy_config.language_prompt,
        skill_id_override=variant.policy_config.skill_id,
        teacher_range=teacher_range,
        progress_monitor=progress_monitor,
        z_ceiling=z_ceiling,
        retry_controller=retry_controller,
        retry_options=retry_options,
    )
    print(
        f"[assembly] {skill_name} = {vla_skill_cls.__name__} (variant={variant.name}, "
        f"type={variant.policy_type}, mode={variant.policy_config.mode}, "
        f"ckpt={variant.policy_config.ckpt_ref})",
        file=sys.stderr,
    )
    print(
        f"[assembly] {skill_name}: dispatch_waist={dispatch_waist} "
        f"skill_id={'class' if variant.policy_config.skill_id is None else variant.policy_config.skill_id} "
        f"prompt={variant.policy_config.language_prompt!r} "
        f"teacher_range={'on' if teacher_range is not None else 'off'} "
        f"progress_monitor={'on' if progress_monitor is not None else 'off'}",
        file=sys.stderr,
    )
    return BuiltSkill(skill=skill, policy=policy, dispatch_waist=dispatch_waist)


# =========================================================================
# 頭の手順 (Issue #141 D2)
# =========================================================================


def head_procedure_skill_names(
    skill_name: str, *, include_hand: bool = True, grasp: bool = False
) -> list[str]:
    """頭の手順の skill 名。実体を作らずに列だけ組む場所 (本番の stage) 用。

    `build_head_procedure` はこの関数から名前を取るので、両者は必ず一致する。
    """
    names: list[str] = []
    if include_hand:
        names.append(f"hand_open_{skill_name}")
    names.append(f"arm_pre_motion_for_{skill_name}")
    if include_hand:
        names.append(f"hand_grasp_{skill_name}" if grasp else f"hand_pose_{skill_name}")
    names.append(f"hold_pose_for_{skill_name}")
    return names


def build_head_procedure(
    *,
    skill_config: dict,
    skill_name: str,
    initial_pose: Any,
    hand_actuator: Any,
    hold_sec: float,
    include_hand: bool,
) -> list:
    """頭の手順の skill 列を作る (Issue #141 D2)。評価と本番の両方がこれを使う。

    手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持。
    物を持って始まる skill (`requires_separate_hand_initialization`) は、最後の手の
    動きを「掴む幅へ (接触で止まってよい)」にする。

    Args:
        skill_config: `yaml.safe_load(skill_config.yaml)` の結果。
        skill_name: 開始姿勢と開度の出どころになる学習 skill。
        initial_pose: その skill の `SkillInitialPose`。
        hand_actuator: Dex1 の actuator。
        hold_sec: 最後に開始姿勢で保持する秒数。
        include_hand: Dex1 を実際に動かすか (`--use-real-hand`)。False なら手の skill を
            入れない (mock の手は測定値が動かないので、到達を待てない)。
    """
    from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
        CollisionAwareArmPreMotionSkill,
    )
    from inference.desktop.lower_policy.skills.hand_pre_motion import HandPreMotionSkill
    from inference.desktop.lower_policy.skills.hold_pose import HoldPoseSkill

    grasp = bool(initial_pose.requires_separate_hand_initialization)
    names = head_procedure_skill_names(
        skill_name, include_hand=include_hand, grasp=grasp
    )
    skills: list = []
    if include_hand:
        skills.append(
            HandPreMotionSkill.from_config(
                skill_config,
                skill_name,
                target="open",
                hand_actuator=hand_actuator,
                name=names[0],
            )
        )
    arm_settings = dict(skill_config.get("arm_pre_motion") or {})
    unknown = sorted(
        set(arm_settings)
        - {
            "velocity_limit_rad_s",
            "acceleration_limit_rad_s2",
            "measured_tolerance_rad",
            "stage_timeout_s",
        }
    )
    if unknown:
        raise ValueError(f"arm_pre_motion has unknown keys: {unknown}")
    skills.append(
        CollisionAwareArmPreMotionSkill(
            tuple(initial_pose.arm_position_rad.tolist()),
            skill_name=names[1] if include_hand else names[0],
            **arm_settings,
        )
    )
    if include_hand:
        skills.append(
            HandPreMotionSkill.from_config(
                skill_config,
                skill_name,
                target="grasp" if grasp else "pose",
                hand_actuator=hand_actuator,
                name=names[2],
            )
        )
    skills.append(HoldPoseSkill(hold_sec, name=names[-1]))
    built = [skill.name for skill in skills]
    if built != names:
        raise RuntimeError(f"head procedure name mismatch: {built} != {names}")
    return skills


# =========================================================================
# 観測の組み立て (評価と本番で同じ 1 つの関数、Issue #141 D3)
# =========================================================================


def select_head_view(rgb: Any, view: str) -> Any:
    """packed stereo の head 画像から、検出に渡す側の目を選ぶ。

    policy に渡す `head_rgb` は packed のまま保つ (学習は左右両方を見る)。
    """
    if view not in HEAD_VIEWS:
        raise ValueError(f"head view must be one of {HEAD_VIEWS}, got {view!r}")
    if view == "packed":
        return rgb
    width = int(np.asarray(rgb).shape[1])
    if width < 2 or width % 2 != 0:
        raise ValueError(
            f"packed stereo image must have a positive even width, got {width}"
        )
    split = width // 2
    return rgb[:, :split] if view == "left" else rgb[:, split:]


def build_observation(
    *,
    head: Any,
    detections: Optional[list] = None,
    joint_state_source: Any = None,
    dex1_state_source: Any = None,
    wrist_left_source: Any = None,
    wrist_right_source: Any = None,
) -> dict:
    """1 tick 分の観測を組む (評価経路の `_collect_obs` をここへ移したもの)。

    `obs["t"]` はホストの monotonic の ns。カメラの header.stamp は
    `_camera_generations` に、受信時刻は `_camera_received_monotonic_ns` に入れる
    (止まったカメラの判定は受信時刻で行う。画像は latest-only なので、USB が抜けても
    最後の画像が残り続ける)。

    Args:
        head: head camera の FrameData (packed stereo)。未受信なら None。
        detections: policy に渡す検出 (`obs["cleaned"]`)。None = 検出が無い tick。
        *_source: `get()` を持つ source。None なら観測にその欄を作らない。
    """
    obs: dict = {"t": time.monotonic_ns(), "cleaned": detections}
    camera_frames: dict = {}
    camera_generations: dict = {}
    camera_received_ns: dict = {}

    if head is not None:
        head_rgb = np.asarray(getattr(head, "rgb", head))
        obs["head_rgb"] = head_rgb
        # 受信時刻は「frame が届いたか」なので、左右に割れるかとは別に記録する
        # (割れない形の画像は別の異常で、止まったカメラとは区別する)。
        generation = int(getattr(head, "t", obs["t"]))
        received = int(getattr(head, "received_monotonic_ns", None) or obs["t"])
        for role in ("head_left", "head_right"):
            camera_generations[role] = generation
            camera_received_ns[role] = received
        if head_rgb.ndim == 3 and head_rgb.shape[1] % 2 == 0:
            split = head_rgb.shape[1] // 2
            camera_frames["head_left"] = head_rgb[:, :split]
            camera_frames["head_right"] = head_rgb[:, split:]

    if joint_state_source is not None:
        obs["joint_state"] = joint_state_source.get()
    if dex1_state_source is not None:
        obs["hand_state"] = dex1_state_source.get()
    for key, role, source in (
        ("wrist_left_rgb", "left_wrist", wrist_left_source),
        ("wrist_right_rgb", "right_wrist", wrist_right_source),
    ):
        if source is None:
            continue
        frame = source.get()
        obs[key] = frame
        if frame is None:
            continue
        camera_frames[role] = np.asarray(getattr(frame, "rgb", frame))
        camera_generations[role] = int(getattr(frame, "t", obs["t"]))
        camera_received_ns[role] = int(
            getattr(frame, "received_monotonic_ns", None) or obs["t"]
        )

    obs["_camera_frames"] = camera_frames
    obs["_camera_generations"] = camera_generations
    obs["_camera_received_monotonic_ns"] = camera_received_ns
    return obs


def required_camera_roles(*, require_wrist: bool) -> tuple[str, ...]:
    """止まっていないことを確かめるカメラの役割。"""
    roles = ("head_left", "head_right")
    return roles + (("left_wrist", "right_wrist") if require_wrist else ())


def camera_stale_roles(
    obs: dict,
    *,
    require_wrist: bool,
    max_age_s: float = CAMERA_STALE_TIMEOUT_S,
    now_ns: Optional[int] = None,
) -> dict[str, float]:
    """止まっている / 来ていないカメラと、その古さ [s] を返す。

    画像は latest-only なので、USB が抜けた後も最後の画像が残る。判定は必ず DDS の
    受信時刻で行う (画像があることを根拠にしない)。
    """
    now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    received = obs.get("_camera_received_monotonic_ns")
    received = received if isinstance(received, dict) else {}
    stale: dict[str, float] = {}
    for role in required_camera_roles(require_wrist=require_wrist):
        timestamp = received.get(role)
        if timestamp is None or int(timestamp) <= 0:
            stale[role] = float("inf")
            continue
        age_s = max(0.0, (now_ns - int(timestamp)) / 1e9)
        if age_s > max_age_s:
            stale[role] = age_s
    return stale
