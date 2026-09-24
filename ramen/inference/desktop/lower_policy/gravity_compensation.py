"""Official G1-29 arm gravity feed-forward for physical arm targets.

Ported from issue-70-flip-table-data-augmentation branch (Phase 5、Issue #128)。
issue-70 版は xr_teleoperate wrapper が生成する `g1_29_model_cache.pkl` pickle
を検出して使う設計 (実機担当が xr_teleoperate をセットアップ済み前提)。

**我々の変更 (Self-Contained path)**: 我々 repo に既にある G1 URDF
(`inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf`)
から pinocchio reduced model を build する経路を追加。xr_teleoperate 外部依存
を排除、cache 準備タスクなしで動く。

# 何をやる

G1 arm 14 DOF (waist/leg を lock した reduced model) の RNEA torque を per-tick
計算し、Actuator の lowcmd.motor_cmd[i].tau に埋める feed-forward。

**効果**:
- Position 制御単体 = 静的 error 残る (motor が gravity と kp×error で平衡、drooping)
- Gravity FF あり = motor に「重力補償 tau を先に、その上で kp×error」 → error≈0
- 特に **insert (SOTA π0.5 = 52.5% 成功率の最難関 skill)** で cm 単位の droop
  差が命取りになるため critical。

# 使い方

    # Self-Contained (推奨、xr_teleoperate 不要):
    gravity = OfficialG1ArmGravityCompensator.from_urdf(
        "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/"
        "g1_29dof_mode_15_with_dex1_1.urdf"
    )
    # or default (auto-detect URDF path in repo tree):
    gravity = OfficialG1ArmGravityCompensator.from_default_urdf()

    # per-tick (Actuator の publish loop 250Hz):
    torque = gravity.torque_nm(current_arm_q_14d)  # (14,) Nm
    lowcmd.motor_cmd[i].tau = torque[i] for arm joints

    # Legacy (xr_teleoperate pickle cache 使用時):
    gravity = OfficialG1ArmGravityCompensator()   # sys.path 検索
    # or 明示 path:
    gravity = OfficialG1ArmGravityCompensator(cache_path=Path("..."))

# 依存

- `pinocchio ==3.1.0` (root runtime env に既存)
- URDF file (repo 内、`inference/orin/...` 経路で参照)
- xr_teleoperate pickle cache は optional (legacy path)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np


EXPECTED_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# Fail-closed validation bound (not a replacement for the official RNEA output).
# All three pre-motion poses are below 7 Nm with the pinned model.
MAX_ABS_GRAVITY_TORQUE_NM = 15.0

# Locked joint names for URDF reduction (arm 14 joint 以外を全部 lock)。
# Verify (Phase 5 実装時): 我々 URDF `g1_29dof_mode_15_with_dex1_1.urdf` を
# pinocchio に load すると full nq=33 (12 leg + 3 waist + 14 arm + 4 dex1 finger)、
# arm 14 に reduce するには legs (12) + waist (3) + dex1 finger (4) = 19 joint 要 lock。
# Dex1-1 の gravity 影響は arm gravity RNEA に含めなくてよい (Dex1 独自 SDK 制御、
# arm actuator は grip 動作を tau で補償しない)。
_LOCKED_JOINT_NAMES = (
    # Legs (12): hip x3 + knee x1 + ankle x2 per side
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3): yaw + roll + pitch
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Dex1-1 finger joints (4): arm gravity RNEA に含めない (別 SDK 制御)
    "left_dex1_finger_joint_1", "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1", "right_dex1_finger_joint_2",
)

# Default URDF path (repo 相対)。g1_urdf_fk.py の DEFAULT_URDF_PATH と同一 asset。
_DEFAULT_URDF_RELATIVE_PATH = (
    "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/"
    "g1_29dof_mode_15_with_dex1_1.urdf"
)


def find_pinned_g1_model_cache() -> Path:
    """Legacy: xr_teleoperate wrapper が生成した pickle cache を sys.path から検出。

    issue-70 準拠経路。実機担当が xr_teleoperate セットアップ済みで、その
    teleop/ dir が sys.path に入ってる場合のみ動く。exactly 1 個見つからないと
    fail (「複数」or「無し」で silently 誤ったのを使う risk 回避)。
    """
    candidates: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry).resolve()
        candidates.extend(
            (
                root / "teleop/g1_29_model_cache.pkl",
                root / "g1_29_model_cache.pkl",
            )
        )
    existing = tuple(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise RuntimeError(
            "expected exactly one pinned xr_teleoperate G1-29 model cache, "
            f"found {[str(path) for path in existing]}"
        )
    return existing[0]


def find_default_urdf_path() -> Path:
    """Repo root から _DEFAULT_URDF_RELATIVE_PATH を探索。

    実行元 dir に依存しないよう、本 module 位置から repo root を推定
    (lower_policy/ の 3 上位)、そこから相対 path で URDF 検出。
    """
    # __file__ = inference/desktop/lower_policy/gravity_compensation.py
    # repo_root = 3 上位
    module_file = Path(__file__).resolve()
    repo_root = module_file.parents[3]
    urdf_path = repo_root / _DEFAULT_URDF_RELATIVE_PATH
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"G1 URDF not found at expected path: {urdf_path}. "
            f"Repo root inferred as {repo_root}."
        )
    return urdf_path


def build_reduced_arm_model_from_urdf(urdf_path: Path | str):
    """URDF → pinocchio 14 DOF arm reduced model + data。

    - Full URDF (29 DOF + floating base + hands) を parse
    - leg 12 + waist 3 joint を lock (fixed) して reduce
    - 残る 14 joint (arm) の RNEA モデルを返す

    Returns:
        (model, data) tuple。model.nq == model.nv == 14、model.names[1:] は
        EXPECTED_ARM_JOINT_NAMES と一致。

    Raises:
        RuntimeError: reduction 後 nq != 14 or joint order 不一致。
    """
    # lazy: env-isolated dependency (pinocchio は runtime env のみ)
    import pinocchio as pin

    path = Path(urdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")

    # Full model build (floating base 有り URDF なら pinocchio が自動 handle、
    # ここでは fixed base として build = arm gravity 計算のみ目的)
    full_model = pin.buildModelFromUrdf(str(path))

    # Lock joint IDs を EXPECTED locked names から解決 (URDF に無い名前は skip)。
    locked_joint_ids = []
    missing = []
    for name in _LOCKED_JOINT_NAMES:
        if full_model.existJointName(name):
            locked_joint_ids.append(full_model.getJointId(name))
        else:
            missing.append(name)
    if missing:
        # Legs/waist が URDF に無い = arm-only URDF (=既に reduced)、そのまま使う
        # else fallback (見つからないなら reduce 不要)
        pass

    # Reduce (locked joint を fixed とみなす、q_ref = 0)
    if locked_joint_ids:
        q_ref = np.zeros(full_model.nq)
        reduced = pin.buildReducedModel(full_model, locked_joint_ids, q_ref)
    else:
        reduced = full_model

    if reduced.nq != 14 or reduced.nv != 14:
        raise RuntimeError(
            f"reduced model must have nq=nv=14, got nq={reduced.nq} nv={reduced.nv}"
        )
    names = tuple(str(name) for name in reduced.names[1:])
    if names != EXPECTED_ARM_JOINT_NAMES:
        raise RuntimeError(
            f"reduced-model joint order mismatch:\n"
            f"  expected: {EXPECTED_ARM_JOINT_NAMES}\n"
            f"  got:      {names}"
        )
    return reduced, reduced.createData()


class OfficialG1ArmGravityCompensator:
    """Compute the same Pinocchio RNEA term returned by official G1 arm IK.

    Construction paths:
      - `__init__(cache_path=None)`: **legacy** = pickle cache from xr_teleoperate
        wrapper (issue-70 verbatim behavior). `sys.path` を検索、明示 path 可。
      - `from_urdf(urdf_path)`: **Self-Contained** = 我々 repo URDF から build。
        xr_teleoperate 不要、実機担当 setup 不要で動く。
      - `from_default_urdf()`: 上記の default URDF path 版。

    Instance attributes:
        cache_path: pickle cache path (legacy path only、`from_urdf` 経路では None)
        cache_sha256: pickle SHA256 (legacy) or URDF SHA256 (self-contained)
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        """Legacy pickle cache 経路 (issue-70 verbatim)。

        Args:
            cache_path: pickle path 明示、None なら `find_pinned_g1_model_cache()`。
        """
        import pinocchio as pin

        self._pin = pin
        self.cache_path = (
            find_pinned_g1_model_cache()
            if cache_path is None
            else Path(cache_path).resolve()
        )
        payload = self.cache_path.read_bytes()
        cache = pickle.loads(payload)  # noqa: S301 - pinned official local artifact
        model = cache.get("reduced_model")
        if model is None or model.nq != 14 or model.nv != 14:
            raise RuntimeError("pinned G1 reduced model must have nq=nv=14")
        names = tuple(str(name) for name in model.names[1:])
        if names != EXPECTED_ARM_JOINT_NAMES:
            raise RuntimeError(f"unexpected G1 reduced-model joint order: {names}")
        self._model = model
        self._data = model.createData()
        self.cache_sha256 = hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_urdf(cls, urdf_path: Path | str) -> "OfficialG1ArmGravityCompensator":
        """Self-Contained: URDF から reduced model を build して instance 生成。

        pickle cache を使わず、pinocchio で URDF → reduced model を直接構築。
        xr_teleoperate 外部依存を排除、我々 repo だけで完結。

        Args:
            urdf_path: G1-29 URDF file path。arm 14 joint + leg/waist 15 joint を
                含む full URDF (reduce 対象)、または既に arm-only な reduced URDF
                (reduce skip)。
        """
        import pinocchio as pin

        instance = cls.__new__(cls)
        instance._pin = pin
        instance.cache_path = None  # not from pickle
        model, data = build_reduced_arm_model_from_urdf(urdf_path)
        instance._model = model
        instance._data = data
        # URDF SHA256 を parity 用に保持 (cache_sha256 field 意味を「provenance
        # digest」として一般化、pickle と URDF で同 field を tag 目的で使う)
        urdf_bytes = Path(urdf_path).resolve().read_bytes()
        instance.cache_sha256 = hashlib.sha256(urdf_bytes).hexdigest()
        return instance

    @classmethod
    def from_default_urdf(cls) -> "OfficialG1ArmGravityCompensator":
        """Convenience: `find_default_urdf_path()` + `from_urdf()` の合成。"""
        return cls.from_urdf(find_default_urdf_path())

    def torque_nm(self, arm_position_rad: np.ndarray) -> np.ndarray:
        """指定 arm 姿勢での重力補償 torque [Nm] を返す (14 DOF)。

        `pin.rnea(model, data, q, zeros, zeros)` = 加速度 0、速度 0 での必要 torque
        = 静的釣り合いに要する gravity 相殺分。返り値は Actuator の
        `lowcmd.motor_cmd[i].tau` に埋めて publish 用。

        Args:
            arm_position_rad: (14,) float、EXPECTED_ARM_JOINT_NAMES 順の関節位置。
        """
        q = np.asarray(arm_position_rad, dtype=np.float64)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("gravity compensation input must be finite arms[14]")
        torque = np.asarray(
            self._pin.rnea(
                self._model,
                self._data,
                q,
                np.zeros(self._model.nv),
                np.zeros(self._model.nv),
            ),
            dtype=np.float64,
        )
        if torque.shape != (14,) or not np.isfinite(torque).all():
            raise RuntimeError("official G1 RNEA returned an invalid torque vector")
        maximum = float(np.max(np.abs(torque)))
        if maximum > MAX_ABS_GRAVITY_TORQUE_NM:
            raise RuntimeError(
                "official G1 gravity feed-forward exceeded the deployment bound "
                f"({maximum:.3f}>{MAX_ABS_GRAVITY_TORQUE_NM:.3f} Nm)"
            )
        return torque
