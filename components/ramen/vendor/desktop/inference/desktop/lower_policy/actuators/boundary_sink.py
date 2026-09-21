"""Orchestrator の 19-D action を大会 boundary (`:5556`) へ出す sink。

# なぜ要るか

自前経路 (`entrypoint.py`) は action を `rt/arm_sdk` へ直接 publish する。腕と歩行は
それで動くが、**グリッパは動かない** — `rt/dex1/*/cmd` を購読する serial↔DDS 中継
(`dex1_1_gripper_server`) は我々のラボ構成にしか無く、会場には存在しない
(`docs/setup/thor_pc2_environment.md` §9.6)。

会場でグリッパまで動かせる経路は 1 本だけで、それが boundary:

    我々 → (T,25) を :5556 に publish → 運営 wbc_adapter (pure relay) → WBC → ロボット

`WBC_RUNBOOK` §7 が "wbc_driver.py's gripper path is a verified pure relay (raw values
in, raw values out)" と明記しているとおり、hand 列もそのまま流れる。

# 何を自前で書かないか

publish 側 (bind / 検証 / フレーミング) は **運営の実装をそのまま使う**
(`inference/desktop/boundary/actions.py:DecoupledSink`、vendor・無改変)。
同梱の README が "a local edit here will pass your tests and fail on the robot" と
警告している当のものなので、同等品を自作しない。

自前で持つのは **19-D → 38-D の組み立てだけ**で、そこから先の 38-D → (T,25) は
既存の `taskspace_adapter.groot_chunk_to_taskspace()` を通す。

# 注意

- **`:5556` は client が bind する側。** 運営の adapter がこちらへ dial-in する
  (`boundary/actions.py` が "THE CLIENT BINDS ... trips everyone up once" と警告)。
  本 sink を Thor 上で動かす場合、adapter の `--actions-host` をそちらへ向けてもらう
  必要がある (**2026-09-20 時点で未検証**)。
- `ee_frame_transform` は既定 `None` (= pelvis/root-link frame)。運営 IK が期待する
  EE 原点が未確定のため (2026-09-15 に質問済み・未回答)。判明したら値を渡すだけ。
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Sequence

import numpy as np

from inference.desktop.lower_policy.policies.taskspace_adapter import (
    groot_chunk_to_taskspace,
)
from inference.desktop.lower_policy.skills.vla_skill import (
    ACTION_DIM_TOTAL,
    ARMS_SLICE,
    HAND_SLICE,
    WAIST_SLICE,
)

# G1 canonical body order の脚 12 dof (G1JointIndex 0..11)。腰・腕は action で
# 上書きするが、脚は policy が出さないので **実測値をそのまま使う**。
LEG_DOF = 12
BODY_DOF = 29


def assemble_action38(
    action19: Sequence[float],
    body_q29: Sequence[float],
) -> np.ndarray:
    """19-D action + 実測 body_q(29) → GR00T raw 38-D (root7 + body29 + hand2)。

    提出側 `components/ramen/orchestrator_driver.py` が同じ組み立てをしている。
    FK は root7 を使わない (pelvis 基準の相対 chain) ので、root は identity で埋める。

    Args:
        action19: `vla_skill` の 19-D (waist3 + arms14 + hand2)。
        body_q29: `rt/lowstate` 由来の実測関節角 (G1JointIndex 順)。脚 12 dof のみ使う。

    Returns:
        (38,) float64。`groot_chunk_to_taskspace` にそのまま渡せる。
    """

    action = np.asarray(action19, dtype=np.float64).reshape(-1)
    if action.shape != (ACTION_DIM_TOTAL,):
        raise ValueError(f"action19 must be ({ACTION_DIM_TOTAL},), got {action.shape}")
    body = np.asarray(body_q29, dtype=np.float64).reshape(-1)
    if body.shape != (BODY_DOF,):
        raise ValueError(f"body_q29 must be ({BODY_DOF},), got {body.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("action19 must be finite")
    if not np.all(np.isfinite(body[:LEG_DOF])):
        raise ValueError("body_q29 legs must be finite")

    # root7 = 位置 0 + 単位 quat (w-first)。FK が使わないので値は効かないが、
    # 38-D の形を崩さないために埋める。
    root = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    body29 = np.concatenate([body[:LEG_DOF], action[WAIST_SLICE], action[ARMS_SLICE]])
    return np.concatenate([root, body29, action[HAND_SLICE]])


def assemble_action19(
    waist3: Optional[Sequence[float]],
    arms14: Sequence[float],
    hand2: Optional[Sequence[float]],
    *,
    measured_waist3: Optional[Sequence[float]] = None,
    fallback_hand2: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """腰 / 腕 / 手の 3 断片から 19-D action を組み直す。

    orchestrator が `actuator_send_fn` に渡すのは **腕 14-D だけ**で、腰と手は skill が
    それぞれの actuator へ直接送っている。boundary へ出すには 19-D 全体が要るので、
    mock actuator が保持している直近 target (`latest`) から組み直す
    (提出側 `orchestrator_driver.py` の `assemble_19d` と同じ考え方)。

    Args:
        waist3: 腰 actuator の直近 target。skill がまだ送っていなければ `None`。
        arms14: orchestrator から来た腕 14-D。
        hand2: 手 actuator の直近 target。未送信なら `None`。
        measured_waist3: `waist3` が `None` のときに使う実測腰角。これも無ければ 0。
        fallback_hand2: `hand2` が `None` のときの値。
    """

    arms = np.asarray(arms14, dtype=np.float64).reshape(-1)
    if arms.shape != (14,):
        raise ValueError(f"arms14 must be (14,), got {arms.shape}")

    if waist3 is not None:
        waist = np.asarray(waist3, dtype=np.float64).reshape(-1)
    elif measured_waist3 is not None:
        waist = np.asarray(measured_waist3, dtype=np.float64).reshape(-1)
    else:
        waist = np.zeros(3, dtype=np.float64)
    if waist.shape != (3,):
        raise ValueError(f"waist must be (3,), got {waist.shape}")

    hand = np.asarray(
        hand2 if hand2 is not None else fallback_hand2, dtype=np.float64
    ).reshape(-1)
    if hand.shape != (2,):
        raise ValueError(f"hand must be (2,), got {hand.shape}")

    return np.concatenate([waist, arms, hand])


class BoundaryActionSink:
    """19-D action を (T,25) にして運営 boundary へ publish する。

    `rt/arm_sdk` 直の actuator と**同時には使わない**。両方が同じ関節を動かすため。

    Args:
        fk: `G1WristFK` 相当 (`compute_ee_transforms` を持つもの)。
        port / host: `DecoupledSink` にそのまま渡す。
        ee_frame_transform: root-link → 運営 IK が期待する frame の 4x4。未確定なので
            既定 `None` (変換なし)。
        log_fn: `dict` を 1 件受け取る callable。publish した raw (T,25) を残す
            (`WBC_RUNBOOK` §5:「グリッパが閉じたかは自分の publish 値で分かる」)。
    """

    def __init__(
        self,
        fk: Any,
        *,
        port: int = 5556,
        host: str = "*",
        ee_frame_transform: Optional[np.ndarray] = None,
        log_fn: Any = None,
    ) -> None:
        # boundary は zmq / cv2 / msgpack を引くので lazy import
        # (default env から本 module を import しても壊さない)。
        from inference.desktop.boundary import DecoupledSink

        self._fk = fk
        self._ee_frame_transform = ee_frame_transform
        self._log_fn = log_fn
        self._sink = DecoupledSink(port=port, host=host)
        self._sent = 0
        print(
            f"[boundary] DecoupledSink bound on {host}:{port} "
            "(the organizer's adapter dials in to this)",
            file=sys.stderr,
        )

    @property
    def sent_count(self) -> int:
        return self._sent

    def send_action(self, action19: Sequence[float], body_q29: Sequence[float]) -> None:
        """1 tick 分の 19-D action を (1,25) chunk として publish する。"""

        action38 = assemble_action38(action19, body_q29)
        chunk = groot_chunk_to_taskspace(
            action38[None, :], self._fk, ee_frame_transform=self._ee_frame_transform
        )
        # 検証は DecoupledSink.send_chunk が中でやる (不正なら ActionError)。
        self._sink.send_chunk(chunk)
        self._sent += 1
        if self._log_fn is not None:
            row = np.asarray(chunk[0], dtype=np.float64)
            self._log_fn(
                {
                    "event": "boundary_taskspace",
                    "seq": self._sent,
                    # hand 列は「掴んだか」の一次証拠。-1=open / +1=closed。
                    "left_hand": row[0:2].tolist(),
                    "right_hand": row[2:4].tolist(),
                    "left_ee_pos": row[4:7].tolist(),
                    "right_ee_pos": row[11:14].tolist(),
                    "taskspace_25": row.tolist(),
                    # 逆算用: この tick で読めた実測関節角。静止保持中の後半を使えば
                    # 「指令した EE」と「到達した関節を自前 FK に通した EE」の差 =
                    # 運営 IK が期待する frame とのオフセットが解ける (EE 原点が
                    # どの資料にも無いため、実測から求めるしかない)。
                    "measured_body_q29": np.asarray(
                        body_q29, dtype=np.float64
                    ).tolist(),
                }
            )

    def close(self) -> None:
        """冪等。"""

        sink, self._sink = self._sink, None
        if sink is not None:
            sink.close()
