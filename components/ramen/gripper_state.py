"""`:5557` から Dex1-1 の実測開度 (`gripper_q`) だけを読む 2 本目の SUB。

# なぜ 2 本目が要るか

運営の state bridge は 2026-09-21 版から**グリッパの実測を既に載せている**:

    reference/orin_bridge/real_orin_state.py:26-31,41-46,67
      # Dex1 gripper motors. They are NOT part of the 29-joint body vector, so
      # the original real_orin.py never captured them ... motor_state is 35 long
      # on this rig and both indices report.
      # Mapping used by run_wbc_with_dex1.py: q=0.0 closed, q=-5.30 open.
      GRIPPER_INDEX = {"left": 31, "right": 33}
      _latest_state["gripper_q"] = {
          side: {"q": …, "dq": …, "tau_est": …} for side, i in GRIPPER_INDEX.items()}
      msgpack.packb({"body_q": …, "base_quat": …, "gripper_q": …})

ところが `boundary/states.py` の `_decode` は REQUIRED/OPTIONAL の 4 キー
(`body_q` / `base_quat` / `left_hand_q` / `right_hand_q`) しか取り出さないので、
`gripper_q` は捨てられる。`boundary/` は運営所有で改変できない。

そこで**同じ endpoint に購読だけの 2 本目**を張る。ZMQ の PUB は購読者数に
依存しないので、運営側の挙動は一切変わらない (契約違反にならない)。

# これが取れると何が変わるか

- `SyntheticDex1StateSource` (自分の指令のエコー) を実測に置換できる。
  `insert_table_leg` / `rotate_leg_to_tighten` の `hand_state` が学習分布に戻る。
- `pick_leg_hybrid` の interlock が**本物になる**。合成では `実測 - 指令` が
  恒等的に 0 なので、把持判定が一度も発火しない。
- `tau_est` も来る。位置の追従誤差より直接的な把持信号になる (現状は未使用)。

⚠️ 2026-09-20 に調べた `rt/dex1/*/state` は**存在しない**。運営自身が
`tools/diagnose_dex1.py:44` で「推測名で、NVIDIA のコードにも unitree_sdk2py
にも無い」と書いている。DDS を触る必要はない。
"""

from __future__ import annotations

import math
import time

import msgpack
import zmq

#: `boundary/states.py` と同じ topic prefix / port。
STATE_TOPIC = "g1_debug"
DEFAULT_PORT = 5557

#: この時間 新しい sample が来なければ「読めていない」とみなす。
#:
#: bridge は 50 Hz で publish する (`real_orin_state.py:82`) ので、0.5s を超えるのは
#: **bridge 自体が止まったとき**。カメラの鮮度判定と同じ基準にしてある。
#:
#: ⚠️ **ここで期限切れにしないと、下流の staleness 判定が原理的に発火しない。**
#: `MeasuredDex1StateSource` は「dict が来たかどうか」で新鮮さを更新するので、
#: 古い値を返し続けると毎 tick 更新されてしまい、`stale_after_s` に入れない。
#: `real_orin_state.py` (:5557) だけが落ちて `real_orin_cameras.py` (:5555) が
#: 生きている場合、カメラ鮮度では検出できず、**凍結した把持状態を「実測」として
#: model に渡し続ける**ことになる。
STALE_AFTER_S = 0.5

#: 運営が `gripper_q` に入れる per-side のキー。
SIDES = ("left", "right")
FIELDS = ("q", "dq", "tau_est")


class GripperStateStream:
    """`:5557` の `g1_debug` から `gripper_q` だけを取り出す購読専用 SUB。

    `boundary.StateStream` と同じ socket 設定 (SUB + CONFLATE) なので、
    最新の 1 通だけが残り backlog は溜まらない。bridge は 50 Hz で publish する
    (`real_orin_state.py:82` の `state_publish_loop(5557, 50.0)`)。

    Args:
        host: 運営の state endpoint のホスト (client の ``--orin`` と同じ)。
        port: 既定 5557。
        stale_after_s: この時間 新しい sample が来なければ None を返す。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        stale_after_s: float = STALE_AFTER_S,
    ) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        context = zmq.Context.instance()
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._stale_after_s = float(stale_after_s)
        self._latest: dict | None = None
        self._latest_at: float | None = None

    def poll(self) -> dict | None:
        """最新の `gripper_q` を返す。無い / 古ければ None。

        非 blocking。**`stale_after_s` を超えた値は返さない** —
        `boundary.StateStream.latest()` と違い、古い値を握り続けない。理由は
        `STALE_AFTER_S` の comment。

        Returns:
            ``{"left": {"q": float, "dq": float, "tau_est": float}, "right": {...}}``
            または None。**単位は運営の生のモータ角** (q=0.0 閉 / -5.30 開)。
            model 空間への変換は `orchestrator_io.gripper_q_to_model_rad`。
        """
        while True:
            try:
                blob = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            decoded = self._decode(blob)
            if decoded is not None:
                self._latest = decoded
                self._latest_at = time.monotonic()
        if self.age_s is None or self.age_s > self._stale_after_s:
            return None
        return self._latest

    @property
    def age_s(self) -> float | None:
        """最後に受けた sample の経過秒。1 度も来ていなければ None。"""
        if self._latest_at is None:
            return None
        return time.monotonic() - self._latest_at

    @staticmethod
    def _decode(blob: bytes) -> dict | None:
        """`gripper_q` を取り出す。欠落・不正なら None (例外にしない)。

        bridge は `motor_state` が 35 未満のリグでは `gripper_q` を None のまま
        送る (`real_orin_state.py:40`)。それは異常ではなく「このリグでは読めない」
        という情報なので、呼び出し側が合成へ落とせるよう静かに None を返す。
        """
        prefix = STATE_TOPIC.encode("utf-8")
        if not blob.startswith(prefix):
            return None
        try:
            msg = msgpack.unpackb(blob[len(prefix) :], raw=False)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(msg, dict):
            return None
        raw = msg.get("gripper_q")
        if not isinstance(raw, dict):
            return None

        out: dict[str, dict[str, float]] = {}
        for side in SIDES:
            entry = raw.get(side)
            if not isinstance(entry, dict):
                return None
            values = {}
            for field in FIELDS:
                value = entry.get(field)
                if value is None:
                    return None
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(number):
                    return None
                values[field] = number
            out[side] = values
        return out

    def close(self) -> None:
        self._socket.close(linger=0)
