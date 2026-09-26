"""天板の回転進捗を YOLO-OBB から追う monitor (Issue #137 Phase 2)。

# なぜ要るか

rotate_table_base の実機 run は、手先が教師と同等の軌跡を描いていても天板が
回っていなかった (Run1 の正味回転 0.4deg、教師 1 ストローク 16.4deg)。
「手が動いたか」ではなく「天板が回ったか」を見ないと空振りを検知できない。

overlay 経路では YOLO-OBB が既に 30Hz で回っていて `obs["cleaned"]` に
`table_top` (class 4) が入るので、追加のセンサも追加の推論も要らない。

# 角度の取り方

矩形 OBB の角度は 180deg 周期。頂点順は ultralytics の `xyxyxyxyn` 準拠で、
**最初の辺 (verts[1] - verts[0])** を使う。「長い辺を使う」方が安定しそうに
見えるが、実測では table_top の 47.7% が長短比 1.15 未満のほぼ正方形で、
長短の判定が反転して 90deg 跳ぶ (フレーム間差が 20deg を超える割合が
最初の辺 0.98% に対し長辺ルールは 5.2%)。よって頂点順をそのまま使う。

# 上位 policy の Kabsch との関係

`skill_planner.geometry.kabsch_rotation_angle_deg` は同じ「天板がどれだけ
回ったか」を別の方法で測る (`enter_pick_table_leg` が Kabsch 18deg で発火)。
**判別力は同等**で、教師 1050 セグメントの「前半 30% vs 末尾 10%」分離で
AUC は first-edge 0.895 / Kabsch 0.893。同じ検知率どうしなら早期発火は
first-edge 6.2% / Kabsch 8.1%。

ただし **値のスケールが違う** (末尾 10% の中央値: first-edge 13.1deg /
Kabsch 51.9deg)。Kabsch は 4 通り cyclic shift の最良フィットを採るため、
正方形に近い矩形で 0deg <-> 90deg の flip が入り値が大きく出る。上位の
18deg はそのスケール上で調整された値なので、**本 monitor の閾値と直接
比較してはいけない**。優劣の問題ではなく単位が違う。

# 累積ではなく「参照からの差」

累積 (フレーム間差の総和) は 1 回の誤検出が恒久的なオフセットとして残る。
本 monitor は **skill 開始時**の中央値を参照として保持し、毎 tick の中央値との
差を取る (retry の attempt 境界では張り直さない — 部分的に回った分を捨てない
ため)。差は ±90deg に収まるが、教師 1 ストロークは 16.4deg なので足りる。

# 閾値の根拠 (教師 1050 セグメント、実測)

「回転がまだ起きていない最初の 0.5s に推定値がどれだけ振れるか」をノイズ床と
して測った (教師が 8deg に達するのは p50 1.8s なので、0.5s 時点の値は実質ノイズ)。

| 設計 | ノイズ床 median | p95 | 0.5s 内の誤検知 (>=8deg が 6 frame 連続) |
|---|---|---|---|
| 累積 (棄却なし) | 2.43deg | 28.5deg | 7.0% |
| 中央値参照 K=5 | 1.66deg | 24.2deg | 3.4% |
| **中央値参照 K=9** | **1.26deg** | **17.6deg** | **1.2%** |

誤検知 = 「回ったと誤判定して retry を打ち切る」方向なので、検知率
(K=9/S=6 で 88.7%) より誤検知を優先して K=9 / sustain 6 frame を既定にした。
sustain を 9 frame にすると誤検知 0.0% / 検知率 84.8%。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


TABLE_TOP_CLASS_NAME = "table_top"


def _wrap_deg(delta: float) -> float:
    """180deg 周期の差を (-90, 90] に畳む。"""
    return ((delta + 90.0) % 180.0) - 90.0


def _obb_angle_deg(verts: np.ndarray) -> float:
    """OBB 4 頂点 → 最初の辺の角度 [deg]、[0, 180)。"""
    edge = np.asarray(verts, dtype=np.float64)[1] - np.asarray(verts, dtype=np.float64)[0]
    return float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)


def _median_angle(window: Sequence[float]) -> float:
    """180deg 周期の角度列の中央値。最新値を基準に畳んでから median を取る。"""
    base = window[-1]
    return base + float(np.median([_wrap_deg(v - base) for v in window]))


@dataclass(frozen=True)
class RotateProgress:
    """1 tick 分の進捗判定。

    Attributes:
        rotation_deg: 参照姿勢からの符号付き回転 [deg]。(-90, 90]。
        sustained: `success_deg` を `sustain_frames` 連続で超えたか。
        usable: 参照が確立済で、かつ検出が stale でない = 判定を信用してよい。
            False の間は「回っていない」ではなく「判定不能」なので、retry の
            発火にも成功判定にも使わないこと。
        detected: 今 tick で table_top を検出したか。
        stale_s: 最後に検出できてからの経過秒。
    """

    rotation_deg: float
    sustained: bool
    usable: bool
    detected: bool
    stale_s: float


class RotateProgressMonitor:
    """`obs["cleaned"]` の table_top OBB から天板の回転量を追う。

    Args:
        success_deg: 「回った」とみなす閾値 [deg]。教師の 1 ストロークは
            median 16.4deg、p25 でも 10.1deg。
        median_window: 角度の中央値窓 [frame]。参照もこの窓で作る。
        sustain_frames: 閾値超えを何 frame 連続で要求するか。
        stale_timeout_s: 未検出がこの秒数を超えたら `usable=False`。
        min_confidence: これ未満の検出は無視する。
        class_name: 追う class 名。
    """

    def __init__(
        self,
        *,
        success_deg: float = 8.0,
        median_window: int = 9,
        sustain_frames: int = 6,
        stale_timeout_s: float = 1.0,
        min_confidence: float = 0.15,
        class_name: str = TABLE_TOP_CLASS_NAME,
    ) -> None:
        if success_deg <= 0.0 or success_deg >= 90.0:
            raise ValueError(f"success_deg must be in (0, 90), got {success_deg}")
        if median_window < 1 or sustain_frames < 1:
            raise ValueError(
                "median_window and sustain_frames must be >= 1, got "
                f"{median_window} / {sustain_frames}"
            )
        if stale_timeout_s <= 0.0:
            raise ValueError(f"stale_timeout_s must be > 0, got {stale_timeout_s}")
        self._success_deg = float(success_deg)
        self._median_window = int(median_window)
        self._sustain_frames = int(sustain_frames)
        self._stale_timeout_s = float(stale_timeout_s)
        self._min_confidence = float(min_confidence)
        self._class_name = class_name
        self.reset()

    def reset(self) -> None:
        """**skill 開始時に 1 回だけ**呼ぶ。参照を張り直し進捗をゼロに戻す。

        retry の attempt 境界では呼ばないこと。参照を張り直すと 1 回目で
        部分的に回した分が忘れられ、「全体でどれだけ回ったか」が見えなくなる。
        """
        self._window: deque[float] = deque(maxlen=self._median_window)
        self._reference: float | None = None
        self._rotation_deg = 0.0
        self._sustain_count = 0
        self._last_detect_ns: int | None = None

    def update(self, detections: Any, t_ns: int) -> RotateProgress:
        """1 tick 進める。

        Args:
            detections: `obs["cleaned"]` (OBBDetection の列) or None。
            t_ns: tick の timestamp [ns]。

        Returns:
            RotateProgress。
        """
        angle = self._pick_angle(detections)
        if angle is None:
            stale_s = (
                float("inf")
                if self._last_detect_ns is None
                else max(0.0, (t_ns - self._last_detect_ns) / 1e9)
            )
            # 単発の検出漏れで判定を捨てない。手が天板にかぶる区間があるので、
            # `stale_timeout_s` までは直前の判定を維持する。sustain は伸ばさない
            # (観測できていない間を「連続」に数えない) が、reset もしない —
            # reset すると検出が間欠な区間で sustain が永久に立たなくなる。
            fresh = stale_s <= self._stale_timeout_s
            if not fresh:
                self._sustain_count = 0
            return RotateProgress(
                rotation_deg=self._rotation_deg,
                sustained=fresh and self._sustain_count >= self._sustain_frames,
                usable=fresh and self._reference is not None,
                detected=False,
                stale_s=stale_s,
            )

        self._last_detect_ns = t_ns
        self._window.append(angle)
        if len(self._window) < self._median_window:
            # 参照がまだ立っていない = 判定不能。
            return RotateProgress(
                rotation_deg=0.0,
                sustained=False,
                usable=False,
                detected=True,
                stale_s=0.0,
            )
        current = _median_angle(list(self._window))
        if self._reference is None:
            self._reference = current
        self._rotation_deg = _wrap_deg(current - self._reference)
        if abs(self._rotation_deg) >= self._success_deg:
            self._sustain_count += 1
        else:
            self._sustain_count = 0
        return RotateProgress(
            rotation_deg=self._rotation_deg,
            sustained=self._sustain_count >= self._sustain_frames,
            usable=True,
            detected=True,
            stale_s=0.0,
        )

    def _pick_angle(self, detections: Any) -> float | None:
        """最も confidence の高い table_top の角度。無ければ None。"""
        if not detections:
            return None
        best = None
        for det in detections:
            if getattr(det, "class_name", None) != self._class_name:
                continue
            conf = float(getattr(det, "confidence", 0.0))
            if conf < self._min_confidence:
                continue
            if best is None or conf > best[0]:
                best = (conf, det.verts)
        return None if best is None else _obb_angle_deg(best[1])


def load_progress_monitor_for_skill(
    skill_config: Any, skill_name: str
) -> "RotateProgressMonitor | None":
    """skill_config.yaml から monitor を解決。`progress` block が無ければ None。

    `load_motion_limits_for_skill` と同じ解決方針だが、こちらは **default を
    持たない**。監視は skill 固有 (回転する天板があるのは rotate 系だけ) なので、
    明示的に書いた skill でのみ有効化する。

    Args:
        skill_config: yaml.safe_load(skill_config.yaml) の結果 (top-level dict)。
        skill_name: `skills.<name>` の key。

    Returns:
        RotateProgressMonitor、または `progress` 未記載なら None。
    """
    skills = skill_config.get("skills") if hasattr(skill_config, "get") else None
    entry = (skills or {}).get(skill_name) or {}
    cfg = entry.get("progress")
    if not isinstance(cfg, dict):
        return None
    unknown = set(cfg) - {
        "success_deg", "median_window", "sustain_frames",
        "stale_timeout_s", "min_confidence", "class_name",
    }
    if unknown:
        raise ValueError(
            f"skills.{skill_name}.progress has unknown keys: {sorted(unknown)}"
        )
    return RotateProgressMonitor(**cfg)
