"""memory の入力 (Issue #141 Phase 7、監査 doc §2「変種の候補 1: memory の入力」)。

「天板や手が区間の開始時からどれだけ動いたか」と「自分がここまでどう動いてきたか」を 51 個の数にまとめる。
学習と推論が同じ `MemoryTracker` を frame / tick ごとに回す: 学習は区間 (episode) ごとに frame 順に回して表にし
(`data_lerobot.py`)、推論は skill の開始時に `reset()` して tick ごとに `update()` する。numpy だけで書く (推論の env でも動く)。

# 1 tick の入力

- head_left の YOLO-OBB の検出: class_id (K,) / conf (K,) / verts (K, 4, 2)。verts は 0〜1 の座標 (ultralytics の
  xyxyxyxyn、学習の OBB の cache と推論の検出で同じ)。conf ≥ 0.30 だけを使う (overlay と同じ)
- 手先の位置 (6,): 1 tick 前に送った指令 (19D) を FK した左手先 xyz + 右手先 xyz [m]。指令がまだ無い区間の先頭は None。
  学習は教師の指令 (1 frame 前)、推論は実際に送った指令なので、実機の関節角の古さ (INF-18) の影響を受けない

# 出力 (51,) の並び (`MEMORY_LAYOUT`)

| 部分 | 中身 | 個数 |
|---|---|---:|
| 1 個しか無い class (workspace / table_top / hand_left / hand_right) | 中心 x / y・大きさ w / h・角度の、最初に見えた frame との差 + 今見えているか (0/1) | 4 × 6 |
| 複数ある class (leg / leg_tip / hole) | 今の個数 + 平均の中心 x / y の、最初に見えた frame との差 | 3 × 3 |
| 自分の動き | 手先の速度 (左 xyz + 右 xyz) の移動平均 (τ 0.25 / 0.5 / 1.0 s) | 3 × 6 |

- 1 個しか無い class が複数検出されたら conf が一番高いもの。複数ある class は、どれがどれかを時間方向に追うと
  検出の出入りで取り違えるので、個数と平均にまとめる
- 大きさは最初の辺 (verts[1] − verts[0]) と 2 番目の辺の長さ。角度は最初の辺の角度 [deg] で、progress monitor
  (`inference/desktop/lower_policy/skills/rotate_progress.py`) と同じ取り方 (0〜1 の座標のまま)。差は ±90° に折り返す。
  全 class を同じ規則で入れる (天板と workspace の角度の差を両方入れるので、workspace に対する天板の角度は model が作れる)
- 最初に見えるまでは差 0。見逃した frame は前の差を保つ (見えているか = 0、個数 = 0)
- 速度 = (今の手先 − 1 tick 前の手先) × 30 [m/s] (1 tick = 1/30 s)。区間の先頭から 2 tick は 0。
  移動平均 m ← a·m + (1 − a)·v、a = exp(−(1/30) / τ)、0 から始める
"""

from __future__ import annotations

import numpy as np

MEMORY_VERSION = 1
FPS = 30.0
CONF_MIN = 0.30
# head_left (overlay を描くカメラと同じ)
MEMORY_CAMERA = "observation.images.cam_0"
# YOLO-OBB v11b の class 番号 (data/yolo_obb/configs/dataset_v11b.yaml)
SINGLE_CLASSES: dict[str, int] = {"workspace": 0, "table_top": 4, "hand_left": 6, "hand_right": 5}
MULTI_CLASSES: dict[str, int] = {"leg": 1, "leg_tip": 2, "hole": 3}
TAUS_S: tuple[float, ...] = (0.25, 0.5, 1.0)

_SINGLE_FIELDS = ("dcx", "dcy", "dw", "dh", "dangle", "visible")
_MULTI_FIELDS = ("count", "dcx", "dcy")
MEMORY_LAYOUT: tuple[str, ...] = (
    *(f"{c}.{f}" for c in SINGLE_CLASSES for f in _SINGLE_FIELDS),
    *(f"{c}.{f}" for c in MULTI_CLASSES for f in _MULTI_FIELDS),
    *(f"motion.tau{tau:g}.{side}.{axis}" for tau in TAUS_S for side in ("left", "right") for axis in "xyz"),
)
MEMORY_DIM = len(MEMORY_LAYOUT)
# 自分の動きの 18 個 (学習時にまとめて隠す範囲)
MOTION_SLICE = slice(MEMORY_DIM - 6 * len(TAUS_S), MEMORY_DIM)


def _wrap_deg(delta: float) -> float:
    """180° 周期の差を ±90° に畳む (rotate_progress と同じ)。"""
    return ((delta + 90.0) % 180.0) - 90.0


def _box_geometry(verts: np.ndarray) -> np.ndarray:
    """OBB 4 頂点 (4, 2) → [中心 x, 中心 y, 最初の辺の長さ, 2 番目の辺の長さ, 最初の辺の角度 [deg]、0〜180)]。"""
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[1]
    return np.array(
        [
            verts[:, 0].mean(),
            verts[:, 1].mean(),
            np.hypot(e1[0], e1[1]),
            np.hypot(e2[0], e2[1]),
            np.degrees(np.arctan2(e1[1], e1[0])) % 180.0,
        ]
    )


class MemoryTracker:
    """区間 (skill) の中で 1 tick ずつ memory を更新する。区間の開始時に `reset()`。"""

    def __init__(self) -> None:
        self._alpha = np.exp(-(1.0 / FPS) / np.asarray(TAUS_S))[:, None]   # (τ の数, 1)
        self.reset()

    def reset(self) -> None:
        self._single_ref: dict[str, np.ndarray | None] = {c: None for c in SINGLE_CLASSES}
        self._single_diff = {c: np.zeros(5) for c in SINGLE_CLASSES}
        self._multi_ref: dict[str, np.ndarray | None] = {c: None for c in MULTI_CLASSES}
        self._multi_diff = {c: np.zeros(2) for c in MULTI_CLASSES}
        self._prev_hand: np.ndarray | None = None
        self._ema = np.zeros((len(TAUS_S), 6))

    def update(
        self,
        class_id: np.ndarray,
        conf: np.ndarray,
        verts: np.ndarray,
        hand_pos: np.ndarray | None,
    ) -> np.ndarray:
        """1 tick 進めて (51,) float32 を返す。

        Args:
            class_id: (K,) 検出の class 番号
            conf: (K,) 検出の confidence
            verts: (K, 4, 2) 0〜1 の座標
            hand_pos: (6,) 1 tick 前に送った指令の左右の手先 [m]、無ければ None
        """
        keep = np.asarray(conf) >= CONF_MIN
        class_id = np.asarray(class_id)[keep]
        conf = np.asarray(conf)[keep]
        verts = np.asarray(verts, dtype=np.float64)[keep]

        out: list[float] = []
        for name, cid in SINGLE_CLASSES.items():
            idx = np.flatnonzero(class_id == cid)
            if idx.size:
                geometry = _box_geometry(verts[idx[np.argmax(conf[idx])]])
                if self._single_ref[name] is None:
                    self._single_ref[name] = geometry
                diff = geometry - self._single_ref[name]
                diff[4] = _wrap_deg(diff[4])
                self._single_diff[name] = diff
            out.extend(self._single_diff[name])
            out.append(float(idx.size > 0))
        for name, cid in MULTI_CLASSES.items():
            idx = np.flatnonzero(class_id == cid)
            if idx.size:
                center = verts[idx].mean(axis=(0, 1))
                if self._multi_ref[name] is None:
                    self._multi_ref[name] = center
                self._multi_diff[name] = center - self._multi_ref[name]
            out.append(float(idx.size))
            out.extend(self._multi_diff[name])

        velocity = np.zeros(6)
        if hand_pos is not None:
            hand_pos = np.asarray(hand_pos, dtype=np.float64)
            if self._prev_hand is not None:
                velocity = (hand_pos - self._prev_hand) * FPS
        self._prev_hand = hand_pos
        self._ema = self._alpha * self._ema + (1.0 - self._alpha) * velocity[None]
        out.extend(self._ema.ravel())
        return np.asarray(out, dtype=np.float32)
