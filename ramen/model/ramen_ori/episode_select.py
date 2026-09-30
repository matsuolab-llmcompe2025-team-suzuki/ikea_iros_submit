"""教師の手順が揃った区間だけで学習する (Issue #188、flip の追加学習)。

flip の教師 (成功 429 区間) は手順がばらつく (大まかに分けても 39 通り、右腕が先に最高点へ行く区間が 25%)。
model が区間ごとに違う手順の断片を拾い、実機で「左腕だけで傾ける → 左手を離す → 戻る」を繰り返した (09-30)。
記録された関節から区間ごとに手順を判定し、揃った区間だけを train / val / test に残す。

判定 (どれも None / False なら見ない):
- left_peak_first: 左肩 pitch が最小 (最も上がる) になる時刻が、右肩 pitch より前か同時
- right_push_max_rad: 右肩 pitch の最小がこれより小さい (右腕でも持ち上げて押す)
- max_peak_gap_s: 左右の最高点の時刻の差がこれ以下 (両腕でほぼ同時に持ち上げる)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    SOURCE_JOINT_SLICES,
    SOURCE_ROOT_POSE_DIM,
)

# robot_q_current (36) の左右の肩 pitch (胴体の位置・向き 7 + 関節の番号)
LEFT_SHOULDER_PITCH = SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["left_arm"][0]
RIGHT_SHOULDER_PITCH = SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["right_arm"][0]
FPS = 30

_KEYS = frozenset({"left_peak_first", "right_push_max_rad", "max_peak_gap_s"})


@dataclass(frozen=True)
class EpisodeSelectConfig:
    left_peak_first: bool = False
    right_push_max_rad: Optional[float] = None
    max_peak_gap_s: Optional[float] = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "EpisodeSelectConfig":
        unknown = sorted(set(cfg) - _KEYS)
        if unknown:
            raise ValueError(
                f"episode_select に知らない key {unknown} (使えるのは {sorted(_KEYS)})"
            )
        push = cfg.get("right_push_max_rad")
        gap = cfg.get("max_peak_gap_s")
        out = cls(
            left_peak_first=bool(cfg.get("left_peak_first", False)),
            right_push_max_rad=None if push is None else float(push),
            max_peak_gap_s=None if gap is None else float(gap),
        )
        if out.max_peak_gap_s is not None and out.max_peak_gap_s < 0:
            raise ValueError(
                f"episode_select.max_peak_gap_s={out.max_peak_gap_s} は 0 以上"
            )
        return out


def select_episodes(
    q_current: np.ndarray, lengths: Sequence[int], config: EpisodeSelectConfig
) -> np.ndarray:
    """区間ごとに残すか (len(lengths),) bool。q_current は (N, 36)、区間が lengths の順に連続して並ぶ。"""
    lengths = [int(n) for n in lengths]
    if len(q_current) != sum(lengths):
        raise ValueError(
            f"行数が合わない: q_current {len(q_current)} / 区間の合計 {sum(lengths)}"
        )
    keep = np.ones(len(lengths), dtype=bool)
    start = 0
    for i, n in enumerate(lengths):
        seg = q_current[start : start + n]
        start += n
        if n == 0:
            keep[i] = False
            continue
        left, right = seg[:, LEFT_SHOULDER_PITCH], seg[:, RIGHT_SHOULDER_PITCH]
        t_left, t_right = int(np.argmin(left)), int(np.argmin(right))
        if config.left_peak_first and t_left > t_right:
            keep[i] = False
        if (
            config.right_push_max_rad is not None
            and not right.min() < config.right_push_max_rad
        ):
            keep[i] = False
        if (
            config.max_peak_gap_s is not None
            and abs(t_right - t_left) / FPS > config.max_peak_gap_s
        ):
            keep[i] = False
    return keep
