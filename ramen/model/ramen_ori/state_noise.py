"""学習の状態の入力の腕を教師からずらす (Issue #188、flip の追加学習。DART と同じ考え)。

実機では状態が教師から外れる (09-30 の flip: 右腕が教師より 0.9 rad 低いまま) と、model はその外れた状態を
なぞって戻れなかった。学習の sample の一部で、今と 1 frame 前の腕 14 関節に同じずれを足し、正解の指令は教師の
まま学習する (外れた所から教師の動きへ戻ることを学ばせる)。今と 1 frame 前に同じ量を足すので速度は変わらない。
追従のずれ・手先 (FK) はずらした腕から作り直す (dataset 側)。正解の指令にはノイズを足さない (足すとノイズを真似る)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.ramen_ori.venue_arm import ARM_DIM, ARM_SLICE_Q36

_KEYS = frozenset({"prob", "sigma_rad", "max_rad"})


@dataclass(frozen=True)
class StateNoiseConfig:
    """- prob: 学習の sample をずらす確率 / sigma_rad: 関節ごとのずれの標準偏差 / max_rad: ずれの上限 (絶対値)"""

    prob: float = 0.5
    sigma_rad: float = 0.15
    max_rad: float = 0.4

    @classmethod
    def from_dict(cls, cfg: dict) -> "StateNoiseConfig":
        unknown = sorted(set(cfg) - _KEYS)
        if unknown:
            raise ValueError(
                f"state_noise に知らない key {unknown} (使えるのは {sorted(_KEYS)})"
            )
        out = cls(
            prob=float(cfg.get("prob", cls.prob)),
            sigma_rad=float(cfg.get("sigma_rad", cls.sigma_rad)),
            max_rad=float(cfg.get("max_rad", cls.max_rad)),
        )
        if not 0.0 <= out.prob <= 1.0:
            raise ValueError(f"state_noise.prob={out.prob} は 0〜1")
        if out.sigma_rad <= 0 or out.max_rad <= 0:
            raise ValueError(
                f"state_noise の sigma_rad={out.sigma_rad} / max_rad={out.max_rad} は正の値"
            )
        return out

    def offset_from_normal(self, z: np.ndarray) -> np.ndarray:
        """標準正規の乱数 (14,) → 腕のずれ (14,)。乱数は呼ぶ側 (DataLoader の worker ごとの torch) が引く。"""
        z = np.asarray(z, dtype=np.float32).reshape(ARM_DIM)
        return np.clip(z * self.sigma_rad, -self.max_rad, self.max_rad).astype(
            np.float32
        )


def add_arm_offset(q_current: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """(2, 36) の robot_q_current (row 0 = 1 frame 前、row 1 = 今) の腕 14 関節に同じずれを足した copy。"""
    out = np.array(q_current, dtype=np.float32, copy=True)
    out[:, ARM_SLICE_Q36] += np.asarray(offset, dtype=np.float32)
    return out
