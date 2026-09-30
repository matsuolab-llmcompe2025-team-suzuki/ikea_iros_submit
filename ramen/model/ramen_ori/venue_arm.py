"""会場の運営 WBC の「遅い・柔らかい腕」を学習の状態の入力に入れる (Issue #188)。

学習データの腕は指令にほぼ遅れずについてくる (自前の実機 80〜110 ms)。会場では運営 adapter が chunk ごとに
1 行目を実測へ合わせ直して 1 rad/s までに抑え、柔らかい WBC の腕が一次遅れでついてくる
(09-29: pick 380〜390 ms・台を回す 270 ms)。policy は遅れた腕を見て、教師の指令から外れていく。

ここでは教師の指令を会場の模擬 (`evaluate/model_evaluation/tools/arm_tracking_sim.simulate`。09-29 の log で
遅れを再現した物をそのまま使う) に通し、「会場ならこうなる」腕 14 関節を区間ごと・腕の反応 (tau) ごとに作る。
dataset (`data_lerobot.RamenOriLerobotDataset`) は学習の sample の一部で、今と 1 frame 前の腕をこの表の値に
置き換える。追従のずれ・速度・手先 (FK) は置き換えた関節から作り直す。正解の指令は変えない
(遅れた腕を見ても教師の指令を出し続けることを学ばせる)。

限界: 台は動かない (押し返しは入れない)、重力の垂れ・Dex1 は入れない、会場への送り方は 1 つ (preset)。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import multiprocessing
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    SOURCE_JOINT_SLICES,
    SOURCE_ROOT_POSE_DIM,
    UPPER_BODY_SOURCE_INDEX_MAP,
)

# robot_q_current / robot_q_desired (36) の腕 14 関節。先頭 7 は胴体の位置・向きで、関節 i は [7 + i]
ARM_SLICE_Q36 = slice(
    SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["left_arm"][0],
    SOURCE_ROOT_POSE_DIM + SOURCE_JOINT_SLICES["right_arm"][1],
)
ARM_DIM = ARM_SLICE_Q36.stop - ARM_SLICE_Q36.start
FPS = 30

_CONFIG_KEYS = frozenset({"prob", "taus", "preset", "eval_tau", "workers"})


@dataclass(frozen=True)
class VenueArmConfig:
    """``data.venue_arm`` の設定。

    - prob: 学習の sample を置き換える確率。残りは元の (自前の実機の) 腕のまま
    - taus: 腕の反応の時定数 [s]。表を tau ごとに作り、置き換えるときに 1 つを等確率で選ぶ。0.15 で 09-29 の遅れを再現
    - preset: 私たちの送り方 (skill_config.yaml の ``boundary_arm_tracking.presets``)
    - eval_tau: val / test の sample (学習以外) を常にこの tau で置き換える (評価用、taus のどれか)。None = 置き換えない
    - workers: 表を作る process 数 (0 = CPU 数)
    """

    prob: float = 0.5
    taus: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3)
    preset: str = "standard"
    eval_tau: Optional[float] = None
    workers: int = 0

    @classmethod
    def from_dict(cls, cfg: dict) -> "VenueArmConfig":
        unknown = sorted(set(cfg) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"venue_arm に知らない key {unknown} (使えるのは {sorted(_CONFIG_KEYS)})"
            )
        taus = tuple(float(t) for t in cfg.get("taus", cls.taus))
        eval_tau = cfg.get("eval_tau")
        out = cls(
            prob=float(cfg.get("prob", cls.prob)),
            taus=taus,
            preset=str(cfg.get("preset", cls.preset)),
            eval_tau=None if eval_tau is None else float(eval_tau),
            workers=int(cfg.get("workers", cls.workers)),
        )
        if not 0.0 <= out.prob <= 1.0:
            raise ValueError(f"venue_arm.prob={out.prob} は 0〜1")
        if not out.taus or any(t <= 0 for t in out.taus):
            raise ValueError(f"venue_arm.taus={out.taus} は正の値を 1 つ以上")
        if out.eval_tau is not None and out.eval_tau not in out.taus:
            raise ValueError(
                f"venue_arm.eval_tau={out.eval_tau} は taus {out.taus} のどれか"
            )
        return out

    @property
    def eval_index(self) -> Optional[int]:
        return None if self.eval_tau is None else self.taus.index(self.eval_tau)


# 運営の boundary (inference/desktop/boundary、手を入れない) は送受信の部品を import 時に読む。模擬は送り口を
# 記録係に差し替えて何も送受信しないので、学習の env (これらが無い) では「使われたら止まる」代わりの module を置いて
# import だけ通す。入っている env では本物を使う
_TRANSPORT_MODULES = ("msgpack", "zmq", "cv2")


class _Unavailable:
    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self._name} は学習の env に無い (会場の腕の模擬は送受信しない)")

    def __getattr__(self, attr: str) -> "_Unavailable":
        return _Unavailable(f"{self._name}.{attr}")


def _transport_stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def _missing(attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(f"{name}.{attr}")

    module.__getattr__ = _missing
    return module


def _sim_module():
    # 模擬は推論のコード (送り方の本物) を読むので、使うときだけ import する
    for name in _TRANSPORT_MODULES:
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            sys.modules[name] = _transport_stub(name)
    import inference.desktop.boundary  # noqa: F401  (送り口が関数の中で import する。ここで一度読んでおく)
    from evaluate.model_evaluation.tools import arm_tracking_sim

    return arm_tracking_sim


_SKILL_CONFIG: Optional[dict] = None


def _skill_config() -> dict:
    global _SKILL_CONFIG
    if _SKILL_CONFIG is None:
        import yaml

        _SKILL_CONFIG = yaml.safe_load(
            _sim_module().SKILL_CONFIG.read_text(encoding="utf-8")
        )
    return _SKILL_CONFIG


def simulate_episode(
    command19: np.ndarray, measured14: np.ndarray, tau: float, preset: str
) -> np.ndarray:
    """1 区間の教師の指令 (n, 19) を会場の模擬に通し、各 frame で指令を送る直前の腕 (n, 14) を返す。

    command19 は waist 3 + 腕 14 + hand 2 (推論の 19-D と同じ順)。腕の初期値は記録された腕の 1 frame 目。
    """
    n = len(command19)
    if n == 0:
        return np.zeros((0, ARM_DIM), dtype=np.float32)
    sim = _sim_module()
    case = sim.Case(preset, preset)
    t = np.arange(n, dtype=np.float64) / FPS
    # 送り口は起動時に設定を print する (区間ごとに出ると学習の log が埋まる)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        out = sim.simulate(
            t,
            np.asarray(command19, dtype=np.float64),
            np.asarray(measured14, dtype=np.float64),
            sim._assist(case, _skill_config()),
            tau=float(tau),
            case=case,
        )
    return out["measured"].astype(np.float32)


def _simulate_job(job: tuple) -> np.ndarray:
    return simulate_episode(*job)


def command19_from(robot_q_desired: np.ndarray, hand_cmd: np.ndarray) -> np.ndarray:
    """(N, 36) + (N, 2) の教師の指令 → 推論と同じ 19-D (waist 3 + 腕 14 + hand 2)。"""
    command38 = np.concatenate([robot_q_desired, hand_cmd], axis=1)
    return command38[:, np.asarray(UPPER_BODY_SOURCE_INDEX_MAP)]


def build_tables(
    command19: np.ndarray,
    measured14: np.ndarray,
    lengths: Sequence[int],
    config: VenueArmConfig,
) -> np.ndarray:
    """区間ごとに模擬を回した腕の表 (len(taus), N, 14)。行は hf_dataset の行 (区間が lengths の順に連続して並ぶ)。

    区間の境目で腕の状態は引き継がない (区間ごとに記録された腕の 1 frame 目から始める)。
    """
    lengths = [int(n) for n in lengths]
    total = sum(lengths)
    if len(command19) != total or len(measured14) != total:
        raise ValueError(
            f"行数が合わない: command {len(command19)} / measured {len(measured14)} / 区間の合計 {total}"
        )
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    jobs = [
        (command19[s : s + n], measured14[s : s + n], tau, config.preset)
        for tau in config.taus
        for s, n in zip(starts, lengths)
    ]
    workers = config.workers or os.cpu_count() or 1
    if workers <= 1 or len(jobs) <= 1:
        results = [_simulate_job(job) for job in jobs]
    else:
        # fork なら親の import (推論のコード) をそのまま使える。macOS の既定 (spawn) でも動く
        method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context(method)
        ) as pool:
            results = list(pool.map(_simulate_job, jobs, chunksize=16))
    tables = np.zeros((len(config.taus), total, ARM_DIM), dtype=np.float32)
    it = iter(results)
    for k in range(len(config.taus)):
        for s, n in zip(starts, lengths):
            tables[k, s : s + n] = next(it)
    return tables


def replace_arms(
    q_current: np.ndarray, arm_current: np.ndarray, arm_prev: Optional[np.ndarray]
) -> np.ndarray:
    """(2, 36) の robot_q_current (row 0 = 1 frame 前、row 1 = 今) の腕 14 関節を置き換えた copy。

    arm_prev が None (区間の先頭) なら row 0 は元のまま (区間の外なので速度は 0 になる)。
    """
    out = np.array(q_current, dtype=np.float32, copy=True)
    out[1, ARM_SLICE_Q36] = arm_current
    if arm_prev is not None:
        out[0, ARM_SLICE_Q36] = arm_prev
    return out
