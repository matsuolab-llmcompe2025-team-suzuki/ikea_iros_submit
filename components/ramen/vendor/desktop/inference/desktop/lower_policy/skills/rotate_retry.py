"""空振りしたら戻って段を下げて再試行する FSM (Issue #137 Phase 4)。

# なぜ要るか

実機の rotate_table_base は 1 回目のストロークから空振りしていて、その後
止まっていた。学習データ側にリカバリ動作が無いことは実測で確認済み
(curation の failure 1390 セグメントは 2 本目が **43mm 浅くなる**、深くなるのは
23% だけ = 「下げて補正」はデータに入っていない)。よって model 側にリカバリを
期待できないので、**実行層で「model が知っている状態」に戻して打ち直す**。

空振り時は天板が動かない (実測 Run1 の正味回転 0.4deg) ので、腕だけ開始姿勢に
戻せば学習開始状態がほぼ完全に復元される、という前提に立っている。

# 段 (rung)

段ごとに `z_ceiling` を下げる。値は教師 1401 セグメントの最下点分布から採る:

    段 1: なし (素のモデル)
    段 2: 122.8mm  (教師 最下点 median)
    段 3: 112.1mm  (教師 最下点 p5) ← 最終段

「いつか成功する」ではなく「教師分布の底で打ち止め」になるので、下げ幅の上限が
データで定義される。実機実測の最下点は Run1 147.9mm / Run2 192.4mm。

# 成功時に skill を終わらせない

「回り切ったので次の skill へ」は **上位 policy が既に持っている**
(`skill_planner.enter_conditions.enter_pick_table_leg` が Kabsch 18deg /
aspect 比 1.35 で発火)。本 FSM が skill を終了させると責務が二重になるので、
成功検知後は **新しい attempt を打たなくなるだけ** (HOLDING) にして、遷移は
上位に委ねる。評価経路 (`run_skill.py`、上位 policy 無し) では max_seconds まで
hold のまま終わる。

# 予算

`attempt_seconds` 3.0s は教師の「8deg 到達時刻」分布から。到達は p50 1.8s /
p75 2.4s / p90 3.0s で、3.0s までに 89% が到達する。成功検知で attempt は
早期終了するので、3.0s を使い切るのは失敗した attempt だけ。3 段 x 3.0s +
戻り 2 回で 12s には収まらないため、retry 有効時は `max_seconds_hard` を
15s に上げる (config 側)。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np


class RetryPhase(str, Enum):
    """FSM の状態。"""

    RUNNING = "running"        # policy を回している
    RETURNING = "returning"    # 開始姿勢へ戻している (policy は呼ばない)
    HOLDING = "holding"        # 成功検知後。上位の遷移待ちでその場保持


class ReturnMode(str, Enum):
    """戻り方。"""

    FULL_RETURN = "full_return"                  # 開始姿勢まで完全に戻す
    LIFT_AND_REAPPROACH = "lift_and_reapproach"  # 手先 z だけ開始高さまで上げる


@dataclass(frozen=True)
class RetryDecision:
    """1 tick 分の指示。

    Attributes:
        phase: 現在の状態。
        ceiling_m: RUNNING 中に適用する z 天井 [m]。None は素のモデル。
        attempt_index: 0 起点の attempt 番号 (= 段 index)。
        flush_execution_state: この tick が attempt の境界で、policy /
            limiter / monitor の内部状態を捨てるべきか。
        reason: 直前の遷移理由 (診断用)。
    """

    phase: RetryPhase
    ceiling_m: float | None
    attempt_index: int
    flush_execution_state: bool
    reason: str


class RotateRetryController:
    """空振り検知 → 戻り → 段を下げて再試行、の状態遷移だけを持つ。

    腕の目標値そのものは `VlaSkill` 側が組む (本 class は「今どの段か」「戻れ」
    「もう打つな」だけを返す)。FK も policy も知らないので単体 test しやすい。

    Args:
        rungs: 段ごとの z 天井 [m]。`None` は天井なし (素のモデル)。
            先頭が 1 回目。
        attempt_seconds: 1 attempt の上限秒。成功検知で早期終了する。
        return_timeout_s: 戻り動作の上限秒。これを超えたら諦めて次段に進む
            (戻れないまま無限に粘ると skill の予算を食い潰すため)。
    """

    def __init__(
        self,
        *,
        rungs: Sequence[float | None],
        attempt_seconds: float = 3.0,
        return_timeout_s: float = 3.0,
    ) -> None:
        rungs = list(rungs)
        if not rungs:
            raise ValueError("rungs must not be empty")
        depths = [r for r in rungs if r is not None]
        if any(d <= 0.0 for d in depths):
            raise ValueError(f"rung ceilings must be > 0 m, got {rungs}")
        if depths != sorted(depths, reverse=True):
            raise ValueError(
                f"rungs must descend (段が進むほど深く), got {rungs}"
            )
        if attempt_seconds <= 0.0 or return_timeout_s <= 0.0:
            raise ValueError(
                "attempt_seconds and return_timeout_s must be > 0, got "
                f"{attempt_seconds} / {return_timeout_s}"
            )
        self._rungs = rungs
        self._attempt_seconds = float(attempt_seconds)
        self._return_timeout_s = float(return_timeout_s)
        self.reset()

    @property
    def rungs(self) -> list[float | None]:
        return list(self._rungs)

    def reset(self) -> None:
        """episode 開始時に呼ぶ。"""
        self._phase = RetryPhase.RUNNING
        self._attempt_index = 0
        self._phase_started_ns: int | None = None
        self._reason = "start"

    def update(
        self, *, t_ns: int, progress: Any, return_complete: bool
    ) -> RetryDecision:
        """1 tick 進める。

        Args:
            t_ns: tick の timestamp [ns]。
            progress: `RotateProgressMonitor.update` の戻り値。`usable` かつ
                `sustained` のときだけ成功扱いにする (判定不能を成功にしない)。
            return_complete: RETURNING 中に戻り目標へ到達したか。他 phase では
                無視される。

        Returns:
            RetryDecision。
        """
        if self._phase_started_ns is None:
            self._phase_started_ns = t_ns
        elapsed_s = max(0.0, (t_ns - self._phase_started_ns) / 1e9)
        flush = False

        if self._phase is RetryPhase.RUNNING:
            if self._is_success(progress):
                self._enter(RetryPhase.HOLDING, t_ns, "rotation_detected")
            elif elapsed_s >= self._attempt_seconds:
                if self._attempt_index + 1 < len(self._rungs):
                    self._enter(RetryPhase.RETURNING, t_ns, "attempt_timeout")
                else:
                    # 最終段。戻っても下げる先が無いので、そのまま回し続ける。
                    # 打ち止めで hold するより上位に遷移の機会を残す方がよい。
                    self._reason = "last_rung_continues"
                    self._phase_started_ns = t_ns
        elif self._phase is RetryPhase.RETURNING:
            if return_complete or elapsed_s >= self._return_timeout_s:
                self._attempt_index += 1
                flush = True
                self._enter(
                    RetryPhase.RUNNING,
                    t_ns,
                    "return_complete" if return_complete else "return_timeout",
                )
        else:  # HOLDING: 上位 policy の遷移待ち。自分からは戻らない。
            pass

        return RetryDecision(
            phase=self._phase,
            # HOLDING は「成功したので新しい attempt を打たない」状態であって
            # 拘束を解く状態ではない。成功した段の天井は維持する (解くと、上位
            # policy の遷移待ちの間にモデルが自由に潜れてしまう)。RETURNING は
            # `lift_to` で持ち上げる局面なので天井を掛けない。
            ceiling_m=(
                self._rungs[self._attempt_index]
                if self._phase in (RetryPhase.RUNNING, RetryPhase.HOLDING)
                else None
            ),
            attempt_index=self._attempt_index,
            flush_execution_state=flush,
            reason=self._reason,
        )

    @staticmethod
    def _is_success(progress: Any) -> bool:
        return bool(
            progress is not None
            and getattr(progress, "usable", False)
            and getattr(progress, "sustained", False)
        )

    def _enter(self, phase: RetryPhase, t_ns: int, reason: str) -> None:
        self._phase = phase
        self._phase_started_ns = t_ns
        self._reason = reason


def load_retry_controller_for_skill(
    skill_config: Any, skill_name: str
) -> tuple["RotateRetryController | None", dict]:
    """skill_config.yaml から controller と付随設定を解決。

    `retry` block が無ければ `(None, {})` = 完全に現状動作。

    Returns:
        (controller, options)。options は `VlaSkill` 側が使う
        `return_mode` / `return_tolerance_rad` / `lift_margin_m` /
        `max_seconds_hard`。
    """
    skills = skill_config.get("skills") if hasattr(skill_config, "get") else None
    entry = (skills or {}).get(skill_name) or {}
    cfg = entry.get("retry")
    if not isinstance(cfg, dict):
        return None, {}
    known = {
        "rungs_mm", "attempt_seconds", "return_timeout_s", "return_mode",
        "return_tolerance_rad", "lift_margin_mm", "max_seconds_hard",
    }
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(
            f"skills.{skill_name}.retry has unknown keys: {sorted(unknown)}"
        )
    rungs = [
        None if v is None else float(v) / 1000.0
        for v in cfg.get("rungs_mm", [None])
    ]
    controller = RotateRetryController(
        rungs=rungs,
        attempt_seconds=float(cfg.get("attempt_seconds", 3.0)),
        return_timeout_s=float(cfg.get("return_timeout_s", 3.0)),
    )
    options = {
        "return_mode": ReturnMode(cfg.get("return_mode", "full_return")),
        "return_tolerance_rad": float(cfg.get("return_tolerance_rad", 0.05)),
        "lift_margin_m": float(cfg.get("lift_margin_mm", 0.0)) / 1000.0,
        "max_seconds_hard": cfg.get("max_seconds_hard"),
    }
    return controller, options


def arm_return_complete(
    measured_arm_14: np.ndarray, target_arm_14: np.ndarray, tolerance_rad: float
) -> bool:
    """full_return の到達判定 (全 14 joint が許容内)。"""
    return bool(
        np.max(np.abs(np.asarray(measured_arm_14) - np.asarray(target_arm_14)))
        <= tolerance_rad
    )
