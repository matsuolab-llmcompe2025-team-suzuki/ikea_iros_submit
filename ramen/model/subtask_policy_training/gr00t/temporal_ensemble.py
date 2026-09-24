"""Timestamp-aligned temporal ensembling for chunk-based policy outputs.

Ported from issue-70-flip-table-data-augmentation branch (2026-08-30、Phase 2)
and adapted from `PhysicalTargetTemporalEnsembler` (16-D、arms14 + dex1[2])
to a generic-dim ensembler. Reason: our VlaSkill contract is 19-D
(waist3 + arms14 + hand2)、waist は Gr00tPolicy Design A (統一 19D blend) で
arm/hand と同 ensembler で処理する。詳細な設計判断は Issue #128 の Phase 2
kickoff メモ参照。

# 前提

- adapter (`slice_53d_to_19d`) が RELATIVE→ABSOLUTE 復元済 physical target
  space の 19D chunk を渡す前提。issue-70 の
  `logical_chunk_to_physical_targets` に相当する 53→19 変換は Gr00tPolicy 側
  で先行実行。
- Blend は physical target space (absolute) で行う。RELATIVE space での blend
  は origin_step が違う delta を平均する = 意味を持たないため NG。
- Weight = `exp(decay_lambda * age_in_steps)`、`decay_lambda=-0.1` 想定
  (issue-70 検証値、age=5 で weight~0.6 / age=10 で weight~0.37)。
- `decay_lambda=None` = 単純に最新 candidate のみ返す (blend disable、debug 用)。

# 使い方

    ensembler = TargetTemporalEnsembler(dim=19, decay_lambda=-0.1)
    # tick 毎: chunk (chunk_len, 19) を投入、current step の blended target 取得
    ensembler.add_chunk(origin_step=t, absolute_targets=chunk_19d)
    blended = ensembler.target(step=t)  # (19,)
    # 次 tick t+1 では別 chunk が来て step t+1 の candidate に加算、
    # step t+1 の target = 昔の chunk(offset=1) + 新 chunk(offset=0) の
    # weighted mean
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class _Candidate:
    origin_step: int
    target: np.ndarray


class TargetTemporalEnsembler:
    """Blend overlapping chunk predictions at a given physical target step.

    Attributes:
        dim: 各 target vector の次元 (VlaSkill 19D なら 19)。add_chunk / target で
            この dim を strict 検証する。
        decay_lambda: age に応じた exp decay 係数、None なら最新 candidate 返す。
    """

    def __init__(self, *, dim: int, decay_lambda: float | None = -0.1) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        if decay_lambda is not None and not math.isfinite(float(decay_lambda)):
            raise ValueError("decay_lambda must be finite or None")
        self.dim = int(dim)
        self.decay_lambda = None if decay_lambda is None else float(decay_lambda)
        self._candidates: dict[int, list[_Candidate]] = {}

    def reset(self) -> None:
        """Skill 遷移 / episode 開始時に呼ぶ。全 candidate を破棄。"""
        self._candidates.clear()

    def add_chunk(
        self,
        *,
        origin_step: int,
        absolute_targets: Sequence[Sequence[float]],
    ) -> None:
        """chunk (chunk_len, dim) を candidate として登録する。

        Args:
            origin_step: この chunk が予測を開始した step index (通常は
                inference が呼ばれた tick 番号)。chunk[offset=0] が step
                origin_step を予測、chunk[offset=k] が step origin_step+k を
                予測。
            absolute_targets: (chunk_len, dim) shape の action。physical target
                space の absolute value である前提 (RELATIVE space の delta
                だと blend が無意味)。
        """
        targets = np.asarray(absolute_targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != self.dim:
            raise ValueError(
                f"absolute_targets must have shape [T, {self.dim}], "
                f"got {targets.shape}"
            )
        if not np.isfinite(targets).all():
            raise ValueError("absolute_targets contains NaN or Inf")
        for offset, target in enumerate(targets):
            step = int(origin_step) + offset
            self._candidates.setdefault(step, []).append(
                _Candidate(origin_step=int(origin_step), target=target.copy())
            )

    def target(self, step: int) -> np.ndarray:
        """指定 step の blended target を返す。

        Blending rule (candidates が複数の場合):
            - decay_lambda is None: origin_step 最大 (= 最新) を採用、blend しない
            - それ以外: weight[i] = exp(decay_lambda * (step - candidate[i].origin))
              の weighted mean。古い chunk ほど weight が小さくなる。

        Args:
            step: 取得したい step index。add_chunk で登録されてない step は
                KeyError。

        Returns:
            (dim,) float64、blended target。

        Note:
            呼び出し後、step より小さい全 step の candidate は破棄される
            (メモリ増加防止)。同 step の再問い合わせは可能。
        """
        step = int(step)
        candidates = self._candidates.get(step, [])
        if not candidates:
            raise KeyError(f"no candidate for step {step}")
        if self.decay_lambda is None:
            result = max(
                candidates, key=lambda candidate: candidate.origin_step
            ).target.copy()
        else:
            weights = np.asarray(
                [
                    math.exp(
                        self.decay_lambda * max(0, step - candidate.origin_step)
                    )
                    for candidate in candidates
                ],
                dtype=np.float64,
            )
            stacked = np.stack(
                [candidate.target for candidate in candidates], axis=0
            )
            result = np.average(stacked, axis=0, weights=weights)
        self._discard_before(step)
        return result

    def candidate_count(self, step: int) -> int:
        """指定 step に登録されている candidate 数 (debug / test 用)。"""
        return len(self._candidates.get(int(step), []))

    def _discard_before(self, step: int) -> None:
        """指定 step 未満の全 candidate を破棄 (メモリ枠固定)。"""
        for old_step in [value for value in self._candidates if value < step]:
            del self._candidates[old_step]
