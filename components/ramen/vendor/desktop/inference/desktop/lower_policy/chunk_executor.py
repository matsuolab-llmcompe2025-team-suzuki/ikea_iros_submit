"""chunk 出力 policy の async 再計画 + temporal ensemble (Issue #139)。

`Gr00tPolicy.predict()` が持つ実行層 (Issue #128 Phase 2/4、#137 で hardening) を
model 非依存に切り出したもの。policy 側は「その tick の観測で推論して絶対値
chunk を返す関数」を渡すだけで、毎 tick blend 済みの目標 1 行を受け取る。

# 振る舞い (GR00T 版と同一)

- `replan_family=None`: 毎 tick 同期推論 → ensembler に投入 → その tick の目標。
- `replan_family` 設定時:
    - 初回 tick だけ同期推論して seed chunk を得る。
    - `execution_steps` ごとに 1 本の推論を別 thread で先行 submit し、chunk
      境界で promote する。promote 時は submit から経過した tick 分の先頭を飛ばす。
    - 境界に間に合わなければ直前の目標をそのまま保持する (30Hz の制御 thread
      で同期推論に落とさない。RAMEN-Ori 版の sync_fallback とはここが違う)。
- 目標は常に TargetTemporalEnsembler の blend 結果 (`temporal_lambda=None` なら
  最新 chunk)。blend は絶対値空間で行うので、渡す chunk は絶対値であること。

# 推論が止まった時 (GR00T 版から追加)

TE には native horizon 全体 (ACT は 100 行) を入れるので、新しい chunk が届かなくても
候補はしばらく尽きない。そのままだと推論が止まった後も古い観測で立てた計画を最大
約 3 s open-loop で実行し続ける。そこで、最新 chunk の観測時刻 (origin) から
「再計画 1 周期 + lead + max_age」を超えたら blend をやめて直前の目標を保持し、
保持が `max_hold_s` 続いたら例外にする (run_skill が腕を controlled release する)。

WHY: GR00T / RAMEN-Ori は実機検証済みの自前実装のまま残している。次にどちらかの
async 経路を直す時に本 module へ移す (Issue #139 で決定)。
"""

from __future__ import annotations

import math
import sys
from typing import Callable

import numpy as np

from inference.desktop.lower_policy.async_replanning import (
    AsyncActionChunkPipeline,
    family_replanning_schedule,
)
from model.subtask_policy_training.gr00t.temporal_ensemble import (
    TargetTemporalEnsembler,
)


# その tick の観測を閉じ込めた推論関数。(T, action_dim) の絶対値 chunk と latency_ms。
PredictChunk = Callable[[], tuple[np.ndarray, float]]


def _seconds_to_ticks(seconds: float, control_hz: float) -> int:
    # 0.30 s × 30 Hz が 9.000000000000002 になって 10 tick に切り上がらないよう丸めてから
    return math.ceil(round(seconds * control_hz, 6))


class ChunkExecutor:
    """chunk policy の実行層 (async 再計画 + temporal ensemble + 遅延時の保持)。

    Args:
        action_dim: chunk 1 行の次元 (VlaSkill 契約なら 19)。
        temporal_lambda: ensemble の decay 係数。None なら blend せず最新 chunk。
        replan_family: `FAMILY_REPLANNING_PROFILES` の key。None なら毎 tick 同期推論。
        execution_steps: 1 chunk を消費する tick 数 (= 再計画の周期)。
        thread_name_prefix: async 推論 thread と warning の名前 (診断用)。
        control_hz: step() を呼ぶ周期。max_age / max_hold_s を tick に直すのに使う。
        max_hold_s: 推論が止まって目標を保持し続けてよい時間。超えたら例外。
    """

    def __init__(
        self,
        *,
        action_dim: int,
        temporal_lambda: float | None,
        replan_family: str | None,
        execution_steps: int,
        thread_name_prefix: str,
        control_hz: float = 30.0,
        max_hold_s: float = 1.0,
    ) -> None:
        # 最新 chunk の origin からこの tick 数を超えたら、推論が止まったとみなして保持する。
        # 正常時の最大の開き = 再計画 1 周期 + lead、そこへ遅延の許容 max_age を足す。
        self._fresh_limit_ticks: int | None = None
        if replan_family is not None:
            # family の typo を初回 tick (= 腕の制御開始後) ではなく生成時に落とす。
            replan_after, max_age = family_replanning_schedule(replan_family, execution_steps)
            lead_steps = execution_steps - replan_after
            self._fresh_limit_ticks = (
                execution_steps + lead_steps + _seconds_to_ticks(max_age, control_hz)
            )
        self._max_hold_ticks = _seconds_to_ticks(max_hold_s, control_hz)
        self._replan_family = replan_family
        self._execution_steps = execution_steps
        self._thread_name_prefix = thread_name_prefix
        self._ensembler = TargetTemporalEnsembler(
            dim=action_dim, decay_lambda=temporal_lambda
        )
        self._pipeline: AsyncActionChunkPipeline | None = None
        self._current_step = 0
        self._pending_submit_step: int | None = None
        self._last_emitted_target: np.ndarray | None = None
        self._async_hold_ticks = 0
        self._consecutive_hold_ticks = 0
        # 最後に TE へ入れた chunk の origin (その chunk を推論した観測の tick)
        self._fresh_origin_step: int | None = None
        # async thread からも書く。float の代入は atomic なので lock は不要。
        self._last_predict_latency_ms: float | None = None

    def step(self, predict_chunk: PredictChunk) -> tuple[np.ndarray, dict]:
        """1 tick 進めて、この tick に送る目標を返す。

        Args:
            predict_chunk: この tick の観測で推論する関数。同期で呼ぶか async
                thread に渡すか、呼ばないかは本 class が決める。

        Returns:
            (target, metadata)。target は (action_dim,) float64。metadata の key は
            Gr00tPolicy と同じ (実機 log の集計をそのまま使うため)。

        Raises:
            RuntimeError: 保持できる目標がまだ無い (初回推論が空 chunk を返した)、
                または推論が止まって保持が `max_hold_s` を超えた。
            その他: 同期推論の例外と、async 推論の例外 (promote 時に再送出)。
        """
        pipeline_index: int | None = None
        if self._replan_family is None or self._pipeline is None:
            chunk, latency_ms = predict_chunk()
            self._last_predict_latency_ms = float(latency_ms)
            self._ensembler.add_chunk(
                origin_step=self._current_step, absolute_targets=chunk
            )
            self._fresh_origin_step = self._current_step
            if self._replan_family is not None:
                self._init_pipeline(np.asarray(chunk))
            chunk_source = "sync"
        else:
            had_pending = self._pipeline.prediction_pending
            promoted = self._pipeline.promote_if_ready()
            if promoted is not None and self._pending_submit_step is not None:
                submit_step = self._pending_submit_step
                # TE には native horizon 全体を渡す。execution_steps は再計画の周期で
                # あって、遅れて届いた chunk が重なって blend できる範囲ではない。
                self._ensembler.add_chunk(
                    origin_step=submit_step, absolute_targets=promoted.actions
                )
                self._fresh_origin_step = submit_step
                # chunk は submit 時刻基準。promote までに過ぎた先頭は実行しない。
                self._pipeline.skip_consumed_prefix(
                    max(0, self._current_step - submit_step)
                )
                self._pending_submit_step = None
                chunk_source = "async_promoted"
            else:
                chunk_source = "async_none_this_tick"
                # 期限切れで捨てられると pipeline 側の pending が消える。こちらの
                # latch も外して次の submit を即出せるようにする。
                if had_pending and not self._pipeline.prediction_pending:
                    self._pending_submit_step = None
            # pipeline の index を進める (wants_prediction の判定に要る)
            _, pipeline_index = self._pipeline.next_action()
            if self._pipeline.wants_prediction and self._pending_submit_step is None:
                self._submit(predict_chunk)

        candidate_count = self._ensembler.candidate_count(self._current_step)
        fresh_chunk_age_ticks = (
            self._current_step - self._fresh_origin_step
            if self._fresh_origin_step is not None
            else None
        )
        stalled = (
            self._fresh_limit_ticks is not None
            and fresh_chunk_age_ticks is not None
            and fresh_chunk_age_ticks > self._fresh_limit_ticks
        )
        if candidate_count and not stalled:
            target = self._ensembler.target(step=self._current_step)
            self._last_emitted_target = target.copy()
            self._consecutive_hold_ticks = 0
        elif self._last_emitted_target is not None:
            # 推論が chunk 境界に間に合わない / 止まった。制御 thread で同期推論はせず、
            # 直前の目標を正確に保持して arm_sdk への指令を途切れさせない。
            target = self._last_emitted_target.copy()
            self._async_hold_ticks += 1
            self._consecutive_hold_ticks += 1
            chunk_source = "async_hold"
            if self._consecutive_hold_ticks > self._max_hold_ticks:
                raise RuntimeError(
                    f"[{self._thread_name_prefix}] inference stalled: no fresh chunk for "
                    f"{fresh_chunk_age_ticks} ticks, held the last target for "
                    f"{self._consecutive_hold_ticks} ticks"
                )
        else:
            raise RuntimeError("chunk executor has no target to hold")
        self._current_step += 1

        pipeline = self._pipeline
        return target, {
            "blended_from_n_candidates": int(candidate_count),
            "temporal_lambda": self._ensembler.decay_lambda,
            "replan_family": self._replan_family,
            "chunk_source": chunk_source,
            "pipeline_index": pipeline_index,
            "pending_submit_step": self._pending_submit_step,
            "async_hold_ticks": self._async_hold_ticks,
            "fresh_chunk_age_ticks": fresh_chunk_age_ticks,
            "async_deadline_miss_ticks": (
                int(pipeline.deadline_miss_ticks) if pipeline is not None else None
            ),
            "async_stale_discard_count": (
                int(pipeline.stale_discard_count) if pipeline is not None else None
            ),
            "async_last_stale_discard_age_ms": (
                pipeline.last_stale_discard_age_ms if pipeline is not None else None
            ),
            "predict_latency_ms": self._last_predict_latency_ms,
        }

    def reset(self) -> None:
        """skill 遷移 / episode 開始時に呼ぶ。前 skill の chunk を新 skill に混ぜない。"""
        self._ensembler.reset()
        self._current_step = 0
        if self._pipeline is not None:
            # bounded close。pending 推論が 0.5s で終わらなければ daemon thread の
            # まま残し (process 終了で消える)、その例外も新 skill には持ち込まない。
            try:
                self._pipeline.close(timeout_s=0.5)
            except Exception:
                pass
            self._pipeline = None
        self._pending_submit_step = None
        self._last_emitted_target = None
        self._async_hold_ticks = 0
        self._consecutive_hold_ticks = 0
        self._fresh_origin_step = None
        self._last_predict_latency_ms = None

    def close(self, abort_pending: Callable[[], None] | None = None) -> None:
        """pipeline を bounded shutdown する。

        Args:
            abort_pending: 0.5s で推論が終わらない時に呼ぶ callback。worker
                process を落として pipe の read を解く用途。
        """
        if self._pipeline is not None:
            self._pipeline.close(timeout_s=0.5, abort_pending=abort_pending)
            self._pipeline = None

    def _init_pipeline(self, seed_chunk: np.ndarray) -> None:
        """初回の同期 seed chunk から AsyncActionChunkPipeline を作る。"""
        replan_after, max_age = family_replanning_schedule(
            self._replan_family, self._execution_steps
        )
        chunk_len = int(seed_chunk.shape[0])
        if (
            self._ensembler.decay_lambda is not None
            and self._execution_steps >= chunk_len
        ):
            # 重なり = chunk_len - execution_steps。0 だと候補が常に 1 個で blend が
            # 一度も起きず、chunk 境界の不連続がそのまま腕に出る (#137 の実測)。
            print(
                f"[{self._thread_name_prefix}] WARNING: execution_steps="
                f"{self._execution_steps} >= chunk_len={chunk_len}: consecutive "
                "chunks never overlap, so the temporal ensemble can never blend.",
                file=sys.stderr,
            )
        self._pipeline = AsyncActionChunkPipeline(
            initial_actions=np.asarray(seed_chunk, dtype=np.float64),
            execution_steps=self._execution_steps,
            replan_after_steps=replan_after,
            max_prediction_age_s=max_age,
            thread_name_prefix=self._thread_name_prefix,
        )
        # seed の row 0 はこの tick で送る。pipeline の時計を tick 0 から揃える。
        self._pipeline.skip_consumed_prefix(1)

    def _submit(self, predict_chunk: PredictChunk) -> None:
        """この tick の観測で async 推論を 1 本出す。"""
        submit_step = self._current_step

        def predictor() -> tuple[np.ndarray, float, dict]:
            chunk, latency_ms = predict_chunk()
            self._last_predict_latency_ms = float(latency_ms)
            return (
                np.asarray(chunk, dtype=np.float64),
                float(latency_ms),
                {"submit_step": submit_step},
            )

        self._pipeline.submit(predictor, anchor_generation=(submit_step,))
        self._pending_submit_step = submit_step
