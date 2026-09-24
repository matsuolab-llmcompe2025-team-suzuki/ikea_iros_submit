"""区間 1 → 2 の境界検出 (Issue #136、determinded.md D-9)。

    遷移 = 把持インターロック AND VLM が 1

インターロックは「グリッパが何かを噛んで止まった」しか言えず**それが脚かは
言えない**。VLM は「脚を掴んでいるように見える」は言えるが**実際に噛んで
いるかは見えない**。互いの弱点を埋め合う。

# 連続確認

ポーリング + 後戻りなしは、平均ではなく**最大**を取る。1 回でも 1 が出たら
確定で進むので、ポーリング回数だけ誤発火が積み上がる。そこで
「両方が `confirm_count` 回続けて立ったら進む」にする。途中で 1 回でも
外れたらカウントは 0 に戻る。`confirm_count = 1` で無効化できる。

代償は待ち時間だけ (1 Hz なら 1 回ぶん = 1 秒)。実測では区間 2 の長さが
median 2.70 秒なので、1 秒の遅れは許容範囲。

# VLM を呼ぶ頻度

制御ループは 30 Hz 程度で回るが、VLM はそんなに速く応答しない。
`min_interval_sec` で間引き、**呼ばなかった tick は前回の答えを再利用しない**
(カウントも進めない)。つまり `confirm_count` は「VLM を呼んだ回数」で数える。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from inference.desktop.pick_leg_hybrid.interlock import (
    RIGHT,
    GraspInterlockConfig,
    is_grasping,
)
from inference.desktop.pick_leg_hybrid.vlm import (
    GRASP_QUESTION,
    VlmAnswer,
    VlmBoundaryClient,
    VlmConfig,
)


@dataclass(frozen=True)
class BoundaryConfig:
    """1→2 境界検出の設定。

    Attributes:
        confirm_count: 何回続けて成立したら進むか。1 で連続確認を無効化。
        min_interval_sec: VLM 呼び出しの最短間隔 [s]。
        require_interlock: インターロックを AND 条件に含めるか。
            False にすると VLM 単独判定 (評価用、実機では非推奨)。
        max_answer_age_sec: 非同期の VLM 応答を採用してよい、問い合わせた観測からの
            経過 [s]。これより古い応答は捨てる (Codex #8)。
    """

    confirm_count: int = 2
    min_interval_sec: float = 1.0
    require_interlock: bool = True
    max_answer_age_sec: float = 6.0

    def __post_init__(self) -> None:
        if self.confirm_count < 1:
            raise ValueError(f"confirm_count must be >= 1, got {self.confirm_count}")
        if self.min_interval_sec < 0.0:
            raise ValueError(
                f"min_interval_sec must be >= 0, got {self.min_interval_sec}"
            )
        if not self.max_answer_age_sec > 0.0:
            raise ValueError(
                f"max_answer_age_sec must be > 0, got {self.max_answer_age_sec}"
            )


@dataclass
class BoundaryDecision:
    """1 tick ぶんの判定結果 (log にそのまま流せる)。"""

    fire: bool = False
    interlock: bool = False
    vlm_called: bool = False
    vlm_value: Optional[int] = None
    hits: int = 0
    latency_sec: float = 0.0
    error: str = ""
    # 非同期モード: この tick に問い合わせを出したか / 応答待ちか / 捨てた応答。
    vlm_submitted: bool = False
    vlm_pending: bool = False
    vlm_discarded: str = ""


class _PendingQuery:
    """1 本だけ走らせる VLM 問い合わせ (daemon thread)。"""

    def __init__(
        self,
        fn: Callable[[], VlmAnswer],
        *,
        observed_t: float,
        generation: int,
    ) -> None:
        import threading

        self.observed_t = float(observed_t)
        self.generation = int(generation)
        self.answer: Optional[VlmAnswer] = None
        self.error: Optional[BaseException] = None
        self._done = threading.Event()

        def _run() -> None:
            try:
                self.answer = fn()
            except BaseException as exc:  # noqa: BLE001 - 判定不能として扱う
                self.error = exc
            finally:
                self._done.set()

        threading.Thread(target=_run, name="vlm-boundary", daemon=True).start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)


class GraspBoundaryDetector:
    """区間 1 → 2 の境界を検出する。

    `update()` を毎 tick 呼ぶ。True が返ったフレームで区間 2 へ進む。

    ``async_vlm=True`` (実機の既定) では VLM を別 thread で問い合わせ、応答を
    待つ間も `update()` は即座に戻る (Codex #8)。以前は制御 tick の中で HTTP
    応答 (最大 ``timeout_sec``) を待ち、その間の指令更新と安全監視が止まっていた。
    応答には「問い合わせた観測の時刻」と「世代」を付け、インターロックが外れた・
    reset された後の応答、``max_answer_age_sec`` より古い応答は捨てる。
    """

    def __init__(
        self,
        *,
        cfg: Optional[BoundaryConfig] = None,
        interlock_cfg: Optional[GraspInterlockConfig] = None,
        vlm_cfg: Optional[VlmConfig] = None,
        vlm_client: Optional[VlmBoundaryClient] = None,
        reference_images_b64: Sequence[str] = (),
        question: str = GRASP_QUESTION,
        async_vlm: bool = False,
    ) -> None:
        """
        Args:
            cfg: 境界検出の設定。
            interlock_cfg: 把持インターロックの閾値。
            vlm_cfg: VLM 設定 (`vlm_client` を渡す場合は無視される)。
            vlm_client: VLM クライアント。None なら `vlm_cfg` から作る。
            reference_images_b64: 参照画像 [A(遷移前), B(遷移後)]。
                overlay 済みのものを渡すこと (D-7)。
            question: 質問文。
            async_vlm: True なら VLM を別 thread で問い合わせる (制御を止めない)。
        """
        self.cfg = cfg or BoundaryConfig()
        self.interlock_cfg = interlock_cfg or GraspInterlockConfig()
        self.vlm = vlm_client or VlmBoundaryClient(vlm_cfg)
        self.references = list(reference_images_b64)
        self.question = question
        self.async_vlm = bool(async_vlm)
        self._hits = 0
        self._last_call_t: Optional[float] = None
        self._generation = 0
        self._pending: Optional[_PendingQuery] = None

    @property
    def hits(self) -> int:
        """現在の連続成立回数。"""
        return self._hits

    @property
    def pending(self) -> bool:
        """応答待ちの問い合わせがあるか。"""
        return self._pending is not None

    def reset(self) -> None:
        """カウントと呼び出し履歴を消す。エピソード境界で呼ぶ。

        走っている問い合わせは世代を進めて無効にする (応答が来ても捨てる)。
        """
        self._hits = 0
        self._last_call_t = None
        self._generation += 1
        self._pending = None

    def update(
        self,
        *,
        t: float,
        hand_state: Sequence[float],
        hand_cmd: Sequence[float],
        images_b64: "Sequence[str] | Callable[[], Sequence[str]]",
        current_phase_text: str = "",
        interlock: Optional[bool] = None,
    ) -> BoundaryDecision:
        """1 tick ぶん判定する。

        Args:
            t: 現在時刻 [s] (単調増加すればよい)。
            hand_state / hand_cmd: (2,) Dex1 の実測 / 指令。
            images_b64: overlay 済みの [過去…, 現在] の base64 JPEG、または
                それを返す callable (問い合わせを出す tick だけ呼ばれる)。
                参照画像はこの前に自動で連結される。
            current_phase_text: 現在フェーズの説明 (VLM への文脈)。
            interlock: 呼び出し側が判定したインターロック。None なら
                ``hand_state`` と ``hand_cmd`` の実測差で判定する。手の実測が無い
                経路 (合成 state) では差が常に 0 なので、呼び出し側が別の条件で
                渡す (Codex #2)。

        Returns:
            `BoundaryDecision`。`fire=True` なら区間 2 へ進む。
        """
        out = BoundaryDecision()

        if self.cfg.require_interlock:
            out.interlock = (
                is_grasping(hand_state, hand_cmd, RIGHT, self.interlock_cfg)
                if interlock is None
                else bool(interlock)
            )
            if not out.interlock:
                # インターロックが落ちている間は VLM を呼ばない。
                # 呼ぶだけ無駄で、Thor 上で GR00T と資源を取り合うため。
                # 走っている問い合わせは古い把持についての答えなので捨てる。
                if self._hits or self._pending is not None:
                    self._generation += 1
                self._pending = None
                self._hits = 0
                out.hits = 0
                return out
        else:
            out.interlock = True

        if self.async_vlm:
            return self._update_async(out, t, images_b64, current_phase_text)

        if not self._due(t):
            # 間引かれた tick。カウントは進めない (据え置き)。
            out.hits = self._hits
            return out

        self._last_call_t = t
        out.vlm_called = True
        answer: VlmAnswer = self.vlm.ask(
            self.question,
            list(self.references) + list(_images(images_b64)),
            current_phase_text=current_phase_text,
        )
        self._apply(out, answer)
        return out

    def _update_async(
        self,
        out: BoundaryDecision,
        t: float,
        images_b64: "Sequence[str] | Callable[[], Sequence[str]]",
        current_phase_text: str,
    ) -> BoundaryDecision:
        pending = self._pending
        if pending is not None and pending.done:
            self._pending = None
            if pending.generation != self._generation:
                out.vlm_discarded = "superseded"
            elif t - pending.observed_t > self.cfg.max_answer_age_sec:
                out.vlm_discarded = (
                    f"stale ({t - pending.observed_t:.2f}s > "
                    f"{self.cfg.max_answer_age_sec:g}s)"
                )
                self._hits = 0
            else:
                answer = pending.answer
                if answer is None:
                    answer = VlmAnswer(
                        value=None,
                        raw="",
                        latency_sec=0.0,
                        error=repr(pending.error),
                    )
                out.vlm_called = True
                self._apply(out, answer)
                return out
        if self._pending is None and self._due(t):
            self._last_call_t = t
            images = list(self.references) + list(_images(images_b64))
            generation = self._generation
            self._pending = _PendingQuery(
                lambda: self.vlm.ask(
                    self.question, images, current_phase_text=current_phase_text
                ),
                observed_t=t,
                generation=generation,
            )
            out.vlm_submitted = True
        out.vlm_pending = self._pending is not None
        out.hits = self._hits
        return out

    def _apply(self, out: BoundaryDecision, answer: VlmAnswer) -> None:
        out.vlm_value = answer.value
        out.latency_sec = answer.latency_sec
        out.error = answer.error
        # 判定不能 (None) は「まだ」として扱う (安全側、D-5)
        self._hits = self._hits + 1 if answer.is_yes else 0
        out.hits = self._hits
        out.fire = self._hits >= self.cfg.confirm_count

    def _due(self, t: float) -> bool:
        if self._last_call_t is None:
            return True
        return (t - self._last_call_t) >= self.cfg.min_interval_sec


def _images(images_b64: "Sequence[str] | Callable[[], Sequence[str]]") -> Sequence[str]:
    return images_b64() if callable(images_b64) else images_b64
