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
    """

    confirm_count: int = 2
    min_interval_sec: float = 1.0
    require_interlock: bool = True

    def __post_init__(self) -> None:
        if self.confirm_count < 1:
            raise ValueError(f"confirm_count must be >= 1, got {self.confirm_count}")
        if self.min_interval_sec < 0.0:
            raise ValueError(
                f"min_interval_sec must be >= 0, got {self.min_interval_sec}"
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


class GraspBoundaryDetector:
    """区間 1 → 2 の境界を検出する。

    `update()` を毎 tick 呼ぶ。True が返ったフレームで区間 2 へ進む。
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
        """
        self.cfg = cfg or BoundaryConfig()
        self.interlock_cfg = interlock_cfg or GraspInterlockConfig()
        self.vlm = vlm_client or VlmBoundaryClient(vlm_cfg)
        self.references = list(reference_images_b64)
        self.question = question
        self._hits = 0
        self._last_call_t: Optional[float] = None

    @property
    def hits(self) -> int:
        """現在の連続成立回数。"""
        return self._hits

    def reset(self) -> None:
        """カウントと呼び出し履歴を消す。エピソード境界で呼ぶ。"""
        self._hits = 0
        self._last_call_t = None

    def update(
        self,
        *,
        t: float,
        hand_state: Sequence[float],
        hand_cmd: Sequence[float],
        images_b64: Sequence[str],
        current_phase_text: str = "",
    ) -> BoundaryDecision:
        """1 tick ぶん判定する。

        Args:
            t: 現在時刻 [s] (単調増加すればよい)。
            hand_state / hand_cmd: (2,) Dex1 の実測 / 指令。
            images_b64: overlay 済みの [過去…, 現在] の base64 JPEG。
                参照画像はこの前に自動で連結される。
            current_phase_text: 現在フェーズの説明 (VLM への文脈)。

        Returns:
            `BoundaryDecision`。`fire=True` なら区間 2 へ進む。
        """
        out = BoundaryDecision()

        if self.cfg.require_interlock:
            out.interlock = is_grasping(
                hand_state, hand_cmd, RIGHT, self.interlock_cfg
            )
            if not out.interlock:
                # インターロックが落ちている間は VLM を呼ばない。
                # 呼ぶだけ無駄で、Thor 上で GR00T と資源を取り合うため。
                self._hits = 0
                out.hits = 0
                return out
        else:
            out.interlock = True

        if not self._due(t):
            # 間引かれた tick。カウントは進めない (据え置き)。
            out.hits = self._hits
            return out

        self._last_call_t = t
        out.vlm_called = True
        answer: VlmAnswer = self.vlm.ask(
            self.question,
            list(self.references) + list(images_b64),
            current_phase_text=current_phase_text,
        )
        out.vlm_value = answer.value
        out.latency_sec = answer.latency_sec
        out.error = answer.error

        # 判定不能 (None) は「まだ」として扱う (安全側、D-5)
        self._hits = self._hits + 1 if answer.is_yes else 0
        out.hits = self._hits
        out.fire = self._hits >= self.cfg.confirm_count
        return out

    def _due(self, t: float) -> bool:
        if self._last_call_t is None:
            return True
        return (t - self._last_call_t) >= self.cfg.min_interval_sec
