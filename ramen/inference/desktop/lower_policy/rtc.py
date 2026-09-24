"""Real-Time Chunking (RTC) の policy 非依存な共有部品 (Issue #137 Phase A)。

chunk 単位で action を出す policy は、新 chunk を推論している間も旧 chunk を
実行し続ける。推論が終わった時点で新 chunk の先頭 `d` 本は **既に送信済み** で
書き換えられない。この境界で新旧 chunk が食い違うと手先が振動する
(実機 1.5-2.0Hz / 7-35mm、Issue #137 で計測)。

RTC はこれを inpainting として解き、denoising の速度場を step ごとに
スケールすることで「確定済みの先頭は凍結、その先は前 chunk から徐々に離す」
という制約を入れる。学習は不要で推論時のみの手法。

# 本 module の位置づけ

`async_replanning.py` と同じく **model を一切知らない** 層。GR00T / RAMEN-Ori の
両方から使えるように、config・遅延推定・leftover 保持・重みベクトル生成だけを
持つ。実際の適用経路は policy ごとに非対称:

- **GR00T**: 重みの適用は我々ではなく LeRobot (`policies/groot/groot_n1_7.py`)
  が行う。我々は re-anchor した prefix と frozen 数を渡すだけ。
- **RAMEN-Ori absolute action**: sampler
  (`model/ramen_ori/action_expert.py: sample_action`) が自前なので、
  `build_velocity_strength()` の出力を Euler ループに直接掛ける。
- **RAMEN-Ori relative action**: normalized delta-q を時系列に累積するため、soft
  ramp の誤差も後続 row へ累積する。Issue #137 B-3 実機試験で発散を確認した
  ため禁止する。async replanning + temporal ensemble のみ使用する。

# 重み式について (重要な設計判断)

`build_velocity_strength()` は **本家 GR00T の式を移植したもの** で、RTC 論文
(arXiv:2506.07339) の式 5 (ΠGDM soft mask) ではない。理由は、GR00T 経路では
LeRobot 内部の実装が使われるため、我々が論文式を採ると **同じ YAML 設定でも
GR00T と RAMEN-Ori の挙動が別物になり比較が成立しない** から。移植元は
`lerobot/policies/groot/groot_n1_7.py` の RTC 分岐 (`vel_strength` 構築部)。
一致は `tests/test_rtc.py` の parity test で機械的に保証している。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


# `RtcConfig.frozen_steps` に指定できる自動推定 sentinel。
AUTO_FROZEN_STEPS: Literal["auto"] = "auto"

# 本家 GR00T の既定 ramp rate (`groot_n1_7.py` の GR00T N1.7 default config)。
DEFAULT_RAMP_RATE: float = 6.0

# ramp 正規化時の 0 除算回避 (本家 `ramp[-1].clamp_min(1e-8)` と同値)。
_RAMP_NORM_EPS: float = 1e-8


@dataclass(frozen=True)
class RtcConfig:
    """policy variant ごとの RTC 設定 (`policy_config.yaml` の `rtc:` ブロック)。

    Attributes:
        enabled: RTC を使うか。**既定 False** = 従来経路そのまま (回帰ゼロ)。
        frozen_steps: 凍結する先頭 step 数 = 推論遅延 `d` の tick 数。
            `"auto"` (既定) なら `DelayEstimator` が実測 latency と実測 tick
            周期から毎回算出する。int を書けばその値に固定 (再現実験用)。
            実測値は GR00T 85-98ms / 33.3ms tick → 3、RAMEN-Ori 51-80ms /
            61.9ms tick → 1-2 (Issue #137、wandb 実機 run より)。
        overlap_steps: 前 chunk を参照する先頭 step 数。`None` (既定) なら
            `PolicyConfig.execution_steps` を流用する。実行時は
            `min(len(leftover), overlap_steps)` で必ずクランプされる。
        ramp_rate: frozen 区間の外側で前 chunk から離れていく指数 ramp の rate。
            大きいほど速く自由になる。既定は本家 GR00T と同じ 6.0。
    """

    enabled: bool = False
    frozen_steps: int | Literal["auto"] = AUTO_FROZEN_STEPS
    overlap_steps: int | None = None
    ramp_rate: float = DEFAULT_RAMP_RATE
    # RAMEN-Ori relative action は normalized delta-q の累積表現であり、B-3
    # 実機試験で RTC soft-prefix の発散を確認済み。通常は fail-closed にし、
    # 原因比較を行う隔離された実験 slot だけが明示的に解除できるようにする。
    allow_experimental_relative_action: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if not isinstance(self.allow_experimental_relative_action, bool):
            raise TypeError(
                "allow_experimental_relative_action must be bool, got "
                f"{type(self.allow_experimental_relative_action).__name__}"
            )
        if self.frozen_steps != AUTO_FROZEN_STEPS:
            if isinstance(self.frozen_steps, bool) or not isinstance(
                self.frozen_steps, int
            ):
                raise TypeError(
                    f"frozen_steps must be int or {AUTO_FROZEN_STEPS!r}, "
                    f"got {self.frozen_steps!r}"
                )
            if self.frozen_steps < 0:
                raise ValueError(
                    f"frozen_steps must be >= 0, got {self.frozen_steps}"
                )
        if self.overlap_steps is not None:
            if isinstance(self.overlap_steps, bool) or not isinstance(
                self.overlap_steps, int
            ):
                raise TypeError(
                    f"overlap_steps must be int or None, got {self.overlap_steps!r}"
                )
            if self.overlap_steps < 1:
                raise ValueError(
                    f"overlap_steps must be >= 1, got {self.overlap_steps}"
                )
        if not math.isfinite(float(self.ramp_rate)) or float(self.ramp_rate) <= 0.0:
            raise ValueError(
                f"ramp_rate must be a finite positive float, got {self.ramp_rate!r}"
            )
        # 両方 int 指定のときだけ相対関係を静的に検証できる ("auto" は実行時決定、
        # overlap=None は execution_steps 依存なので policy 側で最終検証する)。
        if (
            self.frozen_steps != AUTO_FROZEN_STEPS
            and self.overlap_steps is not None
            and self.frozen_steps > self.overlap_steps
        ):
            raise ValueError(
                f"frozen_steps ({self.frozen_steps}) must be <= overlap_steps "
                f"({self.overlap_steps})"
            )


def build_velocity_strength(
    *,
    chunk_len: int,
    frozen_steps: int,
    overlap_steps: int,
    ramp_rate: float = DEFAULT_RAMP_RATE,
) -> np.ndarray:
    """denoising の速度場に掛ける (chunk_len,) の step 別スケールを返す。

    区間は 3 つ:

    - ``[0, frozen_steps)``   → 0.0。速度場が乗らない = 前 chunk の値のまま凍結。
      推論中に既に送信済みの step がここに当たる。
    - ``[frozen_steps, overlap_steps)`` → 0 から 1 への指数 ramp。前 chunk から
      徐々に離れる緩衝区間。
    - ``[overlap_steps, chunk_len)`` → 1.0。前 chunk と重ならないので自由生成。

    Args:
        chunk_len: policy の action chunk 長 (GR00T rotate/insert=16、
            furniture_groot=40)。
        frozen_steps: 凍結 step 数 (= 推論遅延 `d`)。0 なら凍結なし。
        overlap_steps: 前 chunk を参照する step 数。`frozen_steps` 以上、
            `chunk_len` 以下。
        ramp_rate: 指数 ramp の rate (正の有限値)。

    Returns:
        (chunk_len,) float32。呼出側が action 次元に broadcast して使う。

    Raises:
        ValueError: 引数が ``0 <= frozen_steps <= overlap_steps <= chunk_len``
            を満たさない、または ramp_rate が非正 / 非有限。
    """
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
    if frozen_steps < 0:
        raise ValueError(f"frozen_steps must be >= 0, got {frozen_steps}")
    if overlap_steps < frozen_steps:
        raise ValueError(
            f"overlap_steps ({overlap_steps}) must be >= frozen_steps "
            f"({frozen_steps})"
        )
    if overlap_steps > chunk_len:
        raise ValueError(
            f"overlap_steps ({overlap_steps}) must be <= chunk_len ({chunk_len})"
        )
    if not math.isfinite(float(ramp_rate)) or float(ramp_rate) <= 0.0:
        raise ValueError(f"ramp_rate must be finite positive, got {ramp_rate!r}")

    strength = np.ones(chunk_len, dtype=np.float32)
    strength[:frozen_steps] = 0.0
    # 本家と同じ構成: 端点 0.0 / 1.0 を含む linspace で ramp を作り、末尾で
    # 正規化してから両端を落とす。intermediate=0 なら ramp[1:-1] は空になり、
    # 代入も空 slice なので純粋なハードマスクとして成立する。
    intermediate_steps = overlap_steps - frozen_steps
    t = np.linspace(0.0, 1.0, intermediate_steps + 2, dtype=np.float64)
    ramp = 1.0 - np.exp(-float(ramp_rate) * t)
    ramp = ramp / max(float(ramp[-1]), _RAMP_NORM_EPS)
    strength[frozen_steps:overlap_steps] = ramp[1:-1].astype(np.float32, copy=False)
    return strength


class DelayEstimator:
    """推論 latency の sliding window から凍結 step 数を保守的に推定する。

    RTC 論文 (Algorithm 1) と同じく **window 内の max** を採る。`d` を過小評価
    すると「既に送信済みの step を書き換える」ことになり、まさに RTC が防ごうと
    している不連続が出る。過大評価は自由に書ける区間が減るだけで済むので、
    非対称なコストに合わせて保守側へ倒す。

    tick 周期は policy ごとに違う (実測 GR00T 30.0Hz / RAMEN-Ori 16.2Hz) ため
    config 定数にせず、`frozen_steps()` の呼出側が実測値を渡す。

    window は **8 サンプル**。latency は predict 1 回につき 1 サンプルで、async
    replan では 8 tick に 1 回しか predict しないため、8 サンプル ≈ 8 replan ≈
    2 秒分の履歴になる。32 にしていた時は ≈8.5 秒分となり、1 回の遅い推論が
    frozen を上げたまま長く戻らなかった (実 ckpt smoke で frozen 3→6 に張り付く
    のを実測)。論文の意図は「**直近の**遅延に対して保守的」なので、max を採る
    保守性は保ったまま外れ値からの復帰を速くする。
    """

    def __init__(self, *, window: int = 8) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._samples: deque[float] = deque(maxlen=int(window))

    def reset(self) -> None:
        """skill 遷移 / episode 開始時に呼ぶ。全 sample を破棄。"""
        self._samples.clear()

    def add(self, latency_s: float) -> None:
        """latency サンプルを 1 件追加 (秒)。非有限 / 負値は無視する。"""
        value = float(latency_s)
        if not math.isfinite(value) or value < 0.0:
            return
        self._samples.append(value)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def max_latency_s(self) -> float | None:
        """window 内の最大 latency (秒)。sample が無ければ None。"""
        return max(self._samples) if self._samples else None

    def frozen_steps(self, tick_period_s: float, *, cap: int) -> int | None:
        """実測 tick 周期から凍結 step 数を返す。sample が無ければ None。

        Args:
            tick_period_s: 実測の制御 tick 周期 (秒、正の有限値)。
            cap: 上限 (通常 `overlap_steps`)。0 以上。

        Returns:
            `ceil(max_latency / tick_period)` を `[0, cap]` にクランプした値。
            sample が 1 件も無い場合は None (呼出側は RTC を使わない)。

        Note:
            端数は切り上げる。latency が 2.6 tick なら新 chunk が載る前に
            3 tick 目の送信が始まっているため、2 では凍結が足りない。
        """
        if cap < 0:
            raise ValueError(f"cap must be >= 0, got {cap}")
        if not math.isfinite(float(tick_period_s)) or float(tick_period_s) <= 0.0:
            raise ValueError(
                f"tick_period_s must be finite positive, got {tick_period_s!r}"
            )
        latency = self.max_latency_s
        if latency is None:
            return None
        steps = math.ceil(latency / float(tick_period_s))
        return max(0, min(int(steps), int(cap)))


class ChunkLeftoverBuffer:
    """直前 chunk を保持し、未実行の残り (RTC prefix) を返す。

    保持する space は **policy の model action space** であること。GR00T なら
    53D 論理 action、RAMEN-Ori なら学習時の action 空間。VlaSkill 契約の 19D に
    落とした後の値を入れると、prefix をモデルに戻せなくなる。
    """

    def __init__(self, *, action_dim: int) -> None:
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        self.action_dim = int(action_dim)
        self._chunk: np.ndarray | None = None
        self._origin_step: int | None = None

    def reset(self) -> None:
        """skill 遷移 / episode 開始時に呼ぶ。保持中の chunk を破棄。"""
        self._chunk = None
        self._origin_step = None

    @property
    def origin_step(self) -> int | None:
        """保持中 chunk の起点 step。空なら None。"""
        return self._origin_step

    def store(self, chunk: np.ndarray, *, origin_step: int) -> None:
        """chunk を (chunk_len, action_dim) で保持する。既存分は置き換える。"""
        array = np.asarray(chunk)
        if array.ndim != 2 or array.shape[1] != self.action_dim:
            raise ValueError(
                f"chunk must be (chunk_len, {self.action_dim}), "
                f"got shape {array.shape}"
            )
        if array.shape[0] < 1:
            raise ValueError("chunk must contain at least one step")
        self._chunk = array.astype(np.float32, copy=True)
        self._origin_step = int(origin_step)

    def remaining(self, current_step: int) -> np.ndarray | None:
        """`current_step` 時点で未実行の残りを返す。

        Returns:
            (remaining_len, action_dim) float32 の copy。保持 chunk が無い /
            使い切っている場合は None。

        Raises:
            ValueError: `current_step` が保持 chunk の起点より前。
        """
        if self._chunk is None or self._origin_step is None:
            return None
        consumed = int(current_step) - self._origin_step
        if consumed < 0:
            raise ValueError(
                f"current_step ({current_step}) precedes stored origin_step "
                f"({self._origin_step})"
            )
        if consumed >= self._chunk.shape[0]:
            return None
        return self._chunk[consumed:].copy()


def validate_rtc_against_chunk_len(cfg, chunk_len: int) -> None:
    """RTC 設定を ckpt の chunk 長と突き合わせる。

    `overlap_steps` / `frozen_steps` が chunk 長を超えていないかは YAML 読み込み
    時点では判定できない (chunk_size は ckpt の config.json にしかない)。実行時に
    クランプされて黙って別の値で走るのを避けるため、config.json を読んだ直後
    = **重み load の前** に落とす。
    """
    rtc = cfg.rtc
    if not rtc.enabled:
        return
    if chunk_len < 1:
        raise ValueError(f"checkpoint reports a non-positive chunk_size: {chunk_len}")
    overlap = rtc.overlap_steps if rtc.overlap_steps is not None else cfg.execution_steps
    if overlap > chunk_len:
        raise ValueError(
            f"rtc.overlap_steps={overlap} exceeds the checkpoint chunk_size "
            f"({chunk_len}). Lower rtc.overlap_steps (or execution_steps when "
            f"overlap_steps is null) in policy_config.yaml."
        )
    if rtc.frozen_steps != AUTO_FROZEN_STEPS and rtc.frozen_steps > overlap:
        raise ValueError(
            f"rtc.frozen_steps={rtc.frozen_steps} exceeds the resolved "
            f"rtc.overlap_steps={overlap}"
        )
