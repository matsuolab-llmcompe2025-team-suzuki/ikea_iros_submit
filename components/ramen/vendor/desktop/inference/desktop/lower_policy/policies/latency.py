"""Latency budget constants + benchmark helpers (Issue #125 Phase 8)。

# Pipeline scheduling model

RAMEN-Ori / GR00T の action_expert は chunk_len=16 (0.53 秒 @ 30Hz) の action
chunk を 1 回の predict で先出しする。caller (orchestrator) は chunk 消費中に
次の predict を async で走らせ、YOLO も並行実行する = pipeline scheduling で
per-step tick (33ms) の壁を越える設計。

    time →   ┌────── tick 0 (33ms) ──────┐┌── tick 1 ──┐...
    action:  [use chunk[0]]                [use chunk[1]]  ...
    model:   [predict for next chunk...........................done]
    YOLO:    [detection for perception....................done]

Per-step CPU (obs 取得 + action pick from buffer + actuator send) は 33ms 予算に
収まる必要がある。Model forward + YOLO は 530ms (chunk_len × step_time) 予算で
並行実行。

# 予算値の位置付け

**Placeholder**: 本 module の budget 定数は理論値 (30Hz control + chunk_len 16
から derive)。実際の実機 (Orin / Sakura) 上での measurement で必要に応じて調整
する (Phase 8 は skeleton、実測は Phase 9 の entrypoint dry-run + real hardware
tuning で行う)。

# 使用例 (実機で real ckpt load 後)

```python
from inference.desktop.lower_policy.policies.latency import measure_predict_latency

# 5 iter avg で predict() の per-tick latency を測る
stats = measure_predict_latency(policy, sample_observation, n_iter=5)
print(f"mean={stats['mean_ms']:.1f} p95={stats['p95_ms']:.1f}")
```
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


# ---- Pipeline scheduling budget (Placeholder、実機で調整前提) ---- #

# 制御周期。G1 real robot は 30Hz が標準 (LeRobot G1_WBT dataset も 30fps)。
CONTROL_FPS: int = 30
STEP_TIME_MS: float = 1000.0 / CONTROL_FPS  # 33.33 ms

# Action chunk 消費期間。Model は次 chunk を chunk 消費期間内に予測完了させる必要。
# RAMEN-Ori chunk_len=16 / GR00T chunk_size=16、両方とも同じ。
ACTION_CHUNK_LEN: int = 16
MODEL_FORWARD_BUDGET_MS: float = STEP_TIME_MS * ACTION_CHUNK_LEN  # 533.33 ms

# YOLO は Model forward と並行実行、per-frame or on-demand。
# perception 側の schedule に依存するが、safe upper bound は chunk 消費期間相当。
YOLO_FORWARD_BUDGET_MS: float = MODEL_FORWARD_BUDGET_MS  # 533.33 ms (async 前提)

# Per-step CPU (obs 取得 + action pick + actuator send)。tick loop の壁時計制約。
# preprocess_frame / pack_obb_tokens 等の CPU work はここに含まれるが、実際は
# Model forward の tick に組み込まれ per-chunk 単位で 1 回だけ行う想定
# (Skill wrapper が buffer 保持)。
PER_STEP_CPU_BUDGET_MS: float = STEP_TIME_MS  # 33.33 ms


def measure_latency(
    fn,
    *args,
    n_iter: int = 5,
    warmup: int = 1,
    **kwargs,
) -> dict[str, float]:
    """任意 callable の wall-clock latency を n_iter 回測って統計を返す。

    Args:
        fn: 測定対象 callable。
        *args, **kwargs: fn に渡す引数。
        n_iter: 測定 iter 数 (default 5)。
        warmup: warmup iter 数 (計測に含めない、default 1)。JIT / cuDNN
                autotune の初期コストを外す。

    Returns:
        dict with keys: min_ms / mean_ms / p50_ms / p95_ms / max_ms / all_ms (list)。
    """
    # warmup (計測に含めない)
    for _ in range(warmup):
        fn(*args, **kwargs)

    times_ms: list[float] = []
    for _ in range(n_iter):
        t0 = time.monotonic_ns()
        fn(*args, **kwargs)
        t1 = time.monotonic_ns()
        times_ms.append((t1 - t0) / 1e6)

    arr = np.asarray(times_ms)
    return {
        "min_ms": float(arr.min()),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "all_ms": times_ms,
    }


def measure_predict_latency(
    policy,
    observation,
    n_iter: int = 5,
    warmup: int = 1,
) -> dict[str, float]:
    """Policy.predict() の latency 測定 (実機 real ckpt load 後に使う)。

    Args:
        policy: Policy instance (from_ckpt() 済、warmup() 実行後推奨)。
        observation: Observation instance (dummy or real snapshot)。
        n_iter / warmup: measure_latency と同じ。

    Returns:
        dict with mean_ms / p95_ms etc. (measure_latency と同 format)。

    Note: 実機 (Orin / Sakura) で real ckpt load 後に走らせる想定。local PC の
    弱 GPU (8GB) では GR00T (3B params) が VRAM 不足で load 失敗する可能性
    (RAMEN-Ori ~200M は local でも動く見込み)。
    """
    return measure_latency(policy.predict, observation, n_iter=n_iter, warmup=warmup)


def check_budget(stats: dict[str, float], budget_ms: float, tag: str = "") -> dict[str, Any]:
    """measure_latency 結果を budget と照らして judgment 返す (assert 用ではなく report 用)。

    Args:
        stats: measure_latency の戻り値。
        budget_ms: 判定基準 (STEP_TIME_MS / MODEL_FORWARD_BUDGET_MS 等)。
        tag: report label (component name)。

    Returns:
        dict with: within_budget (bool) / margin_ms (float, +=余裕 -=超過) /
                  ratio (mean/budget) / tag / summary (str)。
    """
    mean_ms = stats["mean_ms"]
    margin = budget_ms - mean_ms
    ratio = mean_ms / budget_ms if budget_ms > 0 else float("inf")
    return {
        "tag": tag,
        "budget_ms": budget_ms,
        "mean_ms": mean_ms,
        "p95_ms": stats.get("p95_ms", 0.0),
        "within_budget": mean_ms <= budget_ms,
        "margin_ms": margin,
        "ratio": ratio,
        "summary": (
            f"[{tag}] mean={mean_ms:.2f}ms p95={stats.get('p95_ms', 0):.2f}ms "
            f"budget={budget_ms:.2f}ms ratio={ratio:.2f} "
            f"({'OK' if margin >= 0 else 'OVER'} by {abs(margin):.2f}ms)"
        ),
    }
