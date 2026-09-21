"""policy に渡す YOLO-OBB の検出を、遅れなしで整える filter (Issue #141 束 1-9、共通化の D3)。

上位 Planner 用の cleaner (`cleaner.py` + `stream.py`) は次の frame を 2 回待つので、tick T の出力は
frame T−2 の枠になる。policy (overlay・memory・progress monitor) は学習時に「画像と同じ frame の生の検出」で
描いた画像を見ているので、枠が遅れると学習と違う入力になる。この filter は前の frame だけで判定し、
今の frame の枠を返す。

1 tick の処理 (設定は `configs/cleanup.yaml` の `policy:` 節):

1. conf < `conf_min` を落とす (学習の overlay と同じ閾値 0.30)
2. 直前の `continue_frames` frame に残した枠と IoU ≥ `continue_iou` の枠 (続いている物) は残す。
   それ以外 (新しい物) は、直前の `persist_frames` frame の生の検出すべてに同じ class で IoU ≥ `continue_iou` の枠が
   あれば残す (1 frame だけ出る誤検知を消す。新しい物は `persist_frames` frame 遅れて描かれる。0 ならすぐ残す)
3. class ごとの上限 (`max_count`) の扱いは `over_cap` で選ぶ:
   - `fill_by_conf`: 2 で残した枠が上限を超えたら、続いている枠を先に、残りを conf の高い順で上限まで残す
     (class が丸ごと消えることは無い。`over_max_continue_iou` は使わない)
   - `keep_continuing`: 生の検出の数が上限を超えた frame では、直前に残した枠と IoU ≥ `over_max_continue_iou` の枠だけ残す
     (無ければその class はこの frame で消える。Planner 用の cleaner と同じ規則)
4. `smoothing: median3_past` なら、2〜3 の結果の頂点を直前 2 frame と合わせた中央値でならす
   (過去の frame だけを使うので、動いている物は約 1 frame 遅れる。`none` ならならさない)
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import yaml

from inference.desktop.perception.cleaner import _median_single_frame
from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.geometry import obb_aabb_iou

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "cleanup.yaml"
SMOOTHING_MODES = ("none", "median3_past")
OVER_CAP_MODES = ("fill_by_conf", "keep_continuing")


def load_policy_filter_config(path: Path | None = None) -> dict:
    """`cleanup.yaml` の `policy:` 節を返す。"""
    config = yaml.safe_load((path or _DEFAULT_CONFIG_PATH).read_text())
    return config["policy"]


def _overlaps(det: OBBDetection, others: list[OBBDetection], iou_min: float) -> bool:
    return any(
        o.class_name == det.class_name and obb_aabb_iou(det.verts, o.verts) >= iou_min
        for o in others
    )


class PolicyDetectionFilter:
    """tick ごとに `push(raw)` を呼び、今の frame の枠を返す (None を返さない)。"""

    def __init__(self, config: dict) -> None:
        self._conf_min = float(config["conf_min"])
        self._max_count = {str(k): int(v) for k, v in config["max_count"].items()}
        self._over_max_iou = float(config["over_max_continue_iou"])
        self._continue_iou = float(config["continue_iou"])
        self._continue_frames = int(config["continue_frames"])
        self._persist_frames = int(config["persist_frames"])
        self._over_cap = str(config["over_cap"])
        self._smoothing = str(config["smoothing"])
        self._smoothing_iou = float(config["smoothing_iou_min"])
        if self._over_cap not in OVER_CAP_MODES:
            raise ValueError(f"over_cap must be one of {OVER_CAP_MODES}, got {self._over_cap!r}")
        if self._continue_frames < 1:
            raise ValueError(f"continue_frames must be >= 1, got {self._continue_frames}")
        if self._persist_frames < 0:
            raise ValueError(f"persist_frames must be >= 0, got {self._persist_frames}")
        if self._smoothing not in SMOOTHING_MODES:
            raise ValueError(f"smoothing must be one of {SMOOTHING_MODES}, got {self._smoothing!r}")
        self.reset()

    def reset(self) -> None:
        # 直前に残した枠 (ならす前)。新しい順
        self._kept: deque[list[OBBDetection]] = deque(maxlen=max(self._continue_frames, 2))
        # 直前の生の検出 (conf で切った後)。新しい順
        self._raw: deque[list[OBBDetection]] = deque(maxlen=max(self._persist_frames, 1))

    def push(self, raw: list[OBBDetection]) -> list[OBBDetection]:
        curr = [d for d in raw if d.confidence >= self._conf_min]
        recent_kept = [d for frame in list(self._kept)[: self._continue_frames] for d in frame]
        prev_kept = self._kept[0] if len(self._kept) > 0 else []
        older_kept = self._kept[1] if len(self._kept) > 1 else []
        persist_window = list(self._raw)[: self._persist_frames]
        persist_ready = len(persist_window) == self._persist_frames

        by_class: dict[str, list[OBBDetection]] = {}
        for d in curr:
            by_class.setdefault(d.class_name, []).append(d)

        kept: list[OBBDetection] = []
        for name, dets in by_class.items():
            cap = self._max_count.get(name)
            if cap is not None and len(dets) > cap and self._over_cap == "keep_continuing":
                kept.extend(d for d in dets if _overlaps(d, prev_kept, self._over_max_iou))
                continue
            continuing: list[OBBDetection] = []
            fresh: list[OBBDetection] = []
            for d in dets:
                if _overlaps(d, recent_kept, self._continue_iou):
                    continuing.append(d)
                elif persist_ready and all(
                    _overlaps(d, frame, self._continue_iou) for frame in persist_window
                ):
                    fresh.append(d)
            if cap is None or len(continuing) + len(fresh) <= cap:
                kept.extend(continuing + fresh)
                continue
            # 上限を超えたら、続いている枠を先に、残りを conf の高い順で上限まで
            chosen = sorted(continuing, key=lambda d: -d.confidence)[:cap]
            chosen += sorted(fresh, key=lambda d: -d.confidence)[: cap - len(chosen)]
            kept.extend(chosen)

        self._kept.appendleft(kept)
        self._raw.appendleft(curr)
        if self._smoothing == "none":
            return kept
        # 中央値は 3 つの枠の対称な関数なので、「次」の枠の代わりに 2 frame 前を渡す (過去だけでならす)
        return _median_single_frame(prev_kept, kept, older_kept, self._smoothing_iou)
