"""Detection cleaning を streaming で駆動する pipeline (`DetectionStream`)。

Orchestrator の tick loop で `push(raw_T)` を呼び、latency 後に確定 cleaned dets を
`Optional[list[OBBDetection]]` で返す。stream 終了 (LeRobot ep 終端) では `flush()`
で buffer の残りを吐き出す (ROS2 real-time では呼ばない)。

内部で `cleaner.clean_frame` (Step 1) と `cleaner._median_single_frame`
(Step 2) を per-push 呼ぶ形の薄い wrapper。cleaning logic は既存 `cleaner.py`
に閉じ、この module は buffering と delay pipeline 制御だけを持つ (責務分離)。

latency (raw_T を push してから、その raw に対応する cleaned が emit されるまでの
追加 push 回数):
    - median filter enabled = 2 push delay (Step 1 で 1、median 3-tap で 1)
      warmup: 最初の 3 push は None、4 push 目から定常 emit
    - median filter disabled = 1 push delay (Step 1 のみ)
      warmup: 最初の 1 push は None、2 push 目から定常 emit

端 frame (index=0 / index=n-1) の扱い:
    - median enabled 時、streaming は端 frame を emit しない (中央 index 1..n-2 のみ)。
      batch (`clean_all_frames`) は端 frame を 2-tap median で emit するが、streaming
      では端 frame の情報保持が buffering 設計を複雑化するため drop する design。
      real-time では端の概念が無い、ep replay では端 2 frame の欠損は許容範囲。
    - median disabled 時、raw_curr の Step 1 clean 結果を flush() で emit するため、
      端 frame も emit される (batch と一致)。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from inference.desktop.perception.cleaner import (
    _median_single_frame,
    clean_frame,
    load_cleanup_config,
)
from inference.desktop.perception.yolo_obb import OBBDetection


class DetectionStream:
    """detection cleaning を streaming で駆動する薄い pipeline。

    Args:
        config: cleanup config dict。None なら `load_cleanup_config()` で default
            yaml から load (batch API と同じ config を共有)。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_cleanup_config()
        self._max_count: dict[str, int] = dict(config["max_count"])
        self._over_max: float = float(config["over_max_continue_iou"])
        self._under_max: float = float(config["under_max_similar_iou"])
        mf_config = config.get("median_filter", {})
        self._median_enabled: bool = bool(mf_config.get("enabled", False))
        self._median_iou: float = float(mf_config.get("iou_match_min", 0.5))

        # Step 1 用 ring:
        # raw_curr: 直前 push で受け取った raw (次 push 時に次_raw を得て clean 対象になる)
        # cleaned_prev: 直前 push で clean した cleaned (Step 1 の prev_cleaned 引数用)
        self._raw_curr: Optional[list[OBBDetection]] = None
        self._cleaned_prev: Optional[list[OBBDetection]] = None

        # Step 2 用 ring (median): maxlen=3 の deque
        # size 3 揃った瞬間に中央 (ring[1]) を 3-tap median emit + 最古 (ring[0]) shift out。
        self._median_ring: deque[list[OBBDetection]] = deque(maxlen=3)

    def push(
        self, raw: list[OBBDetection]
    ) -> Optional[list[OBBDetection]]:
        """1 frame の raw dets を投入、確定した cleaned dets を返す (未充填は None)。

        median enabled: 4 push 目 (T=3) 以降で median 版 cleaned_(T-2) を返す。
            warmup 期間 = 最初の 3 push は None。
        median disabled: 2 push 目 (T=1) 以降で cleaned_(T-1) を返す。
            warmup 期間 = 最初の 1 push は None。
        """
        # === Step 1: 直前の raw_curr (= T-1 の raw) を clean ===
        cleaned: Optional[list[OBBDetection]] = None
        if self._raw_curr is not None:
            cleaned = clean_frame(
                self._raw_curr,
                self._cleaned_prev,  # T-2 の cleaned (最初は None)
                raw,                  # T の raw = next_raw for T-1
                self._max_count,
                self._over_max,
                self._under_max,
            )
            self._cleaned_prev = cleaned
        self._raw_curr = raw

        if not self._median_enabled:
            return cleaned  # Step 1 のみ、1 frame delay で emit

        # === Step 2: median filter (3-tap ring) ===
        if cleaned is None:
            return None  # Step 1 warmup 中
        self._median_ring.append(cleaned)
        if len(self._median_ring) < 3:
            return None  # median warmup 中

        # size 3 揃った: 中央の 3-tap median を emit (次 push で最古 ring[0] は自動 shift out)
        return _median_single_frame(
            self._median_ring[0],
            self._median_ring[1],
            self._median_ring[2],
            self._median_iou,
        )

    def flush(self) -> list[list[OBBDetection]]:
        """stream 終了時、buffer の残り frame を吐き出す。

        median enabled: raw_curr を next_raw=[] で Step 1 clean → cleaned_(n-1) を得て
            median ring に append。ring[1] (= cleaned_(n-2)) の 3-tap median を 1 frame
            emit する。cleaned_(n-1) の 2-tap median と cleaned_0 の 2-tap median は
            design 上 drop (module docstring 参照)。
        median disabled: raw_curr の Step 1 clean 結果 (cleaned_(n-1)) を 1 frame emit。
        """
        remaining: list[list[OBBDetection]] = []

        # Step 1 flush: raw_curr が残ってれば next_raw=[] で clean
        cleaned_last: Optional[list[OBBDetection]] = None
        if self._raw_curr is not None:
            cleaned_last = clean_frame(
                self._raw_curr,
                self._cleaned_prev,
                [],  # next_raw なし = 終端 frame
                self._max_count,
                self._over_max,
                self._under_max,
            )

        if not self._median_enabled:
            if cleaned_last is not None:
                remaining.append(cleaned_last)
            self._reset()
            return remaining

        # median enabled: cleaned_last を ring に追加、size 3 なら中央を emit
        if cleaned_last is not None:
            self._median_ring.append(cleaned_last)
        if len(self._median_ring) == 3:
            remaining.append(
                _median_single_frame(
                    self._median_ring[0],
                    self._median_ring[1],
                    self._median_ring[2],
                    self._median_iou,
                )
            )
        # 注: size < 3 の残り frame (ep が 2 frame 以下等の極端 case) は端 frame 扱いで drop。

        self._reset()
        return remaining

    def _reset(self) -> None:
        """flush 後の内部 state クリア (再利用は想定してないが安全側で)。"""
        self._raw_curr = None
        self._cleaned_prev = None
        self._median_ring.clear()

    def snapshot_buffer(self) -> dict:
        """streaming buffer を JSON serializable dict で dump。

        Orchestrator が skill 遷移時 snapshot log を emit する時に呼ぶ。resume 時に
        cleaner を prewarm するため、`raw_curr` / `cleaned_prev` / `median_ring` の
        3 buffer を dict 化する。復元 API (`restore_from_buffer(dict)`) は別途追加。

        Returns:
            {
                "raw_curr": list[dict] or None,
                "cleaned_prev": list[dict] or None,
                "median_ring": list[list[dict]]  (0-3 個の frame),
                "median_enabled": bool,
            }
            各 dict は `_obb_to_dict` により
            `{class_id, class_name, confidence, verts}` (verts は list[list[float]])。
        """
        return {
            "raw_curr": (
                [_obb_to_dict(d) for d in self._raw_curr]
                if self._raw_curr is not None
                else None
            ),
            "cleaned_prev": (
                [_obb_to_dict(d) for d in self._cleaned_prev]
                if self._cleaned_prev is not None
                else None
            ),
            "median_ring": [
                [_obb_to_dict(d) for d in frame] for frame in self._median_ring
            ],
            "median_enabled": self._median_enabled,
        }

    def restore_from_buffer(self, buffer: dict) -> None:
        """`snapshot_buffer()` で dump した dict から streaming buffer を復元。

        次 push() から warmup を経由せず定常 emit に入れる (log 途中再開時の
        cleaner prewarm 用)。既存 buffer は全て上書きされる。

        Raises:
            ValueError: median_enabled が現 instance の config と mismatch
                (run 間で config が変わった場合、復元しても後続処理が壊れる)。
        """
        expected = bool(buffer["median_enabled"])
        if expected != self._median_enabled:
            raise ValueError(
                f"median_enabled mismatch: snapshot={expected}, "
                f"instance={self._median_enabled}"
            )
        self._raw_curr = _restore_frame(buffer["raw_curr"])
        self._cleaned_prev = _restore_frame(buffer["cleaned_prev"])
        self._median_ring.clear()
        for frame in buffer["median_ring"]:
            restored = _restore_frame(frame)
            # frame slot は必ず list (snapshot 側で None は raw_curr/cleaned_prev のみ)
            assert restored is not None
            self._median_ring.append(restored)


def _obb_to_dict(det: OBBDetection) -> dict:
    """OBBDetection dataclass → JSON serializable dict。

    verts (np.ndarray shape (4, 2)) は list[list[float]] に変換、他 field は
    素の scalar なのでそのまま。
    """
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "verts": det.verts.tolist(),
    }


def _dict_to_obb(d: dict) -> OBBDetection:
    """`_obb_to_dict` の逆変換。verts は np.ndarray に戻す (frozen dataclass 復元)。

    数値 dtype は float32 統一 (元の YOLO 出力と揃える。JSON からは float64 で
    復元されるため明示 cast)。
    verts の shape (4, 2) は fail-fast 検証 (malformed log や手動編集 log で
    後続 IoU 計算が silent に誤値を返すのを防ぐ)。
    """
    import numpy as np

    verts = np.asarray(d["verts"], dtype=np.float32)
    if verts.shape != (4, 2):
        raise ValueError(
            f"OBBDetection.verts must have shape (4, 2), got {verts.shape}"
        )
    return OBBDetection(
        class_id=int(d["class_id"]),
        class_name=str(d["class_name"]),
        confidence=float(d["confidence"]),
        verts=verts,
    )


def _restore_frame(raw: object) -> Optional[list[OBBDetection]]:
    """snapshot dict の frame slot (list[dict] or None) を list[OBBDetection] に戻す。"""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"frame slot must be list or None, got {type(raw).__name__}")
    return [_dict_to_obb(d) for d in raw]
