"""YOLO 推論の偽検出を除外/抑制する 2 段パイプライン。

Step 1 (class-wise persistence check):
    case A (count > max_count[class]):
        前 frame cleaned dets と IoU >= over_max_continue_iou の検出のみ保持
        それ以外は spike として drop
    case B (count <= max_count[class]):
        Continuing (前 frame cleaned dets と IoU >= under_max_similar_iou): 保持
        New (前 frame と一致なし): 次 frame raw dets と IoU >= under_max_similar_iou なら保持
        どちらでもなければ drop (突然発生 1-frame 検出)

Step 2 (3-frame temporal median filter):
    各 detection D の verts を (D, prev_aligned, next_aligned) の element-wise
    median で置換。prev/next は same-class の best IoU match、pivot-alignment
    で頂点順を D に揃えてから median を取る。1-frame vertex-order flip
    (YOLO artifact) を除去、stateless (raw のみ入力) で lock-in 発生無し。

将来的には yolo_obb.py の predict 内に統合予定。

Config は `inference/desktop/perception/configs/cleanup.yaml` から load。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.geometry import align_verts_by_min_shift, obb_aabb_iou

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "cleanup.yaml"


def load_cleanup_config(path: Path | None = None) -> dict:
    """cleanup.yaml を load (perception layer 別枠、skill_planner と分離)。"""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    return yaml.safe_load(path.read_text())


def clean_frame(
    curr: list[OBBDetection],
    prev_cleaned: list[OBBDetection] | None,
    next_raw: list[OBBDetection] | None,
    max_count: dict[str, int],
    over_max_iou: float,
    under_max_iou: float,
) -> list[OBBDetection]:
    """1 frame の detections をクリーンアップ。詳細は module docstring 参照。

    Args:
        curr: 現 frame の raw detections
        prev_cleaned: 前 frame の cleaned detections (None = 最初の frame)
        next_raw: 次 frame の raw detections (None = 最終 frame)
        max_count: class → max 検出数
        over_max_iou: case A の 前 frame 継続判定 IoU 閾値
        under_max_iou: case B の 前/次 frame 類似判定 IoU 閾値

    Returns:
        cleaned detections for this frame。
    """
    by_class: dict[str, list[OBBDetection]] = {}
    for d in curr:
        by_class.setdefault(d.class_name, []).append(d)

    result: list[OBBDetection] = []
    for cname, dets in by_class.items():
        max_n = max_count.get(cname)
        prev_of_class = [p for p in (prev_cleaned or []) if p.class_name == cname]
        next_of_class = [n for n in (next_raw or []) if n.class_name == cname]

        if max_n is not None and len(dets) > max_n:
            # case A: max 超え → 前 frame と IoU >= over_max_iou のみ保持
            for d in dets:
                if any(
                    obb_aabb_iou(d.verts, p.verts) >= over_max_iou for p in prev_of_class
                ):
                    result.append(d)
        else:
            # case B: max 内 → continuing OR (new かつ 次 frame で確認)
            for d in dets:
                is_continuing = any(
                    obb_aabb_iou(d.verts, p.verts) >= under_max_iou
                    for p in prev_of_class
                )
                if is_continuing:
                    result.append(d)
                    continue
                # New: 次 frame で類似 detection あれば adoption
                will_persist = any(
                    obb_aabb_iou(d.verts, n.verts) >= under_max_iou
                    for n in next_of_class
                )
                if will_persist:
                    result.append(d)
                # else: drop
    return result


def _find_best_match(
    target_verts: np.ndarray,
    candidates: list[OBBDetection],
    iou_min: float,
) -> OBBDetection | None:
    """candidates 中 target_verts と IoU 最大 (>= iou_min) の detection。無ければ None。"""
    if not candidates:
        return None
    best = None
    best_iou = iou_min
    for c in candidates:
        iou = obb_aabb_iou(target_verts, c.verts)
        if iou > best_iou:
            best_iou = iou
            best = c
    return best


def _median_single_frame(
    prev: list[OBBDetection],
    curr: list[OBBDetection],
    nxt: list[OBBDetection],
    iou_min: float,
) -> list[OBBDetection]:
    """1 frame 分の median filter (prev, curr, next の per-detection pivot median)。

    curr の各 detection D について、prev / next の same-class best IoU match を
    集めて D.verts の頂点順に cyclic shift alignment、element-wise median で
    verts を置換。neighbor 空 or match 無しなら raw passthrough (端 frame 相当)。

    batch API (`median_filter_pass`) と streaming API (`DetectionStream`) の
    共通 core。
    """
    result: list[OBBDetection] = []
    for d in curr:
        same_class_prev = [p for p in prev if p.class_name == d.class_name]
        same_class_next = [p for p in nxt if p.class_name == d.class_name]
        p_match = _find_best_match(d.verts, same_class_prev, iou_min)
        n_match = _find_best_match(d.verts, same_class_next, iou_min)

        aligned = [d.verts]  # self を pivot
        if p_match is not None:
            aligned.append(align_verts_by_min_shift(d.verts, p_match.verts))
        if n_match is not None:
            aligned.append(align_verts_by_min_shift(d.verts, n_match.verts))

        if len(aligned) >= 2:
            new_verts = np.median(np.stack(aligned, axis=0), axis=0).astype(np.float32)
        else:
            new_verts = d.verts  # neighbor 無し (端 frame or match 失敗) → raw passthrough

        result.append(
            OBBDetection(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                verts=new_verts,
            )
        )
    return result


def median_filter_pass(
    all_dets: list[list[OBBDetection]],
    config: dict,
) -> list[list[OBBDetection]]:
    """3-frame temporal median filter on OBB verts with pivot alignment (batch)。

    各 frame T について `_median_single_frame(prev=T-1, curr=T, next=T+1)` を実行。
    端 frame (t=0 / t=n-1) は neighbor 片側なし = 2-tap median (self + 片側).

    設計意図:
        - 1-frame vertex-order flip (YOLO artifact) を median で除去
        - 静止 / 持続的変化は 3 frame が中央値付近に収束 → 保持
        - stateless: prev の median 結果は使わず raw のみ入力 → lock-in 発生無し

    Args:
        all_dets: Step 1 (max_count + persistence) 通過後の per-frame detections。
        config: {"iou_match_min": 0.5} を含む dict。

    Returns:
        median filtered detections per frame。
    """
    iou_min = float(config.get("iou_match_min", 0.5))
    n = len(all_dets)
    result: list[list[OBBDetection]] = [[] for _ in range(n)]
    for t in range(n):
        prev = all_dets[t - 1] if t > 0 else []
        nxt = all_dets[t + 1] if t + 1 < n else []
        result[t] = _median_single_frame(prev, all_dets[t], nxt, iou_min)
    return result


def clean_all_frames(
    all_dets: list[list[OBBDetection]],
    config: dict | None = None,
) -> list[list[OBBDetection]]:
    """全 frame の detections を逐次クリーンアップ (T-1 は cleaned、T+1 は raw を参照)。

    Pipeline:
      Step 1: clean_frame (max_count + persistence) を per-frame 逐次適用
      Step 2 (optional): median_filter_pass (config で enabled 時のみ)

    Args:
        all_dets: [[Frame0 raw dets], [Frame1 raw dets], ...]
        config: None なら default yaml から load

    Returns:
        cleaned per-frame detections。
    """
    if config is None:
        config = load_cleanup_config()
    max_count: dict[str, int] = dict(config["max_count"])
    over_max_iou = float(config["over_max_continue_iou"])
    under_max_iou = float(config["under_max_similar_iou"])

    cleaned: list[list[OBBDetection]] = []
    for t, curr in enumerate(all_dets):
        prev_cleaned = cleaned[t - 1] if t > 0 else None
        next_raw = all_dets[t + 1] if t + 1 < len(all_dets) else None
        cleaned.append(
            clean_frame(
                curr,
                prev_cleaned,
                next_raw,
                max_count,
                over_max_iou,
                under_max_iou,
            )
        )

    # Step 2: median filter (optional、旧 rotation_smoothing 置換)
    mf_config = config.get("median_filter", {})
    if mf_config.get("enabled", False):
        cleaned = median_filter_pass(cleaned, mf_config)

    return cleaned
