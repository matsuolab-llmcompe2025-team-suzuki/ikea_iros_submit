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

import os
import sys
from pathlib import Path

import numpy as np
import yaml

from inference.desktop.perception.yolo_obb import OBBDetection
from inference.desktop.skill_planner.geometry import align_verts_by_min_shift, obb_aabb_iou

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "cleanup.yaml"

# Issue #140: median の prev/next 対応づけを振動に強くする centroid fallback の
# runtime 切替 env (実機 A/B 用、YAML 編集/再ビルド不要)。
# - RAMEN_MEDIAN_MATCH_MAX_CENTROID_DIST: >0 で fallback 有効 (normalized 画面比、例 0.12)。
#   0 / 未設定 なら YAML の median_filter.match_max_centroid_dist を使う。
# - RAMEN_MEDIAN_MATCH_AMBIGUITY_RATIO: 同 class 複数時の誤対応ガード比。
_ENV_MATCH_MAX_CENTROID_DIST = "RAMEN_MEDIAN_MATCH_MAX_CENTROID_DIST"
_ENV_MATCH_AMBIGUITY_RATIO = "RAMEN_MEDIAN_MATCH_AMBIGUITY_RATIO"


def load_cleanup_config(path: Path | None = None) -> dict:
    """cleanup.yaml を load (perception layer 別枠、skill_planner と分離)。"""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    return yaml.safe_load(path.read_text())


def _env_float(name: str) -> float | None:
    """env var を float で読む。未設定/空/不正は None (=無視して config default へ)。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        # 黙って既定に落ちると、A/B の run が「入れたつもりの設定」と違う条件で回る。
        print(f"[cleaner] {name}={raw!r} は数値として読めないので使わない", file=sys.stderr)
        return None


def resolve_median_match_params(mf_config: dict) -> tuple[float, float]:
    """median の対応づけ params (max_centroid_dist, ambiguity_ratio) を解決する。

    優先順位: env override > YAML (median_filter.*) > 既定 (fallback 無効)。
    - max_centroid_dist: 0.0 = centroid fallback 無効 = 従来の IoU-only 挙動 (既定)。
    - ambiguity_ratio: 同 class 複数時、nearest が 2 番目より (ratio 倍) 近い時のみ採用。
    """
    max_dist = _env_float(_ENV_MATCH_MAX_CENTROID_DIST)
    if max_dist is None:
        max_dist = float(mf_config.get("match_max_centroid_dist", 0.0))
    ratio = _env_float(_ENV_MATCH_AMBIGUITY_RATIO)
    if ratio is None:
        ratio = float(mf_config.get("match_ambiguity_ratio", 2.0))
    return max(0.0, max_dist), ratio


def _centroid(verts: np.ndarray) -> np.ndarray:
    """OBB 4 頂点の重心 (normalized [0, 1])。"""
    return verts.mean(axis=0)


def clean_frame(
    curr: list[OBBDetection],
    prev_cleaned: list[OBBDetection] | None,
    next_raw: list[OBBDetection] | None,
    max_count: dict[str, int],
    over_max_iou: float,
    under_max_iou: float,
    over_max_mode: str = "keep_continuing",
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
            # case A: max 超え。keep_continuing = 前 frame と IoU >= over_max_iou のみ保持
            # (該当ゼロなら class 全消し)、fill_by_conf = 継続を先に残し残りを conf 降順で max_n まで (Issue #141 束 1-9)
            continuing = [
                d for d in dets
                if any(obb_aabb_iou(d.verts, p.verts) >= over_max_iou for p in prev_of_class)
            ]
            if over_max_mode == "keep_continuing":
                result.extend(continuing)
            else:
                chosen = sorted(continuing, key=lambda d: -d.confidence)[:max_n]
                rest = [d for i, d in enumerate(dets) if not any(d is c for c in continuing)]
                chosen += sorted(rest, key=lambda d: -d.confidence)[: max_n - len(chosen)]
                result.extend(chosen)
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
    max_centroid_dist: float = 0.0,
    ambiguity_ratio: float = 2.0,
) -> OBBDetection | None:
    """candidates 中 target_verts に対応づく detection を返す。無ければ None。

    1) 従来: IoU 最大 (> iou_min) の detection。
    2) Issue #140 (opt-in、`max_centroid_dist > 0` 時のみ): IoU が全滅した時、
       振動でフレーム間移動が大きく IoU 対応が切れた case として、**重心距離が
       nearest** の同 class detection を fallback 採用する。誤対応 (別の脚を掴む)
       防止に:
         - nearest 重心距離 <= max_centroid_dist (normalized) を要求
         - 候補複数時は nearest が 2 番目より ambiguity_ratio 倍以上近い時のみ採用
           (単一候補なら無条件 OK)
    """
    if not candidates:
        return None
    # 1) IoU match (従来動作、max_centroid_dist=0 なら完全に従来と一致)
    best = None
    best_iou = iou_min
    for c in candidates:
        iou = obb_aabb_iou(target_verts, c.verts)
        if iou > best_iou:
            best_iou = iou
            best = c
    if best is not None:
        return best

    # 2) centroid fallback (IoU 全滅時のみ、opt-in)
    if max_centroid_dist <= 0.0:
        return None
    tc = _centroid(target_verts)
    # (dist, index, det): index は tie-break 用 (OBBDetection は非比較のため)
    scored = sorted(
        (float(np.linalg.norm(_centroid(c.verts) - tc)), i, c)
        for i, c in enumerate(candidates)
    )
    nearest_d, _, nearest_c = scored[0]
    if nearest_d > max_centroid_dist:
        return None
    if len(scored) >= 2 and nearest_d * ambiguity_ratio > scored[1][0]:
        # nearest と 2 番目が近い = どの物体か曖昧 → 誤対応回避で不採用
        return None
    return nearest_c


def _median_single_frame(
    prev: list[OBBDetection],
    curr: list[OBBDetection],
    nxt: list[OBBDetection],
    iou_min: float,
    max_centroid_dist: float = 0.0,
    ambiguity_ratio: float = 2.0,
) -> list[OBBDetection]:
    """1 frame 分の median filter (prev, curr, next の per-detection pivot median)。

    curr の各 detection D について、prev / next の same-class best match を
    集めて D.verts の頂点順に cyclic shift alignment、element-wise median で
    verts を置換。neighbor 空 or match 無しなら raw passthrough (端 frame 相当)。

    対応づけは IoU (>= iou_min)。Issue #140: `max_centroid_dist > 0` の時のみ、
    IoU 全滅 (振動でフレーム間移動大) の case で重心 nearest fallback を使う。

    batch API (`median_filter_pass`) と streaming API (`DetectionStream`) の
    共通 core。
    """
    result: list[OBBDetection] = []
    for d in curr:
        same_class_prev = [p for p in prev if p.class_name == d.class_name]
        same_class_next = [p for p in nxt if p.class_name == d.class_name]
        p_match = _find_best_match(
            d.verts, same_class_prev, iou_min, max_centroid_dist, ambiguity_ratio
        )
        n_match = _find_best_match(
            d.verts, same_class_next, iou_min, max_centroid_dist, ambiguity_ratio
        )

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
    max_centroid_dist, ambiguity_ratio = resolve_median_match_params(config)
    n = len(all_dets)
    result: list[list[OBBDetection]] = [[] for _ in range(n)]
    for t in range(n):
        prev = all_dets[t - 1] if t > 0 else []
        nxt = all_dets[t + 1] if t + 1 < n else []
        result[t] = _median_single_frame(
            prev, all_dets[t], nxt, iou_min, max_centroid_dist, ambiguity_ratio
        )
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
    over_max_mode = str(config.get("over_max_mode", "keep_continuing"))

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
                over_max_mode,
            )
        )

    # Step 2: median filter (optional、旧 rotation_smoothing 置換)
    mf_config = config.get("median_filter", {})
    if mf_config.get("enabled", False):
        cleaned = median_filter_pass(cleaned, mf_config)

    return cleaned
