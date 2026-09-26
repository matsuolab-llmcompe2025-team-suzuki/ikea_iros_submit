"""OBB 幾何 helper (Kabsch / aspect / IoU / edge dist / vertex alignment)。

Enter rule 群から呼ばれる。Pick は Kabsch rotation angle と aspect ratio、
insert は AABB IoU 近似で hand ∩ leg overlap を計算、move_base_next / flip は
hand AABB と table_top 左辺 segment の距離判定に使う。
"""

from __future__ import annotations

import numpy as np


def obb_aabb_iou(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """2 OBB を AABB 化 (verts の x/y min/max) して IoU を返す。

    回転無視の近似だが hand ∩ leg のような「どの程度重なってるか」の判定には
    十分。精密 OBB IoU が必要になれば cv2.rotatedRectangleIntersection に置換。

    Args:
        v_a: shape (4, 2) の OBB 頂点。
        v_b: shape (4, 2) の OBB 頂点。

    Returns:
        IoU 値 [0.0, 1.0]。無交差 / 面積 0 は 0.0。

    Raises:
        ValueError: verts の shape が (4, 2) でない。
    """
    if v_a.shape != (4, 2) or v_b.shape != (4, 2):
        raise ValueError(f"expected (4, 2), got {v_a.shape} and {v_b.shape}")
    ax1, ay1 = float(v_a[:, 0].min()), float(v_a[:, 1].min())
    ax2, ay2 = float(v_a[:, 0].max()), float(v_a[:, 1].max())
    bx1, by1 = float(v_b[:, 0].min()), float(v_b[:, 1].min())
    bx2, by2 = float(v_b[:, 0].max()), float(v_b[:, 1].max())
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def obb_aspect_ratio(verts: np.ndarray) -> float:
    """OBB 4 頂点 (cyclic 順) から aspect ratio = 長辺 / 短辺 を返す。

    Rectangle は対辺等長なので隣接 2 辺 (v0-v1, v1-v2) の長さで判定。
    正方形 → 1.0、細長い → 大きい値。

    Args:
        verts: shape (4, 2) の OBB 頂点 (cyclic 順)。

    Returns:
        aspect ratio (>= 1.0)。短辺が 0 に近い場合は inf。

    Raises:
        ValueError: verts の shape が (4, 2) でない。
    """
    if verts.shape != (4, 2):
        raise ValueError(f"expected (4, 2), got {verts.shape}")
    e1 = float(np.linalg.norm(verts[1] - verts[0]))
    e2 = float(np.linalg.norm(verts[2] - verts[1]))
    long_e = max(e1, e2)
    short_e = min(e1, e2)
    if short_e < 1e-9:
        return float("inf")
    return long_e / short_e


def left_edge_of_obb(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OBB の 4 edges 中「左辺」= midpoint x が最小の edge の 2 端点を返す。

    Cyclic edge (v_i, v_(i+1)%4) を全て見て midpoint x で選択。回転に robust。

    Args:
        verts: shape (4, 2) の OBB 頂点 (cyclic 順)。

    Returns:
        (p1, p2) 左辺の 2 端点。

    Raises:
        ValueError: verts の shape が (4, 2) でない。
    """
    if verts.shape != (4, 2):
        raise ValueError(f"expected (4, 2), got {verts.shape}")
    edges = [(verts[i], verts[(i + 1) % 4]) for i in range(4)]
    return min(edges, key=lambda e: (e[0][0] + e[1][0]) / 2)


def bottom_left_vertex_y(verts: np.ndarray) -> float:
    """OBB 4 頂点中「左下」に最も近い vertex の y 座標を返す。

    画像座標系 (y↑ 下方向、x↑ 右方向) で、左下 = 最下 (max y) かつ最左 (min x)。
    score = y - x を最大化する vertex が「左下」に一番近い。

    Args:
        verts: shape (4, 2) の OBB 頂点。

    Returns:
        「左下」vertex の y 座標。
    """
    scores = verts[:, 1] - verts[:, 0]
    return float(verts[int(np.argmax(scores)), 1])


def aabb_reaches_obb_left_edge(
    tt_verts: np.ndarray,
    hand_xmin: float,
    hand_ymin: float,
    hand_xmax: float,
    hand_ymax: float,
    dist_threshold: float,
    hand_in_frac_max: float,
) -> bool:
    """hand AABB が OBB の左辺に到達しているかを OR 2 条件で判定する。

    到達条件 (OR):
        (a) 左辺 segment と AABB の最短距離 <= dist_threshold (近接 or 接触)
        (b) 0 < (hand ∩ OBB 面積 / hand 面積) <= hand_in_frac_max
            (hand が OBB に部分重なり = 端を掴んでる姿勢、full-inside は除外)

    (b) は overlap fraction ベース。rot_leg 中は hand が leg を掴んで OBB 内に
    深く入る (frac ≈ 0.9-1.0)、flip / base 準備で edge に移動すると閾値以下に
    落ちる、という物理観察に基づく。0 < を要求するのは hand が完全に OBB 外の
    場合の誤発火防止。

    Args:
        tt_verts: shape (4, 2) の OBB 頂点 (通常 table_top)。
        hand_xmin: hand AABB 左端 x。
        hand_ymin: hand AABB 上端 y。
        hand_xmax: hand AABB 右端 x。
        hand_ymax: hand AABB 下端 y。
        dist_threshold: (a) の距離閾値 (normalized 座標系)。
        hand_in_frac_max: (b) の overlap fraction 上限。

    Returns:
        (a) または (b) が成立していれば True。
    """
    # (a) dist check
    p1, p2 = left_edge_of_obb(tt_verts)
    dist = seg_to_aabb_dist(p1, p2, hand_xmin, hand_ymin, hand_xmax, hand_ymax)
    if dist <= dist_threshold:
        return True
    # (b) partial overlap check (AABB approximation)
    tt_xmin = float(tt_verts[:, 0].min())
    tt_ymin = float(tt_verts[:, 1].min())
    tt_xmax = float(tt_verts[:, 0].max())
    tt_ymax = float(tt_verts[:, 1].max())
    ixmin = max(hand_xmin, tt_xmin)
    iymin = max(hand_ymin, tt_ymin)
    ixmax = min(hand_xmax, tt_xmax)
    iymax = min(hand_ymax, tt_ymax)
    inter = max(0.0, ixmax - ixmin) * max(0.0, iymax - iymin)
    hand_area = (hand_xmax - hand_xmin) * (hand_ymax - hand_ymin)
    if hand_area <= 0 or inter <= 0:
        return False
    return (inter / hand_area) <= hand_in_frac_max


def seg_to_aabb_dist(
    p1: np.ndarray,
    p2: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    n_samples: int = 50,
) -> float:
    """線分 p1-p2 と AABB の最短距離を返す。

    n_samples 個の点を線分上に取り、各点から AABB までの距離の min。線分上の点が
    AABB 内部にある場合は距離 0。n_samples=50 で precision ~ seg_length/100
    (typical threshold 0.02 に十分)。

    Args:
        p1: 線分の一端 (shape (2,))。
        p2: 線分の他端 (shape (2,))。
        xmin: AABB 左端 x。
        ymin: AABB 上端 y。
        xmax: AABB 右端 x。
        ymax: AABB 下端 y。
        n_samples: 線分の sampling 数 (default 50)。

    Returns:
        最短距離 (input と同じ座標系)。
    """
    min_d = float("inf")
    for i in range(n_samples + 1):
        t = i / n_samples
        px = p1[0] * (1 - t) + p2[0] * t
        py = p1[1] * (1 - t) + p2[1] * t
        dx = max(0.0, xmin - px, px - xmax)
        dy = max(0.0, ymin - py, py - ymax)
        d = (dx * dx + dy * dy) ** 0.5
        if d < min_d:
            min_d = d
    return min_d


def align_verts_by_min_shift(pivot: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """verts を pivot と最小 sum-of-squared distance となる cyclic shift で並べ替える。

    YOLO の頂点出力順は frame ごとに shift しうるので、複数 frame を平均・median
    する前に順序を合わせる必要がある。

    Args:
        pivot: shape (4, 2) の基準頂点 (この順序に合わせる)。
        verts: shape (4, 2) の並べ替え対象頂点。

    Returns:
        pivot に対して best cyclic shift 適用後の verts (shape (4, 2))。

    Raises:
        ValueError: pivot / verts の shape が (4, 2) でない。
    """
    if pivot.shape != (4, 2) or verts.shape != (4, 2):
        raise ValueError(f"expected (4, 2), got {pivot.shape} and {verts.shape}")
    best_shift, best_err = 0, float("inf")
    for shift in range(4):
        idxs = [(shift + i) % 4 for i in range(4)]
        err = float(((verts[idxs] - pivot) ** 2).sum())
        if err < best_err:
            best_err = err
            best_shift = shift
    idxs = [(best_shift + i) % 4 for i in range(4)]
    return verts[idxs]


def mean_verts_pivot_aligned(verts_list: list[np.ndarray]) -> np.ndarray:
    """複数 frame の OBB verts を末尾を pivot に alignment 後 element-wise 平均する。

    末尾 (最新 frame) を pivot にする理由は、base_run 開始直前の姿勢が最も現況に
    近い pose 想定のため。

    Args:
        verts_list: shape (4, 2) の OBB verts の list (時系列順)。

    Returns:
        shape (4, 2) の平均 verts。

    Raises:
        ValueError: verts_list が空。
    """
    if not verts_list:
        raise ValueError("verts_list is empty")
    pivot = verts_list[-1]
    aligned = [pivot]
    for v in verts_list[:-1]:
        aligned.append(align_verts_by_min_shift(pivot, v))
    return np.stack(aligned, axis=0).mean(axis=0)


def kabsch_rotation_angle_deg(v_ref: np.ndarray, v_cur: np.ndarray) -> float:
    """Kabsch algorithm で ref → cur の rectangle 回転角 (deg) を求める。

    YOLO OBB の頂点順序は frame ごとに一貫しないので、4 通り cyclic shift を試して
    最小 reconstruction error の shift の rotation を採用。Rectangle は 180°
    symmetry なので値域 [-90, 90] に正規化。

    正方形 (aspect ratio ~ 1) では 4 shift が近似同 error になり angle が数値
    不安定になる (0° ⇔ 90° flip 発生)。この場合 caller 側で hysteresis 等で
    補正が必要。

    Args:
        v_ref: shape (4, 2) の基準 OBB 頂点。
        v_cur: shape (4, 2) の現 frame OBB 頂点。

    Returns:
        回転角 (deg)、値域 [-90, 90]。

    Raises:
        ValueError: verts の shape が (4, 2) でない。
    """
    if v_ref.shape != (4, 2) or v_cur.shape != (4, 2):
        raise ValueError(f"expected (4, 2), got {v_ref.shape} and {v_cur.shape}")
    vr = v_ref - v_ref.mean(axis=0)
    vc = v_cur - v_cur.mean(axis=0)
    best_ang, best_err = 0.0, float("inf")
    for shift in range(4):
        idxs = [(shift + i) % 4 for i in range(4)]
        vc_shifted = vc[idxs]
        H = vr.T @ vc_shifted
        U, S, Vt = np.linalg.svd(H)
        d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        v_rot = vr @ R.T
        err = float(((v_rot - vc_shifted) ** 2).sum())
        if err < best_err:
            best_err = err
            ang_rad = float(np.arctan2(R[1, 0], R[0, 0]))
            best_ang = float(np.degrees(ang_rad))
    while best_ang > 90:
        best_ang -= 180
    while best_ang < -90:
        best_ang += 180
    return best_ang
