"""OBB coord warp under 2D affine (Issue #129 Phase E、2026-08-31)。

Run 2/6 (obb_source=precomputed_token) で geometric aug (RandomAffine ±5°/±5% translate)
を head cam image に適用する時、Fusion に流す **OBB coord token も同じ affine で warp**
する必要がある。この helper は image と coord の一貫性を保証する低レベル primitive。

# なぜ必要

Coord token (C-2) は 8 dim xyxyxyxy 座標 (normalized [0,1]) を Fusion に注入し、model が
"この box に注目せよ" を attention weight で学習する仕組み。image を回転/平行移動すると
box の位置も同 座標変換されないと coord と image が mismatch → 学習信号がノイズ化。

hint (overlay 経路) と違い coord は explicit position、warping は必須。

# 使い方 (Phase G で geometric aug 追加時に呼ばれる)

    from torchvision.transforms import v2
    from model.ramen_ori.obb_warp import warp_obb_coord_under_affine

    # RandomAffine 生成の affine_matrix を取得 (torchvision v2 の case か
    # 自前 matrix 生成に応じて画像と同 matrix を coord にも apply)
    affine = v2.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05))
    # ... affine を image に apply ...
    # ... 同 matrix を coord にも apply ...
    warped_verts = warp_obb_coord_under_affine(verts, affine_matrix_normalized)

# 座標系の注意

本 helper は「verts と affine_matrix が同一座標系」を前提とする pure math primitive。
- verts が normalized [0,1] → affine_matrix も normalized [0,1] 空間の変換
- verts が pixel → affine_matrix も pixel 空間の変換
呼出側で座標系を揃える (Phase G の統合時に torchvision の RandomAffine matrix を [0,1]
空間に変換して渡す形にする)。
"""

from __future__ import annotations

import torch


def warp_obb_coord_under_affine(
    verts: torch.Tensor,
    affine_matrix: torch.Tensor,
    clamp: bool = False,
) -> torch.Tensor:
    """OBB 4-vert (top_K, 8) を 2D affine matrix で warp。

    verts の各 (x, y) 頂点に affine を適用:
        x' = a*x + b*y + c
        y' = d*x + e*y + f
    where affine_matrix = [[a, b, c], [d, e, f]]

    Args:
        verts: (..., top_K, 8) tensor、xyxyxyxy 順の 4 頂点 × 2 座標 (x, y)。
               座標系は任意 (normalized [0,1] or pixel)、affine_matrix と一致要。
        affine_matrix: (2, 3) tensor、標準 2D affine (image coord → image coord)。
        clamp: True なら結果を [0, 1] に clamp (normalized 前提の safety)。
               画像範囲外に飛ぶ box を明示 (Phase G で valid_mask=False に落とす予定)。

    Returns:
        warped_verts: verts と同 shape (..., top_K, 8) の warp 済 tensor。dtype 保存。

    Raises:
        ValueError: verts last dim != 8、affine_matrix shape != (2, 3)。
    """
    if verts.shape[-1] != 8:
        raise ValueError(f"verts last dim must be 8 (4 verts × 2 xy), got {verts.shape}")
    if affine_matrix.shape != (2, 3):
        raise ValueError(
            f"affine_matrix must be (2, 3), got {tuple(affine_matrix.shape)}"
        )

    batch_shape = verts.shape[:-1]  # (..., top_K)
    v = verts.reshape(*batch_shape, 4, 2)                # (..., top_K, 4, 2)
    x = v[..., 0]                                        # (..., top_K, 4)
    y = v[..., 1]

    a = affine_matrix[0, 0]
    b = affine_matrix[0, 1]
    c = affine_matrix[0, 2]
    d = affine_matrix[1, 0]
    e = affine_matrix[1, 1]
    f = affine_matrix[1, 2]

    x_new = a * x + b * y + c
    y_new = d * x + e * y + f
    v_new = torch.stack([x_new, y_new], dim=-1)          # (..., top_K, 4, 2)
    out = v_new.reshape(*batch_shape, 8)                 # (..., top_K, 8)

    if clamp:
        out = out.clamp(0.0, 1.0)
    return out


def make_rotation_translation_affine(
    angle_rad: float,
    tx: float = 0.0,
    ty: float = 0.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """[test/reference helper] rotation + translation の 2D affine matrix を生成。

    Center 周りの rotation → translation。RandomAffine (torchvision) の per-sample
    matrix を再現する簡易実装、Phase G の integration test / unit test で使用予定。

    Args:
        angle_rad: 回転角 (radian、positive = 反時計回り)
        tx, ty:    translation (normalized [0,1] 想定なら 0.05 = 5% shift)
        center:    回転中心 (default = 画像中心 (0.5, 0.5))

    Returns:
        (2, 3) affine matrix。normalized [0,1] 座標に適用可能。
    """
    import math

    cx, cy = center
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # T_back @ R @ T_to_origin  (center 周り rotation)
    # Then add tx/ty translation on top.
    # Composed:
    #   x' = cos*x - sin*y + (cx - cx*cos + cy*sin) + tx
    #   y' = sin*x + cos*y + (cy - cx*sin - cy*cos) + ty
    return torch.tensor(
        [
            [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a + tx],
            [sin_a, cos_a, cy - cx * sin_a - cy * cos_a + ty],
        ],
        dtype=torch.float32,
    )
