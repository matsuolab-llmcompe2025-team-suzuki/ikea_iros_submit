# 運営 IK と同じ運動学の URDF（publish 用 FK 専用）

`g1_29dof_with_hand.urdf` は **会場で `(T,25)` の EE を計算するときだけ**使う。
policy の state 入力（`compute_ee_state`）や hybrid pick の内部の運動学は、学習と同じ
`g1_29dof_mode_15_with_dex1_1.urdf` のまま使う。

## なぜ別の URDF が要るか

運営の IK（`reference/wbc_adapter/ik.py`）と WBC は `g1_29dof_with_hand.urdf` で解く。
運営機の実物（`~/g1_bridge/robot/assets/g1_urdf/g1_29dof_with_hand.urdf`、
2026-09-19 監査で取得）の md5 は `093c36ba3284c6cce5f2b62041626b79` で、
NVIDIA `GR00T-WholeBodyControl`（運営 pin の `a0732b6`）の同名ファイルと byte 一致する。

我々の `mode_15_with_dex1_1` とは腕の鎖が 4 か所違う（waist_roll / waist_pitch / 
shoulder_pitch / wrist_yaw の origin）。我々の URDF で FK した EE を渡すと、運営 IK は
「同じ手首位置」を別の関節角で作る。GB10 上で運営 `ik.py` を無改変で回した実測
（Issue #164）:

| publish に使う FK | 運営 IK accept | 手首位置のずれ (median) |
|---|---|---|
| mode_15（旧） | 90〜97% | 4.9 mm |
| この URDF | 95〜100% | 0.1 mm |

## 出所

- `unitreerobotics/unitree_ros` の `robots/g1_description/g1_29dof_with_hand.urdf`
  （commit `9926cc2f179ae3b86f4f74087bd32ef0c8b6fd90`、BSD-3-Clause、`LICENSE` 同梱）。
  このファイルの md5 は `2bad065c162186a19222ec81f99b91b6`。
- NVIDIA 版とは byte 一致しないが、**腰・腕の joint（origin・axis・limit）は全て一致**する。
  違いは手の親指の limit と IMU の frame 名だけで、FK（pelvis → wrist_yaw_link）の差は 0。
  NVIDIA の repo はライセンス表記が "Other" のため、同じ運動学の Unitree 版を置く。

## 会場で確かめること

PC2 で `md5sum ~/g1_bridge/robot/assets/g1_urdf/g1_29dof_with_hand.urdf` が
`093c36ba3284c6cce5f2b62041626b79` であること。違っていたら運営 IK の運動学が変わっている。
