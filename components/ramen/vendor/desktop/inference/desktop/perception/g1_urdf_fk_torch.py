"""GR00T 19-dim action → 両手 wrist 3D 位置 の torch autograd 対応 FK (Issue #129 Phase 0 D0-2)。

L4 (FK anchor loss) のために、predicted action `q_pred` (GR00T 19-dim、torch tensor with
`requires_grad=True`) を URDF FK に通して left/right wrist 3D 座標を出す。gradient は
autograd で伝播、joint 角度誤差 → 3D 位置誤差の chain rule が自然に成立。

# 既存 `g1_urdf_fk.py` (numpy) との関係

- Chain parse (`_parse_urdf_chain`) と constants (`G1_JOINT_NAMES`, `LEFT_WRIST_LINK`,
  `RIGHT_WRIST_LINK`, `DEFAULT_URDF_PATH`) は既存 numpy module を import して再利用。
- Init 時に numpy chain data を torch buffer に変換、per-forward は pure torch matmul chain。
- **既存 numpy version には無変更** = Issue #125 で merged 済 inference stack への regression risk ゼロ。

# GR00T 19-dim 空間との対応

GR00T action は `model.subtask_policy_training.joint_layout.GROOT_ACTION_NAMES` の
19-dim layout (waist 3 + left arm 7 + right arm 7 + gripper 2)。SDK 29-joint 空間との
mapping は `GROOT_TO_SDK29_INDEX`。

wrist FK chain 内 joint (waist 3 + arm 7 = 10) は全て GR00T 覆域内 (legs 12 dim は wrist
chain 不参加、gripper 2 dim も同様)。ここでは chain step ごとに直接 GR00T index を
参照して angle を取り出す (SDK 29 padding 不要)。

# Shape

    forward(q_groot):
        q_groot: (..., 19)   torch.float, requires_grad OK
        returns: (left_pos, right_pos) each (..., 3)   pelvis frame 座標
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# 既存 numpy 経路を import (chain parse + constants + tool offset を再利用)。
# Issue #134: LEFT/RIGHT_WRIST_TOOL_OFFSET_M は numpy 側の single source of truth。
from inference.desktop.perception.g1_urdf_fk import (
    DEFAULT_URDF_PATH,
    G1_JOINT_NAMES,
    LEFT_WRIST_LINK,
    LEFT_WRIST_TOOL_OFFSET_M,
    RIGHT_WRIST_LINK,
    RIGHT_WRIST_TOOL_OFFSET_M,
    ChainJoint,
    _parse_urdf_chain,
)
from model.subtask_policy_training.joint_layout import GROOT_TO_SDK29_INDEX


def _axis_angle_matrix_torch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula。axis (3,) unit + angle (...,) → R (..., 3, 3)。

    numpy 版 `_axis_angle_matrix` の torch equivalent、autograd 対応。
    axis は非 unit 想定でも正規化して安全化 (chain init 時に normalize 済想定だが念のため)。
    """
    norm = torch.linalg.norm(axis)
    if float(norm) < 1e-9:
        # 縮退 axis (回転無し扱い) → identity を batch shape で返す
        eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
        return eye.expand(*angle.shape, 3, 3).clone()
    a = axis / norm
    x, y, z = a[0], a[1], a[2]

    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1.0 - c

    # 各要素 shape (...,)、最後 stack で (..., 3, 3)
    row0 = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], dim=-1)
    row1 = torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], dim=-1)
    row2 = torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _to_4x4(R: torch.Tensor) -> torch.Tensor:
    """R (..., 3, 3) → T (..., 4, 4) with zero translation, bottom row [0,0,0,1]。"""
    batch_shape = R.shape[:-2]
    T = torch.zeros(*batch_shape, 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., 3, 3] = 1.0
    return T


def _sdk29_to_groot_index(sdk_idx: int) -> int:
    """SDK 29-joint index → GR00T 19-dim index の reverse lookup。

    Raises:
        ValueError: SDK idx が GR00T 覆域外 (legs 0-11、GR00T action に含まれない)。
    """
    for groot_idx, s in enumerate(GROOT_TO_SDK29_INDEX):
        if s == sdk_idx:
            return groot_idx
    raise ValueError(
        f"SDK joint index {sdk_idx} ({G1_JOINT_NAMES[sdk_idx]}) is not covered by "
        f"GR00T 19-dim action layout (legs 0-11 は wrist chain 不参加、gripper は SDK 外)"
    )


class G1WristFKTorch(nn.Module):
    """GR00T 19-dim → 両手 wrist 3D 位置 (pelvis frame) の torch FK module。

    - init: URDF file を parse (既存 numpy) → chain の fixed_T / axis を torch buffer に
      変換 + chain 各 step の GR00T index を precompute。
    - forward: q_groot (..., 19) を受け取り、chain step ごとに R_joint 計算 → 4x4 matmul
      chain → 左右 wrist の (..., 3) 位置を返す。

    Buffer:
        _left_fixed_Ts:    (N_left, 4, 4) 各 chain step の parent → joint frame fixed transform
        _left_axes:        (N_left, 3)    各 chain step の joint rotation axis (unit)
        _left_groot_idx:   (N_left,) long 各 chain step が使う GR00T action index
        _left_tool_offset: (3,) wrist_yaw_link frame での tool point offset (Issue #134、
                           numpy G1WristFK と意味論一致 = Dex1 grasp point)
        _right_*: 右手側同様
    """

    def __init__(
        self,
        left_chain: Sequence[ChainJoint],
        right_chain: Sequence[ChainJoint],
        left_tool_offset: torch.Tensor | np.ndarray | None = None,
        right_tool_offset: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        left_fixed_Ts, left_axes, left_groot_idx = self._chain_to_tensors(left_chain, dtype)
        right_fixed_Ts, right_axes, right_groot_idx = self._chain_to_tensors(right_chain, dtype)

        self.register_buffer("_left_fixed_Ts", left_fixed_Ts)
        self.register_buffer("_left_axes", left_axes)
        self.register_buffer("_left_groot_idx", left_groot_idx)
        self.register_buffer("_right_fixed_Ts", right_fixed_Ts)
        self.register_buffer("_right_axes", right_axes)
        self.register_buffer("_right_groot_idx", right_groot_idx)

        # Issue #134: numpy G1WristFK と意味論を揃えるため tool_offset を適用。
        # None なら module default (LEFT/RIGHT_WRIST_TOOL_OFFSET_M) を tensor 化。
        # persistent=False: 定数なので checkpoint に保存しない (既 ckpt との互換維持、
        # 学習 param でも学習中更新される state でもない)。
        self.register_buffer(
            "_left_tool_offset",
            self._validate_offset(left_tool_offset, LEFT_WRIST_TOOL_OFFSET_M, "left_tool_offset", dtype),
            persistent=False,
        )
        self.register_buffer(
            "_right_tool_offset",
            self._validate_offset(right_tool_offset, RIGHT_WRIST_TOOL_OFFSET_M, "right_tool_offset", dtype),
            persistent=False,
        )

    @staticmethod
    def _validate_offset(
        value: torch.Tensor | np.ndarray | None,
        default: np.ndarray,
        name: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """None → default、値有りなら shape (3,) + finite verify して tensor 化。"""
        if value is None:
            return torch.tensor(default, dtype=dtype)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(dtype=dtype)
        else:
            tensor = torch.tensor(np.asarray(value), dtype=dtype)
        if tensor.shape != (3,):
            raise ValueError(f"{name} must be shape (3,), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite, got {tensor.tolist()}")
        return tensor

    @staticmethod
    def _chain_to_tensors(
        chain: Sequence[ChainJoint], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fixed_Ts = torch.stack(
            [torch.tensor(cj.fixed_T, dtype=dtype) for cj in chain], dim=0
        )
        axes = torch.stack([torch.tensor(cj.axis, dtype=dtype) for cj in chain], dim=0)
        groot_idx = torch.tensor(
            [_sdk29_to_groot_index(cj.joint_index) for cj in chain], dtype=torch.long
        )
        return fixed_Ts, axes, groot_idx

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path = DEFAULT_URDF_PATH,
        joint_names: Sequence[str] = G1_JOINT_NAMES,
        left_tool_offset: torch.Tensor | np.ndarray | None = None,
        right_tool_offset: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "G1WristFKTorch":
        joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
        left_chain = _parse_urdf_chain(urdf_path, LEFT_WRIST_LINK, joint_name_to_index)
        right_chain = _parse_urdf_chain(urdf_path, RIGHT_WRIST_LINK, joint_name_to_index)
        return cls(
            left_chain=left_chain,
            right_chain=right_chain,
            left_tool_offset=left_tool_offset,
            right_tool_offset=right_tool_offset,
            dtype=dtype,
        )

    def _fk_chain_positions(
        self,
        q_groot: torch.Tensor,
        fixed_Ts: torch.Tensor,
        axes: torch.Tensor,
        groot_idx: torch.Tensor,
        tool_offset: torch.Tensor,
    ) -> torch.Tensor:
        """chain 経由で pelvis → target_link + tool_offset の 3D 位置を計算 (batch 対応)。

        Issue #134: `tool_offset` を wrist_yaw_link frame で加算 (numpy `G1WristFK.
        compute_ee_state` と同じ order: `pos = origin + rot @ offset`)、意味論 = Dex1
        grasp point。

        Args:
            q_groot:     (..., 19) GR00T action
            fixed_Ts:    (N, 4, 4)
            axes:        (N, 3)
            groot_idx:   (N,) long
            tool_offset: (3,) wrist_yaw_link frame の tool point offset (buffer 経由)

        Returns:
            pos: (..., 3) pelvis frame 座標
        """
        # chain step 毎に該当 GR00T angle を index_select
        # (..., N) の angle tensor
        angles = torch.index_select(q_groot, dim=-1, index=groot_idx)

        batch_shape = q_groot.shape[:-1]
        # T の accumulator を identity で init。batch shape 分 expand + clone。
        T = torch.eye(4, dtype=q_groot.dtype, device=q_groot.device)
        T = T.expand(*batch_shape, 4, 4).clone() if batch_shape else T.clone()

        N = int(fixed_Ts.shape[0])
        for i in range(N):
            angle_i = angles[..., i]  # (...,)
            R_i = _axis_angle_matrix_torch(axes[i], angle_i)  # (..., 3, 3)
            T_joint = _to_4x4(R_i)  # (..., 4, 4)
            # fixed_Ts[i] は (4, 4)、broadcast で (..., 4, 4) と matmul
            step_T = fixed_Ts[i] @ T_joint
            T = T @ step_T

        # tool_offset apply: pos = origin + rot @ offset (numpy 側と同じ order)
        # rot (..., 3, 3) @ offset (3,) → (..., 3)、torch matmul は最後 2 dim を
        # matrix-vector product として扱う (numpy と同じ semantics)
        origin = T[..., :3, 3]                                       # (..., 3)
        rot = T[..., :3, :3]                                         # (..., 3, 3)
        return origin + torch.matmul(rot, tool_offset.to(rot.dtype)) # (..., 3)

    def forward(self, q_groot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GR00T 19-dim action → (left_wrist_pos, right_wrist_pos) 各 (..., 3)。

        Args:
            q_groot: (..., 19) torch tensor、任意 batch shape、requires_grad OK

        Returns:
            (left_pos, right_pos): 各 (..., 3) tensor、pelvis frame 座標。
        """
        if q_groot.shape[-1] != 19:
            raise ValueError(f"q_groot last dim must be 19 (GR00T), got {q_groot.shape}")
        left_pos = self._fk_chain_positions(
            q_groot,
            self._left_fixed_Ts,
            self._left_axes,
            self._left_groot_idx,
            self._left_tool_offset,
        )
        right_pos = self._fk_chain_positions(
            q_groot,
            self._right_fixed_Ts,
            self._right_axes,
            self._right_groot_idx,
            self._right_tool_offset,
        )
        return left_pos, right_pos
