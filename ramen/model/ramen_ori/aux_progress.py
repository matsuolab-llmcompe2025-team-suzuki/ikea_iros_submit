"""区間の進みの補助 head (Issue #183)。画像の token だけから「区間のどこまで来たか」(0〜1) を当てる。

flip_table の実機 (2026-09-27、wandb mwsku9iq) では、台が倒れきる前に左手を離して台が元に戻った。
デモでは「台が倒れた」ときと「腕がある姿勢まで上がった」ときがほぼ重なるので、model は読み取りやすい
腕の姿勢 (state) を合図にしがちになる。ここでは **画像の token だけ** から区間の進みを当てさせ、
画像の表現に「台がどこまで回ったか」を持たせる。

# 入力と出力

- 入力: `VisionEncoder` の token (B, N_cams * pool_size**2, d_model)。state・skill は見ない
- 出力: (B,) の進み (sigmoid で 0〜1)
- 正解: frame の区間内の位置 = frame_index / (区間の長さ − 1) (data の `progress_target: true` で batch に入る)

backbone は凍結なので、この head の勾配が効くのは `VisionEncoder.projection` (行動の経路と共有) と head 自身。
推論では使わない (`RamenOriPolicy.predict_action` は aux_head を呼ばない)。

# Loss (model.py 側)

`RamenOriPolicy.compute_loss_parts` で `aux_head.target_key` ("progress_target") が batch にあるとき:

    total_loss = action_loss + aux_weight * L1(progress_pred, progress_target)

L1 なので値はそのまま「進みの誤差の平均」(0.1 = 区間の 10%)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressHead(nn.Module):
    """画像の token を 1 個の学習する query で attention pooling し、MLP で進み (0〜1) を出す。

    Args:
        d_model: vision token の次元 (`model.d_model`)
        hidden_dim: MLP の中間の次元
        num_heads: pooling の attention の head 数 (d_model を割り切ること)
    """

    target_key = "progress_target"
    loss_name = "aux_progress"

    def __init__(self, d_model: int = 512, hidden_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} は num_heads={num_heads} で割り切れること")
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, vis_tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, d_model) → (B,) の進み (0〜1)。"""
        query = self.query.expand(vis_tokens.shape[0], -1, -1).to(vis_tokens.dtype)
        pooled, _ = self.attn(query, vis_tokens, vis_tokens, need_weights=False)   # (B, 1, d_model)
        return torch.sigmoid(self.mlp(pooled)[:, 0, 0])

    @staticmethod
    def loss(pred: torch.Tensor, batch: dict) -> torch.Tensor:
        """L1 (fp32 で計算)。"""
        return F.l1_loss(pred.float(), batch[ProgressHead.target_key].float())
