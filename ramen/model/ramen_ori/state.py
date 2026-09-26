"""E-δ State encoder module (RAMEN-Ori、Issue #115)。

Proprioception (joint + tracking_err + velocity + hand + EE pose) 71D を 1 token に
圧縮して Fusion transformer に流す。Dataset B に torque/force が無い前提 (design doc
§4.3 前提) の代替 signal として、tracking_err と velocity で "力覚っぽさ" を近似する。

# Shape

- input: (B, state_dim)、state_dim=71 default
    - joint 19D + tracking_err 19D (1 frame 前の指令 - q_current) + velocity 19D
      (2 frame 差分) + hand_state 2D + EE pose 12D
- output: (B, 1, d_model)、Fusion に concat する 1 token

# Design

    [LayerNorm(state_dim)] → Linear(state_dim, hidden) → GELU → Linear(hidden, d_model)

- 入力 LN (`input_norm=True`、Phase K まで): 71D の各 dim は scale がバラバラ (joint rad ±1.6、
  hand 0〜4.5、EE m+rad 混在) なので sample ごとの LN で揃えていた。ただし LN の割る数がほぼ hand で
  決まり、hand の開閉で腕の数字の意味が変わる (Issue #141 RO-6)。
  Issue #141 以降は model の Normalizer (`normalization.py`) で次元ごとに z-score し、LN は外す
  (`input_norm=False`)。既定は Phase K の ckpt と同じ構造 (推論が base.yaml から組み立てるため)
- GELU: Fusion transformer (G-1 Standard self-attn) と activation を揃える
- hidden_dim default = d_model (param ~300K、latency 無視できる)
- init: nn.Linear default (kaiming_uniform)、embedding のような特別 scale 化は不要
  (Fusion 入力段の LN が Fusion 側 attention の入力を再正規化する前提)

# dropout (Issue #141 Phase 6、変種「state の使い方」)

`dropout={group: 確率}` を渡すと、train mode のときだけ sample ごと・group ごとに独立に state を隠す
(値を 0 = 正規化後の平均にする)。model が自分の姿勢の続きを出すことに頼って画像を見なくなる癖 (copycat) を抑える狙い。
隠したかの印 (0/1) を group ごとに 1 個、state の後ろに足す (隠した 0 と本当に平均だった値を区別するため)。
eval mode (val と推論) では隠さず、印は 0。推論に渡す state は 71D のまま。
group の範囲は `state_derive.STATE71_DROPOUT_GROUPS`。0 を平均として使うので、model の正規化と組み、入口の LN とは組まない。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ramen_ori.state_derive import RAMEN_ORI_STATE_DIM, STATE71_DROPOUT_GROUPS


class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 71,
        d_model: int = 512,
        hidden_dim: int | None = None,
        input_norm: bool = True,
        dropout: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model
        self.state_dim = state_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # 隠す group (STATE71_DROPOUT_GROUPS の並び)、確率、group ごとに隠す次元の mask
        self.dropout_groups: tuple[str, ...] = ()
        if dropout:
            unknown = sorted(set(dropout) - set(STATE71_DROPOUT_GROUPS))
            if unknown:
                raise ValueError(f"state dropout の group {unknown} は無い (group: {list(STATE71_DROPOUT_GROUPS)})")
            if state_dim != RAMEN_ORI_STATE_DIM:
                raise ValueError(f"state dropout の group は 71D の並び (state_dim={state_dim})")
            if input_norm:
                raise ValueError("state dropout は正規化後の 0 を平均として使うので、入口の LayerNorm (input_norm) とは組まない")
            if not all(0.0 <= p <= 1.0 for p in dropout.values()):
                raise ValueError(f"state dropout の確率は 0〜1: {dict(dropout)}")
            self.dropout_groups = tuple(g for g in STATE71_DROPOUT_GROUPS if g in dropout)
            mask = torch.zeros(len(self.dropout_groups), state_dim, dtype=torch.bool)
            for i, group in enumerate(self.dropout_groups):
                for sl in STATE71_DROPOUT_GROUPS[group]:
                    mask[i, sl] = True
            self.register_buffer("_dropout_mask", mask, persistent=False)
            self.register_buffer(
                "_dropout_prob",
                torch.tensor([float(dropout[g]) for g in self.dropout_groups]),
                persistent=False,
            )

        self.input_ln = nn.LayerNorm(state_dim) if input_norm else nn.Identity()
        self.fc1 = nn.Linear(state_dim + len(self.dropout_groups), hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.input_ln(state)
        if self.dropout_groups:
            B = x.shape[0]
            if self.training:
                hide = torch.rand(B, len(self.dropout_groups), device=x.device) < self._dropout_prob
                hidden_dims = (hide[:, :, None] & self._dropout_mask[None]).any(dim=1)
                x = torch.cat([x.masked_fill(hidden_dims, 0.0), hide.to(x.dtype)], dim=-1)
            else:
                x = torch.cat([x, x.new_zeros(B, len(self.dropout_groups))], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.unsqueeze(1)
