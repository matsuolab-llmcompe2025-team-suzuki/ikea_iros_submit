"""Learned policies (VLA) の loader 集合 (Issue #125、C-axis inference integration)。

`skills/` (rule-based walk / arm pose 等) と兄弟。GR00T / RAMEN-Ori を "learned
low-level policy" として wrap し、dispatcher 経由で rule vs learned を切替可能に
する。

# 位置付け

- `base.py`: Policy protocol + Observation / PolicyAction / PolicyConfig / CameraKey
- `groot.py`: GR00T loader (Phase 2、issue #125)
- `ramen_ori.py`: RAMEN-Ori loader (Phase 3、3 mode = none / precomputed_token /
   overlay、issue #125)

# 訓練時仕様の厳密追従

Preprocessing (resize / normalize / OBB tokenization / state derive) は
`model/ramen_ori/data_lerobot.py` と `model/ramen_ori/state_derive.py` を単一の
真実として扱う。実装差異が発生した場合は inference 側を training 側に合わせる
(逆は行わない)。
"""
