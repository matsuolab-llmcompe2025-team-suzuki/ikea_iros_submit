"""Team RAMEN の decoupled-lane policy 実装 (GR00T → (T,25) task-space)。

- `g1_urdf_fk` / `taskspace_adapter`: iros_2026_ramen から vendor した pure-numpy
  FK + action adapter (self-contained Thor image のため複製)。
- `policy`: boundary `components/server.py` の Policy を差し替える実装。model 推論を
  `InferenceBackend` 抽象の背後に置き、obs → 絶対 body29 → FK → (T,25) の plumbing を
  GPU 無しで検証可能にする。
"""
