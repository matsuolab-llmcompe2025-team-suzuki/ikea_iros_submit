"""pick_table_leg の hybrid (VLM → 学習 policy → G1 IK MP → ルールの持ち替え)。"""

# 区間 1 (掴むまで) を任せてよい学習 policy (policy_config.yaml の policies: の key)。先頭が既定。
# 区間 1 は学習 policy の skill (VlaSkill) に step を任せ、hybrid が見るのは手の指令だけなので policy の種類は問わない。
# - groot_pick_legs_v1: 実機で確かめた本番の既定 (Issue #148)
# - pick_table_leg_ramen_ori_venue_all6_20k: 会場の遅い腕に慣らした RAMEN-Ori (Issue #188)。hybrid の中では実機で未確認
HYBRID_PHASE1_VARIANTS = (
    "groot_pick_legs_v1",
    "pick_table_leg_ramen_ori_venue_all6_20k",
)
