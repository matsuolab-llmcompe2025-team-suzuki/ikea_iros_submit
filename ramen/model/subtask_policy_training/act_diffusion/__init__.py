"""ACT / Diffusion Policy (LeRobot 内蔵 policy) 固有の smoke / run 設定 (Issue #139)。

学習 pipeline 本体 (materialize / resolve / train_lerobot.sh / frame cache) は policy 非依存で
`scripts/` にあり、ここには ACT / DP でしか使わないものだけを置く。
"""
