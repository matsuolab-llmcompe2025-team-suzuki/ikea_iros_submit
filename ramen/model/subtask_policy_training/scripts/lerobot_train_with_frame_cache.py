"""Wrapper: JPG frame cache patch を install してから lerobot-train を起動 (Issue #122)。

Env var ``LEROBOT_FRAME_CACHE_ENABLE=true`` が set されている時のみ patch が active
になる (未設定なら install 済でも full fallback = 挙動不変)。train_lerobot.sh が
lerobot-train CLI の代わりに この module を invoke する。

# 使い方
    LEROBOT_FRAME_CACHE_ENABLE=true \\
    python -m model.subtask_policy_training.scripts.lerobot_train_with_frame_cache \\
        [-- lerobot-train CLI args]

# 挙動
    1. apply_patch() で decode_video_frames を monkey-patch (LeRobot 0.5.1/0.6.0 両対応)
    2. Issue #122 D-4: OBB_OVERLAY_ENABLE=true なら obb_overlay_setup.setup_from_env() で
       OverlayRenderer を post-decode hook として register (GR00T 等、model 側 config を
       持たない policy でも overlay を効かせるため)。RAMEN-Ori は自身の dataloader で
       register するので本 wrapper 経由でなくても overlay 動くが、GR00T 経路はここで
       register する必要がある。
    3. lerobot.scripts.lerobot_train.main() を invoke (entry_points.txt の lerobot-train と同 entry)
"""

from __future__ import annotations

import sys


def main() -> None:
    # Patch を lerobot import 前に apply (module attribute への差替なので import 後でも
    # 動くが、明示的に前に置くことで load 順の意図を明確化)
    from model.subtask_policy_training.scripts.lerobot_frame_cache_patch import apply_patch

    applied = apply_patch()
    if not applied:
        print(
            "[warn] lerobot_frame_cache_patch.apply_patch() failed "
            "(lerobot import 不可?)、frame cache 無効で継続",
            file=sys.stderr,
        )

    # Issue #122 D-4: OBB overlay hook を env driven で register (GR00T 等の wrapper 経由学習で有効)。
    # env 未 set なら no-op (backward compat)。setup 内で必要 env の validation が走る。
    from model.subtask_policy_training.scripts.obb_overlay_setup import setup_from_env

    setup_from_env()

    # entry_points.txt: lerobot-train = lerobot.scripts.lerobot_train:main
    from lerobot.scripts.lerobot_train import main as _lerobot_train_main

    _lerobot_train_main()


if __name__ == "__main__":
    main()
