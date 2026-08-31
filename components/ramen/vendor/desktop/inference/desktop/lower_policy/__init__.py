"""Lower Policy: Wrapper が dispatch する skill 実行層。

構成:
    dispatcher.py       LowerPolicy Protocol 実装 (skill 名 → Skill instance)
    skills/             個別 skill (state machine のみ、SDK 依存なし)
    actuators/          actuation 経路 (Mock / SDK / 将来 Isaac Sim 等)
    scripts/            stand-alone script (実機依頼用等)
"""
