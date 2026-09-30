import sys

import pytest

from tools.gb10 import operator_probe
from tools.gb10.operator_probe import (
    choice_reply, full_stage_plan, initial_lift_cm, lift_values, owned_groot_worker,
)


def test_full_plan_covers_all_leg_rounds_and_flip():
    plan = full_stage_plan()
    assert list(plan) == [1, 2, 3, 4, 5]
    assert sum(map(len, plan.values())) == 16
    assert plan[1][0] == "pick_table_leg"
    assert plan[5] == ["flip_table"]
    assert all(plan[s] == plan[2] for s in (3, 4))


def process(root, pid, children, executable, *args):
    path = root / str(pid)
    (path / "task" / str(pid)).mkdir(parents=True)
    (path / "task" / str(pid) / "children").write_text(" ".join(map(str, children)))
    (path / "cmdline").write_bytes(b"\0".join(value.encode() for value in (executable, *args)) + b"\0")


MODULE = "inference.desktop.lower_policy.policies.groot_worker"


def test_only_descendant_python_worker_is_selected(tmp_path):
    process(tmp_path, 1, [2], "/usr/bin/python3", "pty_run.py")
    process(tmp_path, 2, [3], "/usr/bin/pixi", "python", "-m", MODULE)
    process(tmp_path, 3, [], "/runtime/bin/python", "-m", MODULE)
    process(tmp_path, 99, [], "/runtime/bin/python", "-m", MODULE)
    assert owned_groot_worker(1, tmp_path) == 3


def test_multiple_workers_fail_closed(tmp_path):
    process(tmp_path, 1, [2, 3], "/usr/bin/python3", "pty_run.py")
    for pid in (2, 3):
        process(tmp_path, pid, [], "/runtime/bin/python", "-m", MODULE)
    with pytest.raises(RuntimeError, match="exactly one"):
        owned_groot_worker(1, tmp_path)


def test_no_worker_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="exactly one"):
        owned_groot_worker(1, tmp_path)


def test_worker_spawned_by_non_main_thread(tmp_path):
    process(tmp_path, 1, [], "/usr/bin/pixi")
    thread = tmp_path / "1" / "task" / "8"
    thread.mkdir()
    (thread / "children").write_text("2")
    process(tmp_path, 2, [], "/runtime/bin/python", "-m", MODULE)
    assert owned_groot_worker(1, tmp_path) == 2


def test_start_lift_heights_come_from_the_init_line_and_each_key():
    """開始待ちの U / D (本体 #188): 起動時の元の高さと、押すたびの高さを読む。"""
    text = (
        "[init] pick_table_leg: start height keys U/D (右手 +2.5 cm（元の開始姿勢から。U で 1 cm 上げる・"
        "D で下げる、0〜8 cm）)\r\npick_table_leg／開始待ち\r\n[start-lift] 右手 +2.5 cm（…）\r\n"
        "[start-lift] 右手 +3.5 cm（…）\r\n[start-lift] 両手（右手の量） +4.0 cm（…）\r\n"
    )
    assert initial_lift_cm(text, "pick_table_leg") == 2.5
    assert initial_lift_cm(text, "flip_table") is None
    values = lift_values(text)
    assert [cm for cm, _ in values] == [2.5, 3.5, 4.0]
    assert [cm for cm, _ in lift_values(text, values[0][1])] == [3.5, 4.0]


def test_choice_replies_skip_the_status_line():
    """R の後の数字キーへの返事 (本体 #188)。画面の状態の 1 行は返事と取り違えない。"""
    skill = "rotate_table_base"
    status = f"[model] {skill}: いま a ｜ 0 読み込み済み ｜ 1 読み込み中\r\n"
    assert choice_reply(status, skill) is None
    loading = status + f"[model] {skill}: 1 b はまだ読み込み中（読むのは腕が止まっている間だけ。…）\r\n"
    kind, line, end = choice_reply(loading, skill)
    assert kind == "loading" and line.startswith(f"[model] {skill}: 1 b は")
    switched = loading + status + f"[model] {skill}: a → b に切り替えた（次の試行から）\r\n"
    assert choice_reply(switched, skill, end)[0] == "accepted"
    assert choice_reply(f"[model] {skill}: a のまま\r\n", skill)[:2] == ("accepted", f"[model] {skill}: a のまま")
    assert choice_reply(f"[model] {skill}: 1 b は読み込みに失敗したので選べない\n", skill)[0] == "refused"
    assert choice_reply("[model] pick_table_leg: a → b に切り替えた\n", skill) is None


@pytest.mark.parametrize("argv", [
    ["--case", "choose"],                                   # stage が要る
    ["--case", "choose", "--stage", "0"],                   # Stage 0 に policy の skill は無い
    ["--case", "lift", "--stage", "2", "--skill", "flip_table"],
    ["--case", "stage", "--stage", "2", "--skill", "pick_table_leg"],
    ["--case", "choose", "--stage", "2", "--choice-key", "0"],  # 0 は既定。候補の数字だけ
])
def test_choose_and_lift_arguments_fail_before_anything_starts(monkeypatch, tmp_path, argv):
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["operator_probe", *argv, "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        operator_probe.main()
    assert exc.value.code == 2
    assert not output.exists()
