import pytest

from tools.gb10.operator_probe import owned_groot_worker


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
