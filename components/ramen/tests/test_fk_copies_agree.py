"""submit 内に **2 つある** G1WristFK が同じ数値を出すことを固定する。

    components/ramen/g1_urdf_fk.py
        手動コピー。`policy.py` / `groot53_worker.py` が import する。
        単体 skill 経路 (`RAMEN_POLICY=groot_pick_real` / `groot_53d_real`) の
        `ee_state` はこちらで計算される。

    components/ramen/vendor/.../perception/g1_urdf_fk.py
        `tools/sync_vendor_desktop.sh` が本体から自動同期する。
        orchestrator 経路 (`groot_orchestrator`) は vendored `vla_skill` 経由で
        こちらを使う。

手動コピーが存在する理由は **URDF path の解決だけ**。上流は repo root からの
相対 (`parents[3]`) で、container では CWD が `/app` のため解決できず
FileNotFoundError で全 53D skill が起動失敗する (IAC eval 指摘)。手動コピーは
`assets/` を `__file__` 相対で持つ。

**その 1 点以外は同じ数値でなければならない。**

## なぜこの test が要るか (2026-09-22、Issue #7 Phase 1)

手動コピーは `sync_vendor_desktop.sh` の対象外なので、本体を更新しても
自動では追従しない。実際に **tool offset が古いまま放置されていた**:

```
手動コピー   WRIST_TOOL_OFFSET_M       = [0.05,  0.0,   0.0  ]   公称 5cm、左右共通
本体         LEFT_WRIST_TOOL_OFFSET_M  = [0.087, -0.034, 0.092]  LeRobot GT との
             RIGHT_WRIST_TOOL_OFFSET_M = [0.116, -0.057, -0.015] fit で決定 (#132)

→ compute_ee_state が 左 0.1048 m / 右 0.0885 m ずれていた
```

`ee_state` は model の **state 入力**なので、学習分布から 10cm 外れた値を
policy に渡していたことになる。エラーは出ず「なんとなく掴まない」としか見えない。

しかも `policy.py` には

    # 両者の compute_ee_state は数値一致 (20 random q で max diff 0.0) を確認済。

と書かれていた。**書かれた時点では正しく、その後本体だけが動いた。** コメントは
腐るのでテストにする。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_RAMEN = _HERE.parent
_ROOT = _RAMEN.parent.parent
_VENDOR = _RAMEN / "vendor" / "desktop"
for _p in (str(_VENDOR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, path: Path):
    """同名 module を 2 本同時に持つため、file 指定で個別に読む。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def copies():
    manual = _load("_fk_manual_copy", _RAMEN / "g1_urdf_fk.py")
    vendored = _load(
        "_fk_vendored_copy", _VENDOR / "inference/desktop/perception/g1_urdf_fk.py"
    )
    return manual, vendored


@pytest.fixture(scope="module")
def instances(copies):
    manual, vendored = copies
    # vendored 側の DEFAULT_URDF_PATH は container で解決できない (それがこの
    # 手動コピーの存在理由) ので、手動コピー側の URDF を明示的に渡す。
    return (
        manual.G1WristFK.from_urdf(),
        vendored.G1WristFK.from_urdf(manual.DEFAULT_URDF_PATH),
    )


@pytest.mark.parametrize(
    "const",
    ["LEFT_WRIST_TOOL_OFFSET_M", "RIGHT_WRIST_TOOL_OFFSET_M", "G1_JOINT_NAMES"],
)
def test_shared_constants_match(copies, const):
    """定数のズレは数値のズレより先に出る。落ちたら手動コピーを再同期する。"""
    manual, vendored = copies
    a, b = getattr(manual, const), getattr(vendored, const)
    assert np.array_equal(np.asarray(a), np.asarray(b)), (
        f"{const} が食い違っている: 手動コピー={a!r} / vendored={b!r}。"
        " components/ramen/g1_urdf_fk.py は sync_vendor_desktop.sh の対象外なので、"
        " 本体 inference/desktop/perception/g1_urdf_fk.py から手で同期すること"
        " (URDF path の局所適応だけは残す)"
    )


def test_compute_ee_state_agrees(instances):
    """model の **state 入力**。ここがズレると学習分布から静かに外れる。"""
    manual, vendored = instances
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-0.6, 0.6, size=29)
        worst = max(
            worst,
            float(
                np.abs(
                    np.asarray(manual.compute_ee_state(q), np.float64)
                    - np.asarray(vendored.compute_ee_state(q), np.float64)
                ).max()
            ),
        )
    assert worst == 0.0, (
        f"compute_ee_state が最大 {worst:.4f} ずれている。単体 skill 経路"
        " (RAMEN_POLICY=groot_pick_real / groot_53d_real) だけが学習分布から"
        " 外れた ee_state を policy に渡すことになる"
    )


def test_compute_ee_transforms_agrees(instances):
    """publish する `(T,25)` の EE。運営 IK が **zero tool offset** で解く側。"""
    manual, vendored = instances
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        q = rng.uniform(-0.6, 0.6, size=29)
        for a, b in zip(
            manual.compute_ee_transforms(q), vendored.compute_ee_transforms(q)
        ):
            worst = max(
                worst,
                float(
                    np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()
                ),
            )
    assert worst == 0.0, f"compute_ee_transforms が最大 {worst:.4e} ずれている"


def test_the_manual_copy_only_differs_in_the_urdf_path(copies):
    """**局所適応は URDF path 1 箇所だけ**であること。

    ここが増えると「何がどこまで同じか」が分からなくなり、今回のような
    サイレントなズレを見逃す。docstring / コメントの差は許す。
    """
    import ast
    import difflib

    def _code(path: Path) -> list[str]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body.pop(0)
                    if not body:
                        body.append(ast.Pass())
        return ast.dump(ast.fix_missing_locations(tree), indent=1).splitlines()

    manual = _code(_RAMEN / "g1_urdf_fk.py")
    vendored = _code(_VENDOR / "inference/desktop/perception/g1_urdf_fk.py")
    hunks = [
        line
        for line in difflib.unified_diff(manual, vendored, lineterm="", n=0)
        if line.startswith("@@")
    ]
    assert len(hunks) <= 2, (
        f"局所適応が {len(hunks)} 箇所に増えている ({hunks})。URDF path 以外の差は"
        " 手動コピーに入れないこと。必要なら上流を直して再同期する"
    )
