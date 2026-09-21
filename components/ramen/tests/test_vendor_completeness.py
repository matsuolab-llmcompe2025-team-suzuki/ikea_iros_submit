"""vendor tree だけで推論コードが解決できることを固定する。

# なぜ要るか

提出 image は `docker/Dockerfile.thor.groot` の `COPY . ./` でこの repo を丸ごと焼く。
推論コードの本体は `iros_2026_ramen` 側にあり、ここはそのコピー
(`tools/sync_vendor_desktop.sh`)。

**`inference/` は `model/` に依存している。** 学習専用ではなく実行時依存で、
2026-09-21 時点で 13 箇所:

    chunk_executor.py     model.subtask_policy_training.gr00t.temporal_ensemble
    groot.py              dex1_hand_synergy / temporal_ensemble
    ramen_ori.py          temporal_ensemble / memory_features / fk /
                          relative_action / model / vision_backbone / build
    g1_urdf_fk_torch.py   model.subtask_policy_training.joint_layout

sync script は `model/` の一部しか運ばないので、**本体側に新しい `model.*` import が
増えると、気づかないまま欠けた image を焼く**ことになる。しかも症状は会場での
ImportError で、それまで一切見えない。実際 `joint_layout` が欠けていた
(2026-09-21 に発見)。

ここでは `inference/` 配下から出る `model.*` import だけを見る。`model/` の中の
学習コードが学習専用モジュールを import しているのは image には関係ない。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE.parent / "vendor"
_DESKTOP = _VENDOR / "desktop"
#: `groot_pick_legs.py` の VENDOR PATCH が repo root を 1 つ上まで遡るので、
#: `vendor/` 直下 (pick worker がここ) も解決先に含める。
_ROOTS = (_DESKTOP, _VENDOR)


def _model_imports_from_inference() -> list[tuple[str, Path]]:
    """vendored `inference/` の本番コードが張る `model.*` import を集める。"""
    found: list[tuple[str, Path]] = []
    for py in (_DESKTOP / "inference").rglob("*.py"):
        if "tests" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # vendored ツリーに壊れた file は無い想定
            continue
        rel = py.relative_to(_DESKTOP)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("model.")
            ):
                found.append((node.module, rel))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("model."):
                        found.append((alias.name, rel))
    return sorted(set(found))


def _resolves(module: str) -> bool:
    rel = Path(module.replace(".", "/"))
    for root in _ROOTS:
        if (root / rel.with_suffix(".py")).exists():
            return True
        if (root / rel / "__init__.py").exists():
            return True
    return False


def test_the_vendor_tree_actually_has_model_imports_to_check():
    """この test が空振りしていないこと (rglob が壊れたら気づけるように)。"""
    assert len(_model_imports_from_inference()) >= 10


@pytest.mark.parametrize(
    "module,source",
    _model_imports_from_inference(),
    ids=lambda v: str(v).replace("/", "."),
)
def test_every_model_import_from_inference_resolves_inside_the_vendor(module, source):
    """image に焼かれる tree だけで解決すること。

    落ちたら `tools/sync_vendor_desktop.sh` にその module を足す。**会場で初めて
    ImportError になる類**なので、ここで止める。
    """
    assert _resolves(module), (
        f"{source} が import する {module} が vendor に無い。"
        "tools/sync_vendor_desktop.sh に追加すること"
    )


def test_the_pick_worker_script_is_vendored():
    """pick が subprocess で spawn する実体が image に入っていること。

    `vendor/desktop` の外 (`vendor/model/`) にあり、2026-09-21 まで sync script の
    対象外だった。無いと pick が `FileNotFoundError` で 1 本も掴めない。
    """
    worker = (
        _VENDOR / "model/subtask_policy_training/deployment/real_groot_n17_worker.py"
    )
    assert worker.is_file(), f"pick worker が vendor に無い: {worker}"
    assert worker.stat().st_size > 1000
