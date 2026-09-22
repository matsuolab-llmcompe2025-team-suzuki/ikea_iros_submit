"""`boundary/` が運営の配布物そのままであることを固定する。

運営は 2026-09-22 の回答で明言している:

    "Please do not modify the organizer code in your submission."

`boundary/` は運営 interface package の `boundary/` をそのまま置いたもので、
`:5555` / `:5556` / `:5557` の wire 形式を決めている。ここを我々が書き換えると、

  - 運営の adapter と **無言で**食い違う (msgpack は型が合えば通る)
  - 失格の根拠になる

の両方が起きる。

**ハッシュの出所は運営の公開物であって、我々の現物ではない。**
自分の今のファイルから採った値を固定すると「今のファイルは今のファイルと同じ」
という円環になり、既に混入している改変を正しいものとして固定してしまう。

    採取元   github.com/iacevaltest/iros_g1_orin_package
             VERSION 2026.09.21 / commit e92f9b6815a127090e7c9d78760ba82953a1b1c0
    採取日   2026-09-22 (Issue #154 の再確認 Phase 0)

⚠️ **上流が新版を出したときは、このテストは通ったまま古い版に固定される。**
それはここでは検出できない (CI に運営 repo は無い)。上流の更新確認は手で行う:

    gh repo clone iacevaltest/iros_g1_orin_package /tmp/organizer_upstream
    diff -r <手元の package> /tmp/organizer_upstream -x .git

上流が動いていたら VERSION を確認し、**定数ごと**更新する。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_BOUNDARY = Path(__file__).resolve().parents[3] / "boundary"

#: 運営 package 2026.09.21 (commit e92f9b6) の `boundary/` の sha256。
ORGANIZER_BOUNDARY_SHA256 = {
    "__init__.py": "f4bafe5ea93c635b529f57f6b45d9051130b68fe506b181b7e33ef2542436613",
    "actions.py": "e86f6a3c9edcf21db56fb9ebe4480e013776b716c49a3b60c42d845fda237d45",
    "cameras.py": "9e41044ee75459bad6320405999bf5ad2d241295424145952ec8c830cf8c03f0",
    "states.py": "9d79af1995d2f9ddea770e1e5e4a6ff13219f3856a9996a12a47e45c47cde258",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(ORGANIZER_BOUNDARY_SHA256))
def test_boundary_file_is_unmodified_organizer_code(name: str) -> None:
    path = _BOUNDARY / name
    assert path.is_file(), f"boundary/{name} が無い"
    assert _sha256(path) == ORGANIZER_BOUNDARY_SHA256[name], (
        f"boundary/{name} が運営の配布物と違う。**運営コードの改変は禁止**"
        " (運営回答 2026-09-22)。会場で一時的にいじったまま戻し忘れていないか"
        " 確認すること。上流が新版を出したのなら VERSION を確認して"
        " ORGANIZER_BOUNDARY_SHA256 ごと更新する"
    )


def test_no_extra_files_were_added_to_boundary() -> None:
    """余計なファイルを置かないこと。

    ハッシュ照合は**宣言したファイルしか見ない**ので、`boundary/helpers.py` の
    ような追加ファイルはそれだけでは素通りする。運営の配布物は 4 ファイル。
    """
    found = {
        p.name
        for p in _BOUNDARY.iterdir()
        if p.is_file() and not p.name.endswith(".pyc")
    }
    assert found == set(ORGANIZER_BOUNDARY_SHA256), (
        f"boundary/ の構成が運営の配布物と違う: {sorted(found)}。"
        " 我々のコードは components/ 配下に置くこと"
    )
