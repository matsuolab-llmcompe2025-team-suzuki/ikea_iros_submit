"""読み終わった重みの file をページキャッシュから落とす (Issue #188 ② 段 2)。

`--gpu-models plan` は model を 1 本ずつ読む。GB10・Thor はメモリが GPU と共有 (統合メモリ)
なので、読んだ重みの file がページキャッシュに残ると、その分だけ次の model が使えるメモリを
圧迫する。1 本読み終わるたびに `posix_fadvise(DONTNEED)` で、その file のページを手放す。

- 落とすのは今どこにも map されていないページだけ (map 中のページは kernel が残す)。
  別の model が同じ file を後で読むときは、ディスクから読み直すだけ。
- `posix_fadvise` が無い OS (macOS) では何もしない。
- 重みの場所は `ckpt_ref` から決める: `repo@revision` は HF の cache の snapshot の中、
  それ以外は手元の path (file か directory)。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

_COMMIT = re.compile(r"[0-9a-f]{40}")


def hf_hub_cache_dir() -> Path:
    """HF の cache の場所 (`huggingface_hub` と同じ決め方)。"""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        pass
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "huggingface" / "hub"


def hf_snapshot_files(
    repo_id: str, revision: Optional[str] = None, *, cache_dir: Optional[Path] = None
) -> list[Path]:
    """cache にある、その repo・revision の file (symlink の先の実体)。無ければ空。"""
    root = (cache_dir or hf_hub_cache_dir()) / f"models--{repo_id.replace('/', '--')}"
    commit = revision
    if commit is None or not _COMMIT.fullmatch(commit):
        ref = root / "refs" / (revision or "main")
        try:
            commit = ref.read_text(encoding="utf-8").strip()
        except OSError:
            return []
    snapshot = root / "snapshots" / commit
    return _files_under(snapshot)


def ckpt_ref_files(ckpt_ref: str, *, cache_dir: Optional[Path] = None) -> list[Path]:
    """`ckpt_ref` ("repo@revision" か手元の path) の重みの file。"""
    local = Path(ckpt_ref).expanduser()
    if local.exists():
        return _files_under(local)
    repo_id, _, revision = ckpt_ref.partition("@")
    return hf_snapshot_files(repo_id, revision or None, cache_dir=cache_dir)


def drop_page_cache(paths: Iterable[Path]) -> tuple[int, int]:
    """file のページキャッシュを手放す。(落とした file の数, その大きさの合計 [byte]) を返す。"""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return 0, 0
    count = size = 0
    for path in dict.fromkeys(Path(p) for p in paths):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            fadvise(fd, 0, 0, dontneed)
            size += os.fstat(fd).st_size
            count += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return count, size


def _files_under(path: Path) -> list[Path]:
    """file ならそれ、directory なら中の file 全部 (symlink は先の実体に直す)。"""
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        return []
    files = []
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in sorted(filenames):
            candidate = (Path(dirpath) / name).resolve()
            if candidate.is_file():
                files.append(candidate)
    return files
