"""vendor 同期の直後に再適用する Team RAMEN 固有パッチ。

`tools/sync_vendor_desktop.sh` が rsync の直後に呼ぶ。

# なぜ diff ファイルではなく Python か

パッチ当て先 (`iros_2026_ramen` の推論コード) は大会直前まで動く。行番号つきの
unified diff は近傍が 1 行動いただけで reject され、しかも「当たらなかった」ことに
気付かないまま image を焼く事故になりやすい。

ここでは **完全一致のアンカー文字列**で置換し、見つからなければその場で
`PatchError` を投げる。上流がアンカーを書き換えたら **同期が失敗して止まる**ので、
パッチが黙って消えることはない。既に適用済みなら skip する (冪等)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


class PatchError(RuntimeError):
    """アンカーが見つからない / 複数ある。上流の変更に追従が必要。"""


@dataclass(frozen=True)
class Patch:
    """vendor tree 内の 1 ファイルに対する 1 箇所の置換。"""

    path: str  # vendor/desktop/ からの相対 path
    why: str  # 失敗時に何を直せばよいか分かる説明
    before: str
    after: str


# container は pixi を持たない。本家は GR00T worker を `pixi run` で起動するので、
# そのままでは 53D skill が FileNotFoundError で全滅する。lerobot 0.6.1 の python を
# 環境変数 RAMEN_WORKER_PYTHON_53D で受け取って直接 exec する経路を足し、
# pixi 経路は fallback に落とす (ラボでは従来どおり動く)。
_GROOT_WORKER_PYTHON = Patch(
    path="inference/desktop/lower_policy/policies/groot.py",
    why=(
        "GR00T 53D worker の起動を pixi 依存から外すパッチ。"
        "上流 groot.py の _GrootWorkerClient.__init__ が変わった可能性がある。"
        "Dockerfile.thor.groot の RAMEN_WORKER_PYTHON_53D と対。"
    ),
    before="""        self._mode = cfg.mode
        self._overlay_jpeg_subsampling = cfg.overlay_jpeg_subsampling
        manifest = repo_root / "inference/desktop/pixi.toml"
        pixi = shutil.which("pixi") or "/home/ubuntu/.pixi/bin/pixi"
        if not Path(pixi).is_file() or not manifest.is_file():
            raise FileNotFoundError(
                f"GR00T worker runtime is unavailable: pixi={pixi} manifest={manifest}"
            )
        self._socket_path = Path(
            f"/tmp/iros_2026_ramen_groot_{os.getpid()}_{uuid.uuid4().hex}.sock"
        )
        command = [
            pixi,
            "run",
            "--manifest-path",
            str(manifest),
            "python",
            "-m",
""",
    after="""        self._mode = cfg.mode
        self._overlay_jpeg_subsampling = cfg.overlay_jpeg_subsampling
        self._socket_path = Path(
            f"/tmp/iros_2026_ramen_groot_{os.getpid()}_{uuid.uuid4().hex}.sock"
        )
        # VENDOR PATCH (Team RAMEN、boundary container 用): worker は
        # RAMEN_WORKER_PYTHON_53D (lerobot 0.6.1 の python) で直接起動する。
        # container に pixi は無いので、本家の pixi run 経路は fallback に回す。
        worker_python = os.environ.get("RAMEN_WORKER_PYTHON_53D") or os.environ.get(
            "RAMEN_WORKER_PYTHON"
        )
        if worker_python and not Path(worker_python).is_file():
            worker_python = shutil.which(worker_python) or worker_python
        if worker_python and Path(worker_python).is_file():
            launch_prefix = [str(worker_python)]
        else:
            manifest = repo_root / "inference/desktop/pixi.toml"
            pixi = shutil.which("pixi") or "/home/ubuntu/.pixi/bin/pixi"
            if not Path(pixi).is_file() or not manifest.is_file():
                raise FileNotFoundError(
                    f"GR00T worker runtime is unavailable: pixi={pixi} manifest={manifest}"
                )
            launch_prefix = [pixi, "run", "--manifest-path", str(manifest), "python"]
        command = [
            *launch_prefix,
            "-m",
""",
)


PATCHES: tuple[Patch, ...] = (_GROOT_WORKER_PYTHON,)


def apply_patch(vendor_root: Path, patch: Patch) -> str:
    """1 件を適用し、"applied" / "already-applied" を返す。"""
    target = vendor_root / patch.path
    if not target.is_file():
        raise PatchError(f"{patch.path} が vendor tree に無い。\n  {patch.why}")

    text = target.read_text(encoding="utf-8")
    if patch.after in text:
        return "already-applied"

    hits = text.count(patch.before)
    if hits != 1:
        raise PatchError(
            f"{patch.path}: アンカーが {hits} 箇所 (1 箇所であるべき)。\n"
            f"  {patch.why}\n"
            f"  上流を読んで tools/vendor_patches.py の before/after を更新すること。"
        )

    target.write_text(text.replace(patch.before, patch.after), encoding="utf-8")
    return "applied"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: vendor_patches.py <vendor/desktop の path>", file=sys.stderr)
        return 2

    vendor_root = Path(argv[1]).resolve()
    for patch in PATCHES:
        status = apply_patch(vendor_root, patch)
        print(f"[vendor-patch] {status:16s} {patch.path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except PatchError as exc:
        print(f"[vendor-patch] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
