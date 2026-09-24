#!/usr/bin/env python3
"""会場の run が読む重みを HF cache に取る / 取れているかを確かめる (一覧と USB の手順は WEIGHTS.md)。

会場は実行時にネットに出ない (HF_HUB_OFFLINE=1)。重みが 1 つでも cache に無いと、その場で止まる。

# 実行時と同じ呼び方で cache に入れる

私たちの model は、ramen/ (本体のコピー) の本物の解決関数をネットのある状態で呼ぶ:
  RAMEN-Ori  resolve_hf_ckpt                   GR00T 53D  _resolve_groot_checkpoint_root (+ base model)
  pick       _PickLegsWorkerClient._resolve_checkpoint                YOLO  resolve_yolo_ckpt_ref
どの model を使うかの正本は policy_config.yaml (default_variant_by_skill と yolo) と hybrid の YAML
(vlm.model) なので、model を差し替えてもこの一覧は自動で追従する。外部の package の中で読まれる物
(lingbot・Cosmos・VLM) は、実行時と同じく main を丸ごと取る (ネット無しで main を引くには refs/main も要る)。

# 使い方 (image の中で。runtime と同じ huggingface_hub の版・cache の形になる)

  取る (ネットのある所。token は .env から: set -a; . ./.env; set +a):
    docker run --rm -e HF_HUB_OFFLINE=0 -e HF_TOKEN -v <hf_cache>:/root/.cache/huggingface <image> \\
      pixi run --as-is -e runtime python /app/tools/prefetch_weights.py
  確かめる (ネット無し):
    docker run --rm -v <hf_cache>:/root/.cache/huggingface:ro <image> \\
      pixi run --as-is -e runtime python /app/tools/prefetch_weights.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RAMEN_ROOT = Path(__file__).resolve().parents[1] / "ramen"
POLICY_CONFIG = RAMEN_ROOT / "inference/desktop/lower_policy/configs/policy_config.yaml"
HYBRID_CONFIG = (
    RAMEN_ROOT / "inference/desktop/pick_leg_hybrid/configs/pick_leg_hybrid.yaml"
)

#: GR00T の base model。revision は各 ckpt の config.json の base_model_revision (無ければこの既定。
#: groot.py の Gr00tPolicy と同じ値。test で突き合わせる)。
GROOT_BASE_REPO = "nvidia/GR00T-N1.7-3B"
GROOT_BASE_DEFAULT_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
#: 外部の package が main で読むもの
LINGBOT_REPO = "robbyant/lingbot-vision-vit-base"  # lingbot_vision.load_pretrained_backbone("base")
COSMOS_REPO = (
    "nvidia/Cosmos-Reason2-2B"  # lerobot の GR00T N1.7 の tokenizer・前処理 (gated)
)


@dataclass
class Item:
    label: str
    repo_id: str
    revision: str  # commit hash か "main"
    scope: str
    fetch: Callable[[], Path]  # 実行時と同じ呼び方。ネット無しなら cache から返すか例外
    needs_groot_base: bool = (
        False  # GR00T 53D の ckpt (config.json の base_model_revision で base を取る)
    )


def _split_ref(ref: str) -> tuple[str, str | None]:
    repo_id, sep, revision = ref.rpartition("@")
    return (repo_id, revision) if sep else (ref, None)


def _snapshot_main(repo_id: str) -> Callable[[], Path]:
    def fetch() -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id))  # revision 無し = main (実行時と同じ)

    return fetch


def _groot_base_item(revision: str) -> Item:
    from inference.desktop.lower_policy.policies.groot import (
        _resolve_groot_checkpoint_root,
    )

    return Item(
        "GR00T の base model",
        GROOT_BASE_REPO,
        revision,
        "全部",
        lambda: _resolve_groot_checkpoint_root(f"{GROOT_BASE_REPO}@{revision}"),
    )


def build_items() -> list[Item]:
    """今の設定から、会場の Stage 0〜5 が読む重みの一覧を作る (ネットには出ない)。

    GR00T の base model は ckpt の config.json で revision が決まるので、ここには入れず
    main() が ckpt を解決した後に足す。
    """
    if str(RAMEN_ROOT) not in sys.path:
        sys.path.insert(0, str(RAMEN_ROOT))
    import yaml

    from inference.desktop.lower_policy.policies.config_loader import (
        load_default_variant_by_skill,
        load_policy_variant,
        load_yolo_settings,
    )

    items: list[Item] = []
    types: set[str] = set()
    for skill, variant in load_default_variant_by_skill(POLICY_CONFIG).items():
        entry = load_policy_variant(POLICY_CONFIG, variant)
        cfg = entry.policy_config
        repo_id, revision = _split_ref(str(cfg.ckpt_ref))
        types.add(entry.policy_type)
        label = f"{skill} ({variant})"
        if entry.policy_type == "ramen_ori":
            from inference.desktop.lower_policy.policies.ramen_ori import (
                resolve_hf_ckpt,
            )

            def fetch(r=repo_id, v=revision, f=cfg.ckpt_filename) -> Path:
                return Path(resolve_hf_ckpt(r, v, f))

            scope = cfg.ckpt_filename or "ckpt_step_*.pt の最新"
            items.append(Item(label, repo_id, revision or "main", scope, fetch))
        elif entry.policy_type == "groot":
            from inference.desktop.lower_policy.policies.groot import (
                _resolve_groot_checkpoint_root,
            )

            def fetch(ref=str(cfg.ckpt_ref), sub=cfg.checkpoint_subdir) -> Path:
                return _resolve_groot_checkpoint_root(ref, sub)

            scope = (
                f"{cfg.checkpoint_subdir}/ だけ" if cfg.checkpoint_subdir else "全部"
            )
            items.append(
                Item(
                    label,
                    repo_id,
                    revision or "main",
                    scope,
                    fetch,
                    needs_groot_base=True,
                )
            )
        elif entry.policy_type == "groot_pick_legs":
            from inference.desktop.lower_policy.policies.groot_pick_legs import (
                RUNTIME_CHECKPOINT_FILES,
                _PickLegsWorkerClient,
            )

            def fetch(c=cfg) -> Path:
                return _PickLegsWorkerClient._resolve_checkpoint(RAMEN_ROOT, c)

            scope = f"{cfg.checkpoint_subdir}/ の実行時の file ({len(RUNTIME_CHECKPOINT_FILES)} 種)"
            items.append(Item(label, repo_id, revision or "main", scope, fetch))
        else:
            raise ValueError(
                f"{skill}: 事前取得の方法を知らない policy_type {entry.policy_type!r}"
            )

    yolo = load_yolo_settings(POLICY_CONFIG)
    from inference.desktop.perception.yolo_obb import resolve_yolo_ckpt_ref

    yolo_repo, yolo_revision = _split_ref(yolo.ckpt_ref)
    items.append(
        Item(
            "YOLO (overlay)",
            yolo_repo,
            yolo_revision or "main",
            yolo.ckpt_file,
            lambda: resolve_yolo_ckpt_ref(yolo.ckpt_ref, yolo.ckpt_file),
        )
    )
    if "ramen_ori" in types:
        items.append(
            Item(
                "RAMEN-Ori の画像 backbone",
                LINGBOT_REPO,
                "main",
                "全部",
                _snapshot_main(LINGBOT_REPO),
            )
        )
    if types & {"groot", "groot_pick_legs"}:
        items.append(
            Item(
                "GR00T の backbone (tokenizer・前処理)",
                COSMOS_REPO,
                "main",
                "全部 (gated)",
                _snapshot_main(COSMOS_REPO),
            )
        )
    if "groot_pick_legs" in types:  # hybrid pick の区間 1→2 の判定
        vlm_repo = yaml.safe_load(HYBRID_CONFIG.read_text(encoding="utf-8"))["vlm"][
            "model"
        ]
        items.append(
            Item(
                "VLM (hybrid pick、Stage 1〜4)",
                vlm_repo,
                "main",
                "全部",
                _snapshot_main(vlm_repo),
            )
        )
    return items


def _base_revision(checkpoint_root: Path) -> str:
    config = json.loads((checkpoint_root / "config.json").read_text(encoding="utf-8"))
    return str(config.get("base_model_revision") or GROOT_BASE_DEFAULT_REVISION)


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size  # cache の symlink の先 (blob) の大きさ
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _check_access(repo_ids: list[str]) -> list[str]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    problems = []
    for repo_id in repo_ids:
        try:
            HfApi().auth_check(repo_id)
        except GatedRepoError:
            problems.append(
                f"{repo_id}: gated。この token の account で HF の license を承認する"
            )
        except RepositoryNotFoundError:
            problems.append(
                f"{repo_id}: この token では見えない (private なら読める account の token を使う)"
            )
    return problems


def _fetch(item: Item) -> Path | None:
    try:
        path = item.fetch()
    except Exception as exc:  # noqa: BLE001 - 取れない / cache に無い理由を名指しで出す
        print(
            f"  MISSING  {item.label}: {item.repo_id}@{item.revision} [{item.scope}] ({type(exc).__name__}: {exc})"
        )
        return None
    print(
        f"  OK       {item.label}: {item.repo_id}@{item.revision} [{item.scope}] "
        f"{_size_bytes(path) / 1e9:.2f} GB"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="ネット無しで、cache に全部あるかだけ見る"
    )
    args = parser.parse_args()

    if args.check:
        os.environ["HF_HUB_OFFLINE"] = (
            "1"  # huggingface_hub の import より前に。確認では絶対に取りに行かない
        )
    items = build_items()
    from huggingface_hub import constants

    if not args.check:
        if constants.HF_HUB_OFFLINE:
            print(
                "error: 取得には -e HF_HUB_OFFLINE=0 が要る (image の既定はオフライン)",
                file=sys.stderr,
            )
            return 2
        problems = _check_access(
            sorted({item.repo_id for item in items} | {GROOT_BASE_REPO})
        )
        if problems:
            print(
                "error: token で読めない repo がある (何も取らずに止める):",
                file=sys.stderr,
            )
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2

    print(
        f"[weights] cache: {constants.HF_HUB_CACHE}  mode: {'check (offline)' if args.check else 'download'}"
    )
    missing = 0
    base_revisions: set[str] = set()
    groot_ckpts = 0
    for item in items:
        path = _fetch(item)
        if path is None:
            missing += 1
        elif item.needs_groot_base:
            try:
                base_revisions.add(_base_revision(path))
            except (OSError, ValueError) as exc:  # 取れたはずの ckpt の config が読めない = 壊れた cache
                missing += 1
                print(f"  MISSING  {item.label}: {path}/config.json が読めない ({type(exc).__name__})")
        groot_ckpts += item.needs_groot_base
    if groot_ckpts and not base_revisions:
        # ckpt が 1 つも無く config を読めない。既定の revision で base だけは確かめる
        base_revisions.add(GROOT_BASE_DEFAULT_REVISION)
    for revision in sorted(base_revisions):
        if _fetch(_groot_base_item(revision)) is None:
            missing += 1
    print(f"[weights] {'all present' if not missing else f'{missing} missing'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
