"""完了済みの LeRobot checkpoint を HF Hub に定期同期する常駐プロセス (Issue #139)。

Sakura の `/nvme` は電源断で消えるので、学習途中の checkpoint を HF に逃がす。
`act_diffusion/run.sh` が学習と並行に背景起動し、学習終了時に `--final` で 1 回だけ呼ぶ。

repo 構成:
    checkpoints/<step>/pretrained_model/...   全 step (評価・選抜用)
    checkpoints/<step>/training_state/...     最新 step だけ (resume 用。古い step の分は削除)
    <root>/...                                `--final` で最新 step の pretrained_model を配置
                                              (from_pretrained でそのまま読める)

学習を邪魔しないための前提:
- 別プロセスで GPU は使わない。run.sh が `nice -n 19` + `ionice -c3` で起動する
- LeRobot は step dir を書き終えてから `checkpoints/last` を張り替える
  (lerobot_train.py の save_checkpoint → update_last_checkpoint)。`last` が指す step 以下の
  dir だけを完了済みとして扱い、書き込み中の dir には触れない
- 失敗しても学習には波及しない。state は commit 成功後にだけ進め、次の周回で再試行する
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from model.subtask_policy_training.scripts.upload_policy import (
    PRESERVED_REMOTE_FILES,
    validate_model_dir,
)

LAST_LINK = "last"
PRETRAINED_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
CHECKPOINTS_PREFIX = "checkpoints/"


@dataclass
class SyncState:
    uploaded_steps: list[str] = field(default_factory=list)
    training_state_step: str | None = None

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(sorted(payload.get("uploaded_steps", [])), payload.get("training_state_step"))

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=1), encoding="utf-8")
        tmp.replace(path)


@dataclass
class Plan:
    adds: list[tuple[Path, str]] = field(default_factory=list)   # (local file, path_in_repo)
    deletes: list[str] = field(default_factory=list)             # path_in_repo (folder は末尾 "/")
    state: SyncState = field(default_factory=SyncState)

    def is_empty(self) -> bool:
        return not self.adds and not self.deletes


def completed_steps(checkpoints_dir: Path) -> list[str]:
    """`last` が指す step 以下の step dir 名を昇順で返す。`last` が無ければ空。"""
    link = checkpoints_dir / LAST_LINK
    try:
        last = int(os.readlink(link).rstrip("/").split("/")[-1])
    except (OSError, ValueError):
        # 未作成 or update_last_checkpoint の unlink → symlink の隙間。次の周回で拾う
        return []
    steps = [p.name for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted((s for s in steps if int(s) <= last), key=int)


def _files_under(local_dir: Path, repo_prefix: str) -> list[tuple[Path, str]]:
    return [
        (path, f"{repo_prefix}{path.relative_to(local_dir).as_posix()}")
        for path in sorted(local_dir.rglob("*"))
        if path.is_file()
    ]


def plan_sync(checkpoints_dir: Path, state: SyncState) -> Plan:
    steps = completed_steps(checkpoints_dir)
    plan = Plan(state=SyncState(list(state.uploaded_steps), state.training_state_step))
    if not steps:
        return plan
    for step in steps:
        if step in plan.state.uploaded_steps:
            continue
        model_dir = checkpoints_dir / step / PRETRAINED_DIR
        validate_model_dir(model_dir)
        plan.adds += _files_under(model_dir, f"{CHECKPOINTS_PREFIX}{step}/{PRETRAINED_DIR}/")
        plan.state.uploaded_steps.append(step)
    latest = steps[-1]
    if state.training_state_step != latest:
        plan.adds += _files_under(
            checkpoints_dir / latest / TRAINING_STATE_DIR,
            f"{CHECKPOINTS_PREFIX}{latest}/{TRAINING_STATE_DIR}/",
        )
        if state.training_state_step is not None:
            plan.deletes.append(f"{CHECKPOINTS_PREFIX}{state.training_state_step}/{TRAINING_STATE_DIR}/")
        plan.state.training_state_step = latest
    plan.state.uploaded_steps.sort(key=int)
    return plan


def plan_final_root(checkpoints_dir: Path, remote_files: set[str]) -> Plan:
    """最新 step の pretrained_model を repo root に置き、root の古い model file を消す。

    `checkpoints/` 配下と README / .gitattributes には触れない。
    """
    steps = completed_steps(checkpoints_dir)
    plan = Plan()
    if not steps:
        return plan
    model_dir = checkpoints_dir / steps[-1] / PRETRAINED_DIR
    validate_model_dir(model_dir)
    plan.adds = _files_under(model_dir, "")
    new_root = {repo_path for _, repo_path in plan.adds}
    plan.deletes = sorted(
        path
        for path in remote_files
        if not path.startswith(CHECKPOINTS_PREFIX)
        and path not in PRESERVED_REMOTE_FILES
        and path not in new_root
    )
    return plan


def existing_deletes(deletes: list[str], remote_files: set[str]) -> list[str]:
    """repo に実在する path だけを残す (folder は末尾 "/" で、配下に file があれば実在)。

    Hub は存在しない path の delete を含む commit を丸ごと拒否する。commit が Hub に届いたのに
    応答で失敗した後の再試行では、前回消した training_state をもう一度消そうとするので、
    照合しないと以後の同期が毎回失敗し続ける。
    """
    return [
        path
        for path in deletes
        if (any(f.startswith(path) for f in remote_files) if path.endswith("/") else path in remote_files)
    ]


class Uploader:
    def __init__(self, repo_id: str, *, private: bool, dry_run: bool) -> None:
        self.repo_id = repo_id
        self.dry_run = dry_run
        self._private = private
        self._repo_ready = False
        self._api = None
        if not dry_run:
            from huggingface_hub import HfApi  # lazy: dry-run では network 依存を持ち込まない

            self._api = HfApi()

    def _ensure_repo(self) -> None:
        # 起動時の一時的な network 失敗で uploader ごと落ちないよう、最初に Hub を触る直前に作る
        # (watch loop の try の中で呼ばれるので、失敗しても次の周回で再試行される)
        if not self._repo_ready:
            self._api.create_repo(self.repo_id, repo_type="model", private=self._private, exist_ok=True)
            self._repo_ready = True

    def remote_files(self) -> set[str]:
        if self.dry_run:
            return set()
        self._ensure_repo()
        return set(self._api.list_repo_files(self.repo_id, repo_type="model"))

    def commit(self, plan: Plan, message: str) -> None:
        log(f"{'[dry-run] ' if self.dry_run else ''}{message}: +{len(plan.adds)} files, -{len(plan.deletes)} paths")
        for _, repo_path in plan.adds:
            log(f"  + {repo_path}")
        for repo_path in plan.deletes:
            log(f"  - {repo_path}")
        if self.dry_run:
            return
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete  # lazy: 同上

        # 新規 repo の最初の plan は delete が無く remote_files() を通らないので、ここで作る
        self._ensure_repo()
        deletes = existing_deletes(plan.deletes, self.remote_files()) if plan.deletes else []
        operations = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in plan.adds]
        operations += [CommitOperationDelete(path_in_repo=r, is_folder=r.endswith("/")) for r in deletes]
        if not operations:
            return
        self._api.create_commit(
            repo_id=self.repo_id, repo_type="model", operations=operations, commit_message=message
        )


def sync_once(uploader: Uploader, checkpoints_dir: Path, state_path: Path | None, state: SyncState) -> SyncState:
    plan = plan_sync(checkpoints_dir, state)
    if plan.is_empty():
        return state
    uploader.commit(plan, f"sync checkpoints up to {plan.state.uploaded_steps[-1]}")
    if state_path is not None:
        plan.state.save(state_path)
    return plan.state


def log(message: str) -> None:
    print(f"[ckpt_uploader {time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--interval", type=float, default=60.0, help="監視周期 [s]")
    parser.add_argument("--final", action="store_true", help="1 回同期して root に最新 model を置き終了")
    parser.add_argument("--dry-run", action="store_true", help="network を使わず予定だけ出力 (state も保存しない)")
    parser.add_argument("--public", action="store_true", help="repo を public で作る (既定 private)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # state は OUTPUT_DIR 直下 (checkpoints/ の外) に置き、LeRobot の checkpoint 管理と混ぜない
    state_path = None if args.dry_run else args.checkpoints_dir.parent / "hf_upload_state.json"
    state = SyncState.load(state_path) if state_path else SyncState()
    uploader = Uploader(args.repo_id, private=not args.public, dry_run=args.dry_run)

    if args.final:
        if not args.checkpoints_dir.is_dir():
            log(f"no checkpoints dir: {args.checkpoints_dir} (学習が ckpt 保存前に終了)")
            return 0
        status = 0
        try:
            sync_once(uploader, args.checkpoints_dir, state_path, state)
        except Exception as exc:  # noqa: BLE001 (途中 ckpt が失敗しても root の最新 model は置きにいく)
            log(f"final checkpoint sync failed: {type(exc).__name__}: {exc}")
            status = 1
        root_plan = plan_final_root(args.checkpoints_dir, uploader.remote_files())
        if not root_plan.is_empty():
            uploader.commit(root_plan, "publish latest checkpoint at repo root")
        return status

    stopping = False

    def request_stop(signum, frame) -> None:  # noqa: ARG001 (signal handler の固定 signature)
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(f"watching {args.checkpoints_dir} → {args.repo_id} every {args.interval:.0f}s")
    while not stopping:
        if args.checkpoints_dir.is_dir():
            try:
                state = sync_once(uploader, args.checkpoints_dir, state_path, state)
            except Exception as exc:  # noqa: BLE001 (学習を止めないため全例外を握って次周回で再試行)
                log(f"sync failed, retry next round: {type(exc).__name__}: {exc}")
        deadline = time.monotonic() + args.interval
        while not stopping and time.monotonic() < deadline:
            time.sleep(1.0)
    log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
