"""Merge multiple RAMEN-Ori training log files into a single wandb run (Issue #122)。

# 背景

Crash → resume で `wandb.resume_run_id` を使い既存 run に append すると、既存 run 側の
"current step" が resume 時 start_step より進んでいる場合 (crash 直前 buffer にあった
log の step 番号が upload 前に消えつつ server は max step だけ track する挙動)、
resume で log される step X (X < current step) は全て `Steps must be monotonically
increasing` として reject される。実測 (2026-08-19 Run 1 crash 時):

- crash-time run: wandb API 上に step 1000-95000 の 95 points (1000 step 間隔)
- crash 直前 wandb server 側 max step: 96001 (buffer にあり実 upload 前に crash)
- Resume の step 95001-96000 log 全て rejected → wandb plot 上で 5k step 分の空白

# このスクリプトの仕事

Local output.log は完全に残っているので、それを parse → 新 wandb run に monotonic
step で再 log すると、crash + resume の完全な curve が 1 run で見える。

# 使い方

    pixi run python -m model.ramen_ori.scripts.merge_wandb_runs \
      --source-log /path/to/crash-time/output.log \
      --resume-log /path/to/resume/output.log \
      --start-step 95000 \
      --entity ken05-matuo-llm-88_llm_2025_suzuki \
      --project ramen_ori \
      --merged-run-name default_100k_merged \
      --source-run-id gd6av7mu  # optional: config を継承する元 run

train.py の post-`[done]` hook から呼ぶ:

    from model.ramen_ori.scripts.merge_wandb_runs import merge_from_local_logs
    merge_from_local_logs(source_log=..., resume_log=..., start_step=..., ...)

# Merge rule

- source_log の step <= start_step: 全 record 採用
- resume_log の step > start_step: 全 record 採用
- Overlap (step == start_step): source 優先 (ckpt 保存直前の trajectory)
- resume_log の step <= start_step (startup artifact): 除外
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable


_STEP_RE = re.compile(
    r"^\[step\s+(?P<step>\d+)\]\s+loss=(?P<loss>[\d.e+-]+)\s+lr=(?P<lr>[\d.e+-]+)"
    r"(?:\s+it/s=(?P<its>[\d.e+-]+))?"
)
_VAL_RE = re.compile(
    r"^\[val\]\s+step=(?P<step>\d+)\s+(?P<key>val/[a-z_]+)=(?P<value>[\d.e+-]+)"
)


def parse_log(log_path: Path) -> dict[int, dict[str, float]]:
    """Parse output.log → {step: {metric_name: value}}。

    train_loss / train_lr / train_it_per_sec / val/loss_ema 等を抽出。
    同じ step に train + val 両方あれば merge。
    """
    records: dict[int, dict[str, float]] = {}
    with log_path.open() as f:
        for line in f:
            m = _STEP_RE.match(line)
            if m:
                step = int(m["step"])
                r = records.setdefault(step, {})
                r["train/loss"] = float(m["loss"])
                r["train/lr"] = float(m["lr"])
                if m["its"]:
                    r["train/it_per_sec"] = float(m["its"])
                continue
            m = _VAL_RE.match(line)
            if m:
                step = int(m["step"])
                key = m["key"]
                r = records.setdefault(step, {})
                r[key] = float(m["value"])
    return records


def merge_records(
    source: dict[int, dict[str, float]],
    resume: dict[int, dict[str, float]],
    start_step: int,
) -> dict[int, dict[str, float]]:
    """source (step <= start_step) + resume (step > start_step) を merge。

    Overlap は source 優先 (ckpt 保存前 trajectory を採用)。
    Resume の startup artifact (step <= start_step) は除外。
    """
    merged: dict[int, dict[str, float]] = {}
    for step, metrics in source.items():
        if step <= start_step:
            merged[step] = dict(metrics)
    for step, metrics in resume.items():
        if step > start_step:
            merged[step] = dict(metrics)
    return merged


def _fetch_source_config(entity: str, project: str, source_run_id: str) -> dict:
    """Source run の wandb config を取得 (metadata 継承用)。Fail 時は空 dict。"""
    try:
        import wandb  # lazy: wandb を必ずしも import 不要な test 経路のため

        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{source_run_id}")
        return dict(run.config)
    except Exception as e:
        print(f"[merge_wandb] WARN: source config fetch failed ({type(e).__name__}: {e})")
        return {}


def _fetch_wandb_history(entity: str, project: str, run_id: str) -> dict[int, dict[str, float]]:
    """既存 wandb run から history を取得 → {step: {metric: value}}。

    local log が消失しているケース (crash run の tee 先が上書き済) 用。
    scan_history で全 step (train / val 全 metric) を pull し、log-parse と同 shape に変換。
    """
    import wandb  # lazy

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    records: dict[int, dict[str, float]] = {}
    for row in run.scan_history():
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        r = records.setdefault(step, {})
        for k, v in row.items():
            if k.startswith("_") or v is None:
                continue
            if isinstance(v, (int, float)):
                r[k] = float(v)
    return records


def _fetch_wandb_history_with_timestamps(
    entity: str, project: str, run_id: str
) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    """history を pull し、metric records + step→timestamp map を返す。

    system metric injection 用 (system rows は timestamp 基準で train step にマップする)。
    """
    import wandb  # lazy

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    records: dict[int, dict[str, float]] = {}
    step_ts: dict[int, float] = {}
    for row in run.scan_history():
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        r = records.setdefault(step, {})
        ts = row.get("_timestamp")
        if isinstance(ts, (int, float)):
            step_ts[step] = float(ts)
        for k, v in row.items():
            if k.startswith("_") or v is None:
                continue
            if isinstance(v, (int, float)):
                r[k] = float(v)
    return records, step_ts


def _fetch_system_stream(entity: str, project: str, run_id: str) -> list[tuple[float, dict[str, float]]]:
    """system stream (GPU util / mem / CPU 等) を pull し、[(timestamp, {system.*: value})] を返す。"""
    import wandb  # lazy

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    rows: list[tuple[float, dict[str, float]]] = []
    # samples=None で全 row (large だが 200-1000 rows 程度なので許容)
    for row in run.history(stream="system", pandas=False, samples=100000):
        ts = row.get("_timestamp")
        if not isinstance(ts, (int, float)):
            continue
        metrics = {k: float(v) for k, v in row.items() if k.startswith("system.") and isinstance(v, (int, float))}
        if metrics:
            rows.append((float(ts), metrics))
    return rows


def _inject_system_metrics(
    records: dict[int, dict[str, float]],
    step_ts: dict[int, float],
    system_rows: list[tuple[float, dict[str, float]]],
    step_filter: Callable[[int], bool] = lambda s: True,
) -> None:
    """system rows を最寄り train step (timestamp 基準) に inject し records に in-place で足す。

    system_rows は timestamp 順にソート済想定。step_filter で採用範囲を絞る。
    """
    if not step_ts or not system_rows:
        return
    sorted_steps = sorted(s for s in step_ts if step_filter(s))
    if not sorted_steps:
        return
    ts_to_step = [(step_ts[s], s) for s in sorted_steps]
    ts_to_step.sort()
    # 各 system row を最寄り step にマップ (二分探索は簡潔さのため線形で十分な row 数)
    import bisect
    ts_arr = [t for t, _ in ts_to_step]
    step_arr = [s for _, s in ts_to_step]
    for sys_ts, sys_metrics in system_rows:
        idx = bisect.bisect_left(ts_arr, sys_ts)
        # left / right の近い方
        candidates = []
        if idx > 0:
            candidates.append((abs(ts_arr[idx - 1] - sys_ts), step_arr[idx - 1]))
        if idx < len(ts_arr):
            candidates.append((abs(ts_arr[idx] - sys_ts), step_arr[idx]))
        if not candidates:
            continue
        _, nearest_step = min(candidates)
        records.setdefault(nearest_step, {}).update(sys_metrics)


def _download_wandb_output_log(entity: str, project: str, run_id: str, dest_path: Path) -> bool:
    """指定 run の output.log を wandb から DL。Fail 時 False。"""
    import wandb  # lazy

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        for f in run.files():
            if f.name == "output.log":
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                f.download(root=str(dest_path.parent), replace=True)
                # wandb は root/name に置くので rename
                downloaded = dest_path.parent / f.name
                if downloaded != dest_path and downloaded.exists():
                    downloaded.rename(dest_path)
                return dest_path.exists()
        return False
    except Exception as e:
        print(f"[merge_wandb] WARN: output.log download failed ({run_id}): {type(e).__name__}: {e}")
        return False


def _emit_log_content(log_path: Path, label: str) -> None:
    """Log file の内容全体を stdout に emit (wandb Logs tab に反映させるため)。"""
    print(f"\n{'=' * 78}\n=== {label}: {log_path} ({log_path.stat().st_size} bytes) ===\n{'=' * 78}")
    print(log_path.read_text())
    print(f"{'=' * 78}\n=== END {label} ===\n{'=' * 78}\n")


def _attach_log_files(run, source_log: Path, resume_log: Path) -> None:
    """source + resume + concat した merged.log を wandb run.dir に配置 → 自動 sync で Files tab に。"""
    import shutil

    run_dir = Path(run.dir)
    dest_source = run_dir / "source_train.log"
    dest_resume = run_dir / "resume_train.log"
    dest_merged = run_dir / "merged_train.log"
    shutil.copy(source_log, dest_source)
    shutil.copy(resume_log, dest_resume)
    with dest_merged.open("w") as f:
        f.write(f"### SOURCE: {source_log}\n")
        f.write(dest_source.read_text())
        f.write(f"\n### RESUME: {resume_log}\n")
        f.write(dest_resume.read_text())
    print(f"[merge_wandb] attached: source_train.log / resume_train.log / merged_train.log")


def merge_from_local_logs(
    source_log: Path,
    resume_log: Path,
    start_step: int,
    entity: str,
    project: str,
    merged_run_name: str,
    source_run_id: str | None = None,
    tags: list[str] | None = None,
    emit_log_content: bool = True,
) -> str:
    """2 個の output.log を parse → 新 wandb run に monotonic step で再 log。

    Returns: 新 run の URL。

    emit_log_content=True (default): source + resume の log 内容を stdout に emit し、
    wandb Logs tab で閲覧可能にする。加えて log file 本体を Files tab に attach する。
    """
    import wandb  # lazy

    src_records = parse_log(source_log)
    res_records = parse_log(resume_log)
    merged = merge_records(src_records, res_records, start_step=start_step)
    if not merged:
        raise RuntimeError(
            f"no records to merge (source={len(src_records)}, resume={len(res_records)}, "
            f"start_step={start_step})"
        )

    config: dict = {}
    if source_run_id:
        config.update(_fetch_source_config(entity, project, source_run_id))
    config["merged_from"] = {
        "source_log": str(source_log),
        "resume_log": str(resume_log),
        "start_step": start_step,
        "source_run_id": source_run_id,
        "source_records": len(src_records),
        "resume_records": len(res_records),
        "merged_records": len(merged),
    }

    run = wandb.init(
        project=project,
        entity=entity,
        name=merged_run_name,
        config=config,
        tags=(tags or []) + ["merged"],
        job_type="merge",
        reinit="finish_previous",  # PR #127 review HIGH: reinit=True は wandb 0.24 で deprecated
    )
    print(
        f"[merge_wandb] created new run: {run.url} "
        f"({len(merged)} steps, source={len(src_records)}, resume={len(res_records)})"
    )

    # Source + resume log の raw 内容を Logs tab に流し込む (debug 用に元 log が
    # merged run 側で参照できるように)。ログサイズが数 MB 級なら emit_log_content=False で off 可能。
    if emit_log_content:
        _emit_log_content(source_log, "SOURCE LOG")
        _emit_log_content(resume_log, "RESUME LOG")

    # Log file 本体を Files tab に attach (run.dir に copy → wandb 自動 sync)。
    _attach_log_files(run, source_log, resume_log)

    # Sorted step 順に log (monotonic guarantee)。commit=True で 1 step ごと確定。
    for step in sorted(merged):
        run.log(merged[step], step=step, commit=True)

    run.finish()
    print(f"[merge_wandb] finished merged run: {run.url}")
    return str(run.url)


def merge_from_wandb_runs(
    source_run_id: str,
    resume_run_id: str,
    start_step: int,
    entity: str,
    project: str,
    merged_run_name: str,
    config_source_run_id: str | None = None,
    tags: list[str] | None = None,
    include_system_metrics: bool = True,
    attach_output_logs: bool = True,
    log_download_dir: Path | None = None,
) -> str:
    """既存 2 wandb run から history を pull → 新 wandb run に monotonic step で merge。

    log 消失で `merge_from_local_logs` が使えない case (crash run の tee 先が resume で
    上書きされた等) 用の代替。config は config_source_run_id (default: source_run_id) から継承。

    include_system_metrics=True で system.* stream (GPU util / mem / CPU) も timestamp 基準で
    最寄り train step にマップして inject。attach_output_logs=True で 2 run の output.log を
    wandb API から DL → 新 run の Files tab に attach + Logs tab に emit。
    """
    import wandb  # lazy

    src_records, src_ts = _fetch_wandb_history_with_timestamps(entity, project, source_run_id)
    res_records, res_ts = _fetch_wandb_history_with_timestamps(entity, project, resume_run_id)

    if include_system_metrics:
        src_sys = _fetch_system_stream(entity, project, source_run_id)
        res_sys = _fetch_system_stream(entity, project, resume_run_id)
        _inject_system_metrics(src_records, src_ts, src_sys, step_filter=lambda s: s <= start_step)
        _inject_system_metrics(res_records, res_ts, res_sys, step_filter=lambda s: s > start_step)
        print(f"[merge_wandb] injected system rows: source={len(src_sys)}, resume={len(res_sys)}")

    merged = merge_records(src_records, res_records, start_step=start_step)
    if not merged:
        raise RuntimeError(
            f"no records to merge (source={len(src_records)}, resume={len(res_records)}, "
            f"start_step={start_step})"
        )

    config = _fetch_source_config(entity, project, config_source_run_id or source_run_id)
    config["merged_from"] = {
        "source_run_id": source_run_id,
        "resume_run_id": resume_run_id,
        "start_step": start_step,
        "source_records": len(src_records),
        "resume_records": len(res_records),
        "merged_records": len(merged),
        "mode": "wandb_history",
        "system_metrics_injected": include_system_metrics,
    }

    run = wandb.init(
        project=project,
        entity=entity,
        name=merged_run_name,
        config=config,
        tags=(tags or []) + ["merged"],
        job_type="merge",
        reinit="finish_previous",  # PR #127 review HIGH: reinit=True は wandb 0.24 で deprecated
    )
    print(
        f"[merge_wandb] created new run: {run.url} "
        f"({len(merged)} steps, source={len(src_records)}, resume={len(res_records)})"
    )

    # output.log を wandb API から DL → attach + Logs tab に emit
    if attach_output_logs:
        dl_dir = log_download_dir or Path(run.dir) / "_source_logs"
        dl_dir.mkdir(parents=True, exist_ok=True)
        src_log = dl_dir / f"source_{source_run_id}_output.log"
        res_log = dl_dir / f"resume_{resume_run_id}_output.log"
        ok_src = _download_wandb_output_log(entity, project, source_run_id, src_log)
        ok_res = _download_wandb_output_log(entity, project, resume_run_id, res_log)
        if ok_src:
            _emit_log_content(src_log, f"SOURCE LOG ({source_run_id})")
        if ok_res:
            _emit_log_content(res_log, f"RESUME LOG ({resume_run_id})")
        # attach: copy to run.dir で wandb 自動 sync
        import shutil

        if ok_src:
            shutil.copy(src_log, Path(run.dir) / "source_train.log")
        if ok_res:
            shutil.copy(res_log, Path(run.dir) / "resume_train.log")
        if ok_src and ok_res:
            merged_log = Path(run.dir) / "merged_train.log"
            with merged_log.open("w") as f:
                f.write(f"### SOURCE ({source_run_id}): {src_log.name}\n")
                f.write(src_log.read_text())
                f.write(f"\n### RESUME ({resume_run_id}): {res_log.name}\n")
                f.write(res_log.read_text())
        print(f"[merge_wandb] attached output logs: source={ok_src}, resume={ok_res}")

    for step in sorted(merged):
        run.log(merged[step], step=step, commit=True)

    run.finish()
    print(f"[merge_wandb] finished merged run: {run.url}")
    return str(run.url)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-log", type=Path, required=True, help="crash-time output.log path")
    p.add_argument("--resume-log", type=Path, required=True, help="resume-time output.log path")
    p.add_argument("--start-step", type=int, required=True, help="resume で load した ckpt の step (source: <= / resume: >)")
    p.add_argument("--entity", type=str, required=True)
    p.add_argument("--project", type=str, required=True)
    p.add_argument("--merged-run-name", type=str, required=True)
    p.add_argument("--source-run-id", type=str, default=None, help="config metadata を継承する元 run id (optional)")
    p.add_argument("--tag", action="append", default=[], help="wandb tag (複数指定可)")
    p.add_argument(
        "--no-emit-log-content",
        dest="emit_log_content",
        action="store_false",
        help="stdout への log content emit を off (Logs tab に流し込まない、大きい log 用)",
    )
    args = p.parse_args()

    if not args.source_log.exists():
        raise FileNotFoundError(f"--source-log not found: {args.source_log}")
    if not args.resume_log.exists():
        raise FileNotFoundError(f"--resume-log not found: {args.resume_log}")

    url = merge_from_local_logs(
        source_log=args.source_log,
        resume_log=args.resume_log,
        start_step=args.start_step,
        entity=args.entity,
        project=args.project,
        merged_run_name=args.merged_run_name,
        source_run_id=args.source_run_id,
        tags=args.tag,
        emit_log_content=args.emit_log_content,
    )
    print(f"MERGED_RUN_URL={url}")


if __name__ == "__main__":
    _cli()
