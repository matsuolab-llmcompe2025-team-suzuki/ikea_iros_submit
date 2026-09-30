"""run の記録から、腕が指令にどれだけついてきたかを skill ごとに出す (Issue #188)。

会場で run の後に Thor 上で数秒で読み、次の run の送り方 (``--arm-tuning`` など) を決めるための道具。
numpy と標準ライブラリだけで動く (image の runtime env でも、手元の Mac でも)。

    python inference/desktop/lower_policy/scripts/tracking_summary.py outputs            # 最新の run
    python inference/desktop/lower_policy/scripts/tracking_summary.py outputs/production_runs/<run>

読むもの: ``production_runs/<run>/actions.jsonl`` (policy の指令 ``chunk_0``) と ``states.jsonl``
(実測 ``joint_position``)、あれば ``orch_logs/orch_<run>.jsonl`` の ``boundary_joint`` (送った回数・
先読み・押している判定)。腕だけ policy の予定の少し先を送った run (``policy_metadata`` に
``plan_target_arms`` がある) は、送った値ではなく予定の今の値を指令とみなす (送った値と比べると、
わざと先に出した分まで遅れに見える)。

出すもの (skill ごと):
- 遅れ: 実測を何 ms ずらすと指令に一番合うか (10 ms 刻み、0〜600 ms)
- ずれ: 遅れをそろえた後の |指令 − 実測| の平均と、指令の速さ別
- 押し負け: 遅れをそろえた後も残る、関節ごとの符号つきの差 (指令 − 実測)。0.03 rad 以上だけ
参考 (09-29 の Issue #188): 自前の実機 (arm_sdk) は遅れ 80〜110 ms・ずれ 0.010〜0.013 rad、
会場 (今日の送り方) は pick 390 ms、台を回す 270 ms・右肘の押し負け −0.15 rad。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ARM_NAMES = (
    "L.sp", "L.sr", "L.sy", "L.el", "L.wr", "L.wp", "L.wy",
    "R.sp", "R.sr", "R.sy", "R.el", "R.wr", "R.wp", "R.wy",
)  # fmt: skip
_ARMS_IN_ACTION = slice(3, 17)  # 19-D action の腕 14-D
_ARMS_IN_BODY = slice(15, 29)  # 29-D 実測の腕 14-D
_GRID_S = 0.01
_MAX_LAG_S = 0.6
_SEGMENT_GAP_S = 0.5
_MIN_ROWS = 60
_DEFICIT_REPORT_RAD = 0.03
_SPEED_BINS = ((0.0, 0.3), (0.3, 0.8), (0.8, float("inf")))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 書きかけの最後の行など
    return rows


def resolve_run(path: Path) -> Path:
    """run の directory。``outputs`` や ``production_runs`` を渡したら最新の run。"""
    if (path / "actions.jsonl").exists():
        return path
    runs_root = (
        path / "production_runs" if (path / "production_runs").is_dir() else path
    )
    runs = sorted(p for p in runs_root.iterdir() if (p / "actions.jsonl").exists())
    if not runs:
        raise FileNotFoundError(f"no production run with actions.jsonl under {path}")
    return runs[-1]


def load_run(run: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """skill ごとの区間 [(時刻 s, 指令 (n,14), 実測 (n,14))]。途切れ (0.5 s 超) で分ける。"""
    actions = _read_jsonl(run / "actions.jsonl")
    states = {row.get("t"): row for row in _read_jsonl(run / "states.jsonl")}
    per_skill: dict[str, list[tuple[float, list, list]]] = {}
    for action in actions:
        state = states.get(action.get("t"))
        chunk = action.get("chunk_0")
        joints = None if state is None else state.get("joint_position")
        if not chunk or not joints or len(chunk) < 17 or len(joints) < 29:
            continue
        skill = str(action.get("skill"))
        per_skill.setdefault(skill, []).append(
            (float(action["t"]) / 1e9, _intent_arms(action, chunk), joints[_ARMS_IN_BODY])
        )
    out: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for skill, rows in per_skill.items():
        t = np.array([r[0] for r in rows])
        intent = np.array([r[1] for r in rows], dtype=np.float64)
        measured = np.array([r[2] for r in rows], dtype=np.float64)
        ok = np.isfinite(t) & np.isfinite(intent).all(1) & np.isfinite(measured).all(1)
        t, intent, measured = t[ok], intent[ok], measured[ok]
        order = np.argsort(t)
        t, intent, measured = t[order], intent[order], measured[order]
        cuts = np.concatenate(
            [[0], np.flatnonzero(np.diff(t) > _SEGMENT_GAP_S) + 1, [len(t)]]
        )
        segments = [
            (t[a:b] - t[a], intent[a:b], measured[a:b])
            for a, b in zip(cuts[:-1], cuts[1:])
            if b - a >= _MIN_ROWS
        ]
        if segments:
            out[skill] = segments
    return out


def _intent_arms(action: dict, chunk: list) -> list:
    """比べる指令の腕 14-D。先の予定を送った tick は予定の今の値 (Issue #188)。"""
    plan = (action.get("policy_metadata") or {}).get("plan_target_arms")
    if isinstance(plan, list) and len(plan) == 14:
        return plan
    return chunk[_ARMS_IN_ACTION]


def preview_stats(run: Path) -> dict[str, dict]:
    """skill ごとに、腕だけ送った先の予定の大きさ (Issue #188)。送っていない skill は出さない。"""
    per_skill: dict[str, list[tuple[float, int, int]]] = {}
    for action in _read_jsonl(run / "actions.jsonl"):
        meta = action.get("policy_metadata") or {}
        if "arm_preview_steps" not in meta or action.get("t") is None:
            continue
        per_skill.setdefault(str(action.get("skill")), []).append(
            (
                float(action["t"]) / 1e9,
                int(meta["arm_preview_steps"]),
                int(meta.get("arm_preview_used_steps", 0)),
            )
        )
    out: dict[str, dict] = {}
    for skill, rows in per_skill.items():
        t = np.sort(np.array([r[0] for r in rows]))
        used = np.array([r[2] for r in rows], dtype=np.float64)
        dt = np.diff(t)
        dt = dt[(dt > 0.0) & (dt < _SEGMENT_GAP_S)]
        tick_s = float(np.median(dt)) if dt.size else 1.0 / 30.0
        out[skill] = {
            "requested_ticks": max(r[1] for r in rows),
            "used_ticks_mean": float(used.mean()),
            "used_ms_mean": float(used.mean() * tick_s * 1000.0),
            "unused_fraction": float(np.mean(used == 0)),
        }
    return out


def _resample(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    grid = np.arange(0.0, t[-1], _GRID_S)
    return np.stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])], axis=1)


def tracking_metrics(
    segments: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Optional[dict]:
    """区間をまとめた遅れ・ずれ・押し負け。動きが無ければ None。"""
    grids = [(_resample(t, i), _resample(t, m)) for t, i, m in segments if t[-1] > 1.0]
    if not grids:
        return None
    shifts = range(0, int(round(_MAX_LAG_S / _GRID_S)) + 1)
    totals = []
    for s in shifts:
        err, n = 0.0, 0
        for intent, measured in grids:
            if len(intent) <= s:
                continue
            d = np.abs(intent[: len(intent) - s] - measured[s:])
            err += float(d.sum())
            n += d.size
        totals.append(err / n if n else np.inf)
    best = int(np.argmin(totals))
    residual, speed = [], []
    for intent, measured in grids:
        if len(intent) <= best + 1:
            continue
        residual.append(intent[: len(intent) - best] - measured[best:])
        speed.append(np.abs(np.gradient(intent, _GRID_S, axis=0))[: len(intent) - best])
    residual = np.concatenate(residual)
    speed = np.concatenate(speed)
    by_speed = {}
    for lo, hi in _SPEED_BINS:
        mask = (speed >= lo) & (speed < hi)
        if mask.sum() >= 50:
            by_speed[f"{lo:g}-{hi:g}"] = float(np.abs(residual[mask]).mean())
    deficit = residual.mean(axis=0)
    return {
        "lag_ms": best * _GRID_S * 1000.0,
        "mean_abs_error_rad": float(np.abs(residual).mean()),
        "error_by_speed_rad": by_speed,
        "deficit_rad": {n: float(v) for n, v in zip(ARM_NAMES, deficit)},
        "seconds": float(sum(len(i) for i, _ in grids) * _GRID_S),
        "segments": len(grids),
    }


def publish_stats(orch_log: Path) -> Optional[dict]:
    """orch log (run と同じ名前、1 run に 1 本) の boundary_joint から、送った回数・
    押している割合・先読みの大きさ。"""
    if not orch_log.exists():
        return None
    rows = [r for r in _read_jsonl(orch_log) if r.get("event") == "boundary_joint"]
    if not rows:
        return None
    reasons: dict[str, int] = {}
    for r in rows:
        reason = str(r.get("publish_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    blocked = [bool(r["arm_blocked"]) for r in rows if "arm_blocked" in r]
    lead = [np.abs(r["arm_lead_rad"]).max() for r in rows if r.get("arm_lead_rad")]
    return {
        "published": len(rows),
        "reasons": reasons,
        "preset": next(
            (r["arm_tracking_preset"] for r in rows if "arm_tracking_preset" in r), None
        ),
        "blocked_fraction": float(np.mean(blocked)) if blocked else None,
        "max_lead_rad": float(np.max(lead)) if lead else None,
    }


def summarize(run: Path, orch_logs: Optional[Path] = None) -> dict:
    run = resolve_run(run)
    report: dict = {"run": run.name, "skills": {}}
    previews = preview_stats(run)
    for skill, segments in load_run(run).items():
        metrics = tracking_metrics(segments)
        if metrics is not None:
            if skill in previews:
                metrics["preview"] = previews[skill]
            report["skills"][skill] = metrics
    logs = orch_logs if orch_logs is not None else run.parent.parent / "orch_logs"
    publish = publish_stats(logs / f"orch_{run.name}.jsonl")
    if publish is not None:
        report["publish"] = publish
    return report


def format_report(report: dict) -> str:
    lines = [
        f"== {report['run']}  (参考: 自前の実機は遅れ 80〜110 ms・ずれ 0.010〜0.013 rad)"
    ]
    publish = report.get("publish")
    if publish:
        blocked = publish.get("blocked_fraction")
        lead = publish.get("max_lead_rad")
        lines.append(
            f"   送り方: preset={publish.get('preset') or '(記録なし = 09-29 と同じ)'}  "
            f"送った回数 {publish['published']} {publish['reasons']}"
            + (f"  押している割合 {blocked * 100:.0f}%" if blocked is not None else "")
            + (f"  先読みの最大 {lead:.3f} rad" if lead is not None else "")
        )
    if not report["skills"]:
        lines.append("   (腕が 1 s 以上動いた skill が無い)")
    for skill, m in report["skills"].items():
        speed = "  ".join(
            f"{k} rad/s: {v:.3f}" for k, v in m["error_by_speed_rad"].items()
        )
        lines.append(
            f"-- {skill}: {m['seconds']:.0f} s / {m['segments']} 区間  遅れ {m['lag_ms']:.0f} ms  "
            f"ずれ {m['mean_abs_error_rad']:.3f} rad  ({speed})"
        )
        preview = m.get("preview")
        if preview:
            lines.append(
                f"   先の予定: 腕だけ平均 {preview['used_ms_mean']:.0f} ms 先 "
                f"({preview['used_ticks_mean']:.1f} / {preview['requested_ticks']} tick、"
                f"届かず今の値のまま {preview['unused_fraction'] * 100:.0f}%)。"
                "遅れ・ずれ・押し負けは予定の今の値との比較"
            )
        deficit = [
            f"{n}={v:+.3f}"
            for n, v in m["deficit_rad"].items()
            if abs(v) >= _DEFICIT_REPORT_RAD
        ]
        lines.append(
            "   押し負け (指令−実測、遅れをそろえた後): "
            + (" ".join(deficit) if deficit else f"{_DEFICIT_REPORT_RAD} rad 以上なし")
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path",
        type=Path,
        help="run の directory、または outputs / production_runs (最新の run)",
    )
    parser.add_argument(
        "--orch-logs",
        type=Path,
        default=None,
        help="orch_logs の場所 (既定は run の隣)",
    )
    parser.add_argument("--json", action="store_true", help="JSON で出す")
    args = parser.parse_args(argv)
    report = summarize(args.path, args.orch_logs)
    print(
        json.dumps(report, ensure_ascii=False, indent=1)
        if args.json
        else format_report(report)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
