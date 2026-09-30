"""会場の腕の追加学習 (Issue #188) の前に、実データで 3 つを確かめる。学習の env (Sakura) で、データを戻した後に回す。

1. 手先の入力の FK: FK は推論と同じ 29 関節 (robot_q_current[7:36]) と、以前の学習が使っていた [:29]
   (胴体 7 + 関節 0〜21) の両方。記録された ee_state は FK と定義 (座標・offset) が違い、正しい FK でも 10 cm 前後
   離れる (rotate の実データ) ので、距離では切り方を判定できない。記録の手先の**動きの向き** (0.1 s ごとの変化、
   定義の違いは打ち消される) と FK の動きの向きの一致 (cos) と、動く量で判定する
2. 模擬の遅れ: 教師の指令を tau ごとに会場の模擬に通した腕の遅れ (会場の log と同じ指標 tracking_metrics)。
   記録された腕 (自前の実機) の遅れも出す。09-29 の会場: 台を回す 260〜280 ms・pick 370〜390 ms、自前 80〜110 ms
3. 表を作る時間 (区間の一部で測り、全体・tau 5 通りを見積もる)

usage (repo root で):
    python -m model.ramen_ori.scripts.check_venue_arm \\
        --src /nvme/unified_cache/rotate_table_base/src --skill rotate_table_base --json /nvme/logs/check_venue_arm_rotate.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from model.ramen_ori.venue_arm import (
    ARM_SLICE_Q36,
    FPS,
    command19_from,
    simulate_episode,
)

REPO = Path(__file__).resolve().parents[3]
SKILL_CONFIG = REPO / "inference/desktop/lower_policy/configs/skill_config.yaml"
TAUS = (0.05, 0.1, 0.15, 0.2, 0.3)
COLUMNS = (
    "index",
    "episode_index",
    "observation.state.robot_q_current",
    "observation.state.ee_state",
    "action.robot_q_desired",
    "action.hand_cmd",
)


def load_columns(src: Path) -> dict[str, np.ndarray]:
    """LeRobot v3 の data/*.parquet を行の番号順に読む。"""
    files = sorted((src / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{src}/data に parquet が無い")
    tables = [pq.read_table(f, columns=list(COLUMNS)) for f in files]
    out = {}
    for name in COLUMNS:
        chunks = [t.column(name).to_numpy(zero_copy_only=False) for t in tables]
        col = np.concatenate(chunks)
        out[name] = np.stack(col).astype(np.float32) if col.dtype == object else col
    order = np.argsort(out["index"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def episodes(episode_index: np.ndarray) -> list[tuple[int, int]]:
    """(先頭の行, 長さ) の list。区間は連続して並んでいる前提 (違えば止める)。"""
    cuts = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(episode_index)]])
    if len(set(episode_index[starts].tolist())) != len(starts):
        raise ValueError("区間が連続して並んでいない")
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def check_fk(cols: dict, skill: str, n_samples: int, seed: int) -> dict:
    from inference.desktop.assembly import FkFactory

    fk = FkFactory().for_skill(
        yaml.safe_load(SKILL_CONFIG.read_text(encoding="utf-8")), skill
    )
    if fk is None:
        raise FileNotFoundError("G1 の URDF が無い")
    q, rec = (
        cols["observation.state.robot_q_current"],
        cols["observation.state.ee_state"],
    )
    rows = np.random.default_rng(seed).choice(
        len(q), size=min(n_samples, len(q)), replace=False
    )
    result = {}
    for label, sl in (("joints_7_36", slice(7, 36)), ("old_0_29", slice(0, 29))):
        ee = np.stack([fk.compute_ee_state(q[r, sl]) for r in rows])
        for side, pos in (("left", slice(0, 3)), ("right", slice(6, 9))):
            d = np.linalg.norm(ee[:, pos] - rec[rows, pos], axis=1) * 1000
            result[f"{label}/{side}_mm"] = {
                "mean": float(d.mean()), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            }  # fmt: skip
    return result


def check_fk_motion(cols: dict, eps: list[tuple[int, int]], skill: str, n_episodes: int, seed: int) -> dict:
    """記録の手先と FK の手先の、0.1 s ごとの動きの向きの一致 (cos) と動く量 [mm]。区間ごとの中央値の中央値。"""
    from inference.desktop.assembly import FkFactory

    fk = FkFactory().for_skill(yaml.safe_load(SKILL_CONFIG.read_text(encoding="utf-8")), skill)
    q, rec = cols["observation.state.robot_q_current"], cols["observation.state.ee_state"]
    labels = ("joints_7_36", "old_0_29")
    cos_by = {(lab, side): [] for lab in labels for side in ("left", "right")}
    move_by = {(lab, side): [] for lab in labels + ("recorded",) for side in ("left", "right")}
    pick = np.random.default_rng(seed).choice(len(eps), size=min(n_episodes, len(eps)), replace=False)
    for i in pick:
        start, n = eps[i]
        rows = list(range(start, start + n, 3))   # 0.1 s おき
        r = rec[rows]
        for label, sl in (("joints_7_36", slice(7, 36)), ("old_0_29", slice(0, 29))):
            e = np.stack([fk.compute_ee_state(q[k, sl]) for k in rows])
            for side, pos in (("left", slice(0, 3)), ("right", slice(6, 9))):
                dr, de = np.diff(r[:, pos], axis=0), np.diff(e[:, pos], axis=0)
                moving = np.linalg.norm(dr, axis=1) > 2e-3   # 記録の手先が 0.1 s で 2 mm 以上動いた所
                if moving.sum() < 5:
                    continue
                dr, de = dr[moving], de[moving]
                cos = (dr * de).sum(1) / (np.linalg.norm(dr, axis=1) * np.linalg.norm(de, axis=1) + 1e-9)
                cos_by[(label, side)].append(float(np.median(cos)))
                move_by[(label, side)].append(float(np.linalg.norm(de, axis=1).mean() * 1000))
                if label == "joints_7_36":
                    move_by[("recorded", side)].append(float(np.linalg.norm(dr, axis=1).mean() * 1000))
    result = {}
    for (label, side), v in cos_by.items():
        result[f"{label}/{side}_cos"] = float(np.median(v)) if v else None
    for (label, side), v in move_by.items():
        result[f"{label}/{side}_move_mm"] = float(np.median(v)) if v else None
    return result


def check_lag(
    cols: dict, eps: list[tuple[int, int]], n_episodes: int, seed: int, preset: str
) -> dict:
    from inference.desktop.lower_policy.scripts.tracking_summary import tracking_metrics

    command19 = command19_from(cols["action.robot_q_desired"], cols["action.hand_cmd"])
    measured = cols["observation.state.robot_q_current"][:, ARM_SLICE_Q36]
    pick = np.random.default_rng(seed).choice(
        len(eps), size=min(n_episodes, len(eps)), replace=False
    )
    chosen = [eps[i] for i in sorted(pick)]

    def _segments(arm_of):
        out = []
        for s, n in chosen:
            t = np.arange(n) / FPS
            out.append((t, command19[s : s + n, 3:17], arm_of(s, n)))
        return out

    result = {
        "episodes": len(chosen),
        "recorded": _summary(
            tracking_metrics(_segments(lambda s, n: measured[s : s + n]))
        ),
    }
    robot_seconds = sum(n for _, n in chosen) / FPS
    for tau in TAUS:
        t0 = time.time()
        segs = _segments(
            lambda s, n: simulate_episode(
                command19[s : s + n], measured[s : s + n], tau, preset
            )
        )
        wall = time.time() - t0
        result[f"tau_{tau:g}"] = {
            **_summary(tracking_metrics(segs)),
            "sim_ms_per_robot_s": wall / robot_seconds * 1000,
        }
    return result


def _summary(metrics: dict | None) -> dict:
    if metrics is None:
        return {"lag_ms": None, "mean_abs_error_rad": None}
    return {
        "lag_ms": metrics["lag_ms"],
        "mean_abs_error_rad": metrics["mean_abs_error_rad"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="/nvme/unified_cache/<skill>/src (meta・data)",
    )
    ap.add_argument("--skill", required=True)
    ap.add_argument("--fk-samples", type=int, default=3000)
    ap.add_argument("--fk-episodes", type=int, default=40)
    ap.add_argument("--lag-episodes", type=int, default=150)
    ap.add_argument("--preset", default="standard")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="表の見積もりに使う process 数 (0 = CPU 数)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import os

    cols = load_columns(args.src)
    eps = episodes(cols["episode_index"])
    total_s = len(cols["index"]) / FPS
    print(
        f"[data] {args.skill}: {len(eps)} 区間 / {len(cols['index'])} frame ({total_s / 60:.0f} 分)"
    )
    fk = check_fk(cols, args.skill, args.fk_samples, args.seed)
    print("[fk] 記録された手先との距離 [mm] (mean / median / p90):")
    for key, v in fk.items():
        print(f"  {key:22s} {v['mean']:7.1f} / {v['median']:7.1f} / {v['p90']:7.1f}")
    motion = check_fk_motion(cols, eps, args.skill, args.fk_episodes, args.seed)
    print("[fk] 手先の動きの向きの一致 (cos、1 = 同じ向き・0 = 無関係) / 0.1 s の移動量 [mm] (記録が動いた所):")
    for label in ("joints_7_36", "old_0_29", "recorded"):
        cos = "" if label == "recorded" else (
            f"cos 左 {motion[f'{label}/left_cos']:+.2f} 右 {motion[f'{label}/right_cos']:+.2f}  "
        )
        print(f"  {label:12s} {cos}移動 左 {motion[f'{label}/left_move_mm']:5.1f} 右 {motion[f'{label}/right_move_mm']:5.1f}")
    fk.update(motion)
    lag = check_lag(cols, eps, args.lag_episodes, args.seed, args.preset)
    print(
        f"[lag] 教師の指令に対する腕の遅れ ({lag['episodes']} 区間、会場の log と同じ指標):"
    )
    print(
        f"  記録された腕 (自前の実機)  遅れ {lag['recorded']['lag_ms']} ms  ずれ {lag['recorded']['mean_abs_error_rad']:.4f} rad"
    )
    workers = args.workers or os.cpu_count() or 1
    for tau in TAUS:
        r = lag[f"tau_{tau:g}"]
        est = r["sim_ms_per_robot_s"] / 1000 * total_s / workers
        print(f"  模擬 tau {tau:4.2f} s         遅れ {r['lag_ms']} ms  ずれ {r['mean_abs_error_rad']:.4f} rad"
              f"  (表 1 本 ≈ {est:.0f} s、{workers} process)")  # fmt: skip
    if args.json:
        args.json.write_text(
            json.dumps(
                {"skill": args.skill, "fk": fk, "lag": lag},
                indent=1,
                ensure_ascii=False,
            )
        )
        print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
