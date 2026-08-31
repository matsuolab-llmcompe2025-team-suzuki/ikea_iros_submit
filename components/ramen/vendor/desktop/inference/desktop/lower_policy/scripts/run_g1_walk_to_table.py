"""G1 walk-to-table 実機動作確認script (Issue #64)。

# 実行前に必ず確認する安全前提

- ハーネス or 天井吊り支持棒あり (**初回歩行は必須**)
- E-stop リモコン (Unitree 標準付属) を人が握って待機
- clearance: 前方 3m 以上を確保
- バッテリ 50% 以上、満充電推奨
- 硬い床は不可、マット敷き推奨

# 使用例

## dry-run (SDK 呼ばず MockActuator で疎通確認、main env で OK)

    pixi run python -m inference.desktop.lower_policy.scripts.run_g1_walk_to_table --dry-run

## 実機 (低速 0.15 m/s、1 秒) — `runtime` env が必要

    pixi run -e runtime python -m inference.desktop.lower_policy.scripts.run_g1_walk_to_table \\
        --interface eth0 --duration 1 --vx 0.15

interface 名の調べ方: `ip a`。G1 に直結の Ethernet iface (`enp2s0` / `eth0` 等)。

# 実機起動時の前提

本 script は robot が **walking FSM state 501 (3-DoF waist Regular Mode)** に
既に入っていることを前提とする。速度指令の前後に姿勢/FSM命令は送らない。
Dampからのstartupはoperatorの責任、本scriptのscope外。
"""

from __future__ import annotations

import argparse
import sys
import time

from inference.desktop.lower_policy.actuators.mock import MockWalkActuator
from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
from inference.desktop.lower_policy.skills.move_to_table import (
    DEFAULT_WALK_VX,
    MoveToTable,
)

# `--vx` 上限。実機で慎重に見て安全と言える範囲 (SDK example の 0.3 でも
# 実測で速すぎたため 0.5 を hard cap に。実測で調整可)。argparse 側で強制。
_VX_MAX = 0.5
_DURATION_MAX = 10.0


def _clamped_vx(s: str) -> float:
    """argparse 用 `--vx` type: [0, _VX_MAX] にない値を弾く。"""
    v = float(s)
    if not 0.0 <= v <= _VX_MAX:
        raise argparse.ArgumentTypeError(
            f"vx must be in [0, {_VX_MAX}], got {v}"
        )
    return v


def _bounded_duration(s: str) -> float:
    """argparse用duration: 正の有限値かつ安全上限以下に制限する。"""
    v = float(s)
    if not 0.1 <= v <= _DURATION_MAX:
        raise argparse.ArgumentTypeError(
            f"duration must be in [0.1, {_DURATION_MAX}], got {v}"
        )
    return v


def _build_actuator(dry_run: bool, interface: str, domain_id: int):
    if dry_run:
        return MockWalkActuator()
    # lazy import: dry-run path では SDK に触れない
    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator

    return G1SDKWalkActuator(interface=interface, domain_id=domain_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="G1 walk-to-table 実機動作確認 (低速前進 → 停止)",
    )
    parser.add_argument(
        "--interface",
        default="eth0",
        help="DDS network interface (`ip a` で確認、実機直結の Ethernet)",
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--expected-fsm-id",
        type=int,
        default=501,
        help="実行を許可する高レベルFSM ID。3-DoF waist Regular Modeは501",
    )
    parser.add_argument(
        "--vx",
        type=_clamped_vx,
        default=0.185,
        help=(
            "前進速度 [m/s]。default 0.185 は Issue #64 実機決め打ち運用値、"
            f"entrypoint.py の pipeline default と揃えてある (skill 内の"
            f" DEFAULT_WALK_VX={DEFAULT_WALK_VX} は gait threshold 実測値で意味が別)。"
            f" 公式example は 0.3、上限 {_VX_MAX}"
        ),
    )
    parser.add_argument(
        "--duration",
        type=_bounded_duration,
        default=1.0,
        help=f"前進時間 [s]。実機診断は1.0推奨。範囲0.1-{_DURATION_MAX}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="SDK 呼ばず MockActuator で疎通確認 (main env で実行可能)",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="実機sport API/FSMを読み取るだけで速度・姿勢指令を送らない",
    )
    args = parser.parse_args()

    actuator = _build_actuator(args.dry_run, args.interface, args.domain_id)

    if not args.dry_run:
        status = actuator.get_loco_status()
        print(
            "[preflight] "
            f"sport_api={status.client_api_version}/{status.server_api_version} "
            f"fsm_id={status.fsm_id} fsm_mode={status.fsm_mode}",
            flush=True,
        )
        if status.client_api_version != status.server_api_version:
            print("ERROR: sport client/server API version mismatch", file=sys.stderr)
            return 2
        if args.diagnose_only:
            print("[diagnose-only] no motor command sent")
            return 0
        if status.fsm_id != args.expected_fsm_id:
            print(
                "ERROR: refusing velocity command: "
                f"expected fsm_id={args.expected_fsm_id}, got {status.fsm_id}",
                file=sys.stderr,
            )
            return 2
        if 0.0 < args.vx < DEFAULT_WALK_VX:
            print(
                f"WARNING: vx < {DEFAULT_WALK_VX} m/s may be below the "
                "gait-start range; "
                "RPC success alone does not prove locomotion.",
                file=sys.stderr,
            )
        print("!! 実機 mode !!", file=sys.stderr)
        print(
            "  harness / E-stop / 3m clearance / battery 50%+ を確認したか?",
            file=sys.stderr,
        )
        print("  [Enter] で続行、Ctrl-C で中止", file=sys.stderr)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("aborted.", file=sys.stderr)
            return 1

    # Pass --duration to the SDK as well as keeping the host-side stop below.
    # The latter is retained as a fail-safe if the firmware-side timeout fails.
    skill = MoveToTable(actuator, vx=args.vx, duration=args.duration)
    policy = SkillDispatchLowerPolicy({"move_to_table": skill})

    print(f"[start] vx={args.vx} m/s, duration={args.duration}s")
    policy.start("move_to_table", {})
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        # SDK 側にも有限の停止時刻を渡すが、Ctrl-C 時を含めて明示停止する。
        print("\n[interrupt] stopping robot ...", file=sys.stderr)
    finally:
        print("[stop]")
        policy.stop()

    if args.dry_run:
        events = actuator.events
        print(f"\nMockActuator events ({len(events)}):")
        for e in events:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
