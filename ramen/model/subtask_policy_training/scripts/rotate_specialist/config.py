"""Issue #129 rotate specialist run config (YAML) を shell export 文字列に変換。

Usage:
    python scripts/rotate_specialist/config.py \\
        --config configs/rotate_specialist/run1_t01_only.yaml \\
        --format shell

Output:
    export GROOT_LEFT_HAND_LOSS_WEIGHT=2.0
    export GROOT_BILATERAL_LOSS_WEIGHT=0.0
    ...

Shell wrapper (`scripts/rotate_specialist/run.sh`) からは:
    eval "$(python scripts/rotate_specialist/config.py --config X.yaml --format shell)"
で env を setup。

# 責務境界

- 本 script は YAML 読取 + 値検証 + shell 用 escape だけを行う (env の実 export は shell 側)
- Patch install や train_lerobot.sh 起動は run.sh (launcher) が担当
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ENV_KEYS = (
    "GROOT_LEFT_HAND_LOSS_WEIGHT",
    "GROOT_BILATERAL_LOSS_WEIGHT",
    "GROOT_FK_ANCHOR_LOSS_WEIGHT",
    "SUBTASK",
    "POLICY_TYPE",
    "HF_AUTOPUSH_REPO_ID",
)


def load_config(path: Path) -> dict[str, Any]:
    """YAML file を load、structure 検証。"""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a dict, got {type(cfg).__name__}")
    for req in ("run_id", "env"):
        if req not in cfg:
            raise ValueError(f"config {path} missing required key: {req}")
    env = cfg["env"]
    if not isinstance(env, dict):
        raise ValueError(f"config env must be a dict, got {type(env).__name__}")
    missing = [k for k in REQUIRED_ENV_KEYS if k not in env]
    if missing:
        raise ValueError(f"config {path} env missing required keys: {missing}")
    return cfg


def env_to_shell(env: dict[str, Any]) -> str:
    """env dict を shell export 文字列 (改行区切り) に変換。値は shlex.quote で安全化。"""
    lines = []
    for k, v in env.items():
        if not isinstance(k, str) or not k.replace("_", "").isalnum():
            raise ValueError(f"invalid env var name: {k!r}")
        v_str = str(v)  # bool は 'True'/'False'、float は '2.0' 等
        lines.append(f"export {k}={shlex.quote(v_str)}")
    return "\n".join(lines)


def env_to_summary(env: dict[str, Any]) -> str:
    """env dict を人間読みやすい summary 文字列に。"""
    return "\n".join(f"  {k}={v}" for k, v in env.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "--format",
        choices=("shell", "summary"),
        default="shell",
        help="output format (shell = export lines、summary = human readable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.format == "shell":
        print(env_to_shell(cfg["env"]))
    elif args.format == "summary":
        print(f"# rotate specialist run: {cfg.get('run_id')}")
        print(f"# {cfg.get('description', '')}")
        print("# env:")
        print(env_to_summary(cfg["env"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
