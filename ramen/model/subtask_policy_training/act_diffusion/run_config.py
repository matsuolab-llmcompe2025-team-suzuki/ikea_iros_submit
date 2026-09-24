"""ACT / Diffusion Policy の run 設定 (YAML) を検証済みの shell export に変換する (Issue #139)。

`scripts/rotate_specialist/config.py` と同じ interface だが、あちらは必須 key に GR00T の
aux loss 重み (`GROOT_*`) を含むため ACT / DP には使えない。

使用例 (`act_diffusion/run.sh` から):
    eval "$(python act_diffusion/run_config.py --config X.yaml --set TRAIN_STEPS=40 --format shell)"
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_POLICY_TYPES = ("act", "diffusion")
REQUIRED_KEYS = (
    "SUBTASK",
    "POLICY_TYPE",
    "POLICY_REPO_ID",
    "TRAIN_BATCH_SIZE",
    "TRAIN_STEPS",
    "TRAIN_SAVE_FREQ",
)
# 学習終了時の LeRobot 標準 upload は repo 上の「最終 model 以外」を削除する
# (scripts/upload_policy.py の delete_patterns、LeRobot の push_to_hub も root に上書き)。
# 途中 ckpt は ckpt_uploader が checkpoints/<step>/ に積むので、両方を必ず無効にする。
MUST_BE_FALSE = ("UPLOAD_AFTER_TRAIN", "PUSH_TO_HUB")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("env"), dict):
        raise ValueError(f"{path}: top-level `env` mapping is required")
    return cfg


def to_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_env(cfg: dict[str, Any], overrides: list[str]) -> dict[str, str]:
    env = {str(k): to_env_value(v) for k, v in cfg["env"].items()}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be KEY=VALUE: {item!r}")
        env[key] = value
    bad_names = [k for k in env if not ENV_NAME.match(k)]
    if bad_names:
        raise ValueError(f"invalid env names: {bad_names}")
    missing = [k for k in REQUIRED_KEYS if not env.get(k)]
    if missing:
        raise ValueError(f"missing required keys: {missing}")
    if env["POLICY_TYPE"] not in SUPPORTED_POLICY_TYPES:
        raise ValueError(f"POLICY_TYPE must be one of {SUPPORTED_POLICY_TYPES}: {env['POLICY_TYPE']!r}")
    enabled = [k for k in MUST_BE_FALSE if env.get(k, "false").lower() != "false"]
    if enabled:
        raise ValueError(
            f"{enabled} must be false: the final upload deletes checkpoints/<step>/ "
            "that ckpt_uploader keeps on the same repo"
        )
    for key in MUST_BE_FALSE:
        env[key] = "false"
    return env


def format_shell(env: dict[str, str]) -> str:
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())


def format_summary(cfg: dict[str, Any], env: dict[str, str]) -> str:
    lines = [f"run_id: {cfg.get('run_id', '(none)')}"]
    lines += [f"  {k}={v}" for k, v in env.items()]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--format", choices=("shell", "summary"), default="shell")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    env = resolve_env(cfg, args.overrides)
    print(format_shell(env) if args.format == "shell" else format_summary(cfg, env))
    return 0


if __name__ == "__main__":
    sys.exit(main())
