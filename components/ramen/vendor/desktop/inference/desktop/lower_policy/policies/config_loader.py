"""Policy variant config loader + factory (Issue #125 Phase 7)。

`inference/desktop/lower_policy/configs/policy_config.yaml` から variant
定義を読み、PolicyConfig を組み立て、対応 Policy class (Gr00tPolicy /
RamenOriPolicy) を返す factory。

# 使用例

```python
from inference.desktop.lower_policy.policies.config_loader import (
    load_policy_variant, list_variants,
)

# 全 variant list
print(list_variants("configs/policy_config.yaml"))
# → ["groot_baseline", "groot_overlay", "ramen_ori_default", ...]

# variant → Policy (lazy load、ckpt DL + model instantiate は from_ckpt() 内)
cfg, policy_cls = load_policy_variant("configs/policy_config.yaml", "ramen_ori_default")
policy = policy_cls.from_ckpt(cfg)
```

# YAML schema

```yaml
policies:
  <variant_name>:
    policy_type: "groot" | "ramen_ori"
    mode: "none" | "overlay" | "precomputed_token"
    ckpt_ref: "hf_repo@sha" | "/local/path" | null
    device: "cuda" | "cpu"
    dtype: "fp32" | "bf16" | "fp16"
    hydra_overrides: optional list[str] for RAMEN-Ori architecture variants

yolo:
  ckpt_ref: "hf_repo@sha"
```

# safe_load 使用

YAML は `yaml.safe_load` で parse (`!!python/object` tag RCE 回避、CLAUDE.md 準拠)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from inference.desktop.lower_policy.policies.base import PolicyConfig


DEFAULT_CONFIG_PATH = Path(
    "inference/desktop/lower_policy/configs/policy_config.yaml"
)


@dataclass(frozen=True)
class VariantEntry:
    """1 variant の parse 済 config。

    Attributes:
        name: variant identifier (yaml section key、例 "ramen_ori_default")。
        policy_type: "groot" or "ramen_ori"。factory dispatch key。
        policy_config: PolicyConfig instance (mode / ckpt_ref / device / dtype /
            cams が埋まった状態、cams は policy_type から自動決定)。
        yolo_ckpt_ref: shared YOLO weight ref (overlay / precomputed_token mode
            で使う、mode=none では None が返る場合あり)。
    """

    name: str
    policy_type: str
    policy_config: PolicyConfig
    yolo_ckpt_ref: str | None


def _load_yaml(path: str | Path) -> dict:
    """safe_load で YAML を読む (RCE 回避、CLAUDE.md 準拠)。"""
    with Path(path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"policy config must be a dict, got {type(data)}")
    return data


def list_variants(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """全 variant 名を返す (dict order = YAML insertion order)。"""
    data = _load_yaml(config_path)
    policies = data.get("policies", {})
    if not isinstance(policies, dict):
        raise ValueError(f"policies section must be a dict, got {type(policies)}")
    return list(policies.keys())


def load_policy_variant(
    config_path: str | Path, variant_name: str
) -> VariantEntry:
    """variant 名 → 完全に組み立った VariantEntry (PolicyConfig + factory dispatch key)。

    Args:
        config_path: policy_config.yaml への path。
        variant_name: variant identifier (yaml `policies` section の key)。

    Returns:
        VariantEntry。呼出側は entry.policy_type で dispatch し、entry.policy_config
        を Policy.from_ckpt() に渡す。

    Raises:
        KeyError: variant_name が config に無い。
        ValueError: config field が不正 (policy_type 不明 / ckpt_ref=null 等)。
    """
    data = _load_yaml(config_path)
    policies = data.get("policies", {})
    if variant_name not in policies:
        available = ", ".join(sorted(policies.keys())) if policies else "(none)"
        raise KeyError(
            f"variant {variant_name!r} not in policy config. Available: {available}"
        )

    entry = policies[variant_name]
    if not isinstance(entry, dict):
        raise ValueError(
            f"variant {variant_name!r} entry must be a dict, got {type(entry)}"
        )

    policy_type = entry.get("policy_type")
    if policy_type not in ("groot", "ramen_ori", "groot_pick_legs"):
        raise ValueError(
            f"variant {variant_name!r}: policy_type must be one of "
            f"'groot' / 'ramen_ori' / 'groot_pick_legs', got {policy_type!r}"
        )

    mode = entry.get("mode")
    ckpt_ref = entry.get("ckpt_ref")
    if ckpt_ref is None:
        raise ValueError(
            f"variant {variant_name!r}: ckpt_ref is null (ckpt not yet trained "
            f"or not yet pushed to HF). Cannot instantiate Policy."
        )

    hydra_overrides_raw = entry.get("hydra_overrides", [])
    if not isinstance(hydra_overrides_raw, list) or not all(
        isinstance(value, str) and value for value in hydra_overrides_raw
    ):
        raise ValueError(
            f"variant {variant_name!r}: hydra_overrides must be a list of "
            "non-empty strings"
        )
    if hydra_overrides_raw and policy_type != "ramen_ori":
        raise ValueError(
            f"variant {variant_name!r}: hydra_overrides are only supported for "
            "policy_type='ramen_ori'"
        )

    # cams は policy_type から自動決定 (それぞれの CAMERAS 定数を使う)
    if policy_type == "groot":
        # lazy import to avoid loading heavy policy module unless needed
        from inference.desktop.lower_policy.policies.groot import CAMERAS as GROOT_CAMS

        cams = GROOT_CAMS
    elif policy_type == "groot_pick_legs":
        from inference.desktop.lower_policy.policies.groot_pick_legs import (
            CAMERAS as PICK_LEGS_CAMS,
        )

        cams = PICK_LEGS_CAMS
    else:
        from inference.desktop.lower_policy.policies.ramen_ori import (
            CAMERAS as RAMEN_CAMS,
        )

        cams = RAMEN_CAMS

    # Phase 2/4 optional fields (指定無しは PolicyConfig の default 継承)
    ctor_kwargs: dict = dict(
        mode=mode,
        ckpt_ref=str(ckpt_ref),
        checkpoint_subdir=(
            str(entry["checkpoint_subdir"])
            if entry.get("checkpoint_subdir") is not None
            else None
        ),
        device=str(entry.get("device", "cuda")),
        dtype=str(entry.get("dtype", "fp32")),
        cams=cams,
        hydra_overrides=tuple(hydra_overrides_raw),
    )
    if "temporal_lambda" in entry:
        # None (blend 無効) を明示指定可能、その場合そのまま渡す
        ctor_kwargs["temporal_lambda"] = entry["temporal_lambda"]
    if "replan_family" in entry:
        ctor_kwargs["replan_family"] = entry["replan_family"]
    if "execution_steps" in entry:
        ctor_kwargs["execution_steps"] = int(entry["execution_steps"])
    cfg = PolicyConfig(**ctor_kwargs)

    # shared YOLO ckpt ref (mode!=none の Policy が使う想定、mode=none でも参照可能)
    yolo_section = data.get("yolo", {})
    yolo_ckpt_ref = yolo_section.get("ckpt_ref") if isinstance(yolo_section, dict) else None

    return VariantEntry(
        name=variant_name,
        policy_type=policy_type,
        policy_config=cfg,
        yolo_ckpt_ref=yolo_ckpt_ref,
    )


def resolve_policy_class(policy_type: str):
    """policy_type → Policy class (lazy import で heavy module ロード回避)。

    Args:
        policy_type: "groot" or "ramen_ori"。

    Returns:
        Policy class (Gr00tPolicy or RamenOriPolicy)。呼出側は class.from_ckpt(cfg)
        で instantiate。
    """
    if policy_type == "groot":
        from inference.desktop.lower_policy.policies.groot import Gr00tPolicy

        return Gr00tPolicy
    if policy_type == "groot_pick_legs":
        from inference.desktop.lower_policy.policies.groot_pick_legs import (
            Gr00tPolicyPickLegs,
        )

        return Gr00tPolicyPickLegs
    if policy_type == "ramen_ori":
        from inference.desktop.lower_policy.policies.ramen_ori import RamenOriPolicy

        return RamenOriPolicy
    raise ValueError(f"unknown policy_type: {policy_type!r}")
