"""Policy variant config loader + factory (Issue #125 Phase 7)。

`inference/desktop/lower_policy/configs/policy_config.yaml` から variant
定義を読み、PolicyConfig を組み立て、対応 Policy class (Gr00tPolicy /
RamenOriPolicy / ActDiffusionPolicy 等) を返す factory。

# 使用例

```python
from inference.desktop.lower_policy.policies.config_loader import (
    load_policy_variant, list_variants,
)

# 全 variant list
print(list_variants("configs/policy_config.yaml"))
# → ["groot_overlay", "ramen_ori_default", ...]

# variant → Policy (lazy load、ckpt DL + model instantiate は from_ckpt() 内)
cfg, policy_cls = load_policy_variant("configs/policy_config.yaml", "ramen_ori_default")
policy = policy_cls.from_ckpt(cfg)
```

# YAML schema

```yaml
default_variant_by_skill:
  <skill_name>: <variant_name>   # 本番 run の既定 (Issue #148)

policies:
  <variant_name>:
    policy_type: "groot" | "ramen_ori" | "groot_pick_legs" | "act_diffusion"
    mode: "none" | "overlay" | "precomputed_token"
    ckpt_ref: "hf_repo@sha" | "/local/path" | null
    device: "cuda" | "cpu"
    dtype: "fp32" | "bf16" | "fp16"
    hydra_overrides: optional list[str] for RAMEN-Ori architecture variants
    overlay_jpeg_subsampling: "4:4:4" (default) | "4:2:0"  # mode=overlay のみ、要クォート

yolo:
  ckpt_ref: "hf_repo@sha"
```

# safe_load 使用

YAML は `yaml.safe_load` で parse (`!!python/object` tag RCE 回避、CLAUDE.md 準拠)。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from inference.desktop.lower_policy.policies.base import (
    OVERLAY_JPEG_SUBSAMPLINGS,
    CameraKey,
    PolicyConfig,
)
from inference.desktop.lower_policy.rtc import RtcConfig


DEFAULT_CONFIG_PATH = Path(
    "inference/desktop/lower_policy/configs/policy_config.yaml"
)


@dataclass(frozen=True)
class VariantEntry:
    """1 variant の parse 済 config。

    Attributes:
        name: variant identifier (yaml section key、例 "ramen_ori_default")。
        policy_type: "groot" / "ramen_ori" / "groot_pick_legs" / "act_diffusion"。
            factory dispatch key。
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


def load_default_variant_by_skill(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, str]:
    """`default_variant_by_skill` section を読む (Issue #148)。

    本番 run で「どの skill をどの ckpt で走らせるか」の正本。呼出側 (entrypoint)
    は CLI が渡さなかった slot だけをこれで埋める。

    ここでは値が registry にある variant 名かどうかだけを見る。skill 名が有効か
    は slot と CLI 引数の対応を持っている呼出側が判定する (対応表をここに複製
    しない)。
    """
    data = _load_yaml(config_path)
    section = data.get("default_variant_by_skill")
    if not isinstance(section, dict):
        raise ValueError(
            f"{config_path}: 'default_variant_by_skill' section is required"
        )
    policies = data.get("policies")
    if not isinstance(policies, dict):
        raise ValueError(f"{config_path}: 'policies' section is required")
    defaults: dict[str, str] = {}
    for skill_name, variant in section.items():
        if not isinstance(variant, str) or not variant:
            raise ValueError(
                f"{config_path}: default_variant_by_skill.{skill_name} must be a "
                f"variant name, got {variant!r}"
            )
        if variant not in policies:
            raise ValueError(
                f"{config_path}: default_variant_by_skill.{skill_name}={variant!r} "
                "is not registered under policies"
            )
        defaults[str(skill_name)] = variant
    return defaults


_RTC_KEYS = frozenset(
    {
        "enabled",
        "frozen_steps",
        "overlap_steps",
        "ramp_rate",
        "allow_experimental_relative_action",
    }
)

# RTC の prefix はモデルの action space に載せ直す必要があり、その経路は
# policy 実装ごとに用意する。native raw embodiment の pick_legs は別 schema で
# 未対応なので、silent no-op ではなく起動時に落とす。
_RTC_SUPPORTED_POLICY_TYPES = frozenset({"groot", "ramen_ori"})


def _parse_rtc(raw: object, variant_name: str, policy_type: str) -> RtcConfig:
    """YAML の `rtc:` ブロック → RtcConfig。null は既定 (無効) 扱い。"""
    if raw is None:
        return RtcConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"variant {variant_name!r}: rtc must be a mapping or null, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - _RTC_KEYS)
    if unknown:
        raise ValueError(
            f"variant {variant_name!r}: rtc has unknown keys {unknown} — "
            f"allowed: {sorted(_RTC_KEYS)}"
        )
    kwargs: dict = {}
    if "enabled" in raw:
        kwargs["enabled"] = raw["enabled"]
    if "frozen_steps" in raw:
        # "auto" (実行時に実測 latency から算出) か int 固定値。
        kwargs["frozen_steps"] = raw["frozen_steps"]
    if "overlap_steps" in raw:
        kwargs["overlap_steps"] = raw["overlap_steps"]
    if "ramp_rate" in raw:
        ramp_rate = raw["ramp_rate"]
        if isinstance(ramp_rate, bool) or not isinstance(ramp_rate, (int, float)):
            raise ValueError(
                f"variant {variant_name!r}: rtc.ramp_rate must be a number, "
                f"got {ramp_rate!r}"
            )
        kwargs["ramp_rate"] = float(ramp_rate)
    if "allow_experimental_relative_action" in raw:
        kwargs["allow_experimental_relative_action"] = raw[
            "allow_experimental_relative_action"
        ]
    try:
        rtc = RtcConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"variant {variant_name!r}: invalid rtc — {exc}") from exc
    if rtc.enabled and policy_type not in _RTC_SUPPORTED_POLICY_TYPES:
        raise ValueError(
            f"variant {variant_name!r}: rtc.enabled is only supported for "
            f"policy_type in {sorted(_RTC_SUPPORTED_POLICY_TYPES)}, "
            f"got policy_type={policy_type!r}"
        )
    return rtc


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
    if policy_type not in ("groot", "ramen_ori", "groot_pick_legs", "act_diffusion"):
        raise ValueError(
            f"variant {variant_name!r}: policy_type must be one of "
            f"'groot' / 'ramen_ori' / 'groot_pick_legs' / 'act_diffusion', "
            f"got {policy_type!r}"
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

    # cams は policy_type の default を採用、YAML で明示指定あれば override
    # (Issue #132 Phase C: Phase K Run 5 は 3 cam 学習 = head_left + wrist L/R、
    # slot YAML で `cams: [head_left, wrist_left, wrist_right]` を明示指定して
    # 既存 4 cam default を override する)
    if policy_type == "groot":
        # lazy import to avoid loading heavy policy module unless needed
        from inference.desktop.lower_policy.policies.groot import CAMERAS as GROOT_CAMS

        cams = GROOT_CAMS
    elif policy_type == "groot_pick_legs":
        from inference.desktop.lower_policy.policies.groot_pick_legs import (
            CAMERAS as PICK_LEGS_CAMS,
        )

        cams = PICK_LEGS_CAMS
    elif policy_type == "act_diffusion":
        from inference.desktop.lower_policy.policies.act_diffusion import (
            CAMERAS as ACT_DIFFUSION_CAMS,
        )

        cams = ACT_DIFFUSION_CAMS
    else:
        from inference.desktop.lower_policy.policies.ramen_ori import (
            CAMERAS as RAMEN_CAMS,
        )

        cams = RAMEN_CAMS

    if "cams" in entry:
        raw_cams = entry["cams"]
        if not isinstance(raw_cams, list) or not all(
            isinstance(name, str) and name for name in raw_cams
        ):
            raise ValueError(
                f"variant {variant_name!r}: cams must be a list of non-empty "
                f"CameraKey strings (e.g. 'head_left'), got {raw_cams!r}"
            )
        try:
            cams = tuple(CameraKey(name) for name in raw_cams)
        except ValueError as exc:
            raise ValueError(
                f"variant {variant_name!r}: cams contains unknown CameraKey — "
                f"{exc}"
            ) from exc

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
    # Issue #132 Phase A: per-variant language prompt override (GR00T のみ意味を持つ)。
    # 未指定なら None、VlaSkill 側で class-level LANGUAGE (subtask_training.json:
    # <subtask>.task と一致) にフォールバックする。
    if "language_prompt" in entry:
        raw = entry["language_prompt"]
        if raw is not None and not isinstance(raw, str):
            raise ValueError(
                f"variant {variant_name!r}: language_prompt must be str or null, "
                f"got {type(raw).__name__}"
            )
        ctor_kwargs["language_prompt"] = raw
    # Issue #141 (1-2 / INF-3): per-variant skill_id override (RAMEN-Ori のみ意味を持つ)。
    # 未指定なら None、VlaSkill 側で class-level SKILL_ID にフォールバックする。
    if "skill_id" in entry:
        raw = entry["skill_id"]
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise ValueError(
                f"variant {variant_name!r}: skill_id must be int or null, "
                f"got {type(raw).__name__}"
            )
        ctor_kwargs["skill_id"] = raw
    # Issue #132 Phase C: action space (abs / rel) per-variant switch。
    # rel は RAMEN-Ori Phase K Run 3/5 (normalized Δq arms 14D + abs hand 2D)
    # 用、inference 側で cumsum 復元経路が走る。GR00T では未対応 = 早期 reject。
    if "action_space" in entry:
        action_space = entry["action_space"]
        if not isinstance(action_space, str) or action_space not in ("abs", "rel"):
            raise ValueError(
                f"variant {variant_name!r}: action_space must be 'abs' or 'rel', "
                f"got {action_space!r}"
            )
        if action_space == "rel" and policy_type != "ramen_ori":
            raise ValueError(
                f"variant {variant_name!r}: action_space='rel' is only supported "
                f"for policy_type='ramen_ori', got policy_type={policy_type!r}"
            )
        ctor_kwargs["action_space"] = action_space
    # Issue #137 Phase A: Real-Time Chunking の per-variant switch。未指定なら
    # RtcConfig() = enabled False で既存 variant は従来どおり動く。
    # overlap_steps と chunk_len の関係は ckpt を読むまで確定しないため、ここでは
    # 値域と policy_type 適合だけを検証し、chunk_len との比較は policy 側に委ねる。
    if "rtc" in entry:
        ctor_kwargs["rtc"] = _parse_rtc(entry["rtc"], variant_name, policy_type)
    # Issue #139: overlay 画像を通す jpg の色差 (学習 cache の保存形式に合わせる)。
    # 未指定は PolicyConfig の既定 "4:4:4" (統合 cache)。それより前の焼き込みで学習した
    # ckpt は "4:2:0" を明示する。
    if "overlay_jpeg_subsampling" in entry:
        subsampling = entry["overlay_jpeg_subsampling"]
        if not isinstance(subsampling, str) or subsampling not in OVERLAY_JPEG_SUBSAMPLINGS:
            # YAML 1.1 はクォート無しの 4:2:0 を 60 進数の int (14520) として読む
            hint = " (quote it, e.g. \"4:2:0\")" if isinstance(subsampling, int) else ""
            raise ValueError(
                f"variant {variant_name!r}: overlay_jpeg_subsampling must be one of "
                f"{tuple(OVERLAY_JPEG_SUBSAMPLINGS)}, got {subsampling!r}{hint}"
            )
        if mode != "overlay":
            raise ValueError(
                f"variant {variant_name!r}: overlay_jpeg_subsampling is only used in "
                f"mode='overlay', got mode={mode!r}"
            )
        ctor_kwargs["overlay_jpeg_subsampling"] = subsampling
    cfg = PolicyConfig(**ctor_kwargs)

    # RAMEN-Ori の relative action は各 row が独立した関節 target ではなく、
    # 現姿勢から積算する normalized delta-q である。GR00T の RTC soft ramp を
    # この座標へ掛けると、ramp 区間の誤差が cumsum で後続 row 全体へ蓄積する。
    # Issue #137 B-3 実機試験では 2.4 s 後から関節範囲外 target が連続し、
    # 827/899 tick が safety HOLD になった。単なる parameter tuning では安全性を
    # 保証できないため、relative RAMEN-Ori では RTC を fail-fast で禁止する。
    # async replanning と temporal ensemble は独立機能なので引き続き使用できる。
    if (
        policy_type == "ramen_ori"
        and cfg.action_space == "rel"
        and cfg.rtc.enabled
        and not cfg.rtc.allow_experimental_relative_action
    ):
        raise ValueError(
            f"variant {variant_name!r}: rtc.enabled is unsafe for RAMEN-Ori "
            "action_space='rel' because RTC soft-prefix errors accumulate through "
            "delta-q reconstruction. Keep async replanning/temporal ensemble enabled "
            "and set rtc.enabled=false."
        )

    # shared YOLO ckpt ref (mode!=none の Policy が使う想定、mode=none でも参照可能)
    yolo_section = data.get("yolo", {})
    yolo_ckpt_ref = yolo_section.get("ckpt_ref") if isinstance(yolo_section, dict) else None
    if yolo_ckpt_ref is not None:
        # ckpt の約束との照合に使うので policy にも渡す (Issue #141 P8-3)
        cfg = replace(cfg, yolo_ckpt_ref=str(yolo_ckpt_ref))

    return VariantEntry(
        name=variant_name,
        policy_type=policy_type,
        policy_config=cfg,
        yolo_ckpt_ref=yolo_ckpt_ref,
    )


@dataclass(frozen=True)
class YoloSettings:
    """`policy_config.yaml` の `yolo` section (両経路が共有する検出の設定)。

    Attributes:
        ckpt_ref: `repo@revision`。学習の overlay を焼いた重みと同じものを指す。
        conf: YOLO の confidence の下限。学習 cache を焼いたときと同じ値にする
            (ここを上げると、cache に入っていた枠が推論では出なくなる)。
        imgsz: 推論の入力解像度。学習 cache と違うと検出そのものが変わる。
    """

    ckpt_ref: str
    conf: float
    imgsz: int


def load_yolo_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> YoloSettings:
    """`policy_config.yaml` の `yolo` section を読む (Issue #141 INF-10)。"""
    section = _load_yaml(config_path).get("yolo")
    if not isinstance(section, dict):
        raise ValueError(f"{config_path}: 'yolo' section is required")
    unknown = sorted(set(section) - {"ckpt_ref", "conf", "imgsz"})
    if unknown:
        raise ValueError(f"{config_path}: yolo has unknown keys {unknown}")
    ckpt_ref = section.get("ckpt_ref")
    if not isinstance(ckpt_ref, str) or "@" not in ckpt_ref:
        raise ValueError(
            f"{config_path}: yolo.ckpt_ref must be 'repo@revision', got {ckpt_ref!r}"
        )
    try:
        conf = float(section["conf"])
        imgsz = int(section["imgsz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config_path}: yolo requires numeric conf / imgsz ({exc})"
        ) from exc
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"{config_path}: yolo.conf must be in [0, 1], got {conf}")
    if imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError(
            f"{config_path}: yolo.imgsz must be a positive multiple of 32, got {imgsz}"
        )
    return YoloSettings(ckpt_ref=ckpt_ref, conf=conf, imgsz=imgsz)


def resolve_policy_class(policy_type: str):
    """policy_type → Policy class (lazy import で heavy module ロード回避)。

    Args:
        policy_type: "groot" / "ramen_ori" / "groot_pick_legs" / "act_diffusion"。

    Returns:
        Policy class。呼出側は class.from_ckpt(cfg) で instantiate。
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
    if policy_type == "act_diffusion":
        from inference.desktop.lower_policy.policies.act_diffusion import (
            ActDiffusionPolicy,
        )

        return ActDiffusionPolicy
    raise ValueError(f"unknown policy_type: {policy_type!r}")
