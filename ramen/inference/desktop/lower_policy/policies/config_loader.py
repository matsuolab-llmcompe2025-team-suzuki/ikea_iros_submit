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

alternatives_by_skill:
  <skill_name>: [<variant_name>, ...]   # R のときに選べる候補 (Issue #188)
  pick_table_leg: [<variant_name>, {variant: <variant_name>, pick_leg_hybrid: false}, ...]

model_loading:                          # --gpu-models plan の読み方 (Issue #188)
  preload_through_skill: <skill_name>
  load_while_policy_runs: [<kind>, ...]

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

import re
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
    *,
    variant_set: str | None = None,
) -> dict[str, str]:
    """`default_variant_by_skill` section を読む (Issue #148)。

    本番 run で「どの skill をどの ckpt で走らせるか」の正本。呼出側 (entrypoint)
    は CLI が渡さなかった slot だけをこれで埋める。

    `variant_set` を渡すと、`variant_sets.<名前>` に書いた skill だけを差し替える
    (Issue #159)。会場で model の組み合わせを 1 つの名前でまとめて切り替えるため
    (会場・自前経路とも `--policy-variant-set`)。skill ごとの
    上書きはこの後に呼出側で掛かるので、そちらが優先する。未知の名前・skill・
    variant は起動時に落とす (会場で「黙って既定のまま」を避ける)。

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
    if variant_set is None:
        return defaults
    sets = data.get("variant_sets")
    if not isinstance(sets, dict) or variant_set not in sets:
        available = sorted(sets) if isinstance(sets, dict) else []
        raise ValueError(
            f"{config_path}: unknown variant_set {variant_set!r} "
            f"(available: {available})"
        )
    overlay = sets[variant_set]
    if not isinstance(overlay, dict) or not overlay:
        raise ValueError(
            f"{config_path}: variant_sets.{variant_set} must map skills to variants"
        )
    for skill_name, variant in overlay.items():
        if skill_name not in defaults:
            raise ValueError(
                f"{config_path}: variant_sets.{variant_set}.{skill_name} is not a "
                f"skill in default_variant_by_skill ({sorted(defaults)})"
            )
        if not isinstance(variant, str) or variant not in policies:
            raise ValueError(
                f"{config_path}: variant_sets.{variant_set}.{skill_name}={variant!r} "
                "is not registered under policies"
            )
        defaults[str(skill_name)] = variant
    return defaults


@dataclass(frozen=True)
class PolicyAlternative:
    """R のやり直しで既定の代わりに選べる候補 1 つ (Issue #188)。

    Attributes:
        variant: `policies:` の key。
        pick_leg_hybrid: pick だけ。True = hybrid (VLM → この policy で掴む → IK で運ぶ・持ち替え)、
            False = hybrid を使わず、学習 policy に pick 全部を任せる。None = 既定と同じやり方。
    """

    variant: str
    pick_leg_hybrid: bool | None = None


#: 候補を mapping で書くときの key (pick だけ、Issue #188)。
_ALTERNATIVE_KEYS = frozenset({"variant", "pick_leg_hybrid"})


def load_alternatives_by_skill(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, tuple[PolicyAlternative, ...]]:
    """`alternatives_by_skill` section を読む (Issue #188)。

    skill ごとに、R のやり直しで既定の代わりに選べる候補を並べたもの。候補は variant 名か、
    pick だけ `{variant: <名前>, pick_leg_hybrid: true / false}` (hybrid で使う / 使わず学習 policy に全部任せる)。
    section が無ければ空 (= 候補なし、今までと同じ動き)。既定と同じ候補を除くのは
    呼出側 (既定は CLI や `--policy-variant-set` で変わるので、ここでは決まらない)。
    誤った書き方は起動時に落とす (会場で「黙って候補なし」を避ける)。
    """
    data = _load_yaml(config_path)
    section = data.get("alternatives_by_skill")
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(
            f"{config_path}: 'alternatives_by_skill' must map skills to variant lists"
        )
    skills = data.get("default_variant_by_skill")
    skills = skills if isinstance(skills, dict) else {}
    policies = data.get("policies")
    policies = policies if isinstance(policies, dict) else {}
    alternatives: dict[str, tuple[PolicyAlternative, ...]] = {}
    for skill_name, entries in section.items():
        if skill_name not in skills:
            raise ValueError(
                f"{config_path}: alternatives_by_skill.{skill_name} is not a skill "
                f"in default_variant_by_skill ({sorted(skills)})"
            )
        if not isinstance(entries, list):
            raise ValueError(
                f"{config_path}: alternatives_by_skill.{skill_name} must be a list "
                f"of variant names, got {entries!r}"
            )
        parsed = []
        for entry in entries:
            if isinstance(entry, dict):
                if skill_name != "pick_table_leg" or set(entry) != _ALTERNATIVE_KEYS:
                    raise ValueError(
                        f"{config_path}: alternatives_by_skill.{skill_name} has {entry!r}; "
                        "a mapping is only for pick_table_leg, as "
                        "{variant: <name>, pick_leg_hybrid: true / false}"
                    )
                if not isinstance(entry["pick_leg_hybrid"], bool):
                    raise ValueError(
                        f"{config_path}: alternatives_by_skill.pick_table_leg has {entry!r}; "
                        "pick_leg_hybrid must be true or false"
                    )
                variant, hybrid = entry["variant"], entry["pick_leg_hybrid"]
            else:
                variant, hybrid = entry, None
            if not isinstance(variant, str) or variant not in policies:
                raise ValueError(
                    f"{config_path}: alternatives_by_skill.{skill_name} has "
                    f"{variant!r}, which is not registered under policies"
                )
            parsed.append(PolicyAlternative(variant, hybrid))
        if len(set(parsed)) != len(parsed):
            raise ValueError(
                f"{config_path}: alternatives_by_skill.{skill_name} lists a "
                f"variant twice: {entries}"
            )
        alternatives[str(skill_name)] = tuple(parsed)
    return alternatives




def load_pick_leg_hybrid_default(config_path: str | Path = DEFAULT_CONFIG_PATH) -> bool:
    """`pick_leg_hybrid` を読む: Stage 1〜4 の pick を hybrid で動かすか (Issue #188)。

    `--pick-leg-hybrid` / `--no-pick-leg-hybrid` を付けなかった run の既定。key が無ければ True
    (Issue #148 からの既定、hybrid)。bool 以外は起動時に落とす。
    """
    data = _load_yaml(config_path)
    value = data.get("pick_leg_hybrid", True)
    if not isinstance(value, bool):
        raise ValueError(f"{config_path}: pick_leg_hybrid must be true or false, got {value!r}")
    return value

@dataclass(frozen=True)
class ModelLoading:
    """`model_loading` section (Issue #188、`--gpu-models plan` の読み方)。

    Attributes:
        preload_through_skill: 起動の後に先に読んでおく最後の skill。
        load_while_policy_runs: policy が動いている間にも読んでよい種類
            (`load_plan.LOAD_KINDS`)。空 = 全部腕が止まっている間だけ。
    """

    preload_through_skill: str
    load_while_policy_runs: tuple[str, ...]


def load_model_loading(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ModelLoading:
    """`model_loading` section を読む (Issue #188)。

    `--gpu-models plan` のときだけ呼ぶ。section が無い・key が違う・知らない skill や
    種類は起動時に落とす (会場で「黙って別の読み方」を避ける)。
    """
    from inference.desktop.lower_policy.policies.load_plan import LOAD_KINDS

    data = _load_yaml(config_path)
    section = data.get("model_loading")
    if not isinstance(section, dict):
        raise ValueError(
            f"{config_path}: 'model_loading' section is required for --gpu-models plan"
        )
    expected = {"preload_through_skill", "load_while_policy_runs"}
    if set(section) != expected:
        raise ValueError(
            f"{config_path}: model_loading must have exactly {sorted(expected)}, "
            f"got {sorted(section)}"
        )
    skills = data.get("default_variant_by_skill")
    skills = skills if isinstance(skills, dict) else {}
    through = section["preload_through_skill"]
    if through not in skills:
        raise ValueError(
            f"{config_path}: model_loading.preload_through_skill={through!r} is not a "
            f"skill in default_variant_by_skill ({sorted(skills)})"
        )
    kinds = section["load_while_policy_runs"]
    if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
        raise ValueError(
            f"{config_path}: model_loading.load_while_policy_runs must be a list of "
            f"kinds, got {kinds!r}"
        )
    unknown = sorted(set(kinds) - LOAD_KINDS)
    if unknown:
        raise ValueError(
            f"{config_path}: model_loading.load_while_policy_runs has unknown kinds "
            f"{unknown} (valid: {sorted(LOAD_KINDS)})"
        )
    if len(set(kinds)) != len(kinds):
        raise ValueError(
            f"{config_path}: model_loading.load_while_policy_runs lists a kind twice: "
            f"{kinds}"
        )
    return ModelLoading(str(through), tuple(kinds))

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
    if policy_type not in ("groot", "ramen_ori", "groot_pick_legs", "groot_pick_legs_ee_rel", "act_diffusion"):
        raise ValueError(
            f"variant {variant_name!r}: policy_type must be one of "
            f"'groot' / 'ramen_ori' / 'groot_pick_legs' / "
            f"'groot_pick_legs_ee_rel' / 'act_diffusion', "
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
    elif policy_type in ("groot_pick_legs", "groot_pick_legs_ee_rel"):
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

    # Issue #155: ckpt_filename を読むのは RAMEN-Ori だけ。他の policy に書くと黙って無視されるので止める
    if entry.get("ckpt_filename") is not None and policy_type != "ramen_ori":
        raise ValueError(
            f"variant {variant_name!r}: ckpt_filename is only read by policy_type=ramen_ori "
            f"(got {policy_type!r}); use checkpoint_subdir for GR00T / ACT / DP"
        )

    # Phase 2/4 optional fields (指定無しは PolicyConfig の default 継承)
    ctor_kwargs: dict = dict(
        mode=mode,
        ckpt_ref=str(ckpt_ref),
        checkpoint_subdir=(
            str(entry["checkpoint_subdir"])
            if entry.get("checkpoint_subdir") is not None
            else None
        ),
        ckpt_filename=(
            str(entry["ckpt_filename"])
            if entry.get("ckpt_filename") is not None
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
        ckpt_ref: `repo@commit`。学習の overlay を焼いた重みと同じものを指す。revision は
            commit hash で固定する (cache にあれば通信せずに読むため)。
        ckpt_file: repo の中の重みの path (決め打ち。HF cache の path がこれで決まる)。
        conf: YOLO の confidence の下限。学習 cache を焼いたときと同じ値にする
            (ここを上げると、cache に入っていた枠が推論では出なくなる)。
        imgsz: 推論の入力解像度。学習 cache と違うと検出そのものが変わる。
    """

    ckpt_ref: str
    ckpt_file: str
    conf: float
    imgsz: int


#: huggingface_hub が「cache にあれば通信しない」近道を使える revision の形。
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


def load_yolo_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> YoloSettings:
    """`policy_config.yaml` の `yolo` section を読む (Issue #141 INF-10)。"""
    section = _load_yaml(config_path).get("yolo")
    if not isinstance(section, dict):
        raise ValueError(f"{config_path}: 'yolo' section is required")
    unknown = sorted(set(section) - {"ckpt_ref", "ckpt_file", "conf", "imgsz"})
    if unknown:
        raise ValueError(f"{config_path}: yolo has unknown keys {unknown}")
    ckpt_ref = section.get("ckpt_ref")
    if not isinstance(ckpt_ref, str) or "@" not in ckpt_ref:
        raise ValueError(
            f"{config_path}: yolo.ckpt_ref must be 'repo@revision', got {ckpt_ref!r}"
        )
    revision = ckpt_ref.partition("@")[2]
    if not _COMMIT_HASH.fullmatch(revision):
        raise ValueError(
            f"{config_path}: yolo.ckpt_ref must pin a 40-hex commit hash so a cached "
            f"weight loads without the network, got revision {revision!r}"
        )
    ckpt_file = section.get("ckpt_file")
    if not isinstance(ckpt_file, str) or not ckpt_file.endswith(".pt"):
        raise ValueError(
            f"{config_path}: yolo.ckpt_file must be the .pt path inside the repo, "
            f"got {ckpt_file!r}"
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
    return YoloSettings(ckpt_ref=ckpt_ref, ckpt_file=ckpt_file, conf=conf, imgsz=imgsz)


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
    if policy_type == "groot_pick_legs_ee_rel":
        from inference.desktop.lower_policy.policies.groot_pick_legs_ee_rel import (
            Gr00tPolicyPickLegsEERel,
        )

        return Gr00tPolicyPickLegsEERel
    if policy_type == "ramen_ori":
        from inference.desktop.lower_policy.policies.ramen_ori import RamenOriPolicy

        return RamenOriPolicy
    if policy_type == "act_diffusion":
        from inference.desktop.lower_policy.policies.act_diffusion import (
            ActDiffusionPolicy,
        )

        return ActDiffusionPolicy
    raise ValueError(f"unknown policy_type: {policy_type!r}")
