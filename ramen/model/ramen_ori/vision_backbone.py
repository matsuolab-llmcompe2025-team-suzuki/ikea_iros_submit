"""Vision backbone loader / adapter (RAMEN-Ori、Issue #120 Alt-4)。

A-1 LingBot (default) と A-2 RADIO (NVIDIA) を統一 API で load、
`(backbone: nn.Module, embed_dim: int)` を返す。VisionEncoder は backbone-agnostic
(A-0 DI 済) なので、この loader を train.py から呼ぶだけで A axis swap 可能。

# Supported variants

- `"lingbot-base"` (A-1 default): LingBot ViT-B/16、~86M frozen
- `"lingbot-<v>"` (LingBot 系の任意 variant): "large" 等
- `"radio-b"` (A-2 alternative): NVIDIA RADIOv2.5-B、~86M
- `"radio-l"`: RADIOv2.5-L、~305M

# RADIO の入手経路

`torch.hub.load('NVlabs/RADIO', 'radio_model', ...)`。初回 clone + weight DL、以降 cache。
timm 等の transitive dep が必要になる場合は pixi.toml へ追加要 (初回 forward 時に発覚)。

# RADIO の出力を LingBot 互換 dict に整形

LingBot は `forward(x, is_training) -> {"x_norm_patchtokens": (B, N_patches, D)}` を返す。
`VisionEncoder` は `["x_norm_patchtokens"]` を取り出して 8x8 pool → projection の順。
RADIO は `(summary: (B, D), features: (B, N_patches, D))` を返すので、`features` を
`x_norm_patchtokens` にラベル付け直しするだけ。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def load_vision_backbone(
    variant: str,
    device: str = "cuda",
    dtype: str = "fp32",
) -> tuple[nn.Module, int]:
    """LingBot / RADIO を統一 API で load、(nn.Module, embed_dim) を返す。

    Args:
        variant: "lingbot-<v>" or "radio-<size>"
            - lingbot-base: A-1 default、LingBot ViT-B/16
            - radio-b: A-2 alternative、RADIOv2.5-B
        device: "cuda" | "cpu"
        dtype: "fp32" | "bf16" 等

    Returns:
        (backbone, embed_dim): backbone は `forward(x, is_training=False)` で
        `{"x_norm_patchtokens": (B, N, D)}` dict を返す (LingBot 互換 interface)

    Raises:
        ValueError: unknown variant prefix
    """
    if variant.startswith("lingbot-"):
        # A-1: lazy import で LingBot HF DL trigger
        from lingbot_vision import load_pretrained_backbone

        lb_variant = variant.split("-", 1)[1]  # e.g. "base"
        return load_pretrained_backbone(variant=lb_variant, device=device, dtype=dtype)
    elif variant.startswith("radio-"):
        radio_size = variant.split("-", 1)[1]  # e.g. "b" | "l"
        return _load_radio(radio_size, device=device, dtype=dtype)
    else:
        raise ValueError(
            f"unknown variant {variant!r} (expected 'lingbot-<v>' or 'radio-<size>')"
        )


def _load_radio(
    size: str,
    device: str = "cuda",
    dtype: str = "fp32",
) -> tuple[nn.Module, int]:
    """NVIDIA RADIO を torch.hub 経由で DL、RadioAdapter で LingBot format にラップ。

    Args:
        size: "b" | "l" | "h" (RADIO ViT-B / L / H)
        device: "cuda" | "cpu"
        dtype: "fp32" | "bf16"

    Returns:
        (adapter, embed_dim)
    """
    version_map = {
        "b": "radio_v2.5-b",
        "l": "radio_v2.5-l",
        "h": "radio_v2.5-h",
    }
    if size not in version_map:
        raise ValueError(f"radio size {size!r} not supported (expected 'b'|'l'|'h')")
    hub_version = version_map[size]

    torch_dtype = _resolve_dtype(dtype)

    # torch.hub.load: 初回 clone + weight DL、以降 cache
    raw_model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version=hub_version,
        progress=True,
        force_reload=False,
        # Evaluation is intentionally non-interactive.  Without this explicit
        # decision torch.hub prompts on first use and crashes with EOF before
        # any checkpoint can be validated.  NVlabs/RADIO is the canonical
        # upstream used by the training recipe.
        trust_repo=True,
    )
    # freeze で eval mode + no grad (LingBot と同じ扱い、VisionEncoder 側で freeze=true が
    # 適用されるので backbone 単体 freeze は最終 param 状態確定 用)
    raw_model = raw_model.to(device=device, dtype=torch_dtype)
    raw_model.eval()
    for p in raw_model.parameters():
        p.requires_grad = False

    # RADIO の InputConditioner が持つ `dtype` は plain Python 属性 (buffer/param で
    # ないため raw_model.to(dtype=) で変化しない)。forward で `y = y.to(self.dtype)`
    # と強制 cast されるので、backbone の param dtype (bf16 等) と合わせないと
    # ViTPatchLinear 入力で dtype mismatch → RuntimeError で forward 失敗。
    if hasattr(raw_model, "input_conditioner") and hasattr(raw_model.input_conditioner, "dtype"):
        raw_model.input_conditioner.dtype = torch_dtype

    adapter = RadioAdapter(radio_model=raw_model)
    embed_dim = _extract_radio_embed_dim(raw_model)
    return adapter, embed_dim


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"dtype {dtype!r} not supported (fp32|fp16|bf16)")
    return mapping[dtype]


def _extract_radio_embed_dim(radio_model: nn.Module) -> int:
    """RADIO model から embed_dim を取り出す (実 API に依存、fallback 有)。

    RADIO の公式 API では `model.model.embed_dim` or `model.embed_dim` に存在
    (backbone は internal ViT)。取れない場合は forward 1 回試して shape 推定。
    """
    # 優先 order で試行
    for attr_path in ("model.embed_dim", "embed_dim"):
        obj = radio_model
        ok = True
        for a in attr_path.split("."):
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and isinstance(obj, int):
            return obj
    raise RuntimeError(
        "cannot infer embed_dim from RADIO model — API changed? "
        "Try forward 1 sample to inspect output shape."
    )


def _extract_radio_patch_size(radio_model: nn.Module) -> int:
    """RADIO model から patch_size を取り出す (Issue #122 fix)。

    RADIOv2.5 (b/l/h いずれも ViT) は公式 API で ``model.model.patch_size``
    or ``model.patch_generator.patch_size`` に存在。VisionEncoder が
    ``backbone.patch_size`` を触るため (vision.py:79)、adapter でも同 attr
    を expose する必要あり。実 forward path なしでは推定できないので、
    attr 探索が全部 miss したら明示的に raise。
    """
    for attr_path in (
        "model.patch_size",
        "patch_size",
        "patch_generator.patch_size",
        "model.patch_generator.patch_size",
    ):
        obj = radio_model
        ok = True
        for a in attr_path.split("."):
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok and isinstance(obj, int):
            return obj
    raise RuntimeError(
        "cannot infer patch_size from RADIO model — API changed? "
        "Try setting RadioAdapter.patch_size manually (RADIOv2.5-B/L/H default = 16)."
    )


class RadioAdapter(nn.Module):
    """NVIDIA RADIO の output を LingBot 互換 dict に整形する thin wrapper。

    LingBot: `forward(x, is_training=False) -> {"x_norm_patchtokens": (B, N, D)}`
    RADIO:   `forward(x) -> (summary: (B, D), features: (B, N, D))`

    → features を x_norm_patchtokens に mapping。
    """

    def __init__(self, radio_model: nn.Module) -> None:
        super().__init__()
        self.radio = radio_model
        # LingBot の `.embed_dim` / `.patch_size` attribute を露出
        # (VisionEncoder が vision.py:79 で `backbone.patch_size` を触る、Issue #122)
        self.embed_dim = _extract_radio_embed_dim(radio_model)
        self.patch_size = _extract_radio_patch_size(radio_model)

    def forward(self, x: torch.Tensor, is_training: bool = False) -> dict:
        """
        Args:
            x: (B, 3, H, W) RGB tensor
            is_training: LingBot API 互換、RADIO は無視

        Returns:
            {"x_norm_patchtokens": (B, N_patches, D)}
        """
        # RADIO の raw forward。返り値の unpack は radio_model の実装依存
        # 公式 API: `RadioOutput(summary, features)` NamedTuple、or (summary, features) tuple
        out = self.radio(x)
        # tuple or NamedTuple、features は index 1
        if hasattr(out, "features"):
            features = out.features
        else:
            _, features = out
        return {"x_norm_patchtokens": features}
