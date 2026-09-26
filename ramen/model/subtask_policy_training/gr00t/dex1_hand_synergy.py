"""Bidirectional Dex1-1 to official GR00T G1 hand-synergy conversion."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

Side = Literal["left", "right"]
ValueKind = Literal["state", "action"]

DEX1_OPEN = 4.5
DEX1_CLOSED = 0.0
HAND_DIM = 7
ASSET_PATH = Path(__file__).with_name("assets") / "dex1_g1_synergy.json"


@lru_cache(maxsize=1)
def load_synergy_manifest() -> dict[str, Any]:
    manifest = json.loads(ASSET_PATH.read_text())
    if manifest.get("schema_version") != "dex1_g1_synergy_v1":
        raise ValueError(f"unsupported Dex1 synergy schema in {ASSET_PATH}")
    if manifest.get("joint_order") != [
        "index_0",
        "index_1",
        "middle_0",
        "middle_1",
        "thumb_0",
        "thumb_1",
        "thumb_2",
    ]:
        raise ValueError("Dex1 synergy joint order does not match the official G1 hand order")
    for side in ("left", "right"):
        for key in ("state_open", "state_closed", "action_open", "action_closed"):
            _require_vector(f"{side}.{key}", manifest[side][key], HAND_DIM)
    return manifest


def synergy_manifest_sha256() -> str:
    return hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest()


def dex1_closed_fraction(dex1: float) -> float:
    value = float(dex1)
    if not math.isfinite(value):
        raise ValueError("Dex1 coordinate must be finite")
    return min(1.0, max(0.0, (DEX1_OPEN - value) / (DEX1_OPEN - DEX1_CLOSED)))


def dex1_to_hand(dex1: float, *, side: Side, kind: ValueKind) -> list[float]:
    """Map one Dex1 coordinate to the official seven-joint G1 synergy."""
    manifest = load_synergy_manifest()
    closed_fraction = dex1_closed_fraction(dex1)
    opened = manifest[side][f"{kind}_open"]
    closed = manifest[side][f"{kind}_closed"]
    return [
        float(open_value) + closed_fraction * (float(closed_value) - float(open_value))
        for open_value, closed_value in zip(opened, closed, strict=True)
    ]


def hand_to_dex1(hand: Sequence[float], *, side: Side, kind: ValueKind) -> float:
    """Least-squares project a seven-joint prediction back to Dex1."""
    _require_vector("hand", hand, HAND_DIM)
    manifest = load_synergy_manifest()
    opened = [float(value) for value in manifest[side][f"{kind}_open"]]
    axis = [
        float(closed) - float(opened_value)
        for opened_value, closed in zip(opened, manifest[side][f"{kind}_closed"], strict=True)
    ]
    centered = [float(value) - opened_value for value, opened_value in zip(hand, opened, strict=True)]
    denominator = sum(value * value for value in axis)
    if denominator <= 1e-12:
        raise ValueError(f"{side} {kind} synergy axis has near-zero norm")
    closed_fraction = sum(value * direction for value, direction in zip(centered, axis, strict=True))
    closed_fraction = min(1.0, max(0.0, closed_fraction / denominator))
    return DEX1_OPEN - closed_fraction * (DEX1_OPEN - DEX1_CLOSED)


def projection_residual(hand: Sequence[float], *, side: Side, kind: ValueKind) -> float:
    projected = dex1_to_hand(hand_to_dex1(hand, side=side, kind=kind), side=side, kind=kind)
    return math.sqrt(
        sum((float(value) - expected) ** 2 for value, expected in zip(hand, projected, strict=True))
    )


def _require_vector(name: str, values: Sequence[float], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"expected {expected}-D {name}, got {len(values)}")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} contains NaN or Inf")
