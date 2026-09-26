"""Configuration contract for the flip-table flow-matching policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "team_ramen_flow_matching_v2"
POLICY_CAMERAS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


@dataclass(frozen=True)
class FlowMatchingConfig:
    state_dim: int = 19
    action_dim: int = 19
    action_horizon: int = 24
    n_action_steps: int = 6
    fps: int = 30
    image_height: int = 224
    image_width: int = 288
    model_dim: int = 384
    transformer_layers: int = 6
    transformer_heads: int = 8
    observation_transformer_layers: int = 2
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    spatial_grid_height: int = 4
    spatial_grid_width: int = 5
    flow_inference_steps: int = 10
    image_encoder_weights: str = "imagenet"
    train_image_encoder: bool = True
    deterministic_inference: bool = True
    deterministic_noise_seed: int = 17

    def __post_init__(self) -> None:
        if self.state_dim != 19 or self.action_dim != 19:
            raise ValueError("the G1 upper-body policy contract requires 19-D state and action")
        if self.fps != 30:
            raise ValueError("the real demonstration contract requires 30 fps")
        if self.action_horizon < 2:
            raise ValueError("action_horizon must be at least 2")
        if not 1 <= self.n_action_steps <= self.action_horizon:
            raise ValueError("n_action_steps must be within action_horizon")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.model_dim % self.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")
        if (
            self.transformer_layers < 1
            or self.observation_transformer_layers < 1
            or self.feedforward_multiplier < 1
        ):
            raise ValueError("transformer depth and feedforward multiplier must be positive")
        if self.spatial_grid_height < 1 or self.spatial_grid_width < 1:
            raise ValueError("spatial feature grid dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.flow_inference_steps < 1:
            raise ValueError("flow_inference_steps must be positive")
        if self.image_encoder_weights not in {"imagenet", "none"}:
            raise ValueError("image_encoder_weights must be 'imagenet' or 'none'")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FlowMatchingConfig":
        payload = dict(value)
        schema = payload.pop("schema_version", None)
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported flow-matching schema: {schema!r}")
        return cls(**payload)
