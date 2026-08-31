"""Inference-only compatibility for Team RAMEN Furniture-GR00T artifacts.

The training plugin is deliberately not a runtime dependency.  This module
registers only the serialized config/processor names and recreates the
auxiliary modules that are present in the checkpoint.  Action generation stays
on LeRobot's official :class:`GrootPolicy` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.processor import (
    EnvTransition,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


PINNED_BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
PHASE_COUNT = 7


@PreTrainedConfig.register_subclass("furniture_groot")
@dataclass
class FurnitureGrootRuntimeConfig(GrootConfig):
    """Exact inference-relevant schema of the training-side config."""

    base_model_revision: str = PINNED_BASE_MODEL_REVISION
    auxiliary_contract_version: int = 1
    phase_control_enabled: bool = False
    phase_macro_f1: float | None = None
    recovery_demonstration_count: int = 0
    recovery_prompt_enabled: bool = False
    progress_enabled: bool = True
    progress_loss_weight: float = 0.05
    progress_monotonicity_weight: float = 0.01
    phase_loss_weight: float = 0.05
    success_loss_weight: float = 0.02
    stall_loss_weight: float = 0.02
    insertion_classifier_enabled: bool = False
    insertion_classifier_loss_weight: float = 0.10
    progress_hidden_dim: int = 512
    consistent_gpu_augmentation: bool = True
    valid_action_dim: int = 46
    expert_role: str = "general"
    left_hand_action_loss_weight: float = 1.0
    right_hand_action_loss_weight: float = 1.0
    left_arm_action_loss_weight: float = 1.0
    right_arm_action_loss_weight: float = 1.0
    detach_auxiliary_features: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.base_model_revision != PINNED_BASE_MODEL_REVISION:
            raise ValueError("Furniture-GR00T base revision does not match N1.7 pin")
        if self.chunk_size != 40:
            raise ValueError("Furniture-GR00T requires H40")
        if self.max_state_dim != 132 or self.max_action_dim != 132:
            raise ValueError("Furniture-GR00T requires packed state/action 132D")
        if self.valid_action_dim != 46:
            raise ValueError("Furniture-GR00T action mask must cover slots 0:46")
        if self.auxiliary_contract_version not in {1, 2, 3}:
            raise ValueError("unsupported Furniture-GR00T auxiliary contract")
        if self.expert_role not in {"general", "insertion", "finish"}:
            raise ValueError("unsupported Furniture-GR00T expert role")
        if self.insertion_classifier_enabled and self.auxiliary_contract_version != 3:
            raise ValueError("insertion classifier requires auxiliary contract v3")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [-20, 0]


@ProcessorStepRegistry.register(name="furniture_groot_temporal_progress_v1")
@dataclass
class FurnitureGrootTemporalRuntimeStep(ProcessorStep):
    """Preserve the checkpoint's two-frame RGB axis during inference."""

    progress_horizon: int = 40

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        state = observation.get(OBS_STATE)
        if isinstance(state, torch.Tensor) and state.ndim == 3:
            observation[OBS_STATE] = state[:, -1, :]

        for key, value in observation.items():
            if not key.startswith(f"{OBS_IMAGES}.") or not isinstance(value, torch.Tensor):
                continue
            if key.endswith("_is_pad"):
                if value.ndim == 1:
                    if value.shape[0] != 2:
                        raise ValueError(f"{key} must preserve two frames")
                    observation[key] = value.unsqueeze(0)
                elif value.ndim != 2 or value.shape[1] != 2:
                    raise ValueError(f"{key} must be [2] or [B,2]")
                continue
            if value.ndim == 4:
                if value.shape[0] != 2:
                    raise ValueError(f"{key} must be [2,C,H,W]")
                observation[key] = value.unsqueeze(0)
            elif value.ndim != 5 or value.shape[1] != 2:
                raise ValueError(f"{key} must be [2,C,H,W] or [B,2,C,H,W]")

        # Training-only supervision keys are absent in physical inference.  If
        # a caller supplies them accidentally, fail closed instead of silently
        # changing the action input contract.
        unexpected = [
            key for key in observation
            if key.startswith("observation.") and key.endswith(("_horizon", "_mask"))
        ]
        if unexpected:
            raise ValueError(f"training-only auxiliary observations in inference: {unexpected}")
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(self, features: dict[str, Any]) -> dict[str, Any]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"progress_horizon": self.progress_horizon}


@ProcessorStepRegistry.register(name="furniture_groot_consistent_gpu_augmentation_v1")
@dataclass
class FurnitureGrootInferenceAugmentationStep(ProcessorStep):
    """Deserialize the training augmentation step as a strict inference no-op."""

    enabled: bool = True
    training: bool = False
    device: str = "cuda"
    max_num_transforms: int = 3
    affine_degrees: float = 5.0
    affine_translate: float = 0.05
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.5, 1.5)
    hue_range: tuple[float, float] = (-0.05, 0.05)
    sharpness_range: tuple[float, float] = (0.5, 1.5)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.training:
            raise RuntimeError("training augmentation is forbidden in physical inference")
        return transition

    def transform_features(self, features: dict[str, Any]) -> dict[str, Any]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "training": False,
            "device": self.device,
            "max_num_transforms": self.max_num_transforms,
            "affine_degrees": self.affine_degrees,
            "affine_translate": self.affine_translate,
            "brightness_range": list(self.brightness_range),
            "contrast_range": list(self.contrast_range),
            "saturation_range": list(self.saturation_range),
            "hue_range": list(self.hue_range),
            "sharpness_range": list(self.sharpness_range),
        }


class FurnitureGrootRuntimePolicy(GrootPolicy):
    """Official action model plus checkpoint-compatible diagnostic heads."""

    name = "furniture_groot"
    config_class = FurnitureGrootRuntimeConfig

    def __init__(self, config: FurnitureGrootRuntimeConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        model_config = self._groot_model.config
        input_dim = int(model_config.backbone_embedding_dim) + int(
            model_config.input_embedding_dim
        )

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, config.progress_hidden_dim),
                nn.GELU(),
                nn.Linear(config.progress_hidden_dim, output_dim),
            )

        self.progress_head = head(config.chunk_size)
        if config.auxiliary_contract_version >= 2:
            self.phase_head = head(config.chunk_size * PHASE_COUNT)
            self.success_head = head(config.chunk_size)
            self.stall_head = head(config.chunk_size)
        else:
            self.phase_head = None
            self.success_head = None
            self.stall_head = None
        if config.auxiliary_contract_version >= 3:
            self.left_support_head = head(config.chunk_size)
            self.right_insert_head = head(config.chunk_size)
        else:
            self.left_support_head = None
            self.right_insert_head = None
