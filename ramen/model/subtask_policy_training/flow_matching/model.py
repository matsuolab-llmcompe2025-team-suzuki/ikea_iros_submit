"""Conditional flow-matching network for short upper-body action chunks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .config import FlowMatchingConfig, POLICY_CAMERAS


def _normalization_tensor(stats: dict[str, Any], field: str, size: int) -> torch.Tensor:
    value = torch.as_tensor(stats[field], dtype=torch.float32)
    if value.shape != (size,) or not torch.isfinite(value).all():
        raise ValueError(f"normalization {field} must contain {size} finite values")
    return value


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), dimension // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(2 * frequencies.numel(), dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time[:, None] * self.frequencies[None, :] * (2.0 * math.pi)
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))


class FlowMatchingPolicy(nn.Module):
    """Generate a smooth 19-D target chunk from three RGB views and joint state."""

    def __init__(
        self,
        config: FlowMatchingConfig,
        *,
        state_stats: dict[str, Any],
        action_stats: dict[str, Any],
        load_pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        weights = None
        if config.image_encoder_weights == "imagenet" and load_pretrained_encoder:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        encoder = models.resnet18(weights=weights)
        self.image_encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        self.image_encoder.requires_grad_(config.train_image_encoder)
        self.visual_pool = nn.AdaptiveAvgPool2d(
            (config.spatial_grid_height, config.spatial_grid_width)
        )
        self.visual_projection = nn.Conv2d(512, config.model_dim, kernel_size=1)
        spatial_tokens = config.spatial_grid_height * config.spatial_grid_width
        self.visual_position_embedding = nn.Parameter(
            torch.zeros(1, len(POLICY_CAMERAS), spatial_tokens, config.model_dim)
        )
        nn.init.normal_(self.visual_position_embedding, std=0.02)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.state_token_embedding = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        nn.init.normal_(self.state_token_embedding, std=0.02)
        observation_block = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.model_dim * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.observation_transformer = nn.TransformerEncoder(
            observation_block,
            num_layers=config.observation_transformer_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )
        self.action_projection = nn.Linear(config.action_dim, config.model_dim)
        self.time_embedding = FourierTimeEmbedding(config.model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.action_horizon, config.model_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        block = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.model_dim * config.feedforward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            block,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.action_dim),
        )
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

        state_mean = _normalization_tensor(state_stats, "mean", config.state_dim)
        state_std = _normalization_tensor(state_stats, "std", config.state_dim).clamp_min(1.0e-4)
        action_mean = _normalization_tensor(action_stats, "mean", config.action_dim)
        action_std = _normalization_tensor(action_stats, "std", config.action_dim).clamp_min(1.0e-4)
        action_min = _normalization_tensor(action_stats, "min", config.action_dim)
        action_max = _normalization_tensor(action_stats, "max", config.action_dim)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        generator = torch.Generator(device="cpu").manual_seed(config.deterministic_noise_seed)
        self.register_buffer(
            "deterministic_noise",
            torch.randn(
                1,
                config.action_horizon,
                config.action_dim,
                generator=generator,
            ),
        )

    def train(self, mode: bool = True) -> "FlowMatchingPolicy":
        super().train(mode)
        if not self.config.train_image_encoder:
            self.image_encoder.eval()
        return self

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

    def _encode_observation_tokens(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1:3] != (len(POLICY_CAMERAS), 3):
            raise ValueError(f"images must be [B,3,3,H,W], got {tuple(images.shape)}")
        if state.ndim != 2 or state.shape[1] != self.config.state_dim:
            raise ValueError(f"state must be [B,{self.config.state_dim}], got {tuple(state.shape)}")
        if images.shape[0] != state.shape[0]:
            raise ValueError("image and state batch sizes differ")
        if not torch.isfinite(images).all() or not torch.isfinite(state).all():
            raise ValueError("policy observations contain NaN or Inf")
        batch_size, camera_count = images.shape[:2]
        image_batch = images.reshape(-1, *images.shape[2:]).float()
        image_batch = F.interpolate(
            image_batch,
            size=(self.config.image_height, self.config.image_width),
            mode="bilinear",
            align_corners=False,
        )
        if float(image_batch.max().detach()) > 1.0:
            image_batch = image_batch / 255.0
        image_batch = (image_batch - self.image_mean) / self.image_std
        features = self.visual_projection(self.visual_pool(self.image_encoder(image_batch)))
        features = features.flatten(2).transpose(1, 2)
        features = features.reshape(batch_size, camera_count, features.shape[1], self.config.model_dim)
        features = features + self.visual_position_embedding
        state_token = self.state_projection(self.normalize_state(state)).unsqueeze(1)
        state_token = state_token + self.state_token_embedding
        memory = torch.cat((state_token, features.flatten(1, 2)), dim=1)
        return self.observation_transformer(memory)

    def encode_observation(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return a spatially conditioned summary for the residual actor/critic."""

        return self._encode_observation_tokens(images, state)[:, 0]

    def forward(
        self,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (context.shape[0], self.config.action_horizon, self.config.action_dim)
        if noisy_action.shape != expected:
            raise ValueError(f"noisy_action must be {expected}, got {tuple(noisy_action.shape)}")
        if time.shape != (context.shape[0],):
            raise ValueError(f"time must be [B], got {tuple(time.shape)}")
        if context.ndim == 2:
            context = context.unsqueeze(1)
        if context.ndim != 3 or context.shape[0] != noisy_action.shape[0]:
            raise ValueError("context must be [B,T,D] or [B,D]")
        if action_is_pad is not None and action_is_pad.shape != noisy_action.shape[:2]:
            raise ValueError("action padding mask must match [B,H]")
        tokens = self.action_projection(noisy_action)
        tokens = tokens + self.position_embedding
        tokens = tokens + self.time_embedding(time)[:, None, :]
        return self.velocity_head(
            self.transformer(tokens, context, tgt_key_padding_mask=action_is_pad)
        )

    def flow_loss(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (state.shape[0], self.config.action_horizon, self.config.action_dim)
        if target_action.shape != expected:
            raise ValueError(f"target_action must be {expected}, got {tuple(target_action.shape)}")
        target = self.normalize_action(target_action)
        source = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        noisy = (1.0 - time[:, None, None]) * source + time[:, None, None] * target
        memory = self._encode_observation_tokens(images, state)
        velocity = self.forward(noisy, time, memory, action_is_pad)
        error = (velocity - (target - source)).square().mean(dim=-1)
        if action_is_pad is None:
            return error.mean()
        if action_is_pad.shape != error.shape:
            raise ValueError(f"action_is_pad must be {tuple(error.shape)}, got {tuple(action_is_pad.shape)}")
        valid = (~action_is_pad).to(error.dtype)
        return (error * valid).sum() / valid.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        *,
        inference_steps: int | None = None,
        deterministic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        steps = inference_steps or self.config.flow_inference_steps
        if steps < 1:
            raise ValueError("inference_steps must be positive")
        use_deterministic = (
            self.config.deterministic_inference if deterministic is None else deterministic
        )
        shape = (images.shape[0], self.config.action_horizon, self.config.action_dim)
        if use_deterministic:
            action = self.deterministic_noise.to(device=images.device, dtype=state.dtype).expand(shape)
        else:
            action = torch.randn(shape, device=images.device, dtype=state.dtype, generator=generator)
        context = self._encode_observation_tokens(images, state)
        dt = 1.0 / float(steps)
        for index in range(steps):
            time = torch.full(
                (images.shape[0],), index / float(steps), device=images.device, dtype=state.dtype
            )
            first = self.forward(action, time, context)
            proposal = action + dt * first
            if index + 1 < steps:
                next_time = torch.full_like(time, (index + 1) / float(steps))
                second = self.forward(proposal, next_time, context)
                action = action + 0.5 * dt * (first + second)
            else:
                action = proposal
        result = self.denormalize_action(action)
        return torch.maximum(torch.minimum(result, self.action_max), self.action_min)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "policy_inputs": [*POLICY_CAMERAS, "observation.state"],
            "policy_output": "19D upper-body absolute joint target chunk",
            "privileged_inputs": [],
            "state_stats": {
                "mean": self.state_mean.tolist(),
                "std": self.state_std.tolist(),
            },
            "action_stats": {
                "mean": self.action_mean.tolist(),
                "std": self.action_std.tolist(),
                "min": self.action_min.tolist(),
                "max": self.action_max.tolist(),
            },
        }

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "flow_matching_policy.json").write_text(
            json.dumps(self.checkpoint_metadata(), indent=2) + "\n", encoding="utf-8"
        )
        state = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}
        save_file(state, str(output / "model.safetensors"))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FlowMatchingPolicy":
        root = Path(checkpoint)
        metadata = json.loads((root / "flow_matching_policy.json").read_text(encoding="utf-8"))
        config = FlowMatchingConfig.from_dict(metadata["config"])
        model = cls(
            config,
            state_stats=metadata["state_stats"],
            action_stats=metadata["action_stats"],
            load_pretrained_encoder=False,
        )
        model.load_state_dict(load_file(str(root / "model.safetensors"), device=str(device)))
        return model.to(device).eval()
