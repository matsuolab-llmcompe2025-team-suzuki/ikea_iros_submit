#!/usr/bin/env python3
"""Train the three-camera 19-D flow-matching behavior-cloning policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from model.subtask_policy_training.flow_matching import FlowMatchingConfig, FlowMatchingPolicy
from model.subtask_policy_training.flow_matching.data import (
    FlowMatchingDataset,
    load_dataset_stats,
    load_episode_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--validation-freq", type=int, default=5_000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--validation-action-samples", type=int, default=64)
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--n-action-steps", type=int, default=6)
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--flow-inference-steps", type=int, default=10)
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--no-image-augmentation", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="iros2026-ramen-flip-table")
    parser.add_argument("--wandb-run-name", default="flow-matching-bc")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "steps",
        "batch_size",
        "validation_freq",
        "validation_samples",
        "validation_action_samples",
        "save_freq",
        "log_freq",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.warmup_steps < 0:
        raise ValueError("workers and warmup steps must be non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema decay must be in (0, 1)")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        model_values = dict(model.named_parameters())
        for name, value in ema.named_parameters():
            value.lerp_(model_values[name].detach(), 1.0 - decay)
        model_buffers = dict(model.named_buffers())
        for name, value in ema.named_buffers():
            value.copy_(model_buffers[name])


def learning_rate_scale(step: int, *, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@torch.no_grad()
def validate(
    model: FlowMatchingPolicy,
    loader: DataLoader,
    *,
    device: torch.device,
    max_samples: int,
    max_action_samples: int,
) -> dict[str, float]:
    model.eval()
    losses: list[tuple[float, int]] = []
    samples = 0
    action_samples = 0
    raw_error_sum = 0.0
    normalized_error_sum = 0.0
    body_error_sum = 0.0
    hand_error_sum = 0.0
    valid_steps = 0
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        torch.manual_seed(0)
        for batch in loader:
            if samples >= max_samples:
                break
            remaining = max_samples - samples
            if batch["state"].shape[0] > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}
            data = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = model.flow_loss(
                    data["images"], data["state"], data["action"], data["action_is_pad"]
                )
            batch_size = int(data["state"].shape[0])
            losses.append((float(loss), batch_size))
            samples += batch_size

            if action_samples < max_action_samples:
                action_count = min(batch_size, max_action_samples - action_samples)
                images = data["images"][:action_count]
                state = data["state"][:action_count]
                target = data["action"][:action_count]
                padding = data["action_is_pad"][:action_count]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    predicted = model.sample_actions(images, state).float()
                valid = (~padding).unsqueeze(-1)
                raw_error = (predicted - target).abs()
                normalized_error = (
                    model.normalize_action(predicted) - model.normalize_action(target)
                ).abs()
                step_count = int(valid.sum())
                raw_error_sum += float((raw_error * valid).sum())
                normalized_error_sum += float((normalized_error * valid).sum())
                body_error_sum += float((raw_error[..., :17] * valid).sum())
                hand_error_sum += float((raw_error[..., 17:] * valid).sum())
                valid_steps += step_count
                action_samples += action_count
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    if valid_steps < 1:
        raise RuntimeError("validation action samples contain no valid steps")
    return {
        "flow_loss": sum(value * count for value, count in losses) / sum(count for _, count in losses),
        "action_mae": raw_error_sum / (valid_steps * model.config.action_dim),
        "action_normalized_mae": normalized_error_sum
        / (valid_steps * model.config.action_dim),
        "body_action_mae_rad": body_error_sum / (valid_steps * 17),
        "dex1_action_mae": hand_error_sum / (valid_steps * 2),
        "action_samples": float(action_samples),
    }


def save_training_state(
    directory: Path,
    *,
    ema_model: FlowMatchingPolicy,
    model: FlowMatchingPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_validation_metric: float,
    args: argparse.Namespace,
) -> None:
    temporary = directory.with_name(f".{directory.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    ema_model.save_pretrained(temporary)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_validation_metric": best_validation_metric,
            "args": vars(args),
        },
        temporary / "training_state.pt",
    )
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = load_episode_split(args.split_file)
    state_stats, action_stats = load_dataset_stats(args.dataset_root)
    config = FlowMatchingConfig(
        action_horizon=args.action_horizon,
        n_action_steps=args.n_action_steps,
        model_dim=args.model_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        flow_inference_steps=args.flow_inference_steps,
        train_image_encoder=not args.freeze_image_encoder,
    )
    train_dataset = FlowMatchingDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=split["train"],
        config=config,
        augment=not args.no_image_augmentation,
        revision=args.revision,
    )
    validation_dataset = FlowMatchingDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=split["validation"],
        config=config,
        augment=False,
        revision=args.revision,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = FlowMatchingPolicy(
        config,
        state_stats=state_stats,
        action_stats=action_stats,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_scale(step, warmup=args.warmup_steps, total=args.steps),
    )
    start_step = 0
    best_validation_metric = float("inf")
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        best_validation_metric = float(
            payload.get("best_validation_metric", payload.get("best_validation_loss", float("inf")))
        )
        ema_checkpoint = args.resume.parent
        ema_model = FlowMatchingPolicy.from_pretrained(ema_checkpoint, device=device)
        ema_model.requires_grad_(False)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), **config.to_dict()},
        )

    manifest = {
        "config": config.to_dict(),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_repo_id": args.repo_id,
        "split_file": str(args.split_file.resolve()),
        "split_counts": {name: len(indices) for name, indices in split.items()},
        "train_frames": len(train_dataset),
        "validation_frames": len(validation_dataset),
        "effective_train_epochs": args.steps * args.batch_size / len(train_dataset),
        "device": str(device),
        "policy_inputs": [
            "head-left RGB 640x480",
            "left D405 RGB 640x480",
            "right D405 RGB 640x480",
            "19D upper-body joint state",
        ],
        "policy_output": "19D upper-body absolute joint targets",
        "sim_privileged_policy_inputs": [],
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = args.output_dir / "metrics.jsonl"

    batches = infinite_batches(train_loader)
    started = time.monotonic()
    model.train()
    for step in range(start_step, args.steps):
        data = move_batch(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss = model.flow_loss(
                data["images"], data["state"], data["action"], data["action_is_pad"]
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step + 1}: {loss}")
        loss.backward()
        gradient_norm = clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        completed = step + 1

        if completed % args.log_freq == 0:
            elapsed = time.monotonic() - started
            values = {
                "step": completed,
                "train/loss": float(loss.detach()),
                "train/gradient_norm": float(gradient_norm.detach()),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/steps_per_second": (completed - start_step) / max(elapsed, 1.0e-6),
            }
            if device.type == "cuda":
                values["train/peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1.0e9
            print(json.dumps(values), flush=True)
            append_jsonl(metrics_path, values)
            if wandb_run is not None:
                wandb_run.log(values, step=completed)

        if completed % args.validation_freq == 0 or completed == args.steps:
            validation_metrics = validate(
                ema_model,
                validation_loader,
                device=device,
                max_samples=args.validation_samples,
                max_action_samples=args.validation_action_samples,
            )
            validation_values = {
                "step": completed,
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            print(json.dumps(validation_values), flush=True)
            append_jsonl(metrics_path, validation_values)
            if wandb_run is not None:
                wandb_run.log(validation_values, step=completed)
            selection_metric = validation_metrics["action_normalized_mae"]
            if selection_metric < best_validation_metric:
                best_validation_metric = selection_metric
                save_training_state(
                    args.output_dir / "best",
                    ema_model=ema_model,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=completed,
                    best_validation_metric=best_validation_metric,
                    args=args,
                )
            model.train()

        if completed % args.save_freq == 0 or completed == args.steps:
            save_training_state(
                args.output_dir / f"checkpoint_{completed:08d}",
                ema_model=ema_model,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed,
                best_validation_metric=best_validation_metric,
                args=args,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
