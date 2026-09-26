"""ACT / Diffusion Policy の学習 smoke (合成バッチ、ローカル GPU)。

検証するのは次の 2 点だけで、データ経路 (materialize / frame cache) は対象外。

1. `resolve_training_config.py` が出す値が、そのまま LeRobot の policy config として通るか
   (DP の horizon / n_obs_steps / drop_n_last_frames を追加した Issue 対応の回帰確認)
2. 実 shape (640x480 x 3 cam、19-D state/action) で forward + backward が回るか、
   batch size ごとに VRAM をどれだけ使うか

学習データは使わず、正規化済み想定の合成テンソルを流す (LeRobot 0.6 系では正規化は
policy の外の processor pipeline が担うため、forward には正規化後の値を渡してよい)。

使用例:
    cd model/subtask_policy_training && pixi run smoke-vram
    cd model/subtask_policy_training && pixi run smoke-vram --batch-sizes 1,2,4 --steps 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

FEATURE_ROOT = Path(__file__).resolve().parents[1]
RESOLVER_PATH = FEATURE_ROOT / "scripts" / "resolve_training_config.py"
CONFIG_PATH = FEATURE_ROOT / "configs" / "subtask_training.json"


def load_resolver() -> Any:
    spec = importlib.util.spec_from_file_location("resolve_training_config", RESOLVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load resolver: {RESOLVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolved_values(policy_type: str, subtask: str) -> dict[str, str]:
    """resolve_training_config を実 env と同じ経路で叩いて値を得る。"""
    resolver = load_resolver()
    config = resolver.load_config(CONFIG_PATH)
    previous = {k: os.environ.get(k) for k in ("SUBTASK", "POLICY_TYPE")}
    os.environ["SUBTASK"] = subtask
    os.environ["POLICY_TYPE"] = policy_type
    try:
        return resolver.resolve(config)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_features(values: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.configs.types import FeatureType, PolicyFeature

    def convert(payload: str) -> dict[str, Any]:
        return {
            key: PolicyFeature(type=FeatureType[spec["type"]], shape=tuple(spec["shape"]))
            for key, spec in json.loads(payload).items()
        }

    return convert(values["POLICY_INPUT_FEATURES"]), convert(values["POLICY_OUTPUT_FEATURES"])


def build_policy(policy_type: str, values: dict[str, str], device: torch.device):
    input_features, output_features = build_features(values)
    if policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = ACTConfig(
            chunk_size=int(values["ACT_CHUNK_SIZE"]),
            n_action_steps=int(values["ACT_N_ACTION_STEPS"]),
            input_features=input_features,
            output_features=output_features,
            device=str(device),
        )
        return ACTPolicy(config).to(device), config
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        config = DiffusionConfig(
            horizon=int(values["DIFFUSION_HORIZON"]),
            n_action_steps=int(values["DIFFUSION_N_ACTION_STEPS"]),
            n_obs_steps=int(values["DIFFUSION_N_OBS_STEPS"]),
            num_inference_steps=int(values["DIFFUSION_NUM_INFERENCE_STEPS"]),
            drop_n_last_frames=int(values["DIFFUSION_DROP_N_LAST_FRAMES"]),
            input_features=input_features,
            output_features=output_features,
            device=str(device),
        )
        return DiffusionPolicy(config).to(device), config
    raise ValueError(f"unsupported policy type for smoke: {policy_type!r}")


def synthetic_batch(config: Any, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """正規化済み想定の合成 batch。ACT は単一 obs、DP は n_obs_steps 分を積む。"""
    n_obs = int(getattr(config, "n_obs_steps", 1))
    horizon = int(getattr(config, "horizon", getattr(config, "chunk_size", 1)))
    batch: dict[str, torch.Tensor] = {}
    for key, feature in config.input_features.items():
        shape = tuple(feature.shape)
        dims = (batch_size, n_obs, *shape) if n_obs > 1 else (batch_size, *shape)
        batch[key] = torch.randn(*dims, device=device)
    action_dim = next(iter(config.output_features.values())).shape[0]
    batch["action"] = torch.randn(batch_size, horizon, action_dim, device=device)
    batch["action_is_pad"] = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
    return batch


def run_policy(policy_type: str, subtask: str, batch_sizes: list[int], steps: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = resolved_values(policy_type, subtask)
    policy, config = build_policy(policy_type, values, device)
    n_params = sum(p.numel() for p in policy.parameters())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    print(f"\n=== {policy_type} ({subtask}) ===")
    print(f"params: {n_params / 1e6:.1f} M | device: {device}")
    print(f"config: {describe(policy_type, values)}")
    print(f"{'batch':>6s} {'peak VRAM':>11s} {'step time':>10s} {'loss':>9s}")

    for batch_size in batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        batch = synthetic_batch(config, batch_size, device)
        try:
            elapsed = 0.0
            loss_value = float("nan")
            for step in range(steps):
                start = time.perf_counter()
                loss, _ = policy.forward(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                # 1 step 目は allocator の warm-up を含むので計測から外す。
                if step:
                    elapsed += time.perf_counter() - start
                loss_value = float(loss.detach())
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
            per_step = elapsed / max(steps - 1, 1)
            print(f"{batch_size:6d} {peak:9.2f} GB {per_step * 1e3:8.0f} ms {loss_value:9.3f}")
        except torch.OutOfMemoryError:
            print(f"{batch_size:6d} {'OOM':>11s}")
            break
        finally:
            del batch
    del policy, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def describe(policy_type: str, values: dict[str, str]) -> str:
    keys = {
        "act": ("ACT_CHUNK_SIZE", "ACT_N_ACTION_STEPS"),
        "diffusion": (
            "DIFFUSION_HORIZON",
            "DIFFUSION_N_ACTION_STEPS",
            "DIFFUSION_N_OBS_STEPS",
            "DIFFUSION_NUM_INFERENCE_STEPS",
            "DIFFUSION_DROP_N_LAST_FRAMES",
        ),
    }[policy_type]
    return ", ".join(f"{k.split('_', 1)[1].lower()}={values[k]}" for k in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-types", default="act,diffusion")
    parser.add_argument("--subtask", default="rotate_table_base")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--steps", type=int, default=3, help="計測する step 数 (1 step 目は warm-up)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {name} ({total:.1f} GB)")
    else:
        print("GPU: 無し (CPU で実行、VRAM 計測は不可)")
    for policy_type in (x.strip() for x in args.policy_types.split(",") if x.strip()):
        run_policy(policy_type, args.subtask, batch_sizes, args.steps)


if __name__ == "__main__":
    main()
