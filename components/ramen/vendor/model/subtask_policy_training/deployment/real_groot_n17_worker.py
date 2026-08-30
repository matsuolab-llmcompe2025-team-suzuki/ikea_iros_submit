#!/usr/bin/env python3
"""GPU-only worker for the raw GR00T N1.7 pick-table-leg checkpoint."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.desktop.upper_policy.groot_pick_leg_contract import (
    CAMERA_KEYS,
    EMBODIMENT_TAG,
    LEROBOT_VERSION,
    MODEL_ACTION_DIM,
    MODEL_ACTION_HORIZON,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_STATE_DIM,
    TASK_TEXT,
    validate_checkpoint_metadata,
)
from inference.desktop.upper_policy.worker_protocol import receive_message, send_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--task", default=TASK_TEXT)
    return parser.parse_args()


def _decode_rgb(jpeg: bytes, role: str) -> torch.Tensor:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        shape = None if image is None else tuple(image.shape)
        raise ValueError(f"{role} JPEG must decode to 640x480 BGR, got {shape}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # LeRobot's GR00T processor expects an unbatched CHW PolicyFeature. Its
    # AddBatchDimension step makes this BCHW before the N1.7 pack step.
    # Supplying HWC is misread as C=480 and can allocate an enormous malformed
    # VLM video tensor.
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def _feature_config(checkpoint: Path) -> Any:
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig

    inputs = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        for key in CAMERA_KEYS
    }
    inputs["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(MODEL_STATE_DIM,)
    )
    return GrootConfig(
        base_model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        chunk_size=MODEL_ACTION_HORIZON,
        n_action_steps=MODEL_ACTION_HORIZON,
        input_features=inputs,
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION, shape=(MODEL_ACTION_DIM,)
            )
        },
        action_decode_transform="none",
        device="cuda:0",
        # Physical inference never trains. Keeping the checkpoint's BF16
        # parameters avoids a transient ~19 GB FP32 copy on 32 GB-RAM
        # workstations and matches the N1.7 inference compute dtype.
        use_bf16=True,
        model_params_fp32=False,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_vlln=False,
    )


def _validate_processor(preprocessor: Any, postprocessor: Any) -> None:
    from lerobot.policies.groot.processor_groot import (
        GrootN17ActionDecodeStep,
        GrootN17PackInputsStep,
    )

    pack = next(
        (step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep)),
        None,
    )
    decode = next(
        (step for step in postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep)),
        None,
    )
    if pack is None or decode is None:
        raise ValueError("GR00T N1.7 processor pack/decode steps are missing")
    if pack.embodiment_tag != EMBODIMENT_TAG:
        raise ValueError(f"processor embodiment changed: {pack.embodiment_tag!r}")
    if tuple(pack.video_modality_keys or ()) != ("cam_0", "cam_1", "cam_2", "cam_3"):
        raise ValueError(f"processor camera order changed: {pack.video_modality_keys!r}")
    if pack.valid_action_horizon != MODEL_ACTION_HORIZON:
        raise ValueError(
            f"processor valid horizon changed: {pack.valid_action_horizon}"
        )
    if decode.env_action_dim != MODEL_ACTION_DIM:
        raise ValueError(f"processor decoded action dim changed: {decode.env_action_dim}")
    modality = decode.modality_config or {}
    if tuple((modality.get("state") or {}).get("modality_keys") or ()) != (
        "robot_q",
        "hand",
    ):
        raise ValueError("processor state group order changed")
    if tuple((modality.get("action") or {}).get("modality_keys") or ()) != (
        "robot_q",
        "hand",
    ):
        raise ValueError("processor action group order changed")


def _make_inference_policy(config: Any) -> Any:
    """Load the raw checkpoint without a second full CPU state-dict copy.

    LeRobot 0.6.0's generic raw-checkpoint path does not forward
    ``low_cpu_mem_usage``. A 9.5 GB BF16 checkpoint can therefore briefly use
    more than 30 GB RAM while Transformers materializes the state dict. This
    local subclass changes loading mechanics only; model architecture,
    processors, weights, IK/action semantics, and official checkout remain
    untouched.
    """

    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
    from lerobot.policies.groot.modeling_groot import (
        GrootPolicy,
        _tie_unused_qwen_lm_head,
    )

    def _load_safetensor_shards_one_tensor_at_a_time(
        model: torch.nn.Module,
        checkpoint: str | Path,
        *,
        device: str,
    ) -> None:
        """Avoid materializing a complete multi-gigabyte shard in system RAM."""

        from safetensors import safe_open

        root = Path(checkpoint)
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint weight index is missing or empty")
        model_keys = set(model.state_dict())
        checkpoint_keys = set(weight_map)
        unexpected = sorted(checkpoint_keys - model_keys)
        if unexpected:
            raise ValueError(f"checkpoint has unexpected tensors: {unexpected[:20]}")
        for shard_name in sorted(set(weight_map.values())):
            expected_keys = {
                key for key, value in weight_map.items() if value == shard_name
            }
            with safe_open(
                root / shard_name, framework="pt", device="cpu"
            ) as reader:
                actual_keys = set(reader.keys())
                if actual_keys != expected_keys:
                    raise ValueError(
                        f"checkpoint shard index mismatch for {shard_name}"
                    )
                for key in sorted(actual_keys):
                    tensor = reader.get_tensor(key)
                    target_dtype = (
                        torch.bfloat16
                        if torch.is_floating_point(tensor)
                        else tensor.dtype
                    )
                    set_module_tensor_to_device(
                        model,
                        key,
                        device,
                        value=tensor,
                        dtype=target_dtype,
                    )
                    del tensor

    class _LowMemoryInferenceGrootPolicy(GrootPolicy):
        def _create_groot_model(self) -> Any:
            raw_config = GR00TN17Config.from_pretrained(
                self.config.base_model_path,
                local_files_only=True,
            )
            raw_config.tune_llm = False
            raw_config.tune_visual = False
            raw_config.tune_projector = False
            raw_config.tune_diffusion_model = False
            raw_config.tune_vlln = False
            raw_config.tune_top_llm_layers = 0
            raw_config.use_flash_attention = self.config.use_flash_attention
            raw_config.load_bf16 = True
            raw_config.backbone_trainable_params_fp32 = False
            # Instantiate parameters on the meta device, then stream each
            # safetensors shard straight to the GPU. The generic Transformers
            # path builds a complete CPU model and a second state dict, which
            # exceeds this workstation's 32 GB system RAM.
            with init_empty_weights():
                model = GR00TN17(
                    raw_config,
                    load_backbone_weights=False,
                    transformers_loading_kwargs={
                        "trust_remote_code": True,
                        "local_files_only": True,
                    },
                )
                backbone = getattr(model, "backbone", None)
                qwen_model = getattr(backbone, "model", None)
                if qwen_model is not None:
                    _tie_unused_qwen_lm_head(qwen_model)
            print("[groot-worker] meta model constructed", file=sys.stderr, flush=True)
            _load_safetensor_shards_one_tensor_at_a_time(
                model,
                self.config.base_model_path,
                device=str(self.config.device),
            )
            print("[groot-worker] checkpoint shards loaded", file=sys.stderr, flush=True)
            backbone = getattr(model, "backbone", None)
            qwen_model = getattr(backbone, "model", None)
            if qwen_model is not None:
                _tie_unused_qwen_lm_head(qwen_model)
            missing_meta = [
                f"parameter:{name}"
                for name, value in model.named_parameters()
                if value.device.type == "meta"
            ] + [
                f"buffer:{name}"
                for name, value in model.named_buffers()
                if value.device.type == "meta"
            ]
            if missing_meta:
                raise RuntimeError(
                    "checkpoint left tensors on the meta device: "
                    + ", ".join(missing_meta[:20])
                )
            model.to(str(self.config.device))
            print("[groot-worker] model dispatched", file=sys.stderr, flush=True)
            model.backbone.set_trainable_parameters(
                tune_visual=False,
                tune_llm=False,
                tune_top_llm_layers=0,
            )
            model.action_head.set_trainable_parameters(
                tune_projector=False,
                tune_diffusion_model=False,
                tune_vlln=False,
            )
            model.eval()
            return model

    return _LowMemoryInferenceGrootPolicy(config)


class Runtime:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        *,
        model_repo_id: str = MODEL_REPO_ID,
        model_revision: str = MODEL_REVISION,
        task: str = TASK_TEXT,
    ) -> None:
        if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"physical GR00T inference requires lerobot=={LEROBOT_VERSION}")
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        self.contract = validate_checkpoint_metadata(
            checkpoint,
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            task=task,
        )
        self.task = task
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for physical GR00T inference")
        config = _feature_config(checkpoint)
        config.device = str(self.device)
        with redirect_stdout(sys.stderr):
            self.model = _make_inference_policy(config)
            self.model.to(self.device)
            self.model.eval()
            self.model.reset()
            print("[groot-worker] building official processors", file=sys.stderr, flush=True)
            self.preprocessor, self.postprocessor = (
                make_groot_pre_post_processors_from_pretrained(
                    self.model.config,
                    str(checkpoint),
                    preprocessor_overrides={
                        "device_processor": {"device": str(self.device)},
                        "groot_n1_7_vlm_encode_v1": {"device": str(self.device)},
                    },
                )
            )
            print("[groot-worker] official processors ready", file=sys.stderr, flush=True)
        _validate_processor(self.preprocessor, self.postprocessor)

    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (MODEL_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"state must be finite [{MODEL_STATE_DIM}]")
        cameras = request.get("cameras")
        if not isinstance(cameras, dict) or set(cameras) != set(CAMERA_KEYS):
            raise ValueError(f"camera keys must be exactly {CAMERA_KEYS}")
        task = request.get("task")
        if task != self.task:
            raise ValueError(f"task must exactly match manifest task {self.task!r}")

        raw: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": task,
        }
        for key in CAMERA_KEYS:
            raw[key] = _decode_rgb(bytes(cameras[key]), key)
        started = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(raw)
            normalized = self.model.predict_action_chunk(processed)
            decoded = self.postprocessor(normalized)
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = decoded.detach().cpu().float().numpy()
        if (
            result.ndim != 3
            or result.shape[0] != 1
            or result.shape[1] < MODEL_ACTION_HORIZON
            or result.shape[2] != MODEL_ACTION_DIM
        ):
            raise RuntimeError(
                "decoded GR00T action must be [1,>=16,38], "
                f"got {result.shape}"
            )
        result = result[0, :MODEL_ACTION_HORIZON]
        if not np.isfinite(result).all():
            raise RuntimeError("decoded GR00T action contains NaN or Inf")
        return result, elapsed_ms


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = Runtime(
        args.checkpoint.resolve(),
        args.device,
        model_repo_id=args.model_repo_id,
        model_revision=args.model_revision,
        task=args.task,
    )
    send_message(
        sys.stdout.buffer,
        {
            "type": "ready",
            "contract": runtime.contract,
            "device": str(runtime.device),
        },
    )
    while True:
        try:
            request = receive_message(sys.stdin.buffer)
        except EOFError:
            # Parent runtime exited or was terminated.  This is the normal
            # fail-safe ownership boundary, not a model failure.
            return 0
        if not isinstance(request, dict):
            raise TypeError("policy request must be a mapping")
        if request.get("type") == "close":
            send_message(sys.stdout.buffer, {"type": "closed"})
            return 0
        if request.get("type") != "predict":
            raise ValueError(f"unsupported policy request: {request.get('type')!r}")
        try:
            actions, elapsed_ms = runtime.predict(request)
            send_message(
                sys.stdout.buffer,
                {
                    "type": "prediction",
                    "request_id": int(request["request_id"]),
                    "actions": actions.tolist(),
                    "inference_ms": elapsed_ms,
                },
            )
        except Exception as exc:  # noqa: BLE001
            send_message(
                sys.stdout.buffer,
                {
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


if __name__ == "__main__":
    raise SystemExit(main())
