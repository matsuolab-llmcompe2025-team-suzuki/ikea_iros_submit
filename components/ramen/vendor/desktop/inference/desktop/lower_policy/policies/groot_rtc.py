"""GR00T N1.7 の action decode 逆変換 = RTC prefix builder (Issue #137 Phase B)。

# なぜ必要か

RTC は「前 chunk の未実行分」を denoising の初期値・拘束として渡す。denoising は
**正規化済み model action space** で走るので、prefix もその空間に載せ直さないと
意味の違う値をモデルに与えることになる。

我々が保持しているのは post_processor 通過後の **絶対値 53D** なので、
`GrootN17ActionDecodeStep` が行う

    正規化 relative → (unnormalize) → relative → (現 state 基準で復元) → 絶対

の逆を辿る必要がある。本 module がその逆変換。

# LeRobot 汎用 RTC が使えない理由

`lerobot/policies/rtc/relative.py: reanchor_relative_rtc_prefix()` は汎用の
`RelativeActionsProcessorStep` を前提にしているが、我々の ckpt の postprocessor は
`groot_n1_7_action_decode_v1` 1 step のみ (`use_relative_action=True` /
`use_percentiles=True` / `env_action_dim=53`) で、その step は入っていない。
`lerobot/policies/groot/modeling_groot.py:_prepare_n1_7_rtc_inputs` が
relative ckpt で RTC を早期 return しているのはこのため
("until a GROOT-specific RTC path can pass re-anchored absolute leftovers through")。

# 実 ckpt の action group 構成 (Team-RAMEN rotate/insert 系で実測)

| group | dim | rep | 逆変換 |
|---|---:|---|---|
| left/right_wrist_eef_9d | 9+9 | RELATIVE / EEF XYZ_ROT6D | inv(ref) @ abs → 正規化 |
| left/right_arm | 7+7 | RELATIVE / NON_EEF | abs - ref → 正規化 |
| left/right_hand, waist, base_height_command, navigate_command | 7+7+3+1+3 | ABSOLUTE | 正規化のみ |

stats は checkpoint により per-dim の 1-D または per-horizon の 2-D。RTC の
leftover は新 chunk の先頭へ置かれ、decode 側も stats の先頭行から適用するため、
2-D の場合は encoder も新しい prefix 位置 `0:rows` の行を使う。

# lerobot private symbol への依存

stats 選択ロジック (`_n1_7_decode_stats_for_action`) は decode 側と**完全に同じ
ものを使う**必要があるため、再実装せず import する。private だが
`inference/desktop/pixi.toml` が `lerobot==0.6.1` を pin しているので版は固定。
upgrade 時に壊れたらここで気付けるよう、tests で decode との round trip を張る。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # 型注釈のみ。実体は lazy import (下記 import 理由参照)
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep


def absolute_eef_to_relative(
    action: np.ndarray, reference_state: np.ndarray
) -> np.ndarray:
    """絶対 EEF pose (xyz+rot6d) を reference 基準の相対に戻す。

    `lerobot.policies.groot.utils.relative_eef_to_absolute` の厳密な逆:
    復元が `abs = ref @ rel` なので、こちらは `rel = inv(ref) @ abs`。

    Args:
        action: (B, T, 9) 絶対 EEF pose。
        reference_state: (B, 9) 基準 pose。

    Returns:
        (B, T, 9) float32 の相対 pose。
    """
    # lazy: lerobot は inference/desktop の pixi sub-workspace 専用 (torch +
    # transformers を引く)。default env での import 収集を壊さないため関数内。
    from lerobot.policies.groot.utils import (
        homogeneous_to_xyz_rot6d,
        xyz_rot6d_to_homogeneous,
    )

    if action.ndim != 3 or action.shape[-1] != 9:
        raise ValueError(f"action must be (B, T, 9), got shape {action.shape}")
    if reference_state.ndim != 2 or reference_state.shape[-1] != 9:
        raise ValueError(
            f"reference_state must be (B, 9), got shape {reference_state.shape}"
        )
    out = np.empty_like(action, dtype=np.float64)
    for batch_idx in range(action.shape[0]):
        # rot6d_to_matrix が Gram-Schmidt で正規直交化するので inv は数値的に安定。
        reference_inv = np.linalg.inv(
            xyz_rot6d_to_homogeneous(reference_state[batch_idx])
        )
        for timestep in range(action.shape[1]):
            absolute = xyz_rot6d_to_homogeneous(action[batch_idx, timestep])
            out[batch_idx, timestep] = homogeneous_to_xyz_rot6d(
                reference_inv @ absolute
            )
    return out.astype(np.float32)


def normalize_min_max(
    action: np.ndarray, min_v: np.ndarray, max_v: np.ndarray
) -> np.ndarray:
    """`_unnormalize_min_max` の逆 ([min, max] → [-1, 1])。

    decode 側は unnormalize 前に入力を [-1, 1] へ clip するので、こちらも
    出力を clip して往復を対称にする。stats 範囲外の絶対値は端に張り付く。
    """
    span = np.asarray(max_v, dtype=np.float64) - np.asarray(min_v, dtype=np.float64)
    # 定数次元 (min == max) は 0 除算になるので中央値 0 を返す。decode 側は
    # そこを常に min に潰すため、往復しても情報は元から失われている。
    safe_span = np.where(np.abs(span) < 1e-12, 1.0, span)
    normalized = (action - np.asarray(min_v, dtype=np.float64)) / safe_span * 2.0 - 1.0
    normalized = np.where(np.abs(span) < 1e-12, 0.0, normalized)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def encode_absolute_chunk_to_model_space(
    absolute_chunk: np.ndarray,
    *,
    reference_raw_state: dict[str, np.ndarray],
    decode_step: "GrootN17ActionDecodeStep",
) -> np.ndarray:
    """絶対 53D chunk → 正規化済み model action space (RTC prefix)。

    `GrootN17ActionDecodeStep.__call__` の逆。group ごとに checkpoint 順で
    slice し、RELATIVE group は `reference_raw_state` 基準で相対に戻してから
    decode と同じ stats で正規化する。

    Args:
        absolute_chunk: (rows, env_action_dim) 絶対値。post_processor 出力と
            同じ空間 (= `_sync_predict_chunk_19d` の `action_chunk_53d`)。
        reference_raw_state: `GrootN17PackInputsStep.get_cached_raw_state()` の
            戻り値。checkpoint modality key → (B, D)。**今 tick の観測で
            preprocessor を回した直後のもの**を渡すこと (Isaac の decode と同じ
            基準を使うため)。
        decode_step: stats / modality_config の出所。postprocessor から取る。

    Returns:
        (rows, sum(group dims)) float32、各要素 [-1, 1]。

    Raises:
        ValueError: decode_step が RTC 非対応の設定 (action_decode_transform 有り、
            stats が per-horizon 2-D 等)、または入力 shape 不整合。
        KeyError: RELATIVE group に対応する raw state が無い。
    """
    # lazy: 上記 absolute_eef_to_relative と同じ理由。
    from lerobot.policies.groot.processor_groot import (
        _n1_7_decode_stats_for_action,
    )
    from lerobot.policies.groot.utils import config_value, stat_dim_from_entry

    if absolute_chunk.ndim != 2:
        raise ValueError(
            f"absolute_chunk must be (rows, action_dim), got shape "
            f"{absolute_chunk.shape}"
        )
    if absolute_chunk.shape[0] < 1:
        raise ValueError("absolute_chunk must contain at least one row")
    if not np.all(np.isfinite(absolute_chunk)):
        raise ValueError("absolute_chunk contains non-finite values")
    if decode_step.action_decode_transform is not None:
        # decode 側に不可逆な後処理が挟まる設定。逆変換の正しさを保証できない。
        raise ValueError(
            "RTC prefix cannot be built for a checkpoint with "
            f"action_decode_transform={decode_step.action_decode_transform!r}"
        )
    if decode_step.raw_stats is None or decode_step.modality_config is None:
        raise ValueError("decode_step must carry raw_stats and modality_config")

    action_config: dict[str, Any] = decode_step.modality_config.get("action", {})
    action_keys = action_config.get("modality_keys", [])
    action_configs = action_config.get("action_configs", [])
    if not action_keys:
        raise ValueError("decode_step modality_config has no action modality_keys")

    chunk = np.asarray(absolute_chunk, dtype=np.float64)[None, :, :]  # (1, T, D)
    encoded_groups: list[np.ndarray] = []
    start_idx = 0
    for idx, key in enumerate(action_keys):
        if not isinstance(key, str):
            continue
        stats_entry = decode_step.raw_stats.get("action", {}).get(key, {})
        if not isinstance(stats_entry, dict):
            continue
        dim = stat_dim_from_entry(stats_entry)
        if dim <= 0:
            continue
        if start_idx + dim > chunk.shape[-1]:
            raise ValueError(
                f"absolute_chunk width {chunk.shape[-1]} is too small for action "
                f"group {key!r} at offset {start_idx} (needs {dim})"
            )
        cfg = (
            action_configs[idx]
            if idx < len(action_configs) and isinstance(action_configs[idx], dict)
            else {}
        )
        group = chunk[..., start_idx : start_idx + dim]

        is_relative = decode_step.use_relative_action and (
            config_value(cfg.get("rep")) == "relative"
        )
        if is_relative:
            state_key = cfg.get("state_key") or key
            if state_key not in reference_raw_state:
                raise KeyError(
                    f"Missing cached raw state {state_key!r} for relative action "
                    f"group {key!r}"
                )
            reference = np.asarray(reference_raw_state[state_key], dtype=np.float64)
            action_type = config_value(cfg.get("type"))
            action_format = config_value(cfg.get("format"))
            if action_type == "non_eef":
                group = group - reference[:, None, :]
            elif action_type == "eef" and action_format == "xyz+rot6d":
                group = absolute_eef_to_relative(group, reference).astype(np.float64)
            else:
                raise ValueError(
                    f"Unsupported relative N1.7 action config for {key!r}: {cfg}"
                )

        min_v, max_v = _n1_7_decode_stats_for_action(
            decode_step.raw_stats,
            key,
            cfg,
            use_relative_action=decode_step.use_relative_action,
            use_percentiles=decode_step.use_percentiles,
        )
        if min_v.ndim == 2:
            # Prefix is decoded as a brand-new chunk, so both encode and decode
            # must use rows 0..N-1.  Using the leftover's old horizon indices
            # would not invert GrootN17ActionDecodeStep.__call__.
            if group.shape[1] > min_v.shape[0] or max_v.shape != min_v.shape:
                raise ValueError(
                    f"per-timestep stats for action group {key!r} cannot cover "
                    f"prefix rows={group.shape[1]} (min={min_v.shape}, "
                    f"max={max_v.shape})"
                )
            min_v = min_v[: group.shape[1]]
            max_v = max_v[: group.shape[1]]
        elif min_v.ndim != 1 or max_v.ndim != 1:
            raise ValueError(
                f"unsupported stats rank for action group {key!r}: "
                f"min={min_v.shape}, max={max_v.shape}"
            )
        encoded_groups.append(normalize_min_max(group, min_v, max_v))
        start_idx += dim

    if not encoded_groups:
        raise ValueError("no action groups were encoded")
    prefix = np.concatenate(encoded_groups, axis=-1)[0].astype(np.float32)
    if not np.all(np.isfinite(prefix)):
        # 退化した rot6d (全ゼロ等) は Gram-Schmidt で NaN になる。そのまま
        # denoising の初期値に混ぜると chunk 全体が壊れるので、ここで止める。
        raise ValueError("encoded RTC prefix contains non-finite values")
    return prefix


class _RtcPrefixMixin:
    """`_prepare_n1_7_rtc_inputs` を override し、relative ckpt でも RTC を通す。

    上流実装は `use_relative_actions=True` の ckpt で早期 return する。絶対値の
    leftover を model space に載せ直す経路が無いためで、その変換は
    `encode_absolute_chunk_to_model_space()` が担う。したがって本 override は
    **prefix が既に model space である**ことを前提に、長さ管理と options 構築
    だけを行う。

    上流が持つ「値がゼロの行はパディングとみなして削る」heuristic は採らない。
    正規化後に真値ゼロの行は普通に出るので、行数は呼出側が厳密に管理する。
    """

    def _prepare_n1_7_rtc_inputs(
        self,
        inputs: dict,
        *,
        inference_delay: object,
        prev_chunk_left_over: object,
    ) -> tuple[dict, dict | None]:
        # lazy: env-isolated dependencies (lerobot 環境でのみ available)
        import torch

        if prev_chunk_left_over is None:
            return inputs, None
        if not isinstance(prev_chunk_left_over, torch.Tensor):
            raise TypeError(
                "prev_chunk_left_over must be a torch.Tensor already expressed in "
                "the normalized model action space"
            )
        prefix = prev_chunk_left_over
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        elif prefix.ndim != 3:
            raise ValueError(
                f"prev_chunk_left_over must be (T, A) or (B, T, A), got "
                f"{tuple(prefix.shape)}"
            )
        state = inputs.get("state")
        if state is None:
            raise ValueError("GR00T RTC requires `state` in the preprocessed batch")
        batch_size = int(state.shape[0])
        if prefix.shape[0] == 1 and batch_size > 1:
            prefix = prefix.expand(batch_size, -1, -1).clone()
        elif prefix.shape[0] != batch_size:
            raise ValueError(
                f"prev_chunk_left_over batch {prefix.shape[0]} does not match the "
                f"current batch {batch_size}"
            )

        model_horizon = int(
            getattr(self._groot_model.config, "action_horizon", self.config.chunk_size)
        )
        if prefix.shape[1] > model_horizon:
            # prefix は新 chunk の先頭に載るので、溢れたら**末尾**を捨てる
            # (上流は逆向きに切るが、あちらは prefix の意味づけが異なる)。
            prefix = prefix[:, :model_horizon]
        rows = int(prefix.shape[1])
        if rows < 1:
            return inputs, None

        max_action_dim = int(
            getattr(self._groot_model.config, "max_action_dim", self.config.max_action_dim)
        )
        if prefix.shape[2] > max_action_dim:
            prefix = prefix[:, :, :max_action_dim]
        elif prefix.shape[2] < max_action_dim:
            pad = torch.zeros(
                prefix.shape[0],
                prefix.shape[1],
                max_action_dim - prefix.shape[2],
                dtype=prefix.dtype,
                device=prefix.device,
            )
            prefix = torch.cat([prefix, pad], dim=2)

        try:
            frozen_steps = int(inference_delay or 0)
        except (TypeError, ValueError):
            frozen_steps = 0
        frozen_steps = max(0, min(frozen_steps, rows))

        # ramp rate は per-variant で振れるよう GrootConfig.rtc_ramp_rate を優先。
        # 未設定なら checkpoint の値 (GR00T N1.7 既定 6.0)。
        ramp_rate = getattr(self.config, "rtc_ramp_rate", None)
        if ramp_rate is None:
            ramp_rate = getattr(self._groot_model.config, "rtc_ramp_rate", 6.0)

        options = {
            # overlap = 渡した行数そのもの。呼出側が overlap_steps 分だけを
            # 渡す契約にしているので、ここで別途 clamp する必要がない。
            "action_horizon": rows,
            "rtc_overlap_steps": rows,
            "rtc_frozen_steps": frozen_steps,
            "rtc_ramp_rate": float(ramp_rate),
        }
        inputs = dict(inputs)
        inputs["action"] = prefix.to(device=state.device, dtype=state.dtype)
        return inputs, options


def make_rtc_policy_class(base: type | None = None) -> type:
    """RTC prefix override を混ぜた GrootPolicy サブクラスを返す。

    `base` は `Gr00tPolicy.from_ckpt` が選ぶ runtime policy class
    (generic なら None = LeRobot の GrootPolicy、furniture なら
    `FurnitureGrootRuntimePolicy`)。どちらにも同じ override を載せられるよう
    mixin 合成にしてある。
    """
    # lazy: env-isolated dependencies (lerobot 環境でのみ available)
    from lerobot.policies.groot.modeling_groot import GrootPolicy as _LrGroot

    resolved = base or _LrGroot
    return type(f"Rtc{resolved.__name__}", (_RtcPrefixMixin, resolved), {})
