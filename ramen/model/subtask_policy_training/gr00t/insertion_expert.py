"""Reviewed M3-M4 supervision for the flip-table insertion experts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "flip_table_insertion_supervision_v1"
PHASE_SAMPLING_SCHEMA_VERSION = "team_ramen_phase_sampling_plan/v3"
SUPPORTED_TRAJECTORY_TYPES = frozenset({"full_success", "recovery_success"})
HARD_NEGATIVE_REASONS = frozenset(
    {
        "left_support_lost",
        "right_hand_above_tabletop",
        "insufficient_insertion_depth",
        "right_wrist_limit",
    }
)


@dataclass(frozen=True)
class InsertionSupervision:
    """Per-frame labels that never become policy inputs."""

    episode_index: int
    length: int
    expert_role: str
    trajectory_type: str
    m3_frame: int
    m4_frame: int
    interval_start: int
    interval_end: int
    action_loss_eligible: tuple[bool, ...]
    eef_loss_eligible: tuple[bool, ...]
    left_support_ready: tuple[float, ...]
    left_support_valid: tuple[bool, ...]
    right_insert_complete: tuple[float, ...]
    right_insert_valid: tuple[bool, ...]
    dex1_transition: tuple[bool, ...]
    hard_negative_reason: str | None = None

    @property
    def is_success_interval(self) -> bool:
        return self.hard_negative_reason is None and any(self.action_loss_eligible)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_index": self.episode_index,
            "length": self.length,
            "expert_role": self.expert_role,
            "trajectory_type": self.trajectory_type,
            "m3_frame": self.m3_frame,
            "m4_frame": self.m4_frame,
            "interval_start": self.interval_start,
            "interval_end": self.interval_end,
            "action_loss_eligible": list(self.action_loss_eligible),
            "eef_loss_eligible": list(self.eef_loss_eligible),
            "left_support_ready": list(self.left_support_ready),
            "left_support_valid": list(self.left_support_valid),
            "right_insert_complete": list(self.right_insert_complete),
            "right_insert_valid": list(self.right_insert_valid),
            "dex1_transition": list(self.dex1_transition),
            "hard_negative_reason": self.hard_negative_reason,
            "policy_input": False,
        }


def build_success_supervision(
    progress: Mapping[str, Any],
    *,
    trajectory_type: str,
    eef_loss_eligible: bool | Sequence[bool] = True,
    pre_roll_frames: int = 15,
    post_roll_frames: int = 10,
    classifier_guard_frames: int = 5,
) -> InsertionSupervision:
    """Build a reviewed insertion window from ordered M3 and M4 milestones.

    M3/M4 labels are only accepted when they were explicitly reviewed. This
    prevents the hand-command heuristic used by the generic progress sidecar
    from silently becoming insertion ground truth.
    """

    if trajectory_type not in SUPPORTED_TRAJECTORY_TYPES:
        raise ValueError(f"unsupported successful trajectory type: {trajectory_type!r}")
    length = _positive_int(progress.get("length"), "length")
    episode_index = _non_negative_int(progress.get("episode_index"), "episode_index")
    milestones = progress.get("milestones")
    if not isinstance(milestones, Mapping):
        raise ValueError("progress annotation lacks milestones")
    m3 = _reviewed_frame(milestones, "M3", length)
    m4 = _reviewed_frame(milestones, "M4", length)
    if m4 <= m3:
        raise ValueError("M4 must occur after M3")
    if pre_roll_frames < 0 or post_roll_frames < 0 or classifier_guard_frames < 1:
        raise ValueError("invalid insertion supervision window")

    action_start = max(0, m3 - pre_roll_frames)
    action_end = min(length, m4 + post_roll_frames + 1)
    action = _range_mask(length, action_start, action_end)
    eef = _eef_mask(eef_loss_eligible, length, action)

    support_start = max(0, m3 - classifier_guard_frames + 1)
    left_support = tuple(float(index >= support_start) for index in range(length))
    left_valid = tuple(index >= action_start for index in range(length))
    insert_complete = tuple(float(index >= m4) for index in range(length))
    insert_valid = tuple(index >= m3 for index in range(length))
    dex1 = _bool_vector(progress.get("dex1_transition_mask"), length, "dex1_transition_mask")

    return InsertionSupervision(
        episode_index=episode_index,
        length=length,
        expert_role="insertion",
        trajectory_type=trajectory_type,
        m3_frame=m3,
        m4_frame=m4,
        interval_start=action_start,
        interval_end=action_end,
        action_loss_eligible=action,
        eef_loss_eligible=eef,
        left_support_ready=left_support,
        left_support_valid=left_valid,
        right_insert_complete=insert_complete,
        right_insert_valid=insert_valid,
        dex1_transition=dex1,
    )


def build_finish_supervision(
    progress: Mapping[str, Any],
    *,
    trajectory_type: str,
    eef_loss_eligible: bool | Sequence[bool] = True,
    pre_roll_frames: int = 10,
) -> InsertionSupervision:
    """Build the successful M4-M6 action window for the finish expert."""

    if trajectory_type not in SUPPORTED_TRAJECTORY_TYPES:
        raise ValueError(f"unsupported successful trajectory type: {trajectory_type!r}")
    length = _positive_int(progress.get("length"), "length")
    episode_index = _non_negative_int(progress.get("episode_index"), "episode_index")
    milestones = progress.get("milestones")
    if not isinstance(milestones, Mapping):
        raise ValueError("progress annotation lacks milestones")
    m3 = _reviewed_frame(milestones, "M3", length)
    m4 = _reviewed_frame(milestones, "M4", length)
    m6 = _reviewed_frame(milestones, "M6", length)
    if not m3 < m4 < m6:
        raise ValueError("finish supervision requires ordered M3 < M4 < M6")
    if pre_roll_frames < 0:
        raise ValueError("finish pre-roll must be non-negative")
    action_start = max(0, m4 - pre_roll_frames)
    action_end = length
    action = _range_mask(length, action_start, action_end)
    eef = _eef_mask(eef_loss_eligible, length, action)
    dex1 = _bool_vector(progress.get("dex1_transition_mask"), length, "dex1_transition_mask")
    return InsertionSupervision(
        episode_index=episode_index,
        length=length,
        expert_role="finish",
        trajectory_type=trajectory_type,
        m3_frame=m3,
        m4_frame=m4,
        interval_start=action_start,
        interval_end=action_end,
        action_loss_eligible=action,
        eef_loss_eligible=eef,
        left_support_ready=tuple(float(index >= m3) for index in range(length)),
        left_support_valid=tuple(index >= action_start for index in range(length)),
        right_insert_complete=tuple(float(index >= m4) for index in range(length)),
        right_insert_valid=tuple(index >= m3 for index in range(length)),
        dex1_transition=dex1,
    )


def build_hard_negative_supervision(
    *,
    episode_index: int,
    length: int,
    reason: str,
    left_support_ready: Sequence[float],
    right_insert_complete: Sequence[float],
) -> InsertionSupervision:
    """Create classifier-only supervision from a failed real rollout."""

    if reason not in HARD_NEGATIVE_REASONS:
        raise ValueError(f"unsupported hard-negative reason: {reason!r}")
    length = _positive_int(length, "length")
    episode_index = _non_negative_int(episode_index, "episode_index")
    support = _probability_vector(left_support_ready, length, "left_support_ready")
    insertion = _probability_vector(
        right_insert_complete, length, "right_insert_complete"
    )
    return InsertionSupervision(
        episode_index=episode_index,
        length=length,
        expert_role="insertion_classifier_negative",
        trajectory_type="failure_diagnostic",
        m3_frame=-1,
        m4_frame=-1,
        interval_start=-1,
        interval_end=-1,
        action_loss_eligible=(False,) * length,
        eef_loss_eligible=(False,) * length,
        left_support_ready=support,
        left_support_valid=(True,) * length,
        right_insert_complete=insertion,
        right_insert_valid=(True,) * length,
        dex1_transition=(False,) * length,
        hard_negative_reason=reason,
    )


def validate_supervision(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported insertion supervision schema")
    length = _positive_int(record.get("length"), "length")
    if record.get("expert_role") not in {
        "insertion",
        "finish",
        "insertion_classifier_negative",
    }:
        raise ValueError("invalid insertion supervision expert role")
    for key in (
        "action_loss_eligible",
        "eef_loss_eligible",
        "left_support_ready",
        "left_support_valid",
        "right_insert_complete",
        "right_insert_valid",
        "dex1_transition",
    ):
        if not isinstance(record.get(key), list) or len(record[key]) != length:
            raise ValueError(f"{key} must contain exactly {length} values")
    if record.get("policy_input") is not False:
        raise ValueError("insertion labels must be auxiliary targets only")
    if record.get("hard_negative_reason") is not None and any(
        bool(value) for value in record["action_loss_eligible"]
    ):
        raise ValueError("hard negatives cannot provide action supervision")


def build_insertion_sampling_plan(
    base_plan: Mapping[str, Any],
    supervision_records: Sequence[Mapping[str, Any]],
    *,
    insertion_sidecar_sha256: str,
    dex1_transition_multiplier: float = 4.0,
) -> dict[str, Any]:
    """Restrict sampling to reviewed expert windows.

    The generic v4 phase plan covers the whole task.  Reusing its weights for
    an insertion-only expert would mostly sample frames with a masked action
    loss.  This derived plan retains the immutable episode topology while
    assigning nonzero weight only to reviewed M3-M4 or hard-negative windows.
    """

    if base_plan.get("schema_version") != PHASE_SAMPLING_SCHEMA_VERSION:
        raise ValueError("insertion sampling requires a v3 phase plan")
    if dex1_transition_multiplier <= 0.0:
        raise ValueError("Dex1 transition multiplier must be positive")
    _validate_sha256(insertion_sidecar_sha256, "insertion sidecar")

    supervision: dict[int, Mapping[str, Any]] = {}
    for record in supervision_records:
        validate_supervision(record)
        episode_index = _non_negative_int(record.get("episode_index"), "episode_index")
        if episode_index in supervision:
            raise ValueError(f"duplicate insertion supervision for episode {episode_index}")
        supervision[episode_index] = record

    source_episodes = base_plan.get("episodes")
    if not isinstance(source_episodes, list) or not source_episodes:
        raise ValueError("base phase plan has no episodes")
    episodes: list[dict[str, Any]] = []
    reviewed_frames = 0
    action_frames = 0
    dex1_frames = 0
    seen: set[int] = set()
    for source in source_episodes:
        if not isinstance(source, Mapping):
            raise ValueError("base phase plan contains an invalid episode")
        episode_index = _non_negative_int(source.get("episode_index"), "episode_index")
        length = _positive_int(source.get("length"), "length")
        if episode_index in seen:
            raise ValueError(f"base phase plan repeats episode {episode_index}")
        seen.add(episode_index)
        record = supervision.get(episode_index)
        weights = [0.0] * length
        action_loss_eligible = False
        auxiliary_loss_eligible = False
        if record is not None:
            if int(record["length"]) != length:
                raise ValueError(
                    f"episode {episode_index} supervision length differs from phase plan"
                )
            action = [bool(value) for value in record["action_loss_eligible"]]
            left_valid = [bool(value) for value in record["left_support_valid"]]
            right_valid = [bool(value) for value in record["right_insert_valid"]]
            dex1 = [bool(value) for value in record["dex1_transition"]]
            if record.get("hard_negative_reason") is None:
                sampled = action
            else:
                sampled = [left or right for left, right in zip(left_valid, right_valid, strict=True)]
            for index, selected in enumerate(sampled):
                if not selected:
                    continue
                weights[index] = (
                    float(dex1_transition_multiplier) if dex1[index] and action[index] else 1.0
                )
                reviewed_frames += 1
                action_frames += int(action[index])
                dex1_frames += int(dex1[index] and action[index])
            action_loss_eligible = any(action)
            auxiliary_loss_eligible = any(
                selected and (left_valid[index] or right_valid[index])
                for index, selected in enumerate(sampled)
            )
            if not any(weight > 0.0 for weight in weights):
                raise ValueError(f"episode {episode_index} has no reviewed sampled frame")
        train_eligible = any(weight > 0.0 for weight in weights)
        episodes.append(
            {
                "episode_index": episode_index,
                "length": length,
                "recovery": False,
                "train_eligible": train_eligible,
                "action_loss_eligible": action_loss_eligible,
                "auxiliary_loss_eligible": auxiliary_loss_eligible,
                "phase_supervision_eligible": False,
                "frame_weights": weights,
            }
        )
    unknown = sorted(set(supervision) - seen)
    if unknown:
        raise ValueError(f"insertion supervision references unknown episodes: {unknown[:10]}")
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("base phase plan episode indices must be contiguous")
    return {
        "schema_version": PHASE_SAMPLING_SCHEMA_VERSION,
        "sampling_scope": "reviewed_flip_table_insertion",
        "progress_sidecar_sha256": base_plan.get("progress_sidecar_sha256"),
        "insertion_sidecar_sha256": insertion_sidecar_sha256,
        "late_phase_multiplier": 1.0,
        "dex1_transition_multiplier": float(dex1_transition_multiplier),
        "recovery_multiplier": 1.0,
        "recovery_prompt_enabled": False,
        "excluded_episode_indices": [],
        "action_ineligible_episode_indices": sorted(
            episode["episode_index"]
            for episode in episodes
            if not episode["action_loss_eligible"]
        ),
        "summary": {
            "episode_count": len(episodes),
            "reviewed_episode_count": len(supervision),
            "train_eligible_episode_count": sum(
                int(episode["train_eligible"]) for episode in episodes
            ),
            "action_loss_eligible_episode_count": sum(
                int(episode["action_loss_eligible"]) for episode in episodes
            ),
            "auxiliary_loss_eligible_episode_count": sum(
                int(episode["auxiliary_loss_eligible"]) for episode in episodes
            ),
            "reviewed_sampled_frame_count": reviewed_frames,
            "action_loss_frame_count": action_frames,
            "dex1_transition_frame_count": dex1_frames,
        },
        "episodes": episodes,
    }


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} SHA-256 is malformed")


def _reviewed_frame(milestones: Mapping[str, Any], name: str, length: int) -> int:
    value = milestones.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing {name} milestone")
    if value.get("source") != "reviewed_phase_frames" or value.get("valid") is not True:
        raise ValueError(f"{name} must be a reviewed valid milestone")
    frame = _non_negative_int(value.get("frame"), f"{name}.frame")
    if frame >= length:
        raise ValueError(f"{name}.frame is outside the episode")
    return frame


def _range_mask(length: int, start: int, end: int) -> tuple[bool, ...]:
    return tuple(start <= index < end for index in range(length))


def _eef_mask(
    value: bool | Sequence[bool], length: int, action_mask: Sequence[bool]
) -> tuple[bool, ...]:
    if isinstance(value, bool):
        source = (value,) * length
    else:
        source = _bool_vector(value, length, "eef_loss_eligible")
    return tuple(bool(source[index] and action_mask[index]) for index in range(length))


def _bool_vector(value: Any, length: int, name: str) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(bool(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _probability_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(float(item) for item in value)
    if len(result) != length or any(not 0.0 <= item <= 1.0 for item in result):
        raise ValueError(f"{name} must contain {length} probabilities")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
