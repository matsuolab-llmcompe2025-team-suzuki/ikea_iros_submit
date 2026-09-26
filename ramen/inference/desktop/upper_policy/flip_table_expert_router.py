"""One-way expert routing for the observable flip-table insertion transition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Expert(str, Enum):
    EARLY = "early"
    INSERTION = "insertion"
    FINISH = "finish"


@dataclass(frozen=True)
class InsertionEstimate:
    left_support_ready: float
    right_insert_complete: float
    tabletop_edge_in_grasp_region: float
    tabletop_between_fingers: float
    insertion_depth_ready: float
    right_wrist_limit_margin_rad: float

    def __post_init__(self) -> None:
        probabilities = (
            self.left_support_ready,
            self.right_insert_complete,
            self.tabletop_edge_in_grasp_region,
            self.tabletop_between_fingers,
            self.insertion_depth_ready,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("insertion probabilities must lie in [0, 1]")
        if not np.isfinite(self.right_wrist_limit_margin_rad):
            raise ValueError("right wrist limit margin must be finite")


@dataclass(frozen=True)
class ExpertRouterConfig:
    support_threshold: float = 0.8
    support_confirm_frames: int = 5
    insertion_threshold: float = 0.9
    insertion_confirm_frames: int = 5
    visual_gate_threshold: float = 0.9
    minimum_wrist_limit_margin_rad: float = 0.15
    early_execution_steps: int = 10
    insertion_execution_steps: int = 4
    finish_execution_steps: int = 5
    blend_control_steps: int = 10

    def __post_init__(self) -> None:
        for value in (
            self.support_threshold,
            self.insertion_threshold,
            self.visual_gate_threshold,
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError("router probability thresholds must lie in (0, 1]")
        if self.support_confirm_frames < 1 or self.insertion_confirm_frames < 1:
            raise ValueError("router confirmation windows must be positive")
        if self.minimum_wrist_limit_margin_rad <= 0:
            raise ValueError("wrist limit margin must be positive")
        if self.blend_control_steps < 2:
            raise ValueError("expert transition needs at least two blend steps")
        if self.insertion_execution_steps != 4:
            raise ValueError("insertion expert must replan every four 30 Hz steps")


@dataclass(frozen=True)
class RouterDecision:
    expert: Expert
    execution_steps: int
    transitioned: bool
    right_hand_close_allowed: bool
    reason: str


class FlipTableExpertRouter:
    """Latch Early -> Insertion -> Finish using real-robot observations only."""

    def __init__(self, config: ExpertRouterConfig | None = None) -> None:
        self.config = config or ExpertRouterConfig()
        self.reset()

    def reset(self) -> None:
        self.expert = Expert.EARLY
        self._support_count = 0
        self._insert_count = 0

    def update(self, estimate: InsertionEstimate) -> RouterDecision:
        transitioned = False
        reason = "latched"
        if self.expert is Expert.EARLY:
            self._support_count = (
                self._support_count + 1
                if estimate.left_support_ready >= self.config.support_threshold
                else 0
            )
            if self._support_count >= self.config.support_confirm_frames:
                self.expert = Expert.INSERTION
                transitioned = True
                reason = "left_support_ready_confirmed"

        close_allowed = self._right_hand_close_allowed(estimate)
        if self.expert is Expert.INSERTION:
            insertion_ready = (
                estimate.right_insert_complete >= self.config.insertion_threshold
                and close_allowed
            )
            self._insert_count = self._insert_count + 1 if insertion_ready else 0
            if self._insert_count >= self.config.insertion_confirm_frames:
                self.expert = Expert.FINISH
                transitioned = True
                reason = "right_insert_complete_confirmed"

        return RouterDecision(
            expert=self.expert,
            execution_steps=self.execution_steps,
            transitioned=transitioned,
            right_hand_close_allowed=(
                self.expert is Expert.FINISH or close_allowed
            ),
            reason=reason,
        )

    @property
    def execution_steps(self) -> int:
        if self.expert is Expert.EARLY:
            return self.config.early_execution_steps
        if self.expert is Expert.INSERTION:
            return self.config.insertion_execution_steps
        return self.config.finish_execution_steps

    def _right_hand_close_allowed(self, estimate: InsertionEstimate) -> bool:
        threshold = self.config.visual_gate_threshold
        return (
            estimate.tabletop_edge_in_grasp_region >= threshold
            and estimate.tabletop_between_fingers >= threshold
            and estimate.insertion_depth_ready >= threshold
            and estimate.right_wrist_limit_margin_rad
            >= self.config.minimum_wrist_limit_margin_rad
        )


def blend_absolute_targets(
    previous_target: np.ndarray,
    next_target: np.ndarray,
    *,
    steps: int = 10,
) -> np.ndarray:
    """Blend decoded physical 16D targets at an expert boundary."""

    previous = np.asarray(previous_target, dtype=np.float64)
    following = np.asarray(next_target, dtype=np.float64)
    if previous.shape != (16,) or following.shape != (16,):
        raise ValueError("expert blending requires two physical 16D targets")
    if not np.isfinite(previous).all() or not np.isfinite(following).all():
        raise ValueError("expert targets must be finite")
    if steps < 2:
        raise ValueError("expert transition needs at least two blend steps")
    alpha = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float64)[:, None]
    return previous[None, :] + alpha * (following - previous)[None, :]


def gate_right_dex1_target(
    target: np.ndarray,
    measured: np.ndarray,
    *,
    close_allowed: bool,
    minimum_open_value: float = 4.0,
) -> np.ndarray:
    """Block right Dex1 closing until every observable insertion gate passes.

    Physical targets are ``arms14 + left/right Dex1`` with ``4.5=open`` and
    ``0=closed``.  While closing is blocked, the right hand remains at least as
    open as its measured value and the configured minimum.
    """

    command = np.asarray(target, dtype=np.float64)
    state = np.asarray(measured, dtype=np.float64)
    if command.shape != (16,) or state.shape != (16,):
        raise ValueError("Dex1 gating requires physical 16D target and state")
    if not np.isfinite(command).all() or not np.isfinite(state).all():
        raise ValueError("Dex1 gating inputs must be finite")
    if not 0.0 <= minimum_open_value <= 4.5:
        raise ValueError("minimum Dex1 opening must lie in [0, 4.5]")
    gated = command.copy()
    if not close_allowed:
        gated[15] = max(gated[15], state[15], minimum_open_value)
    return gated
