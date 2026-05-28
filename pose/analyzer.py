"""Lightweight motion analysis used by the prototype session loop."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MovementState:
    """Motion snapshot produced for every UI tick."""

    phase: float
    phase_value: float
    movement_state: str
    hip_angle: float
    repetition_detected: bool


class MovementAnalyzer:
    """Simulate a movement-analysis pipeline that can later be replaced by pose landmarks."""

    def __init__(
        self,
        phase_speed: float = 2.4,
        repetition_threshold: float = 0.92,
        base_hip_angle: float = 170.0,
        hip_angle_amplitude: float = 8.0,
    ) -> None:
        """Store the movement model parameters."""

        self.phase_speed = phase_speed
        self.repetition_threshold = repetition_threshold
        self.base_hip_angle = base_hip_angle
        self.hip_angle_amplitude = hip_angle_amplitude
        self.reset()

    def reset(self) -> None:
        """Reset the internal phase state before a new session."""

        self.current_phase = 0.0
        self._previous_state = "OPUSZCZANIE"

    def step(self, delta_seconds: float) -> MovementState:
        """Advance the simulated motion model and return the current state."""

        self.current_phase += delta_seconds * self.phase_speed
        phase_value = math.sin(self.current_phase)
        movement_state = "PODNOSZENIE" if phase_value > 0 else "OPUSZCZANIE"
        repetition_detected = (
            self._previous_state == "OPUSZCZANIE"
            and movement_state == "PODNOSZENIE"
            and phase_value > self.repetition_threshold
        )
        self._previous_state = movement_state
        hip_angle = self.base_hip_angle + math.sin(self.current_phase * 1.3) * self.hip_angle_amplitude
        return MovementState(
            phase=self.current_phase,
            phase_value=phase_value,
            movement_state=movement_state,
            hip_angle=hip_angle,
            repetition_detected=repetition_detected,
        )
