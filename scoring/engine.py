"""Training session scoring and feedback logic."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pose.analyzer import MovementState


@dataclass(frozen=True)
class ScoreSnapshot:
    """Immutable score state shown in the UI."""

    repetitions: int
    warnings: int
    quality: int
    feedback: str
    repetition_event: bool = False
    warning_event: bool = False


class SessionScorer:
    """Accumulate repetition, quality and feedback data for the current session."""

    def __init__(
        self,
        rng: random.Random | None = None,
        initial_quality: int = 91,
        minimum_quality: int = 60,
        recovery_ceiling: int = 95,
        feedback: str = "Gotowy do uruchomienia sesji.",
    ) -> None:
        """Inject the random source and scoring boundaries."""

        self._rng = rng or random.Random()
        self.initial_quality = initial_quality
        self.minimum_quality = minimum_quality
        self.recovery_ceiling = recovery_ceiling
        self.initial_feedback = feedback
        self.reset()

    def reset(self) -> None:
        """Clear the session counters and restore the baseline quality."""

        self.repetitions = 0
        self.warnings = 0
        self.quality = self.initial_quality
        self.feedback = self.initial_feedback

    def update(self, movement: MovementState | None) -> ScoreSnapshot:
        """Update the score from a movement snapshot and return the new state."""

        if movement is None:
            return self.snapshot()

        repetition_event = False
        warning_event = False

        if movement.repetition_detected:
            self.repetitions += 1
            repetition_event = True
            self.feedback = f"Wykryto powtorzenie {self.repetitions}"

        if abs(movement.phase_value) < 0.15 and self._rng.random() < 0.03:
            self.warnings += 1
            warning_event = True
            self.quality = max(self.minimum_quality, self.quality - 1)
            self.feedback = "Utrzymaj stabilniejszy tor ruchu bioder i pelny zakres wyprostu."

        if self.quality < self.recovery_ceiling and self._rng.random() < 0.02:
            self.quality += 1

        if self.repetitions and self.repetitions % 5 == 0:
            self.feedback = "Dobra praca: utrzymano poprawny zakres ruchu."

        return self.snapshot(repetition_event=repetition_event, warning_event=warning_event)

    def snapshot(self, repetition_event: bool = False, warning_event: bool = False) -> ScoreSnapshot:
        """Return the current score without modifying it."""

        return ScoreSnapshot(
            repetitions=self.repetitions,
            warnings=self.warnings,
            quality=self.quality,
            feedback=self.feedback,
            repetition_event=repetition_event,
            warning_event=warning_event,
        )
