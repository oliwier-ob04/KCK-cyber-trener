"""Training session scoring and feedback logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        rng: Any = None,  # Przywrócone wstecznie, aby app.py się nie wywalał przy inicjalizacji
        initial_quality: int = 100,
        minimum_quality: int = 60,
        recovery_ceiling: int = 95,  # Przywrócone na wypadek, gdyby app.py też to przekazywał
        feedback: str = "Gotowy do uruchomienia sesji.",
    ) -> None:
        """Inject the scoring boundaries."""
        self.minimum_quality = minimum_quality
        self.initial_quality = initial_quality
        self.initial_feedback = feedback
        self.reset()

    def reset(self) -> None:
        """Clear the session counters and restore the baseline quality."""
        self.repetitions = 0
        self.warnings = 0
        self.quality = self.initial_quality
        self.feedback = self.initial_feedback
        
    def grade_hip_cycle(self, hip_angle: float) -> bool:
        """Ocena tułowia w całym cyklu ruchu."""
        import math
        if math.isnan(hip_angle):
            return False
        return 90.0 <= hip_angle <= 190.0

    def grade_hip_top_hold(self, hip_angle: float) -> bool:
        """Ocena tułowia podczas 1-sekundowego zatrzymania na górze."""
        import math
        if math.isnan(hip_angle):
            return False
        return 170.0 <= hip_angle <= 190.0

    def grade_knee_stable(self, knee_side: float) -> bool:
        """Ocena stabilności kąta kolana podczas zatrzymania na górze."""
        import math
        if math.isnan(knee_side):
            return False
        return 80.0 <= knee_side <= 100.0

    def register_repetition(self, correct_hip_frames: int, correct_knee_frames: int, total_frames: int) -> str:
        """Advance the repetition counter, compute the quality, and return the feedback message."""
        self.repetitions += 1
        
        if total_frames > 0:
            hip_score = (correct_hip_frames / total_frames) * 60.0
            knee_score = (correct_knee_frames / total_frames) * 40.0
            calculated_quality = int(hip_score + knee_score)
            self.quality = max(self.minimum_quality, min(100, calculated_quality))
        else:
            self.quality = 100

        # Tworzymy oficjalny komunikat dla tego powtórzenia
        self.feedback = f"Powtórzenie {self.repetitions} | Jakość: {self.quality}%"
        return self.feedback

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