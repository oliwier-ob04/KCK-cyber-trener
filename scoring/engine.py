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
        
    def grade_hip_going_up(self, hip_angle: float) -> bool:
        """Ocena klatki bioder podczas ruchu w górę."""
        import math
        if math.isnan(hip_angle):
            return False
        # Podczas ruchu w górę akceptujemy szerszy zakres klatek roboczych
        return 90.0 <= hip_angle <= 190.0

    def grade_hip_holding(self, hip_angle: float) -> bool:
        """Ocena klatki bioder podczas utrzymania na górze (blokada)."""
        import math
        if math.isnan(hip_angle):
            return False
        # Ścisły wyprost w fazie holding
        return 170.0 <= hip_angle <= 190.0

    def grade_knee_going_up(self, knee_front: float, knee_side: float) -> bool:
        """Ocena klatki kolan (przód i bok) podczas ruchu w górę."""
        import math
        if math.isnan(knee_front) or math.isnan(knee_side):
            return False
        # Przykładowe kryteria stabilizacji: przód stabilny, bok zgina się odpowiednio do fazy
        # Wstaw tutaj swoje idealne widełki matematyczne
        return (160.0 <= knee_front <= 190.0) and (80.0 <= knee_side <= 140.0)

    def grade_knee_holding(self, knee_front: float, knee_side: float) -> bool:
        """Ocena klatki kolan (przód i bok) podczas utrzymania na górze."""
        import math
        if math.isnan(knee_front) or math.isnan(knee_side):
            return False
        # Na górze kolana powinny być w pełnej stabilizacji kątowej
        # Wstaw tutaj swoje idealne widełki matematyczne
        return (170.0 <= knee_front <= 190.0) and (160.0 <= knee_side <= 190.0)

    def register_repetition(self, correct_hip_frames: int, correct_knee_frames: int, total_frames: int) -> str:
        """Advance the repetition counter, compute the quality, and return the feedback message."""
        self.repetitions += 1
        
        if total_frames > 0:
            hip_score = (correct_hip_frames / total_frames) * 50.0
            knee_score = (correct_knee_frames / total_frames) * 50.0
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