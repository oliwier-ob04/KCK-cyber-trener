"""Hip thrust exercise profile used by the current prototype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechniqueRow:
    """A single technique hint row shown in the feedback card."""

    label: str
    value: str
    color: str


@dataclass(frozen=True)
class ExerciseProfile:
    """Static content describing one supported exercise."""

    title: str
    description: str
    tips: tuple[str, ...]
    technique_rows: tuple[TechniqueRow, ...]
    default_feedback: str


def build_hip_thrust_exercise() -> ExerciseProfile:
    """Build the default hip thrust exercise profile."""

    return ExerciseProfile(
        title="Podglad z kamer + warstwa AR",
        description=(
            "To jest dzialajacy panel pod szkielet aplikacji. Jesli dostepne jest OpenCV, "
            "aplikacja pokazuje realny obraz z kamer. Gdy podlaczona jest tylko jedna kamera, "
            "drugi panel pozostaje czarny."
        ),
        tips=(
            "• utrzymaj miednice w linii",
            "• pelny wyprost bez przeprostu",
            "• ruch bioder pionowo",
            "• jedna osoba w kadrze",
        ),
        technique_rows=(
            TechniqueRow("Kregoslup", "neutralny", "#22c55e"),
            TechniqueRow("Biodra", "w gornej pozycji", "#38bdf8"),
            TechniqueRow("Kolana", "stabilne", "#22c55e"),
            TechniqueRow("Tempo", "do dopracowania", "#f59e0b"),
        ),
        default_feedback="Gotowy do uruchomienia sesji.",
    )
