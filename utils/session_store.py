"""JSON persistence for training sessions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionResult:
    """A single saved workout session."""

    started_at: float
    ended_at: float
    repetitions: int
    warnings: int
    quality: int
    source: str
    avg_quality: float | None = None
    avg_rep_time_seconds: float | None = None
    total_series_seconds: int | None = None

    @property
    def duration_seconds(self) -> int:
        """Return the session duration rounded down to seconds."""

        return max(0, int(self.ended_at - self.started_at))


class SessionStore:
    """Load and append workout results from a local JSON history file."""

    def __init__(self, path: Path) -> None:
        """Store the destination file path."""

        self.path = path

    def load(self) -> list[dict[str, Any]]:
        """Load the persisted results or return an empty list."""

        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def append(self, result: SessionResult) -> None:
        """Insert the newest session at the top of the history file."""

        items = self.load()
        payload = asdict(result) | {"duration_seconds": result.duration_seconds}
        items.insert(0, payload)
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
