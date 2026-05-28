"""Time formatting helpers."""

from __future__ import annotations


def format_elapsed_seconds(total_seconds: int) -> str:
    """Format elapsed seconds as MM:SS."""

    minutes, seconds = divmod(max(0, total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"
