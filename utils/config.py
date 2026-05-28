"""Application configuration and shared constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the Cyber Trener application."""

    app_name: str = "Cyber Trener"
    window_title: str = "Cyber Trener - dzialajacy prototyp"
    window_state: str = "zoomed"
    background_color: str = "#0b1220"
    sidebar_color: str = "#0f172a"
    card_color: str = "#111c33"
    remote_peer_host: str = "10.218.165.145"
    remote_peer_port: int = 5001
    listen_host: str = "0.0.0.0"
    listen_port: int = 5000
    send_interval_seconds: float = 0.016
    update_interval_ms: int = 33
    camera_scan_max_index: int = 2
    default_quality: int = 91
    minimum_quality: int = 60
    history_limit: int = 8
    max_camera_slots: int = 2
    history_file: Path = field(default_factory=lambda: _project_root() / "cyber_trener_history.json")


def build_default_config() -> AppConfig:
    """Create the default application configuration."""

    return AppConfig()
