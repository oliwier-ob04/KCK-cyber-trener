"""Application configuration and shared constants."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root directory."""

    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the Cyber Trener application."""

    app_name: str = "Cyber-Trener"
    window_title: str = "Cyber-Trener"
    window_state: str = "zoomed"
    background_color: str = "#0b1220"
    sidebar_color: str = "#0f172a"
    card_color: str = "#111c33"
    # Extended theme palette (used by UI styling)
    header_bg: str = "#071326"
    panel_wrapper_bg: str = "#08111f"
    panel_holder_bg: str = "#111c33"
    chip_bg: str = "#1e293b"
    list_bg: str = "#0b1220"
    text_primary: str = "#f8fafc"
    text_secondary: str = "#cbd5e1"
    text_muted: str = "#94a3b8"
    accent_cyan: str = "#06b6d4"
    accent_green: str = "#22c55e"
    accent_orange: str = "#f59e0b"
    danger_color: str = "#ef4444"
    # Camera preview corner radius in pixels
    camera_corner_radius: int = 16
    # Thickness of top/bottom bars above/below camera container
    camera_edge_bar_thickness: int = 16
    remote_peer_host: str = "10.218.165.145"
    remote_peer_port: int = 5001
    listen_host: str = "0.0.0.0"
    listen_port: int = 5000
    send_interval_seconds: float = 0.016
    update_interval_ms: int = 33
    camera_scan_max_index: int = 2
    default_quality: int = 91
    minimum_quality: int = 60
    front_tolerance_degrees: float = 10.0
    side_tolerance_degrees: float = 10.0
    side_back_tolerance_degrees: float = 10.0
    history_limit: int = 8
    max_camera_slots: int = 2
    history_file: Path = field(default_factory=lambda: _project_root() / "cyber_trener_history.json")
    settings_file: Path = field(default_factory=lambda: _project_root() / "cyber_trener_settings.json")


def _load_json_settings(path: Path) -> dict[str, float]:
    """Load persisted tolerance values from disk."""

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    settings: dict[str, float] = {}
    for key in ("front_tolerance_degrees", "side_tolerance_degrees", "side_back_tolerance_degrees"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            settings[key] = max(1.0, min(45.0, float(value)))
    return settings


def save_persisted_settings(path: Path, config: AppConfig) -> None:
    """Persist the current tolerance values so the next run restores them."""

    payload = {
        "front_tolerance_degrees": config.front_tolerance_degrees,
        "side_tolerance_degrees": config.side_tolerance_degrees,
        "side_back_tolerance_degrees": config.side_back_tolerance_degrees,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_default_config() -> AppConfig:
    """Create the default application configuration."""

    base = AppConfig()
    loaded = _load_json_settings(base.settings_file)
    if not loaded:
        return base

    return AppConfig(
        app_name=base.app_name,
        window_title=base.window_title,
        window_state=base.window_state,
        background_color=base.background_color,
        sidebar_color=base.sidebar_color,
        card_color=base.card_color,
        header_bg=base.header_bg,
        panel_wrapper_bg=base.panel_wrapper_bg,
        panel_holder_bg=base.panel_holder_bg,
        chip_bg=base.chip_bg,
        list_bg=base.list_bg,
        text_primary=base.text_primary,
        text_secondary=base.text_secondary,
        text_muted=base.text_muted,
        accent_cyan=base.accent_cyan,
        accent_green=base.accent_green,
        accent_orange=base.accent_orange,
        danger_color=base.danger_color,
        camera_corner_radius=base.camera_corner_radius,
        camera_edge_bar_thickness=base.camera_edge_bar_thickness,
        remote_peer_host=base.remote_peer_host,
        remote_peer_port=base.remote_peer_port,
        listen_host=base.listen_host,
        listen_port=base.listen_port,
        send_interval_seconds=base.send_interval_seconds,
        update_interval_ms=base.update_interval_ms,
        camera_scan_max_index=base.camera_scan_max_index,
        default_quality=base.default_quality,
        minimum_quality=base.minimum_quality,
        front_tolerance_degrees=loaded.get("front_tolerance_degrees", base.front_tolerance_degrees),
        side_tolerance_degrees=loaded.get("side_tolerance_degrees", base.side_tolerance_degrees),
        side_back_tolerance_degrees=loaded.get("side_back_tolerance_degrees", base.side_back_tolerance_degrees),
        history_limit=base.history_limit,
        max_camera_slots=base.max_camera_slots,
        history_file=base.history_file,
        settings_file=base.settings_file,
    )
