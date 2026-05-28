"""Threaded IP camera implementation built on OpenCV streams."""

from __future__ import annotations

from typing import Any

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from cameras.base import BaseCamera


class IPCamera(BaseCamera):
    """Capture frames from an IP camera, RTSP stream or HTTP stream."""

    def __init__(
        self,
        stream_url: str,
        reconnect_interval: float = 1.5,
        frame_interval: float = 0.03,
    ) -> None:
        """Create a camera wrapper for the supplied stream URL."""

        super().__init__(name=f"ip-camera-{stream_url}", reconnect_interval=reconnect_interval, frame_interval=frame_interval)
        self.stream_url = stream_url

    def switch_camera(self, source: str) -> None:
        """Switch to a different stream URL and reconnect."""

        self.stream_url = source
        self._release_capture()

    def _open_capture(self) -> Any | None:
        """Open the network stream using OpenCV."""

        if cv2 is None or not self.stream_url:
            return None

        try:
            capture = cv2.VideoCapture(self.stream_url)
        except Exception:
            return None
        return capture
