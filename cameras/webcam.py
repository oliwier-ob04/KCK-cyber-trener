"""Threaded webcam camera implementation built on OpenCV."""

from __future__ import annotations

from typing import Any

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from cameras.base import BaseCamera


class WebcamCamera(BaseCamera):
    """Capture frames from a local webcam device index."""

    def __init__(
        self,
        device_index: int,
        reconnect_interval: float = 1.0,
        frame_interval: float = 0.02,
        target_width: int = 1280,
        target_height: int = 720,
        buffer_size: int = 1,
    ) -> None:
        """Create a webcam source wrapper for the given device index."""

        super().__init__(name=f"webcam-{device_index}", reconnect_interval=reconnect_interval, frame_interval=frame_interval)
        self.device_index = device_index
        self.target_width = target_width
        self.target_height = target_height
        self.buffer_size = buffer_size
        self._backend_order: tuple[int | None, ...] = self._build_backend_order()

    def switch_camera(self, source: int) -> None:
        """Switch to a different webcam index and reconnect."""

        self.device_index = int(source)
        self._release_capture()

    def _open_capture(self) -> Any | None:
        """Open the webcam using the Windows-friendly backend when available."""

        if cv2 is None:
            return None

        for backend in self._backend_order:
            capture = None
            try:
                if backend is None:
                    capture = cv2.VideoCapture(self.device_index)
                else:
                    capture = cv2.VideoCapture(self.device_index, backend)
            except Exception:
                capture = None

            if capture is None:
                continue

            try:
                if not capture.isOpened():
                    capture.release()
                    continue

                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                except Exception:
                    pass
                try:
                    capture.set(cv2.CAP_PROP_FPS, 30)
                except Exception:
                    pass
                return capture
            except Exception:
                try:
                    capture.release()
                except Exception:
                    pass
                continue

        return None

    @classmethod
    def probe_device(cls, device_index: int) -> bool:
        """Return True when a device index can be opened by OpenCV."""

        camera = cls(device_index)
        capture = camera._open_capture()
        if capture is None:
            return False
        try:
            return True
        finally:
            try:
                capture.release()
            except Exception:
                pass

    @staticmethod
    def _build_backend_order() -> tuple[int | None, ...]:
        """Return the preferred backend order for Windows webcams."""

        if cv2 is None:
            return (None,)

        candidates: list[int | None] = []
        for backend_name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            backend = getattr(cv2, backend_name, None)
            if isinstance(backend, int) and backend not in candidates:
                candidates.append(backend)
        candidates.append(None)
        return tuple(candidates)
