"""Base camera abstractions for threaded frame acquisition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
import time
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class BaseCamera(ABC):
    """Base class for camera sources that run a dedicated acquisition thread."""

    def __init__(
        self,
        name: str,
        reconnect_interval: float = 1.0,
        frame_interval: float = 0.03,
    ) -> None:
        """Store shared thread and reconnect state."""

        self.name = name
        self.reconnect_interval = reconnect_interval
        self.frame_interval = frame_interval
        self._frame_lock = threading.RLock()
        self._current_frame: np.ndarray | None = None
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_error_at = 0.0
        self._read_retry_count = 5
        self._read_retry_delay = 0.05

    @property
    def is_running(self) -> bool:
        """Return True when the acquisition thread is active."""

        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def is_available(self) -> bool:
        """Return True when OpenCV is available in the current environment."""

        return cv2 is not None

    def start(self) -> None:
        """Start the acquisition thread if it is not already running."""

        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name=self.name, daemon=True)
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """Stop the acquisition thread and release the underlying capture handle."""

        self._stop_event.set()
        self._running = False
        self._release_capture()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_current_frame(self) -> np.ndarray | None:
        """Return a copy of the latest frame as a numpy array."""

        with self._frame_lock:
            if self._current_frame is None:
                return None
            return self._current_frame.copy()

    @abstractmethod
    def switch_camera(self, source: Any) -> None:
        """Change the underlying source and reconnect on the next capture cycle."""

    @abstractmethod
    def _open_capture(self) -> Any | None:
        """Open the underlying OpenCV capture object."""

    def _capture_loop(self) -> None:
        """Continuously read frames, reconnecting automatically after failures."""

        while not self._stop_event.is_set():
            if not self.is_available:
                time.sleep(self.reconnect_interval)
                continue

            capture = self._ensure_capture()
            if capture is None:
                self._mark_failure()
                time.sleep(self.reconnect_interval)
                continue

            delivered_frame = False
            for _attempt in range(self._read_retry_count):
                if self._stop_event.is_set():
                    break
                try:
                    ok, frame = capture.read()
                except Exception:
                    ok, frame = False, None

                if ok and frame is not None:
                    if frame.size > 0:
                        self._store_frame(frame)
                        delivered_frame = True
                        time.sleep(self.frame_interval)
                        break

                time.sleep(self._read_retry_delay)

            if delivered_frame:
                continue

            self._mark_failure()
            self._release_capture()
            time.sleep(self.reconnect_interval)

    def _ensure_capture(self) -> Any | None:
        """Return an open capture object, opening it if needed."""

        if self._capture is not None:
            try:
                if self._capture.isOpened():
                    return self._capture
            except Exception:
                pass
            self._release_capture()

        capture = self._open_capture()
        if capture is None:
            return None
        try:
            if not capture.isOpened():
                capture.release()
                return None
        except Exception:
            try:
                capture.release()
            except Exception:
                pass
            return None

        self._capture = capture
        return capture

    def _release_capture(self) -> None:
        """Release the current OpenCV capture if one exists."""

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def _store_frame(self, frame: np.ndarray) -> None:
        """Persist the latest frame in a thread-safe way."""

        with self._frame_lock:
            self._current_frame = frame.copy()

    def _mark_failure(self) -> None:
        """Record the latest capture failure timestamp."""

        self._last_error_at = time.time()
