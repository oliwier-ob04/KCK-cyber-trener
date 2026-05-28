"""Device scanning and camera-slot management."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

if cv2 is not None:
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        elif hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(3)
    except Exception:
        pass

from cameras.base import BaseCamera
from cameras.ipcamera import IPCamera
from cameras.webcam import WebcamCamera


@dataclass(frozen=True)
class CameraSource:
    """Describe a camera slot source in a way the UI can render."""

    kind: Literal["webcam", "ip", "remote"]
    value: int | str | None = None
    label: str | None = None


class CameraManager:
    """Probe local cameras, manage slot assignments and read frames.

    The manager keeps one thread per physical camera source and exposes
    simple slot-based access for the Tk UI layer.
    """

    def __init__(
        self,
        scan_max_index: int = 2,
        slot_count: int = 2,
        remote_frame_provider: Callable[[], Image.Image | None] | None = None,
        auto_start: bool = False,
    ) -> None:
        """Store the device scan range and initialize the camera registry."""

        self.scan_max_index = scan_max_index
        self.slot_count = slot_count
        self.remote_frame_provider = remote_frame_provider
        self.auto_start = auto_start
        self._lock = threading.RLock()
        self.available_devices: list[int] = []
        self.slot_sources: list[CameraSource | None] = [None for _ in range(slot_count)]
        self._cameras: dict[str, BaseCamera] = {}
        self.refresh_devices()

    @property
    def available(self) -> bool:
        """Return True when OpenCV is available in the environment."""

        return cv2 is not None

    def start(self) -> None:
        """Start every managed threaded camera source."""

        with self._lock:
            for camera in self._cameras.values():
                camera.start()

    def stop(self) -> None:
        """Stop every managed threaded camera source."""

        with self._lock:
            for camera in self._cameras.values():
                camera.stop()

    def close(self) -> None:
        """Backward-compatible alias for stop()."""

        self.stop()

    def refresh_devices(self) -> list[int]:
        """Rescan the local webcam indices and refresh the slot defaults."""

        discovered = self.discover_webcams(self.scan_max_index)
        with self._lock:
            self.available_devices = discovered
            webcam_keys = {self._camera_key("webcam", index) for index in discovered}
            for key in list(self._cameras.keys()):
                if key.startswith("webcam:") and key not in webcam_keys:
                    self._cameras[key].stop()
                    del self._cameras[key]

            for slot_index, source in enumerate(self.slot_sources):
                if source is not None and source.kind == "webcam" and source.value is not None and int(source.value) not in self.available_devices:
                    self.slot_sources[slot_index] = None

        return self.available_devices

    @staticmethod
    def discover_webcams(max_index: int = 2) -> list[int]:
        """Probe the first N webcam indices and return the ones that open."""

        if cv2 is None:
            return []

        available_devices: list[int] = []
        for index in range(max_index):
            try:
                if WebcamCamera.probe_device(index):
                    available_devices.append(index)
            except Exception:
                continue
        return available_devices

    def register_ip_camera(self, label: str, stream_url: str) -> None:
        """Register an IP camera source and start it if auto-start is enabled."""

        with self._lock:
            key = self._camera_key("ip", label)
            camera = self._cameras.get(key)
            if camera is None:
                camera = IPCamera(stream_url)
                self._cameras[key] = camera
            else:
                camera.switch_camera(stream_url)
            if self.auto_start:
                camera.start()

    def register_remote_source(self, frame_provider: Callable[[], Image.Image | None]) -> None:
        """Register the optional remote frame provider used by the UI."""

        self.remote_frame_provider = frame_provider

    def set_slot_source(self, slot_index: int, source: CameraSource | None) -> None:
        """Assign a source to a preview slot."""

        self.switch_camera(slot_index, source)

    def switch_camera(self, slot_index: int, source: CameraSource | int | str | None) -> None:
        """Switch the specified slot to a new source.

        The source may be a CameraSource object, a webcam index, a raw URL
        for an IP camera or None.
        """

        if slot_index < 0 or slot_index >= len(self.slot_sources):
            return

        selection = self._normalize_source(source)
        with self._lock:
            previous = self.slot_sources[slot_index]
            if previous is not None and selection is not None and self._source_key(previous) == self._source_key(selection):
                return
            if previous is not None and self._is_source_unused(previous, exclude_slot=slot_index):
                self._stop_source(previous)
            self.slot_sources[slot_index] = selection
            if selection is not None and selection.kind in {"webcam", "ip"}:
                camera = self._resolve_camera(selection)
                if camera is not None and not camera.is_running:
                    camera.start()

    def format_source(self, source: CameraSource | None) -> str:
        """Convert a source object into the combobox label."""

        if source is None:
            return "Brak"
        if source.kind == "remote":
            return "Zdalna"
        if source.kind == "ip":
            return source.label or str(source.value or "IP")
        return f"Kamera {source.value}"

    def parse_source(self, label: str) -> CameraSource | None:
        """Convert a combobox label back into a source object."""

        if label == "Brak":
            return None
        if label == "Zdalna":
            return CameraSource(kind="remote", value="remote", label="Zdalna")
        if label.startswith("Kamera "):
            try:
                index = int(label.split(" ", 1)[1])
                return CameraSource(kind="webcam", value=index, label=f"Kamera {index}")
            except (ValueError, IndexError):
                return None
        if label.startswith("rtsp://") or label.startswith("http://") or label.startswith("https://"):
            return CameraSource(kind="ip", value=label, label=label)
        return None

    def source_options(self) -> list[str]:
        """Return all selectable labels for the camera comboboxes."""

        options = ["Brak"] + [f"Kamera {index}" for index in self.available_devices]
        if self.remote_frame_provider is not None:
            options.append("Zdalna")
        return options

    def slot_source_labels(self) -> list[str]:
        """Return the current slot assignments as combobox labels."""

        return [self.format_source(source) for source in self.slot_sources]

    def get_selected_frame(self, slot_index: int) -> tuple[bool, np.ndarray | None, str | int | None]:
        """Compatibility wrapper that returns the current frame for a slot."""

        return self.read(slot_index)

    def read_camera(self, device_index: int) -> tuple[bool, np.ndarray | None, int | None]:
        """Return the latest frame for a concrete webcam index."""

        with self._lock:
            camera = self._ensure_webcam(device_index)
            if camera is None:
                return False, None, device_index
            frame = camera.get_current_frame()
            return frame is not None, frame, device_index

    def read(self, slot_index: int) -> tuple[bool, np.ndarray | None, str | int | None]:
        """Read from the source assigned to a preview slot."""

        if slot_index < 0 or slot_index >= len(self.slot_sources):
            return False, None, None

        with self._lock:
            source = self.slot_sources[slot_index]
        if source is None:
            return False, None, None

        if source.kind == "remote":
            if self.remote_frame_provider is None or cv2 is None:
                return False, None, "remote"
            remote_frame = self.remote_frame_provider()
            if remote_frame is None:
                return False, None, "remote"
            frame = cv2.cvtColor(np.array(remote_frame), cv2.COLOR_RGB2BGR)
            return True, frame, "remote"

        if source.kind == "ip":
            camera = self._resolve_camera(source)
            if camera is None:
                return False, None, source.value or source.label or "ip"
            frame = camera.get_current_frame()
            return frame is not None, frame, source.value or source.label or "ip"

        index = int(source.value or 0)
        return self.read_camera(index)

    def close(self) -> None:
        """Release every cached OpenCV capture."""

        self.stop()

    def _ensure_webcam(self, device_index: int) -> WebcamCamera | None:
        """Return the webcam object for a device index, creating it on demand."""

        key = self._camera_key("webcam", device_index)
        camera = self._cameras.get(key)
        if camera is not None:
            return camera if isinstance(camera, WebcamCamera) else None
        if cv2 is None:
            return None
        camera = WebcamCamera(device_index)
        self._cameras[key] = camera
        return camera

    def _resolve_camera(self, source: CameraSource) -> BaseCamera | None:
        """Translate a source selection into a managed camera object."""

        if source.kind == "webcam":
            index = int(source.value or 0)
            return self._ensure_webcam(index)
        if source.kind == "ip":
            key = self._camera_key("ip", source.value or source.label or "ip")
            camera = self._cameras.get(key)
            if camera is None and isinstance(source.value, str):
                camera = IPCamera(source.value)
                self._cameras[key] = camera
            return camera
        return None

    def _normalize_source(self, source: CameraSource | int | str | None) -> CameraSource | None:
        """Normalize external source formats into CameraSource objects."""

        if source is None:
            return None
        if isinstance(source, CameraSource):
            return source
        if isinstance(source, int):
            return CameraSource(kind="webcam", value=source, label=f"Kamera {source}")
        if source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"):
            return CameraSource(kind="ip", value=source, label=source)
        if source == "Zdalna":
            return CameraSource(kind="remote", value="remote", label="Zdalna")
        return None

    def _source_key(self, source: CameraSource) -> str:
        """Return the registry key for a normalized source."""

        if source.value is not None:
            return self._camera_key(source.kind, source.value)
        if source.label is not None:
            return self._camera_key(source.kind, source.label)
        return self._camera_key(source.kind, "unknown")

    def _is_source_unused(self, source: CameraSource, exclude_slot: int | None = None) -> bool:
        """Check whether a source is still referenced by any other slot."""

        source_key = self._source_key(source)
        for slot_index, slot_source in enumerate(self.slot_sources):
            if exclude_slot is not None and slot_index == exclude_slot:
                continue
            if slot_source is not None and self._source_key(slot_source) == source_key:
                return False
        return True

    def _stop_source(self, source: CameraSource) -> None:
        """Stop and remove a source that is no longer referenced."""

        if source.kind not in {"webcam", "ip"}:
            return
        key = self._source_key(source)
        camera = self._cameras.pop(key, None)
        if camera is not None:
            camera.stop()

    @staticmethod
    def _camera_key(kind: str, value: int | str) -> str:
        """Build a stable registry key for a source."""

        return f"{kind}:{value}"
