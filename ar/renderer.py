"""Convert camera frames into Tk-compatible images and back-end payloads."""

from __future__ import annotations

from typing import Any

import tkinter as tk

from PIL import Image, ImageTk

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class FrameRenderer:
    """Render OpenCV and PIL images for the Tk UI."""

    def __init__(self, master: tk.Misc) -> None:
        """Keep a reference to the Tk root used for PhotoImage objects."""

        self.master = master

    def _cover_frame(self, frame: Any, max_width: int, max_height: int) -> Any:
        """Resize and crop a frame so it fills the available area."""

        height, width = frame.shape[:2]
        ratio = max(max_width / width, max_height / height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        crop_x = max(0, (new_size[0] - max_width) // 2)
        crop_y = max(0, (new_size[1] - max_height) // 2)
        return resized[crop_y : crop_y + max_height, crop_x : crop_x + max_width]

    def frame_to_photo(self, frame: Any, max_width: int, max_height: int) -> ImageTk.PhotoImage:
        """Convert an OpenCV BGR frame into a Tk photo image."""

        if cv2 is None:
            return self.black_photo(max_width, max_height)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = self._cover_frame(rgb, max_width, max_height)
        image = Image.fromarray(resized)
        return ImageTk.PhotoImage(image, master=self.master)

    def pil_to_photo(self, image: Image.Image, max_width: int, max_height: int) -> ImageTk.PhotoImage:
        """Convert a PIL image into a Tk photo image."""

        rgb = image.convert("RGB")
        width, height = rgb.size
        ratio = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
        resized = rgb.resize(new_size, resample)
        return ImageTk.PhotoImage(resized, master=self.master)

    def black_photo(self, width: int = 640, height: int = 480) -> ImageTk.PhotoImage:
        """Create the default offline placeholder image."""

        image = Image.new("RGB", (width, height), (2, 6, 15))
        return ImageTk.PhotoImage(image, master=self.master)

    @staticmethod
    def encode_frame_as_jpeg(frame: Any, quality: int = 10) -> bytes | None:
        """Encode an OpenCV frame as JPEG payload bytes."""

        if frame is None or cv2 is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded.tobytes() if ok else None
