"""Convert camera frames into Tk-compatible images and back-end payloads."""

from __future__ import annotations

from typing import Any

import tkinter as tk

from PIL import Image, ImageTk, ImageDraw

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class FrameRenderer:
    """Render OpenCV and PIL images for the Tk UI."""

    def __init__(self, master: tk.Misc, config=None) -> None:
        """Keep a reference to the Tk root used for PhotoImage objects and theme config."""

        self.master = master
        self.config = config

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
        return self._make_photo_with_rounded_corners(image, max_width, max_height)

    def pil_to_photo(self, image: Image.Image, max_width: int, max_height: int) -> ImageTk.PhotoImage:
        """Convert a PIL image into a Tk photo image."""

        rgb = image.convert("RGB")
        width, height = rgb.size
        # Use cover behavior: resize so image fills area, then crop center
        ratio = max(max_width / width, max_height / height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
        resized = rgb.resize(new_size, resample)

        # Crop center to exact target size
        left = max(0, (resized.width - max_width) // 2)
        top = max(0, (resized.height - max_height) // 2)
        right = left + max_width
        bottom = top + max_height
        cropped = resized.crop((left, top, right, bottom))
        return self._make_photo_with_rounded_corners(cropped, max_width, max_height)

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    def _make_photo_with_rounded_corners(self, image: Image.Image, width: int, height: int) -> ImageTk.PhotoImage:
        """Apply rounded-corner mask and composite onto background matching panel color."""
        # Determine corner radius from config or default
        radius = getattr(self.config, "camera_corner_radius", 16) if self.config else 16
        bg_hex = getattr(self.config, "panel_holder_bg", "#000000") if self.config else "#000000"
        bg_rgb = self._hex_to_rgb(bg_hex)

        base = Image.new("RGB", (width, height), bg_rgb)

        # Ensure image size equals target
        img = image.resize((width, height), getattr(Image, "LANCZOS", Image.BICUBIC))

        # Create mask with rounded rectangle
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)

        base.paste(img, (0, 0), mask)
        return ImageTk.PhotoImage(base, master=self.master)

    def black_photo(self, width: int = 640, height: int = 480) -> ImageTk.PhotoImage:
        """Create the default offline placeholder image."""
        bg_hex = getattr(self.config, "panel_holder_bg", "#000000") if self.config else "#000000"
        bg_rgb = self._hex_to_rgb(bg_hex)
        image = Image.new("RGB", (width, height), bg_rgb)
        # apply rounded corners as well
        return self._make_photo_with_rounded_corners(image, width, height)

    @staticmethod
    def encode_frame_as_jpeg(frame: Any, quality: int = 10) -> bytes | None:
        """Encode an OpenCV frame as JPEG payload bytes."""

        if frame is None or cv2 is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded.tobytes() if ok else None
