"""Standalone webcam sender for Cyber Trener."""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import cv2
from PIL import Image, ImageOps, ImageTk


@dataclass(frozen=True)
class SenderSettings:
    """Connection and capture settings for the remote camera sender."""

    target_host: str
    target_port: int
    camera_index: int
    fps: float
    jpeg_quality: int
    reconnect_delay: float = 1.0


class CameraSender:
    """Capture frames from a webcam and stream them to Cyber Trener."""

    def __init__(self, settings: SenderSettings) -> None:
        self.settings = settings
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status = "Gotowy do uruchomienia."

    def start(self) -> None:
        """Start the capture and transmit loop."""

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture and transmit loop."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        self._set_status("Zatrzymano.")

    def latest_frame(self):
        """Return a copy of the most recent camera frame."""

        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def status(self) -> str:
        """Return the current connection status."""

        with self._status_lock:
            return self._status

    def _set_status(self, text: str) -> None:
        with self._status_lock:
            self._status = text

    def _set_latest_frame(self, frame) -> None:
        with self._frame_lock:
            self._latest_frame = frame.copy()

    @staticmethod
    def _close_socket(sock: socket.socket | None) -> None:
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass

    def _run(self) -> None:
        backend = getattr(cv2, "CAP_DSHOW", 0)
        capture = cv2.VideoCapture(self.settings.camera_index, backend) if backend else cv2.VideoCapture(self.settings.camera_index)
        if not capture.isOpened():
            self._set_status(f"Nie mogę otworzyć kamery {self.settings.camera_index}.")
            capture.release()
            return

        socket_client: socket.socket | None = None
        next_connect_attempt = 0.0
        frame_interval = 1.0 / max(1.0, float(self.settings.fps))
        host = self.settings.target_host.strip()
        port = int(self.settings.target_port)

        try:
            while not self._stop_event.is_set():
                loop_started = time.time()
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._set_status("Brak obrazu z kamery.")
                    self._close_socket(socket_client)
                    socket_client = None
                    sleep_for = frame_interval
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue

                self._set_latest_frame(frame)

                if socket_client is None and host and port > 0 and loop_started >= next_connect_attempt:
                    try:
                        socket_client = socket.create_connection((host, port), timeout=2.0)
                        socket_client.settimeout(1.0)
                        self._set_status(f"Połączono z {host}:{port}.")
                    except OSError:
                        self._set_status(f"Brak połączenia z {host}:{port}.")
                        next_connect_attempt = loop_started + self.settings.reconnect_delay

                if socket_client is not None:
                    payload = self._encode_frame_as_jpeg(frame, quality=self.settings.jpeg_quality)
                    if payload is None:
                        self._set_status("Nie udało się zakodować klatki.")
                    else:
                        try:
                            socket_client.sendall(struct.pack("!I", len(payload)))
                            socket_client.sendall(payload)
                        except OSError:
                            self._close_socket(socket_client)
                            socket_client = None
                            next_connect_attempt = loop_started + self.settings.reconnect_delay
                            self._set_status(f"Utracono połączenie z {host}:{port}.")

                sleep_for = frame_interval - (time.time() - loop_started)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            capture.release()
            self._close_socket(socket_client)

    @staticmethod
    def _encode_frame_as_jpeg(frame, quality: int = 70) -> bytes | None:
        if frame is None:
            return None
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, max(1, min(95, int(quality)))],
        )
        return encoded.tobytes() if ok else None


class TrenerCameraApp:
    """Small GUI for the sender computer."""

    def __init__(self, root: tk.Tk, defaults: SenderSettings) -> None:
        self.root = root
        self.root.title("TrenerCamera")
        self.root.configure(bg="#0b1220")
        self.root.minsize(980, 680)

        self.host_var = tk.StringVar(value=defaults.target_host)
        self.port_var = tk.StringVar(value=str(defaults.target_port))
        self.camera_var = tk.StringVar(value=str(defaults.camera_index))
        self.fps_var = tk.StringVar(value=str(defaults.fps))
        self.quality_var = tk.StringVar(value=str(defaults.jpeg_quality))
        self.status_var = tk.StringVar(value="Ustaw adres cyber-trenera i kliknij Start.")
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._sender: CameraSender | None = None
        self._preview_size = (840, 472)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._refresh_ui)

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg="#0b1220")
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        header = tk.Frame(outer, bg="#0f172a")
        header.pack(fill="x", pady=(0, 14))
        tk.Label(
            header,
            text="TrenerCamera",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 2))
        tk.Label(
            header,
            text="Wysyła obraz z kamerki do cyber-trenera i pokazuje podgląd tego samego strumienia.",
            bg="#0f172a",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=18, pady=(0, 16))

        body = tk.Frame(outer, bg="#0b1220")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        settings_card = tk.Frame(body, bg="#111c33", padx=16, pady=16)
        settings_card.grid(row=0, column=0, sticky="nsw", padx=(0, 14))

        preview_card = tk.Frame(body, bg="#111c33", padx=16, pady=16)
        preview_card.grid(row=0, column=1, sticky="nsew")
        preview_card.rowconfigure(1, weight=1)
        preview_card.columnconfigure(0, weight=1)

        tk.Label(settings_card, text="Połączenie", bg="#111c33", fg="#f8fafc", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self._entry_row(settings_card, "Host cyber-trenera", self.host_var)
        self._entry_row(settings_card, "Port", self.port_var)
        self._entry_row(settings_card, "Kamera", self.camera_var)
        self._entry_row(settings_card, "FPS", self.fps_var)
        self._entry_row(settings_card, "Jakość JPEG", self.quality_var)

        buttons = tk.Frame(settings_card, bg="#111c33")
        buttons.pack(fill="x", pady=(14, 0))
        tk.Button(buttons, text="Start", command=self.start_sender, bg="#22c55e", fg="#08111f", relief="flat", padx=14, pady=8).pack(fill="x")
        tk.Button(buttons, text="Stop", command=self.stop_sender, bg="#ef4444", fg="#ffffff", relief="flat", padx=14, pady=8).pack(fill="x", pady=(8, 0))
        tk.Label(
            settings_card,
            text="Połączenie używa tego samego formatu co cyber-trener: 4 bajty długości + JPEG.",
            bg="#111c33",
            fg="#94a3b8",
            justify="left",
            wraplength=260,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(14, 0))

        tk.Label(preview_card, text="Podgląd wysyłanego obrazu", bg="#111c33", fg="#f8fafc", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.preview_label = tk.Label(preview_card, text="Uruchom Start, aby zobaczyć kamerę.", bg="#08111f", fg="#cbd5e1", font=("Segoe UI", 11))
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(12, 12))

        status_bar = tk.Frame(preview_card, bg="#08111f")
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        tk.Label(status_bar, textvariable=self.status_var, bg="#08111f", fg="#cbd5e1", anchor="w", font=("Segoe UI", 10)).pack(fill="x", padx=12, pady=10)

    def _entry_row(self, parent: tk.Widget, label: str, variable: tk.StringVar) -> None:
        row = tk.Frame(parent, bg="#111c33")
        row.pack(fill="x", pady=(10, 0))
        tk.Label(row, text=label, bg="#111c33", fg="#cbd5e1", anchor="w", font=("Segoe UI", 10)).pack(fill="x")
        tk.Entry(row, textvariable=variable, bg="#0b1220", fg="#f8fafc", insertbackground="#f8fafc", relief="flat").pack(fill="x", pady=(4, 0))

    def _read_settings(self) -> SenderSettings | None:
        try:
            return SenderSettings(
                target_host=self.host_var.get().strip(),
                target_port=int(self.port_var.get().strip()),
                camera_index=int(self.camera_var.get().strip()),
                fps=max(1.0, float(self.fps_var.get().strip())),
                jpeg_quality=max(1, min(95, int(self.quality_var.get().strip()))),
            )
        except ValueError:
            self.status_var.set("Sprawdź wartości host/port/kamera/FPS/jakość.")
            return None

    def start_sender(self) -> None:
        """Start or restart the sender using the form values."""

        settings = self._read_settings()
        if settings is None:
            return

        self.stop_sender()
        self._sender = CameraSender(settings)
        self._sender.start()
        self.status_var.set("Uruchomiono nadawanie.")

    def stop_sender(self) -> None:
        """Stop the sender if it is running."""

        if self._sender is not None:
            self._sender.stop()
            self._sender = None

    def _refresh_ui(self) -> None:
        sender = self._sender
        if sender is not None:
            self.status_var.set(sender.status())
            frame = sender.latest_frame()
            if frame is not None:
                self._show_frame(frame)
            elif self._preview_photo is None:
                self.preview_label.configure(text="Czekam na obraz z kamery.", image="")
        else:
            if self._preview_photo is None:
                self.preview_label.configure(text="Uruchom Start, aby zobaczyć kamerę.", image="")

        self.root.after(80, self._refresh_ui)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = ImageOps.contain(image, self._preview_size, method=getattr(Image, "LANCZOS", Image.BICUBIC))
        preview = Image.new("RGB", self._preview_size, "#08111f")
        offset_x = max(0, (self._preview_size[0] - image.width) // 2)
        offset_y = max(0, (self._preview_size[1] - image.height) // 2)
        preview.paste(image, (offset_x, offset_y))
        self._preview_photo = ImageTk.PhotoImage(preview, master=self.root)
        self.preview_label.configure(image=self._preview_photo, text="")

    def _on_close(self) -> None:
        self.stop_sender()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stand-alone sender for Cyber Trener remote camera input.")
    parser.add_argument("--host", default="", help="IP or hostname of the cyber-trener computer.")
    parser.add_argument("--port", type=int, default=5000, help="TCP port used by cyber-trener (default: 5000).")
    parser.add_argument("--camera", type=int, default=0, help="Local webcam index to stream.")
    parser.add_argument("--fps", type=float, default=20.0, help="Frame rate for transmission.")
    parser.add_argument("--quality", type=int, default=65, help="JPEG quality for the stream.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = SenderSettings(
        target_host=args.host,
        target_port=args.port,
        camera_index=args.camera,
        fps=args.fps,
        jpeg_quality=args.quality,
    )

    root = tk.Tk()
    TrenerCameraApp(root, defaults)
    root.mainloop()


if __name__ == "__main__":
    main()
