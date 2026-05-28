"""Application controller for Cyber Trener."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from ar.renderer import FrameRenderer
from cameras.manager import CameraManager
from cameras.network import FrameRelayService
from exercises.hip_thrust import ExerciseProfile, build_hip_thrust_exercise
from pose.analyzer import MovementAnalyzer
from scoring.engine import ScoreSnapshot, SessionScorer
from ui.styles import StyleManager
from ui.view import CyberTrainerView, ViewCallbacks
from utils.config import AppConfig, build_default_config
from utils.session_store import SessionResult, SessionStore
from utils.time_utils import format_elapsed_seconds


@dataclass(frozen=True)
class HistoryEntry:
    """Formatted session row shown in the history list."""

    repetitions: int
    duration_seconds: int
    quality: int
    source: str

    def format(self) -> str:
        """Render the history entry as a compact one-line label."""

        return (
            f"{self.repetitions:02d} powt. | {self.duration_seconds}s | "
            f"{self.quality}% | {self.source}"
        )


class CyberTrainerApp:
    """Wire the view, camera stack, scoring engine and transport layer together."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Build the application with injectable defaults."""

        self.config = config or build_default_config()
        self.exercise: ExerciseProfile = build_hip_thrust_exercise()
        self.store = SessionStore(self.config.history_file)
        self._rng = random.Random()
        self.scorer = SessionScorer(
            rng=self._rng,
            initial_quality=self.config.default_quality,
            minimum_quality=self.config.minimum_quality,
            feedback=self.exercise.default_feedback,
        )
        self.analyzer = MovementAnalyzer()

        self.session_active = False
        self.session_paused = False
        self.session_started_at = 0.0
        self.last_tick = time.time()
        self._last_camera_scan_at = 0.0
        self._remote_frame: Image.Image | None = None
        self._remote_frame_lock = threading.Lock()

        self.view = CyberTrainerView(
            config=self.config,
            exercise=self.exercise,
            callbacks=ViewCallbacks(
                on_start_session=self.start_session,
                on_toggle_pause=self.toggle_pause,
                on_end_session=self.end_session,
                on_save_result=self.save_result,
                on_close=self._on_close,
                on_camera_source_changed=self._on_camera_source_changed,
            ),
        )
        StyleManager(self.view, self.config).apply()
        self.renderer = FrameRenderer(self.view)
        self.camera = CameraManager(
            scan_max_index=self.config.camera_scan_max_index,
            slot_count=self.config.max_camera_slots,
            remote_frame_provider=self._get_remote_frame,
        )
        self.transport = FrameRelayService(
            peer_host=self.config.remote_peer_host,
            peer_port=self.config.remote_peer_port,
            listen_host=self.config.listen_host,
            listen_port=self.config.listen_port,
            send_interval_seconds=self.config.send_interval_seconds,
            frame_provider=lambda: self.camera.read(0),
            on_frame_received=self._store_remote_frame,
        )

        self._load_history()
        self._refresh_camera_sources(force=True)
        self._update_camera_status()
        self.view.set_feedback(self.exercise.default_feedback)
        self.view.set_metrics(self.scorer.snapshot(), self._format_elapsed())

    def run(self) -> None:
        """Start background services and enter the Tk event loop."""

        self.transport.start()
        self.view.after(self.config.update_interval_ms, self._update_loop)
        self.view.mainloop()

    def start_session(self) -> None:
        """Start a new training session if one is not already active."""

        if self.session_active:
            return

        self.session_active = True
        self.session_paused = False
        self.session_started_at = time.time()
        self.last_tick = self.session_started_at
        self.analyzer.reset()
        self.scorer.reset()

        self._update_camera_status()
        active_cameras = len(self.camera.available_devices)
        if active_cameras == 0:
            self.view.set_feedback(
                "Sesja uruchomiona bez aktywnej kamery. Panele pozostana czarne do czasu wykrycia urzadzenia."
            )
        elif active_cameras == 1:
            self.view.set_feedback("Sesja uruchomiona. Jedna kamera jest aktywna, drugi panel pozostanie czarny.")
        else:
            self.view.set_feedback("Sesja uruchomiona. Obraz jest pobierany z kamer w czasie rzeczywistym.")

        self.view.append_event("Sesja rozpoceta")

    def toggle_pause(self) -> None:
        """Toggle the active session between paused and running."""

        if not self.session_active:
            self.view.set_feedback("Najpierw uruchom sesje.")
            return

        self.session_paused = not self.session_paused
        state = "wznowiona" if not self.session_paused else "wstrzymana"
        self.view.set_feedback(f"Sesja {state}.")
        self.view.append_event(f"Sesja {state}")

    def end_session(self) -> None:
        """End the current session and persist the result automatically."""

        if not self.session_active:
            return

        self.session_active = False
        self.session_paused = False
        self.view.set_connection_status("Kamera nieaktywna")
        self.view.set_feedback("Sesja zakonczona. Mozesz zapisac wynik lub rozpoczac nowa probe.")
        self.view.append_event("Sesja zakonczona")
        self.save_result(auto=True)

    def save_result(self, auto: bool = False) -> None:
        """Persist the current session snapshot in the local JSON history."""

        if not self.session_started_at:
            return

        now = time.time()
        snapshot = self.scorer.snapshot()
        result = SessionResult(
            started_at=self.session_started_at,
            ended_at=now,
            repetitions=snapshot.repetitions,
            warnings=snapshot.warnings,
            quality=snapshot.quality,
            source=self.view.source_label.get(),
        )
        try:
            self.store.append(result)
            self._load_history()
            if not auto:
                self.view.append_event("Wynik zapisany lokalnie")
                self.view.set_feedback("Wynik zapisany do lokalnej historii treningow.")
        except Exception as exc:
            self.view.append_event(f"Blad zapisu: {exc}")
            self.view.set_feedback("Nie udalo sie zapisac wyniku.")

    def _on_camera_source_changed(self, slot_index: int, source_label: str) -> None:
        """Update the camera slot assignment after a combobox change."""

        self.camera.set_slot_source(slot_index, self.camera.parse_source(source_label))
        self.view.set_camera_sources(
            self.camera.source_options(),
            self.camera.slot_source_labels(),
        )

    def _store_remote_frame(self, image: Image.Image) -> None:
        """Cache the latest remote image received over the network."""

        with self._remote_frame_lock:
            self._remote_frame = image

    def _get_remote_frame(self) -> Image.Image | None:
        """Return the latest network image for the remote camera source."""

        with self._remote_frame_lock:
            return self._remote_frame

    def _load_history(self) -> None:
        """Load the local history file and refresh the sidebar list."""

        items = self.store.load()
        entries = [
            HistoryEntry(
                repetitions=int(item.get("repetitions", 0)),
                duration_seconds=int(item.get("duration_seconds", 0)),
                quality=int(item.get("quality", 0)),
                source=str(item.get("source", "unknown")),
            ).format()
            for item in items[: self.config.history_limit]
        ]
        self.view.set_history(entries)

    def _refresh_camera_sources(self, force: bool = False) -> None:
        """Rescan local devices and keep the UI comboboxes in sync."""

        now = time.time()
        if not force and self.camera.available_devices:
            return

        self._last_camera_scan_at = now
        self.camera.refresh_devices()
        self.view.set_camera_sources(
            self.camera.source_options(),
            self.camera.slot_source_labels(),
        )

    def _update_camera_status(self) -> None:
        """Update the sidebar status text based on available devices."""

        if not self.camera.available:
            self.view.set_source_label("Demo mode")
            self.view.set_connection_status("OpenCV nie jest dostepne - wyswietlany jest czarny ekran")
            return

        active = len(self.camera.available_devices)
        if active == 0:
            self.view.set_source_label("Demo mode")
            self.view.set_connection_status("Nie wykryto zadnej kamery - oba panele pozostaja czarne")
        elif active == 1:
            self.view.set_source_label("1 camera")
            self.view.set_connection_status("Wykryto jedna kamere - drugi panel pozostaje czarny")
        else:
            self.view.set_source_label(f"{active} cameras")
            self.view.set_connection_status("Wykryto co najmniej dwie kamery")

    def _update_camera_panels(self) -> None:
        """Render the active camera sources into the two preview panels."""

        for slot_index in range(self.config.max_camera_slots):
            panel = self.view.get_camera_panel(slot_index)
            max_width = max(320, panel.holder.winfo_width() or 640)
            max_height = max(240, panel.holder.winfo_height() or 480)
            ok, frame, source_index = self.camera.read(slot_index)
            if ok and frame is not None:
                photo = self.renderer.frame_to_photo(frame, max_width, max_height)
                self.view.update_camera_panel(
                    slot_index=slot_index,
                    photo=photo,
                    status_text=f"kamera {source_index}",
                    online=True,
                )
            else:
                photo = self.renderer.black_photo(max_width, max_height)
                self.view.update_camera_panel(
                    slot_index=slot_index,
                    photo=photo,
                    status_text="brak sygnalu",
                    online=False,
                )

    def _update_metrics(self) -> None:
        """Advance the motion analysis and refresh the visible metrics."""

        if self.session_active and not self.session_paused:
            now = time.time()
            delta = now - self.last_tick
            self.last_tick = now
            movement = self.analyzer.step(delta)
            snapshot = self.scorer.update(movement)
            if movement.repetition_detected:
                self.view.append_event(f"Wykryto powtorzenie {snapshot.repetitions}")
            if snapshot.warning_event:
                self.view.append_event("Ostrzezenie o technice")
                self.view.set_feedback(snapshot.feedback)
        else:
            snapshot = self.scorer.snapshot()

        self.view.set_metrics(snapshot, self._format_elapsed())

        if snapshot.repetitions and snapshot.repetitions % 5 == 0:
            self.view.set_feedback("Dobra praca: utrzymano poprawny zakres ruchu.")

    def _format_elapsed(self) -> str:
        """Return the formatted elapsed session time."""

        if not self.session_started_at:
            return "00:00"
        elapsed = int(time.time() - self.session_started_at) if self.session_active else int(self.last_tick - self.session_started_at)
        return format_elapsed_seconds(elapsed)

    def _update_loop(self) -> None:
        """Run one UI tick and reschedule the next frame."""

        self._update_metrics()
        self._update_camera_panels()
        self.view.after(self.config.update_interval_ms, self._update_loop)

    def _on_close(self) -> None:
        """Stop background services and close the window cleanly."""

        self.transport.stop()
        self.camera.close()
        self.view.destroy()
