"""Application controller for Cyber Trener."""

from __future__ import annotations

import random
import threading
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from ar.renderer import FrameRenderer
from cameras.manager import CameraManager
from cameras.network import FrameRelayService
from exercises.hip_thrust import ExerciseProfile, build_hip_thrust_exercise
from pose.analyzer import MovementAnalyzer, PoseMetrics
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
        self.analyzers = {i: MovementAnalyzer() for i in range(self.config.max_camera_slots)}

        self.session_active = False
        self.session_paused = False
        self.session_started_at = 0.0
        self.last_tick = time.time()
        self.session_mode = "idle"  # idle -> arming -> active -> paused
        self.session_elapsed = 0.0
        self._rep_anchor_elapsed: float | None = None
        self._rep_count = 0
        self._tempo_samples: list[float] = []
        self._last_pose_zone = "unknown"
        self._latest_pose_metrics = PoseMetrics(False)
        self._last_camera_scan_at = 0.0
        self._remote_frame: Image.Image | None = None
        self._remote_frame_lock = threading.Lock()

        # --- Zmienne nowej maszyny stanów i kalkulatora jakości powtórzeń ---
        self._rep_state = "START"  # START -> GOING_UP -> TOP_HOLDING -> LOCK_REQUIRE_DOWN
        self._top_hold_started_at: float | None = None
        self._rep_started_elapsed = 0.0
        self._current_rep_frames_count = 0
        self._current_rep_hip_correct_frames = 0
        self._current_rep_knee_correct_frames = 0
        self._last_calculated_quality = 100

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
            camera_only=True,
        )
        StyleManager(self.view, self.config).apply()
        self.renderer = FrameRenderer(self.view, config=self.config)
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
        self.view.set_workout_status("Gotowy", "Naciśnij Start albo unieś rękę, aby rozpocząć ustawianie pozycji startowej.")
        self.view.set_pose_metrics(PoseMetrics(False))

    def _sync_score_state(self) -> None:
        """Keep the scorer object aligned with the controller counters."""

        self.scorer.repetitions = self._rep_count
        self.scorer.quality = max(self.config.minimum_quality, min(100, self._last_calculated_quality))

    def run(self) -> None:
        """Start background services and enter the Tk event loop."""

        self.camera.refresh_devices()
        for slot_index, device_id in enumerate(self.camera.available_devices):
            if slot_index < self.config.max_camera_slots:
                self.camera.switch_camera(slot_index, device_id)
        self.transport.start()
        self.camera.start()
        for i in range(50):
            any_frame_available = False
            for slot_idx in range(self.config.max_camera_slots):
                ok, frame, _ = self.camera.read(slot_idx)
                if ok and frame is not None:
                    any_frame_available = True
                    break
            if any_frame_available:
                break
            time.sleep(0.1)
        self.view.after(self.config.update_interval_ms, self._update_loop)
        self.view.mainloop()

    def start_session(self) -> None:
        """Start a new training session if one is not already active."""

        if self.session_mode in {"arming", "active", "paused"}:
            return

        self._arm_session(trigger="button")

    def _arm_session(self, trigger: str) -> None:
        """Prepare a fresh series and wait for the correct start pose."""

        self.session_mode = "arming"
        self.session_active = False
        self.session_paused = False
        self.session_started_at = 0.0
        self.session_elapsed = 0.0
        self.last_tick = time.time()
        self._rep_anchor_elapsed = None
        self._rep_count = 0
        self._tempo_samples = []
        self._last_pose_zone = "unknown"
        
        # Reset zmiennych nowej maszyny stanów przy starcie sesji
        self._rep_state = "START"
        self._top_hold_started_at = None
        self._rep_started_elapsed = 0.0
        self._current_rep_frames_count = 0
        self._current_rep_hip_correct_frames = 0
        self._current_rep_knee_correct_frames = 0
        self._last_calculated_quality = 100

        for analyzer in self.analyzers.values():
            analyzer.reset()
        self.scorer.reset()

        self.view.set_workout_status("Ustaw start", "Ustaw biodro-kolano-stopa w okolicach 90° i pokaż pozycję startową.")
        self.view.set_feedback("Ustaw pozycję startową. Kąt biodro-kolano-stopa ma być około 90°.")
        self.view.append_event(f"Seria przygotowywana ({trigger})")
        self._sync_score_state()

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

        if self.session_mode == "idle":
            self.start_session()
            return

        if self.session_mode == "arming":
            self.view.set_feedback("Najpierw ustaw pozycję startową.")
            return

        if self.session_mode == "active":
            self.session_mode = "paused"
            self.session_paused = True
            self.view.set_workout_status("Pauza", "Seria wstrzymana. Czas i tempo nie rosną w trakcie pauzy.")
            self.view.set_feedback("Seria wstrzymana.")
            self.view.append_event("Seria wstrzymana")
            return

        if self.session_mode == "paused":
            self.session_mode = "active"
            self.session_paused = False
            self.last_tick = time.time()
            self.view.set_workout_status("Aktywna", "Seria wznowiona.")
            self.view.set_feedback("Seria wznowiona.")
            self.view.append_event("Seria wznowiona")

    def end_session(self) -> None:
        """End the current session and persist the result automatically."""

        if self.session_mode == "idle":
            return

        self.session_mode = "idle"
        self.session_active = False
        self.session_paused = False
        self.view.set_connection_status("Kamera nieaktywna")
        self.view.set_workout_status("Zakończona", "Seria zakończona. Możesz rozpocząć nową próbę.")
        self.view.set_feedback("Seria zakończona.")
        self.view.append_event("Seria zakończona")
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

    def _update_camera_panels(self) -> PoseMetrics:
        """Render the active camera sources into the two preview panels."""

        best_metrics = PoseMetrics(False)
        for slot_index in range(self.config.max_camera_slots):
            panel = self.view.get_camera_panel(slot_index)
            area = getattr(panel, "inner_image_area", panel.holder)
            max_width = max(320, area.winfo_width() or 640)
            max_height = max(240, area.winfo_height() or 480)
            ok, frame, source_index = self.camera.read(slot_index)
            if ok and frame is not None:
                metrics, frame = self.analyzers[slot_index].analyze_frame(frame)
                if metrics.pose_detected and metrics.visibility >= best_metrics.visibility:
                    best_metrics = metrics
                
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

        self._latest_pose_metrics = best_metrics
        return best_metrics

    def _update_metrics(self, pose_metrics: PoseMetrics | None = None) -> None:
        """Advance the motion analysis and refresh the visible metrics."""

        pose_metrics = pose_metrics or self._latest_pose_metrics
        now = time.time()
        delta = now - self.last_tick
        self.last_tick = now

        if self.session_mode == "active":
            self.session_elapsed += delta

        self.view.set_pose_metrics(pose_metrics)

        # 1. Określenie poprawności kątów klatka po klatce (zgodnie z zielonym podświetleniem ze szkieletu)
        hip_correct = False
        knee_correct = False

        if pose_metrics.pose_detected:
            if pose_metrics.side == "front":
                # Przód: kolano zielone gdy różnica < 10.0 stopni
                knee_correct = (pose_metrics.knee_angle_front < 10.0)
                hip_correct = True  # z przodu nie mierzymy bocznego biodra, traktujemy jako prawidłowe
            else:
                # Bok: biodro zielone w pełnym wyproście 180 +/- 5 stopni
                hip_correct = (175.0 <= pose_metrics.upper_body_angle <= 185.0)
                # Bok: łydka zielona w pionie 90 +/- 5 stopni
                knee_correct = (85.0 <= pose_metrics.knee_angle_side <= 95.0)

        # Obsługa stanów sesji
        if self.session_mode == "idle":
            if pose_metrics.hand_raised:
                self._arm_session(trigger="gesture")
            else:
                self.view.set_workout_status("Gotowy", "Naciśnij Start albo unieś rękę, aby rozpocząć ustawianie pozycji startowej.")

        elif self.session_mode == "arming":
            self.view.set_workout_status("Ustaw start", pose_metrics.message)
            self.view.set_feedback(pose_metrics.message)
            
            # Przejście do treningu automatycznie po zejściu do pozycji niskiej (90-135 stopni w biodrach)
            if pose_metrics.pose_detected and pose_metrics.side != "front":
                if 90.0 <= pose_metrics.upper_body_angle <= 135.0:
                    self.session_mode = "active"
                    self.session_active = True
                    self.session_started_at = now
                    self.session_elapsed = 0.0
                    self.last_tick = now
                    self._rep_state = "START"
                    self.view.set_workout_status("Aktywna", "Rozpocznij wznos bioder.")
                    self.view.set_feedback("Trening aktywny!")

        elif self.session_mode == "paused":
            self.view.set_workout_status("Pauza", "Seria wstrzymana.")

        elif self.session_mode == "active":
            self.view.set_workout_status("Aktywna", pose_metrics.message)

            if pose_metrics.pose_detected and pose_metrics.side != "front":
                hip_angle = pose_metrics.upper_body_angle

                # --- MASZYNA STANÓW POWTÓRZENIA ---
                if self._rep_state == "START":
                    # Rozpoczynamy od pozycji opuszczonej (biodra 90 - 135 stopni)
                    if 90.0 <= hip_angle <= 135.0:
                        self._rep_state = "GOING_UP"
                        self._rep_started_elapsed = self.session_elapsed
                        self._current_rep_frames_count = 0
                        self._current_rep_hip_correct_frames = 0
                        self._current_rep_knee_correct_frames = 0

                elif self._rep_state == "GOING_UP":
                    # Ruch w górę: zliczamy ramki i ich poprawność
                    self._current_rep_frames_count += 1
                    if hip_correct: self._current_rep_hip_correct_frames += 1
                    if knee_correct: self._current_rep_knee_correct_frames += 1

                    # Czy osiągnięto pełny wyprost górny?
                    if 175.0 <= hip_angle <= 185.0:
                        self._rep_state = "TOP_HOLDING"
                        self._top_hold_started_at = now

                elif self._rep_state == "TOP_HOLDING":
                    self._current_rep_frames_count += 1
                    if hip_correct: self._current_rep_hip_correct_frames += 1
                    if knee_correct: self._current_rep_knee_correct_frames += 1

                    # Jeśli użytkownik spadnie z pozycji górnej przed upływem sekundy -> wraca do podnoszenia
                    if not (175.0 <= hip_angle <= 185.0):
                        self._rep_state = "GOING_UP"
                        self._top_hold_started_at = None
                    else:
                        # Weryfikacja utrzymania pozycji przez minimum 1 sekundę
                        if self._top_hold_started_at and (now - self._top_hold_started_at) >= 1.0:
                            self._rep_count += 1
                            rep_duration = max(0.01, self.session_elapsed - self._rep_started_elapsed)
                            self._tempo_samples.append(rep_duration)

                            # Wyliczanie oceny średniej z czasu powtórzenia (50% biodra, 50% kolana)
                            if self._current_rep_frames_count > 0:
                                hip_score = (self._current_rep_hip_correct_frames / self._current_rep_frames_count) * 50.0
                                knee_score = (self._current_rep_knee_correct_frames / self._current_rep_frames_count) * 50.0
                                self._last_calculated_quality = int(hip_score + knee_score)
                            else:
                                self._last_calculated_quality = 100

                            msg = f"Powtórzenie {self._rep_count} | Jakość: {self._last_calculated_quality}%"
                            self.scorer.feedback = msg
                            self.view.append_event(msg)
                            self.view.set_feedback(msg)

                            # Blokada: zaliczone, wymagamy powrotu na dół przed kolejnym powtórzeniem
                            self._rep_state = "LOCK_REQUIRE_DOWN"

                elif self._rep_state == "LOCK_REQUIRE_DOWN":
                    # Dopiero zejście bioder poniżej 135 stopni resetuje maszynę stanów do pozycji START
                    if hip_angle < 135.0:
                        self._rep_state = "START"
            else:
                # Kontynuacja zliczania klatek jako niepoprawne w przypadku chwilowej zguby sylwetki
                if self._rep_state in {"GOING_UP", "TOP_HOLDING"}:
                    self._current_rep_frames_count += 1

        # Wyznaczenie średnich statystyk tempa
        if self._tempo_samples:
            avg_tempo = sum(self._tempo_samples) / len(self._tempo_samples)
            current_tempo = self._tempo_samples[-1]
        else:
            avg_tempo = 0.0
            current_tempo = 0.0

        self.scorer.warnings = max(0, self.scorer.warnings)
        self.scorer.quality = max(self.config.minimum_quality, min(100, self._last_calculated_quality))
        self._sync_score_state()

        snapshot = self.scorer.snapshot()
        self.view.set_metrics(snapshot, self._format_elapsed())
        self.view.set_workout_counters(
            elapsed_text=self._format_elapsed(),
            current_tempo=f"{current_tempo:.2f} s" if current_tempo else "--",
            avg_tempo=f"{avg_tempo:.2f} s" if avg_tempo else "--",
            reps=self._rep_count,
        )

    def _format_elapsed(self) -> str:
        """Return the formatted elapsed session time."""

        if not self.session_started_at:
            return "00:00"
        elapsed = int(time.time() - self.session_started_at) if self.session_active else int(self.last_tick - self.session_started_at)
        return format_elapsed_seconds(elapsed)

    def _update_loop(self) -> None:
        """Run one UI tick and reschedule the next frame."""

        pose_metrics = self._update_camera_panels()
        self._update_metrics(pose_metrics)
        self.view.after(self.config.update_interval_ms, self._update_loop)

    def _on_close(self) -> None:
        """Stop background services and close the window cleanly."""

        self.transport.stop()
        self.camera.close()
        self.view.destroy()