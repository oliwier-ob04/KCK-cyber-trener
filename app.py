"""Application controller for Cyber Trener."""

from __future__ import annotations

import random
import threading
import time
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from PIL import Image

from ar.renderer import FrameRenderer
from cameras.manager import CameraManager
from cameras.network import FrameRelayService
from pose.analyzer import MovementAnalyzer, PoseMetrics
from scoring.engine import ScoreSnapshot, SessionScorer
from ui.styles import StyleManager
from ui.view import CyberTrainerView, ViewCallbacks
from utils.config import AppConfig, build_default_config, save_persisted_settings
from utils.session_store import SessionResult, SessionStore
from audio.voice_feedback import VoiceFeedback
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

    TOP_HOLD_SECONDS = 0.25

    def __init__(self, config: AppConfig | None = None) -> None:
        """Build the application with injectable defaults."""
        self.config = config or build_default_config()
        self.store = SessionStore(self.config.history_file)
        self._rng = random.Random()
        self.scorer = SessionScorer(
            rng=self._rng,
            initial_quality=self.config.default_quality,
            minimum_quality=self.config.minimum_quality,
            feedback="Sugestie",
        )
        self.front_tolerance_degrees = self.config.front_tolerance_degrees
        self.side_tolerance_degrees = self.config.side_tolerance_degrees
        self.side_back_tolerance_degrees = self.config.side_back_tolerance_degrees
        self.analyzers = {
            i: MovementAnalyzer(
                front_tolerance_degrees=self.front_tolerance_degrees,
                side_tolerance_degrees=self.side_tolerance_degrees,
                side_back_tolerance_degrees=self.side_back_tolerance_degrees,
            )
            for i in range(self.config.max_camera_slots)
        }

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
        self._gesture_toggle_block_until = 0.0
        self.voice = VoiceFeedback(min_interval_seconds=1.8, enabled=self.config.voice_feedback_enabled)
        self._last_instruction_at = 0.0

        # --- Zmienne maszyny stanów i kalkulatora jakości powtórzeń ---
        self._rep_state = "WAITING_TOP_START"  # WAITING_TOP_START -> WAITING_FOR_DESCENT -> GOING_DOWN_TO_BOTTOM -> GOING_UP_TO_TOP -> TOP_HOLDING
        self._bottom_reached_in_cycle = False
        self._top_hold_started_at: float | None = None
        self._rep_started_elapsed = 0.0
        self._current_rep_frames_count = 0
        self._current_rep_hip_correct_frames = 0
        self._current_rep_knee_correct_frames = 0
        self._last_calculated_quality = 100
        self._current_rep_issues: set[str] = set()
        self._top_hold_feedback_sent = False
        self._side_metrics: PoseMetrics | None = None
        self._front_metrics: PoseMetrics | None = None

        self.view = CyberTrainerView(
            config=self.config,
            callbacks=ViewCallbacks(
                on_start_session=self.start_session,
                on_primary_action=self.primary_action,
                on_toggle_pause=self.toggle_pause,
                on_end_session=self.end_session,
                on_save_result=self.save_result,
                on_close=self._on_close,
                on_camera_source_changed=self._on_camera_source_changed,
                on_angle_tolerance_changed=self._on_angle_tolerance_changed,
                on_voice_feedback_changed=self._on_voice_feedback_changed,
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
        self.view.set_feedback("Sugestie")
        self.view.set_metrics(self.scorer.snapshot(), self._format_elapsed())
        self.view.set_workout_status("Gotowy", "Jedna dłoń nad głową = start, dwie dłonie = stop.")
        self.view.set_primary_action_label("START")
        self.view.set_pose_metrics(PoseMetrics(False))
        self.view.set_angle_tolerances(
            self.front_tolerance_degrees,
            self.side_tolerance_degrees,
            self.side_back_tolerance_degrees,
        )

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
        for _ in range(50):
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
        if self.session_mode in {"active", "paused"}:
            return
        self._arm_session(trigger="button")

    def primary_action(self) -> None:
        """Handle the main Start/Stop button in the right panel."""
        if self.session_mode == "active":
            self.end_session()
            return
        if self.session_mode == "paused":
            self.toggle_pause()
            return
        self.start_session()

    def _arm_session(self, trigger: str) -> None:
        """Prepare and start a fresh series."""
        self.session_mode = "active"
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
        self._rep_state = "WAITING_TOP_START"
        self._bottom_reached_in_cycle = False
        self._top_hold_started_at = None
        self._rep_started_elapsed = 0.0
        self._current_rep_frames_count = 0
        self._current_rep_hip_correct_frames = 0
        self._current_rep_knee_correct_frames = 0
        self._last_calculated_quality = 100
        self._current_rep_issues = set()
        self._top_hold_feedback_sent = False

        for analyzer in self.analyzers.values():
            analyzer.reset()
        self.scorer.reset()

        self.session_active = True
        self.session_started_at = time.time()
        self.last_tick = self.session_started_at
        self._last_instruction_at = self.session_started_at

        self.view.set_workout_status("Aktywna", "Jedna dłoń nad głową = start, dwie dłonie = stop.")
        self.view.set_feedback("Seria uruchomiona.")
        self.view.set_primary_action_label("STOP")
        self.view.append_event(f"Seria uruchomiona ({trigger})")
        self.voice.say("Seria uruchomiona.")
        self._sync_score_state()

        self._update_camera_status()
        active_cameras = len(self.camera.available_devices)
        if active_cameras == 0:
            self.view.set_feedback("Sesja uruchomiona bez aktywnej kamery. Panele pozostana czarne do czasu wykrycia urzadzenia.")
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

        if self.session_mode == "active":
            self.session_mode = "paused"
            self.session_paused = True
            self.view.set_workout_status("Pauza", "Seria wstrzymana. Czas i tempo nie rosną w trakcie pauzy.")
            self.view.set_feedback("Seria wstrzymana.")
            self.view.set_primary_action_label("START")
            self.view.append_event("Seria wstrzymana")
            return

        if self.session_mode == "paused":
            self.session_mode = "active"
            self.session_paused = False
            self.last_tick = time.time()
            self.view.set_workout_status("Aktywna", "Seria wznowiona.")
            self.view.set_feedback("Seria wznowiona.")
            self.view.set_primary_action_label("STOP")
            self.view.append_event("Seria wznowiona")

    def _set_gesture_toggle_block(self, seconds: float = 1.0) -> None:
        """Prevent immediate gesture re-triggering after a state change."""

        self._gesture_toggle_block_until = time.time() + seconds

    def _gesture_toggle_block_active(self, now: float) -> bool:
        return now < self._gesture_toggle_block_until

    def end_session(self) -> None:
        """End the current session and persist the result automatically."""
        if self.session_mode == "idle":
            return

        self.session_mode = "idle"
        self.session_active = False
        self.session_paused = False
        self._set_gesture_toggle_block()
        self.view.set_connection_status("Kamera nieaktywna")
        self.view.set_workout_status("Zakończona", "Jedna dłoń nad głową = start, dwie dłonie = stop.")
        self.view.set_feedback("Seria zakończona.")
        self.view.set_primary_action_label("START")
        self.view.append_event("Seria zakończona")
        self.voice.say("Seria zakończona.")
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

    def _on_angle_tolerance_changed(self, axis: str, value: float) -> None:
        """Apply angle tolerance changes from the settings view."""

        clamped = max(1.0, min(45.0, float(value)))
        if axis == "front":
            self.front_tolerance_degrees = clamped
            label = "przód"
        elif axis == "side":
            self.side_tolerance_degrees = clamped
            label = "bok: noga"
        elif axis == "side_back":
            self.side_back_tolerance_degrees = clamped
            label = "bok: plecy"
        else:
            return

        for analyzer in self.analyzers.values():
            analyzer.set_tolerances(
                front_tolerance_degrees=self.front_tolerance_degrees,
                side_tolerance_degrees=self.side_tolerance_degrees,
                side_back_tolerance_degrees=self.side_back_tolerance_degrees,
            )

        self.view.set_angle_tolerances(
            self.front_tolerance_degrees,
            self.side_tolerance_degrees,
            self.side_back_tolerance_degrees,
        )
        save_persisted_settings(
            self.config.settings_file,
            replace(
                self.config,
                front_tolerance_degrees=self.front_tolerance_degrees,
                side_tolerance_degrees=self.side_tolerance_degrees,
                side_back_tolerance_degrees=self.side_back_tolerance_degrees,
            ),
        )
        self.view.append_event(f"Zmieniono tolerancję kąta ({label}) na ±{clamped:.1f}°")

    def _on_voice_feedback_changed(self, enabled: bool) -> None:
        """Handle voice feedback toggle from settings."""
        self.config = replace(self.config, voice_feedback_enabled=enabled)
        self.voice.set_enabled(enabled)
        save_persisted_settings(self.config.settings_file, self.config)
        status = "włączone" if enabled else "wyłączone"
        self.view.append_event(f"Komunikaty głosowe {status}")
        self.view.set_feedback(f"Komunikaty głosowe są teraz {status}.")

    def _voice_issue_summary(self, issues: set[str]) -> str | None:
        """Build a short spoken summary for the latest repetition (without repetition number)."""

        if not issues:
            return "Dobrze."

        phrases: list[str] = []
        if "hip" in issues:
            phrases.append("plecy nie trzymały pozycji")
        if "knee_side" in issues:
            phrases.append("kolana z boku nie były stabilne na górze")
        if "knee_front" in issues:
            phrases.append("kolana z przodu były nier\u00f3wne")

        if not phrases:
            return "Uwaga na technikę."

        joined = ", ".join(phrases)
        return f"Uwaga: {joined}."

    def _voice_top_hold_summary(self, hip_ok: bool, knee_side_ok: bool, knee_front_ok: bool | None = None) -> str:
        """Build spoken feedback for the top hold, even when the rep is not yet completed."""

        if hip_ok and knee_side_ok and (knee_front_ok is None or knee_front_ok):
            return "Góra: plecy i kolana w porządku."

        parts: list[str] = []
        if not hip_ok:
            parts.append("plecy za bardzo odjechały")
        if not knee_side_ok:
            parts.append("kolana z boku wymagają stabilizacji")
        if knee_front_ok is not None and not knee_front_ok:
            parts.append("kolana z przodu nierówne")

        return f"Góra: {', '.join(parts)}."

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
        self._side_metrics = None
        self._front_metrics = None
        
        for slot_index in range(self.config.max_camera_slots):
            panel = self.view.get_camera_panel(slot_index)
            area = getattr(panel, "inner_image_area", panel.holder)
            max_width = max(320, area.winfo_width() or 640)
            max_height = max(240, area.winfo_height() or 480)
            ok, frame, source_index = self.camera.read(slot_index)
            if ok and frame is not None:
                metrics, frame = self.analyzers[slot_index].analyze_frame(frame)
                if metrics.pose_detected:
                    # Identify camera type by available metrics
                    has_hip_angle = not math.isnan(metrics.upper_body_angle)
                    has_knee_front = not math.isnan(metrics.knee_angle_front)
                    
                    # Store metrics by view type for dual-camera analysis
                    if has_hip_angle:
                        self._side_metrics = metrics
                    if has_knee_front:
                        self._front_metrics = metrics
                    
                    if metrics.visibility >= best_metrics.visibility:
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

    def _handle_gesture_toggle(self) -> None:
        """Toggle session state using the hand raise gesture."""
        if self.session_mode == "idle":
            self._arm_session(trigger="gesture")
        elif self.session_mode in {"arming", "active", "paused"}:
            self.end_session()

    def _update_metrics(self, pose_metrics: PoseMetrics | None) -> None:
        """Translate raw pose signals into training metrics and updates the UI."""
        if pose_metrics is None:
            return

        now = time.time()
        repetition_event = False

        # Podgląd kątów na żywo
        self.view.set_pose_metrics(pose_metrics)

        # Sprawdzenie, czy dłoń jest uniesiona - ze strony bocznej lub frontu
        metrics_for_gestures = self._side_metrics or self._front_metrics or pose_metrics
        hand_is_raised = getattr(metrics_for_gestures, "hand_raised", False)
        start_gesture = getattr(metrics_for_gestures, "start_gesture", hand_is_raised)
        stop_gesture = getattr(metrics_for_gestures, "stop_gesture", False)
        two_hands_visible = getattr(metrics_for_gestures, "two_hands_visible", False)
        gesture_blocked = self._gesture_toggle_block_active(now)

        # Gesty sterujące sesją
        if self.session_mode == "idle":
            if gesture_blocked:
                return
            if start_gesture:
                self._arm_session(trigger="gesture")
            return

        if self.session_mode == "active":
            if gesture_blocked:
                return
            if stop_gesture:
                self.end_session()
                return

        # Jeśli sesja nie jest aktywna (np. paused), przerywamy dalsze przetwarzanie
        if self.session_mode == "paused" or not self.session_active:
            return

        # Zliczanie czasu trwania sesji
        self.session_elapsed += now - self.last_tick
        self.last_tick = now

        # 2. MASZYNA STANÓW HIP THRUST: top -> dół -> góra -> 1 s hold na górze
        # Pobieramy plecy z kamery bocznej (główna analiza)
        hip_angle = getattr(self._side_metrics, "upper_body_angle", 0.0) if self._side_metrics else 0.0
        # Kolana z boku również z kamery bocznej
        knee_side = getattr(self._side_metrics, "knee_angle_side", 0.0) if self._side_metrics else 0.0
        # Kolana z przodu z kamery frontalnej, jeśli jest dostępna
        knee_front = getattr(self._front_metrics, "knee_angle_front", float("nan")) if self._front_metrics else float("nan")
        
        # Fallback: jeśli mamy tylko jedną kamerę (front), bierz z niej dostępne dane
        if not self._side_metrics and self._front_metrics:
            hip_angle = getattr(self._front_metrics, "upper_body_angle", 0.0)
            knee_side = getattr(self._front_metrics, "knee_angle_side", 0.0)
        # Jeśli mamy tylko boczną, możemy spróbować pobrać kolana z przodu z niej
        if not self._front_metrics and self._side_metrics:
            knee_front = getattr(self._side_metrics, "knee_angle_front", float("nan"))
        
        # Określ, czy mamy dwie kamery czy jedną
        has_both_cameras = self._side_metrics is not None and self._front_metrics is not None

        in_bottom_position = not math.isnan(hip_angle) and (80.0 <= hip_angle <= 140.0)
        in_top_position = not math.isnan(hip_angle) and (165.0 <= hip_angle <= 175.0)

        # Oceniamy tylko zatrzymanie na górze, nie całą drogę ruchu.
        if self._rep_state == "TOP_HOLDING":
            hip_correct = self.scorer.grade_hip_top_hold(hip_angle)
            knee_side_correct = self.scorer.grade_knee_stable(knee_side)
            # Ocena kolan z przodu - jeśli kamera jest dostępna
            knee_front_correct = None
            if not math.isnan(knee_front):
                knee_front_correct = self.scorer.grade_knee_stable(knee_front)
        else:
            hip_correct = False
            knee_side_correct = False
            knee_front_correct = None

        if self._side_metrics and self._side_metrics.pose_detected:
            if self._rep_state == "WAITING_TOP_START":
                # Instruktujący komunikat co 15 sekund jeśli użytkownik czeka
                if now - self._last_instruction_at >= 15.0:
                    self.voice.say_nonblocking("Spróbuj wykonać ćwiczenie. Stanął w górze, schyl się, wstań.")
                    self._last_instruction_at = now
                
                if in_top_position:
                    if self._top_hold_started_at is None:
                        self._top_hold_started_at = now
                    elif (now - self._top_hold_started_at) >= self.TOP_HOLD_SECONDS:
                        self._rep_state = "WAITING_FOR_DESCENT"
                        self._top_hold_started_at = None
                else:
                    self._top_hold_started_at = None

            elif self._rep_state == "WAITING_FOR_DESCENT":
                if not in_top_position:
                    self._rep_state = "GOING_DOWN_TO_BOTTOM"
                    self._rep_started_elapsed = self.session_elapsed
                    self._current_rep_frames_count = 0
                    self._current_rep_hip_correct_frames = 0
                    self._current_rep_knee_correct_frames = 0
                    self._current_rep_issues = set()

            elif self._rep_state == "GOING_DOWN_TO_BOTTOM":
                if in_bottom_position:
                    self._rep_state = "GOING_UP_TO_TOP"

            elif self._rep_state == "GOING_UP_TO_TOP":
                if in_top_position:
                    self._rep_state = "TOP_HOLDING"
                    self._top_hold_started_at = now
                    self._current_rep_frames_count = 0
                    self._current_rep_hip_correct_frames = 0
                    self._current_rep_knee_correct_frames = 0
                    self._top_hold_feedback_sent = False

                    hold_message = self._voice_top_hold_summary(
                        hip_correct,
                        knee_side_correct,
                        knee_front_correct,
                    )
                    self.view.set_feedback(hold_message)
                    self.view.append_event(hold_message)
                    self.voice.say(hold_message)
                    self._top_hold_feedback_sent = True

            elif self._rep_state == "TOP_HOLDING":
                self._current_rep_frames_count += 1
                if hip_correct: self._current_rep_hip_correct_frames += 1
                # Count knee correctness - use side knee primarily, but if front is available, both must be correct
                knee_ok_for_count = knee_side_correct and (knee_front_correct is None or knee_front_correct)
                if knee_ok_for_count:
                    self._current_rep_knee_correct_frames += 1
                
                if not hip_correct:
                    self._current_rep_issues.add("hip")
                if not knee_side_correct:
                    self._current_rep_issues.add("knee_side")
                if knee_front_correct is not None and not knee_front_correct:
                    self._current_rep_issues.add("knee_front")

                if not in_top_position:
                    self._rep_state = "GOING_DOWN_TO_BOTTOM"
                    self._top_hold_started_at = None
                else:
                    if self._top_hold_started_at and (now - self._top_hold_started_at) >= self.TOP_HOLD_SECONDS:
                        msg = self.scorer.register_repetition(
                            correct_hip_frames=self._current_rep_hip_correct_frames,
                            correct_knee_frames=self._current_rep_knee_correct_frames,
                            total_frames=self._current_rep_frames_count
                        )

                        self._rep_count = self.scorer.repetitions
                        self._last_calculated_quality = self.scorer.quality
                        repetition_event = True

                        rep_duration = max(0.01, self.session_elapsed - self._rep_started_elapsed)
                        self._tempo_samples.append(rep_duration)

                        self.view.append_event(msg)
                        self.view.set_feedback(msg)
                        
                        # Announce repetition number first
                        self.voice.say(f"Powtórzenie {self._rep_count}.")
                        
                        voice_message = self._voice_issue_summary(set(self._current_rep_issues))
                        if voice_message:
                            self.voice.say(voice_message)

                        self._rep_state = "WAITING_FOR_DESCENT"
                        self._top_hold_started_at = None
                        self._current_rep_issues = set()
        else:
            if self._rep_state in {"GOING_DOWN_TO_BOTTOM", "GOING_UP_TO_TOP", "TOP_HOLDING"}:
                self._current_rep_frames_count += 1

        # 3. TWOJA ORYGINALNA SYNCHRONIZACJA STATUSU I WIDOKU UI
        self._sync_score_state()
        
        snapshot = self.scorer.snapshot(repetition_event=repetition_event)
        self.view.set_metrics(snapshot, self._format_elapsed())
        
        if self._tempo_samples:
            current_tempo = self._tempo_samples[-1]
            avg_tempo = sum(self._tempo_samples) / len(self._tempo_samples)
        else:
            avg_tempo = 0.0
            current_tempo = 0.0

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
        try:
            self.voice.close()
        except Exception:
            pass
        self.transport.stop()
        self.camera.close()
        self.view.destroy()