"""Tkinter view that renders the Cyber Trener desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import tkinter as tk
from tkinter import ttk

from scoring.engine import ScoreSnapshot
from utils.config import AppConfig


@dataclass(frozen=True)
class ViewCallbacks:
    """Callbacks injected from the controller layer."""

    on_start_session: Callable[[], None]
    on_primary_action: Callable[[], None]
    on_toggle_pause: Callable[[], None]
    on_end_session: Callable[[], None]
    on_save_result: Callable[[], None]
    on_close: Callable[[], None]
    on_camera_source_changed: Callable[[int, str], None]
    on_angle_tolerance_changed: Callable[[str, float], None]
    on_voice_feedback_changed: Callable[[bool], None]
    on_nav_section_changed: Callable[[str], None]


@dataclass
class CameraPanelWidgets:
    """Widget references for a single camera preview panel."""

    frame: tk.Frame
    holder: tk.Frame
    image: tk.Label
    status: tk.Label
    picker: ttk.Combobox
    picker_var: tk.StringVar
    slot_index: int
    # inner_image_area: frame that maintains aspect ratio (16:9)
    inner_image_area: tk.Frame | None = None


class CyberTrainerView(tk.Tk):
    """Own the Tk root window and expose small update methods for the controller."""

    def __init__(
        self,
        config: AppConfig,
        callbacks: ViewCallbacks,
        camera_only: bool = False,
    ) -> None:
        """Create the root window and build the layout."""

        super().__init__()
        self.config = config
        self.callbacks = callbacks
        self.camera_only = camera_only

        self.title(config.window_title)
        self.state(config.window_state)
        self.configure(bg=config.background_color)
        self.protocol("WM_DELETE_WINDOW", self.callbacks.on_close)
        # Fullscreen tracking
        self._is_fullscreen = False
        self.bind("<F11>", self._toggle_fullscreen)

        self.source_label = tk.StringVar(value="Demo mode")
        self.connection_label = tk.StringVar(value="Kamera nieaktywna")
        self.feedback_label = tk.StringVar(value="Sugestie")
        self.metric_reps = tk.StringVar(value="0")
        self.metric_time = tk.StringVar(value="00:00")
        self.metric_quality = tk.StringVar(value=f"{config.default_quality}%")
        self.metric_warnings = tk.StringVar(value="0")
        self.metric_knee_error = tk.StringVar(value="--")
        self.technique_knee_value = tk.StringVar(value="stabilne")
        self.technique_knee_color = self.config.accent_green
        self.technique_knee_label_widget = None
        self.nav_sections = ["Trening", "Historia", "Postępy", "Ustawienia"]
        self.active_nav_section = tk.StringVar(value="Trening")
        self._nav_buttons: dict[str, tk.Button] = {}
        self.left_panel_width = 230
        self.right_panel_width = 310
        self.side_gap = 18
        self.camera_inner_padding = 12

        self._camera_photos: dict[int, object] = {}
        self._panels: list[CameraPanelWidgets] = []
        self.workout_state = tk.StringVar(value="Gotowy")
        self.workout_hint = tk.StringVar(value="Jedna dłoń nad głową = start, dwie dłonie = stop.")
        self.primary_action_label = tk.StringVar(value="START")
        self.pose_side = tk.StringVar(value="--")
        
        # Kąty kolan (przód i bok)
        self.knee_angle_front_text = tk.StringVar(value="--")
        self.knee_angle_side_text = tk.StringVar(value="--")
        
        self.upper_angle_text = tk.StringVar(value="--")
        self.rep_tempo_text = tk.StringVar(value="--")
        self.avg_tempo_text = tk.StringVar(value="--")
        self.elapsed_text = tk.StringVar(value="00:00")
        self.rep_count_text = tk.StringVar(value="0")
        self.hand_signal_text = tk.StringVar(value="Ręka: --")
        self.front_tolerance_var = tk.StringVar(value=f"{self.config.front_tolerance_degrees:.1f}")
        self.side_tolerance_var = tk.StringVar(value=f"{self.config.side_tolerance_degrees:.1f}")
        self.side_back_tolerance_var = tk.StringVar(value=f"{self.config.side_back_tolerance_degrees:.1f}")
        self.front_tolerance_scale = tk.DoubleVar(value=self.config.front_tolerance_degrees)
        self.side_tolerance_scale = tk.DoubleVar(value=self.config.side_tolerance_degrees)
        self.side_back_tolerance_scale = tk.DoubleVar(value=self.config.side_back_tolerance_degrees)
        self._suspend_tolerance_callbacks = False
        self._settings_camera_vars: list[tk.StringVar] = []
        self._settings_camera_boxes: list[ttk.Combobox] = []
        self.voice_feedback_enabled = tk.BooleanVar(value=self.config.voice_feedback_enabled)
        self._build_ui()

    def set_connection_status(self, text: str) -> None:
        """Update the sidebar connection label."""

        self.connection_label.set(text)

    def set_source_label(self, text: str) -> None:
        """Update the compact source chip label."""

        self.source_label.set(text)

    def set_feedback(self, text: str) -> None:
        """Set the main feedback text shown in the central card."""

        self.feedback_label.set(text)

    def set_metrics(self, snapshot: ScoreSnapshot, elapsed_text: str) -> None:
        """Render the metric counters in the statistics card."""

        self.metric_reps.set(str(snapshot.repetitions))
        self.metric_time.set(elapsed_text)
        self.metric_quality.set(f"{snapshot.quality}%")
        self.metric_warnings.set(str(snapshot.warnings))

    def set_workout_status(self, state: str, hint: str) -> None:
        """Update the workout state banner and guidance."""

        self.workout_state.set(state)
        self.workout_hint.set(hint)

    def set_primary_action_label(self, text: str) -> None:
        """Update the main action button label."""

        self.primary_action_label.set(text)

    def is_training_section_active(self) -> bool:
        """Return True when the currently selected navigation section is Trening."""

        return self.active_nav_section.get() == "Trening"

    def set_pose_metrics(self, pose_metrics) -> None:
        """Render the latest pose-derived metrics into the right-side panel."""

        self.pose_side.set(pose_metrics.side if pose_metrics.pose_detected else "--")
        
        self.knee_angle_front_text.set("--" if math.isnan(pose_metrics.knee_angle_front) else f"{pose_metrics.knee_angle_front:.1f}°")
        self.knee_angle_side_text.set("--" if math.isnan(pose_metrics.knee_angle_side) else f"{pose_metrics.knee_angle_side:.1f}°")
        
        self.upper_angle_text.set("--" if math.isnan(pose_metrics.upper_body_angle) else f"{pose_metrics.upper_body_angle:.1f}°")
        self.hand_signal_text.set("Ręka: TAK" if pose_metrics.hand_raised else "Ręka: --")

    def set_workout_counters(self, elapsed_text: str, current_tempo: str, avg_tempo: str, reps: int) -> None:
        """Render workout counters such as elapsed time and tempo."""

        self.elapsed_text.set(elapsed_text)
        self.rep_tempo_text.set(current_tempo)
        self.avg_tempo_text.set(avg_tempo)
        self.rep_count_text.set(str(reps))

    def set_knee_error(self, knee_error: float) -> None:
        """Update the knee alignment error percentage."""

        if math.isnan(knee_error):
            self.metric_knee_error.set("--")
            self.set_knee_technique(float('nan'))
        else:
            knee_error_percent = knee_error * 100
            self.metric_knee_error.set(f"{knee_error_percent:.1f}%")
            self.set_knee_technique(knee_error_percent)

    def set_knee_technique(self, knee_error_percent: float) -> None:
        """Update the knee technique row based on error percentage."""

        if math.isnan(knee_error_percent):
            self.technique_knee_value.set("??% - niewidoczne")
            self.technique_knee_color = self.config.text_muted
        elif knee_error_percent >= 95:
            self.technique_knee_value.set(f"{knee_error_percent:.1f}% - stabilnie")
            self.technique_knee_color = self.config.accent_green
        elif knee_error_percent >= 90:
            self.technique_knee_value.set(f"{knee_error_percent:.1f}% - do dopracowania")
            self.technique_knee_color = self.config.accent_orange
        else:
            self.technique_knee_value.set(f"{knee_error_percent:.1f}% - źle")
            self.technique_knee_color = self.config.danger_color
        
        # Update the label color if widget exists
        if self.technique_knee_label_widget:
            self.technique_knee_label_widget.configure(fg=self.technique_knee_color)

    def set_history(self, entries: Sequence[str]) -> None:
        """Replace the history list with formatted session entries."""

        rendered = list(entries)
        if not rendered:
            rendered = ["Brak zapisanej historii"]

        updated_any_widget = False

        if hasattr(self, "history_box") and not self.camera_only:
            self.history_box.delete(0, tk.END)
            for entry in rendered:
                self.history_box.insert(tk.END, entry)
            updated_any_widget = True

        if hasattr(self, "history_page_box"):
            self.history_page_box.delete(0, tk.END)
            for entry in rendered:
                self.history_page_box.insert(tk.END, entry)
            updated_any_widget = True

        if not updated_any_widget:
            self._pending_history = rendered

    def set_history_records(self, records: Sequence[dict[str, str]]) -> None:
        """Render structured history records as cards on the history page."""

        rendered = list(records)
        self._pending_history_records = rendered
        if not hasattr(self, "_history_cards_frame"):
            return

        for child in self._history_cards_frame.winfo_children():
            child.destroy()

        if not rendered:
            empty = tk.Label(
                self._history_cards_frame,
                text="Brak zapisanej historii",
                bg="#0f172a",
                fg="#9fb3cf",
                font=("Segoe UI", 12),
                pady=16,
            )
            empty.pack(fill="x", padx=6, pady=6)
            return

        for index, record in enumerate(rendered, start=1):
            card = tk.Frame(self._history_cards_frame, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
            card.pack(fill="x", padx=6, pady=(0, 10))

            header = tk.Frame(card, bg="#13213a")
            header.pack(fill="x")
            tk.Label(
                header,
                text=f"Seria {index}",
                bg="#13213a",
                fg="#f8fafc",
                font=("Segoe UI", 11, "bold"),
            ).pack(side="left", padx=12, pady=8)
            tk.Label(
                header,
                text=f"Źródło: {record.get('source', 'unknown')}",
                bg="#13213a",
                fg="#a8c1df",
                font=("Segoe UI", 9),
            ).pack(side="right", padx=12, pady=8)

            body = tk.Frame(card, bg="#10192c")
            body.pack(fill="x", padx=12, pady=10)

            for label, value in [
                ("Powtórzenia", record.get("repetitions", "0")),
                ("Średnia jakość", record.get("avg_quality", "0.0%")),
                ("Średni czas powtórzenia", record.get("avg_rep_time", "0.00 s")),
                ("Całkowity czas serii", record.get("total_series_time", "00:00")),
            ]:
                row = tk.Frame(body, bg="#10192c")
                row.pack(fill="x", pady=2)
                tk.Label(row, text=label, bg="#10192c", fg="#9fb3cf", font=("Segoe UI", 10)).pack(side="left")
                tk.Label(row, text=value, bg="#10192c", fg="#f1f7ff", font=("Segoe UI", 10, "bold")).pack(side="right")

        self._history_cards_frame.update_idletasks()
        if hasattr(self, "_history_canvas"):
            self._history_canvas.configure(scrollregion=self._history_canvas.bbox("all"))

    def show_session_summary_popup(
        self,
        repetitions: int,
        avg_quality: float,
        avg_rep_time_seconds: float,
        total_series_text: str,
    ) -> None:
        """Display a themed session summary popup aligned with the app style."""

        popup = tk.Toplevel(self)
        popup.title("Podsumowanie treningu")
        popup.transient(self)
        popup.grab_set()
        popup.configure(bg="#0b1220")
        popup.resizable(False, False)

        card = tk.Frame(popup, bg="#0f172a", highlightthickness=1, highlightbackground="#314058", bd=0)
        card.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(
            card,
            text="Gratulacje!",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 2))
        tk.Label(
            card,
            text="Podsumowanie zakończonej serii",
            bg="#0f172a",
            fg="#9fb3cf",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=18, pady=(0, 12))

        metrics = tk.Frame(card, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
        metrics.pack(fill="x", padx=18, pady=(0, 12))

        for label, value in [
            ("Liczba powtórzeń", str(repetitions)),
            ("Średnia jakość", f"{avg_quality:.1f}%"),
            ("Średni czas powtórzenia", f"{avg_rep_time_seconds:.2f} s"),
            ("Całkowity czas serii", total_series_text),
        ]:
            row = tk.Frame(metrics, bg="#10192c")
            row.pack(fill="x", padx=12, pady=6)
            tk.Label(row, text=label, bg="#10192c", fg="#9fb3cf", font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, text=value, bg="#10192c", fg="#f1f7ff", font=("Segoe UI", 11, "bold")).pack(side="right")

        tk.Button(
            card,
            text="Zamknij",
            command=popup.destroy,
            bg="#22c55e",
            fg="#08111f",
            activebackground="#2dd06c",
            activeforeground="#08111f",
            relief="flat",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=8,
        ).pack(anchor="e", padx=18, pady=(0, 16))

        popup.update_idletasks()
        width = popup.winfo_reqwidth()
        height = popup.winfo_reqheight()
        x = self.winfo_rootx() + (self.winfo_width() - width) // 2
        y = self.winfo_rooty() + (self.winfo_height() - height) // 2
        popup.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
        popup.focus_set()

    def append_event(self, message: str) -> None:
        """Push a new message into the event log."""

        timestamp = time.strftime("%H:%M:%S")
        # event_log may not exist in camera-only mode; fallback to console
        formatted = f"[{timestamp}] {message}"
        if hasattr(self, "event_log") and not self.camera_only:
            self.event_log.insert(0, formatted)
            if self.event_log.size() > 12:
                self.event_log.delete(tk.END)
        else:
            try:
                print(formatted)
            except Exception:
                # silent fallback if stdout unavailable
                pass

    def set_camera_sources(self, options: Sequence[str], selected_labels: Sequence[str]) -> None:
        """Refresh all camera comboboxes with new available sources."""

        for panel, selected in zip(self._panels, selected_labels, strict=False):
            panel.picker.configure(values=list(options))
            if selected not in options:
                selected = "Brak"
            panel.picker_var.set(selected)

        for box, variable, selected in zip(self._settings_camera_boxes, self._settings_camera_vars, selected_labels, strict=False):
            box.configure(values=list(options))
            if selected not in options:
                selected = "Brak"
            variable.set(selected)

    def set_angle_tolerances(self, front_tolerance: float, side_tolerance: float, side_back_tolerance: float) -> None:
        """Update editable tolerance values shown in the settings page."""

        self._suspend_tolerance_callbacks = True
        try:
            self.front_tolerance_var.set(f"{front_tolerance:.1f}")
            self.side_tolerance_var.set(f"{side_tolerance:.1f}")
            self.side_back_tolerance_var.set(f"{side_back_tolerance:.1f}")
            self.front_tolerance_scale.set(front_tolerance)
            self.side_tolerance_scale.set(side_tolerance)
            self.side_back_tolerance_scale.set(side_back_tolerance)
        finally:
            self._suspend_tolerance_callbacks = False

    def get_camera_panel(self, slot_index: int) -> CameraPanelWidgets:
        """Return a camera panel reference by slot index."""

        return self._panels[slot_index]

    def update_camera_panel(
        self,
        slot_index: int,
        photo: object,
        status_text: str,
        online: bool,
    ) -> None:
        """Display a new frame or the offline placeholder in a camera panel."""

        panel = self._panels[slot_index]
        panel.image.configure(image=photo, text="", fg=self.config.text_muted)
        panel.image.image = photo  # type: ignore[attr-defined]
        panel.status.configure(text=f"● LIVE | {status_text}" if online else "● OFFLINE")
        panel.status.configure(fg=self.config.accent_green if online else self.config.danger_color)
        self._camera_photos[slot_index] = photo

    def _build_ui(self) -> None:
        """Create the page shell, sidebar and cards."""
        try:
            self.attributes("-fullscreen", True)
        except Exception:
            try:
                self.state("zoomed")
            except Exception:
                pass

        root = ttk.Frame(self, style="App.TFrame", padding=(12, 8, 12, 8))
        root.pack(fill="both", expand=True)

        target_w, total_h = self._fit_camera_window()

        shell = tk.Frame(root, bg=self.config.background_color)
        shell.place(relx=0.5, rely=0.5, anchor="center")
        shell.grid_columnconfigure(0, minsize=self.left_panel_width, weight=0)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_columnconfigure(2, minsize=self.right_panel_width, weight=0)
        shell.grid_rowconfigure(0, weight=1)

        left_panel = tk.Frame(shell, bg="#0d1527", highlightthickness=1, highlightbackground="#27324a", bd=0)
        left_panel.grid(row=0, column=0, sticky="nsw", padx=(0, self.side_gap))
        left_panel.configure(width=self.left_panel_width, height=total_h)
        left_panel.grid_propagate(False)
        self._camera_shell = shell
        self._camera_left_panel = left_panel
        self._build_left_panel(left_panel)

        center_panel = tk.Frame(shell, bg=self.config.background_color)
        center_panel.grid(row=0, column=1, sticky="nsew")
        center_panel.configure(width=target_w, height=total_h)
        center_panel.grid_propagate(False)
        self._camera_center_panel = center_panel

        self._page_container = tk.Frame(center_panel, bg=self.config.background_color)
        self._page_container.place(relx=0.5, rely=0.5, anchor="center", width=target_w, height=total_h)

        self._camera_page = tk.Frame(self._page_container, bg=self.config.background_color)
        self._camera_page.place(relx=0.5, rely=0.5, anchor="center", width=target_w, height=total_h)
        canvas = tk.Canvas(self._camera_page, bg=self.config.background_color, highlightthickness=0)
        canvas.place(relx=0.5, rely=0.5, anchor="center", width=target_w, height=total_h)

        radius = getattr(self.config, "camera_corner_radius", 16)
        bg = self.config.panel_holder_bg
        x0, y0, x1, y1 = 0, 0, target_w, total_h
        r = max(0, min(radius, int(min(target_w, total_h) / 2)))
        try:
            canvas.create_rectangle(x0 + r, y0, x1 - r, y1, fill=bg, outline=bg)
            canvas.create_rectangle(x0, y0 + r, x1, y1 - r, fill=bg, outline=bg)
            canvas.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, fill=bg, outline=bg)
            canvas.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, fill=bg, outline=bg)
            canvas.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, fill=bg, outline=bg)
            canvas.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, fill=bg, outline=bg)
        except Exception:
            canvas.create_rectangle(x0, y0, x1, y1, fill=bg, outline=bg)

        inner_frame = tk.Frame(canvas, bg=self.config.panel_holder_bg)
        inner_w = max(1, target_w - (self.camera_inner_padding * 2))
        inner_h = max(1, total_h - (self.camera_inner_padding * 2))
        canvas.create_window(target_w // 2, total_h // 2, window=inner_frame, width=inner_w, height=inner_h)
        self._camera_container = canvas
        self._build_camera_card(inner_frame)

        self._settings_page = tk.Frame(self._page_container, bg=self.config.background_color)
        self._settings_page.place(relx=0.5, rely=0.5, anchor="center", width=target_w, height=total_h)
        self._build_settings_page(self._settings_page)
        self._settings_page.place_forget()

        self._history_page = tk.Frame(self._page_container, bg=self.config.background_color)
        self._history_page.place(relx=0.5, rely=0.5, anchor="center", width=target_w, height=total_h)
        self._build_history_page(self._history_page)
        self._history_page.place_forget()

        right_panel = tk.Frame(shell, bg="#0d1527", highlightthickness=1, highlightbackground="#27324a", bd=0)
        right_panel.grid(row=0, column=2, sticky="nse", padx=(self.side_gap, 0))
        right_panel.configure(width=self.right_panel_width, height=total_h)
        right_panel.grid_propagate(False)
        self._camera_right_panel = right_panel
        self._build_right_panel(right_panel)

    def _fit_camera_window(self) -> None:
        """Resize the main window so two stacked 16:9 panels fit without horizontal bars."""
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        margin_w = 60
        margin_h = 120

        avail_w = max(200, screen_w - margin_w)
        avail_h = max(200, screen_h - margin_h)

        if getattr(self, "camera_only", False):
            avail_w = max(200, avail_w - self.left_panel_width - self.right_panel_width - (self.side_gap * 2))

        width_by_height = int(avail_h * (16 / 18))
        safety_margin = 24
        computed = min(avail_w, width_by_height) - safety_margin
        target_w = max(320, min(avail_w, computed))

        panel_h = int(target_w * 9 / 16)
        total_h = panel_h * 2
        total_h += 60

        return target_w, total_h

    def _toggle_fullscreen(self, event: tk.Event | None = None) -> None:
        """Toggle fullscreen with F11 and adjust camera container size."""
        self._is_fullscreen = not getattr(self, "_is_fullscreen", False)
        try:
            self.attributes("-fullscreen", self._is_fullscreen)
        except Exception:
            try:
                self.state("zoomed" if self._is_fullscreen else "normal")
            except Exception:
                pass

        if getattr(self, "camera_only", False) and hasattr(self, "_camera_container"):
            target_w, total_h = self._fit_camera_window()
            try:
                self._camera_container.place_configure(width=target_w, height=total_h)
                if hasattr(self, "_page_container"):
                    self._page_container.place_configure(width=target_w, height=total_h)
                if hasattr(self, "_camera_center_panel"):
                    self._camera_center_panel.configure(width=target_w, height=total_h)
                if hasattr(self, "_camera_shell"):
                    shell_w = target_w + self.left_panel_width + self.right_panel_width + (self.side_gap * 2)
                    self._camera_shell.place_configure(width=shell_w, height=total_h)
            except Exception:
                pass

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        """Build the sidebar status, navigation and history widgets."""

        shell = tk.Frame(parent, bg="#0b1020", highlightthickness=1, highlightbackground="#27324a", bd=0)
        shell.pack(fill="both", expand=True)

        shell.grid_rowconfigure(0, weight=0)
        shell.grid_rowconfigure(1, weight=0)
        shell.grid_rowconfigure(2, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        header = tk.Frame(shell, bg="#0d1527")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 12))
        tk.Label(
            header,
            text=self.config.app_name,
            bg="#0d1527",
            fg="#f4f7fb",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(
            header,
            text="Glassmorphism shell for training, analysis and progress.",
            bg="#0d1527",
            fg="#9eb0c9",
            font=("Segoe UI", 9),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 14))

        nav = tk.Frame(shell, bg="#0d1527", highlightthickness=1, highlightbackground="#334055")
        nav.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
        tk.Label(
            nav,
            text="Nawigacja",
            bg="#0d1527",
            fg="#dbe7f6",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 8))

        for label in self.nav_sections:
            btn = tk.Button(
                nav,
                text=label,
                command=lambda value=label: self._set_nav_section(value),
                bg="#101a30",
                fg="#d6e1f2",
                activebackground="#17304f",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                highlightthickness=1,
                highlightbackground="#2f415a",
                font=("Segoe UI", 11, "bold"),
                anchor="w",
                padx=14,
                pady=10,
            )
            btn.pack(fill="x", padx=12, pady=5)
            self._nav_buttons[label] = btn

        status = tk.Frame(shell, bg="#0d1527", highlightthickness=1, highlightbackground="#334055")
        status.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        status.grid_columnconfigure(0, weight=1)

        tk.Label(
            status,
            text="Stan",
            bg="#0d1527",
            fg="#dbe7f6",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(
            status,
            textvariable=self.connection_label,
            bg="#0d1527",
            fg="#9eb0c9",
            font=("Segoe UI", 9),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 10))

        chip = tk.Label(
            status,
            textvariable=self.active_nav_section,
            bg="#14233a",
            fg="#7ef0ff",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=5,
        )
        chip.pack(anchor="w", padx=14, pady=(0, 12))

        self.history_box = tk.Listbox(
            status,
            bg="#0b1020",
            fg="#d6e1f2",
            highlightthickness=0,
            selectbackground="#17304f",
            borderwidth=0,
            activestyle="none",
        )
        self.history_box.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._set_nav_section("Trening")

    def _build_left_panel(self, parent: tk.Frame) -> None:
        """Build the left-side glass panel for navigation and brand."""

        panel = tk.Canvas(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        panel.pack(fill="both", expand=True)

        def draw_panel() -> None:
            panel.delete("all")
            w = int(panel.winfo_width() or panel.winfo_reqwidth() or parent.winfo_width() or self.left_panel_width)
            h = int(panel.winfo_height() or panel.winfo_reqheight() or parent.winfo_height() or 200)
            r = 22
            bg = "#0c1324"
            try:
                panel.create_rectangle(0 + r, 0, w - r, h, fill=bg, outline=bg)
                panel.create_rectangle(0, 0 + r, w, h - r, fill=bg, outline=bg)
                panel.create_arc(0, 0, 0 + 2 * r, 0 + 2 * r, start=90, extent=90, fill=bg, outline=bg)
                panel.create_arc(w - 2 * r, 0, w, 0 + 2 * r, start=0, extent=90, fill=bg, outline=bg)
                panel.create_arc(0, h - 2 * r, 0 + 2 * r, h, start=180, extent=90, fill=bg, outline=bg)
                panel.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=bg, outline=bg)
            except Exception:
                panel.create_rectangle(0, 0, w, h, fill=bg, outline=bg)

            inner = tk.Frame(panel, bg=bg)
            panel.create_window(w // 2, h // 2, window=inner, width=max(1, w - 18), height=max(1, h - 18))

            tk.Label(inner, text=self.config.app_name, bg=bg, fg="#f5fbff", font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=16, pady=(16, 2))
            tk.Label(inner, text="Training shell", bg=bg, fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=16, pady=(0, 12))

            tk.Label(inner, text="Menu", bg=bg, fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=16, pady=(2, 8))
            for label in self.nav_sections:
                btn = tk.Button(
                    inner,
                    text=label,
                    command=lambda value=label: self._set_nav_section(value),
                    bg="#101a30",
                    fg="#d6e1f2",
                    activebackground="#17304f",
                    activeforeground="#ffffff",
                    relief="flat",
                    bd=0,
                    highlightthickness=1,
                    highlightbackground="#2f415a",
                    font=("Segoe UI", 11, "bold"),
                    anchor="w",
                    padx=14,
                    pady=10,
                )
                btn.pack(fill="x", padx=14, pady=5)
                self._nav_buttons[label] = btn
            tk.Label(inner, textvariable=self.active_nav_section, bg="#14233a", fg="#7ef0ff", font=("Segoe UI", 9, "bold"), padx=12, pady=5).pack(anchor="w", padx=16, pady=(0, 12))

            self._set_nav_section(self.active_nav_section.get())

        panel.bind("<Configure>", lambda _event: draw_panel())
        draw_panel()

    def _build_right_panel(self, parent: tk.Frame) -> None:
        """Build the right-side glass panel with compact training status cards."""

        canvas = tk.Canvas(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)

        def draw_panel() -> None:
            canvas.delete("all")
            w = int(canvas.winfo_width() or canvas.winfo_reqwidth() or parent.winfo_width() or self.right_panel_width)
            h = int(canvas.winfo_height() or canvas.winfo_reqheight() or parent.winfo_height() or 200)
            r = 22
            bg = "#0c1324"
            try:
                canvas.create_rectangle(0 + r, 0, w - r, h, fill=bg, outline=bg)
                canvas.create_rectangle(0, 0 + r, w, h - r, fill=bg, outline=bg)
                canvas.create_arc(0, 0, 0 + 2 * r, 0 + 2 * r, start=90, extent=90, fill=bg, outline=bg)
                canvas.create_arc(w - 2 * r, 0, w, 0 + 2 * r, start=0, extent=90, fill=bg, outline=bg)
                canvas.create_arc(0, h - 2 * r, 0 + 2 * r, h, start=180, extent=90, fill=bg, outline=bg)
                canvas.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=bg, outline=bg)
            except Exception:
                canvas.create_rectangle(0, 0, w, h, fill=bg, outline=bg)

            inner = tk.Frame(canvas, bg=bg)
            canvas.create_window(w // 2, h // 2, window=inner, width=max(1, w - 18), height=max(1, h - 18))

            tk.Label(inner, textvariable=self.workout_state, bg=bg, fg="#2ee6a6", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=16, pady=(16, 2))
            tk.Label(inner, textvariable=self.workout_hint, bg=bg, fg="#8aa0bf", font=("Segoe UI", 9), wraplength=270, justify="left").pack(anchor="w", padx=16, pady=(0, 14))

            tk.Button(
                inner,
                textvariable=self.primary_action_label,
                command=self.callbacks.on_primary_action,
                bg="#1a8cff",
                fg="#ffffff",
                activebackground="#2f98ff",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                font=("Segoe UI", 11, "bold"),
                padx=18,
                pady=10,
            ).pack(fill="x", padx=12, pady=(0, 10))

            rep_card = tk.Frame(inner, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
            rep_card.pack(fill="x", padx=12, pady=(0, 10))
            tk.Label(rep_card, text="POWTÓRZENIA / TEMPO", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
            tk.Label(rep_card, textvariable=self.rep_count_text, bg="#10192c", fg="#52f0c5", font=("Segoe UI", 24, "bold")).pack(anchor="w", padx=14)
            tk.Label(rep_card, textvariable=self.elapsed_text, bg="#10192c", fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(0, 2))
            tk.Label(rep_card, textvariable=self.rep_tempo_text, bg="#10192c", fg="#7ef0ff", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(0, 12))

            form_card = tk.Frame(inner, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
            form_card.pack(fill="x", padx=12, pady=(0, 10))
            tk.Label(form_card, text="POZYCJA", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
            tk.Label(form_card, textvariable=self.pose_side, bg="#10192c", fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=14)
            
            tk.Label(form_card, text="Kolano Przód (wąsko/szeroko)", bg="#10192c", fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(6, 0))
            tk.Label(form_card, textvariable=self.knee_angle_front_text, bg="#10192c", fg="#7ef0ff", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14)
            
            tk.Label(form_card, text="Kolano Bok (zgięcie)", bg="#10192c", fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(6, 0))
            tk.Label(form_card, textvariable=self.knee_angle_side_text, bg="#10192c", fg="#7ef0ff", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14)
            
            tk.Label(form_card, text="Plecy-biodro-kolano", bg="#10192c", fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(6, 0))
            tk.Label(form_card, textvariable=self.upper_angle_text, bg="#10192c", fg="#7ef0ff", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(0, 12))

            feedback = tk.Frame(inner, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
            feedback.pack(fill="both", expand=True, padx=12, pady=(0, 12))
            tk.Label(feedback, text="INFORMACJE ZWROTNE", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 8))
            tk.Label(feedback, textvariable=self.feedback_label, bg="#10192c", fg="#cfe1f5", font=("Segoe UI", 9), wraplength=260, justify="left").pack(anchor="w", padx=14, pady=(0, 8))
            tk.Label(feedback, textvariable=self.hand_signal_text, bg="#10192c", fg="#8aa0bf", font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(0, 12))

        canvas.bind("<Configure>", lambda _event: draw_panel())
        draw_panel()

    def _update_tolerance(self, axis: str, raw_value: float | str) -> None:
        """Clamp, render and propagate one tolerance value."""

        if self._suspend_tolerance_callbacks:
            return

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return

        value = max(1.0, min(45.0, value))
        if axis == "front":
            self.front_tolerance_scale.set(value)
            self.front_tolerance_var.set(f"{value:.1f}")
        elif axis == "side":
            self.side_tolerance_scale.set(value)
            self.side_tolerance_var.set(f"{value:.1f}")
        elif axis == "side_back":
            self.side_back_tolerance_scale.set(value)
            self.side_back_tolerance_var.set(f"{value:.1f}")
        else:
            return

        self.callbacks.on_angle_tolerance_changed(axis, value)

    def _adjust_tolerance(self, axis: str, delta: float) -> None:
        """Increment or decrement tolerance via large + / - buttons."""

        if axis == "front":
            current = self.front_tolerance_scale.get()
        elif axis == "side":
            current = self.side_tolerance_scale.get()
        elif axis == "side_back":
            current = self.side_back_tolerance_scale.get()
        else:
            return
        self._update_tolerance(axis, current + delta)

    def _apply_tolerance_from_entry(self, axis: str, _event: tk.Event | None = None) -> None:
        """Commit tolerance typed manually in the entry field."""

        if axis == "front":
            variable = self.front_tolerance_var
        elif axis == "side":
            variable = self.side_tolerance_var
        else:
            variable = self.side_back_tolerance_var
        raw = variable.get().strip().replace(",", ".")
        self._update_tolerance(axis, raw)

    def _build_settings_page(self, parent: tk.Frame) -> None:
        """Build the settings screen with angle thresholds and camera assignment."""

        card = tk.Frame(parent, bg="#0d1527", highlightthickness=1, highlightbackground="#27324a", bd=0)
        card.pack(fill="both", expand=True, padx=6, pady=6)

        scroll_canvas = tk.Canvas(card, bg="#0d1527", highlightthickness=0, bd=0)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(card, orient="vertical", command=scroll_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        content = tk.Frame(scroll_canvas, bg="#0d1527")
        content_window = scroll_canvas.create_window((0, 0), window=content, anchor="nw")

        def _sync_settings_scroll_region(_event: tk.Event | None = None) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
            scroll_canvas.itemconfigure(content_window, width=scroll_canvas.winfo_width())

        content.bind("<Configure>", _sync_settings_scroll_region)
        scroll_canvas.bind("<Configure>", _sync_settings_scroll_region)

        tk.Label(content, text="Ustawienia", bg="#0d1527", fg="#f5fbff", font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(
            content,
            text="Dostosuj tolerancję przodu, boku nogi i boku pleców oraz przypisanie kamer do widoków.",
            bg="#0d1527",
            fg="#8aa0bf",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=20, pady=(0, 14))

        status_box = tk.Frame(content, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
        status_box.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(status_box, text="Status kamer", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(status_box, textvariable=self.connection_label, bg="#10192c", fg="#9eb0c9", font=("Segoe UI", 10), wraplength=760, justify="left").pack(anchor="w", padx=14, pady=(0, 12))

        tolerance_box = tk.Frame(content, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
        tolerance_box.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(tolerance_box, text="Kąt dopuszczalny (± stopni)", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 8))
        front_card = tk.Frame(tolerance_box, bg="#0f1a2f", highlightthickness=1, highlightbackground="#314058", bd=0)
        front_card.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(front_card, text="Przód", bg="#0f1a2f", fg="#cfe1f5", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        front_controls = tk.Frame(front_card, bg="#0f1a2f")
        front_controls.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(
            front_controls,
            text="−",
            command=lambda: self._adjust_tolerance("front", -0.5),
            bg="#132743",
            fg="#e8f3ff",
            activebackground="#1a355a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 14, "bold"),
            padx=14,
            pady=4,
            width=3,
        ).pack(side="left")
        front_scale = tk.Scale(
            front_controls,
            from_=1.0,
            to=45.0,
            orient="horizontal",
            resolution=0.5,
            showvalue=False,
            variable=self.front_tolerance_scale,
            command=lambda value: self._update_tolerance("front", value),
            bg="#0f1a2f",
            fg="#cfe1f5",
            troughcolor="#1c2f4f",
            activebackground="#35d0ff",
            highlightthickness=0,
            sliderlength=22,
        )
        front_scale.pack(side="left", fill="x", expand=True, padx=10)
        tk.Button(
            front_controls,
            text="+",
            command=lambda: self._adjust_tolerance("front", 0.5),
            bg="#132743",
            fg="#e8f3ff",
            activebackground="#1a355a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 14, "bold"),
            padx=14,
            pady=4,
            width=3,
        ).pack(side="left")
        front_entry = tk.Entry(
            front_controls,
            textvariable=self.front_tolerance_var,
            width=6,
            justify="center",
            bg="#0b1020",
            fg="#dbe7f6",
            insertbackground="#dbe7f6",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#314058",
            font=("Segoe UI", 11, "bold"),
        )
        front_entry.pack(side="left", padx=(10, 0))
        tk.Label(front_controls, text="°", bg="#0f1a2f", fg="#cfe1f5", font=("Segoe UI", 11, "bold")).pack(side="left", padx=(4, 0))
        front_entry.bind("<Return>", lambda event: self._apply_tolerance_from_entry("front", event))
        front_entry.bind("<FocusOut>", lambda event: self._apply_tolerance_from_entry("front", event))

        side_card = tk.Frame(tolerance_box, bg="#0f1a2f", highlightthickness=1, highlightbackground="#314058", bd=0)
        side_card.pack(fill="x", padx=14, pady=(0, 12))
        tk.Label(side_card, text="Bok - noga", bg="#0f1a2f", fg="#cfe1f5", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        side_controls = tk.Frame(side_card, bg="#0f1a2f")
        side_controls.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(
            side_controls,
            text="−",
            command=lambda: self._adjust_tolerance("side", -0.5),
            bg="#132743",
            fg="#e8f3ff",
            activebackground="#1a355a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 14, "bold"),
            padx=14,
            pady=4,
            width=3,
        ).pack(side="left")
        side_scale = tk.Scale(
            side_controls,
            from_=1.0,
            to=45.0,
            orient="horizontal",
            resolution=0.5,
            showvalue=False,
            variable=self.side_tolerance_scale,
            command=lambda value: self._update_tolerance("side", value),
            bg="#0f1a2f",
            fg="#cfe1f5",
            troughcolor="#1c2f4f",
            activebackground="#35d0ff",
            highlightthickness=0,
            sliderlength=22,
        )
        side_scale.pack(side="left", fill="x", expand=True, padx=10)
        tk.Button(
            side_controls,
            text="+",
            command=lambda: self._adjust_tolerance("side", 0.5),
            bg="#132743",
            fg="#e8f3ff",
            activebackground="#1a355a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 14, "bold"),
            padx=14,
            pady=4,
            width=3,
        ).pack(side="left")
        side_entry = tk.Entry(
            side_controls,
            textvariable=self.side_tolerance_var,
            width=6,
            justify="center",
            bg="#0b1020",
            fg="#dbe7f6",
            insertbackground="#dbe7f6",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#314058",
            font=("Segoe UI", 11, "bold"),
        )
        side_entry.pack(side="left", padx=(10, 0))
        tk.Label(side_controls, text="°", bg="#0f1a2f", fg="#cfe1f5", font=("Segoe UI", 11, "bold")).pack(side="left", padx=(4, 0))
        side_entry.bind("<Return>", lambda event: self._apply_tolerance_from_entry("side", event))
        side_entry.bind("<FocusOut>", lambda event: self._apply_tolerance_from_entry("side", event))

        back_card = tk.Frame(tolerance_box, bg="#0f1a2f", highlightthickness=1, highlightbackground="#314058", bd=0)
        back_card.pack(fill="x", padx=14, pady=(0, 12))
        tk.Label(back_card, text="Bok - plecy", bg="#0f1a2f", fg="#cfe1f5", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        back_controls = tk.Frame(back_card, bg="#0f1a2f")
        back_controls.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(
            back_controls,
            text="−",
            command=lambda: self._adjust_tolerance("side_back", -0.5),
            bg="#132743",
            fg="#e8f3ff",
            activebackground="#1a355a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 14, "bold"),
            padx=14,
            pady=4,
            width=3,
        ).pack(side="left")
        back_scale = tk.Scale(
            back_controls,
            from_=1.0,
            to=45.0,
            orient="horizontal",
            resolution=0.5,
            showvalue=False,
            variable=self.side_back_tolerance_scale,
            command=lambda value: self._update_tolerance("side_back", value),
            bg="#0f1a2f",
            fg="#cfe1f5",
            troughcolor="#1c2f4f",
            activebackground="#35d0ff",
            highlightthickness=0,
            sliderlength=22,
        )
        back_scale.pack(side="left", fill="x", expand=True, padx=10)
        tk.Button(
            back_controls,
            text="+",
            command=lambda: self._adjust_tolerance("side_back", 0.5),
            bg="#132743",
            fg="#e8f3ff",
            activebackground="#1a355a",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            font=("Segoe UI", 14, "bold"),
            padx=14,
            pady=4,
            width=3,
        ).pack(side="left")
        back_entry = tk.Entry(
            back_controls,
            textvariable=self.side_back_tolerance_var,
            width=6,
            justify="center",
            bg="#0b1020",
            fg="#dbe7f6",
            insertbackground="#dbe7f6",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#314058",
            font=("Segoe UI", 11, "bold"),
        )
        back_entry.pack(side="left", padx=(10, 0))
        tk.Label(back_controls, text="°", bg="#0f1a2f", fg="#cfe1f5", font=("Segoe UI", 11, "bold")).pack(side="left", padx=(4, 0))
        back_entry.bind("<Return>", lambda event: self._apply_tolerance_from_entry("side_back", event))
        back_entry.bind("<FocusOut>", lambda event: self._apply_tolerance_from_entry("side_back", event))

        voice_box = tk.Frame(content, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
        voice_box.pack(fill="x", padx=16, pady=(0, 12))
        voice_label = tk.Label(voice_box, text="Komunikaty głosowe", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold"))
        voice_label.pack(anchor="w", padx=14, pady=(12, 8))
        voice_frame = tk.Frame(voice_box, bg="#10192c")
        voice_frame.pack(fill="x", padx=14, pady=(0, 12))
        voice_checkbox = tk.Checkbutton(
            voice_frame,
            text="Włącz komunikaty głosowe",
            variable=self.voice_feedback_enabled,
            command=lambda: self.callbacks.on_voice_feedback_changed(self.voice_feedback_enabled.get()),
            bg="#10192c",
            fg="#cfe1f5",
            activebackground="#10192c",
            activeforeground="#ffffff",
            selectcolor="#10192c",
            font=("Segoe UI", 10),
            relief="flat",
            bd=0,
        )
        voice_checkbox.pack(anchor="w")

        sources_box = tk.Frame(content, bg="#10192c", highlightthickness=1, highlightbackground="#314058", bd=0)
        sources_box.pack(fill="x", padx=16, pady=(0, 14))
        tk.Label(sources_box, text="Przypisanie kamer do widoków", bg="#10192c", fg="#dbe7f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 8))

        self._settings_camera_vars = []
        self._settings_camera_boxes = []
        source_values = ["Brak"]
        for slot_index in range(self.config.max_camera_slots):
            row = tk.Frame(sources_box, bg="#10192c")
            row.pack(fill="x", padx=14, pady=(0, 8))
            tk.Label(row, text=f"Widok {slot_index + 1}", bg="#10192c", fg="#cfe1f5", font=("Segoe UI", 10)).pack(side="left")
            variable = tk.StringVar(value="Brak")
            combo = ttk.Combobox(
                row,
                textvariable=variable,
                values=source_values,
                width=18,
                state="readonly",
                style="CameraSelect.TCombobox",
            )
            combo.pack(side="right")
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _event, index=slot_index, var=variable: self.callbacks.on_camera_source_changed(index, var.get()),
            )
            self._settings_camera_vars.append(variable)
            self._settings_camera_boxes.append(combo)

    def _set_nav_section(self, section: str) -> None:
        """Highlight the selected sidebar section."""

        self.active_nav_section.set(section)
        for label, button in self._nav_buttons.items():
            if label == section:
                button.configure(bg="#17304f", fg="#ffffff", highlightbackground="#6de7ff")
            else:
                button.configure(bg="#101a30", fg="#d6e1f2", highlightbackground="#2f415a")

        try:
            self.callbacks.on_nav_section_changed(section)
        except Exception:
            pass

        if not self.camera_only:
            return

        if not hasattr(self, "_camera_page") or not hasattr(self, "_settings_page") or not hasattr(self, "_history_page"):
            return

        if section == "Ustawienia":
            self._camera_page.place_forget()
            self._history_page.place_forget()
            self._settings_page.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
        elif section == "Historia":
            self._camera_page.place_forget()
            self._settings_page.place_forget()
            self._history_page.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)
        else:
            self._settings_page.place_forget()
            self._history_page.place_forget()
            self._camera_page.place(relx=0.5, rely=0.5, anchor="center", relwidth=1.0, relheight=1.0)

    def _build_history_page(self, parent: tk.Frame) -> None:
        """Build a simple page with saved results from previous workout series."""

        card = tk.Frame(parent, bg="#0f172a", highlightthickness=1, highlightbackground="#314058", bd=0)
        card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.96, relheight=0.96)

        tk.Label(
            card,
            text="Historia ćwiczeń",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(
            card,
            text="Zapisane wyniki poprzednich serii: powtórzenia, średnia jakość, średni czas powtórzenia i całkowity czas serii.",
            bg="#0f172a",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
            wraplength=920,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        list_wrap = tk.Frame(card, bg="#0f172a")
        list_wrap.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self._history_canvas = tk.Canvas(list_wrap, bg="#0f172a", highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(list_wrap, orient="vertical", command=self._history_canvas.yview)
        self._history_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._history_canvas.pack(side="left", fill="both", expand=True)

        self._history_cards_frame = tk.Frame(self._history_canvas, bg="#0f172a")
        self._history_window_id = self._history_canvas.create_window((0, 0), window=self._history_cards_frame, anchor="nw")

        def _resize_cards(_event: tk.Event) -> None:
            width = max(100, self._history_canvas.winfo_width())
            self._history_canvas.itemconfigure(self._history_window_id, width=width)
            self._history_canvas.configure(scrollregion=self._history_canvas.bbox("all"))

        self._history_cards_frame.bind("<Configure>", _resize_cards)
        self._history_canvas.bind("<Configure>", _resize_cards)

        pending = getattr(self, "_pending_history", None)
        if pending:
            self.set_history(pending)
        pending_records = getattr(self, "_pending_history_records", None)
        if pending_records is not None:
            self.set_history_records(pending_records)

    def _build_main(self, parent: ttk.Frame) -> None:
        """Build the main cards for camera preview, controls, feedback and stats."""

        parent.columnconfigure(0, weight=5)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=10, minsize=620)
        parent.rowconfigure(1, weight=1, minsize=160)

        camera_card = ttk.Frame(parent, style="App.TFrame", padding=0)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16), pady=(0, 16))
        controls_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        controls_card.grid(row=0, column=1, sticky="nsew", pady=(0, 16))
        feedback_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        feedback_card.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        stats_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        stats_card.grid(row=1, column=1, sticky="nsew")

        self._build_camera_card(camera_card)
        self._build_controls_card(controls_card)
        self._build_feedback_card(feedback_card)
        self._build_stats_card(stats_card)

    def _build_camera_card(self, parent: ttk.Frame) -> None:
        """Build the side-by-side camera preview area."""
        if not self.camera_only:
            ttk.Label(parent, text="hip thrust", style="Section.TLabel").pack(anchor="w")
            ttk.Label(
                parent,
                text="Opis ćwiczenia hip thrust",
                style="CardText.TLabel",
                wraplength=760,
                justify="left",
            ).pack(anchor="w", pady=(6, 14))

        view_wrap = ttk.Frame(parent, style="Card.TFrame")
        view_wrap.pack(fill="both", expand=True)
        view_wrap.columnconfigure(0, weight=1)
        view_wrap.rowconfigure(0, weight=1, minsize=260)
        view_wrap.rowconfigure(1, weight=1, minsize=260)

        top_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(0, 0, 0, 6))
        bottom_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(0, 6, 0, 0))
        top_panel.grid(row=0, column=0, sticky="nsew")
        bottom_panel.grid(row=1, column=0, sticky="nsew")

        self._panels = [
            self._create_camera_panel(top_panel, "Widok 1", 0),
            self._create_camera_panel(bottom_panel, "Widok 2", 1),
        ]

        bar_thick = getattr(self.config, "camera_edge_bar_thickness", 8)
        try:
            if getattr(self, "camera_only", False) and hasattr(self, "_camera_container") and isinstance(self._camera_container, tk.Canvas):
                canvas = self._camera_container

                def draw_bars() -> None:
                    w = int(canvas.winfo_width() or canvas.winfo_reqwidth() or 0)
                    h = int(canvas.winfo_height() or canvas.winfo_reqheight() or 0)
                    if w <= 0 or h <= 0:
                        canvas.after(50, draw_bars)
                        return

                    r = getattr(self.config, "camera_corner_radius", 16)
                    bg = self.config.panel_holder_bg

                    def rounded_rect(x0, y0, x1, y1, radius):
                        try:
                            canvas.create_rectangle(x0 + radius, y0, x1 - radius, y1, fill=bg, outline=bg)
                            canvas.create_rectangle(x0, y0 + radius, x1, y1 - radius, fill=bg, outline=bg)
                            canvas.create_arc(x0, y0, x0 + 2 * radius, y0 + 2 * radius, start=90, extent=90, fill=bg, outline=bg)
                            canvas.create_arc(x1 - 2 * radius, y0, x1, y0 + 2 * radius, start=0, extent=90, fill=bg, outline=bg)
                            canvas.create_arc(x0, y1 - 2 * radius, x0 + 2 * radius, y1, start=180, extent=90, fill=bg, outline=bg)
                            canvas.create_arc(x1 - 2 * radius, y1 - 2 * radius, x1, y1, start=270, extent=90, fill=bg, outline=bg)
                        except Exception:
                            canvas.create_rectangle(x0, y0, x1, y1, fill=bg, outline=bg)

                    rounded_rect(0, 0, w, bar_thick, min(r, bar_thick))
                    rounded_rect(0, h - bar_thick, w, h, min(r, bar_thick))
                    rounded_rect(0, 0, bar_thick, h, min(r, bar_thick))
                    rounded_rect(w - bar_thick, 0, w, h, min(r, bar_thick))

                canvas.after(50, draw_bars)
            else:
                top_bar = tk.Frame(parent, bg=self.config.panel_holder_bg, height=bar_thick)
                top_bar.place(relx=0, rely=0, relwidth=1, anchor="nw")
                bottom_bar = tk.Frame(parent, bg=self.config.panel_holder_bg, height=bar_thick)
                bottom_bar.place(relx=0, rely=1, relwidth=1, anchor="sw")
                left_bar = tk.Frame(parent, bg=self.config.panel_holder_bg, width=bar_thick)
                left_bar.place(relx=0, rely=0, relheight=1, anchor="nw")
                right_bar = tk.Frame(parent, bg=self.config.panel_holder_bg, width=bar_thick)
                right_bar.place(relx=1, rely=0, relheight=1, anchor="ne")
        except Exception:
            pass

    def _create_camera_panel(self, parent: ttk.Frame, title: str, slot_index: int) -> CameraPanelWidgets:
        """Create one preview panel with a camera source combobox."""
        wrapper = tk.Frame(parent, bg=self.config.panel_wrapper_bg, highlightthickness=0, bd=0)
        wrapper.pack(fill="both", expand=True)

        if not self.camera_only:
            header = tk.Frame(wrapper, bg=self.config.header_bg, padx=8, pady=6)
            header.pack(fill="x", pady=(0, 8))
            tk.Label(header, text=title, bg=self.config.header_bg, fg=self.config.text_secondary, font=("Segoe UI", 12, "bold")).pack(side="left")

        picker_var = tk.StringVar(value="Brak")
        picker_parent = header if (not self.camera_only) else wrapper
        picker = ttk.Combobox(
            picker_parent,
            textvariable=picker_var,
            values=["Brak"],
            width=12,
            state="readonly",
            style="CameraSelect.TCombobox",
        )
        if not self.camera_only:
            picker.pack(side="left", padx=(12, 0))
        picker.bind("<<ComboboxSelected>>", lambda _event, index=slot_index, variable=picker_var: self.callbacks.on_camera_source_changed(index, variable.get()))

        status = tk.Label(picker_parent, text="● SZUKANIE", bg=self.config.header_bg if (not self.camera_only) else self.config.panel_wrapper_bg, fg=self.config.accent_orange, font=("Segoe UI", 10, "bold"))
        if not self.camera_only:
            status.pack(side="right")

        holder = tk.Frame(wrapper, bg=self.config.panel_holder_bg)
        holder.pack(fill="both", expand=True)
        holder.pack_propagate(False)

        image_area = tk.Frame(holder, bg=self.config.panel_holder_bg)
        image_area.place(relx=0.5, rely=0.5, anchor="center")

        def _enforce_aspect(event: tk.Event) -> None:
            w = event.width
            h = event.height
            pref_h = int(w * 9 / 16)
            if pref_h <= h:
                new_w = w
                new_h = pref_h
            else:
                new_h = h
                new_w = int(h * 16 / 9)
            image_area.place_configure(width=new_w, height=new_h)

        holder.bind("<Configure>", _enforce_aspect)

        image_label = tk.Label(image_area, bg=self.config.panel_holder_bg, anchor="center")
        image_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        return CameraPanelWidgets(
            frame=wrapper,
            holder=holder,
            inner_image_area=image_area,
            image=image_label,
            status=status,
            picker=picker,
            picker_var=picker_var,
            slot_index=slot_index,
        )

    def _build_controls_card(self, parent: ttk.Frame) -> None:
        """Build the session control buttons and technique tips."""

        ttk.Label(parent, text="Panel sterowania", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Przyciski steruja sesja treningowa. To tutaj pozniej podepnie sie logike komunikacji z kamera, analiza i zapis danych.",
            style="CardText.TLabel",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        controls = [
            ("Start sesji", self.config.accent_cyan, self.callbacks.on_start_session),
            ("Pauza / wznow", self.config.panel_wrapper_bg, self.callbacks.on_toggle_pause),
            ("Zakoncz", self.config.danger_color, self.callbacks.on_end_session),
            ("Zapisz wynik", self.config.accent_green, self.callbacks.on_save_result),
        ]
        for label, color, command in controls:
            btn = tk.Button(
                parent,
                text=label,
                command=command,
                bg=color,
                fg="#ffffff",
                activebackground=color,
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                font=("Segoe UI", 11, "bold"),
                height=2,
            )
            btn.pack(fill="x", pady=6)

        ttk.Label(parent, text="Sugestie techniczne", style="Section.TLabel").pack(anchor="w", pady=(18, 8))
        ttk.Label(parent, text="\n".join(["Sugestie"]), style="CardText.TLabel", justify="left").pack(anchor="w")

    def _build_feedback_card(self, parent: ttk.Frame) -> None:
        """Build the feedback and system event cards."""

        ttk.Label(parent, text="Ocena techniki i komunikaty", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            textvariable=self.feedback_label,
            style="CardText.TLabel",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(6, 10))

        content = ttk.Frame(parent, style="Card.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)

        left = tk.Frame(content, bg=self.config.card_color)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(content, bg=self.config.card_color)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(left, text="Ocena techniki", bg=self.config.card_color, fg=self.config.text_secondary, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))

        tk.Label(right, text="Komunikaty systemu", bg=self.config.card_color, fg=self.config.text_secondary, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.event_log = tk.Listbox(right, bg=self.config.list_bg, fg=self.config.text_secondary, highlightthickness=0, selectbackground=self.config.chip_bg, borderwidth=0, activestyle="none")
        self.event_log.pack(fill="both", expand=True)

    def _build_stats_card(self, parent: ttk.Frame) -> None:
        """Build the metrics and placeholder chart card."""

        ttk.Label(parent, text="Statystyki sesji", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Tutaj trafia podsumowanie bieżącej sesji i historia zapisów lokalnych.",
            style="CardText.TLabel",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(6, 10))

        stats = tk.Frame(parent, bg=self.config.card_color)
        stats.pack(fill="x")
        for label, variable in [
            ("Powtorzenia", self.metric_reps),
            ("Czas sesji", self.metric_time),
            ("Jasnosc oceny", self.metric_quality),
            ("Ostrzezenia", self.metric_warnings),
            ("Poprawność kolan", self.metric_knee_error),
        ]:
            row = tk.Frame(stats, bg=self.config.card_color)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=self.config.card_color, fg=self.config.text_muted, font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, textvariable=variable, bg=self.config.card_color, fg=self.config.text_primary, font=("Segoe UI", 10, "bold")).pack(side="right")

        chart = tk.Canvas(parent, bg=self.config.list_bg, highlightthickness=0, height=80)
        chart.pack(fill="x", pady=(14, 0))
        chart.create_rectangle(8, 10, 332, 68, outline=self.config.panel_wrapper_bg, width=2)
        chart.create_text(170, 39, text="Miejsce na wykres postepow", fill=self.config.text_muted, font=("Segoe UI", 10, "bold"))
        self.chart_canvas = chart
