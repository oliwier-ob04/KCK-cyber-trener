"""Tkinter view that renders the Cyber Trener desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import tkinter as tk
from tkinter import ttk

from exercises.hip_thrust import ExerciseProfile
from scoring.engine import ScoreSnapshot
from utils.config import AppConfig


@dataclass(frozen=True)
class ViewCallbacks:
    """Callbacks injected from the controller layer."""

    on_start_session: Callable[[], None]
    on_toggle_pause: Callable[[], None]
    on_end_session: Callable[[], None]
    on_save_result: Callable[[], None]
    on_close: Callable[[], None]
    on_camera_source_changed: Callable[[int, str], None]


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


class CyberTrainerView(tk.Tk):
    """Own the Tk root window and expose small update methods for the controller."""

    def __init__(
        self,
        config: AppConfig,
        exercise: ExerciseProfile,
        callbacks: ViewCallbacks,
    ) -> None:
        """Create the root window and build the layout."""

        super().__init__()
        self.config = config
        self.exercise = exercise
        self.callbacks = callbacks

        self.title(config.window_title)
        self.state(config.window_state)
        self.configure(bg=config.background_color)
        self.protocol("WM_DELETE_WINDOW", self.callbacks.on_close)

        self.source_label = tk.StringVar(value="Demo mode")
        self.connection_label = tk.StringVar(value="Kamera nieaktywna")
        self.feedback_label = tk.StringVar(value=exercise.default_feedback)
        self.metric_reps = tk.StringVar(value="0")
        self.metric_time = tk.StringVar(value="00:00")
        self.metric_quality = tk.StringVar(value=f"{config.default_quality}%")
        self.metric_warnings = tk.StringVar(value="0")
        self.metric_letter = tk.StringVar(value="Brak")

        self._camera_photos: dict[int, object] = {}
        self._panels: list[CameraPanelWidgets] = []
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

    def set_detected_letter(self, letter: str) -> None:
        """Update the detected gesture letter."""

        self.metric_letter.set(letter)

    def set_history(self, entries: Sequence[str]) -> None:
        """Replace the history list with formatted session entries."""

        self.history_box.delete(0, tk.END)
        if not entries:
            self.history_box.insert(tk.END, "Brak zapisanej historii")
            return
        for entry in entries:
            self.history_box.insert(tk.END, entry)

    def append_event(self, message: str) -> None:
        """Push a new message into the event log."""

        timestamp = time.strftime("%H:%M:%S")
        self.event_log.insert(0, f"[{timestamp}] {message}")
        if self.event_log.size() > 12:
            self.event_log.delete(tk.END)

    def set_camera_sources(self, options: Sequence[str], selected_labels: Sequence[str]) -> None:
        """Refresh all camera comboboxes with new available sources."""

        for panel, selected in zip(self._panels, selected_labels, strict=False):
            panel.picker.configure(values=list(options))
            if selected not in options:
                selected = "Brak"
            panel.picker_var.set(selected)

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
        panel.image.configure(image=photo, text="", fg="#9aa9c2")
        panel.image.image = photo  # type: ignore[attr-defined]
        panel.status.configure(text=f"● LIVE | {status_text}" if online else "● OFFLINE")
        panel.status.configure(fg="#34d399" if online else "#f87171")
        self._camera_photos[slot_index] = photo

    def _build_ui(self) -> None:
        """Create the page shell, sidebar and cards."""

        root = ttk.Frame(self, style="App.TFrame", padding=(20, 8, 20, 20))
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text=self.config.app_name, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Dzialajacy szkielet aplikacji do komunikacji komputer <-> kamera oraz oceny techniki hip thrust.",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(root, style="App.TFrame")
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=280)
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)

        main = ttk.Frame(body, style="App.TFrame")
        main.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)
        self._build_main(main)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        """Build the sidebar status, navigation and history widgets."""

        top = ttk.Frame(parent, style="Sidebar.TFrame", padding=18)
        top.pack(fill="x")

        title_box = tk.Frame(top, bg="#1e293b")
        title_box.pack(fill="x", pady=(0, 14))
        tk.Label(title_box, text=self.config.app_name, bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Label(
            title_box,
            text="Prototyp do dalszego rozwoju. Kamera i analiza sa podlaczone przez warstwe backendu.",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        ttk.Label(top, text="Stan polaczenia", background=self.config.sidebar_color, foreground="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 4))
        ttk.Label(top, textvariable=self.connection_label, background=self.config.sidebar_color, foreground="#94a3b8", font=("Segoe UI", 10), wraplength=230).pack(anchor="w")
        ttk.Label(top, textvariable=self.source_label, style="Chip.TLabel").pack(anchor="w", pady=(10, 0))

        ttk.Label(top, text="Obszar roboczy", background=self.config.sidebar_color, foreground="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(18, 4))
        ttk.Label(
            top,
            text="• podglad kamery\n• start / pauza sesji\n• zapis historii\n• podstawowe metryki",
            background=self.config.sidebar_color,
            foreground="#94a3b8",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        nav = ttk.Frame(parent, style="Sidebar.TFrame", padding=(18, 12))
        nav.pack(fill="x")
        ttk.Label(nav, text="Ekrany aplikacji", background=self.config.sidebar_color, foreground="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        for label in ["Panel glowny", "Analiza techniki", "Historia treningow", "Statystyki", "Ustawienia"]:
            ttk.Button(nav, text=label, style="Nav.TButton").pack(fill="x", pady=4)

        self.history_box = tk.Listbox(parent, bg="#0b1220", fg="#cbd5e1", highlightthickness=0, selectbackground="#1e293b", borderwidth=0, activestyle="none")
        self.history_box.pack(fill="both", expand=True, padx=18, pady=(6, 18))

    def _build_main(self, parent: ttk.Frame) -> None:
        """Build the main cards for camera preview, controls, feedback and stats."""

        parent.columnconfigure(0, weight=5)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=10, minsize=620)
        parent.rowconfigure(1, weight=1, minsize=160)

        camera_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
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

        ttk.Label(parent, text=self.exercise.title, style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text=self.exercise.description,
            style="CardText.TLabel",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        view_wrap = ttk.Frame(parent, style="Card.TFrame")
        view_wrap.pack(fill="both", expand=True)
        view_wrap.columnconfigure(0, weight=1)
        view_wrap.columnconfigure(1, weight=1)
        view_wrap.rowconfigure(0, weight=1, minsize=420)

        left_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(0, 0, 10, 0))
        right_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(10, 0, 0, 0))
        left_panel.grid(row=0, column=0, sticky="nsew")
        right_panel.grid(row=0, column=1, sticky="nsew")

        self._panels = [
            self._create_camera_panel(left_panel, "Widok 1", 0),
            self._create_camera_panel(right_panel, "Widok 2", 1),
        ]

    def _create_camera_panel(self, parent: ttk.Frame, title: str, slot_index: int) -> CameraPanelWidgets:
        """Create one preview panel with a camera source combobox."""

        wrapper = tk.Frame(parent, bg="#08111f", highlightthickness=1, highlightbackground="#253655")
        wrapper.pack(fill="both", expand=True)

        header = tk.Frame(wrapper, bg="#071326", padx=8, pady=6)
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text=title, bg="#071326", fg="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(side="left")

        picker_var = tk.StringVar(value="Brak")
        picker = ttk.Combobox(
            header,
            textvariable=picker_var,
            values=["Brak"],
            width=12,
            state="readonly",
            style="CameraSelect.TCombobox",
        )
        picker.pack(side="left", padx=(12, 0))
        picker.bind("<<ComboboxSelected>>", lambda _event, index=slot_index, variable=picker_var: self.callbacks.on_camera_source_changed(index, variable.get()))

        status = tk.Label(header, text="● SZUKANIE", bg="#071326", fg="#f59e0b", font=("Segoe UI", 10, "bold"))
        status.pack(side="right")

        holder = tk.Frame(wrapper, bg="#000000")
        holder.pack(fill="both", expand=True)
        holder.pack_propagate(False)

        image_label = tk.Label(holder, bg="#000000", anchor="center")
        image_label.place(relx=0.5, rely=0.5, anchor="center")

        return CameraPanelWidgets(
            frame=wrapper,
            holder=holder,
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
            ("Start sesji", "#2563eb", self.callbacks.on_start_session),
            ("Pauza / wznow", "#334155", self.callbacks.on_toggle_pause),
            ("Zakoncz", "#dc2626", self.callbacks.on_end_session),
            ("Zapisz wynik", "#059669", self.callbacks.on_save_result),
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
        ttk.Label(parent, text="\n".join(self.exercise.tips), style="CardText.TLabel", justify="left").pack(anchor="w")

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

        left = tk.Frame(content, bg="#111c33")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = tk.Frame(content, bg="#111c33")
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(left, text="Ocena techniki", bg="#111c33", fg="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        for row_def in self.exercise.technique_rows:
            row = tk.Frame(left, bg="#111c33")
            row.pack(fill="x", pady=4)
            tk.Label(row, text=row_def.label, bg="#111c33", fg="#94a3b8", font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, text=row_def.value, bg="#111c33", fg=row_def.color, font=("Segoe UI", 10, "bold")).pack(side="right")

        tk.Label(right, text="Komunikaty systemu", bg="#111c33", fg="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.event_log = tk.Listbox(right, bg="#0b1220", fg="#cbd5e1", highlightthickness=0, selectbackground="#1e293b", borderwidth=0, activestyle="none")
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

        stats = tk.Frame(parent, bg="#111c33")
        stats.pack(fill="x")
        for label, variable in [
            ("Powtorzenia", self.metric_reps),
            ("Czas sesji", self.metric_time),
            ("Jasnosc oceny", self.metric_quality),
            ("Ostrzezenia", self.metric_warnings),
            ("Rozpoznana litera", self.metric_letter),
        ]:
            row = tk.Frame(stats, bg="#111c33")
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg="#111c33", fg="#94a3b8", font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, textvariable=variable, bg="#111c33", fg="#f8fafc", font=("Segoe UI", 10, "bold")).pack(side="right")

        chart = tk.Canvas(parent, bg="#0b1220", highlightthickness=0, height=80)
        chart.pack(fill="x", pady=(14, 0))
        chart.create_rectangle(8, 10, 332, 68, outline="#2b3a55", width=2)
        chart.create_text(170, 39, text="Miejsce na wykres postepow", fill="#64748b", font=("Segoe UI", 10, "bold"))
        self.chart_canvas = chart
