import json
import math
import random
import time
import tkinter as tk
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

if cv2 is not None:
    # Try to reduce OpenCV logging noise when probing devices (may not be available in all builds)
    try:
        # Newer OpenCV Python API
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        try:
            # Fallback older API
            cv2.setLogLevel(3)
        except Exception:
            # If neither available, we still catch exceptions during probing below
            pass


APP_DIR = Path(__file__).resolve().parent
HISTORY_FILE = APP_DIR / "cyber_trener_history.json"


@dataclass
class SessionResult:
    started_at: float
    ended_at: float
    repetitions: int
    warnings: int
    quality: int
    source: str

    @property
    def duration_seconds(self) -> int:
        return max(0, int(self.ended_at - self.started_at))


class SessionStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def append(self, result: SessionResult) -> None:
        items = self.load()
        items.insert(0, asdict(result) | {"duration_seconds": result.duration_seconds})
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


class CameraManager:
    def __init__(self, scan_max_index: int = 6):
        self.scan_max_index = scan_max_index
        self.available_devices: list[int] = []
        self.slot_sources: list[int | None] = [None, None]
        self.captures: dict[int, any] = {}
        self.refresh_devices()

    @property
    def available(self) -> bool:
        return cv2 is not None

    def refresh_devices(self) -> list[int]:
        self.available_devices = []
        if not self.available:
            return self.available_devices

        for index in range(self.scan_max_index):
            try:
                # Prefer DirectShow on Windows to avoid some backend noise
                try:
                    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                except Exception:
                    capture = cv2.VideoCapture(index)
                if capture is not None and capture.isOpened():
                    self.available_devices.append(index)
                if capture is not None:
                    capture.release()
            except Exception:
                # ignore noisy backend errors for out-of-range indices
                continue

        if self.slot_sources[0] is None and self.available_devices:
            self.slot_sources[0] = self.available_devices[0]
        if self.slot_sources[1] is None and len(self.available_devices) > 1:
            self.slot_sources[1] = self.available_devices[1]

        for slot_idx, device_index in enumerate(self.slot_sources):
            if device_index is not None and device_index not in self.available_devices:
                self.slot_sources[slot_idx] = None

        for device_index in list(self.captures.keys()):
            if device_index not in self.available_devices:
                self.captures[device_index].release()
                del self.captures[device_index]

        return self.available_devices

    def set_slot_source(self, slot_index: int, device_index: int | None) -> None:
        if slot_index < 0 or slot_index >= len(self.slot_sources):
            return
        self.slot_sources[slot_index] = device_index

    def _ensure_capture(self, device_index: int):
        if device_index in self.captures:
            return self.captures[device_index]

        try:
            try:
                capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            except Exception:
                capture = cv2.VideoCapture(device_index)
        except Exception:
            return None

        if not capture or not capture.isOpened():
            try:
                if capture:
                    capture.release()
            except Exception:
                pass
            return None

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.captures[device_index] = capture
        return capture

    def read(self, slot_index: int):
        if slot_index < 0 or slot_index >= len(self.slot_sources):
            return False, None, None

        device_index = self.slot_sources[slot_index]
        if device_index is None:
            return False, None, None

        capture = self._ensure_capture(device_index)
        if capture is None:
            return False, None, device_index

        ok, frame = capture.read()
        if not ok:
            capture.release()
            if device_index in self.captures:
                del self.captures[device_index]
            capture = self._ensure_capture(device_index)
            if capture is None:
                return False, None, device_index
            ok, frame = capture.read()

        return ok, frame, device_index

    def close(self) -> None:
        for capture in self.captures.values():
            if capture is not None:
                capture.release()
        self.captures = {}


class CyberTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cyber Trener - dzialajacy prototyp")
        self.state("zoomed")
        self.configure(bg="#0b1220")

        self.camera = CameraManager()
        self.store = SessionStore(HISTORY_FILE)

        self.session_active = False
        self.session_paused = False
        self.session_started_at = 0.0
        self.last_tick = time.time()
        self.rep_count = 0
        self.warning_count = 0
        self.quality_score = 91
        self.current_phase = 0.0
        self.repetition_state = "OPUSZCZANIE"
        self.source_label = tk.StringVar(value="Demo mode")
        self.connection_label = tk.StringVar(value="Kamera nieaktywna")
        self.feedback_label = tk.StringVar(value="Gotowy do uruchomienia sesji.")
        self.metric_reps = tk.StringVar(value="0")
        self.metric_time = tk.StringVar(value="00:00")
        self.metric_quality = tk.StringVar(value="91%")
        self.metric_warnings = tk.StringVar(value="0")
        self._camera_images: dict[str, ImageTk.PhotoImage | None] = {"front": None, "side": None}
        self._last_camera_scan_at = 0.0
        self.camera_option_values = ["Brak"]

        self._setup_style()
        self._build_ui()
        self._load_history()
        self._refresh_camera_devices(force=True)
        self._update_camera_status()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._update_loop)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("App.TFrame", background="#0b1220")
        style.configure("Sidebar.TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#111c33", relief="flat")
        style.configure("CardInner.TFrame", background="#111c33")
        style.configure("Section.TLabel", background="#111c33", foreground="#e2e8f0", font=("Segoe UI", 12, "bold"))
        style.configure("CardText.TLabel", background="#111c33", foreground="#cbd5e1", font=("Segoe UI", 10))
        style.configure("Meta.TLabel", background="#111c33", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#0b1220", foreground="#f8fafc", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#0b1220", foreground="#94a3b8", font=("Segoe UI", 10))
        style.configure("Chip.TLabel", background="#1e293b", foreground="#e2e8f0", font=("Segoe UI", 9, "bold"), padding=(10, 4))
        style.configure("Nav.TButton", background="#111c33", foreground="#e2e8f0", borderwidth=0, padding=(12, 10), font=("Segoe UI", 10, "bold"))
        style.configure("CameraSelect.TCombobox", fieldbackground="#0f172a", background="#0f172a", foreground="#e2e8f0", arrowsize=14)
        style.map("Nav.TButton", background=[("active", "#1f2e4d")], foreground=[("active", "#ffffff")])

    def _build_ui(self):
        root = ttk.Frame(self, style="App.TFrame", padding=(20, 8, 20, 20))
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill="x", pady=(0, 16))

        ttk.Label(header, text="Cyber Trener", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Dzialajacy szkielet aplikacji do komunikacji komputer <-> kamera oraz oceny techniki hip thrust.",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(root, style="App.TFrame")
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=240)
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)

        main = ttk.Frame(body, style="App.TFrame")
        main.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)
        self._build_main(main)

    def _build_sidebar(self, parent):
        top = ttk.Frame(parent, style="Sidebar.TFrame", padding=18)
        top.pack(fill="x")

        title_box = tk.Frame(top, bg="#1e293b")
        title_box.pack(fill="x", pady=(0, 14))
        tk.Label(title_box, text="Cyber Trener", bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Label(
            title_box,
            text="Prototyp do dalszego rozwoju. Kamera i analiza sa podlaczone przez warstwe backendu.",
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9),
            wraplength=220,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        ttk.Label(top, text="Stan polaczenia", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 4))
        ttk.Label(top, textvariable=self.connection_label, background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 10), wraplength=230).pack(anchor="w")
        ttk.Label(top, textvariable=self.source_label, style="Chip.TLabel").pack(anchor="w", pady=(10, 0))

        ttk.Label(top, text="Obszar roboczy", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(18, 4))
        ttk.Label(
            top,
            text="• podglad kamery\n• start / pauza sesji\n• zapis historii\n• podstawowe metryki",
            background="#0f172a",
            foreground="#94a3b8",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        nav = ttk.Frame(parent, style="Sidebar.TFrame", padding=(18, 12))
        nav.pack(fill="x")
        ttk.Label(nav, text="Ekrany aplikacji", background="#0f172a", foreground="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        for label in ["Panel glowny", "Analiza techniki", "Historia treningow", "Statystyki", "Ustawienia"]:
            ttk.Button(nav, text=label, style="Nav.TButton").pack(fill="x", pady=4)

        self.history_box = tk.Listbox(parent, bg="#0b1220", fg="#cbd5e1", highlightthickness=0, selectbackground="#1e293b", borderwidth=0, activestyle="none")
        self.history_box.pack(fill="both", expand=True, padx=18, pady=(6, 18))

    def _build_main(self, parent):
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

    def _build_camera_card(self, parent):
        ttk.Label(parent, text="Podglad z kamer + warstwa AR", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="To jest dzialajacy panel pod szkielet aplikacji. Jesli dostepne jest OpenCV, aplikacja pokazuje realny obraz z kamer. Gdy podlaczona jest tylko jedna kamera, drugi panel pozostaje czarny.",
            style="CardText.TLabel",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        view_wrap = ttk.Frame(parent, style="Card.TFrame")
        view_wrap.pack(fill="both", expand=True)
        view_wrap.columnconfigure(0, weight=1)
        view_wrap.columnconfigure(1, weight=1)
        view_wrap.rowconfigure(0, weight=1, minsize=420)

        front_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(0, 0, 10, 0))
        side_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(10, 0, 0, 0))
        front_panel.grid(row=0, column=0, sticky="nsew")
        side_panel.grid(row=0, column=1, sticky="nsew")

        self.front_view = self._create_camera_panel(front_panel, "KAMERA 1", slot_index=0)
        self.side_view = self._create_camera_panel(side_panel, "KAMERA 2", slot_index=1)

    def _create_camera_panel(self, parent, title, slot_index: int):
        wrapper = tk.Frame(parent, bg="#08111f", highlightthickness=1, highlightbackground="#253655")
        wrapper.pack(fill="both", expand=True)

        header = tk.Frame(wrapper, bg="#071326", padx=8, pady=6)
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text=title, bg="#071326", fg="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(side="left")

        picker_var = tk.StringVar(value="Brak")
        picker = ttk.Combobox(
            header,
            textvariable=picker_var,
            values=self.camera_option_values,
            width=12,
            state="readonly",
            style="CameraSelect.TCombobox",
        )
        picker.pack(side="left", padx=(12, 0))
        picker.bind("<<ComboboxSelected>>", lambda _event, s=slot_index, v=picker_var: self._on_camera_selected(s, v.get()))

        status = tk.Label(header, text="● SZUKANIE", bg="#071326", fg="#f59e0b", font=("Segoe UI", 10, "bold"))
        status.pack(side="right")

        image_holder = tk.Frame(wrapper, bg="#000000")
        image_holder.pack(fill="both", expand=True)
        image_holder.pack_propagate(False)

        image_label = tk.Label(image_holder, bg="#000000", anchor="center")
        image_label.place(relx=0.5, rely=0.5, anchor="center")
        return {"frame": wrapper, "holder": image_holder, "image": image_label, "status": status, "picker": picker, "picker_var": picker_var, "slot": slot_index}

    def _format_camera_option(self, device_index: int | None) -> str:
        if device_index is None:
            return "Brak"
        return f"Kamera {device_index}"

    def _parse_camera_option(self, option: str) -> int | None:
        if option == "Brak":
            return None
        if option.startswith("Kamera "):
            try:
                return int(option.split(" ")[1])
            except (ValueError, IndexError):
                return None
        return None

    def _on_camera_selected(self, slot_index: int, selected_option: str):
        self.camera.set_slot_source(slot_index, self._parse_camera_option(selected_option))
        self._update_camera_status()

    def _refresh_camera_devices(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_camera_scan_at) < 3.0:
            return

        self._last_camera_scan_at = now
        devices = self.camera.refresh_devices()
        self.camera_option_values = ["Brak"] + [self._format_camera_option(index) for index in devices]

        for view in [self.front_view, self.side_view]:
            picker = view["picker"]
            picker_var = view["picker_var"]
            slot = view["slot"]

            picker.configure(values=self.camera_option_values)
            selected = self._format_camera_option(self.camera.slot_sources[slot])
            if selected not in self.camera_option_values:
                selected = "Brak"
            picker_var.set(selected)

    def _cover_frame(self, frame, max_width: int, max_height: int):
        height, width = frame.shape[:2]
        ratio = max(max_width / width, max_height / height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        crop_x = max(0, (new_size[0] - max_width) // 2)
        crop_y = max(0, (new_size[1] - max_height) // 2)
        return resized[crop_y : crop_y + max_height, crop_x : crop_x + max_width]

    def _frame_to_photo(self, frame, max_width: int, max_height: int):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = self._cover_frame(rgb, max_width, max_height)
        image = Image.fromarray(resized)
        return ImageTk.PhotoImage(image)

    def _black_photo(self, width: int = 640, height: int = 480):
        image = Image.new("RGB", (width, height), (2, 6, 15))
        return ImageTk.PhotoImage(image)

    def _update_camera_view(self, view, slot_index: int, label_text: str):
        image_label = view["image"]
        status_label = view["status"]
        holder = view["holder"]
        max_width = max(320, holder.winfo_width() or 640)
        max_height = max(240, holder.winfo_height() or 480)

        ok, frame, source_index = self.camera.read(slot_index)
        if ok and frame is not None:
            photo = self._frame_to_photo(frame, max_width, max_height)
            image_label.configure(image=photo, bg="#000000")
            image_label.image = photo
            image_label.place(relx=0.5, rely=0.5, anchor="center")
            self._camera_images[label_text] = photo
            status_label.configure(text=f"kamera {source_index}")
        else:
            photo = self._black_photo(max_width, max_height)
            image_label.configure(image=photo, bg="#000000")
            image_label.image = photo
            image_label.place(relx=0.5, rely=0.5, anchor="center")
            self._camera_images[label_text] = photo
            status_label.configure(text="● OFFLINE", fg="#f87171")

            image_label.configure(
                text="BRAK SYGNALU\nPodlacz druga kamere lub sprawdz uprawnienia.",
                fg="#9aa9c2",
                font=("Segoe UI", 11, "bold"),
                justify="center",
                compound="center",
            )
            return

        image_label.configure(text="")
        status_label.configure(text=f"● LIVE | kamera {source_index}", fg="#34d399")

    def _update_camera_status(self):
        self._refresh_camera_devices()
        if not self.camera.available:
            self.source_label.set("Demo mode")
            self.connection_label.set("OpenCV nie jest dostepne - wyswietlany jest czarny ekran")
            return

        active = len(self.camera.available_devices)
        selected = len([slot for slot in self.camera.slot_sources if slot is not None])
        if active == 0:
            self.source_label.set("Demo mode")
            self.connection_label.set("Nie wykryto zadnej kamery - oba panele pozostaja czarne")
        else:
            self.source_label.set(f"wybrane: {selected}/2")
            self.connection_label.set(f"Wykryte kamery: {active}. Wybierz, ktora ma trafic do KAMERA 1 i KAMERA 2.")

    def _build_controls_card(self, parent):
        ttk.Label(parent, text="Panel sterowania", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            parent,
            text="Przyciski steruja sesja treningowa. To tutaj pozniej podepnie sie logike komunikacji z kamera, analiza i zapis danych.",
            style="CardText.TLabel",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        controls = [
            ("Start sesji", "#2563eb", self.start_session),
            ("Pauza / wznow", "#334155", self.toggle_pause),
            ("Zakoncz", "#dc2626", self.end_session),
            ("Zapisz wynik", "#059669", self.save_result),
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
        tips = (
            "• utrzymaj miednice w linii\n"
            "• pelny wyprost bez przeprostu\n"
            "• ruch bioder pionowo\n"
            "• jedna osoba w kadrze"
        )
        ttk.Label(parent, text=tips, style="CardText.TLabel", justify="left").pack(anchor="w")

    def _build_feedback_card(self, parent):
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
        self.tech_rows = []
        for name, value, color in [
            ("Kregoslup", "neutralny", "#22c55e"),
            ("Biodra", "w gornej pozycji", "#38bdf8"),
            ("Kolana", "stabilne", "#22c55e"),
            ("Tempo", "do dopracowania", "#f59e0b"),
        ]:
            row = tk.Frame(left, bg="#111c33")
            row.pack(fill="x", pady=4)
            tk.Label(row, text=name, bg="#111c33", fg="#94a3b8", font=("Segoe UI", 10)).pack(side="left")
            tk.Label(row, text=value, bg="#111c33", fg=color, font=("Segoe UI", 10, "bold")).pack(side="right")
            self.tech_rows.append((name, row))

        tk.Label(right, text="Komunikaty systemu", bg="#111c33", fg="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.event_log = tk.Listbox(right, bg="#0b1220", fg="#cbd5e1", highlightthickness=0, selectbackground="#1e293b", borderwidth=0, activestyle="none")
        self.event_log.pack(fill="both", expand=True)

    def _build_stats_card(self, parent):
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

    def _load_history(self):
        self.history_box.delete(0, tk.END)
        items = self.store.load()
        if not items:
            self.history_box.insert(tk.END, "Brak zapisanej historii")
            return

        for item in items[:8]:
            duration = item.get("duration_seconds", 0)
            self.history_box.insert(
                tk.END,
                f"{int(item.get('repetitions', 0)):02d} powt. | {duration}s | {int(item.get('quality', 0))}% | {item.get('source', 'unknown')}",
            )

    def _append_event(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.insert(0, f"[{timestamp}] {message}")
        if self.event_log.size() > 12:
            self.event_log.delete(tk.END)

    def start_session(self):
        if self.session_active:
            return

        self.session_active = True
        self.session_paused = False
        self.session_started_at = time.time()
        self.last_tick = self.session_started_at
        self.rep_count = 0
        self.warning_count = 0
        self.quality_score = 91
        self.current_phase = 0.0

        self._refresh_camera_devices(force=True)
        self._update_camera_status()
        selected = len([slot for slot in self.camera.slot_sources if slot is not None])
        if selected == 0:
            self.feedback_label.set("Sesja uruchomiona bez aktywnej kamery. Panele pozostana czarne do czasu wykrycia urzadzenia.")
        elif selected == 1:
            self.feedback_label.set("Sesja uruchomiona. Jedna kamera jest aktywna, drugi panel pozostanie czarny.")
        else:
            self.feedback_label.set("Sesja uruchomiona. Obraz jest pobierany z kamer w czasie rzeczywistym.")

        self._append_event("Sesja rozpoceta")

    def toggle_pause(self):
        if not self.session_active:
            self.feedback_label.set("Najpierw uruchom sesje.")
            return

        self.session_paused = not self.session_paused
        state = "wznowiona" if not self.session_paused else "wstrzymana"
        self.feedback_label.set(f"Sesja {state}.")
        self._append_event(f"Sesja {state}")

    def end_session(self):
        if not self.session_active:
            return

        self.session_active = False
        self.session_paused = False
        self.camera.close()
        self.connection_label.set("Kamera nieaktywna")
        self.feedback_label.set("Sesja zakonczona. Mozesz zapisac wynik lub rozpoczac nowa probe.")
        self._append_event("Sesja zakonczona")
        self.save_result(auto=True)

    def save_result(self, auto: bool = False):
        if not self.session_started_at:
            return

        now = time.time()
        result = SessionResult(
            started_at=self.session_started_at,
            ended_at=now,
            repetitions=self.rep_count,
            warnings=self.warning_count,
            quality=self.quality_score,
            source=self.source_label.get(),
        )
        try:
            self.store.append(result)
            self._load_history()
            if not auto:
                self._append_event("Wynik zapisany lokalnie")
                self.feedback_label.set("Wynik zapisany do lokalnej historii treningow.")
        except Exception as exc:
            self._append_event(f"Blad zapisu: {exc}")
            self.feedback_label.set("Nie udalo sie zapisac wyniku.")

    def _hip_angle(self) -> float:
        return 170 + math.sin(self.current_phase * 1.3) * 8

    def _update_loop(self):
        self._update_metrics()
        self._update_camera_status()
        self._update_camera_view(self.front_view, 0, "front")
        self._update_camera_view(self.side_view, 1, "side")
        self.after(33, self._update_loop)

    def _update_metrics(self):
        if not self.session_active or self.session_paused:
            self.metric_time.set(self._format_elapsed())
            self.metric_reps.set(str(self.rep_count))
            self.metric_quality.set(f"{self.quality_score}%")
            self.metric_warnings.set(str(self.warning_count))
            return

        now = time.time()
        delta = now - self.last_tick
        self.last_tick = now
        self.current_phase += delta * 2.4

        phase_value = math.sin(self.current_phase)
        previous_state = self.repetition_state
        self.repetition_state = "PODNOSZENIE" if phase_value > 0 else "OPUSZCZANIE"

        if previous_state == "OPUSZCZANIE" and self.repetition_state == "PODNOSZENIE" and phase_value > 0.92:
            self.rep_count += 1
            self._append_event(f"Wykryto powtorzenie {self.rep_count}")

        if abs(phase_value) < 0.15 and random.random() < 0.03:
            self.warning_count += 1
            self.quality_score = max(60, self.quality_score - 1)
            self.feedback_label.set("Utrzymaj stabilniejszy tor ruchu bioder i pelny zakres wyprostu.")

        if self.quality_score < 95 and random.random() < 0.02:
            self.quality_score += 1

        self.metric_time.set(self._format_elapsed())
        self.metric_reps.set(str(self.rep_count))
        self.metric_quality.set(f"{self.quality_score}%")
        self.metric_warnings.set(str(self.warning_count))

        if self.rep_count and self.rep_count % 5 == 0:
            self.feedback_label.set("Dobra praca: utrzymano poprawny zakres ruchu.")

    def _format_elapsed(self) -> str:
        if not self.session_started_at:
            return "00:00"
        elapsed = int(time.time() - self.session_started_at) if self.session_active else int(self.last_tick - self.session_started_at)
        minutes, seconds = divmod(max(0, elapsed), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _on_close(self):
        self.camera.close()
        self.destroy()


def main():
    app = CyberTrainerApp()
    app.mainloop()


if __name__ == "__main__":
    main()