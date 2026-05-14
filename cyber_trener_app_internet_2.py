import io
import json
import math
import queue
import random
import socket
import struct
import threading
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

REMOTE_PEER_HOST = "10.218.165.145"
REMOTE_PEER_PORT = 5001
LOCAL_LISTEN_HOST = "0.0.0.0"
LOCAL_LISTEN_PORT = 5000


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
    def __init__(self, app, source_indices: tuple[int, ...] = (0, 1)):
        self.app = app
        self.source_indices = source_indices
        self.captures: list[dict] = []
        self._open_sources()

    @property
    def available(self) -> bool:
        return cv2 is not None

    def _open_sources(self) -> None:
        self.close()
        if not self.available:
            return

        for index in self.source_indices:
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.captures.append({"type": "camera", "index": index, "capture": capture})
            else:
                capture.release()

        # Add remote source
        self.captures.append({"type": "remote", "index": "remote"})

    def refresh(self) -> None:
        if not self.captures:
            self._open_sources()

    def read(self, slot_index: int):
        if slot_index >= len(self.captures):
            return False, None, None
        entry = self.captures[slot_index]
        if entry["type"] == "camera":
            capture = entry["capture"]
            ok, frame = capture.read()
            return ok, frame, entry["index"]
        elif entry["type"] == "remote":
            if self.app._remote_pil is not None:
                # Convert PIL to numpy array
                import numpy as np
                frame = np.array(self.app._remote_pil)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame, "remote"
            else:
                return False, None, "remote"

    def close(self) -> None:
        for entry in self.captures:
            if entry["type"] == "camera" and entry.get("capture") is not None:
                entry["capture"].release()
        self.captures = []


class CyberTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cyber Trener - dzialajacy prototyp")
        self.state("zoomed")
        self.configure(bg="#0b1220")

        self.camera = CameraManager(self)
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
        self._remote_photo: ImageTk.PhotoImage | None = None
        self._remote_pil = None
        self.source1_var = tk.StringVar(value="Kamera 0")
        self.source2_var = tk.StringVar(value="Kamera 1")

        self.peer_host = REMOTE_PEER_HOST
        self.peer_port = REMOTE_PEER_PORT
        self.listen_host = LOCAL_LISTEN_HOST
        self.listen_port = LOCAL_LISTEN_PORT
        self._network_server_thread: threading.Thread | None = None
        self._network_server_socket: socket.socket | None = None
        self._network_stop_event = threading.Event()
        self._network_send_thread: threading.Thread | None = None
        self._network_send_stop_event = threading.Event()
        self._incoming_frames: "queue.Queue[bytes]" = queue.Queue(maxsize=10)
        self._network_send_interval = 0.016
        self._send_socket: socket.socket | None = None

        self._setup_style()
        self._build_ui()
        self._load_history()
        self._start_frame_server()
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

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=280)
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
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=9, minsize=560)
        parent.rowconfigure(1, weight=1, minsize=180)

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
            text="Wybierz źródło dla każdego widoku za pomocą dropdownów.",
            style="CardText.TLabel",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        view_wrap = ttk.Frame(parent, style="Card.TFrame")
        view_wrap.pack(fill="both", expand=True)
        view_wrap.columnconfigure(0, weight=1)
        view_wrap.columnconfigure(1, weight=1)
        view_wrap.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(0, 0, 10, 0))
        right_panel = ttk.Frame(view_wrap, style="Card.TFrame", padding=(10, 0, 0, 0))
        left_panel.grid(row=0, column=0, sticky="nsew")
        right_panel.grid(row=0, column=1, sticky="nsew")

        self.left_view = self._create_camera_panel(left_panel, "Widok 1", self.source1_var)
        self.right_view = self._create_camera_panel(right_panel, "Widok 2", self.source2_var)

    def _create_camera_panel(self, parent, title, source_var):
        wrapper = tk.Frame(parent, bg="#08111f")
        wrapper.pack(fill="both", expand=True)

        header = tk.Frame(wrapper, bg="#08111f")
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text=title, bg="#08111f", fg="#e2e8f0", font=("Segoe UI", 12, "bold")).pack(side="left")
        combo = ttk.Combobox(header, textvariable=source_var, values=["Kamera 0", "Kamera 1", "Zdalna"], state="readonly", width=10)
        combo.pack(side="right", padx=(10, 0))
        status = tk.Label(header, text="szukanie kamery...", bg="#08111f", fg="#94a3b8", font=("Segoe UI", 10))
        status.pack(side="right")

        image_label = tk.Label(wrapper, bg="#000000", width=640, height=480, anchor="center")
        image_label.pack(fill="both", expand=True)
        return {"frame": wrapper, "image": image_label, "status": status}

    def _map_source(self, source_str: str) -> int:
        if source_str == "Kamera 0":
            return 0
        elif source_str == "Kamera 1":
            return 1
        elif source_str == "Zdalna":
            return 2
        return 0

    def _fit_frame(self, frame, max_width: int, max_height: int):
        height, width = frame.shape[:2]
        ratio = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _frame_to_photo(self, frame, max_width: int, max_height: int):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = self._fit_frame(rgb, max_width, max_height)
        image = Image.fromarray(resized)
        return ImageTk.PhotoImage(image)

    def _pil_to_photo(self, image: Image.Image, max_width: int, max_height: int):
        rgb = image.convert("RGB")
        width, height = rgb.size
        ratio = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
        resized = rgb.resize(new_size, resample)
        return ImageTk.PhotoImage(resized)

    def _encode_frame_as_jpeg(self, frame, quality: int = 5):
        if frame is None or cv2 is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded.tobytes() if ok else None

    def _recv_exact(self, sock: socket.socket, length: int) -> bytes | None:
        data = bytearray()
        while len(data) < length:
            try:
                packet = sock.recv(length - len(data))
            except socket.timeout:
                return None
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _start_frame_server(self) -> None:
        if self._network_server_thread and self._network_server_thread.is_alive():
            return
        self._network_stop_event.clear()
        self._network_server_thread = threading.Thread(target=self._frame_server_loop, daemon=True)
        self._network_server_thread.start()

        if self._network_send_thread and self._network_send_thread.is_alive():
            return
        self._network_send_stop_event.clear()
        self._network_send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._network_send_thread.start()

    def _stop_frame_server(self) -> None:
        self._network_stop_event.set()
        if self._network_server_socket is not None:
            try:
                self._network_server_socket.close()
            except Exception:
                pass
        if self._network_server_thread is not None:
            self._network_server_thread.join(timeout=1.0)
            self._network_server_thread = None

        self._network_send_stop_event.set()
        if self._network_send_thread is not None:
            self._network_send_thread.join(timeout=1.0)
            self._network_send_thread = None
        if self._send_socket is not None:
            try:
                self._send_socket.close()
            except Exception:
                pass
            self._send_socket = None

    def _frame_server_loop(self) -> None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._network_server_socket = server_socket
        try:
            server_socket.bind((self.listen_host, self.listen_port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            while not self._network_stop_event.is_set():
                try:
                    conn, addr = server_socket.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(2.0)
                    while not self._network_stop_event.is_set():
                        size_bytes = self._recv_exact(conn, 4)
                        if not size_bytes:
                            break
                        payload_length = struct.unpack("!I", size_bytes)[0]
                        payload = self._recv_exact(conn, payload_length)
                        if not payload:
                            break
                        try:
                            self._incoming_frames.put_nowait(payload)
                        except queue.Full:
                            continue
                        self.after(0, self._process_incoming_frames)
        except Exception as exc:
            pass
        finally:
            server_socket.close()
            self._network_server_socket = None

    def _send_loop(self) -> None:
        while not self._network_send_stop_event.is_set():
            if self._send_socket is None:
                try:
                    self._send_socket = socket.create_connection((self.peer_host, self.peer_port), timeout=1.0)
                    self._send_socket.settimeout(0.1)
                except Exception:
                    self._send_socket = None
                    time.sleep(1.0)
                    continue
            try:
                ok, frame, source_index = self.camera.read(0)
                if not ok or frame is None:
                    time.sleep(self._network_send_interval)
                    continue
                payload = self._encode_frame_as_jpeg(frame, quality=10)
                if not payload:
                    time.sleep(self._network_send_interval)
                    continue
                self._send_socket.sendall(struct.pack("!I", len(payload)))
                self._send_socket.sendall(payload)
            except Exception:
                if self._send_socket:
                    try:
                        self._send_socket.close()
                    except:
                        pass
                self._send_socket = None
            time.sleep(self._network_send_interval)

    def _process_incoming_frames(self) -> None:
        while not self._incoming_frames.empty():
            payload = self._incoming_frames.get()
            try:
                incoming_image = Image.open(io.BytesIO(payload)).convert("RGB")
                self._remote_pil = incoming_image
                self._remote_photo = None
                self.remote_view["status"].configure(text="zdalny obraz")
            except Exception:
                pass


    def _send_camera_frame(self) -> None:
        if cv2 is None:
            return
        ok, frame, source_index = self.camera.read(0)
        if not ok or frame is None:
            return
        payload = self._encode_frame_as_jpeg(frame, quality=15)
        if not payload:
            return
        try:
            with socket.create_connection((self.peer_host, self.peer_port), timeout=0.01) as sock:
                sock.sendall(struct.pack("!I", len(payload)))
                sock.sendall(payload)
        except Exception as exc:
            pass

    def _black_photo(self, width: int = 640, height: int = 480):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        return ImageTk.PhotoImage(image)

    def _update_camera_view(self, view, slot_index: int, label_text: str):
        image_label = view["image"]
        status_label = view["status"]
        max_width = max(320, image_label.winfo_width() or 640)
        max_height = max(240, image_label.winfo_height() or 480)

        ok, frame, source_index = self.camera.read(slot_index)
        if ok and frame is not None:
            photo = self._frame_to_photo(frame, max_width, max_height)
            image_label.configure(image=photo, bg="#000000")
            image_label.image = photo
            self._camera_images[label_text] = photo
            status_label.configure(text=f"kamera {source_index}")
        else:
            photo = self._black_photo(max_width, max_height)
            image_label.configure(image=photo, bg="#000000")
            image_label.image = photo
            self._camera_images[label_text] = photo
            status_label.configure(text="brak sygnalu")

    def _update_camera_status(self):
        self.camera.refresh()
        if not self.camera.available:
            self.source_label.set("Demo mode")
            self.connection_label.set("OpenCV nie jest dostepne - wyswietlany jest czarny ekran")
            return

        active = len(self.camera.captures)
        if active == 0:
            self.source_label.set("Demo mode")
            self.connection_label.set("Nie wykryto zadnej kamery - oba panele pozostaja czarne")
        elif active == 1:
            self.source_label.set("1 camera")
            self.connection_label.set("Wykryto jedna kamere - drugi panel pozostaje czarny")
        else:
            self.source_label.set(f"{active} cameras")
            self.connection_label.set("Wykryto co najmniej dwie kamery")

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

        self._update_camera_status()
        if len(self.camera.captures) == 0:
            self.feedback_label.set("Sesja uruchomiona bez aktywnej kamery. Panele pozostana czarne do czasu wykrycia urzadzenia.")
        elif len(self.camera.captures) == 1:
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
        self._update_camera_view(self.left_view, self._map_source(self.source1_var.get()), "left")
        self._update_camera_view(self.right_view, self._map_source(self.source2_var.get()), "right")
        self.after(16, self._update_loop)

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
        self._stop_frame_server()
        self.camera.close()
        self.destroy()


def main():
    app = CyberTrainerApp()
    app.mainloop()


if __name__ == "__main__":
    main()