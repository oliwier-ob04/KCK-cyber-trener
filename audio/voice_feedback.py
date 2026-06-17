"""Queued speech output for training feedback."""

from __future__ import annotations

import queue
import threading
import time

import pyttsx3


class VoiceFeedback:
    """Speak short training messages without overlapping them."""

    def __init__(self, min_interval_seconds: float = 1.5, enabled: bool = True) -> None:
        self.min_interval_seconds = max(0.2, float(min_interval_seconds))
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._last_spoken_at = 0.0
        self._stop_event = threading.Event()
        self._enabled = enabled

        try:
            # Test if pyttsx3 works by creating a temporary engine
            test_engine = pyttsx3.init()
            test_engine.setProperty("rate", 175)
            test_engine.setProperty("volume", 1.0)
            del test_engine
        except Exception:
            self._enabled = False
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, message: str) -> None:
        """Queue a message if it is not a duplicate burst."""

        text = message.strip()
        if not text or not self._enabled:
            return

        with self._lock:
            now = time.time()
            if now - self._last_spoken_at < self.min_interval_seconds:
                return
            self._last_spoken_at = now

        self._queue.put(text)

    def close(self) -> None:
        """Stop the worker thread."""

        self._stop_event.set()
        self._queue.put("")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            message = self._queue.get()
            try:
                if self._stop_event.is_set() or not message:
                    continue
                # Create a new engine for each message to avoid stability issues
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.setProperty("volume", 1.0)
                engine.say(message)
                engine.runAndWait()
                del engine
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def say_nonblocking(self, message: str) -> None:
        """Queue a message without rate limiting (for important repeating messages)."""
        text = message.strip()
        if not text or not self._enabled:
            return
        self._queue.put(text)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable voice feedback."""
        self._enabled = enabled
