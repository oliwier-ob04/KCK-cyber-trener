"""Socket transport for remote frame exchange."""

from __future__ import annotations

import io
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable

from PIL import Image

from ar.renderer import FrameRenderer


class FrameRelayService:
    """Send local camera frames to a peer and receive remote frames."""

    def __init__(
        self,
        peer_host: str,
        peer_port: int,
        listen_host: str,
        listen_port: int,
        send_interval_seconds: float,
        frame_provider: Callable[[], tuple[bool, Any | None, int | str | None]],
        on_frame_received: Callable[[Image.Image], None] | None = None,
    ) -> None:
        """Store the transport endpoints and callbacks."""

        self.peer_host = peer_host
        self.peer_port = peer_port
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.send_interval_seconds = send_interval_seconds
        self.frame_provider = frame_provider
        self.on_frame_received = on_frame_received
        self._incoming_frames: queue.Queue[bytes] = queue.Queue(maxsize=10)
        self._receive_stop_event = threading.Event()
        self._send_stop_event = threading.Event()
        self._receive_thread: threading.Thread | None = None
        self._send_thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None
        self._send_socket: socket.socket | None = None
        self._latest_remote_frame: Image.Image | None = None
        self._remote_frame_lock = threading.Lock()

    def start(self) -> None:
        """Start the background send and receive threads if they are not running."""

        if self._receive_thread is None or not self._receive_thread.is_alive():
            self._receive_stop_event.clear()
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

        if self._send_thread is None or not self._send_thread.is_alive():
            self._send_stop_event.clear()
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()

    def stop(self) -> None:
        """Stop the transport and close any open sockets."""

        self._receive_stop_event.set()
        self._send_stop_event.set()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
        if self._send_socket is not None:
            try:
                self._send_socket.close()
            except Exception:
                pass
        if self._receive_thread is not None:
            self._receive_thread.join(timeout=1.0)
            self._receive_thread = None
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None
        self._server_socket = None
        self._send_socket = None

    def get_remote_frame(self) -> Image.Image | None:
        """Return the latest image received from the remote peer."""

        with self._remote_frame_lock:
            return self._latest_remote_frame

    def _receive_loop(self) -> None:
        """Listen for incoming payloads and decode them into PIL frames."""

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket = server_socket
        try:
            server_socket.bind((self.listen_host, self.listen_port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            while not self._receive_stop_event.is_set():
                try:
                    conn, _addr = server_socket.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(2.0)
                    while not self._receive_stop_event.is_set():
                        size_bytes = self._recv_exact(conn, 4)
                        if not size_bytes:
                            break
                        payload_length = struct.unpack("!I", size_bytes)[0]
                        payload = self._recv_exact(conn, payload_length)
                        if not payload:
                            break
                        try:
                            image = Image.open(io.BytesIO(payload)).convert("RGB")
                        except Exception:
                            continue
                        with self._remote_frame_lock:
                            self._latest_remote_frame = image
                        if self.on_frame_received is not None:
                            self.on_frame_received(image)
        except Exception:
            pass
        finally:
            try:
                server_socket.close()
            except Exception:
                pass
            self._server_socket = None

    def _send_loop(self) -> None:
        """Keep sending the primary local frame to the configured peer."""

        while not self._send_stop_event.is_set():
            if self._send_socket is None:
                try:
                    self._send_socket = socket.create_connection((self.peer_host, self.peer_port), timeout=1.0)
                    self._send_socket.settimeout(0.1)
                except Exception:
                    self._send_socket = None
                    time.sleep(1.0)
                    continue
            try:
                ok, frame, _source_index = self.frame_provider()
                if not ok or frame is None:
                    time.sleep(self.send_interval_seconds)
                    continue
                payload = FrameRenderer.encode_frame_as_jpeg(frame, quality=10)
                if not payload:
                    time.sleep(self.send_interval_seconds)
                    continue
                self._send_socket.sendall(struct.pack("!I", len(payload)))
                self._send_socket.sendall(payload)
            except Exception:
                if self._send_socket is not None:
                    try:
                        self._send_socket.close()
                    except Exception:
                        pass
                self._send_socket = None
            time.sleep(self.send_interval_seconds)

    @staticmethod
    def _recv_exact(sock: socket.socket, length: int) -> bytes | None:
        """Read an exact number of bytes from a socket or return None."""

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
