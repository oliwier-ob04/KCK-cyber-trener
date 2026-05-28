"""Tk ttk style setup for the Cyber Trener UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from utils.config import AppConfig


class StyleManager:
    """Apply the shared application theme to the root window."""

    def __init__(self, root: tk.Misc, config: AppConfig) -> None:
        """Store the Tk root and theme colors."""

        self.root = root
        self.config = config

    def apply(self) -> None:
        """Register the ttk styles used across the application."""

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=self.config.background_color)
        style.configure("Sidebar.TFrame", background=self.config.sidebar_color)
        style.configure("Card.TFrame", background=self.config.card_color, relief="flat")
        style.configure("CardInner.TFrame", background=self.config.card_color)
        style.configure(
            "Section.TLabel",
            background=self.config.card_color,
            foreground="#e2e8f0",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=self.config.card_color,
            foreground="#cbd5e1",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Meta.TLabel",
            background=self.config.card_color,
            foreground="#94a3b8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Title.TLabel",
            background=self.config.background_color,
            foreground="#f8fafc",
            font=("Segoe UI", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.config.background_color,
            foreground="#94a3b8",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Chip.TLabel",
            background="#1e293b",
            foreground="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 4),
        )
        style.configure(
            "Nav.TButton",
            background=self.config.card_color,
            foreground="#e2e8f0",
            borderwidth=0,
            padding=(12, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "CameraSelect.TCombobox",
            fieldbackground=self.config.sidebar_color,
            background=self.config.sidebar_color,
            foreground="#e2e8f0",
            arrowsize=14,
        )
        style.map("Nav.TButton", background=[("active", "#1f2e4d")], foreground=[("active", "#ffffff")])
