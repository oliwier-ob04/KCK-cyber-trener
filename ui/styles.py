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

        # Labels
        style.configure(
            "Section.TLabel",
            background=self.config.card_color,
            foreground=self.config.text_secondary,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=self.config.card_color,
            foreground=self.config.text_secondary,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Meta.TLabel",
            background=self.config.card_color,
            foreground=self.config.text_muted,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Title.TLabel",
            background=self.config.background_color,
            foreground=self.config.text_primary,
            font=("Segoe UI", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.config.background_color,
            foreground=self.config.text_muted,
            font=("Segoe UI", 10),
        )

        # Chips and navigation
        style.configure(
            "Chip.TLabel",
            background=self.config.chip_bg,
            foreground=self.config.text_secondary,
            font=("Segoe UI", 9, "bold"),
            padding=(10, 4),
        )
        style.configure(
            "Nav.TButton",
            background=self.config.card_color,
            foreground=self.config.text_secondary,
            borderwidth=0,
            padding=(12, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Nav.TButton",
            background=[("active", self.config.panel_wrapper_bg)],
            foreground=[("active", self.config.text_primary)],
        )

        # Combobox
        style.configure(
            "CameraSelect.TCombobox",
            fieldbackground=self.config.sidebar_color,
            background=self.config.sidebar_color,
            foreground=self.config.text_secondary,
            arrowsize=14,
        )
