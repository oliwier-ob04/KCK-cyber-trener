"""Application entry point for Cyber Trener."""

from __future__ import annotations

from app import CyberTrainerApp


def main() -> None:
    """Start the Cyber Trener desktop application."""

    app = CyberTrainerApp()
    app.run()


if __name__ == "__main__":
    main()
