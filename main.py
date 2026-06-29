"""
VNS — GNSS-Free Navigation System
Entry point: launches the desktop application.

Usage:
    python main.py
"""
import sys
from pathlib import Path

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).parent / "app"))

from src.vns.app.desktop_app import VNSApp


if __name__ == "__main__":
    app = VNSApp()
    app.run()
