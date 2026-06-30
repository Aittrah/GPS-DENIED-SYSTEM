"""FR-1 — Satellite Image Loader UI (MPVI module).

Run standalone:
    python -m src.vns.mpvi.map_loader
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette, QFont
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
        QSplitter, QStatusBar, QFrame, QSizePolicy, QGroupBox,
    )
except ImportError as _e:
    raise SystemExit(f"PyQt6 is required: pip install PyQt6\n  ({_e})")

from vns.database import ReferenceDatabase, SatelliteImageRecord
from vns.mpvi.geo_validator import validate_image

# ---------------------------------------------------------------------------
# Colours (dark theme matching existing app palette)
# ---------------------------------------------------------------------------
_BG = "#0d1117"
_PANEL = "#161b22"
_BORDER = "#30363d"
_TEXT = "#e6edf3"
_MUTED = "#8b949e"
_GREEN = "#3fb950"
_ORANGE = "#d29922"
_RED = "#f85149"
_BLUE = "#58a6ff"

_ACCEPTED_FILTER = "Satellite Images (*.jpg *.jpeg *.png *.tif *.tiff *.geotiff)"


# ---------------------------------------------------------------------------
# Worker thread for image validation (keeps UI responsive)
# ---------------------------------------------------------------------------

class _ValidatorWorker(QThread):
    done = pyqtSignal(dict)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        result = validate_image(self._path)
        self.done.emit(result)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MapLoaderWindow(QMainWindow):
    """Satellite image loader and ReferenceDatabase importer."""

    def __init__(self, db: Optional[ReferenceDatabase] = None) -> None:
        super().__init__()
        self._db = db or ReferenceDatabase()
        self._current_path: Optional[Path] = None
        self._current_meta: Optional[dict] = None
        self._worker: Optional[_ValidatorWorker] = None

        self._build_ui()
        self._apply_theme()
        self.setWindowTitle("VNS — Satellite Image Loader (FR-1)")
        self.resize(1024, 680)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top toolbar
        toolbar = QHBoxLayout()
        self._btn_browse = QPushButton("Browse Files…")
        self._btn_browse.setFixedHeight(34)
        self._btn_browse.clicked.connect(self._on_browse)
        toolbar.addWidget(self._btn_browse)
        toolbar.addStretch()
        root.addLayout(toolbar)

        # Main splitter: file list | preview + metadata
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left: file list
        left = QGroupBox("Selected Files")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        self._file_list = QListWidget()
        self._file_list.setAlternatingRowColors(True)
        self._file_list.currentItemChanged.connect(self._on_file_selected)
        left_layout.addWidget(self._file_list)
        splitter.addWidget(left)

        # Right: preview + metadata
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Preview pane
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        preview_frame.setMinimumHeight(320)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        self._preview_label = QLabel("No image selected")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(400, 300)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self._preview_label)
        right_layout.addWidget(preview_frame, stretch=3)

        # Metadata pane
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setContentsMargins(8, 8, 8, 8)
        meta_layout.setSpacing(4)
        self._meta_labels: dict[str, QLabel] = {}
        for key in ("Filename", "Size", "Format", "Geo-ref", "CRS", "Bounds (W/S/E/N)"):
            row = QHBoxLayout()
            key_lbl = QLabel(f"{key}:")
            key_lbl.setFixedWidth(140)
            key_lbl.setStyleSheet(f"color: {_MUTED};")
            val_lbl = QLabel("—")
            val_lbl.setWordWrap(True)
            self._meta_labels[key] = val_lbl
            row.addWidget(key_lbl)
            row.addWidget(val_lbl, stretch=1)
            meta_layout.addLayout(row)
        right_layout.addWidget(meta_group, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([260, 760])
        root.addWidget(splitter, stretch=1)

        # Bottom action bar
        action_bar = QHBoxLayout()
        self._btn_add = QPushButton("Add to Database")
        self._btn_add.setFixedHeight(34)
        self._btn_add.setEnabled(False)
        self._btn_add.clicked.connect(self._on_add_to_db)
        action_bar.addStretch()
        action_bar.addWidget(self._btn_add)
        root.addLayout(action_bar)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(f"Database: {len(self._db)} image(s) loaded.")

    def _apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }}
            QGroupBox {{
                border: 1px solid {_BORDER};
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 4px;
                color: {_MUTED};
                font-size: 11px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
            }}
            QListWidget {{
                background-color: {_PANEL};
                border: 1px solid {_BORDER};
                border-radius: 4px;
                color: {_TEXT};
                alternate-background-color: {_BG};
            }}
            QListWidget::item:selected {{
                background-color: #1f6feb;
            }}
            QFrame[frameShape="5"] {{
                background-color: {_PANEL};
                border: 1px solid {_BORDER};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: #21262d;
                color: {_TEXT};
                border: 1px solid {_BORDER};
                border-radius: 4px;
                padding: 4px 14px;
            }}
            QPushButton:hover {{
                background-color: #30363d;
                border-color: {_BLUE};
            }}
            QPushButton:disabled {{
                color: {_MUTED};
                border-color: {_BORDER};
            }}
            QStatusBar {{
                background-color: {_PANEL};
                color: {_MUTED};
                border-top: 1px solid {_BORDER};
            }}
            QSplitter::handle {{
                background-color: {_BORDER};
            }}
        """)
        self._btn_browse.setStyleSheet(
            self._btn_browse.styleSheet() + f"QPushButton {{ color: {_BLUE}; border-color: {_BLUE}; }}"
        )
        self._btn_add.setStyleSheet(
            self._btn_add.styleSheet() + f"QPushButton:enabled {{ background-color: #1a4a2e; color: {_GREEN}; border-color: {_GREEN}; }}"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Satellite Images", "", _ACCEPTED_FILTER
        )
        if not paths:
            return
        self._file_list.clear()
        for p in paths:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self._file_list.addItem(item)
        self._file_list.setCurrentRow(0)

    def _on_file_selected(self, current: Optional[QListWidgetItem], _prev) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        self._current_path = path
        self._current_meta = None
        self._btn_add.setEnabled(False)
        self._clear_metadata()
        self._status.showMessage(f"Validating {path.name}…")

        # Run validation in background thread
        if self._worker and self._worker.isRunning():
            self._worker.quit()
        self._worker = _ValidatorWorker(path)
        self._worker.done.connect(self._on_validated)
        self._worker.start()

    def _on_validated(self, meta: dict) -> None:
        self._current_meta = meta
        path = self._current_path
        if path is None:
            return

        self._update_metadata(path, meta)
        self._update_preview(path, meta)

        if not meta["valid"]:
            self._status.showMessage(f"Error: {meta['error']}", 0)
            self._set_status_color(_RED)
            self._btn_add.setEnabled(False)
        elif meta.get("warning"):
            self._status.showMessage(f"Warning: {meta['warning']}", 0)
            self._set_status_color(_ORANGE)
            self._btn_add.setEnabled(True)
        else:
            self._status.showMessage(f"Ready — {path.name} is geo-referenced and valid.")
            self._set_status_color(_GREEN)
            self._btn_add.setEnabled(True)

    def _on_add_to_db(self) -> None:
        path = self._current_path
        meta = self._current_meta
        if path is None or meta is None or not meta["valid"]:
            return

        record = SatelliteImageRecord(
            path=str(path.resolve()),
            filename=path.name,
            file_size_bytes=path.stat().st_size,
            format=meta["format"],
            has_georef=meta["has_georef"],
            bbox=meta["bbox"],
            crs=meta["crs"],
            added_at=time.time(),
        )
        self._db.add_image(record)
        self._btn_add.setEnabled(False)
        self._status.showMessage(
            f"Saved — {path.name} added to database ({len(self._db)} total)."
        )
        self._set_status_color(_GREEN)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _clear_metadata(self) -> None:
        for lbl in self._meta_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet("")
        self._preview_label.setText("Loading…")
        self._preview_label.setPixmap(QPixmap())

    def _update_metadata(self, path: Path, meta: dict) -> None:
        size_kb = path.stat().st_size / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"

        self._meta_labels["Filename"].setText(path.name)
        self._meta_labels["Size"].setText(size_str)
        self._meta_labels["Format"].setText(meta["format"])

        if meta["valid"]:
            georef_text = "Yes" if meta["has_georef"] else "No (warning)"
            color = _GREEN if meta["has_georef"] else _ORANGE
            self._meta_labels["Geo-ref"].setText(georef_text)
            self._meta_labels["Geo-ref"].setStyleSheet(f"color: {color};")
        else:
            self._meta_labels["Geo-ref"].setText("Invalid")
            self._meta_labels["Geo-ref"].setStyleSheet(f"color: {_RED};")

        self._meta_labels["CRS"].setText(meta["crs"] or "—")
        if meta["bbox"]:
            w, s, e, n = meta["bbox"]
            self._meta_labels["Bounds (W/S/E/N)"].setText(
                f"{w:.5f}, {s:.5f}, {e:.5f}, {n:.5f}"
            )
        else:
            self._meta_labels["Bounds (W/S/E/N)"].setText("—")

    def _update_preview(self, path: Path, meta: dict) -> None:
        pixmap: Optional[QPixmap] = None
        suffix = path.suffix.lower()

        try:
            if suffix in {".tif", ".tiff", ".geotiff"}:
                pixmap = _load_geotiff_pixmap(path)
            else:
                pixmap = QPixmap(str(path))
        except Exception:
            pass

        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                self._preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)
        else:
            self._preview_label.setText("Preview unavailable")

    def _set_status_color(self, color: str) -> None:
        self._status.setStyleSheet(
            f"QStatusBar {{ background-color: {_PANEL}; color: {color}; border-top: 1px solid {_BORDER}; }}"
        )


# ---------------------------------------------------------------------------
# GeoTIFF → QPixmap
# ---------------------------------------------------------------------------

def _load_geotiff_pixmap(path: Path) -> Optional[QPixmap]:
    """Read first three bands of a GeoTIFF and return a QPixmap."""
    try:
        import rasterio  # type: ignore
        import numpy as np
        with rasterio.open(path) as src:
            count = src.count
            if count >= 3:
                r = src.read(1).astype(float)
                g = src.read(2).astype(float)
                b = src.read(3).astype(float)
            elif count == 1:
                band = src.read(1).astype(float)
                r = g = b = band
            else:
                return None

            def _norm(arr: "np.ndarray") -> "np.ndarray":
                mn, mx = arr.min(), arr.max()
                if mx == mn:
                    return np.zeros_like(arr, dtype=np.uint8)
                return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

            import numpy as np
            h, w = r.shape
            rgb = np.stack([_norm(r), _norm(g), _norm(b)], axis=2)
            img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(img)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MapLoaderWindow()
    win.show()
    sys.exit(app.exec())
