from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

logger = logging.getLogger("vns.validation")


class ImageFrameLogger:
    """Best-effort writer that saves downward-camera frames to disk.

    Wired to the ``logging.log_images`` config flag so a simulation run leaves
    the processed camera frames recoverable as individual PNGs under
    ``<log_dir>/images/`` plus a JSONL manifest (one line per saved frame with
    its sequence number, timestamp, filename, and pixel mean/std). It is a
    no-op when disabled and mirrors :class:`JsonlEvaluationLogger`'s lifecycle
    (constructed enabled/disabled, ``close()`` on node shutdown).

    ``stride`` decimates the stream — ``stride=1`` saves every frame, ``stride=5``
    saves every fifth — to keep disk usage sane for long runs at 30 Hz.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: str | Path,
        stride: int = 1,
        subdir: str = "images",
    ) -> None:
        self._enabled = enabled
        self._stride = max(1, int(stride))
        self._seq = 0
        self._saved = 0
        self._image_dir: Path | None = None
        self._manifest: TextIO | None = None
        self._manifest_path: Path | None = None

        if enabled:
            self._open(log_dir=log_dir, subdir=subdir)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._image_dir is not None

    @property
    def image_dir(self) -> Path | None:
        return self._image_dir

    @property
    def manifest_path(self) -> Path | None:
        return self._manifest_path

    @property
    def saved_count(self) -> int:
        return self._saved

    def _open(self, *, log_dir: str | Path, subdir: str) -> None:
        try:
            base = Path(log_dir).expanduser() / subdir
            base.mkdir(parents=True, exist_ok=True)
            self._image_dir = base
            self._manifest_path = base / f"manifest_{int(time.time())}.jsonl"
            self._manifest = self._manifest_path.open("a", encoding="utf-8")
            logger.info("Camera frames will be saved to: %s", base)
        except OSError as exc:
            self._enabled = False
            self._image_dir = None
            self._manifest = None
            self._manifest_path = None
            logger.warning(
                "Failed to initialize image logger at %s: %s", log_dir, exc
            )

    def save(self, image: Any, *, timestamp: float | None = None) -> bool:
        """Save one frame (decimated by ``stride``). Returns True if written."""
        if not self.enabled or self._image_dir is None:
            return False

        seq = self._seq
        self._seq += 1
        if seq % self._stride != 0:
            return False

        try:
            arr = np.asarray(image)
            filename = f"frame_{seq:06d}.png"
            path = self._image_dir / filename
            if not cv2.imwrite(str(path), arr):
                raise OSError(f"cv2.imwrite returned False for {path}")
            self._saved += 1
            if self._manifest is not None:
                record: dict[str, Any] = {
                    "seq": seq,
                    "timestamp": None if timestamp is None else float(timestamp),
                    "file": filename,
                    "width": int(arr.shape[1]) if arr.ndim >= 2 else None,
                    "height": int(arr.shape[0]) if arr.ndim >= 2 else None,
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                }
                self._manifest.write(
                    json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                )
                self._manifest.flush()
            return True
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            logger.warning("Failed to save camera frame: %s", exc)
            self.close()
            self._enabled = False
            return False

    def close(self) -> None:
        if self._manifest is None:
            return
        try:
            self._manifest.close()
        finally:
            self._manifest = None
