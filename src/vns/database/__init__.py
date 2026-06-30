"""Reference database for satellite image records."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class SatelliteImageRecord:
    """Metadata record for a loaded satellite reference image."""
    path: str                                          # str for JSON serialisation; use Path(record.path) to get Path
    filename: str
    file_size_bytes: int
    format: str                                        # "GeoTIFF" | "JPEG" | "PNG"
    has_georef: bool
    bbox: Optional[tuple[float, float, float, float]]  # (west, south, east, north) WGS84 degrees
    crs: Optional[str]
    added_at: float = 0.0

    def __post_init__(self) -> None:
        if self.added_at == 0.0:
            self.added_at = time.time()


class ReferenceDatabase:
    """Persistent store for satellite image records used during mission planning."""

    _INDEX_FILE = "index.json"

    def __init__(self, db_dir: Path = Path("data") / "database") -> None:
        self._db_dir = db_dir
        self._records: list[SatelliteImageRecord] = []
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_image(self, record: SatelliteImageRecord) -> None:
        """Add a satellite image record and persist to disk."""
        self._records.append(record)
        self._save()

    def get_all(self) -> list[SatelliteImageRecord]:
        """Return all stored image records."""
        return list(self._records)

    def remove_image(self, path: Path) -> bool:
        """Remove a record by file path. Returns True if found and removed."""
        before = len(self._records)
        self._records = [r for r in self._records if Path(r.path) != path]
        if len(self._records) < before:
            self._save()
            return True
        return False

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self._db_dir / self._INDEX_FILE

    def _load(self) -> None:
        index = self._index_path()
        if not index.exists():
            return
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            for item in data:
                bbox = item.get("bbox")
                self._records.append(SatelliteImageRecord(
                    path=item["path"],
                    filename=item["filename"],
                    file_size_bytes=item["file_size_bytes"],
                    format=item["format"],
                    has_georef=item["has_georef"],
                    bbox=tuple(bbox) if bbox is not None else None,
                    crs=item.get("crs"),
                    added_at=item.get("added_at", 0.0),
                ))
        except Exception:
            # Corrupt index — start fresh, do not crash
            self._records = []

    def _save(self) -> None:
        rows = []
        for r in self._records:
            d = asdict(r)
            rows.append(d)
        self._index_path().write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
