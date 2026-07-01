#!/usr/bin/env python3
"""
Normalize an external real-imagery manifest into the existing database_index.yaml shape.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from vns.utils.paths import relativize_to_base

REQUIRED_IMAGE_FIELDS = ("id", "latitude", "longitude", "altitude", "heading")


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must contain a top-level mapping.")
    return payload


def _resolve_source_path(
    row: dict[str, Any],
    *,
    manifest_path: Path,
    image_root: Path | None,
) -> Path:
    raw_path = row.get("filepath")
    if raw_path is None:
        raw_path = row.get("filename")
    if raw_path is None:
        raw_path = row.get("source_path")
    if raw_path is None:
        raise ValueError(
            f"Image '{row.get('id', '<unknown>')}' must define filepath, filename, or source_path."
        )

    source = Path(str(raw_path))
    if source.is_absolute():
        return source

    base_dir = image_root if image_root is not None else manifest_path.parent
    return (base_dir / source).resolve()


def _require_image_row(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"images[{index}] must be a mapping.")
    for key in REQUIRED_IMAGE_FIELDS:
        if key not in row:
            raise ValueError(f"images[{index}] is missing required field '{key}'.")
    return row


def _copy_if_requested(
    *,
    source_path: Path,
    image_id: str,
    copy_images_to: Path | None,
) -> Path:
    if copy_images_to is None:
        return source_path.resolve()

    copy_images_to.mkdir(parents=True, exist_ok=True)
    destination = copy_images_to / f"{image_id}{source_path.suffix.lower()}"
    shutil.copy2(source_path, destination)
    return destination.resolve()


def _database_metadata(
    rows: list[dict[str, Any]],
    database: dict[str, Any],
) -> dict[str, Any]:
    lats = [float(row["latitude"]) for row in rows]
    lons = [float(row["longitude"]) for row in rows]
    alts = [float(row["altitude"]) for row in rows]
    center = database.get("center")
    bounds = database.get("bounds")

    if not isinstance(center, dict):
        center = {
            "latitude": sum(lats) / len(lats),
            "longitude": sum(lons) / len(lons),
            "altitude": sum(alts) / len(alts),
        }

    if not isinstance(bounds, dict):
        bounds = {
            "min_latitude": min(lats),
            "max_latitude": max(lats),
            "min_longitude": min(lons),
            "max_longitude": max(lons),
        }

    return {
        "name": str(database.get("name", "Imported Real Imagery Reference Database")),
        "version": str(database.get("version", "1.0.0")),
        "created": str(
            database.get("created", datetime.now(timezone.utc).isoformat())
        ),
        "location": str(database.get("location", "unspecified")),
        "center": {
            "latitude": float(center["latitude"]),
            "longitude": float(center["longitude"]),
            "altitude": float(center["altitude"]),
        },
        "bounds": {
            "min_latitude": float(bounds["min_latitude"]),
            "max_latitude": float(bounds["max_latitude"]),
            "min_longitude": float(bounds["min_longitude"]),
            "max_longitude": float(bounds["max_longitude"]),
        },
        "image_count": len(rows),
    }


def import_manifest_to_index(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    image_root: str | Path | None = None,
    copy_images_to: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize an external manifest into database_index.yaml format."""
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    image_root_path = None if image_root is None else Path(image_root).resolve()
    copy_images_path = None if copy_images_to is None else Path(copy_images_to).resolve()

    manifest = _load_manifest(manifest_path)
    raw_rows = manifest.get("images")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Manifest must contain a non-empty 'images' list.")

    output_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []

    for index, raw_row in enumerate(raw_rows):
        row = _require_image_row(raw_row, index)
        source_path = _resolve_source_path(
            row,
            manifest_path=manifest_path,
            image_root=image_root_path,
        )
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")

        staged_path = _copy_if_requested(
            source_path=source_path,
            image_id=str(row["id"]),
            copy_images_to=copy_images_path,
        )
        filepath = relativize_to_base(staged_path, output_path.parent)

        normalized_row = {
            "id": str(row["id"]),
            "filepath": filepath,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "altitude": float(row["altitude"]),
            "heading": float(row["heading"]),
            "timestamp": str(
                row.get("timestamp", manifest.get("captured_at", ""))
            ),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
        output_rows.append(normalized_row)
        normalized_rows.append(normalized_row)

    payload = {
        "database": _database_metadata(
            normalized_rows,
            manifest.get("database", {}) if isinstance(manifest.get("database"), dict) else {},
        ),
        "images": output_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import external real imagery into database_index.yaml format."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the external YAML or JSON manifest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output database_index.yaml file.",
    )
    parser.add_argument(
        "--image-root",
        help="Base directory for relative image filenames in the manifest.",
    )
    parser.add_argument(
        "--copy-images-to",
        help="Optional destination directory for staging imported imagery.",
    )
    args = parser.parse_args(argv)

    payload = import_manifest_to_index(
        manifest_path=args.manifest,
        output_path=args.output,
        image_root=args.image_root,
        copy_images_to=args.copy_images_to,
    )
    print(f"Imported {payload['database']['image_count']} image(s) into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
