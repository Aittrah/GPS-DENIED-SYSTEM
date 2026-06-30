#!/usr/bin/env python3
"""Shared helpers for synthetic and Gazebo-backed reference-image workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReferencePoint:
    id: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    description: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapturedImage:
    id: str
    filepath: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    timestamp: str
    width: int
    height: int
    extra: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_reference_config(config_path: str | Path) -> tuple[dict[str, Any], list[ReferencePoint]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    points = [
        ReferencePoint(
            id=str(point_data["id"]),
            latitude=float(point_data["latitude"]),
            longitude=float(point_data["longitude"]),
            altitude=float(point_data["altitude"]),
            heading=float(point_data["heading"]),
            description=str(point_data.get("description", "")),
            extra={
                key: value
                for key, value in point_data.items()
                if key
                not in {
                    "id",
                    "latitude",
                    "longitude",
                    "altitude",
                    "heading",
                    "description",
                }
            },
        )
        for point_data in config.get("reference_points", [])
    ]
    return config, points


def generate_database_index(
    captured_images: list[CapturedImage],
    output_dir: Path,
    config: dict[str, Any],
    *,
    database_source_type: str,
) -> Path:
    output_dir = output_dir.resolve()
    database_meta = dict(config.get("database", {}))
    database_meta["created"] = utc_now_iso()
    database_meta["image_count"] = len(captured_images)
    database_meta["source_type"] = database_source_type

    index = {
        "database": database_meta,
        "images": [],
    }

    for image in captured_images:
        image_path = Path(image.filepath).resolve()
        relative_path = image_path.relative_to(output_dir).as_posix()
        entry = {
            "id": image.id,
            "filepath": relative_path,
            "latitude": image.latitude,
            "longitude": image.longitude,
            "altitude": image.altitude,
            "heading": image.heading,
            "timestamp": image.timestamp,
            "width": image.width,
            "height": image.height,
        }
        entry.update(image.extra)
        index["images"].append(entry)

    index_path = output_dir / "database_index.yaml"
    with open(index_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(index, handle, default_flow_style=False, sort_keys=False)
    return index_path


def validate_coverage(
    captured_images: list[CapturedImage],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not captured_images:
        return {
            "total_images": 0,
            "adequate": False,
        }

    bounds = config["database"]["bounds"]
    lats = [image.latitude for image in captured_images]
    lons = [image.longitude for image in captured_images]

    coverage = {
        "total_images": len(captured_images),
        "latitude_range": [min(lats), max(lats)],
        "longitude_range": [min(lons), max(lons)],
        "bounds_coverage": {
            "north": max(lats) >= bounds["max_latitude"] - 0.001,
            "south": min(lats) <= bounds["min_latitude"] + 0.001,
            "east": max(lons) >= bounds["max_longitude"] - 0.001,
            "west": min(lons) <= bounds["min_longitude"] + 0.001,
        },
        "headings_covered": len({image.heading for image in captured_images}),
    }
    coverage["adequate"] = (
        coverage["total_images"] >= 10
        and all(coverage["bounds_coverage"].values())
    )
    return coverage
