#!/usr/bin/env python3
"""Augment QAU reference metadata with a regular coverage grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import yaml

from reference_image_utils import load_reference_config

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from vns.validation.reference_coverage import camera_footprint_m

DEFAULT_WIDTH_M = 300.0
DEFAULT_HEIGHT_M = 300.0
DEFAULT_ALTITUDE_AGL_M = 50.0
DEFAULT_OVERLAP_RATIO = 0.15
DEFAULT_HEADING_DEG = 0.0


def load_simulation_camera(simulation_config_path: str | Path) -> dict[str, Any]:
    with open(simulation_config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    camera = payload.get("camera", {})
    return camera if isinstance(camera, dict) else {}


def augment_reference_metadata(
    *,
    metadata_path: str | Path,
    simulation_config_path: str | Path,
    output_path: str | Path,
    width_m: float = DEFAULT_WIDTH_M,
    height_m: float = DEFAULT_HEIGHT_M,
    altitude_agl_m: float = DEFAULT_ALTITUDE_AGL_M,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    heading_deg: float = DEFAULT_HEADING_DEG,
    center_latitude: float | None = None,
    center_longitude: float | None = None,
    min_latitude: float | None = None,
    max_latitude: float | None = None,
    min_longitude: float | None = None,
    max_longitude: float | None = None,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path).resolve()
    simulation_config_path = Path(simulation_config_path).resolve()
    output_path = Path(output_path).resolve()

    with metadata_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError("Reference metadata must contain a top-level mapping.")

    database = document.get("database", {})
    if not isinstance(database, dict):
        raise ValueError("Reference metadata database section must be a mapping.")

    config, _existing_points = load_reference_config(metadata_path)
    all_points = config.get("reference_points", [])
    if not isinstance(all_points, list):
        raise ValueError("reference_points must be a list.")

    if any(value is not None for value in (min_latitude, max_latitude, min_longitude, max_longitude)):
        bounds_args = (min_latitude, max_latitude, min_longitude, max_longitude)
        if any(value is None for value in bounds_args):
            raise ValueError(
                "Bounding-box generation requires min/max latitude and longitude together."
            )
        center_latitude = (float(min_latitude) + float(max_latitude)) / 2.0
        center_longitude = (float(min_longitude) + float(max_longitude)) / 2.0
        width_m = (float(max_longitude) - float(min_longitude)) * _meters_per_degree_lon(center_latitude)
        height_m = (float(max_latitude) - float(min_latitude)) * _meters_per_degree_lat()

    if center_latitude is None:
        center_latitude = float(database.get("center", {}).get("latitude", 33.7470))
    if center_longitude is None:
        center_longitude = float(database.get("center", {}).get("longitude", 73.1370))

    overlap_ratio = min(max(float(overlap_ratio), 0.0), 0.95)
    heading_deg = float(heading_deg)
    altitude_agl_m = float(altitude_agl_m)
    width_m = float(width_m)
    height_m = float(height_m)

    camera_cfg = load_simulation_camera(simulation_config_path)
    footprint_w, footprint_h = camera_footprint_m(camera_cfg, altitude_agl_m)
    spacing_east = footprint_w * (1.0 - overlap_ratio)
    spacing_north = footprint_h * (1.0 - overlap_ratio)
    cols = max(1, math.ceil(max(0.0, width_m - footprint_w) / spacing_east) + 1)
    rows = max(1, math.ceil(max(0.0, height_m - footprint_h) / spacing_north) + 1)

    base_altitude = float(database.get("center", {}).get("altitude", 550.0))
    west_offset = -((cols - 1) * spacing_east) / 2.0
    north_offset = ((rows - 1) * spacing_north) / 2.0

    legacy_points = [
        _annotate_legacy_point(point)
        for point in all_points
        if str(point.get("coverage_role", "")).strip().lower() != "coverage_grid"
    ]

    coverage_points: list[dict[str, Any]] = []
    for row in range(rows):
        north_m = north_offset - row * spacing_north
        for col in range(cols):
            east_m = west_offset + col * spacing_east
            latitude, longitude = _local_to_latlon(
                center_latitude,
                center_longitude,
                east_m,
                north_m,
            )
            coverage_points.append(
                {
                    "id": f"qau_cov_r{row + 1:02d}_c{col + 1:02d}",
                    "latitude": round(latitude, 7),
                    "longitude": round(longitude, 7),
                    "altitude": base_altitude,
                    "heading": heading_deg,
                    "description": f"Coverage grid row {row + 1} col {col + 1}",
                    "coverage_role": "coverage_grid",
                    "capture_status": "planned",
                    "manual_capture_required": True,
                    "grid_row": row,
                    "grid_col": col,
                }
            )

    min_latitude = center_latitude - (height_m / 2.0) / _meters_per_degree_lat()
    max_latitude = center_latitude + (height_m / 2.0) / _meters_per_degree_lat()
    min_longitude = center_longitude - (width_m / 2.0) / _meters_per_degree_lon(center_latitude)
    max_longitude = center_longitude + (width_m / 2.0) / _meters_per_degree_lon(center_latitude)

    database["center"] = {
        "latitude": round(center_latitude, 7),
        "longitude": round(center_longitude, 7),
        "altitude": base_altitude,
    }
    database["bounds"] = {
        "min_latitude": round(min_latitude, 7),
        "max_latitude": round(max_latitude, 7),
        "min_longitude": round(min_longitude, 7),
        "max_longitude": round(max_longitude, 7),
    }
    database["coverage_plan"] = {
        "width_m": width_m,
        "height_m": height_m,
        "altitude_agl_m": altitude_agl_m,
        "overlap_ratio": overlap_ratio,
        "heading_deg": heading_deg,
        "coverage_grid_rows": rows,
        "coverage_grid_cols": cols,
        "coverage_grid_tile_count": len(coverage_points),
        "total_reference_points": len(legacy_points) + len(coverage_points),
        "frame_footprint_m": {
            "width": round(footprint_w, 3),
            "height": round(footprint_h, 3),
        },
        "grid_spacing_m": {
            "east_west": round(spacing_east, 3),
            "north_south": round(spacing_north, 3),
        },
    }

    document["database"] = database
    document["reference_points"] = legacy_points + coverage_points
    document["capture_flight_path"] = {
        "altitude": altitude_agl_m,
        "speed": 5,
        "waypoints": _lawnmower_waypoints(coverage_points),
    }
    return document


def summarize_metadata(document: dict[str, Any]) -> dict[str, Any]:
    reference_points = document.get("reference_points", [])
    coverage_points = [
        point
        for point in reference_points
        if str(point.get("coverage_role", "")).strip().lower() == "coverage_grid"
    ]
    database = document.get("database", {})
    coverage_plan = database.get("coverage_plan", {})
    return {
        "total_reference_points": len(reference_points),
        "coverage_grid_tile_count": len(coverage_points),
        "coverage_grid_rows": coverage_plan.get("coverage_grid_rows"),
        "coverage_grid_cols": coverage_plan.get("coverage_grid_cols"),
        "supported_bounds": database.get("bounds"),
        "frame_footprint_m": coverage_plan.get("frame_footprint_m"),
        "grid_spacing_m": coverage_plan.get("grid_spacing_m"),
    }


def write_metadata(document: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a regular QAU coverage grid inside the existing reference metadata format."
    )
    parser.add_argument(
        "--input",
        default="../database/qau_reference_metadata.yaml",
        help="Input metadata YAML file to augment.",
    )
    parser.add_argument(
        "--output",
        default="../database/qau_reference_metadata.yaml",
        help="Output metadata YAML path.",
    )
    parser.add_argument(
        "--simulation-config",
        default="../config/simulation.yaml",
        help="Path to simulation.yaml for camera intrinsics.",
    )
    parser.add_argument("--center-latitude", type=float, help="Supported-area center latitude.")
    parser.add_argument("--center-longitude", type=float, help="Supported-area center longitude.")
    parser.add_argument("--min-latitude", type=float, help="Supported-area minimum latitude.")
    parser.add_argument("--max-latitude", type=float, help="Supported-area maximum latitude.")
    parser.add_argument("--min-longitude", type=float, help="Supported-area minimum longitude.")
    parser.add_argument("--max-longitude", type=float, help="Supported-area maximum longitude.")
    parser.add_argument("--width-m", type=float, default=DEFAULT_WIDTH_M, help="Supported-area width in metres.")
    parser.add_argument("--height-m", type=float, default=DEFAULT_HEIGHT_M, help="Supported-area height in metres.")
    parser.add_argument(
        "--altitude-agl",
        type=float,
        default=DEFAULT_ALTITUDE_AGL_M,
        help="Reference capture altitude above ground level in metres.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=DEFAULT_OVERLAP_RATIO,
        help="Desired grid overlap ratio from 0.0 to 0.95.",
    )
    parser.add_argument(
        "--heading-deg",
        type=float,
        default=DEFAULT_HEADING_DEG,
        help="Heading to assign to generated coverage-grid entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned coverage summary without writing the output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output metadata file.",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input).resolve()
    output_path = (script_dir / args.output).resolve()
    simulation_config_path = (script_dir / args.simulation_config).resolve()

    if output_path.exists() and not args.overwrite and not args.dry_run:
        print(
            "Refusing to overwrite existing metadata. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    document = augment_reference_metadata(
        metadata_path=input_path,
        simulation_config_path=simulation_config_path,
        output_path=output_path,
        width_m=args.width_m,
        height_m=args.height_m,
        altitude_agl_m=args.altitude_agl,
        overlap_ratio=args.overlap_ratio,
        heading_deg=args.heading_deg,
        center_latitude=args.center_latitude,
        center_longitude=args.center_longitude,
        min_latitude=args.min_latitude,
        max_latitude=args.max_latitude,
        min_longitude=args.min_longitude,
        max_longitude=args.max_longitude,
    )
    summary = summarize_metadata(document)
    print("Reference metadata coverage plan")
    print("==============================")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.dry_run:
        print("Dry run only; metadata file was not written.")
        return 0

    write_metadata(document, output_path)
    print(f"Updated metadata written to: {output_path}")
    return 0


def _annotate_legacy_point(point: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(point)
    point_id = str(annotated.get("id", ""))
    coverage_role = annotated.get("coverage_role")
    if coverage_role is None:
        annotated["coverage_role"] = "grid_anchor" if point_id.startswith("grid_") else "legacy_landmark"
    annotated.setdefault("capture_status", "synthetic_only")
    annotated.setdefault("manual_capture_required", True)
    return annotated


def _local_to_latlon(
    origin_latitude: float,
    origin_longitude: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    latitude = origin_latitude + north_m / _meters_per_degree_lat()
    longitude = origin_longitude + east_m / _meters_per_degree_lon(origin_latitude)
    return latitude, longitude


def _meters_per_degree_lat() -> float:
    return 111320.0


def _meters_per_degree_lon(latitude: float) -> float:
    return 111320.0 * math.cos(math.radians(latitude))


def _lawnmower_waypoints(points: list[dict[str, Any]]) -> list[list[float]]:
    by_row: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        by_row.setdefault(int(point["grid_row"]), []).append(point)

    waypoints: list[list[float]] = []
    for row in sorted(by_row):
        row_points = sorted(by_row[row], key=lambda point: int(point["grid_col"]))
        if row % 2 == 1:
            row_points.reverse()
        for point in row_points:
            waypoints.append([point["latitude"], point["longitude"]])
    return waypoints


if __name__ == "__main__":
    raise SystemExit(main())
