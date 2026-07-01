from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from vns.database.reference_db import ReferenceDatabase
from vns.utils.paths import resolve_from_base

DEFAULT_CAMERA = {
    "width": 640.0,
    "height": 480.0,
    "fx": 554.25,
    "fy": 554.25,
}
DEFAULT_ALTITUDE_AGL_M = 50.0
DEFAULT_OVERLAP_RATIO = 0.0
SUPPORTED_COVERAGE_ROLES = {"coverage_grid", "grid_anchor"}
KNOWN_SOURCE_TYPES = {"synthetic", "gazebo_camera", "unknown"}
GRID_SPACING_TOLERANCE_M = 2.0


@dataclass(frozen=True)
class CoverageEntry:
    id: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    metadata: dict[str, Any] = field(default_factory=dict)
    filepath: str | None = None


@dataclass
class CoverageValidationReport:
    metadata_entry_count: int = 0
    index_entry_count: int | None = None
    database_entry_count: int | None = None
    coverage_tile_count: int = 0
    supporting_tile_count: int = 0
    unique_coordinate_count: int = 0
    estimated_coverage_pct: float | None = None
    frame_footprint_m: tuple[float, float] | None = None
    expected_grid_spacing_m: tuple[float, float] | None = None
    source_type_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def load_yaml_document(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a top-level mapping.")
    return payload


def camera_footprint_m(
    camera_cfg: Mapping[str, Any],
    altitude_agl_m: float,
) -> tuple[float, float]:
    width = float(camera_cfg.get("width", DEFAULT_CAMERA["width"]))
    height = float(camera_cfg.get("height", DEFAULT_CAMERA["height"]))
    fx = float(camera_cfg.get("fx", DEFAULT_CAMERA["fx"]))
    fy = float(camera_cfg.get("fy", DEFAULT_CAMERA["fy"]))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive to estimate coverage.")
    return (
        width * float(altitude_agl_m) / fx,
        height * float(altitude_agl_m) / fy,
    )


def format_report(report: CoverageValidationReport) -> str:
    lines = [
        "Reference Coverage Validation",
        "=============================",
        f"Metadata entries:        {report.metadata_entry_count}",
        (
            "Index entries:           "
            f"{report.index_entry_count if report.index_entry_count is not None else 'n/a'}"
        ),
        (
            "Database entries:        "
            f"{report.database_entry_count if report.database_entry_count is not None else 'n/a'}"
        ),
        f"Coverage-grid tiles:     {report.coverage_tile_count}",
        f"Coverage-support tiles:  {report.supporting_tile_count}",
        f"Unique coordinates:      {report.unique_coordinate_count}",
    ]

    if report.frame_footprint_m is not None:
        lines.append(
            "Frame footprint (m):    "
            f"{report.frame_footprint_m[0]:.2f} x {report.frame_footprint_m[1]:.2f}"
        )
    if report.expected_grid_spacing_m is not None:
        lines.append(
            "Expected spacing (m):   "
            f"{report.expected_grid_spacing_m[0]:.2f} x {report.expected_grid_spacing_m[1]:.2f}"
        )
    if report.estimated_coverage_pct is not None:
        lines.append(
            f"Estimated coverage:     {report.estimated_coverage_pct:.2f}%"
        )

    source_counts = report.source_type_counts or {"unknown": 0}
    lines.append(
        "Source types:           "
        + ", ".join(
            f"{source_type}={count}"
            for source_type, count in sorted(source_counts.items())
        )
    )

    if report.errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {message}" for message in report.errors)
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {message}" for message in report.warnings)

    return "\n".join(lines)


def load_camera_config(simulation_config_path: str | Path | None) -> dict[str, Any]:
    if simulation_config_path is None:
        return dict(DEFAULT_CAMERA)
    payload = load_yaml_document(simulation_config_path)
    camera = payload.get("camera", {})
    return camera if isinstance(camera, dict) else dict(DEFAULT_CAMERA)


def validate_reference_coverage(
    *,
    metadata_path: str | Path,
    index_path: str | Path | None = None,
    database_path: str | Path | None = None,
    simulation_config_path: str | Path | None = None,
) -> CoverageValidationReport:
    metadata_doc = load_yaml_document(metadata_path)
    report = CoverageValidationReport()
    metadata_entries = _metadata_entries(metadata_doc, report)
    report.metadata_entry_count = len(metadata_entries)
    report.unique_coordinate_count = len(
        {(round(entry.latitude, 7), round(entry.longitude, 7)) for entry in metadata_entries}
    )

    database_meta = metadata_doc.get("database", {})
    if not isinstance(database_meta, dict):
        report.add_error("metadata.database must be a mapping.")
        return report

    coverage_plan = database_meta.get("coverage_plan", {})
    coverage_plan = coverage_plan if isinstance(coverage_plan, dict) else {}

    camera_cfg = load_camera_config(simulation_config_path)
    altitude_agl_m = _coverage_altitude_agl(coverage_plan)
    overlap_ratio = _coverage_overlap_ratio(coverage_plan)
    footprint_w, footprint_h = camera_footprint_m(camera_cfg, altitude_agl_m)
    report.frame_footprint_m = (footprint_w, footprint_h)
    report.expected_grid_spacing_m = (
        footprint_w * (1.0 - overlap_ratio),
        footprint_h * (1.0 - overlap_ratio),
    )

    coverage_tiles = [entry for entry in metadata_entries if _coverage_role(entry) == "coverage_grid"]
    supporting_tiles = [
        entry for entry in metadata_entries if _coverage_role(entry) in SUPPORTED_COVERAGE_ROLES
    ]
    report.coverage_tile_count = len(coverage_tiles)
    report.supporting_tile_count = len(supporting_tiles)

    _validate_entry_values(metadata_entries, "metadata", report)
    _validate_duplicate_coordinates(metadata_entries, report)
    _validate_grid_spacing(
        coverage_tiles,
        report,
        database_meta,
    )

    supported_area_m2 = _supported_area_m2(database_meta, coverage_plan)
    if supported_area_m2 is not None and supported_area_m2 > 0.0:
        coverage_basis = coverage_tiles if coverage_tiles else supporting_tiles
        union_area = _estimate_union_area(
            coverage_basis,
            footprint_w,
            footprint_h,
            database_meta,
        )
        report.estimated_coverage_pct = min(100.0, (union_area / supported_area_m2) * 100.0)

    if index_path is not None:
        index_doc = load_yaml_document(index_path)
        index_entries = _index_entries(index_doc, report)
        report.index_entry_count = len(index_entries)
        _validate_entry_values(index_entries, "index", report)
        _validate_index_against_metadata(
            metadata_entries,
            index_entries,
            Path(index_path).resolve().parent,
            report,
        )
        report.source_type_counts = _source_type_counts(index_entries)
    elif database_path is None:
        report.source_type_counts = _source_type_counts(metadata_entries)

    if database_path is not None:
        db_entries = _database_entries(database_path, report)
        report.database_entry_count = len(db_entries)
        _validate_entry_values(db_entries, "database", report)
        _validate_database_against_metadata(metadata_entries, db_entries, report)
        if not report.source_type_counts:
            report.source_type_counts = _source_type_counts(db_entries)

    if report.coverage_tile_count == 0:
        report.add_warning(
            "No entries are tagged as coverage_grid; coverage percentage is based on the support subset."
        )
    if report.supporting_tile_count == 0:
        report.add_warning(
            "No entries are tagged with coverage-support roles (coverage_grid / grid_anchor)."
        )

    return report


def _metadata_entries(
    metadata_doc: Mapping[str, Any],
    report: CoverageValidationReport,
) -> list[CoverageEntry]:
    rows = metadata_doc.get("reference_points", [])
    if not isinstance(rows, list):
        report.add_error("metadata.reference_points must be a list.")
        return []
    entries: list[CoverageEntry] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            report.add_error(f"metadata.reference_points[{index}] must be a mapping.")
            continue
        entry = _coerce_entry(row, source_label=f"metadata.reference_points[{index}]")
        if entry is not None:
            entries.append(entry)
    return entries


def _index_entries(
    index_doc: Mapping[str, Any],
    report: CoverageValidationReport,
) -> list[CoverageEntry]:
    rows = index_doc.get("images", [])
    if not isinstance(rows, list):
        report.add_error("index.images must be a list.")
        return []
    entries: list[CoverageEntry] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            report.add_error(f"index.images[{index}] must be a mapping.")
            continue
        entry = _coerce_entry(row, source_label=f"index.images[{index}]")
        if entry is None:
            continue
        filepath = row.get("filepath")
        entries.append(
            CoverageEntry(
                id=entry.id,
                latitude=entry.latitude,
                longitude=entry.longitude,
                altitude=entry.altitude,
                heading=entry.heading,
                metadata=entry.metadata,
                filepath=str(filepath) if filepath is not None else None,
            )
        )
    return entries


def _database_entries(
    database_path: str | Path,
    report: CoverageValidationReport,
) -> list[CoverageEntry]:
    db = ReferenceDatabase.load(str(Path(database_path).resolve()))
    entries: list[CoverageEntry] = []
    for row in db.entries.values():
        entries.append(
            CoverageEntry(
                id=row.id,
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                altitude=float(row.altitude),
                heading=float(row.heading),
                metadata=dict(row.metadata),
                filepath=str(row.source_path),
            )
        )
    return entries


def _coerce_entry(row: Mapping[str, Any], *, source_label: str) -> CoverageEntry | None:
    required = ("id", "latitude", "longitude", "altitude", "heading")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{source_label} is missing required fields: {', '.join(missing)}")
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"id", "latitude", "longitude", "altitude", "heading", "filepath"}
    }
    return CoverageEntry(
        id=str(row["id"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        altitude=float(row["altitude"]),
        heading=float(row["heading"]),
        metadata=metadata,
        filepath=str(row["filepath"]) if "filepath" in row else None,
    )


def _coverage_role(entry: CoverageEntry) -> str:
    return str(entry.metadata.get("coverage_role", "")).strip().lower()


def _coverage_altitude_agl(coverage_plan: Mapping[str, Any]) -> float:
    return float(coverage_plan.get("altitude_agl_m", DEFAULT_ALTITUDE_AGL_M))


def _coverage_overlap_ratio(coverage_plan: Mapping[str, Any]) -> float:
    overlap_ratio = float(coverage_plan.get("overlap_ratio", DEFAULT_OVERLAP_RATIO))
    return min(max(overlap_ratio, 0.0), 0.95)


def _supported_area_m2(
    database_meta: Mapping[str, Any],
    coverage_plan: Mapping[str, Any],
) -> float | None:
    width_m = coverage_plan.get("width_m")
    height_m = coverage_plan.get("height_m")
    if width_m is not None and height_m is not None:
        return float(width_m) * float(height_m)

    bounds = database_meta.get("bounds")
    center = database_meta.get("center")
    if not isinstance(bounds, dict) or not isinstance(center, dict):
        return None
    min_lat = float(bounds["min_latitude"])
    max_lat = float(bounds["max_latitude"])
    min_lon = float(bounds["min_longitude"])
    max_lon = float(bounds["max_longitude"])
    center_lat = float(center["latitude"])
    width = (max_lon - min_lon) * _meters_per_degree_lon(center_lat)
    height = (max_lat - min_lat) * _meters_per_degree_lat()
    return abs(width * height)


def _estimate_union_area(
    entries: Iterable[CoverageEntry],
    footprint_w: float,
    footprint_h: float,
    database_meta: Mapping[str, Any],
) -> float:
    center = database_meta.get("center", {})
    if not isinstance(center, dict):
        return 0.0
    origin_lat = float(center.get("latitude", 33.7470))
    origin_lon = float(center.get("longitude", 73.1370))

    rectangles: list[tuple[float, float, float, float]] = []
    half_w = footprint_w / 2.0
    half_h = footprint_h / 2.0
    for entry in entries:
        east_m, north_m = _local_offsets_m(
            entry.latitude,
            entry.longitude,
            origin_lat,
            origin_lon,
        )
        rectangles.append(
            (
                east_m - half_w,
                east_m + half_w,
                north_m - half_h,
                north_m + half_h,
            )
        )

    if not rectangles:
        return 0.0

    x_points = sorted({value for rect in rectangles for value in rect[:2]})
    union_area = 0.0
    for start, end in zip(x_points, x_points[1:]):
        if end <= start:
            continue
        overlaps = []
        for rect in rectangles:
            min_x, max_x, min_y, max_y = rect
            if min_x < end and max_x > start:
                overlaps.append((min_y, max_y))
        if not overlaps:
            continue
        overlaps.sort()
        covered_y = 0.0
        current_start, current_end = overlaps[0]
        for next_start, next_end in overlaps[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
                continue
            covered_y += current_end - current_start
            current_start, current_end = next_start, next_end
        covered_y += current_end - current_start
        union_area += (end - start) * covered_y
    return union_area


def _validate_entry_values(
    entries: Iterable[CoverageEntry],
    label: str,
    report: CoverageValidationReport,
) -> None:
    for entry in entries:
        if not math.isfinite(entry.latitude) or not (-90.0 <= entry.latitude <= 90.0):
            report.add_error(f"{label} entry '{entry.id}' has invalid latitude {entry.latitude!r}.")
        if not math.isfinite(entry.longitude) or not (-180.0 <= entry.longitude <= 180.0):
            report.add_error(f"{label} entry '{entry.id}' has invalid longitude {entry.longitude!r}.")
        if not math.isfinite(entry.altitude) or abs(entry.altitude) > 20000.0:
            report.add_error(f"{label} entry '{entry.id}' has invalid altitude {entry.altitude!r}.")


def _validate_duplicate_coordinates(
    entries: Iterable[CoverageEntry],
    report: CoverageValidationReport,
) -> None:
    grouped: dict[tuple[float, float], list[CoverageEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(round(entry.latitude, 7), round(entry.longitude, 7))].append(entry)

    for coordinates, grouped_entries in grouped.items():
        if len(grouped_entries) <= 1:
            continue
        ids = [entry.id for entry in grouped_entries]
        coverage_duplicates = [
            entry.id for entry in grouped_entries if _coverage_role(entry) == "coverage_grid"
        ]
        if coverage_duplicates:
            report.add_error(
                "duplicate coverage-grid coordinates "
                f"{coordinates}: {', '.join(sorted(ids))}"
            )
            continue
        report.add_warning(
            f"duplicate coordinates {coordinates} retained for legacy entries: {', '.join(sorted(ids))}"
        )


def _validate_grid_spacing(
    coverage_tiles: list[CoverageEntry],
    report: CoverageValidationReport,
    database_meta: Mapping[str, Any],
) -> None:
    if not coverage_tiles:
        return

    center = database_meta.get("center", {})
    if not isinstance(center, dict):
        return
    origin_lat = float(center.get("latitude", 33.7470))
    origin_lon = float(center.get("longitude", 73.1370))

    by_row: dict[int, list[tuple[int, CoverageEntry, float]]] = defaultdict(list)
    by_col: dict[int, list[tuple[int, CoverageEntry, float]]] = defaultdict(list)
    missing_grid_indices = [
        entry.id
        for entry in coverage_tiles
        if "grid_row" not in entry.metadata or "grid_col" not in entry.metadata
    ]
    if missing_grid_indices:
        report.add_error(
            "coverage_grid entries are missing grid_row/grid_col: "
            + ", ".join(sorted(missing_grid_indices))
        )
        return

    for entry in coverage_tiles:
        row = int(entry.metadata["grid_row"])
        col = int(entry.metadata["grid_col"])
        east_m, north_m = _local_offsets_m(entry.latitude, entry.longitude, origin_lat, origin_lon)
        by_row[row].append((col, entry, east_m))
        by_col[col].append((row, entry, north_m))

    expected_spacing = report.expected_grid_spacing_m
    if expected_spacing is None:
        return
    expected_east, expected_north = expected_spacing

    for row, items in by_row.items():
        sorted_items = sorted(items, key=lambda item: item[0])
        for (left_col, left_entry, left_east), (right_col, right_entry, right_east) in zip(
            sorted_items,
            sorted_items[1:],
        ):
            if right_col != left_col + 1:
                continue
            spacing = abs(right_east - left_east)
            if abs(spacing - expected_east) > GRID_SPACING_TOLERANCE_M:
                report.add_error(
                    "unexpected east-west spacing between "
                    f"{left_entry.id} and {right_entry.id}: {spacing:.2f} m "
                    f"(expected {expected_east:.2f} m)"
                )

    for col, items in by_col.items():
        sorted_items = sorted(items, key=lambda item: item[0])
        for (upper_row, upper_entry, upper_north), (lower_row, lower_entry, lower_north) in zip(
            sorted_items,
            sorted_items[1:],
        ):
            if lower_row != upper_row + 1:
                continue
            spacing = abs(upper_north - lower_north)
            if abs(spacing - expected_north) > GRID_SPACING_TOLERANCE_M:
                report.add_error(
                    "unexpected north-south spacing between "
                    f"{upper_entry.id} and {lower_entry.id}: {spacing:.2f} m "
                    f"(expected {expected_north:.2f} m)"
                )


def _validate_index_against_metadata(
    metadata_entries: list[CoverageEntry],
    index_entries: list[CoverageEntry],
    index_dir: Path,
    report: CoverageValidationReport,
) -> None:
    index_ids = {entry.id for entry in index_entries}
    for entry in metadata_entries:
        if entry.id not in index_ids:
            report.add_error(f"metadata entry '{entry.id}' has no matching index/image entry.")

    for entry in index_entries:
        if entry.filepath is None:
            report.add_error(f"index entry '{entry.id}' is missing filepath metadata.")
            continue
        image_path = resolve_from_base(entry.filepath, index_dir)
        if not image_path.exists():
            report.add_error(
                f"index entry '{entry.id}' references a missing image file: {image_path}"
            )
        elif not image_path.is_file():
            report.add_error(
                f"index entry '{entry.id}' filepath is not a file: {image_path}"
            )


def _validate_database_against_metadata(
    metadata_entries: list[CoverageEntry],
    database_entries: list[CoverageEntry],
    report: CoverageValidationReport,
) -> None:
    metadata_ids = {entry.id for entry in metadata_entries}
    database_ids = {entry.id for entry in database_entries}
    for entry_id in sorted(metadata_ids - database_ids):
        report.add_error(f"metadata entry '{entry_id}' is missing from the built database.")
    for entry_id in sorted(database_ids - metadata_ids):
        report.add_warning(f"database entry '{entry_id}' has no matching metadata entry.")


def _source_type_counts(entries: Iterable[CoverageEntry]) -> dict[str, int]:
    counts = Counter(_source_type(entry) for entry in entries)
    for source_type in KNOWN_SOURCE_TYPES:
        counts.setdefault(source_type, 0)
    return dict(sorted(counts.items()))


def _source_type(entry: CoverageEntry) -> str:
    raw = str(entry.metadata.get("source_type", "")).strip().lower()
    if raw in {"synthetic", "gazebo_camera"}:
        return raw

    capture_status = str(entry.metadata.get("capture_status", "")).strip().lower()
    if capture_status == "synthetic_only":
        return "synthetic"
    if capture_status == "captured_gazebo":
        return "gazebo_camera"
    return "unknown"


def _local_offsets_m(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    north_m = (latitude - origin_latitude) * _meters_per_degree_lat()
    east_m = (longitude - origin_longitude) * _meters_per_degree_lon(origin_latitude)
    return east_m, north_m


def _meters_per_degree_lat() -> float:
    return 111320.0


def _meters_per_degree_lon(latitude: float) -> float:
    return 111320.0 * math.cos(math.radians(latitude))
