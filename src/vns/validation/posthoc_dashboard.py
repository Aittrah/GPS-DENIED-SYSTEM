from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vns.database.reference_db import DatabaseEntry, ReferenceDatabase
from vns.vision.extractor import FeatureExtractor
from vns.vision.verification import GeometricVerifier

logger = logging.getLogger("vns.validation.dashboard")


@dataclass(frozen=True)
class ManifestFrame:
    seq: int
    timestamp: float | None
    filename: str
    path: Path


@dataclass(frozen=True)
class DashboardSample:
    index: int
    timestamp: float | None
    frame_filename: str | None
    matched_ref_id: str | None
    navigation_mode: str | None
    gnss_status: str | None
    localization_success: bool
    inlier_count: int | None
    horizontal_error: float | None
    position_error_m: float | None
    frame_time_delta: float | None
    overlay_filename: str | None
    reference_filename: str | None
    reconstructed_inliers: int | None
    note: str | None


@dataclass(frozen=True)
class DashboardBuildResult:
    output_dir: Path
    html_path: Path
    overlay_count: int
    skipped_count: int
    samples: list[DashboardSample]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSONL row %s:%d (%s)",
                    path,
                    line_number,
                    exc,
                )
                continue
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}: line {line_number} must decode to a JSON object."
                )
            rows.append(payload)
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _load_manifest_frames(path: str | Path) -> list[ManifestFrame]:
    manifest_path = Path(path)
    rows = _read_jsonl(manifest_path)
    frames: list[ManifestFrame] = []
    for row in rows:
        filename = str(row["file"])
        frames.append(
            ManifestFrame(
                seq=int(row.get("seq", len(frames))),
                timestamp=_optional_float(row.get("timestamp")),
                filename=filename,
                path=(manifest_path.parent / filename).resolve(),
            )
        )
    frames.sort(key=lambda item: item.seq)
    return frames


def _align_frames(
    records: list[dict[str, Any]],
    frames: list[ManifestFrame],
) -> list[ManifestFrame | None]:
    if not frames:
        return [None] * len(records)

    remaining = list(range(len(frames)))
    aligned: list[ManifestFrame | None] = []

    for record_index, record in enumerate(records):
        if not remaining:
            aligned.append(None)
            continue

        timestamp = _optional_float(record.get("timestamp"))
        if timestamp is None:
            candidate = min(remaining, key=lambda idx: abs(idx - record_index))
        else:
            candidate = min(
                remaining,
                key=lambda idx: (
                    abs((_optional_float(frames[idx].timestamp) or timestamp) - timestamp),
                    abs(idx - record_index),
                ),
            )

        aligned.append(frames[candidate])
        remaining.remove(candidate)

    return aligned


def _reference_image_path(
    database: ReferenceDatabase,
    entry: DatabaseEntry,
) -> Path | None:
    try:
        return database.resolve_source_path(entry.source_path)
    except ValueError:
        logger.warning("Could not resolve source path for %s", entry.id)
        return None


def _draw_overlay(
    query_image: np.ndarray,
    reference_image: np.ndarray,
    query_points: np.ndarray,
    reference_points: np.ndarray,
    output_path: Path,
    *,
    max_points: int = 50,
) -> None:
    if len(query_image.shape) == 2:
        query_vis = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    else:
        query_vis = query_image.copy()
    if len(reference_image.shape) == 2:
        reference_vis = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    else:
        reference_vis = reference_image.copy()

    height = max(query_vis.shape[0], reference_vis.shape[0])
    width = query_vis.shape[1] + reference_vis.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: query_vis.shape[0], : query_vis.shape[1]] = query_vis
    canvas[
        : reference_vis.shape[0],
        query_vis.shape[1] : query_vis.shape[1] + reference_vis.shape[1],
    ] = reference_vis

    count = min(len(query_points), len(reference_points), max_points)
    rng = np.random.default_rng(42)

    for index in range(count):
        color = tuple(int(channel) for channel in rng.integers(64, 256, size=3))
        query_x, query_y = query_points[index]
        ref_x, ref_y = reference_points[index]
        start = (int(round(query_x)), int(round(query_y)))
        end = (
            int(round(ref_x)) + query_vis.shape[1],
            int(round(ref_y)),
        )
        cv2.circle(canvas, start, 4, color, 1)
        cv2.circle(canvas, end, 4, color, 1)
        cv2.line(canvas, start, end, color, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Failed to write overlay image: {output_path}")


def _reconstruct_overlay(
    *,
    record: dict[str, Any],
    frame: ManifestFrame,
    database: ReferenceDatabase,
    extractor: FeatureExtractor,
    verifier: GeometricVerifier,
    output_dir: Path,
    sample_index: int,
) -> tuple[str | None, str | None, int | None, str | None]:
    matched_ref_id = record.get("matched_ref_id")
    if not matched_ref_id:
        return None, None, None, "No matched_ref_id recorded."

    entry = database.entries.get(str(matched_ref_id))
    if entry is None:
        return None, None, None, f"Reference '{matched_ref_id}' is missing from the database."

    if not frame.path.exists():
        return None, None, None, f"Frame image '{frame.filename}' is missing."

    reference_path = _reference_image_path(database, entry)
    if reference_path is None or not reference_path.exists():
        return None, None, None, f"Reference image for '{matched_ref_id}' is missing."

    query_image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if query_image is None:
        return None, None, None, f"Could not read frame image '{frame.filename}'."
    if reference_image is None:
        return None, None, None, f"Could not read reference image '{reference_path.name}'."

    query_keypoints, query_descriptors = extractor.detect_and_compute(query_image)
    verification = verifier.verify(query_keypoints, query_descriptors, entry)
    if verification is None:
        return None, reference_path.name, None, "Offline verification did not recover enough inliers."

    overlay_name = f"sample_{sample_index:04d}_{entry.id}.png"
    overlay_path = output_dir / "overlays" / overlay_name
    _draw_overlay(
        query_image=query_image,
        reference_image=reference_image,
        query_points=verification.inliers_query,
        reference_points=verification.inliers_ref,
        output_path=overlay_path,
    )
    return (
        overlay_name,
        reference_path.name,
        verification.inlier_count,
        None,
    )


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, str]:
    horizontal_errors = [
        float(row["horizontal_error"])
        for row in records
        if row.get("horizontal_error") is not None
    ]
    position_errors = [
        float(row["position_error_m"])
        for row in records
        if row.get("position_error_m") is not None
    ]
    localized = sum(1 for row in records if bool(row.get("localization_success")))
    matched = sum(1 for row in records if row.get("matched_ref_id"))

    def _fmt_metric(values: list[float]) -> str:
        if not values:
            return "N/A"
        return f"{float(np.mean(values)):.3f}"

    return {
        "total": str(len(records)),
        "localized": str(localized),
        "matched": str(matched),
        "mean_horizontal": _fmt_metric(horizontal_errors),
        "mean_position": _fmt_metric(position_errors),
    }


def _evenly_sample_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if max_samples is None or len(records) <= max_samples:
        return list(records)
    if max_samples <= 0:
        return []
    if max_samples == 1:
        return [records[0]]

    last_index = len(records) - 1
    sampled: list[dict[str, Any]] = []
    previous_index = -1
    for step in range(max_samples):
        index = round(step * last_index / (max_samples - 1))
        if index <= previous_index:
            index = previous_index + 1
        if index > last_index:
            index = last_index
        sampled.append(records[index])
        previous_index = index
    return sampled


def _select_display_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Prefer records that can render overlays, then sample across the run.

    A dashboard with only the first N records is often useless for long runs:
    startup frames commonly have no matched_ref_id, even when later records do.
    """
    matched_records = [record for record in records if record.get("matched_ref_id")]
    source = matched_records if matched_records else records
    return _evenly_sample_records(source, max_samples=max_samples)


def _render_html(
    *,
    output_dir: Path,
    log_path: Path,
    manifest_path: Path,
    database_path: Path,
    summary: dict[str, str],
    samples: list[DashboardSample],
) -> str:
    rows = []
    for sample in samples:
        frame_delta = (
            "N/A"
            if sample.frame_time_delta is None
            else f"{sample.frame_time_delta:.3f}s"
        )
        horizontal_error = (
            "N/A"
            if sample.horizontal_error is None
            else f"{sample.horizontal_error:.3f} m"
        )
        position_error = (
            "N/A"
            if sample.position_error_m is None
            else f"{sample.position_error_m:.3f} m"
        )
        overlay_cell = "N/A"
        if sample.overlay_filename is not None:
            overlay_cell = (
                f'<a href="{html.escape("overlays/" + sample.overlay_filename)}">'
                f"{html.escape(sample.overlay_filename)}</a>"
            )
        note = html.escape(sample.note or "")
        rows.append(
            "<tr>"
            f"<td>{sample.index}</td>"
            f"<td>{html.escape(sample.frame_filename or 'N/A')}</td>"
            f"<td>{html.escape(sample.matched_ref_id or 'N/A')}</td>"
            f"<td>{html.escape(sample.navigation_mode or 'N/A')}</td>"
            f"<td>{html.escape(sample.gnss_status or 'N/A')}</td>"
            f"<td>{frame_delta}</td>"
            f"<td>{horizontal_error}</td>"
            f"<td>{position_error}</td>"
            f"<td>{sample.reconstructed_inliers if sample.reconstructed_inliers is not None else 'N/A'}</td>"
            f"<td>{overlay_cell}</td>"
            f"<td>{note}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VNS Post-hoc Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe4;
      --panel: #fffdf8;
      --ink: #1b1f24;
      --muted: #6b7280;
      --line: #d6cfc2;
      --accent: #0f766e;
      --accent-soft: #d8efe9;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #ebe2cf 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(27, 31, 36, 0.08);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .stat {{
      padding: 14px;
      background: var(--accent-soft);
      border-radius: 12px;
    }}
    .stat strong {{
      display: block;
      font-size: 24px;
      color: var(--accent);
    }}
    .panel {{
      padding: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>VNS Post-hoc Dashboard</h1>
      <div class="meta">
        Log: {html.escape(log_path.name)}<br>
        Manifest: {html.escape(manifest_path.name)}<br>
        Database: {html.escape(database_path.name)}
      </div>
      <div class="stats">
        <div class="stat"><span>Total Samples</span><strong>{summary['total']}</strong></div>
        <div class="stat"><span>Localization Successes</span><strong>{summary['localized']}</strong></div>
        <div class="stat"><span>Matched References</span><strong>{summary['matched']}</strong></div>
        <div class="stat"><span>Mean Horizontal Error</span><strong>{summary['mean_horizontal']}</strong></div>
        <div class="stat"><span>Mean Position Error</span><strong>{summary['mean_position']}</strong></div>
      </div>
    </section>
    <section class="panel">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Frame</th>
            <th>Matched Ref</th>
            <th>Mode</th>
            <th>GNSS</th>
            <th>Frame Delta</th>
            <th>Horizontal Error</th>
            <th>Position Error</th>
            <th>Offline Inliers</th>
            <th>Overlay</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def build_dashboard(
    *,
    log_path: str | Path,
    image_manifest_path: str | Path,
    database_path: str | Path,
    output_dir: str | Path,
    max_samples: int | None = 25,
) -> DashboardBuildResult:
    """Generate a static HTML dashboard and offline correspondence overlays."""
    log_path = Path(log_path)
    image_manifest_path = Path(image_manifest_path)
    database_path = Path(database_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(log_path)
    frames = _load_manifest_frames(image_manifest_path)
    database = ReferenceDatabase.load(str(database_path))
    extractor = FeatureExtractor()
    verifier = GeometricVerifier()

    display_records = _select_display_records(records, max_samples=max_samples)
    aligned_frames = _align_frames(display_records, frames)

    samples: list[DashboardSample] = []
    overlay_count = 0
    skipped_count = 0

    for index, (record, frame) in enumerate(zip(display_records, aligned_frames)):
        overlay_name = None
        reference_name = None
        reconstructed_inliers = None
        note = None

        if frame is None:
            skipped_count += 1
            note = "No frame manifest entry was available."
        else:
            overlay_name, reference_name, reconstructed_inliers, note = _reconstruct_overlay(
                record=record,
                frame=frame,
                database=database,
                extractor=extractor,
                verifier=verifier,
                output_dir=output_dir,
                sample_index=index,
            )
            if overlay_name is None:
                skipped_count += 1
            else:
                overlay_count += 1

        frame_time_delta = None
        if frame is not None:
            frame_timestamp = frame.timestamp
            record_timestamp = _optional_float(record.get("timestamp"))
            if frame_timestamp is not None and record_timestamp is not None:
                frame_time_delta = abs(frame_timestamp - record_timestamp)

        samples.append(
            DashboardSample(
                index=index,
                timestamp=_optional_float(record.get("timestamp")),
                frame_filename=None if frame is None else frame.filename,
                matched_ref_id=None
                if record.get("matched_ref_id") is None
                else str(record.get("matched_ref_id")),
                navigation_mode=None
                if record.get("navigation_mode") is None
                else str(record.get("navigation_mode")),
                gnss_status=None
                if record.get("gnss_status") is None
                else str(record.get("gnss_status")),
                localization_success=bool(record.get("localization_success")),
                inlier_count=None
                if record.get("inlier_count") is None
                else int(record.get("inlier_count")),
                horizontal_error=_optional_float(record.get("horizontal_error")),
                position_error_m=_optional_float(record.get("position_error_m")),
                frame_time_delta=frame_time_delta,
                overlay_filename=overlay_name,
                reference_filename=reference_name,
                reconstructed_inliers=reconstructed_inliers,
                note=note,
            )
        )

    html_path = output_dir / "index.html"
    html_path.write_text(
        _render_html(
            output_dir=output_dir,
            log_path=log_path,
            manifest_path=image_manifest_path,
            database_path=database_path,
            summary=_metric_summary(records),
            samples=samples,
        ),
        encoding="utf-8",
    )

    return DashboardBuildResult(
        output_dir=output_dir,
        html_path=html_path,
        overlay_count=overlay_count,
        skipped_count=skipped_count,
        samples=samples,
    )
