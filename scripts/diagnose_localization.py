#!/usr/bin/env python3
"""Diagnose why dashboard overlays/localization records are missing."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from openhouse_utils import (
    load_database_metadata,
    load_jsonl,
    mean,
    png_channel_stats,
    resolve_db_source_path,
)


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "logs"
IMAGE_LOG_DIR = LOG_DIR / "images"
DB_PATH = REPO / "simulation/database/qau_campus.vnsdb"
DB_INDEX_PATH = REPO / "simulation/database/images_real/database_index.yaml"
TEXTURE_PATH = (
    REPO / "simulation/models/qau_ground_plane/materials/textures/qau_satellite.png"
)
DASHBOARD_HTML = REPO / "reports/dashboard/index.html"
VNS_LOG = LOG_DIR / "vns.log"


def _latest_nonempty(pattern: str) -> Path | None:
    candidates = sorted(Path().glob(pattern))
    for path in reversed(candidates):
        if path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    return None


def _matching_manifest(ground_truth_log: Path | None) -> Path | None:
    if ground_truth_log is None:
        return None
    suffix = ground_truth_log.stem.split("_")[-1]
    candidate = IMAGE_LOG_DIR / f"manifest_{suffix}.jsonl"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate.resolve()
    return _latest_nonempty("logs/images/manifest_*.jsonl")


def _sample_query_frames(manifest_rows: list[dict], sample_count: int = 5) -> list[Path]:
    if not manifest_rows:
        return []
    step = max(1, len(manifest_rows) // sample_count)
    indices = list(range(0, len(manifest_rows), step))[:sample_count]
    if indices and indices[-1] != len(manifest_rows) - 1:
        indices[-1] = len(manifest_rows) - 1
    paths: list[Path] = []
    for index in indices:
        filename = manifest_rows[index].get("file")
        if not filename:
            continue
        path = (IMAGE_LOG_DIR / str(filename)).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _query_feature_average(query_image_count: int) -> float | None:
    if not VNS_LOG.is_file():
        return None

    pattern = re.compile(r"Extracted (\d+) features")
    extracted: list[float] = []
    with VNS_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                extracted.append(float(match.group(1)))

    if not extracted:
        return None
    if query_image_count > 0 and len(extracted) >= query_image_count:
        extracted = extracted[-query_image_count:]
    return mean(extracted)


def _dashboard_note() -> str | None:
    if not DASHBOARD_HTML.is_file():
        return None
    text = DASHBOARD_HTML.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"Offline verification did not recover enough inliers\.",
        text,
    )
    if match:
        return match.group(0)
    return None


def build_report() -> str:
    ground_truth_logs = sorted(LOG_DIR.glob("ground_truth_*.jsonl"))
    latest_ground_truth = _latest_nonempty("logs/ground_truth_*.jsonl")
    latest_manifest = _matching_manifest(latest_ground_truth)

    records: list[dict] = []
    invalid_ground_truth_lines = 0
    if latest_ground_truth is not None:
        records, invalid_ground_truth_lines = load_jsonl(latest_ground_truth)

    manifest_rows: list[dict] = []
    invalid_manifest_lines = 0
    if latest_manifest is not None:
        manifest_rows, invalid_manifest_lines = load_jsonl(latest_manifest)

    matched_counter = Counter(
        str(row["matched_ref_id"])
        for row in records
        if row.get("matched_ref_id") is not None
    )
    reason_counter = Counter(
        str(
            row.get("localization_reason")
            or row.get("failure_reason")
            or "unknown"
        )
        for row in records
    )
    matched_count = sum(1 for row in records if row.get("matched_ref_id") is not None)
    success_count = sum(1 for row in records if bool(row.get("localization_success")))

    database_metadata = load_database_metadata(DB_PATH)
    entries = [
        entry
        for entry in database_metadata.get("entries", [])
        if isinstance(entry, dict)
    ]
    reference_feature_counts = [
        float(entry.get("feature_count", 0))
        for entry in entries
        if entry.get("feature_count") is not None
    ]
    reference_paths = [
        resolve_db_source_path(DB_PATH, str(entry.get("source_path", "")))
        for entry in entries
    ]
    reference_real_satellite = (
        DB_INDEX_PATH.is_file()
        and "source_type: real_satellite" in DB_INDEX_PATH.read_text(
            encoding="utf-8", errors="ignore"
        )
        and all(
            path.is_file() and "simulation/database/images_real/" in str(path)
            for path in reference_paths
        )
    )

    query_samples = _sample_query_frames(manifest_rows)
    query_stats = [png_channel_stats(path) for path in query_samples]
    avg_query_green_ratio = mean([item.green_pixel_ratio for item in query_stats]) or 0.0
    avg_query_mean_r = mean([item.mean_r for item in query_stats]) or 0.0
    avg_query_mean_g = mean([item.mean_g for item in query_stats]) or 0.0
    avg_query_mean_b = mean([item.mean_b for item in query_stats]) or 0.0
    query_green_dominant = any(item.is_green_dominant for item in query_stats)

    query_feature_avg = _query_feature_average(len(manifest_rows))
    reference_feature_avg = mean(reference_feature_counts)
    dashboard_note = _dashboard_note()

    logs_are_new_real_texture = (
        not query_green_dominant and reference_real_satellite and matched_count > 0
    )
    mtime_note = ""
    if latest_manifest is not None and TEXTURE_PATH.is_file():
        if latest_manifest.stat().st_mtime < TEXTURE_PATH.stat().st_mtime:
            mtime_note = " (latest log predates the current rebuilt texture file timestamp)"

    top_matched_text = ", ".join(
        f"{ref_id}={count}" for ref_id, count in matched_counter.most_common(10)
    ) or "none"
    top_reasons_text = ", ".join(
        f"{reason}={count}" for reason, count in reason_counter.most_common(5)
    ) or "none"

    if matched_count == 0:
        geometric_reason = "No matched_ref_id records exist, so overlays cannot be reconstructed."
    else:
        geometric_reason = (
            f"Latest run reasons: {top_reasons_text}. "
            f"Dashboard note: {dashboard_note or 'not available'}"
        )

    if success_count == 0:
        dashboard_sampling = (
            "failed logs; the dashboard can only sample matched-but-unsuccessful rows "
            "because localization_success count is 0"
        )
    else:
        dashboard_sampling = "useful logs; successful localization records exist"

    lines = [
        "Localization Diagnosis",
        "======================",
        f"Number of ground_truth logs: {len(ground_truth_logs)}",
        f"Newest ground_truth log path: {latest_ground_truth or 'not found'}",
        (
            f"Total records: {len(records)} valid JSON rows"
            + (
                f" ({invalid_ground_truth_lines} truncated/invalid line(s) ignored)"
                if invalid_ground_truth_lines
                else ""
            )
        ),
        f"matched_ref_id record count: {matched_count}",
        f"localization_success record count: {success_count}",
        f"Top matched reference IDs: {top_matched_text}",
        f"Number of image manifest logs: {len(list(IMAGE_LOG_DIR.glob('manifest_*.jsonl')))}",
        (
            f"Number of query images: {len(manifest_rows)}"
            + (
                f" ({invalid_manifest_lines} invalid line(s) ignored)"
                if invalid_manifest_lines
                else ""
            )
        ),
        (
            "Whether query images are green-dominant: "
            f"{'yes' if query_green_dominant else 'no'} "
            f"(sampled {len(query_stats)} PNG frame(s); mean RGB "
            f"{avg_query_mean_r:.1f}/{avg_query_mean_g:.1f}/{avg_query_mean_b:.1f}, "
            f"green-dominant pixel ratio {avg_query_green_ratio:.3f})"
        ),
        (
            "Whether reference tiles are real satellite tiles: "
            f"{'yes' if reference_real_satellite else 'no'} "
            f"(database_index.yaml source_type={'real_satellite' if reference_real_satellite else 'unknown'})"
        ),
        (
            "Average ORB keypoints in query images: "
            + (
                f"{query_feature_avg:.2f} "
                f"(from the last {len(manifest_rows)} 'Extracted N features' lines in logs/vns.log)"
                if query_feature_avg is not None
                else "unavailable"
            )
        ),
        (
            "Average ORB keypoints in reference tiles: "
            + (
                f"{reference_feature_avg:.2f} (from qau_campus.vnsdb feature_count metadata)"
                if reference_feature_avg is not None
                else "unavailable"
            )
        ),
        f"Reason geometric verification failed: {geometric_reason}",
        (
            "Whether logs are old green-frame logs or new real-texture logs: "
            + (
                f"new real-texture logs{mtime_note}"
                if logs_are_new_real_texture
                else "unclear or older/green-frame logs"
            )
        ),
        f"Whether dashboard is sampling failed logs or useful logs: {dashboard_sampling}",
        (
            "Texture path check: simulation/worlds/uav_test_world.sdf and "
            "simulation/worlds/uav_localization_test.sdf both include model://qau_ground_plane, "
            "which resolves to simulation/models/qau_ground_plane/materials/textures/qau_satellite.png"
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose localization/dashboard failures.")
    parser.add_argument(
        "--output",
        default=str(REPO / "reports/openhouse_proof/localization_diagnosis.txt"),
        help="Where to write the diagnosis report.",
    )
    args = parser.parse_args()

    report = build_report()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
