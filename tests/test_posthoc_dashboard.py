from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vns.database.reference_db import ReferenceDatabase
from vns.validation.posthoc_dashboard import build_dashboard


def _feature_rich_image() -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for x in range(40, 640, 80):
        cv2.line(image, (x, 0), (x, 479), (255, 255, 255), 2)
    for y in range(40, 480, 80):
        cv2.line(image, (0, y), (639, y), (255, 255, 255), 2)
    cv2.rectangle(image, (80, 80), (240, 220), (0, 200, 255), -1)
    cv2.circle(image, (360, 220), 70, (255, 0, 180), 4)
    cv2.putText(
        image,
        "QAU",
        (260, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 0),
        4,
    )
    return image


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_test_database(tmp_path: Path) -> tuple[Path, Path]:
    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), _feature_rich_image())

    database = ReferenceDatabase(name="dashboard-test")
    database.add_image(
        str(reference_path),
        latitude=33.7470,
        longitude=73.1370,
        altitude=550.0,
        heading=0.0,
        image_id="ref_01",
    )
    database_path = tmp_path / "test.vnsdb"
    database.save(str(database_path))
    return database_path, reference_path


def test_build_dashboard_generates_overlay(tmp_path: Path) -> None:
    database_path, reference_path = _build_test_database(tmp_path)

    frame_path = tmp_path / "logs" / "images" / "frame_000000.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), cv2.imread(str(reference_path), cv2.IMREAD_COLOR))
    manifest_path = tmp_path / "logs" / "images" / "manifest.jsonl"
    log_path = tmp_path / "logs" / "ground_truth.jsonl"

    _write_jsonl(
        manifest_path,
        [{"seq": 0, "timestamp": 10.0, "file": frame_path.name}],
    )
    _write_jsonl(
        log_path,
        [
            {
                "timestamp": 10.0,
                "localization_success": True,
                "matched_ref_id": "ref_01",
                "navigation_mode": "VISION",
                "gnss_status": "DENIED",
                "horizontal_error": 1.25,
                "position_error_m": 1.31,
                "inlier_count": 22,
            }
        ],
    )

    result = build_dashboard(
        log_path=log_path,
        image_manifest_path=manifest_path,
        database_path=database_path,
        output_dir=tmp_path / "dashboard",
        max_samples=10,
    )

    assert result.html_path.exists()
    assert result.overlay_count == 1
    overlays = sorted((tmp_path / "dashboard" / "overlays").glob("*.png"))
    assert len(overlays) == 1
    html_text = result.html_path.read_text(encoding="utf-8")
    assert "ref_01" in html_text
    assert overlays[0].name in html_text


def test_build_dashboard_handles_missing_match_ids(tmp_path: Path) -> None:
    database_path, reference_path = _build_test_database(tmp_path)

    frame_path = tmp_path / "logs" / "images" / "frame_000000.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), cv2.imread(str(reference_path), cv2.IMREAD_COLOR))
    manifest_path = tmp_path / "logs" / "images" / "manifest.jsonl"
    log_path = tmp_path / "logs" / "ground_truth.jsonl"

    _write_jsonl(
        manifest_path,
        [{"seq": 0, "timestamp": 15.0, "file": frame_path.name}],
    )
    _write_jsonl(
        log_path,
        [
            {
                "timestamp": 15.0,
                "localization_success": False,
                "matched_ref_id": None,
                "navigation_mode": "FAILSAFE",
                "gnss_status": "DENIED",
                "inlier_count": 0,
            }
        ],
    )

    result = build_dashboard(
        log_path=log_path,
        image_manifest_path=manifest_path,
        database_path=database_path,
        output_dir=tmp_path / "dashboard",
        max_samples=10,
    )

    assert result.html_path.exists()
    assert result.overlay_count == 0
    assert result.skipped_count == 1
    html_text = result.html_path.read_text(encoding="utf-8")
    assert "No matched_ref_id recorded." in html_text


def test_build_dashboard_prefers_matched_records_for_sampling(tmp_path: Path) -> None:
    database_path, reference_path = _build_test_database(tmp_path)

    frame_dir = tmp_path / "logs" / "images"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / "frame_000100.png"
    cv2.imwrite(str(frame_path), cv2.imread(str(reference_path), cv2.IMREAD_COLOR))
    manifest_path = frame_dir / "manifest.jsonl"
    log_path = tmp_path / "logs" / "ground_truth.jsonl"

    manifest_rows = [{"seq": i, "timestamp": float(i), "file": "frame_000100.png"} for i in range(101)]
    log_rows = [
        {
            "timestamp": float(i),
            "localization_success": False,
            "matched_ref_id": None,
            "navigation_mode": "VISION",
            "gnss_status": "DENIED",
            "inlier_count": 0,
        }
        for i in range(100)
    ]
    log_rows.append(
        {
            "timestamp": 100.0,
            "localization_success": True,
            "matched_ref_id": "ref_01",
            "navigation_mode": "VISION",
            "gnss_status": "DENIED",
            "horizontal_error": 1.0,
            "position_error_m": 1.0,
            "inlier_count": 25,
        }
    )

    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(log_path, log_rows)

    result = build_dashboard(
        log_path=log_path,
        image_manifest_path=manifest_path,
        database_path=database_path,
        output_dir=tmp_path / "dashboard",
        max_samples=10,
    )

    assert result.overlay_count == 1
    assert any(sample.matched_ref_id == "ref_01" for sample in result.samples)


def test_build_dashboard_skips_malformed_trailing_jsonl_row(tmp_path: Path) -> None:
    database_path, reference_path = _build_test_database(tmp_path)

    frame_path = tmp_path / "logs" / "images" / "frame_000000.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), cv2.imread(str(reference_path), cv2.IMREAD_COLOR))
    manifest_path = tmp_path / "logs" / "images" / "manifest.jsonl"
    log_path = tmp_path / "logs" / "ground_truth.jsonl"

    _write_jsonl(
        manifest_path,
        [{"seq": 0, "timestamp": 10.0, "file": frame_path.name}],
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {
                "timestamp": 10.0,
                "localization_success": True,
                "matched_ref_id": "ref_01",
                "navigation_mode": "VISION",
                "gnss_status": "DENIED",
                "horizontal_error": 1.25,
                "position_error_m": 1.31,
                "inlier_count": 22,
            }
        ) + "\n")
        handle.write('{"timestamp": 10.1, "localization_success": false')

    result = build_dashboard(
        log_path=log_path,
        image_manifest_path=manifest_path,
        database_path=database_path,
        output_dir=tmp_path / "dashboard",
        max_samples=10,
    )

    assert result.html_path.exists()
    assert result.overlay_count == 1
