"""Unit tests for ReferenceDatabase (FR-8)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vns.database.reference_database import ReferenceDatabase, ORB_PARAMS


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))
    # Add some structure so ORB finds keypoints
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 0), -1)
    cv2.rectangle(img, (300, 300), (450, 450), (0, 255, 255), -1)
    cv2.circle(img, (256, 256), 80, (255, 0, 255), 4)
    return img


def _save_image(img: np.ndarray, path: Path) -> str:
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    img = _synthetic_image(42)
    img_path = tmp_path / "sat.png"
    _save_image(img, img_path)
    return ReferenceDatabase(db_path=str(db_path)), str(img_path), tmp_path


def test_build_creates_db(tmp_path):
    db_path = tmp_path / "test.db"
    img = _synthetic_image(1)
    img_path = tmp_path / "sat.png"
    _save_image(img, img_path)

    db = ReferenceDatabase(db_path=str(db_path))
    count = db.build(str(img_path), lat=33.7470, lon=73.1370, alt=550.0)

    assert db_path.exists(), "Database file should be created"
    assert count > 0, "At least one patch should be stored"


def test_query_returns_top_k(tmp_db):
    db, img_path, tmp_path = tmp_db
    db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)

    uav_patch = _synthetic_image(42)[128:384, 128:384]
    results = db.query(uav_patch, top_k=5)

    assert len(results) <= 5
    assert all("confidence" in r for r in results)


def test_confidence_range(tmp_db):
    db, img_path, _ = tmp_db
    db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)

    uav_patch = _synthetic_image(7)[0:256, 0:256]
    results = db.query(uav_patch, top_k=5)

    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0, (
            f"confidence={r['confidence']} out of [0,1]"
        )


def test_same_orb_params():
    """ORB_PARAMS must be the single source of truth for both build and query."""
    assert "nfeatures" in ORB_PARAMS
    assert "scaleFactor" in ORB_PARAMS
    assert "nlevels" in ORB_PARAMS

    import inspect
    import src.vns.database.reference_database as mod
    src_code = inspect.getsource(mod)

    build_uses = src_code.count("ORB_PARAMS")
    assert build_uses >= 2, (
        "ORB_PARAMS should appear at least twice (build + query)"
    )


def test_get_stats(tmp_db):
    db, img_path, _ = tmp_db
    db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)

    stats = db.get_stats()
    for key in ("total_patches", "unique_images", "lat_range",
                "lon_range", "last_built", "db_size_mb"):
        assert key in stats, f"Missing key: {key}"

    assert stats["total_patches"] > 0
    assert stats["unique_images"] >= 1


def test_clear(tmp_db):
    db, img_path, _ = tmp_db
    db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)
    assert db.get_stats()["total_patches"] > 0

    db.clear()
    assert db.get_stats()["total_patches"] == 0
