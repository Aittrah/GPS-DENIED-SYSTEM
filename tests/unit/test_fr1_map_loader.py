"""Unit tests for FR-1 — geo_validator and ReferenceDatabase."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vns.mpvi.geo_validator import validate_image
from vns.database import ReferenceDatabase, SatelliteImageRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> ReferenceDatabase:
    return ReferenceDatabase(db_dir=tmp_path / "database")


@pytest.fixture()
def synthetic_geotiff(tmp_path: Path) -> Path:
    """Write a minimal in-memory GeoTIFF to a temp file."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    path = tmp_path / "test_scene.tif"
    width, height = 64, 64
    transform = from_bounds(73.10, 33.70, 73.20, 33.80, width, height)
    crs = CRS.from_epsg(4326)
    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture()
def plain_jpeg(tmp_path: Path) -> Path:
    """Write a tiny JPEG (no geo-reference)."""
    PIL_Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "plain.jpg"
    img = PIL_Image.new("RGB", (32, 32), color=(128, 64, 32))
    img.save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# validate_image — GeoTIFF
# ---------------------------------------------------------------------------

def test_validate_geotiff_valid(synthetic_geotiff: Path) -> None:
    result = validate_image(synthetic_geotiff)
    assert result["valid"] is True
    assert result["format"] == "GeoTIFF"
    assert result["has_georef"] is True
    assert result["crs"] is not None
    assert result["bbox"] is not None
    west, south, east, north = result["bbox"]
    assert west < east
    assert south < north
    assert result["error"] is None


def test_validate_geotiff_missing_file() -> None:
    result = validate_image(Path("/nonexistent/path/scene.tif"))
    assert result["valid"] is False
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# validate_image — JPEG/PNG
# ---------------------------------------------------------------------------

def test_validate_jpg_no_georef(plain_jpeg: Path) -> None:
    result = validate_image(plain_jpeg)
    assert result["valid"] is True
    assert result["has_georef"] is False
    assert result["format"] == "JPEG"
    assert result["warning"] is not None
    assert result["bbox"] is None
    assert result["error"] is None


def test_validate_png_no_georef(tmp_path: Path) -> None:
    PIL_Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "plain.png"
    img = PIL_Image.new("RGB", (32, 32), color=(0, 128, 255))
    img.save(path, format="PNG")

    result = validate_image(path)
    assert result["valid"] is True
    assert result["has_georef"] is False
    assert result["format"] == "PNG"


# ---------------------------------------------------------------------------
# validate_image — unsupported formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".xyz", ".csv", ".bmp", ".raw", ""])
def test_validate_unsupported_format(tmp_path: Path, ext: str) -> None:
    path = tmp_path / f"file{ext}"
    path.write_bytes(b"\x00\x01\x02")
    result = validate_image(path)
    assert result["valid"] is False
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# ReferenceDatabase
# ---------------------------------------------------------------------------

def _make_record(path: str = "/data/img.tif") -> SatelliteImageRecord:
    return SatelliteImageRecord(
        path=path,
        filename=Path(path).name,
        file_size_bytes=1024,
        format="GeoTIFF",
        has_georef=True,
        bbox=(73.10, 33.70, 73.20, 33.80),
        crs="EPSG:4326",
        added_at=time.time(),
    )


def test_database_add_and_retrieve(tmp_db: ReferenceDatabase) -> None:
    assert len(tmp_db) == 0
    record = _make_record()
    tmp_db.add_image(record)
    records = tmp_db.get_all()
    assert len(records) == 1
    assert records[0].filename == "img.tif"
    assert records[0].has_georef is True


def test_database_persists_to_disk(tmp_path: Path) -> None:
    """Records survive a reload from the index.json file."""
    db_dir = tmp_path / "database"
    db1 = ReferenceDatabase(db_dir=db_dir)
    db1.add_image(_make_record("/data/scene_a.tif"))
    db1.add_image(_make_record("/data/scene_b.tif"))

    db2 = ReferenceDatabase(db_dir=db_dir)
    assert len(db2) == 2
    filenames = {r.filename for r in db2.get_all()}
    assert "scene_a.tif" in filenames
    assert "scene_b.tif" in filenames


def test_database_remove_image(tmp_db: ReferenceDatabase) -> None:
    tmp_db.add_image(_make_record("/data/remove_me.tif"))
    assert len(tmp_db) == 1
    removed = tmp_db.remove_image(Path("/data/remove_me.tif"))
    assert removed is True
    assert len(tmp_db) == 0


def test_database_corrupt_index_is_handled(tmp_path: Path) -> None:
    """A corrupt index.json should not raise — database starts empty."""
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    (db_dir / "index.json").write_text("NOT VALID JSON", encoding="utf-8")
    db = ReferenceDatabase(db_dir=db_dir)
    assert len(db) == 0
