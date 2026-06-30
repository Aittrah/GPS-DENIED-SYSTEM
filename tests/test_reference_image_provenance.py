from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "simulation" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_reference_database as brd  # noqa: E402
from vns.database.reference_db import ReferenceDatabase  # noqa: E402


def _write_test_image(path: Path) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (590, 430), (255, 255, 255), 3)
    cv2.line(image, (0, 0), (639, 479), (0, 255, 0), 2)
    cv2.circle(image, (320, 240), 60, (0, 0, 255), -1)
    assert cv2.imwrite(str(path), image)


def test_builder_preserves_source_provenance(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "ref.jpg"
    _write_test_image(image_path)

    index_path = image_dir / "database_index.yaml"
    index_document = {
        "database": {
            "name": "Provenance Test",
            "version": "1.0.0",
            "location": "Test",
            "source_type": "synthetic",
            "center": {
                "latitude": 33.7470,
                "longitude": 73.1370,
                "altitude": 550.0,
            },
            "bounds": {
                "min_latitude": 33.7460,
                "max_latitude": 33.7480,
                "min_longitude": 73.1360,
                "max_longitude": 73.1380,
            },
        },
        "images": [
            {
                "id": "ref",
                "filepath": "ref.jpg",
                "latitude": 33.7470,
                "longitude": 73.1370,
                "altitude": 550.0,
                "heading": 0.0,
                "timestamp": "2026-06-30T00:00:00+00:00",
                "width": 640,
                "height": 480,
                "generator": "generate_synthetic_reference_images.py",
            }
        ],
    }
    index_path.write_text(yaml.safe_dump(index_document, sort_keys=False), encoding="utf-8")

    db = brd.build_database(str(index_path))
    output_path = tmp_path / "provenance.vnsdb"
    brd.save_database(db, str(output_path))
    loaded = ReferenceDatabase.load(str(output_path))
    metadata = loaded.entries["ref"].metadata

    assert metadata["source_type"] == "synthetic"
    assert metadata["generator"] == "generate_synthetic_reference_images.py"


def test_committed_synthetic_index_is_labeled() -> None:
    index_path = REPO_ROOT / "simulation" / "database" / "images" / "database_index.yaml"
    index_document = yaml.safe_load(index_path.read_text(encoding="utf-8"))

    assert index_document["database"]["source_type"] == "synthetic"


def test_committed_synthetic_database_entries_are_labeled() -> None:
    db_path = REPO_ROOT / "simulation" / "database" / "qau_campus.vnsdb"
    db = ReferenceDatabase.load(str(db_path))

    assert db.entries
    assert all(
        entry.metadata.get("source_type") == "synthetic"
        for entry in db.entries.values()
    )
