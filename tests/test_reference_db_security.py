"""Security-focused tests for reference database serialization."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from vns.database.reference_db import DatabaseEntry, ReferenceDatabase


def _make_entry(entry_id: str, seed: int = 0) -> DatabaseEntry:
    rng = np.random.default_rng(seed)
    keypoints = rng.random((12, 2)).astype(np.float32) * 100.0
    descriptors = rng.integers(0, 256, size=(12, 32), dtype=np.uint8)
    return DatabaseEntry(
        id=entry_id,
        source_path=f"{entry_id}.jpg",
        latitude=33.747 + seed * 1e-4,
        longitude=73.137 + seed * 1e-4,
        altitude=550.0 + seed,
        heading=0.0,
        capture_time="2024-01-01T00:00:00",
        feature_count=len(keypoints),
        feature_algorithm="ORB",
        keypoints=keypoints,
        descriptors=descriptors,
        metadata={"seed": seed},
    )


def _write_legacy_db(path: Path, with_bovw: bool = True) -> None:
    entry_a = _make_entry("a", seed=1)
    entry_b = _make_entry("b", seed=2)
    data = {
        "version": "1.1.0",
        "name": "legacy",
        "created": "2024-01-01T00:00:00",
        "algorithm": "ORB",
        "bounds": {
            "min_lat": entry_a.latitude,
            "max_lat": entry_b.latitude,
            "min_lon": entry_a.longitude,
            "max_lon": entry_b.longitude,
        },
        "entries": {
            "a": {
                "id": entry_a.id,
                "source_path": entry_a.source_path,
                "latitude": entry_a.latitude,
                "longitude": entry_a.longitude,
                "altitude": entry_a.altitude,
                "heading": entry_a.heading,
                "capture_time": entry_a.capture_time,
                "feature_count": entry_a.feature_count,
                "feature_algorithm": entry_a.feature_algorithm,
                "keypoints": entry_a.keypoints,
                "descriptors": entry_a.descriptors,
                "metadata": entry_a.metadata,
            },
            "b": {
                "id": entry_b.id,
                "source_path": entry_b.source_path,
                "latitude": entry_b.latitude,
                "longitude": entry_b.longitude,
                "altitude": entry_b.altitude,
                "heading": entry_b.heading,
                "capture_time": entry_b.capture_time,
                "feature_count": entry_b.feature_count,
                "feature_algorithm": entry_b.feature_algorithm,
                "keypoints": entry_b.keypoints,
                "descriptors": entry_b.descriptors,
                "metadata": entry_b.metadata,
            },
        },
    }
    if with_bovw:
        vocabulary = np.random.default_rng(5).random((8, 32)).astype(np.float32)
        data["bovw"] = {
            "metric": "cosine",
            "vocabulary": vocabulary,
            "ids": ["a", "b"],
            "histograms": np.vstack(
                [
                    np.linspace(0.0, 1.0, 8, dtype=np.float32),
                    np.linspace(1.0, 0.0, 8, dtype=np.float32),
                ]
            ),
        }
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def test_safe_roundtrip_with_bovw(tmp_path):
    db = ReferenceDatabase(name="safe")
    db.add_entry(_make_entry("a", seed=1))
    db.add_entry(_make_entry("b", seed=2))
    db.vocabulary = np.random.default_rng(10).random((8, 32)).astype(np.float32)
    db.bovw_metric = "cosine"
    db.entries["a"].bovw_histogram = np.ones(8, dtype=np.float32) / np.sqrt(8)
    db.entries["b"].bovw_histogram = np.arange(8, dtype=np.float32)
    db.entries["b"].bovw_histogram /= np.linalg.norm(db.entries["b"].bovw_histogram)

    output = tmp_path / "safe.vnsdb"
    db.save(str(output))

    loaded = ReferenceDatabase.load(str(output))
    assert ReferenceDatabase.detect_format(str(output)) == "safe_archive"
    assert loaded.vocabulary is not None
    assert loaded.bovw_metric == "cosine"
    assert loaded.entries["a"].bovw_histogram is not None
    assert loaded.entries["b"].bovw_histogram is not None


def test_legacy_pickle_is_blocked_by_default(tmp_path):
    legacy = tmp_path / "legacy.vnsdb"
    _write_legacy_db(legacy, with_bovw=False)

    assert ReferenceDatabase.detect_format(str(legacy)) == "legacy_pickle"
    with pytest.raises(ValueError, match="insecure legacy pickle format"):
        ReferenceDatabase.load(str(legacy))


def test_trusted_legacy_loader_and_migration(tmp_path):
    legacy = tmp_path / "legacy.vnsdb"
    migrated = tmp_path / "migrated.vnsdb"
    _write_legacy_db(legacy, with_bovw=True)

    legacy_db = ReferenceDatabase.load_legacy_trusted(str(legacy))
    assert legacy_db.vocabulary is not None
    assert legacy_db.entries["a"].bovw_histogram is not None

    ReferenceDatabase.migrate_legacy_file(str(legacy), str(migrated))
    migrated_db = ReferenceDatabase.load(str(migrated))
    assert migrated_db.vocabulary is not None
    assert migrated_db.entries["b"].bovw_histogram is not None


def test_malformed_safe_archive_is_rejected(tmp_path):
    bad_path = tmp_path / "bad.vnsdb"
    metadata = {
        "format": "vns.safe_database",
        "schema_version": 1,
        "version": "2.0.0",
        "name": "bad",
        "created": "2024-01-01T00:00:00",
        "algorithm": "ORB",
        "bounds": {
            "min_lat": 0.0,
            "max_lat": 0.0,
            "min_lon": 0.0,
            "max_lon": 0.0,
        },
        "entries": [
            {
                "id": "bad",
                "source_path": "bad.jpg",
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude": 0.0,
                "heading": 0.0,
                "capture_time": "",
                "feature_count": 1,
                "feature_algorithm": "ORB",
                "metadata": {},
                "keypoints_key": "k",
                "descriptors_key": "d",
                "bovw_histogram_key": None,
            }
        ],
    }

    with open(bad_path, "wb") as handle:
        np.savez_compressed(
            handle,
            __metadata__=np.array(json.dumps(metadata)),
            k=np.zeros((1, 2), dtype=np.float32),
            d=np.zeros((1, 31), dtype=np.uint8),
        )

    with pytest.raises(ValueError, match="descriptors must have shape"):
        ReferenceDatabase.load(str(bad_path))
