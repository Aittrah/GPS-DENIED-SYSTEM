"""Unit tests for the coarse retrieval stage (FLANN/LSH)."""

import numpy as np
import pytest

from vns.database.reference_db import DatabaseEntry
from vns.vision.retrieval import FlannLshBackend, RetrievalIndex


def _make_entry(entry_id: str, n_desc: int = 50, seed: int = 0) -> DatabaseEntry:
    rng = np.random.RandomState(seed)
    return DatabaseEntry(
        id=entry_id,
        source_path="synthetic",
        latitude=33.747,
        longitude=73.137,
        altitude=550.0,
        heading=0.0,
        capture_time="",
        feature_count=n_desc,
        feature_algorithm="ORB",
        keypoints=rng.rand(n_desc, 2).astype(np.float32) * 500,
        descriptors=rng.randint(0, 256, (n_desc, 32), dtype=np.uint8),
        metadata={},
    )


class TestFlannLshBackend:

    def test_build_and_query(self):
        entries = [
            _make_entry("A", seed=0),
            _make_entry("B", seed=100),
            _make_entry("C", seed=200),
        ]
        backend = FlannLshBackend()
        backend.build_index(entries)

        results = backend.query(entries[0].descriptors, top_k=3)
        assert len(results) > 0
        top_id = results[0][0]
        assert top_id == "A"

    def test_empty_index(self):
        backend = FlannLshBackend()
        backend.build_index([])
        assert backend.query(np.zeros((10, 32), dtype=np.uint8), 5) == []

    def test_empty_query(self):
        entries = [_make_entry("A")]
        backend = FlannLshBackend()
        backend.build_index(entries)
        assert backend.query(np.empty((0, 32), dtype=np.uint8), 5) == []
        assert backend.query(None, 5) == []

    def test_top_k_limits_results(self):
        entries = [_make_entry(f"E{i}", seed=i * 50) for i in range(10)]
        backend = FlannLshBackend()
        backend.build_index(entries)
        results = backend.query(entries[0].descriptors, top_k=3)
        assert len(results) <= 3


class TestRetrievalIndex:

    def test_default_backend(self):
        idx = RetrievalIndex()
        entries = [_make_entry("A"), _make_entry("B", seed=99)]
        idx.build_index(entries)
        results = idx.query(entries[0].descriptors, top_k=2)
        assert len(results) > 0

    def test_custom_backend(self):
        backend = FlannLshBackend(table_number=4, key_size=10)
        idx = RetrievalIndex(backend=backend)
        entries = [_make_entry("X", seed=42)]
        idx.build_index(entries)
        results = idx.query(entries[0].descriptors, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "X"
