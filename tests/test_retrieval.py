"""Unit tests for the coarse retrieval stage (FLANN/LSH)."""

import numpy as np
import pytest

from vns.database.reference_db import DatabaseEntry
from vns.vision.retrieval import (
    BoVWRetrievalBackend,
    FlannLshBackend,
    RetrievalIndex,
)


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

    def test_from_config_uses_bovw_when_requested(self, sample_bovw_database):
        idx = RetrievalIndex.from_config(
            {"mode": "bovw", "bovw": {"enabled": True}},
            sample_bovw_database,
        )

        entry = sample_bovw_database.entries["synth_001"]
        results = idx.query(entry.descriptors, top_k=3)

        assert idx.backend_name == "bovw"
        assert results[0][0] == "synth_001"

    def test_from_config_falls_back_to_flann(self, sample_database):
        idx = RetrievalIndex.from_config(
            {
                "mode": "bovw",
                "bovw": {
                    "enabled": True,
                    "fallback_to_flann": True,
                },
            },
            sample_database,
        )

        entry = next(iter(sample_database.entries.values()))
        results = idx.query(entry.descriptors, top_k=1)

        assert idx.backend_name == "flann"
        assert results[0][0] == entry.id

    def test_from_config_raises_for_missing_bovw_index(self, sample_database):
        with pytest.raises(ValueError, match="BoVW retrieval selected"):
            RetrievalIndex.from_config(
                {"mode": "bovw", "bovw": {"enabled": True}},
                sample_database,
            )


class TestRetrievalGeoGate:
    """Geo-gating + region fallback layered on the BoVW backend (FR-9).

    ``sample_bovw_database`` holds three entries:
        synth_001 @ (33.7470, 73.1370)   - the prior anchor
        synth_002 @ (33.7475, 73.1375)   - ~0.000707 deg from the prior
        synth_003 @ (33.7480, 73.1380)   - ~0.001414 deg from the prior
    """

    PRIOR_AT_001 = (33.7470, 73.1370)
    FAR_PRIOR = (40.0, 70.0)

    def _bovw_index(self, database, **bovw_overrides):
        return RetrievalIndex.from_config(
            {"mode": "bovw", "bovw": {"enabled": True, **bovw_overrides}},
            database,
        )

    def test_geo_gate_filters_distant_candidates(self, sample_bovw_database):
        idx = self._bovw_index(
            sample_bovw_database, geo_gate=True, geo_gate_radius_deg=0.0001
        )
        entry = sample_bovw_database.entries["synth_001"]
        gated = idx.query(entry.descriptors, top_k=3, prior=self.PRIOR_AT_001)
        # Only synth_001 lies within 0.0001 deg of the prior.
        assert {r[0] for r in gated} == {"synth_001"}

    def test_gate_disabled_keeps_all_candidates(self, sample_bovw_database):
        idx = self._bovw_index(
            sample_bovw_database, geo_gate=False, geo_gate_radius_deg=0.0001
        )
        entry = sample_bovw_database.entries["synth_001"]
        with_prior = idx.query(entry.descriptors, top_k=3, prior=self.PRIOR_AT_001)
        no_prior = idx.query(entry.descriptors, top_k=3)
        assert with_prior == no_prior
        assert len(with_prior) > 1  # distant candidates retained

    def test_no_prior_is_appearance_only(self, sample_bovw_database):
        idx = self._bovw_index(
            sample_bovw_database, geo_gate=True, geo_gate_radius_deg=0.0001
        )
        entry = sample_bovw_database.entries["synth_001"]
        # Gate enabled, but without a prior retrieval stays purely appearance-based.
        assert len(idx.query(entry.descriptors, top_k=3, prior=None)) > 1

    def test_gate_never_returns_empty(self, sample_bovw_database, caplog):
        idx = self._bovw_index(
            sample_bovw_database, geo_gate=True, geo_gate_radius_deg=0.0001
        )
        entry = sample_bovw_database.entries["synth_001"]
        unfiltered = idx.query(entry.descriptors, top_k=3)
        with caplog.at_level("WARNING"):
            gated = idx.query(entry.descriptors, top_k=3, prior=self.FAR_PRIOR)
        # A stale/drifted prior must not wipe out every candidate.
        assert gated == unfiltered
        assert any("Geo gate discarded" in m for m in caplog.messages)

    def test_empty_descriptors_fall_back_to_region(self, sample_bovw_database):
        idx = self._bovw_index(
            sample_bovw_database, geo_gate=True, geo_gate_radius_deg=0.0001
        )
        results = idx.query(
            np.empty((0, 32), dtype=np.uint8), top_k=3, prior=self.PRIOR_AT_001
        )
        # Region fallback uses query_radius_deg (default 0.001): 001 + 002 within.
        assert {r[0] for r in results} == {"synth_001", "synth_002"}

    def test_empty_descriptors_without_prior_returns_empty(self, sample_bovw_database):
        idx = self._bovw_index(sample_bovw_database, geo_gate=True)
        results = idx.query(np.empty((0, 32), dtype=np.uint8), top_k=3, prior=None)
        assert results == []

    def test_flann_backend_ignores_prior(self, sample_bovw_database):
        idx = RetrievalIndex.from_config({"mode": "flann"}, sample_bovw_database)
        entry = sample_bovw_database.entries["synth_001"]
        with_prior = idx.query(entry.descriptors, top_k=3, prior=self.PRIOR_AT_001)
        no_prior = idx.query(entry.descriptors, top_k=3)
        assert idx.backend_name == "flann"
        assert with_prior == no_prior


class TestBoVWRetrievalBackend:

    def test_build_and_query(self, sample_bovw_database):
        backend = BoVWRetrievalBackend.from_database(sample_bovw_database)
        backend.build_index(list(sample_bovw_database.entries.values()))

        entry = sample_bovw_database.entries["synth_002"]
        results = backend.query(entry.descriptors, top_k=3)

        assert results[0][0] == "synth_002"

    def test_missing_bovw_data_raises(self, sample_database):
        with pytest.raises(ValueError, match="no BoVW vocabulary"):
            BoVWRetrievalBackend.from_database(sample_database)
