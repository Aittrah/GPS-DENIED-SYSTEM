#!/usr/bin/env python3
"""
Tests for the BoVW coarse-retrieval index.

Covers:
  * the shared offline/online histogram routine (normalization, determinism,
    empty-input handling),
  * K-clamping when the database has fewer descriptors than the requested K,
  * an end-to-end build (synthetic DB -> .vnsdb -> BoVWIndex) where querying an
    image with its OWN descriptors must return that image as the #1 candidate.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "simulation" / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from vns.database.reference_db import ReferenceDatabase  # noqa: E402
from vns.vision.bovw_retrieval import BoVWIndex, compute_bovw_histogram  # noqa: E402
import build_reference_database as brd  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared histogram routine
# --------------------------------------------------------------------------- #

def test_histogram_normalized_and_deterministic():
    rng = np.random.default_rng(1)
    vocab = rng.random((20, 32)).astype(np.float32)
    desc = rng.integers(0, 256, size=(50, 32), dtype=np.uint8)

    h1 = compute_bovw_histogram(desc, vocab, 20)
    h2 = compute_bovw_histogram(desc, vocab, 20)

    assert h1.shape == (20,)
    assert h1.dtype == np.float32
    assert np.isclose(np.linalg.norm(h1), 1.0)
    assert np.array_equal(h1, h2)  # same inputs -> identical histogram


def test_empty_descriptors_returns_zeros():
    vocab = np.random.default_rng(0).random((10, 32)).astype(np.float32)
    h = compute_bovw_histogram(np.empty((0, 32), dtype=np.uint8), vocab, 10)
    assert h.shape == (10,)
    assert not h.any()


# --------------------------------------------------------------------------- #
# K-clamping
# --------------------------------------------------------------------------- #

def test_k_clamped_when_descriptors_fewer_than_k():
    """Requesting K >> available descriptors must clamp K to the descriptor count."""
    db = brd.ReferenceDatabase(
        version="1.1.0",
        name="tiny",
        created="now",
        algorithm="ORB",
        bounds=brd.GeoBounds(0.0, 0.0, 0.0, 0.0),
    )
    rng = np.random.default_rng(0)
    desc = rng.integers(0, 256, size=(30, 32), dtype=np.uint8)
    db.add_entry(brd.DatabaseEntry(
        id="only",
        source_path="",
        latitude=0.0, longitude=0.0, altitude=0.0, heading=0.0,
        capture_time="", feature_count=30, feature_algorithm="ORB",
        keypoints=np.zeros((30, 2)), descriptors=desc,
    ))

    stats = brd.build_bovw_vocabulary(db, vocab_size=1000)

    assert stats["k"] == 30                     # clamped to total descriptors
    assert db.vocabulary.shape[0] == 30
    assert db.entries["only"].bovw_histogram.shape == (30,)


# --------------------------------------------------------------------------- #
# End-to-end build + retrieval
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def synthetic_db(tmp_path_factory):
    """Build a synthetic .vnsdb from the ~23 QAU reference points."""
    tmp = tmp_path_factory.mktemp("bovw")
    images_dir = tmp / "images"

    subprocess.run(
        [sys.executable, str(SCRIPTS / "generate_synthetic_reference_images.py"),
         "--output", str(images_dir)],
        check=True, cwd=str(SCRIPTS),
    )

    db_path = tmp / "test.vnsdb"
    subprocess.run(
        [sys.executable, str(SCRIPTS / "build_reference_database.py"),
         "--input", str(images_dir / "database_index.yaml"),
         "--output", str(db_path),
         "--build-vocab", "--vocab-size", "100"],
        check=True, cwd=str(SCRIPTS),
    )
    return db_path


def test_query_own_descriptors_returns_self(synthetic_db):
    index = BoVWIndex.load(str(synthetic_db))
    db = ReferenceDatabase.load(str(synthetic_db))
    assert db.vocabulary is not None

    checked = 0
    for ref_id, entry in db.entries.items():
        desc = entry.descriptors
        if desc is None or len(desc) == 0:
            continue
        results = index.query(desc, k=5)
        assert results[0][0] == ref_id, (
            f"{ref_id} was not the #1 candidate; got {results[:3]}"
        )
        checked += 1

    assert checked >= 10  # sanity: we actually exercised most reference images


def test_vectorized_query_matches_sklearn(synthetic_db):
    """The fast vectorized query() must agree with the sklearn NN index."""
    index = BoVWIndex.load(str(synthetic_db))
    db = ReferenceDatabase.load(str(synthetic_db))

    for entry in db.entries.values():
        desc = entry.descriptors
        if desc is None or len(desc) == 0:
            continue
        fast = index.query(desc, k=5)
        ref = index.query_sklearn(desc, k=5)
        assert [r[0] for r in fast] == [r[0] for r in ref]
        for (_, df), (_, dr) in zip(fast, ref):
            assert df == pytest.approx(dr, abs=1e-5)


def test_loading_safe_db_without_bovw_raises(tmp_path):
    db = ReferenceDatabase(name="no-bovw")
    db.save(str(tmp_path / "safe_no_bovw.vnsdb"))

    with pytest.raises(ValueError, match="no BoVW index"):
        BoVWIndex.load(str(tmp_path / "safe_no_bovw.vnsdb"))
