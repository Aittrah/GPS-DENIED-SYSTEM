"""Unit tests for FAISSDatabase (FR-9). DINOv2 is mocked to avoid 300MB download."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.vns.database import faiss_database as faiss_database_module

# ── helpers ──────────────────────────────────────────────────────────────────

DESCRIPTOR_DIM = 384
FAISS_AVAILABLE = getattr(faiss_database_module, "_FAISS_AVAILABLE", False)


def _random_descriptors(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    d = rng.random((n, DESCRIPTOR_DIM)).astype(np.float32)
    # L2-normalise so inner-product == cosine similarity
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    return d / norms


def _synthetic_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 0), -1)
    cv2.circle(img, (256, 256), 80, (255, 0, 255), 4)
    return img


def _save_image(img: np.ndarray, path: Path) -> str:
    cv2.imwrite(str(path), img)
    return str(path)


def _make_mock_extractor(n_patches: int = 9) -> MagicMock:
    mock = MagicMock()
    mock.available = True
    mock.model_name = "dinov2_vits14"
    mock.extract.return_value = _random_descriptors(1)[0]
    mock.extract_batch.return_value = _random_descriptors(n_patches)
    return mock


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_with_image(tmp_path):
    """FAISSDatabase with mocked DINOv2 and a saved satellite image."""
    img = _synthetic_image(0)
    img_path = _save_image(img, tmp_path / "sat.png")

    with patch("src.vns.database.faiss_database.DINOv2Extractor",
               return_value=_make_mock_extractor(9)):
        from src.vns.database.faiss_database import FAISSDatabase
        db = FAISSDatabase(
            index_path=str(tmp_path / "faiss.index"),
            meta_path=str(tmp_path / "metadata.pkl"),
        )
        db.extractor = _make_mock_extractor(9)
        yield db, img_path, tmp_path


# ── tests ─────────────────────────────────────────────────────────────────────

class TestFAISSDatabase:

    @pytest.mark.skipif(
        not FAISS_AVAILABLE,
        reason="faiss-cpu is not installed; FAISS index tests require the optional dependency.",
    )
    def test_build_creates_index(self, db_with_image):
        db, img_path, _ = db_with_image
        count = db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)
        assert count > 0
        assert db.index.ntotal > 0

    @pytest.mark.skipif(
        not FAISS_AVAILABLE,
        reason="faiss-cpu is not installed; FAISS persistence tests require the optional dependency.",
    )
    def test_save_and_load(self, db_with_image):
        db, img_path, tmp_path = db_with_image
        db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)
        stats_before = db.get_stats()

        with patch("src.vns.database.faiss_database.DINOv2Extractor",
                   return_value=_make_mock_extractor(9)):
            from src.vns.database.faiss_database import FAISSDatabase
            db2 = FAISSDatabase(
                index_path=str(tmp_path / "faiss.index"),
                meta_path=str(tmp_path / "metadata.pkl"),
            )
            db2.extractor = _make_mock_extractor(9)
            ok = db2.load()

        assert ok
        assert db2.index.ntotal == stats_before["total_descriptors"]
        assert len(db2.metadata) == len(db.metadata)

    @pytest.mark.skipif(
        not FAISS_AVAILABLE,
        reason="faiss-cpu is not installed; FAISS query tests require the optional dependency.",
    )
    def test_query_returns_results(self, db_with_image):
        db, img_path, _ = db_with_image
        db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)

        uav_img = _synthetic_image(1)
        db.extractor.extract_batch.return_value = _random_descriptors(9)

        results = db.query(uav_img, top_k=5, min_confidence=0.0)
        assert isinstance(results, list)

    @pytest.mark.skipif(
        not FAISS_AVAILABLE,
        reason="faiss-cpu is not installed; FAISS query tests require the optional dependency.",
    )
    def test_confidence_range(self, db_with_image):
        db, img_path, _ = db_with_image
        db.build(img_path, lat=33.7470, lon=73.1370, alt=550.0)

        uav_img = _synthetic_image(2)
        db.extractor.extract_batch.return_value = _random_descriptors(9)
        results = db.query(uav_img, top_k=5, min_confidence=0.0)

        for r in results:
            assert 0.0 <= r["confidence"] <= 1.0, (
                f"confidence={r['confidence']} outside [0, 1]"
            )

    def test_descriptor_dim(self):
        from src.vns.database.faiss_database import FAISSDatabase
        assert FAISSDatabase.DESCRIPTOR_DIM == 384
