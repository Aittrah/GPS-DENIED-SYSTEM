from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.vns.preprocessing.calibration import (
    CameraCalibration,
    load_camera_calibration,
)
from src.vns.preprocessing.uav_preprocessor import UAVPreprocessor


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(
            [[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        distortion_coefficients=np.array(
            [0.1, -0.05, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        calibration_size=(512, 512),
    )


def _frame(width: int = 1024, height: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_undistort_is_disabled_by_default(monkeypatch):
    called = False

    def fake_undistort(frame, camera_matrix, distortion):
        nonlocal called
        called = True
        return frame

    monkeypatch.setattr(cv2, "undistort", fake_undistort)
    proc = UAVPreprocessor(calibration=_calibration())

    result = proc.preprocess_frame(_frame())

    assert result is not None
    assert called is False


def test_undistort_scales_camera_matrix(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def fake_undistort(frame, camera_matrix, distortion):
        captured["camera_matrix"] = camera_matrix.copy()
        captured["distortion"] = distortion.copy()
        return frame

    monkeypatch.setattr(cv2, "undistort", fake_undistort)
    proc = UAVPreprocessor(calibration=_calibration(), undistort=True)

    result = proc.preprocess_frame(_frame())

    assert result is not None
    assert captured["camera_matrix"][0, 0] == pytest.approx(200.0)
    assert captured["camera_matrix"][1, 1] == pytest.approx(240.0)
    assert captured["camera_matrix"][0, 2] == pytest.approx(100.0)
    assert captured["camera_matrix"][1, 2] == pytest.approx(80.0)
    assert captured["distortion"][0] == pytest.approx(0.1)


def test_load_camera_calibration_from_yaml(tmp_path):
    calibration_path = tmp_path / "camera_calibration.yaml"
    calibration_path.write_text(
        "\n".join(
            [
                "camera_matrix:",
                "  - [554.25, 0.0, 320.0]",
                "  - [0.0, 554.25, 240.0]",
                "  - [0.0, 0.0, 1.0]",
                "distortion_coefficients: [0.1, -0.02, 0.0, 0.0, 0.0]",
                "image_size: [640, 480]",
            ]
        ),
        encoding="utf-8",
    )

    calibration = load_camera_calibration(calibration_path)

    assert calibration.calibration_size == (640, 480)
    assert calibration.camera_matrix[0, 0] == pytest.approx(554.25)
    assert calibration.distortion_coefficients[1] == pytest.approx(-0.02)


def test_faiss_database_passes_optional_query_calibration_to_preprocessor(tmp_path):
    calibration_path = tmp_path / "camera_calibration.yaml"
    calibration_path.write_text(
        "\n".join(
            [
                "camera_matrix:",
                "  - [554.25, 0.0, 320.0]",
                "  - [0.0, 554.25, 240.0]",
                "  - [0.0, 0.0, 1.0]",
                "distortion: [0.1, 0.0, 0.0, 0.0, 0.0]",
                "image_size: [640, 480]",
            ]
        ),
        encoding="utf-8",
    )

    with patch(
        "src.vns.database.faiss_database.DINOv2Extractor",
        return_value=MagicMock(available=True, model_name="mock"),
    ):
        from src.vns.database.faiss_database import FAISSDatabase

        db = FAISSDatabase(
            index_path=str(tmp_path / "faiss.index"),
            meta_path=str(tmp_path / "metadata.pkl"),
            query_calibration_path=str(calibration_path),
            undistort_queries=True,
        )

    captured: dict[str, object] = {}

    class StubPreprocessor:
        def __init__(self, *, calibration=None, undistort=False):
            captured["calibration"] = calibration
            captured["undistort"] = undistort

        def preprocess_frame(self, frame):
            return frame

    class StubPatchGenerator:
        def generate_patches(self, image, image_center, meters_per_pixel, source, base_id):
            return [
                SimpleNamespace(
                    data=image,
                    patch_id="uav_query_000",
                )
            ]

    db.index = SimpleNamespace(
        ntotal=1,
        search=lambda descriptors, k: (
            np.array([[1.0]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        ),
    )
    db.metadata = [
        {
            "patch_id": "ref_patch",
            "lat": 33.7470,
            "lon": 73.1370,
            "alt": 550.0,
            "source_image": "ref.png",
            "patch_array": _frame(256, 256),
        }
    ]
    db.extractor.extract_batch.return_value = np.ones((1, db.DESCRIPTOR_DIM), dtype=np.float32)
    db.verifier.verify = MagicMock(return_value={"confidence": 1.0, "verified": True})

    with patch("src.vns.database.faiss_database._FAISS_AVAILABLE", True), patch(
        "src.vns.database.faiss_database.faiss",
        SimpleNamespace(normalize_L2=lambda descriptors: None),
    ), patch("src.vns.database.faiss_database.UAVPreprocessor", StubPreprocessor), patch(
        "src.vns.database.faiss_database.PatchGenerator",
        return_value=StubPatchGenerator(),
    ):
        results = db.query(_frame(256, 256), top_k=1, min_confidence=0.0)

    assert results
    assert captured["undistort"] is True
    assert isinstance(captured["calibration"], CameraCalibration)
