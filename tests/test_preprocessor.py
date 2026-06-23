"""Unit tests for the Preprocessor stage."""

import cv2
import numpy as np
import pytest

from vns.vision.preprocessor import Preprocessor


class TestPreprocessor:

    def test_resize(self):
        proc = Preprocessor(target_width=320, target_height=240)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = proc.process(frame)
        assert out.shape == (240, 320)

    def test_no_resize_when_already_target(self):
        proc = Preprocessor(target_width=640, target_height=480)
        frame = np.zeros((480, 640), dtype=np.uint8)
        out = proc.process(frame)
        assert out.shape == (480, 640)

    def test_bgr_to_grayscale(self):
        proc = Preprocessor()
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[:, :, 2] = 200  # red channel
        out = proc.process(bgr)
        assert len(out.shape) == 2

    def test_grayscale_passthrough(self):
        proc = Preprocessor()
        gray = np.full((480, 640), 100, dtype=np.uint8)
        out = proc.process(gray)
        assert len(out.shape) == 2
        assert out.shape == (480, 640)

    def test_clahe_improves_contrast(self):
        proc = Preprocessor(clahe_clip_limit=3.0, clahe_grid_size=8)
        low_contrast = np.full((480, 640), 120, dtype=np.uint8)
        low_contrast[100:200, 100:200] = 130
        out = proc.process(low_contrast)
        assert out.std() >= low_contrast.std()

    def test_output_dtype(self):
        proc = Preprocessor()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        out = proc.process(frame)
        assert out.dtype == np.uint8

    def test_undistort_scales_camera_matrix(self, monkeypatch):
        captured = {}

        def fake_undistort(frame, camera_matrix, distortion):
            captured["camera_matrix"] = camera_matrix.copy()
            captured["distortion"] = distortion.copy()
            return frame

        monkeypatch.setattr(cv2, "undistort", fake_undistort)
        proc = Preprocessor(
            target_width=1280,
            target_height=960,
            camera_matrix=np.array(
                [[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            distortion_coefficients=np.array(
                [0.1, -0.05, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            calibration_size=(640, 480),
            undistort=True,
        )

        out = proc.process(np.zeros((960, 1280, 3), dtype=np.uint8))

        assert out.shape == (960, 1280)
        assert captured["camera_matrix"][0, 0] == pytest.approx(200.0)
        assert captured["camera_matrix"][1, 1] == pytest.approx(240.0)
        assert captured["camera_matrix"][0, 2] == pytest.approx(100.0)
        assert captured["camera_matrix"][1, 2] == pytest.approx(80.0)

    def test_zero_distortion_skips_undistort(self, monkeypatch):
        called = False

        def fake_undistort(frame, camera_matrix, distortion):
            nonlocal called
            called = True
            return frame

        monkeypatch.setattr(cv2, "undistort", fake_undistort)
        proc = Preprocessor(
            camera_matrix=np.eye(3, dtype=np.float32),
            distortion_coefficients=np.zeros(5, dtype=np.float32),
            undistort=True,
        )

        proc.process(np.zeros((480, 640, 3), dtype=np.uint8))

        assert called is False
