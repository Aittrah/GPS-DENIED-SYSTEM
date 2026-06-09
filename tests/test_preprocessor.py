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
