import cv2
import numpy as np

from vns.validation.frame_quality import (
    DEFAULT_THRESHOLDS,
    assess_frame_quality,
)


def make_solid(color_bgr, h=480, w=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def make_textured(h=480, w=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[0:h // 2, 0:w // 2] = (120, 80, 40)
    img[0:h // 2, w // 2:w] = (60, 110, 200)
    img[h // 2:h, 0:w // 2] = (90, 90, 90)
    img[h // 2:h, w // 2:w] = (200, 200, 100)
    for i in range(0, max(h, w), 24):
        cv2.line(img, (i, 0), (i, h), (255, 255, 255), 1)
        cv2.line(img, (0, i), (w, i), (255, 255, 255), 1)
    return img


class TestAssessFrameQuality:
    def test_solid_green_is_rejected(self):
        green = make_solid((50, 150, 50))  # BGR: high G, the Gazebo-default-ground-ish color
        report = assess_frame_quality(green)
        assert not report.passed
        assert report.green_ratio > DEFAULT_THRESHOLDS.max_green_ratio
        assert any("green_ratio" in reason for reason in report.reasons)

    def test_solid_grey_is_rejected(self):
        grey = make_solid((100, 102, 99))
        report = assess_frame_quality(grey)
        assert not report.passed
        assert report.grayscale_std == 0.0
        assert report.orb_keypoint_count == 0

    def test_uniform_dark_frame_is_rejected_for_brightness(self):
        dark = make_solid((2, 2, 2))
        report = assess_frame_quality(dark)
        assert not report.passed
        assert any("brightness_mean" in reason for reason in report.reasons)

    def test_richly_textured_frame_is_accepted(self):
        textured = make_textured()
        report = assess_frame_quality(textured)
        assert report.passed
        assert report.reasons == ()
        assert report.orb_keypoint_count >= DEFAULT_THRESHOLDS.min_orb_keypoints
        assert report.grayscale_std >= DEFAULT_THRESHOLDS.min_grayscale_std

    def test_grayscale_input_is_accepted(self):
        textured = cv2.cvtColor(make_textured(), cv2.COLOR_BGR2GRAY)
        report = assess_frame_quality(textured)
        assert report.passed

    def test_report_str_includes_verdict(self):
        report = assess_frame_quality(make_solid((50, 150, 50)))
        assert "REJECT" in str(report)
        report = assess_frame_quality(make_textured())
        assert "PASS" in str(report)

    def test_as_dict_round_trips_fields(self):
        report = assess_frame_quality(make_textured())
        data = report.as_dict()
        assert data["passed"] is True
        assert isinstance(data["reasons"], list)
        assert data["orb_keypoint_count"] == report.orb_keypoint_count
