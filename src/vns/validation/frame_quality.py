from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("vns.validation")


@dataclass(frozen=True)
class FrameQualityThresholds:
    """Pass/fail thresholds for :func:`assess_frame_quality`.

    Defaults are tuned for a Gazebo downward-camera frame over a textured
    ground plane: a blank/green/low-feature frame (untextured ground, a
    rendering failure, or a frame captured outside the textured footprint)
    should fail at least one of these.
    """

    min_grayscale_std: float = 8.0
    min_orb_keypoints: int = 25
    max_green_ratio: float = 0.6
    min_brightness_mean: float = 10.0
    max_brightness_mean: float = 245.0


DEFAULT_THRESHOLDS = FrameQualityThresholds()

# Same nfeatures/scaleFactor/nlevels convention as
# src/vns/perception/feature_extractor.py, src/vns/vision/extractor.py, and
# simulation/scripts/build_reference_database.py.
_ORB = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

# Green hue band in OpenCV's HSV (H in [0, 179]), with a minimum saturation so
# desaturated grey/khaki ground (real terrain) is not mistaken for a "green"
# untextured Gazebo default-ground-plane color.
_GREEN_HSV_LOW = (35, 60, 40)
_GREEN_HSV_HIGH = (85, 255, 255)


@dataclass(frozen=True)
class FrameQualityReport:
    grayscale_std: float
    orb_keypoint_count: int
    green_ratio: float
    brightness_mean: float
    brightness_std: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "grayscale_std": self.grayscale_std,
            "orb_keypoint_count": self.orb_keypoint_count,
            "green_ratio": self.green_ratio,
            "brightness_mean": self.brightness_mean,
            "brightness_std": self.brightness_std,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "REJECT"
        summary = (
            f"{verdict} std={self.grayscale_std:.1f} "
            f"orb_kp={self.orb_keypoint_count} "
            f"green_ratio={self.green_ratio:.2f} "
            f"brightness={self.brightness_mean:.1f}+/-{self.brightness_std:.1f}"
        )
        if self.reasons:
            summary += " (" + "; ".join(self.reasons) + ")"
        return summary


def assess_frame_quality(
    image: np.ndarray,
    thresholds: FrameQualityThresholds = DEFAULT_THRESHOLDS,
) -> FrameQualityReport:
    """Assess whether ``image`` (BGR or grayscale) looks like a real, textured
    downward-camera frame rather than blank/plain/low-feature ground.

    Does not raise on a bad frame -- returns a report with ``passed=False``
    and human-readable ``reasons`` so callers decide what to do (reject before
    writing, log, etc.).
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        bgr = arr
    elif arr.ndim == 2:
        gray = arr
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Expected a grayscale or BGR image, got shape {arr.shape}")

    grayscale_std = float(gray.std())
    brightness_mean = float(gray.mean())
    brightness_std = grayscale_std

    keypoints, _ = _ORB.detectAndCompute(gray, None)
    orb_keypoint_count = 0 if keypoints is None else len(keypoints)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, _GREEN_HSV_LOW, _GREEN_HSV_HIGH)
    green_ratio = float(green_mask.mean()) / 255.0

    reasons: list[str] = []
    if grayscale_std < thresholds.min_grayscale_std:
        reasons.append(
            f"grayscale_std {grayscale_std:.1f} < {thresholds.min_grayscale_std} "
            "(near-uniform color, likely untextured ground)"
        )
    if orb_keypoint_count < thresholds.min_orb_keypoints:
        reasons.append(
            f"orb_keypoint_count {orb_keypoint_count} < {thresholds.min_orb_keypoints} "
            "(too few features to match against a reference database)"
        )
    if green_ratio > thresholds.max_green_ratio:
        reasons.append(
            f"green_ratio {green_ratio:.2f} > {thresholds.max_green_ratio} "
            "(likely Gazebo default/fallback green ground plane)"
        )
    if brightness_mean < thresholds.min_brightness_mean:
        reasons.append(
            f"brightness_mean {brightness_mean:.1f} < {thresholds.min_brightness_mean} "
            "(frame too dark, e.g. rendering not ready)"
        )
    if brightness_mean > thresholds.max_brightness_mean:
        reasons.append(
            f"brightness_mean {brightness_mean:.1f} > {thresholds.max_brightness_mean} "
            "(frame blown out / overexposed)"
        )

    report = FrameQualityReport(
        grayscale_std=grayscale_std,
        orb_keypoint_count=orb_keypoint_count,
        green_ratio=green_ratio,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        passed=not reasons,
        reasons=tuple(reasons),
    )
    if not report.passed:
        logger.debug("Frame quality check failed: %s", report)
    return report
