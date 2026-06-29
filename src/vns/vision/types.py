"""Shared data types for the visual localization pipeline."""

import time as _time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class VerificationResult:
    """Result of geometric verification for a single candidate."""

    entry_id: str
    inlier_count: int
    confidence: float
    homography: Optional[np.ndarray]
    inliers_query: np.ndarray
    inliers_ref: np.ndarray


@dataclass
class LocalizationResult:
    """Output of the visual localization pipeline."""

    success: bool
    confidence: float
    inlier_count: int
    matched_ref_id: Optional[str]
    reason: str
    pose_ned: Optional[Tuple[float, float, float]]
    pose_geodetic: Optional[Tuple[float, float, float]]
    yaw_rad: float
    timestamp: float = field(default_factory=_time.time)
