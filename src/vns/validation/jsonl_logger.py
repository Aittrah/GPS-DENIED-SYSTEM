from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from vns.core.runtime import FrameProcessingResult

logger = logging.getLogger("vns.validation")


@dataclass(frozen=True)
class GroundTruthSample:
    """Ground-truth pose sample converted into the runtime geodetic frame."""

    latitude: float
    longitude: float
    altitude: float
    heading_deg: float
    timestamp: float


def _coerce_pose(
    pose: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    if pose is None:
        return None
    return float(pose[0]), float(pose[1]), float(pose[2])


def _heading_error_deg(estimated_deg: float, true_deg: float) -> float:
    delta = abs(estimated_deg - true_deg) % 360.0
    return min(delta, 360.0 - delta)


def _horizontal_error_m(
    estimated_pose: tuple[float, float, float],
    truth: GroundTruthSample,
) -> float:
    dlat = estimated_pose[0] - truth.latitude
    dlon = estimated_pose[1] - truth.longitude
    return 111000.0 * math.sqrt(dlat ** 2 + dlon ** 2)


def build_evaluation_record(
    *,
    timestamp: float | None,
    ros_time: float | None,
    result: FrameProcessingResult,
    ground_truth: GroundTruthSample | None,
    gnss_status: str,
    record_without_ground_truth: bool,
) -> dict[str, Any] | None:
    """Build one JSONL evaluation record from a runtime frame result."""
    if ground_truth is None and not record_without_ground_truth:
        return None

    event_time = (
        result.localization.timestamp if timestamp is None else float(timestamp)
    )
    visual_pose = _coerce_pose(result.localization.pose_geodetic)
    estimated_pose = _coerce_pose(result.blended_pose)
    estimated_heading = (
        float(math.degrees(result.localization.yaw_rad))
        if result.localization.success
        else None
    )

    horizontal_error = None
    vertical_error = None
    heading_error = None
    position_error_m = None
    true_lat = None
    true_lon = None
    true_alt = None
    true_heading = None
    gt_timestamp = None

    if ground_truth is not None:
        true_lat = float(ground_truth.latitude)
        true_lon = float(ground_truth.longitude)
        true_alt = float(ground_truth.altitude)
        true_heading = float(ground_truth.heading_deg)
        gt_timestamp = float(ground_truth.timestamp)

        if estimated_pose is not None:
            horizontal_error = _horizontal_error_m(estimated_pose, ground_truth)
            vertical_error = abs(estimated_pose[2] - ground_truth.altitude)
            if estimated_heading is not None:
                heading_error = _heading_error_deg(
                    estimated_heading,
                    ground_truth.heading_deg,
                )
            position_error_m = math.sqrt(
                horizontal_error ** 2 + vertical_error ** 2
            )

    return {
        "timestamp": event_time,
        "ros_time": None if ros_time is None else float(ros_time),
        "ground_truth_timestamp": gt_timestamp,
        "ground_truth_available": ground_truth is not None,
        "estimated_lat": None if estimated_pose is None else estimated_pose[0],
        "estimated_lon": None if estimated_pose is None else estimated_pose[1],
        "estimated_alt": None if estimated_pose is None else estimated_pose[2],
        "visual_lat": None if visual_pose is None else visual_pose[0],
        "visual_lon": None if visual_pose is None else visual_pose[1],
        "visual_alt": None if visual_pose is None else visual_pose[2],
        "true_lat": true_lat,
        "true_lon": true_lon,
        "true_alt": true_alt,
        "estimated_heading": estimated_heading,
        "true_heading": true_heading,
        "horizontal_error": horizontal_error,
        "vertical_error": vertical_error,
        "heading_error": heading_error,
        "position_error_m": position_error_m,
        "vision_confidence": float(result.localization.confidence),
        "localization_success": bool(result.localization.success),
        "gnss_status": gnss_status,
        "navigation_mode": result.navigation_mode,
        "mode": result.navigation_mode,
        "failure_reason": None if result.success else result.reason,
        "localization_reason": result.reason,
        "matched_ref_id": result.localization.matched_ref_id,
        "inlier_count": int(result.localization.inlier_count),
    }


class JsonlEvaluationLogger:
    """Best-effort JSONL writer for evaluation records."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: str | Path,
        filename_prefix: str = "ground_truth",
    ) -> None:
        self._enabled = enabled
        self._handle: TextIO | None = None
        self._log_path: Path | None = None

        if enabled:
            self._open(log_dir=log_dir, filename_prefix=filename_prefix)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._handle is not None

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def _open(self, *, log_dir: str | Path, filename_prefix: str) -> None:
        try:
            path = Path(log_dir).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            self._log_path = path / f"{filename_prefix}_{int(time.time())}.jsonl"
            self._handle = self._log_path.open("a", encoding="utf-8")
            logger.info(
                "Accuracy evaluation logs will be written to: %s",
                self._log_path,
            )
        except OSError as exc:
            self._enabled = False
            self._handle = None
            self._log_path = None
            logger.warning(
                "Failed to initialize evaluation logger at %s: %s",
                log_dir,
                exc,
            )

    def write_frame(
        self,
        *,
        timestamp: float | None,
        ros_time: float | None,
        result: FrameProcessingResult,
        ground_truth: GroundTruthSample | None,
        gnss_status: str,
        record_without_ground_truth: bool,
    ) -> bool:
        if not self.enabled or self._handle is None:
            return False

        record = build_evaluation_record(
            timestamp=timestamp,
            ros_time=ros_time,
            result=result,
            ground_truth=ground_truth,
            gnss_status=gnss_status,
            record_without_ground_truth=record_without_ground_truth,
        )
        if record is None:
            return False

        try:
            self._handle.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )
            self._handle.flush()
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to write evaluation log record: %s", exc)
            self.close()
            self._enabled = False
            return False

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        finally:
            self._handle = None
