"""Pose recovery: homography + reference geo-tag + altitude -> metric NED pose."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from vns.database.reference_db import DatabaseEntry
from vns.utils.coordinates import enu_to_geodetic, geodetic_to_enu
from vns.vision.types import VerificationResult

logger = logging.getLogger("vns.vision.pose_recovery")

# Return type shared by the helper and PoseRecovery.recover:
#   ((lat, lon, alt), (north_m, east_m, down_m), yaw_rad)
Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float], float]


def displacement_to_pose(
    inliers_query: np.ndarray,
    inliers_ref: np.ndarray,
    entry: DatabaseEntry,
    camera_config: Dict,
    altitude: float,
    heading_deg: float,
    geo_origin: Dict[str, float],
) -> Pose:
    """Convert median inlier pixel displacement into a metric pose.

    This is the single source of truth for the geometry: the median pixel
    offset between matched query/reference inliers is scaled by the
    altitude-above-terrain and focal length, rotated by heading, anchored to
    the reference entry's geo-tag, and converted back to geodetic + NED.

    Both :class:`PoseRecovery` (current pipeline) and ``FeatureMatcher`` (legacy
    path) call this so their geometry can never diverge.

    Returns ``((lat, lon, alt), (north_m, east_m, down_m), yaw_rad)``.
    """
    fx = camera_config.get("fx", 554.25)
    fy = camera_config.get("fy", 554.25)
    f = (fx + fy) / 2.0

    du = float(np.median(inliers_query[:, 0] - inliers_ref[:, 0]))
    dv = float(np.median(inliers_query[:, 1] - inliers_ref[:, 1]))

    height = max(1.0, altitude - entry.altitude)

    dx_body = -du * (height / f)
    dy_body = -dv * (height / f)

    yaw_rad = float(np.radians(heading_deg))
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    de = float(dx_body * cos_y + dy_body * sin_y)
    dn = float(-dx_body * sin_y + dy_body * cos_y)

    origin_lat = geo_origin["origin_latitude"]
    origin_lon = geo_origin["origin_longitude"]
    origin_alt = geo_origin["origin_altitude"]

    ref_e, ref_n, _ = geodetic_to_enu(
        entry.latitude, entry.longitude, entry.altitude,
        origin_lat, origin_lon, origin_alt,
    )

    est_e = ref_e + de
    est_n = ref_n + dn
    est_u = altitude - origin_alt

    est_lat, est_lon, _ = enu_to_geodetic(
        est_e, est_n, est_u,
        origin_lat, origin_lon, origin_alt,
    )

    return (est_lat, est_lon, altitude), (est_n, est_e, -est_u), yaw_rad


class PoseRecovery:
    """Convert a geometric-verification result into a metric pose.

    Uses the median inlier pixel displacement, the camera focal length,
    and the drone altitude to recover position in both geodetic and
    local-NED frames.
    """

    def recover(
        self,
        verification: VerificationResult,
        entry: DatabaseEntry,
        camera_config: Dict,
        altitude: float,
        heading_deg: float,
        geo_origin: Dict[str, float],
    ) -> Optional[Pose]:
        """Recover the drone's metric pose.

        Returns:
            ``((lat, lon, alt), (north_m, east_m, down_m), yaw_rad)`` on
            success, or ``None`` if the geometry is degenerate.
        """
        if len(verification.inliers_query) == 0:
            return None

        return displacement_to_pose(
            verification.inliers_query,
            verification.inliers_ref,
            entry,
            camera_config,
            altitude,
            heading_deg,
            geo_origin,
        )
