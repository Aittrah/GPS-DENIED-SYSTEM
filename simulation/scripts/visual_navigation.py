#!/usr/bin/env python3
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, NavSatFix
    from geometry_msgs.msg import PoseStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False

import cv2
import numpy as np

# Add src to system path to import vns
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from vns.config.config_manager import ConfigManager
from vns.core.blender import PositionBlender
from vns.core.gnss_monitor import GnssMonitor, GnssState
from vns.database.reference_db import ReferenceDatabase
from vns.utils.coordinates import enu_to_geodetic, geodetic_to_enu
from vns.utils.logging import setup_logging
from vns.vision.bovw_retrieval import BoVWIndex
from vns.vision.extractor import FeatureExtractor
from vns.vision.matcher import FeatureMatcher

logger = logging.getLogger("vns.node")

class VnsNode:
    """Mock/ROS2 Node running the VNS visual positioning loop."""
    
    def __init__(self, config_path: str, database_path: str) -> None:
        self.config = ConfigManager.load(config_path)
        
        # Setup logging
        setup_logging(
            level=self.config.get("logging.level", "INFO"),
            log_dir=self.config.get("logging.log_dir", "./logs")
        )
        
        # Load database
        logger.info("Loading reference database from %s...", database_path)
        try:
            self.database = ReferenceDatabase.load(database_path)
        except ValueError as exc:
            raise ValueError(
                f"{exc} Migrate the database with: "
                f"vns database migrate --input {database_path} "
                f"--output {database_path}"
            ) from exc
        logger.info("Database loaded with %d entries.", self.database.entry_count)
        
        # Initialize modules
        self.extractor = FeatureExtractor(
            algorithm=self.config.get("feature_extraction.algorithm", "ORB"),
            max_features=self.config.get("feature_extraction.max_features", 500),
            scale_factor=self.config.get("feature_extraction.scale_factor", 1.2),
            n_levels=self.config.get("feature_extraction.n_levels", 8)
        )
        
        self.matcher = FeatureMatcher(
            confidence_threshold=self.config.get("matching.confidence_threshold", 0.6),
            ratio_test_threshold=self.config.get("matching.ratio_test_threshold", 0.75),
            min_matches=self.config.get("matching.min_matches", 10)
        )

        # Coarse retrieval: BoVW appearance index (O(1)-ish top-k) when the
        # database ships one, with graceful fallback to the geographic radius
        # search for legacy / --no-vocab databases.
        self.top_k = self.config.get("retrieval.top_k", 5)
        # Hybrid retrieval: intersect BoVW top-k with a geographic radius, but
        # only when a trustworthy position prior is available (see process_image).
        self.geo_gate = self.config.get("retrieval.bovw.geo_gate", True)
        self.geo_gate_radius_deg = self.config.get("retrieval.bovw.geo_gate_radius_deg", 0.002)
        self.bovw_index: Optional[BoVWIndex] = None
        if self.config.get("retrieval.bovw.enabled", True):
            try:
                self.bovw_index = BoVWIndex.load(database_path)
                logger.info(
                    "BoVW coarse retrieval enabled (K=%d words, metric=%s, top_k=%d, "
                    "geo_gate=%s, gate_radius_deg=%s).",
                    self.bovw_index.k, self.bovw_index.metric, self.top_k,
                    self.geo_gate, self.geo_gate_radius_deg
                )
            except (ValueError, FileNotFoundError) as e:
                logger.warning(
                    "BoVW index unavailable (%s); using geographic query_region fallback.",
                    e
                )
        
        self.gnss_monitor = GnssMonitor(
            degraded_hdop=self.config.get("gnss.degraded_hdop", 5.0),
            degraded_satellites=self.config.get("gnss.degraded_satellites", 4),
            denied_timeout_seconds=self.config.get("gnss.denied_timeout_seconds", 2.0)
        )
        
        self.blender = PositionBlender(
            uncertainty_failsafe_threshold=self.config.get("navigation.uncertainty_failsafe_threshold", 10.0),
            visual_timeout_seconds=self.config.get("navigation.visual_timeout_seconds", 300.0),
            fusion_blend_duration=self.config.get("navigation.fusion_blend_duration", 5.0),
            position_continuity_threshold=self.config.get("navigation.position_continuity_threshold", 2.0)
        )
        
        self.camera_config = self.config.get("camera", {})
        self.geo_origin = self.config.get("geo_reference", {
            "origin_latitude": 33.7470,
            "origin_longitude": 73.1370,
            "origin_altitude": 550.0
        })
        
        # State variables
        self.current_gps_fix: Optional[tuple] = None  # (lat, lon, alt)
        self.current_heading = 0.0  # degrees
        self.current_altitude = 580.0  # MSL
        
        self.ground_truth_pose: Optional[tuple] = None  # (lat, lon, alt, heading)
        
        # Create output log file
        log_dir = Path(self.config.get("logging.log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.gt_log_path = log_dir / f"ground_truth_{int(time.time())}.jsonl"
        self._gt_file = open(self.gt_log_path, "w")
        logger.info("Accuracy evaluation logs will be written to: %s", self.gt_log_path)
        
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()

    def process_gps(self, has_fix: bool, num_sats: int, hdop: float, lat: float, lon: float, alt: float) -> None:
        """Process incoming GPS telemetry."""
        state = self.gnss_monitor.update(has_fix, num_sats, hdop)
        if state != GnssState.DENIED:
            self.current_gps_fix = (lat, lon, alt)
            self.current_altitude = alt
        else:
            self.current_gps_fix = None
            
        logger.debug("GPS updated. Health State: %s", state.value)

    def _retrieve_candidates(
        self,
        des: np.ndarray,
        lat_ref: float,
        lon_ref: float,
        has_prior: bool,
    ) -> list:
        """Coarse retrieval of reference candidates for fine matching.

        Primary path is BoVW appearance retrieval (top-k, ~O(1)). When a
        trustworthy position prior exists (``has_prior``) and the geographic gate
        is enabled, the appearance top-k is intersected with a radius around the
        prior to drop visually-similar-but-distant references. The gate is never
        applied without a prior, and never returns empty: if it would discard
        every candidate (e.g. a stale/drifted prior), the unfiltered BoVW top-k
        is used and the disagreement is logged.

        Falls back to the geographic radius search when no BoVW index is loaded
        or it yields nothing usable (e.g. an empty query descriptor set).
        """
        if self.bovw_index is not None and des is not None and len(des) > 0:
            hits = self.bovw_index.query(des, k=self.top_k)
            candidates = [
                self.database.entries[ref_id]
                for ref_id, _ in hits
                if ref_id in self.database.entries
            ]

            if candidates:
                if self.geo_gate and has_prior:
                    gated = [
                        c for c in candidates
                        if self._within_radius(c, lat_ref, lon_ref, self.geo_gate_radius_deg)
                    ]
                    if gated:
                        logger.debug(
                            "BoVW+geo retrieved %d/%d candidate(s) within %.4f deg of "
                            "(%.6f, %.6f): %s",
                            len(gated), len(candidates), self.geo_gate_radius_deg,
                            lat_ref, lon_ref, [c.id for c in gated],
                        )
                        return gated
                    logger.warning(
                        "Geo gate discarded all %d BoVW candidate(s) near (%.6f, %.6f); "
                        "prior may be stale/drifted -- using unfiltered appearance top-k.",
                        len(candidates), lat_ref, lon_ref,
                    )

                logger.debug(
                    "BoVW retrieved %d candidate(s): %s",
                    len(candidates),
                    [(ref_id, round(dist, 4)) for ref_id, dist in hits],
                )
                return candidates

        # Fallback: geographic radius search around the position prior.
        return self.database.query_region(
            lat_ref, lon_ref,
            self.config.get("database.query_radius_deg", 0.001)
        )

    @staticmethod
    def _within_radius(entry, lat: float, lon: float, radius_deg: float) -> bool:
        """Whether a reference entry lies within ``radius_deg`` of (lat, lon).

        Uses the same planar lat/lon distance as ReferenceDatabase.query_region
        so the gate and the geographic fallback agree.
        """
        dlat = entry.latitude - lat
        dlon = entry.longitude - lon
        return math.sqrt(dlat ** 2 + dlon ** 2) <= radius_deg

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Process a new camera frame, run visual matching, blend state, and log accuracy.
        Returns:
            blended_pos: (lat, lon, alt)
            mode: Navigation mode string
        """
        # Timeout check
        gnss_state = self.gnss_monitor.check_timeout()
        
        # Extract features from query image
        kp, des = self.extractor.detect_and_compute(image)
        
        # Query Reference DB
        # Look around last known position, or geocenter if none
        lat_ref = self.geo_origin["origin_latitude"]
        lon_ref = self.geo_origin["origin_longitude"]
        
        # has_prior gates geographic filtering: only a live GPS fix or a recent
        # visual estimate is trustworthy enough to constrain candidates by
        # position. The default origin is not a real prior.
        has_prior = False
        if self.current_gps_fix is not None:
            lat_ref, lon_ref, _ = self.current_gps_fix
            has_prior = True
        elif self.blender._last_estimate is not None:
            lat_ref, lon_ref, _ = self.blender._last_estimate
            has_prior = True

        candidates = self._retrieve_candidates(des, lat_ref, lon_ref, has_prior)
        
        # Find best match
        best_visual_pos = None
        best_confidence = 0.0
        
        for candidate in candidates:
            res = self.matcher.estimate_relative_pose(
                kp, des, candidate, self.camera_config,
                self.current_altitude, self.current_heading, self.geo_origin
            )
            if res is not None:
                # Estimate confidence again for logging/selection
                _, _, conf = self.matcher.match(kp, des, candidate)
                if conf > best_confidence:
                    best_confidence = conf
                    best_visual_pos = res

        # State blender
        blended, mode = self.blender.blend(self.current_gps_fix, best_visual_pos, gnss_state)
        
        # Log ground truth accuracy if available
        if self.ground_truth_pose is not None:
            gt_lat, gt_lon, gt_alt, gt_heading = self.ground_truth_pose
            
            # Simple geodesic distance conversion
            dlat = blended[0] - gt_lat
            dlon = blended[1] - gt_lon
            horizontal_err = 111000.0 * math.sqrt(dlat**2 + dlon**2)
            vertical_err = abs(blended[2] - gt_alt)
            
            d_heading = abs(self.current_heading - gt_heading) % 360.0
            heading_err = min(d_heading, 360.0 - d_heading)
            
            log_data = {
                "timestamp": time.time(),
                "estimated_lat": blended[0],
                "estimated_lon": blended[1],
                "estimated_alt": blended[2],
                "true_lat": gt_lat,
                "true_lon": gt_lon,
                "true_alt": gt_alt,
                "estimated_heading": self.current_heading,
                "true_heading": gt_heading,
                "horizontal_error": horizontal_err,
                "vertical_error": vertical_err,
                "heading_error": heading_err,
                "mode": mode
            }
            
            self._gt_file.write(json.dumps(log_data) + "\n")
            self._gt_file.flush()
            
            logger.debug(
                "Mode: %s | Horiz Err: %.2fm | Vert Err: %.2fm | Heading Err: %.2fdeg",
                mode, horizontal_err, vertical_err, heading_err
            )
            
        return blended, mode

    def close(self) -> None:
        """Close log file."""
        if hasattr(self, "_gt_file") and not self._gt_file.closed:
            self._gt_file.close()

# ROS2 Node wrapper
if ROS2_AVAILABLE:
    class VnsRosNode(Node):
        """ROS2 node wrapping the VNS navigation logic."""
        
        def __init__(self) -> None:
            super().__init__('vns_node')
            
            # Parameters
            self.declare_parameter('config_file', '')
            self.declare_parameter('database_path', '')
            
            config_file = self.get_parameter('config_file').get_parameter_value().string_value
            database_path = self.get_parameter('database_path').get_parameter_value().string_value
            
            if not config_file or not database_path:
                self.get_logger().error("Configuration or Database path not provided!")
                raise ValueError("Paths not provided")
                
            self.vns = VnsNode(config_file, database_path)
            
            # Subscriptions
            self._img_sub = self.create_subscription(
                Image,
                'camera/image_raw',
                self.image_callback,
                10
            )
            
            self._gps_sub = self.create_subscription(
                NavSatFix,
                'gps/fix',
                self.gps_callback,
                10
            )
            
            self._gt_sub = self.create_subscription(
                PoseStamped,
                'ground_truth',
                self.gt_callback,
                10
            )
            
            # Publisher
            self._pose_pub = self.create_publisher(
                PoseStamped,
                '/vns/vision_pose',
                10
            )
            
            self.get_logger().info("VNS ROS2 Node initialized successfully.")

        def gps_callback(self, msg: NavSatFix) -> None:
            """Process raw NavSatFix from GPS sensor."""
            has_fix = msg.status.status >= 0
            # Mock sat count and HDOP for simulation
            num_sats = 10 if has_fix else 0
            hdop = 1.0 if has_fix else 99.0
            
            self.vns.process_gps(
                has_fix, num_sats, hdop,
                msg.latitude, msg.longitude, msg.altitude
            )

        def gt_callback(self, msg: PoseStamped) -> None:
            """Process Ground Truth pose."""
            # Quaternion to Yaw
            q = msg.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw_rad) % 360.0
            
            # Origin Lat/Lon
            lat = self.vns.geo_origin["origin_latitude"]
            lon = self.vns.geo_origin["origin_longitude"]
            alt = self.vns.geo_origin["origin_altitude"]
            
            # Enu displacement to lat/lon approximately
            # In simulation, PoseStamped coordinates are local ENU relative to spawn/origin
            # Let's map ground truth pose (ENU) back to Geodetic!
            gt_lat, gt_lon, gt_alt = enu_to_geodetic(
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                lat, lon, alt
            )
            
            self.vns.ground_truth_pose = (gt_lat, gt_lon, gt_alt, yaw_deg)
            self.vns.current_heading = yaw_deg

        def image_callback(self, msg: Image) -> None:
            """Process camera frame."""
            if CV_BRIDGE_AVAILABLE:
                cv_img = self.vns.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                # Custom raw converter
                cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                
            blended_pos, mode = self.vns.process_image(cv_img)
            
            # Publish estimated PoseStamped
            out_msg = PoseStamped()
            out_msg.header = msg.header
            
            # Convert lat/lon/alt to local ENU for publication
            origin_lat = self.vns.geo_origin["origin_latitude"]
            origin_lon = self.vns.geo_origin["origin_longitude"]
            origin_alt = self.vns.geo_origin["origin_altitude"]
            
            e, n, u = geodetic_to_enu(
                blended_pos[0], blended_pos[1], blended_pos[2],
                origin_lat, origin_lon, origin_alt
            )
            
            out_msg.pose.position.x = e
            out_msg.pose.position.y = n
            out_msg.pose.position.z = u
            
            # Orientation: heading to quaternion
            yaw_rad = math.radians(self.vns.current_heading)
            out_msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
            out_msg.pose.orientation.w = math.cos(yaw_rad / 2.0)
            
            self._pose_pub.publish(out_msg)

def main(args=None):
    """Main execution block."""
    if not ROS2_AVAILABLE:
        print("ROS2 not available in this environment. VnsNode can only be used as python import.")
        sys.exit(1)
        
    rclpy.init(args=args)
    try:
        node = VnsRosNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node execution failed: {e}")
    finally:
        if 'node' in locals():
            node.vns.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
