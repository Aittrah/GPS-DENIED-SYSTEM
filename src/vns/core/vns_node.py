#!/usr/bin/env python3
"""VNS ROS 2 node — bridges simulation topics to the VNS pipeline.

Subscribes to camera, GPS, IMU, and ground-truth topics from Gazebo,
runs visual feature matching against the reference database, and
publishes vision-based pose estimates.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, NavSatFix

from vns.core.blender import PositionBlender
from vns.core.gnss_monitor import GnssMonitor, GnssState
from vns.database.reference_db import ReferenceDatabase
from vns.vision.extractor import FeatureExtractor
from vns.vision.matcher import FeatureMatcher


class VnsNode(Node):

    def __init__(self):
        super().__init__('vns_node')

        self.declare_parameter('config_file', '')
        self.declare_parameter('database_path', '')
        self.declare_parameter('simulation_mode', True)

        config_path = self.get_parameter('config_file').value
        db_path = self.get_parameter('database_path').value

        self._cfg = self._load_config(config_path)
        self._bridge = CvBridge()

        cam_cfg = self._cfg.get('camera', {})
        feat_cfg = self._cfg.get('feature_extraction', {})
        match_cfg = self._cfg.get('matching', {})
        gnss_cfg = self._cfg.get('gnss', {})
        nav_cfg = self._cfg.get('navigation', {})
        geo_cfg = self._cfg.get('geo_reference', {})

        self._camera_config = cam_cfg
        self._geo_origin = geo_cfg

        self._extractor = FeatureExtractor(
            algorithm=feat_cfg.get('algorithm', 'ORB'),
            max_features=feat_cfg.get('max_features', 500),
            scale_factor=feat_cfg.get('scale_factor', 1.2),
            n_levels=feat_cfg.get('n_levels', 8),
        )
        self._matcher = FeatureMatcher(
            confidence_threshold=match_cfg.get('confidence_threshold', 0.6),
            ratio_test_threshold=match_cfg.get('ratio_test_threshold', 0.75),
            min_matches=match_cfg.get('min_matches', 10),
        )
        self._gnss_monitor = GnssMonitor(
            degraded_hdop=gnss_cfg.get('degraded_hdop', 5.0),
            degraded_satellites=gnss_cfg.get('degraded_satellites', 4),
            denied_timeout_seconds=gnss_cfg.get('denied_timeout_seconds', 2.0),
        )
        self._blender = PositionBlender(
            uncertainty_failsafe_threshold=nav_cfg.get('uncertainty_failsafe_threshold', 10.0),
            visual_timeout_seconds=nav_cfg.get('visual_timeout_seconds', 300.0),
            fusion_blend_duration=nav_cfg.get('fusion_blend_duration', 5.0),
            position_continuity_threshold=nav_cfg.get('position_continuity_threshold', 2.0),
        )

        self._db = self._load_database(db_path)
        self._query_radius = self._cfg.get('database', {}).get('query_radius_deg', 0.001)

        self._last_gps = None
        self._last_altitude = 0.0
        self._last_heading = 0.0

        self._cam_sub = self.create_subscription(
            Image, 'camera/image_raw', self._on_image, 10)
        self._gps_sub = self.create_subscription(
            NavSatFix, 'gps/fix', self._on_gps, 10)
        self._imu_sub = self.create_subscription(
            Imu, 'imu/data', self._on_imu, 10)
        self._gt_sub = self.create_subscription(
            Odometry, 'ground_truth', self._on_ground_truth, 10)

        self._vision_pub = self.create_publisher(PoseStamped, 'vns/vision_pose', 10)

        self._gnss_timer = self.create_timer(1.0, self._check_gnss_timeout)

        self.get_logger().info('VNS node started')

    def _load_config(self, path: str) -> dict:
        if not path or not Path(path).exists():
            self.get_logger().warn(f'Config not found: {path}, using defaults')
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _load_database(self, path: str):
        if not path or not Path(path).exists():
            self.get_logger().warn(f'Database not found: {path}')
            return None
        try:
            db = ReferenceDatabase.load(path)
            self.get_logger().info(
                f'Loaded reference database: {db.entry_count} entries')
            return db
        except Exception as e:
            self.get_logger().error(f'Failed to load database: {e}')
            return None

    def _on_image(self, msg: Image):
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        kp, des = self._extractor.detect_and_compute(cv_image)
        if len(kp) == 0 or self._db is None:
            return

        anchor_lat = self._last_gps[0] if self._last_gps else \
            self._geo_origin.get('origin_latitude', 0.0)
        anchor_lon = self._last_gps[1] if self._last_gps else \
            self._geo_origin.get('origin_longitude', 0.0)

        candidates = self._db.query_region(
            anchor_lat, anchor_lon, self._query_radius)
        if not candidates:
            return

        best_result = None
        best_confidence = 0.0
        for entry in candidates:
            result = self._matcher.estimate_relative_pose(
                kp, des, entry,
                self._camera_config, self._last_altitude,
                self._last_heading, self._geo_origin,
            )
            if result is not None:
                _, _, _ = result
                _, _, conf = self._matcher.match(kp, des, entry)
                if conf > best_confidence:
                    best_confidence = conf
                    best_result = result

        if best_result is None:
            return

        lat, lon, alt = best_result
        visual_pos = (lat, lon, alt)
        gps_pos = self._last_gps
        gnss_state = self._gnss_monitor.state

        blended, mode = self._blender.blend(
            gps_pos, visual_pos, gnss_state, time.time())

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = blended[1]  # lon
        pose_msg.pose.position.y = blended[0]  # lat
        pose_msg.pose.position.z = blended[2]  # alt
        self._vision_pub.publish(pose_msg)

    def _on_gps(self, msg: NavSatFix):
        self._last_gps = (msg.latitude, msg.longitude, msg.altitude)
        has_fix = msg.status.status >= 0
        self._gnss_monitor.update(
            has_fix=has_fix,
            num_satellites=0,
            hdop=99.9 if not has_fix else 1.0,
        )

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
        self._last_heading = np.degrees(yaw_rad)

    def _on_ground_truth(self, msg: Odometry):
        self._last_altitude = msg.pose.pose.position.z

    def _check_gnss_timeout(self):
        self._gnss_monitor.check_timeout()


def main(args=None):
    rclpy.init(args=args)
    node = VnsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
