#!/usr/bin/env python3
"""VNS ROS 2 node — bridges simulation topics to the VNS pipeline.

Subscribes to camera, GPS, IMU, and ground-truth topics from Gazebo,
runs visual feature matching against the reference database, and
sends VISION_POSITION_ESTIMATE to PX4 EKF2 via MAVLink.
"""

import asyncio
import threading
import time
from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, NavSatFix

from vns.config import ConfigManager, InvalidConfigError
from vns.core.runtime import VnsRuntime
from vns.database.reference_db import ReferenceDatabase
from vns.interfaces.mavlink_interface import MAVLinkInterface
from vns.utils.coordinates import geodetic_to_enu
from vns.utils.logging import setup_logging


class VnsNode(Node):

    def __init__(self):
        super().__init__('vns_node')

        self.declare_parameter('config_file', '')
        self.declare_parameter('database_path', '')
        self.declare_parameter('simulation_mode', True)

        config_path = self.get_parameter('config_file').value
        db_path = self.get_parameter('database_path').value

        self._config = self._load_config(config_path)
        logging_cfg = self._config.model.logging
        setup_logging(
            level=logging_cfg.level,
            log_dir=logging_cfg.log_dir,
            log_to_console=logging_cfg.log_to_console,
            log_format=logging_cfg.format,
            max_size_mb=logging_cfg.max_size_mb,
            retention_count=logging_cfg.retention_count,
        )
        self._bridge = CvBridge()
        self._db = self._load_database(db_path or self._config.model.database.path)
        self._runtime = VnsRuntime(self._config, database=self._db)

        ros_cfg = self._config.model.ros
        sim_cfg = self._config.model.simulation

        # MAVLink interface (async) — runs in a background thread
        self._mavlink = MAVLinkInterface.from_config(
            self._config.model.mavlink.model_dump(mode="python")
        )
        self._runtime.attach_mavlink(self._mavlink)
        self._loop = asyncio.new_event_loop()
        self._mavlink_thread = threading.Thread(
            target=self._run_async_loop, daemon=True)
        self._mavlink_thread.start()
        asyncio.run_coroutine_threadsafe(
            self._mavlink_connect(), self._loop)

        self._cam_sub = self.create_subscription(
            Image, ros_cfg.camera_topic, self._on_image, 10)
        self._gps_sub = self.create_subscription(
            NavSatFix, ros_cfg.gps_topic, self._on_gps, 10)
        self._imu_sub = self.create_subscription(
            Imu, ros_cfg.imu_topic, self._on_imu, 10)
        self._gt_sub = self.create_subscription(
            Odometry, sim_cfg.ground_truth_topic, self._on_ground_truth, 10)

        self._vision_pub = self.create_publisher(
            PoseStamped, ros_cfg.vision_pose_topic, 10
        )

        self._gnss_timer = self.create_timer(1.0, self._check_gnss_timeout)

        self.get_logger().info('VNS node started')

    # ── Async helpers ──

    def _run_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _mavlink_connect(self):
        try:
            await self._mavlink.connect()
            self.get_logger().info('MAVLink connected')
        except Exception as e:
            self.get_logger().error(f'MAVLink connection failed: {e}')

    def _send_vision_estimate(
        self, n: float, e: float, d: float, yaw_rad: float
    ):
        """Schedule async send of VISION_POSITION_ESTIMATE to PX4."""
        time_usec = int(time.time() * 1e6)
        asyncio.run_coroutine_threadsafe(
            self._mavlink.send_vision_position_estimate(
                x_m=n, y_m=e, z_m=d,
                yaw_rad=yaw_rad,
                time_usec=time_usec,
            ),
            self._loop,
        )

    # ── Config / DB ──

    def _load_config(self, path: str) -> ConfigManager:
        if not path:
            self.get_logger().warn('Config path not provided, using defaults')
            return ConfigManager()
        if not Path(path).exists():
            self.get_logger().warn(f'Config not found: {path}, using defaults')
            return ConfigManager()
        try:
            return ConfigManager.from_yaml(path)
        except InvalidConfigError as exc:
            self.get_logger().error(f'Invalid config file {path}: {exc}')
            raise

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
            self.get_logger().error(
                f'Failed to load database: {e}. '
                f'Migrate legacy databases with: '
                f'vns database migrate --input {path} --output {path}'
            )
            return None

    # ── Callbacks ──

    def _on_image(self, msg: Image):
        if self._runtime.localizer is None:
            return

        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self._runtime.process_frame(cv_image)
        if not result.success:
            return

        assert result.blended_pose is not None
        blended = result.blended_pose

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = blended[1]  # lon
        pose_msg.pose.position.y = blended[0]  # lat
        pose_msg.pose.position.z = blended[2]  # alt
        self._vision_pub.publish(pose_msg)

        geo_origin = self._runtime.geo_origin
        origin_lat = geo_origin.get('origin_latitude', 0.0)
        origin_lon = geo_origin.get('origin_longitude', 0.0)
        origin_alt = geo_origin.get('origin_altitude', 0.0)
        e, n, u = geodetic_to_enu(
            blended[0], blended[1], blended[2],
            origin_lat, origin_lon, origin_alt)
        self._send_vision_estimate(
            n=n, e=e, d=-u, yaw_rad=result.localization.yaw_rad)

    def _on_gps(self, msg: NavSatFix):
        has_fix = msg.status.status >= 0
        # NavSatFix carries no satellite count, so derive a representative one
        # from the fix flag: a real fix must report healthy (>= degraded_satellites)
        # so the monitor can reach HEALTHY, otherwise GnssMonitor pins to DEGRADED
        # and the blender abandons good GPS for visual navigation.
        self._runtime.update_gps(
            latitude=msg.latitude,
            longitude=msg.longitude,
            altitude=msg.altitude,
            has_fix=has_fix,
            num_satellites=10 if has_fix else 0,
            hdop=1.0 if has_fix else 99.9,
        )

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        self._runtime.update_heading_from_quaternion(
            w=q.w,
            x=q.x,
            y=q.y,
            z=q.z,
        )

    def _on_ground_truth(self, msg: Odometry):
        self._runtime.update_ground_truth_altitude(msg.pose.pose.position.z)

    def _check_gnss_timeout(self):
        self._runtime.check_gnss_timeout()

    def destroy_node(self):
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._mavlink.disconnect(), self._loop
            )
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._mavlink_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VnsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
