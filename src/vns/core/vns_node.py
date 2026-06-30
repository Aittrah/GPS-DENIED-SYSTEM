#!/usr/bin/env python3
"""VNS ROS 2 node — bridges simulation topics to the VNS pipeline.

Subscribes to camera, GPS, IMU, and ground-truth topics from Gazebo,
runs visual feature matching against the reference database, and
sends VISION_POSITION_ESTIMATE to PX4 EKF2 via MAVLink.
"""

import asyncio
import math
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
from vns.utils.coordinates import enu_to_geodetic, geodetic_to_enu
from vns.utils.logging import setup_logging
from vns.validation import ImageFrameLogger, JsonlEvaluationLogger


class VnsNode(Node):

    def __init__(self):
        super().__init__('vns_node')

        self.declare_parameter('config_file', '')
        self.declare_parameter('database_path', '')
        self.declare_parameter('camera_topic', '')
        self.declare_parameter('ground_truth_topic', '')
        self.declare_parameter('evaluation_logging', '')
        self.declare_parameter('simulation_mode', True)

        config_path = self.get_parameter('config_file').value
        db_path = self.get_parameter('database_path').value

        self._config = self._load_config(config_path)
        mavlink_config = self._config.model.mavlink.model_dump(mode="python")
        self.declare_parameter('mavlink_enabled', mavlink_config.get('enabled', True))
        logging_cfg = self._config.model.logging
        sim_cfg = self._config.model.simulation
        self._simulation_mode = bool(self.get_parameter('simulation_mode').value)
        self._evaluation_logging_enabled = self._resolve_optional_bool(
            'evaluation_logging',
            sim_cfg.log_ground_truth,
        )
        self._record_estimates_without_ground_truth = (
            sim_cfg.record_estimates_without_ground_truth
        )
        setup_logging(
            level=logging_cfg.level,
            log_dir=logging_cfg.log_dir,
            log_to_console=logging_cfg.log_to_console,
            log_format=logging_cfg.format,
            max_size_mb=logging_cfg.max_size_mb,
            retention_count=logging_cfg.retention_count,
        )
        self._bridge = CvBridge()
        configured_db_path = self._config.resolve_path(self._config.model.database.path)
        self._db = self._load_database(db_path or str(configured_db_path))
        self._runtime = VnsRuntime(self._config, database=self._db)
        self._evaluation_logger = JsonlEvaluationLogger(
            enabled=self._evaluation_logging_enabled,
            log_dir=logging_cfg.log_dir,
        )
        self._image_logger = ImageFrameLogger(
            enabled=bool(logging_cfg.log_images),
            log_dir=logging_cfg.log_dir,
            stride=logging_cfg.image_log_stride,
        )

        ros_cfg = self._config.model.ros
        camera_topic = self.get_parameter('camera_topic').value or ros_cfg.camera_topic
        ground_truth_topic = (
            self.get_parameter('ground_truth_topic').value or sim_cfg.ground_truth_topic
        )
        self._mavlink_enabled = bool(self.get_parameter('mavlink_enabled').value)
        self._mavlink: MAVLinkInterface | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mavlink_thread: threading.Thread | None = None

        # MAVLink interface (async) — runs in a background thread
        if self._mavlink_enabled:
            mavlink_config['enabled'] = self._mavlink_enabled
            self._mavlink = MAVLinkInterface.from_config(mavlink_config)
            self._runtime.attach_mavlink(self._mavlink)
            self._loop = asyncio.new_event_loop()
            self._mavlink_thread = threading.Thread(
                target=self._run_async_loop, daemon=True)
            self._mavlink_thread.start()
            asyncio.run_coroutine_threadsafe(
                self._mavlink_connect(), self._loop)
        else:
            self.get_logger().info('MAVLink disabled for this run')

        self._cam_sub = self.create_subscription(
            Image, camera_topic, self._on_image, 10)
        self._gps_sub = self.create_subscription(
            NavSatFix, ros_cfg.gps_topic, self._on_gps, 10)
        self._imu_sub = self.create_subscription(
            Imu, ros_cfg.imu_topic, self._on_imu, 10)
        self._gt_sub = None
        if self._simulation_mode and ground_truth_topic:
            self._gt_sub = self.create_subscription(
                Odometry, ground_truth_topic, self._on_ground_truth, 10)

        self._vision_pub = self.create_publisher(
            PoseStamped, ros_cfg.vision_pose_topic, 10
        )

        self._gnss_timer = self.create_timer(1.0, self._check_gnss_timeout)

        self.get_logger().info('VNS node started')

    # ── Async helpers ──

    def _run_async_loop(self):
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _mavlink_connect(self):
        if self._mavlink is None:
            return
        try:
            await self._mavlink.connect()
            self.get_logger().info('MAVLink connected')
        except Exception as e:
            self.get_logger().error(f'MAVLink connection failed: {e}')

    def _send_vision_estimate(
        self, n: float, e: float, d: float, yaw_rad: float
    ):
        """Schedule async send of VISION_POSITION_ESTIMATE to PX4."""
        if not self._mavlink_enabled or self._mavlink is None or self._loop is None:
            return
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
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._image_logger.save(
            cv_image, timestamp=self._stamp_to_seconds(msg.header.stamp)
        )
        result = self._runtime.process_frame(cv_image)
        self._log_evaluation_sample(msg, result)
        if result.blended_pose is None:
            return

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
        yaw_rad = (
            result.localization.yaw_rad
            if result.localization.success
            else self._runtime.last_yaw_rad
        )
        self._send_vision_estimate(
            n=n, e=e, d=-u, yaw_rad=yaw_rad)

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
        stamp = self._stamp_to_seconds(msg.header.stamp)
        self._runtime.update_imu(
            w=q.w,
            x=q.x,
            y=q.y,
            z=q.z,
            accel_x=msg.linear_acceleration.x,
            accel_y=msg.linear_acceleration.y,
            accel_z=msg.linear_acceleration.z,
            gyro_x=msg.angular_velocity.x,
            gyro_y=msg.angular_velocity.y,
            gyro_z=msg.angular_velocity.z,
            timestamp=stamp,
        )

    def _on_ground_truth(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw_rad) % 360.0

        geo_origin = self._runtime.geo_origin
        origin_lat = geo_origin.get('origin_latitude', 0.0)
        origin_lon = geo_origin.get('origin_longitude', 0.0)
        origin_alt = geo_origin.get('origin_altitude', 0.0)

        gt_lat, gt_lon, gt_alt = enu_to_geodetic(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            origin_lat,
            origin_lon,
            origin_alt,
        )
        stamp = self._stamp_to_seconds(msg.header.stamp)
        self._runtime.update_ground_truth_pose(
            latitude=gt_lat,
            longitude=gt_lon,
            altitude=gt_alt,
            heading_deg=yaw_deg,
            timestamp=stamp,
        )

    def _check_gnss_timeout(self):
        self._runtime.check_gnss_timeout()

    def _log_evaluation_sample(self, msg: Image, result) -> None:
        if self._evaluation_logger is None:
            return

        ros_time = self._stamp_to_seconds(msg.header.stamp)
        timestamp = ros_time if ros_time is not None else result.localization.timestamp
        self._evaluation_logger.write_frame(
            timestamp=timestamp,
            ros_time=ros_time,
            result=result,
            ground_truth=self._runtime.last_ground_truth,
            gnss_status=self._runtime.gnss_monitor.state.name,
            record_without_ground_truth=self._record_estimates_without_ground_truth,
        )

    def _resolve_optional_bool(self, parameter_name: str, default: bool) -> bool:
        raw_value = self.get_parameter(parameter_name).value
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value in {"", None}:
            return default
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        self.get_logger().warn(
            f"Invalid boolean override for {parameter_name}: {raw_value!r}; "
            f"using config value {default!r}"
        )
        return default

    @staticmethod
    def _stamp_to_seconds(stamp) -> float | None:
        seconds = getattr(stamp, 'sec', 0)
        nanoseconds = getattr(stamp, 'nanosec', 0)
        if seconds == 0 and nanoseconds == 0:
            return None
        return float(seconds) + float(nanoseconds) / 1e9

    def destroy_node(self):
        if self._evaluation_logger is not None:
            self._evaluation_logger.close()
        if self._image_logger is not None:
            self._image_logger.close()
        if self._mavlink is not None and self._loop is not None and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._mavlink.disconnect(), self._loop
            )
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._mavlink_thread is not None:
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
