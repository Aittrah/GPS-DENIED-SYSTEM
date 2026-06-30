#!/usr/bin/env python3
"""Capture real Gazebo downward-camera reference images under ROS 2 Humble."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

import cv2
import yaml

from reference_image_utils import (
    CapturedImage,
    ReferencePoint,
    generate_database_index,
    load_reference_config,
    validate_coverage,
)

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from vns.utils.coordinates import enu_to_geodetic
from vns.utils.ros_env import check_cv_bridge_compatibility


@dataclass(frozen=True)
class GroundTruthPose:
    latitude: float
    longitude: float
    altitude: float
    heading_deg: float
    timestamp: float | None


def _load_geo_origin(config_path: Path) -> tuple[float, float, float]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    geo = config.get("geo_reference", {})
    return (
        float(geo.get("origin_latitude", 33.7470)),
        float(geo.get("origin_longitude", 73.1370)),
        float(geo.get("origin_altitude", 550.0)),
    )


def _horizontal_error_m(target: ReferencePoint, pose: GroundTruthPose) -> float:
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(target.latitude))
    north_m = (pose.latitude - target.latitude) * m_per_deg_lat
    east_m = (pose.longitude - target.longitude) * m_per_deg_lon
    return float(math.hypot(north_m, east_m))


def _heading_error_deg(target_heading: float, actual_heading: float) -> float:
    delta = (actual_heading - target_heading + 180.0) % 360.0 - 180.0
    return abs(delta)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_cli_path(path_value: str, *, script_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (script_dir / candidate).resolve()


def _import_ros_types() -> tuple[object, object, object, object]:
    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ImportError as exc:
        print(
            "ROS 2 Python packages are not available in this environment. "
            "Source /opt/ros/humble/setup.bash and use the ROS 2 Python "
            "interpreter before running this capture script.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return rclpy, Node, Image, Odometry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture real Gazebo downward-camera reference images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../database/qau_reference_metadata.yaml",
        help="Path to reference metadata YAML file",
    )
    parser.add_argument(
        "--simulation-config",
        type=str,
        default="../config/simulation.yaml",
        help="Path to simulation.yaml for geo_reference origin lookup",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../database/images/gazebo",
        help="Output directory for captured images",
    )
    parser.add_argument(
        "--camera-topic",
        type=str,
        default="/vns_drone/downward_camera/image_raw",
        help="ROS image topic to capture",
    )
    parser.add_argument(
        "--ground-truth-topic",
        type=str,
        default="/vns_drone/ground_truth",
        help="Ground-truth odometry topic",
    )
    parser.add_argument(
        "--position-tolerance-m",
        type=float,
        default=3.0,
        help="Maximum horizontal distance from the target reference point",
    )
    parser.add_argument(
        "--heading-tolerance-deg",
        type=float,
        default=20.0,
        help="Maximum absolute heading error for a valid capture",
    )
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="Capture automatically once the current target is within tolerance",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing captured images in the output directory",
    )
    args = parser.parse_args()

    bridge_check = check_cv_bridge_compatibility()
    print(bridge_check.message)
    if not bridge_check.ok:
        return 1

    script_dir = Path(__file__).resolve().parent
    config_path = _resolve_cli_path(args.config, script_dir=script_dir)
    simulation_config_path = _resolve_cli_path(
        args.simulation_config,
        script_dir=script_dir,
    )
    output_dir = _resolve_cli_path(args.output, script_dir=script_dir)
    config, reference_points = load_reference_config(config_path)
    origin_lat, origin_lon, origin_alt = _load_geo_origin(simulation_config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        collisions = [
            output_dir / f"{point.id}.jpg"
            for point in reference_points
            if (output_dir / f"{point.id}.jpg").exists()
        ]
        if collisions:
            print(
                "Refusing to overwrite existing Gazebo captures. Use --overwrite "
                "or a fresh --output directory.",
                file=sys.stderr,
            )
            return 1

    rclpy, Node, Image, Odometry = _import_ros_types()

    class GazeboReferenceCapture(Node):
        def __init__(self) -> None:
            super().__init__("gazebo_reference_capture")
            self._bridge = bridge_check.bridge
            self._latest_image = None
            self._latest_ground_truth: GroundTruthPose | None = None
            self.create_subscription(Image, args.camera_topic, self._on_image, 10)
            self.create_subscription(
                Odometry,
                args.ground_truth_topic,
                self._on_ground_truth,
                10,
            )

        def ready(self) -> bool:
            return self._latest_image is not None and self._latest_ground_truth is not None

        def current_status(self, target: ReferencePoint) -> tuple[float, float] | None:
            if self._latest_ground_truth is None:
                return None
            return (
                _horizontal_error_m(target, self._latest_ground_truth),
                _heading_error_deg(target.heading, self._latest_ground_truth.heading_deg),
            )

        def capture(self, target: ReferencePoint) -> CapturedImage:
            if self._latest_image is None or self._latest_ground_truth is None:
                raise RuntimeError("Latest camera frame or ground truth is not ready.")

            filepath = output_dir / f"{target.id}.jpg"
            frame = self._latest_image.copy()
            if not cv2.imwrite(str(filepath), frame):
                raise RuntimeError(f"Failed to write {filepath}")

            ground_truth = self._latest_ground_truth
            horizontal_error = _horizontal_error_m(target, ground_truth)
            heading_error = _heading_error_deg(target.heading, ground_truth.heading_deg)
            return CapturedImage(
                id=target.id,
                filepath=str(filepath),
                latitude=target.latitude,
                longitude=target.longitude,
                altitude=target.altitude,
                heading=target.heading,
                timestamp=_utc_now_iso(),
                width=frame.shape[1],
                height=frame.shape[0],
                extra={
                    "source_type": "gazebo_camera",
                    "source_topic": args.camera_topic,
                    "ground_truth_topic": args.ground_truth_topic,
                    "capture_altitude_msl": ground_truth.altitude,
                    "capture_latitude": ground_truth.latitude,
                    "capture_longitude": ground_truth.longitude,
                    "capture_heading_deg": ground_truth.heading_deg,
                    "position_error_m": round(horizontal_error, 3),
                    "heading_error_deg": round(heading_error, 3),
                },
            )

        def _on_image(self, msg) -> None:
            self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        def _on_ground_truth(self, msg) -> None:
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw_rad) % 360.0
            lat, lon, alt = enu_to_geodetic(
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                origin_lat,
                origin_lon,
                origin_alt,
            )
            stamp = msg.header.stamp
            timestamp = None
            if getattr(stamp, "sec", 0) or getattr(stamp, "nanosec", 0):
                timestamp = float(stamp.sec) + float(stamp.nanosec) / 1e9
            self._latest_ground_truth = GroundTruthPose(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                heading_deg=yaw_deg,
                timestamp=timestamp,
            )

    rclpy.init(args=None)
    node = GazeboReferenceCapture()
    captured_images: list[CapturedImage] = []

    try:
        print("Waiting for first camera frame and ground-truth sample...")
        while rclpy.ok() and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.2)

        for index, point in enumerate(reference_points, start=1):
            print(
                f"[{index}/{len(reference_points)}] target={point.id} "
                f"lat={point.latitude:.6f} lon={point.longitude:.6f} "
                f"heading={point.heading:.1f}"
            )
            last_status_log = 0.0
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.2)
                status = node.current_status(point)
                if status is None:
                    continue
                position_error, heading_error = status
                now = time.time()
                if now - last_status_log >= 1.0:
                    print(
                        f"  current error: position={position_error:.2f} m "
                        f"heading={heading_error:.1f} deg"
                    )
                    last_status_log = now

                if position_error > args.position_tolerance_m:
                    continue
                if heading_error > args.heading_tolerance_deg:
                    continue

                if args.auto_capture:
                    captured = node.capture(point)
                    captured_images.append(captured)
                    print(f"  captured {point.id} -> {captured.filepath}")
                    break

                command = input(
                    "  within tolerance. Press Enter to capture, 's' to skip, "
                    "or 'q' to abort: "
                ).strip().lower()
                if command == "q":
                    print("Capture aborted by user.")
                    return 1
                if command == "s":
                    print(f"  skipped {point.id}")
                    break

                captured = node.capture(point)
                captured_images.append(captured)
                print(f"  captured {point.id} -> {captured.filepath}")
                break

        if not captured_images:
            print("No images were captured; no database index written.", file=sys.stderr)
            return 1

        index_path = generate_database_index(
            captured_images,
            output_dir,
            config,
            database_source_type="gazebo_camera",
        )
        coverage = validate_coverage(captured_images, config)
        print(f"Database index saved to: {index_path}")
        print(f"Coverage: {coverage}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
