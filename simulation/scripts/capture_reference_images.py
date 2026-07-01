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
from typing import Any

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
from vns.validation.frame_quality import FrameQualityReport, assess_frame_quality


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


def _resolve_cli_path(
    path_value: str,
    *,
    script_dir: Path,
    default_value: str | None = None,
) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    if default_value is not None and path_value == default_value:
        return (script_dir / candidate).resolve()
    return candidate.resolve()


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


def _missing_publisher_topics(node: Any, topics: list[str]) -> list[str]:
    """Return the subset of `topics` that currently have no active publisher."""
    return [topic for topic in topics if not node.get_publishers_info_by_topic(topic)]


def _wait_for_required_topics(
    node: Any,
    rclpy_module: Any,
    topics: list[str],
    *,
    timeout_sec: float,
    poll_interval_sec: float = 0.5,
) -> list[str]:
    """Poll the ROS graph until every topic in `topics` has a publisher, or
    `timeout_sec` elapses. Returns whichever topics are still missing a
    publisher when the function returns."""
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    missing = _missing_publisher_topics(node, topics)
    while missing and time.monotonic() < deadline:
        rclpy_module.spin_once(node, timeout_sec=poll_interval_sec)
        missing = _missing_publisher_topics(node, topics)
    return missing


def main(argv: list[str] | None = None) -> int:
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
        "--timeout-sec",
        type=float,
        default=30.0,
        help=(
            "Seconds to wait for required topic publishers to appear and for "
            "the first camera/ground-truth sample to arrive before failing "
            "with a clear error (default: 30.0)"
        ),
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
    parser.add_argument(
        "--allow-low-quality",
        action="store_true",
        help=(
            "Capture frames even if they fail the image-quality check (low "
            "ORB feature count, near-uniform color, dominant green, etc.). "
            "For debugging the capture pipeline only -- do not use when "
            "building the production reference database."
        ),
    )
    parser.add_argument(
        "--debug-save-first-frame",
        action="store_true",
        help=(
            "Unconditionally save the first received camera frame to "
            "<output>/debug_first_frame.png with its quality metrics, "
            "regardless of capture pass/fail, then continue normally."
        ),
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    config_default = parser.get_default("config")
    simulation_config_default = parser.get_default("simulation_config")
    output_default = parser.get_default("output")

    config_path = _resolve_cli_path(
        args.config,
        script_dir=script_dir,
        default_value=config_default,
    )
    simulation_config_path = _resolve_cli_path(
        args.simulation_config,
        script_dir=script_dir,
        default_value=simulation_config_default,
    )
    output_dir = _resolve_cli_path(
        args.output,
        script_dir=script_dir,
        default_value=output_default,
    )

    print(f"Reference config: {config_path}")
    print(f"Simulation config: {simulation_config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Camera topic: {args.camera_topic}")
    print(f"Ground-truth topic: {args.ground_truth_topic}")

    if not config_path.exists():
        print(f"Reference config not found: {config_path}", file=sys.stderr)
        return 1
    if not simulation_config_path.exists():
        print(f"Simulation config not found: {simulation_config_path}", file=sys.stderr)
        return 1

    try:
        config, reference_points = load_reference_config(config_path)
    except (OSError, yaml.YAMLError, KeyError, ValueError) as exc:
        print(f"Failed to load reference config {config_path}: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(reference_points)} reference point(s) from {config_path}")

    try:
        origin_lat, origin_lon, origin_alt = _load_geo_origin(simulation_config_path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(
            f"Failed to load simulation config {simulation_config_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    bridge_check = check_cv_bridge_compatibility()
    print(bridge_check.message)
    if not bridge_check.ok:
        return 1

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

        def latest_image_copy(self):
            return None if self._latest_image is None else self._latest_image.copy()

        def current_status(self, target: ReferencePoint) -> tuple[float, float] | None:
            if self._latest_ground_truth is None:
                return None
            return (
                _horizontal_error_m(target, self._latest_ground_truth),
                _heading_error_deg(target.heading, self._latest_ground_truth.heading_deg),
            )

        def capture(self, target: ReferencePoint) -> tuple[CapturedImage | None, FrameQualityReport]:
            """Attempt to capture ``target``. Returns ``(None, report)`` with a
            failing ``report`` if the frame is rejected for quality (and
            ``--allow-low-quality`` was not passed) -- the caller is
            responsible for not appending a ``None`` result and for not
            writing a database index from an all-rejected run."""
            if self._latest_image is None or self._latest_ground_truth is None:
                raise RuntimeError("Latest camera frame or ground truth is not ready.")

            frame = self._latest_image.copy()
            quality = assess_frame_quality(frame)
            if not quality.passed and not args.allow_low_quality:
                return None, quality

            filepath = output_dir / f"{target.id}.jpg"
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
                    "description": target.description,
                    **target.extra,
                    "source_type": "gazebo_camera",
                    "capture_status": "captured_gazebo",
                    "manual_capture_required": False,
                    "source_topic": args.camera_topic,
                    "ground_truth_topic": args.ground_truth_topic,
                    "capture_altitude_msl": ground_truth.altitude,
                    "capture_latitude": ground_truth.latitude,
                    "capture_longitude": ground_truth.longitude,
                    "capture_heading_deg": ground_truth.heading_deg,
                    "position_error_m": round(horizontal_error, 3),
                    "heading_error_deg": round(heading_error, 3),
                    "quality_grayscale_std": round(quality.grayscale_std, 2),
                    "quality_orb_keypoints": quality.orb_keypoint_count,
                    "quality_green_ratio": round(quality.green_ratio, 3),
                    "quality_brightness_mean": round(quality.brightness_mean, 2),
                },
            ), quality

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
    try:
        node = GazeboReferenceCapture()
        captured_images: list[CapturedImage] = []
        try:
            required_topics = [args.camera_topic, args.ground_truth_topic]
            print(
                f"Checking for active publishers on {required_topics} "
                f"(timeout {args.timeout_sec:.0f}s)..."
            )
            missing_topics = _wait_for_required_topics(
                node, rclpy, required_topics, timeout_sec=args.timeout_sec,
            )
            if missing_topics:
                for topic in missing_topics:
                    print(
                        f"No publisher detected on required topic '{topic}' "
                        f"after {args.timeout_sec:.0f}s.",
                        file=sys.stderr,
                    )
                print(
                    "Hint: run `ros2 topic list` in another terminal to confirm "
                    "the simulation is publishing these topics, and make sure "
                    "the simulation stack is running, e.g.:\n"
                    "  ros2 launch vns full_simulation.launch.py headless:=true\n"
                    "or:\n"
                    "  ros2 launch vns localization_test.launch.py headless:=true",
                    file=sys.stderr,
                )
                return 1

            print("Waiting for first camera frame and ground-truth sample...")
            deadline = time.monotonic() + args.timeout_sec
            while rclpy.ok() and not node.ready():
                if time.monotonic() >= deadline:
                    print(
                        f"Timed out after {args.timeout_sec:.0f}s waiting for "
                        f"the first camera frame and ground-truth sample on "
                        f"{args.camera_topic} / {args.ground_truth_topic}.",
                        file=sys.stderr,
                    )
                    return 1
                rclpy.spin_once(node, timeout_sec=0.2)

            if args.debug_save_first_frame:
                debug_frame = node.latest_image_copy()
                debug_path = output_dir / "debug_first_frame.png"
                debug_quality = assess_frame_quality(debug_frame)
                cv2.imwrite(str(debug_path), debug_frame)
                print(f"Debug frame saved to: {debug_path}")
                print(f"Debug frame quality: {debug_quality}")

            rejected_count = 0

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
                        captured, quality = node.capture(point)
                        if captured is None:
                            rejected_count += 1
                            print(f"  REJECTED {point.id}: {quality}", file=sys.stderr)
                            break
                        captured_images.append(captured)
                        print(f"  captured {point.id} -> {captured.filepath} ({quality})")
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

                    captured, quality = node.capture(point)
                    if captured is None:
                        rejected_count += 1
                        print(
                            f"  REJECTED {point.id}: {quality} -- not written. "
                            "Retrying (press Enter again, 's' to skip, 'q' to abort).",
                            file=sys.stderr,
                        )
                        continue
                    captured_images.append(captured)
                    print(f"  captured {point.id} -> {captured.filepath} ({quality})")
                    break

            print(
                f"Captured {len(captured_images)} of {len(reference_points)} "
                f"reference image(s) ({rejected_count} rejected for quality)."
            )

            if not captured_images:
                if rejected_count > 0:
                    print(
                        f"No images were captured: all {rejected_count} candidate "
                        "frame(s) failed the image-quality check (low feature "
                        "count / near-uniform color / dominant green -- likely "
                        "an untextured or unrendered ground plane). No database "
                        "index written. Re-run with --debug-save-first-frame to "
                        "inspect a frame, or --allow-low-quality to bypass "
                        "(debugging only).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "No images were captured: no candidate ever reached the "
                        "position/heading tolerance. No database index written.",
                        file=sys.stderr,
                    )
                return 1

            try:
                index_path = generate_database_index(
                    captured_images,
                    output_dir,
                    config,
                    database_source_type="gazebo_camera",
                )
                coverage = validate_coverage(captured_images, config)
            except (OSError, KeyError, ValueError) as exc:
                print(
                    f"Failed to write database index to {output_dir}: {exc}",
                    file=sys.stderr,
                )
                return 1

            print(f"Database index saved to: {index_path}")
            print(f"Coverage: {coverage}")
            return 0
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
