"""Tests for the Gazebo reference-image capture CLI (FR-11).

Regression coverage for the bug where `capture_reference_images.py` defined
`main()` but never invoked it, so running the script silently exited 0 with
no output and no `database_index.yaml`. No Gazebo is required by any test
here; the missing-topic test uses real (but Gazebo-less) `rclpy` since it is
available in this dev/CI environment.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import uuid

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "simulation" / "scripts"
SCRIPT = SCRIPTS / "capture_reference_images.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import capture_reference_images as cri  # noqa: E402
import reference_image_utils as riu  # noqa: E402


def _write_reference_config(path: Path, *, point_count: int = 1) -> None:
    points = [
        {
            "id": f"point_{i}",
            "latitude": 33.7470 + i * 0.001,
            "longitude": 73.1370 + i * 0.001,
            "altitude": 550.0,
            "heading": 0.0,
            "description": f"test point {i}",
        }
        for i in range(point_count)
    ]
    document = {
        "database": {
            "name": "Test Reference Database",
            "version": "1.0.0",
            "location": "Test",
            "bounds": {
                "min_latitude": 33.740,
                "max_latitude": 33.755,
                "min_longitude": 73.130,
                "max_longitude": 73.145,
            },
        },
        "reference_points": points,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_simulation_config(path: Path) -> None:
    document = {
        "geo_reference": {
            "origin_latitude": 33.7470,
            "origin_longitude": 73.1370,
            "origin_altitude": 550.0,
        }
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_help_flag_exits_zero_and_documents_timeout_flag() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--timeout-sec" in result.stdout


def test_missing_config_exits_nonzero_via_subprocess(tmp_path: Path) -> None:
    """Regression guard: before the fix, this exited 0 with zero output."""
    bad_config = tmp_path / "does_not_exist.yaml"
    sim_config = tmp_path / "simulation.yaml"
    _write_simulation_config(sim_config)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(bad_config),
            "--simulation-config",
            str(sim_config),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert "does_not_exist.yaml" in result.stderr
    assert not (tmp_path / "out" / "database_index.yaml").exists()


def test_missing_config_does_not_invoke_ros_environment_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_config = tmp_path / "does_not_exist.yaml"
    sim_config = tmp_path / "simulation.yaml"
    _write_simulation_config(sim_config)

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "check_cv_bridge_compatibility should not run before config validation"
        )

    monkeypatch.setattr(cri, "check_cv_bridge_compatibility", _fail_if_called)

    exit_code = cri.main(
        [
            "--config",
            str(bad_config),
            "--simulation-config",
            str(sim_config),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Reference config not found" in captured.err


def test_missing_simulation_config_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "qau_reference_metadata.yaml"
    _write_reference_config(config)
    bad_sim_config = tmp_path / "does_not_exist_sim.yaml"

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "check_cv_bridge_compatibility should not run before config validation"
        )

    monkeypatch.setattr(cri, "check_cv_bridge_compatibility", _fail_if_called)

    exit_code = cri.main(
        [
            "--config",
            str(config),
            "--simulation-config",
            str(bad_sim_config),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Simulation config not found" in captured.err


def test_malformed_config_yaml_exits_nonzero_with_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "qau_reference_metadata.yaml"
    config.write_text("database: [oops\n", encoding="utf-8")
    sim_config = tmp_path / "simulation.yaml"
    _write_simulation_config(sim_config)

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "check_cv_bridge_compatibility should not run before config validation"
        )

    monkeypatch.setattr(cri, "check_cv_bridge_compatibility", _fail_if_called)

    exit_code = cri.main(
        [
            "--config",
            str(config),
            "--simulation-config",
            str(sim_config),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Failed to load reference config" in captured.err


def test_missing_topic_publisher_exits_nonzero_with_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("cv_bridge")

    config = tmp_path / "qau_reference_metadata.yaml"
    _write_reference_config(config)
    sim_config = tmp_path / "simulation.yaml"
    _write_simulation_config(sim_config)
    output_dir = tmp_path / "out"

    suffix = uuid.uuid4().hex[:8]
    camera_topic = f"/test_capture_{suffix}/camera"
    ground_truth_topic = f"/test_capture_{suffix}/ground_truth"

    exit_code = cri.main(
        [
            "--config",
            str(config),
            "--simulation-config",
            str(sim_config),
            "--output",
            str(output_dir),
            "--camera-topic",
            camera_topic,
            "--ground-truth-topic",
            ground_truth_topic,
            "--timeout-sec",
            "1.0",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert camera_topic in captured.err
    assert ground_truth_topic in captured.err
    assert "ros2 topic list" in captured.err
    assert not (output_dir / "database_index.yaml").exists()


def test_all_rejected_frames_exit_nonzero_and_write_no_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A solid-color (blank/featureless) camera frame must be rejected by the
    image-quality check: not written to disk, not indexed, and the run must
    exit non-zero with no `database_index.yaml` (FR-11 #10, #11)."""
    pytest.importorskip("rclpy")
    pytest.importorskip("cv_bridge")
    import threading

    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    config = tmp_path / "qau_reference_metadata.yaml"
    _write_reference_config(config, point_count=1)
    sim_config = tmp_path / "simulation.yaml"
    _write_simulation_config(sim_config)
    output_dir = tmp_path / "out"

    suffix = uuid.uuid4().hex[:8]
    camera_topic = f"/test_capture_reject_{suffix}/camera"
    ground_truth_topic = f"/test_capture_reject_{suffix}/ground_truth"

    # Independent rclpy context so this publisher survives cri.main()'s own
    # rclpy.init()/shutdown() on the default global context.
    pub_context = Context()
    rclpy.init(args=None, context=pub_context)
    pub_node = Node(f"test_publisher_{suffix}", context=pub_context)
    # Spin on an executor bound to pub_context — rclpy.spin_once() would use the
    # global executor on the *default* context, which cri.main() shuts down,
    # crashing this thread before any frame is delivered.
    pub_executor = SingleThreadedExecutor(context=pub_context)
    pub_executor.add_node(pub_node)
    bridge = CvBridge()
    image_pub = pub_node.create_publisher(Image, camera_topic, 10)
    odom_pub = pub_node.create_publisher(Odometry, ground_truth_topic, 10)

    # Solid green: a stand-in for an untextured/default Gazebo ground plane.
    solid_green = np.zeros((480, 640, 3), dtype=np.uint8)
    solid_green[:, :] = (50, 150, 50)
    img_msg = bridge.cv2_to_imgmsg(solid_green, encoding="bgr8")

    # Identity pose at the world origin == reference point's lat/lon (0 error).
    odom_msg = Odometry()
    odom_msg.pose.pose.orientation.w = 1.0

    stop = threading.Event()

    def _publish_loop() -> None:
        while not stop.is_set():
            image_pub.publish(img_msg)
            odom_pub.publish(odom_msg)
            pub_executor.spin_once(timeout_sec=0.05)

    thread = threading.Thread(target=_publish_loop, daemon=True)
    thread.start()

    try:
        exit_code = cri.main(
            [
                "--config",
                str(config),
                "--simulation-config",
                str(sim_config),
                "--output",
                str(output_dir),
                "--camera-topic",
                camera_topic,
                "--ground-truth-topic",
                ground_truth_topic,
                "--timeout-sec",
                "10.0",
                "--auto-capture",
            ]
        )
    finally:
        stop.set()
        thread.join(timeout=5.0)
        pub_executor.remove_node(pub_node)
        pub_executor.shutdown()
        pub_node.destroy_node()
        rclpy.shutdown(context=pub_context)

    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert exit_code == 1
    # Per-point rejection is logged to stderr; the run summary
    # ("N rejected for quality") goes to stdout.
    assert "REJECTED" in captured.err
    assert "image-quality check" in captured.err.lower()
    assert "rejected for quality" in combined
    assert not (output_dir / "database_index.yaml").exists()
    assert not (output_dir / "point_0.jpg").exists()


def test_generate_database_index_path_matches_output_convention(tmp_path: Path) -> None:
    output_dir = tmp_path / "gazebo"
    output_dir.mkdir()
    image_path = output_dir / "point_0.jpg"
    image_path.write_bytes(b"fake-jpg-bytes")

    captured_image = riu.CapturedImage(
        id="point_0",
        filepath=str(image_path),
        latitude=33.7470,
        longitude=73.1370,
        altitude=550.0,
        heading=0.0,
        timestamp="2026-07-01T00:00:00+00:00",
        width=640,
        height=480,
    )
    config = {
        "database": {
            "name": "Test",
            "bounds": {
                "min_latitude": 33.740,
                "max_latitude": 33.755,
                "min_longitude": 73.130,
                "max_longitude": 73.145,
            },
        }
    }

    index_path = riu.generate_database_index(
        [captured_image], output_dir, config, database_source_type="gazebo_camera"
    )

    assert index_path == (output_dir / "database_index.yaml").resolve()
    assert index_path.exists()


def test_generate_database_index_tags_gazebo_source_type(tmp_path: Path) -> None:
    output_dir = tmp_path / "gazebo"
    output_dir.mkdir()
    image_path = output_dir / "point_0.jpg"
    image_path.write_bytes(b"fake-jpg-bytes")

    captured_image = riu.CapturedImage(
        id="point_0",
        filepath=str(image_path),
        latitude=33.7470,
        longitude=73.1370,
        altitude=550.0,
        heading=0.0,
        timestamp="2026-07-01T00:00:00+00:00",
        width=640,
        height=480,
        extra={
            "description": "test point",
            "source_type": "gazebo_camera",
            "capture_status": "captured_gazebo",
            "manual_capture_required": False,
            "source_topic": "/vns_drone/downward_camera/image_raw",
            "ground_truth_topic": "/vns_drone/ground_truth",
        },
    )
    config = {
        "database": {
            "name": "Test",
            "bounds": {
                "min_latitude": 33.740,
                "max_latitude": 33.755,
                "min_longitude": 73.130,
                "max_longitude": 73.145,
            },
        }
    }

    index_path = riu.generate_database_index(
        [captured_image], output_dir, config, database_source_type="gazebo_camera"
    )
    index_document = yaml.safe_load(index_path.read_text(encoding="utf-8"))

    assert index_document["database"]["source_type"] == "gazebo_camera"
    entry = index_document["images"][0]
    assert entry["source_type"] == "gazebo_camera"
    assert entry["capture_status"] == "captured_gazebo"
    assert entry["manual_capture_required"] is False
    assert entry["source_topic"] == "/vns_drone/downward_camera/image_raw"
    assert entry["ground_truth_topic"] == "/vns_drone/ground_truth"


def test_resolve_cli_path_uses_cwd_for_explicit_repo_relative_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)

    resolved = cri._resolve_cli_path(
        "simulation/database/images/gazebo",
        script_dir=SCRIPTS,
        default_value="../database/images/gazebo",
    )

    assert resolved == (REPO_ROOT / "simulation" / "database" / "images" / "gazebo").resolve()


def test_resolve_cli_path_keeps_default_paths_relative_to_script_dir() -> None:
    resolved = cri._resolve_cli_path(
        "../database/images/gazebo",
        script_dir=SCRIPTS,
        default_value="../database/images/gazebo",
    )

    assert resolved == (SCRIPTS / "../database/images/gazebo").resolve()
