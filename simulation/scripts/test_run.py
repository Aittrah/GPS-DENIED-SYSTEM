#!/usr/bin/env python3
"""Offline VNS flight-evaluation harness for the canonical runtime.

Drives the packaged ``VnsRuntime`` (``vns_node.py`` -> ``VnsRuntime`` ->
``VisualLocalizer``) directly — no ROS required — over a scripted 30-step
trajectory that begins with a healthy GNSS fix and then goes GNSS-denied,
writing the same JSONL evaluation log the ROS node produces.

Replaces the retired ``simulation/scripts/visual_navigation.py`` parallel
runtime; see ``docs/runtime_consolidation.md``.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

# Add src and root to system path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from vns.config import ConfigManager
from vns.core.runtime import VnsRuntime
from vns.database.reference_db import ReferenceDatabase
from vns.validation.jsonl_logger import GroundTruthSample, JsonlEvaluationLogger


def _resolve_log_dir(config: ConfigManager) -> Path:
    log_dir = config.data.get("logging", {}).get("log_dir", "./logs")
    path = Path(log_dir)
    return path if path.is_absolute() else (root_dir / path).resolve()


def main() -> None:
    print("Starting simulated VNS flight evaluation (canonical runtime)...")

    config_path = root_dir / "simulation" / "config" / "simulation.yaml"
    database_path = root_dir / "simulation" / "database" / "qau_campus.vnsdb"

    config = ConfigManager.from_yaml(str(config_path))

    # Load a reference image from the current database so feature matching is
    # guaranteed to succeed for the scripted offline evaluation run.
    try:
        db = ReferenceDatabase.load(str(database_path))
    except ValueError as exc:
        print(
            f"{exc}\nMigrate the database with: "
            f"vns database migrate --input {database_path} "
            f"--output {database_path}"
        )
        sys.exit(1)

    runtime = VnsRuntime(config, database=db)
    gt_logger = JsonlEvaluationLogger(enabled=True, log_dir=_resolve_log_dir(config))

    first_entry = list(db.entries.values())[0]

    import cv2

    image = cv2.imread(str(db.resolve_source_path(first_entry.source_path)))
    if image is None:
        # Fallback to synthetic if the image can't be read.
        image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Hold a constant heading over the trajectory (centred on the reference).
    yaw = math.radians(first_entry.heading)
    runtime.update_heading_from_quaternion(
        w=math.cos(yaw / 2.0), x=0.0, y=0.0, z=math.sin(yaw / 2.0)
    )

    retrieval_mode = config.data.get("retrieval", {}).get("mode", "flann")
    print(
        f"Running 30-step flight simulation using reference image "
        f"{first_entry.id} (retrieval={retrieval_mode})..."
    )

    # Step 0-9: GPS healthy. Step 10-29: GPS denied (fix lost).
    for step in range(30):
        healthy = step < 10
        runtime.update_gps(
            latitude=first_entry.latitude,
            longitude=first_entry.longitude,
            altitude=first_entry.altitude,
            has_fix=healthy,
            num_satellites=10 if healthy else 0,
            hdop=1.0 if healthy else 99.0,
        )
        runtime.update_ground_truth_pose(
            latitude=first_entry.latitude,
            longitude=first_entry.longitude,
            altitude=first_entry.altitude,
            heading_deg=first_entry.heading,
        )

        result = runtime.process_frame(image)

        gt_logger.write_frame(
            timestamp=None,
            ros_time=None,
            result=result,
            ground_truth=GroundTruthSample(
                latitude=first_entry.latitude,
                longitude=first_entry.longitude,
                altitude=first_entry.altitude,
                heading_deg=first_entry.heading,
                timestamp=time.time(),
            ),
            gnss_status=runtime.gnss_monitor.state.value,
            record_without_ground_truth=False,
        )

        pose = result.blended_pose
        lat, lon, alt = pose if pose is not None else (float("nan"),) * 3
        gnss_label = "OK" if healthy else "DENIED"
        print(
            f"Step {step:02d} | GNSS: {gnss_label:<6} | "
            f"Mode: {result.navigation_mode:<12} | "
            f"Lat: {lat:.6f} | Lon: {lon:.6f} | Alt: {alt:.1f}"
        )
        time.sleep(0.02)

    gt_logger.close()
    print("\nSimulated VNS flight complete.")
    print(f"Evaluation logs written to: {gt_logger.log_path}")


if __name__ == "__main__":
    main()
