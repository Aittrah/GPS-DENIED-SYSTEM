# Reference Database Creation Guide

This guide defines the current FR-11 database workflow for two different cases:

- the **synthetic baseline** used by offline/unit-style tests,
- the **real Gazebo downward-camera capture** path required for meaningful end-to-end PX4/Gazebo validation.

The committed `simulation/database/images/` and `simulation/database/qau_campus.vnsdb`
artifacts are currently the explicit **synthetic baseline**. They are labeled
`source_type: synthetic` and should not be mistaken for live Gazebo camera captures.

---

## 1. ROS2 / Python Preflight

Run this in the Ubuntu 22.04 ROS 2 Humble environment that will launch `vns_node`:

```bash
source /opt/ros/humble/setup.bash
pip3 install "numpy<2"
# Only if pytest/import validation reports it missing:
# pip3 install lark
python3 simulation/scripts/check_ros2_python_env.py
```

If the preflight reports a NumPy / `cv_bridge` incompatibility, fix that before
starting Gazebo. The ROS node now fails early instead of crashing later in the
camera callback with `_ARRAY_API not found`.

---

## 2. Synthetic Baseline

Use the synthetic generator only for offline tests and the ground-texture
workflow. It does **not** produce real Gazebo camera imagery.

```bash
python3 simulation/scripts/generate_synthetic_reference_images.py \
  --config simulation/database/qau_reference_metadata.yaml \
  --output simulation/database/images/synthetic
```

The generated `database_index.yaml` is tagged `source_type: synthetic`. If you
build a database from it, the safe archive preserves that provenance so runtime
and tests can warn when a synthetic DB is being used for a live simulation run.

---

## 3. Real Gazebo Camera Capture

This is the workflow needed for the first meaningful end-to-end VNS simulation.
The capture script subscribes to:

- `/vns_drone/downward_camera/image_raw`
- `/vns_drone/ground_truth`

It stores JPG frames under `simulation/database/images/gazebo/` and writes a
manifest with the existing `database_index.yaml` contract plus provenance fields
such as `source_type`, `source_topic`, `ground_truth_topic`, and
`capture_altitude_msl`.

### 3.1 Start the stack

```bash
ros2 launch vns full_simulation.launch.py headless:=true
```

Expected startup logs include:

- `MAVLink enabled; connection string=udp://:14540`
- `Connected to flight controller on udp://:14540`
- later, after localization starts succeeding:
  `VISION_POSITION_ESTIMATE send path active on udp://:14540`

If you want to validate the visual pipeline before PX4 is involved, use:

```bash
ros2 launch vns localization_test.launch.py headless:=true
```

### 3.2 Capture the reference set

Manually position or fly the drone to each reference point from
`simulation/database/qau_reference_metadata.yaml`, then run:

```bash
python3 simulation/scripts/capture_reference_images.py \
  --config simulation/database/qau_reference_metadata.yaml \
  --simulation-config simulation/config/simulation.yaml \
  --output simulation/database/images/gazebo \
  --camera-topic /vns_drone/downward_camera/image_raw \
  --ground-truth-topic /vns_drone/ground_truth \
  --position-tolerance-m 3.0 \
  --heading-tolerance-deg 20.0
```

For an automated flow once the drone is already being moved to the targets:

```bash
python3 simulation/scripts/capture_reference_images.py \
  --config simulation/database/qau_reference_metadata.yaml \
  --simulation-config simulation/config/simulation.yaml \
  --output simulation/database/images/gazebo \
  --auto-capture
```

By default the script refuses to overwrite an existing capture directory. Use
`--overwrite` only when you intentionally want to replace an earlier run.

The script validates its environment before capturing anything: it prints the
resolved config paths, output directory, topics, and reference-point count on
startup; it confirms `--camera-topic` and `--ground-truth-topic` both have an
active publisher before waiting for data; and `--timeout-sec` (default `30.0`)
bounds both that publisher check and the wait for the first camera/ground-truth
sample. If a required topic has no publisher, or no sample arrives in time, the
script exits non-zero with a message naming the missing topic and a hint to run
`ros2 topic list` and confirm the simulation launch file is running. At the end
it always prints how many images were captured and the path to the written
`database_index.yaml`.

### 3.3 Troubleshooting

- **Script exits immediately with no output**: you are running a stale copy
  predating the FR-11 entrypoint fix. The file must end with
  `if __name__ == "__main__": raise SystemExit(main())`; pull the latest
  `simulation/scripts/capture_reference_images.py`.
- **"No publisher detected on required topic ..."**: the simulation stack
  isn't running yet, or the topic name doesn't match. Run `ros2 topic list`
  and confirm `ros2 launch vns full_simulation.launch.py headless:=true` (or
  `localization_test.launch.py`) is up.
- **"Timed out ... waiting for the first camera frame ..."**: the topic has a
  publisher but nothing has been received yet (e.g. simulation paused).
  Increase `--timeout-sec` or unpause Gazebo.

---

## 4. Build Or Rebuild `qau_campus.vnsdb`

After capturing real Gazebo frames, rebuild the active simulation database:

```bash
python3 simulation/scripts/build_reference_database.py \
  --input simulation/database/images/gazebo/database_index.yaml \
  --output simulation/database/qau_campus.vnsdb \
  --build-vocab
```

The active full-simulation config reads `simulation/database/qau_campus.vnsdb`,
so rebuilding that file is the step that switches the live run away from the
synthetic baseline.

For imported external real imagery, the same builder still works:

```bash
python3 simulation/scripts/import_real_reference_imagery.py \
  --manifest /path/to/real_manifest.yaml \
  --image-root /path/to/source_images \
  --output simulation/database/imported/database_index.yaml \
  --copy-images-to simulation/database/imported/images

python3 simulation/scripts/build_reference_database.py \
  --input simulation/database/imported/database_index.yaml \
  --output simulation/database/imported/qau_real.vnsdb \
  --build-vocab
```

---

## 5. Inspect And Validate

Inspect the rebuilt database:

```bash
vns database inspect simulation/database/qau_campus.vnsdb
```

Recommended checks:

- `python3 -m pytest tests/ -q`
- `python3 simulation/scripts/test_run.py`
- confirm the rebuilt DB entries are no longer tagged `source_type: synthetic`
- confirm the live full simulation no longer logs the synthetic DB warning

For GNSS-denied end-to-end validation, look for logs like:

- `GNSS` state transitions from healthy to denied
- `Mode: VISION` or `Mode: DR` in runtime/evaluation output
- successful reference-image matches instead of `no_geometric_match`
- `VISION_POSITION_ESTIMATE send path active on udp://:14540`
