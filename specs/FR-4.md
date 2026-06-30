# FR-4: UAV Sensor Data Acquisition

## Requirement
> The system shall acquire real-time sensor data from the UAV: camera imagery,
> IMU measurements, and GPS fix (when available), via the MAVLink interface and
> Gazebo Classic SDF plugins.

## Rationale
Every downstream stage — visual localisation, dead reckoning, and sensor fusion
— depends on a continuous, recoverable stream of UAV sensor data. In the Gazebo
Classic simulation the downward camera, IMU, and GPS are published directly by
`gazebo_ros` plugins inside the drone SDF and consumed by the packaged ROS node.
A regression in the launch environment silently broke this: the `type='camera'`
sensor rendered **zero frames** because the Gazebo Classic share directory
(`/usr/share/gazebo-11`, which holds the RTShaderSystem shader libs) was never
placed on `GAZEBO_RESOURCE_PATH` unless the user manually sourced
`/usr/share/gazebo/setup.sh` first. The same gap rendered the ground plane flat
grey instead of the textured `QAUCampus/Ground` material. This requirement
captures correct sensor acquisition and the means to verify and recover the
acquired streams from a completed run.

## Acceptance Criteria
1. The downward-camera topic `/vns_drone/downward_camera/image_raw` publishes at
   **≥ 25 Hz** with non-trivial (non-flat) image content — a sampled frame's
   pixel standard deviation is above a small threshold, confirming the shaders
   and ground texture resolve. **[Met]**
2. The IMU topic `/vns_drone/imu` publishes at its configured rate during a run.
   **[Met]**
3. Both streams are recoverable from a completed simulation run: camera + IMU
   (+ GPS + ground-truth) are captured in the `vns_simulation_bag/` rosbag when
   launched with `record_bag:=true`, and downward-camera frames are additionally
   saved as PNGs with a JSONL manifest under `logs/images/` when
   `logging.log_images` is enabled. **[Met]**
4. Camera/shader and base-model rendering works without manually sourcing
   `/usr/share/gazebo/setup.sh` — the launch files compose
   `GAZEBO_RESOURCE_PATH` / `GAZEBO_MODEL_PATH` themselves via a shared,
   version-robust helper applied consistently across all launch entrypoints.
   **[Met]**

## Dependencies
None

## Linked Source Files
- `simulation/models/iris_downward_cam/model.sdf` — camera/IMU/GPS gazebo_ros sensor plugins and topic names.
- `src/vns/utils/paths.py` — `detect_gazebo_share_dir` / `compose_gazebo_resource_path` / `compose_gazebo_model_path` shared launch env helpers.
- `simulation/launch/full_simulation.launch.py` — composes the gazebo env, spawns the drone, and (optionally) records the rosbag.
- `simulation/launch/UAV_simulation.launch.py`, `simulation/launch/localization_test.launch.py`, `simulation/launch/px4_sitl.launch.py` — same shared env helper applied consistently.
- `src/vns/core/vns_node.py` — camera / IMU / GPS / ground-truth subscribers and the image-logging hook.
- `src/vns/validation/image_logger.py` — `ImageFrameLogger` that persists downward-camera frames + manifest under `logs/images/`.
- `simulation/config/simulation.yaml` — sensor topic names and `logging.log_images` / `image_log_stride`.

## Test Plan

### Unit Tests
- `tests/test_paths_gazebo.py` — `detect_gazebo_share_dir` version selection /
  fallback / absence, and `compose_gazebo_resource_path` /
  `compose_gazebo_model_path` ordering.
- `tests/test_image_logger.py` — `ImageFrameLogger` enabled/disabled behavior,
  stride decimation, PNG + manifest output, idempotent close.

### Integration Tests
- `tests/test_portability.py::test_launch_and_world_assets_are_portable` —
  launch files keep `GAZEBO_RESOURCE_PATH` and contain no absolute `/home/`
  paths after the helper refactor.

### Simulation Tests
- Launch `full_simulation.launch.py` headless with `record_bag:=true`
  **without** sourcing `setup.sh` first; confirm the camera publishes ≥ 25 Hz
  with non-flat frames, the IMU publishes at its rate, and both are recoverable
  from the rosbag and (camera) from `logs/images/`.

## Status
- [ ] Not Started
- [ ] In Progress
- [x] Done

**Implementation evidence:** `GAZEBO_RESOURCE_PATH` / `GAZEBO_MODEL_PATH` are now
composed inside the launch files via the shared, version-robust helpers in
`src/vns/utils/paths.py` (`detect_gazebo_share_dir`,
`compose_gazebo_resource_path`, `compose_gazebo_model_path`), applied
consistently across `full_simulation`, `UAV_simulation`, `localization_test`,
and `px4_sitl`. The `logging.log_images` flag is implemented by
`src/vns/validation/image_logger.py` (`ImageFrameLogger`), wired into
`src/vns/core/vns_node.py`. Unit coverage: `tests/test_paths_gazebo.py`,
`tests/test_image_logger.py` (full suite touching these modules: 43 passed).
Verified end-to-end through `full_simulation.launch.py` (`headless:=true
record_bag:=true`) launched **without** sourcing `/usr/share/gazebo/setup.sh`
and with `GAZEBO_RESOURCE_PATH`/`OGRE_RESOURCE_PATH` explicitly unset:
`/vns_drone/downward_camera/image_raw` published at **29.6 Hz** (5355 msgs over
177.8 s ≈ 30 Hz in the rosbag) with non-flat frames (sample pixel std 31.5–38.4,
ORB features up to 32) and **zero** "Unable to find shader" errors;
`/vns_drone/imu` published at **248.6 Hz** (44058 msgs). Both streams are
recoverable from the `vns_simulation_bag/` rosbag, and 5268 camera frames were
additionally saved as PNGs + a JSONL manifest under `logs/images/`.
