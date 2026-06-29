---
description: Quick-reference card for the VNS development environment — binaries, paths, env vars, ROS topics, and hard rules. Load when asked about the environment, when a binary is not found, or before any simulation or build task.
---

# VNS Environment Reference Card

## System

| Item | Value |
|------|-------|
| OS | Ubuntu 22.04 LTS |
| User | `hp` |
| Home | `/home/hp` |
| Shell | bash |

---

## Binaries — EXISTS vs NOT EXISTS

| Binary | Path | Status | Notes |
|--------|------|--------|-------|
| `gz` | `/usr/bin/gz` | ⚠️ EXISTS but WRONG | `gz sim` is Harmonic — **blocked** |
| `gzserver` | `/usr/bin/gzserver` | ✅ CORRECT | Gazebo Classic 11 physics server |
| `gzclient` | `/usr/bin/gzclient` | ✅ CORRECT | Gazebo Classic 11 GUI |
| `ros2` | `/opt/ros/humble/bin/ros2` | ✅ | ROS_DISTRO=humble |
| `python3` | `/usr/bin/python3` | ✅ | Python 3.10.12 — **always use this** |
| `python` | — | ❌ NOT INSTALLED | Any `python ` command will fail |
| `pip3` | `/home/hp/.local/bin/pip3` | ✅ | Targets python3.10 — **always use this** |
| `pip` | `/home/hp/.local/bin/pip` | ⚠️ EXISTS but WRONG | May target wrong env — **blocked** |
| `px4` | `/home/hp/PX4-Autopilot/build/px4_sitl_default/bin/px4` | ✅ built | Via `make px4_sitl_default none_iris` |

---

## ROS 2

| Item | Value |
|------|-------|
| Distro | humble |
| Setup script | `/opt/ros/humble/setup.bash` |
| Source command | `source /opt/ros/humble/setup.bash` |

---

## Gazebo Classic 11

| Item | Value |
|------|-------|
| Version | **Classic 11** (NOT Gazebo Harmonic) |
| Server binary | `/usr/bin/gzserver` |
| GUI binary | `/usr/bin/gzclient` |
| Server launch flags | `-s libgazebo_ros_init.so -s libgazebo_ros_factory.so` |
| Model path variable | `GAZEBO_MODEL_PATH` (NOT `GZ_SIM_RESOURCE_PATH`) |
| GAZEBO_MODEL_PATH | ⚠️ **UNSET** — must export before any launch |
| Correct model path | `/home/hp/GPS-DENIED-SYSTEM/simulation/models` |
| Sensor bridge | SDF `<plugin>` blocks inside `model.sdf` — NOT `ros_gz_bridge` nodes |

---

## PX4

| Item | Value |
|------|-------|
| Source dir | `/home/hp/PX4-Autopilot` |
| Correct SITL target | `make px4_sitl_default none_iris` |
| Blocked targets | `gz_*` — require Gazebo Harmonic |
| MAVLink VNS port | UDP 14540 |
| MAVLink QGC port | UDP 14550 |
| Connection string | `udp://:14551` (VNS → PX4) |

---

## Project Directory Map

| Path | Contents |
|------|----------|
| `/home/hp/GPS-DENIED-SYSTEM/` | Project root |
| `simulation/models/` | Gazebo UAV/world models |
| `simulation/worlds/` | World SDF files (uav_test_world.sdf full campus; uav_localization_test.sdf map-only + static cam) |
| `simulation/scripts/` | build_reference_database.py, capture_reference_images.py, generate_accuracy_report.py, visual_navigation.py, test_run.py, cli.py |
| `simulation/database/` | qau_campus.vnsdb, images/, qau_reference_metadata.yaml |
| `simulation/launch/` | full_simulation.launch.py, UAV_simulation.launch.py, px4_sitl.launch.py |
| `simulation/config/` | simulation.yaml |
| `src/vns/` | Main Python package |
| `src/vns/core/` | Kalman filter, fusion, dead-reckoning |
| `src/vns/vision/` | Feature extraction, matching |
| `src/vns/interfaces/` | MAVLink, ROS 2 interfaces |
| `src/vns/database/` | Reference DB access |
| `src/vns/config/` | Config management |
| `src/vns/utils/` | Utilities |
| `src/vns/validation/` | Accuracy reporting |

---

## ROS 2 Topics

| Topic | Message Type | Source |
|-------|-------------|--------|
| `/vns_drone/downward_camera/image_raw` | `sensor_msgs/Image` | `libgazebo_ros_camera.so` in model.sdf |
| `/vns_drone/imu` | `sensor_msgs/Imu` | `libgazebo_ros_imu_sensor.so` (inside `<sensor type="imu">`) |
| `/vns_drone/gps_raw` → `/vns_drone/gps` | `sensor_msgs/NavSatFix` | `libgazebo_ros_gps_sensor.so` (gated by gps_gate_node) |
| `/vns_drone/ground_truth` | `nav_msgs/Odometry` | `libgazebo_ros_p3d.so` (remap `odom:=ground_truth`) |
| `/vns/vision_pose` | `geometry_msgs/PoseStamped` | VNS node output |
| `/vns/status` | (custom) | VNS node output |

---

## Geographic Reference (QAU Campus, Islamabad)

| Item | Value |
|------|-------|
| Origin latitude | 33.7470 |
| Origin longitude | 73.1370 |
| Origin altitude | 550.0 m MSL |
| Match radius | 0.001 deg (~100 m) |

---

## Standard Environment Setup (paste before any sim/build work)

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models
```

---

## ⛔ Hard Rules (all enforced by plugin hooks)

| # | Rule | Correct alternative |
|---|------|---------------------|
| 1 | Never `gz sim` | `gzserver <world.sdf> -s libgazebo_ros_init.so -s libgazebo_ros_factory.so` |
| 2 | Never `ros_gz_bridge` / `ros_gz_image` | gazebo_ros SDF plugins in model.sdf |
| 3 | Never set `GZ_SIM_RESOURCE_PATH` | `export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models` |
| 4 | Never `make px4_sitl_default gz_*` | `cd /home/hp/PX4-Autopilot && make px4_sitl_default none_iris` |
| 5 | Never bare `pip install` | `pip3 install <pkg>` |
| 6 | Never bare `python ` | `python3 <script.py>` |
| 7 | Never `/home/ubuntu/` paths | Use `/home/hp/` as base |
| 8 | Never `/opt/px4/` | PX4 is at `/home/hp/PX4-Autopilot` |
