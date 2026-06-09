---
description: Launch the complete GNSS-denied VNS simulation stack (Gazebo Classic 11 + PX4 SITL + ROS 2 Humble + VNS node). Use when the user asks to run, start, or launch the simulation, or to test GNSS-denied visual navigation.
---

# Run VNS Simulation

## Purpose

Start the full GNSS-denied navigation test stack for the QAU Campus scenario.  Components run in **separate terminals**: gzserver (physics), gzclient (GUI), PX4 SITL, drone spawn, and the VNS node.

> **Critical rules (enforced by hooks):**
> - NEVER use `gz sim` — use `gzserver` + `gzclient` separately
> - NEVER use `gz_x500`, `gz_iris`, or any `gz_*` PX4 target — use `none_iris`
> - NEVER use `ros_gz_bridge` — sensors are bridged by SDF plugins inside `model.sdf`
> - NEVER set `GZ_SIM_RESOURCE_PATH` — use `GAZEBO_MODEL_PATH`

---

## Prerequisites (every terminal, every session)

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models
```

---

## Terminal 1 — Gazebo Classic Physics Server

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models

gzserver \
  /home/hp/GPS-DENIED-SYSTEM/simulation/worlds/vns_test_world.sdf \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so \
  --verbose
```

Wait for: `Gazebo multi-robot simulator, version 11.x` and `[Msg] Successfully loaded world`.

Binaries: `/usr/bin/gzserver`  (physics) and `/usr/bin/gzclient` (GUI)

---

## Terminal 2 — Gazebo GUI Client

```bash
gzclient --verbose
```

---

## Terminal 3 — PX4 SITL (Gazebo Classic target)

```bash
cd /home/hp/PX4-Autopilot
HEADLESS=1 make px4_sitl_default none_iris
```

> `none_iris` = Gazebo Classic compatible target.  
> `gz_iris` / `gz_x500` would require Gazebo Harmonic (NOT installed).

PX4 binary: `/home/hp/PX4-Autopilot/build/px4_sitl_default/bin/px4`

MAVLink ports after launch:
- **UDP 14540** → VNS offboard / MAVSDK
- **UDP 14550** → QGroundControl

---

## Terminal 4 — Spawn Drone Model

Wait ~4 s after gzserver starts, then:

```bash
source /opt/ros/humble/setup.bash

ros2 run gazebo_ros spawn_entity.py \
  -file /home/hp/GPS-DENIED-SYSTEM/simulation/models/uav_model/model.sdf \
  -entity vns_drone
```

The spawn service (`/spawn_entity`) is provided by `libgazebo_ros_factory.so` loaded in Terminal 1.

---

## Terminal 5 — VNS Node (GNSS-denied mode)

### Option A — UAV simulation only (no PX4 in this launch file)

```bash
source /opt/ros/humble/setup.bash

ros2 launch /home/hp/GPS-DENIED-SYSTEM/simulation/launch/UAV_simulation.launch.py \
  gps_enabled:=false
```

### Option B — Full stack (Gazebo + PX4 + VNS all-in-one)

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models

ros2 launch /home/hp/GPS-DENIED-SYSTEM/simulation/launch/full_simulation.launch.py \
  gps_enabled:=false
```

### Option C — Headless (CI / no GUI)

```bash
ros2 launch /home/hp/GPS-DENIED-SYSTEM/simulation/launch/full_simulation.launch.py \
  gps_enabled:=false headless:=true
```

---

## Verify Topics

```bash
ros2 topic list | grep vns_drone
```

Expected topics (published by **SDF plugins in model.sdf**, NOT by ros_gz_bridge nodes):

| Topic | Message Type | Source SDF Plugin |
|-------|-------------|-------------------|
| `/vns_drone/camera` | `sensor_msgs/Image` | `libgazebo_ros_camera.so` |
| `/vns_drone/imu` | `sensor_msgs/Imu` | `libgazebo_ros_imu_sensor.so` |
| `/vns_drone/gps` | `sensor_msgs/NavSatFix` | `libgazebo_ros_gps_sensor.so` |
| `/vns_drone/ground_truth` | `nav_msgs/Odometry` | `libgazebo_ros_p3d.so` |

VNS output topics:
- `/vns/vision_pose` — estimated position from visual matching
- `/vns/status` — current navigation mode

---

## Launch File Reference

| File | Purpose |
|------|---------|
| `simulation/launch/full_simulation.launch.py` | Gazebo Classic + PX4 SITL + VNS (complete stack) |
| `simulation/launch/UAV_simulation.launch.py` | Gazebo Classic + VNS only (no PX4) |
| `simulation/launch/px4_sitl.launch.py` | PX4 SITL only (pair with running Gazebo) |

---

## Key Configuration Files

| File | Key Setting |
|------|------------|
| `simulation/config/simulation.yaml` | camera intrinsics, MAVLink port, database path |
| `simulation/config/simulation.yaml` → `mavlink.connection_string` | `"udp://:14551"` |
| `simulation/database/qau_campus.vnsdb` | must exist before launching VNS node |

Build the database first with `/vns:build-reference-database` if it is missing.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `[gzserver] error: world file not found` | Wrong world path | Check `simulation/worlds/vns_test_world.sdf` exists |
| Topics not appearing | SDF plugins missing from model.sdf | Add `libgazebo_ros_*` plugin blocks to model.sdf |
| `spawn_entity: service /spawn_entity not available` | gzserver not ready | Wait longer; ensure `-s libgazebo_ros_factory.so` flag |
| PX4 exits immediately | Gazebo not running | Start gzserver (Terminal 1) before PX4 (Terminal 3) |
| VNS node: `database not found` | qau_campus.vnsdb missing | Run `/vns:build-reference-database` first |
