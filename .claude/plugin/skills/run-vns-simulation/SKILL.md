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

## Prerequisites

Build the `vns` ROS 2 package once (installs `vns_node` to `lib/vns/` so
`ros2 run`/`ros2 launch` can find it):

```bash
cd /home/hp/GPS-DENIED-SYSTEM
source /opt/ros/humble/setup.bash
colcon build --packages-select vns --symlink-install
```

Then, in **every terminal, every session**:

```bash
source /opt/ros/humble/setup.bash
source /home/hp/GPS-DENIED-SYSTEM/install/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models
```

---

## Quickest check — localization smoke-test (no PX4, one command)

The fastest way to confirm the visual localizer publishes a pose fix. It brings up
a map-only world (`uav_localization_test.sdf`) with a **static downward camera at
240 m** over the origin, plus the VNS node (run under `.venv-run`). No PX4, no
drone, no flight — the camera is fixed over the `grid_center` reference tile.

```bash
source /opt/ros/humble/setup.bash
source /home/hp/GPS-DENIED-SYSTEM/install/setup.bash

ros2 launch vns localization_test.launch.py headless:=true   # drop headless:=true for the GUI

# in another terminal:
source /opt/ros/humble/setup.bash
ros2 topic echo /vns/vision_pose --once
#   pose.position.y ≈ 33.7470 (lat), pose.position.x ≈ 73.1370 (lon)  -> success
```

To check the localizer with **no Gazebo at all** (deterministic, CI-friendly):

```bash
/home/hp/GPS-DENIED-SYSTEM/.venv-run/bin/python3 \
  /home/hp/GPS-DENIED-SYSTEM/simulation/scripts/simulate_camera_view.py
```

---

## Terminal 1 — Gazebo Classic Physics Server

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models

gzserver \
  /home/hp/GPS-DENIED-SYSTEM/simulation/worlds/uav_test_world.sdf \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so \
  --verbose
```

Wait for: `Gazebo multi-robot simulator, version 11.x` and `[Msg] Successfully loaded world`.

> World files: `uav_test_world.sdf` is the full QAU campus (buildings + textured
> ground). `uav_localization_test.sdf` is a map-only world with a static downward
> camera at 240 m over the origin — the simplest way to get a visual pose fix
> (no PX4, no drone, no free-fall). To test localization purely offline (no
> Gazebo) run `simulation/scripts/simulate_camera_view.py`.

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
  -file /home/hp/GPS-DENIED-SYSTEM/simulation/models/iris_downward_cam/model.sdf \
  -entity iris_downward_cam -x 0 -y 0 -z 1
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

> **Get the drone to altitude.** The VNS node only *localizes* — it never commands
> flight. After the full stack is up, the drone sits on the ground, where the
> downward camera is too low to match the reference tiles. Arm and climb to ~240 m
> (where the 60°-FOV camera footprint ≈ one 278 m grid tile) via QGroundControl,
> the PX4 `pxh>` shell (`commander takeoff`), or a MAVSDK script, e.g.:
>
> ```bash
> /home/hp/GPS-DENIED-SYSTEM/.venv-run/bin/python3 - <<'PY'
> import asyncio
> from mavsdk import System
> async def main():
>     d = System(); await d.connect(system_address="udp://:14540")
>     async for s in d.core.connection_state():
>         if s.is_connected: break
>     await d.action.set_takeoff_altitude(240.0)
>     await d.action.arm(); await d.action.takeoff()
>     await asyncio.sleep(60)
> asyncio.run(main())
> PY
> ```
>
> Over the full campus world the buildings occlude parts of the map, so matches
> are intermittent in flight. For a clean, repeatable pose fix use the
> localization smoke-test above (static camera, map-only world).

---

## Verify Topics

```bash
ros2 topic list | grep vns_drone
```

Expected topics (published by **SDF plugins in model.sdf**, NOT by ros_gz_bridge nodes):

| Topic | Message Type | Source SDF Plugin |
|-------|-------------|-------------------|
| `/vns_drone/downward_camera/image_raw` | `sensor_msgs/Image` | `libgazebo_ros_camera.so` |
| `/vns_drone/imu` | `sensor_msgs/Imu` | `libgazebo_ros_imu_sensor.so` (inside `<sensor type="imu">`) |
| `/vns_drone/gps_raw` → `/vns_drone/gps` | `sensor_msgs/NavSatFix` | `libgazebo_ros_gps_sensor.so` (gated by `gps_gate_node`) |
| `/vns_drone/ground_truth` | `nav_msgs/Odometry` | `libgazebo_ros_p3d.so` (remap `odom:=ground_truth`) |

VNS output topics:
- `/vns/vision_pose` — estimated position from visual matching
- `/vns/status` — current navigation mode

---

## Launch File Reference

| File | Purpose |
|------|---------|
| `simulation/launch/localization_test.launch.py` | Map-only world + static 240 m camera + VNS (no PX4) — quickest pose-fix check |
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
| `[gzserver] error: world file not found` | Wrong world path | Check `simulation/worlds/uav_test_world.sdf` exists |
| Topics not appearing | SDF plugins missing from model.sdf | Add `libgazebo_ros_*` plugin blocks to model.sdf |
| `spawn_entity: service /spawn_entity not available` | gzserver not ready | Wait longer; ensure `-s libgazebo_ros_factory.so` flag |
| PX4 exits immediately | Gazebo not running | Start gzserver (Terminal 1) before PX4 (Terminal 3) |
| VNS node: `database not found` | qau_campus.vnsdb missing | Run `/vns:build-reference-database` first |
