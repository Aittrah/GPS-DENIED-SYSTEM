# Running the GPS-DENIED-SYSTEM (VNS) end to end

This is the hands-on companion to [`scripts/run_project.sh`](../scripts/run_project.sh).
Every scenario below has a one-line command; the script encodes the environment
fixes this machine needs so you don't have to remember them.

```bash
cd /home/hp/GPS-DENIED-SYSTEM
./scripts/run_project.sh            # prints the command list
```

## Environments used

| Venv          | Python | What it's for                                             |
|---------------|--------|-----------------------------------------------------------|
| `.venv-test`  | 3.10   | pytest, desktop app, DB build (has faiss-cpu + rasterio)  |
| `.venv-run`   | 3.10   | sim-side helpers (numpy 1.26 = cv_bridge-safe, mavsdk)     |
| system `python3` | 3.10 | ROS nodes (rclpy/cv_bridge come from `/opt/ros/humble`)  |

> The interactive shell auto-sources ROS (`~/.bashrc`). That is required for the
> simulation but **breaks pytest** (ROS ships `launch_testing` pytest plugins that
> crash pytest 9). The script handles both cases; if you run pytest by hand, see
> the `test` note below.

---

## 1. Tests — `./scripts/run_project.sh test`

Runs the unit + integration suite in two passes:

```bash
env -u PYTHONPATH .venv-test/bin/python3 -m pytest tests/ -q          # main suite
/usr/bin/python3 -m pytest tests/test_vns_node.py \                   # rclpy-gated
    tests/test_capture_reference_images.py -q -p no:launch_testing -p no:launch_ros
```

Expect **all green** (≈272 tests). `env -u PYTHONPATH` drops ROS from the path so
its incompatible pytest plugins don't load; the second pass keeps ROS (for rclpy)
but disables the two bad plugins by name.

## 2. Desktop app — `./scripts/run_project.sh app`

Launches the tkinter mission planner (`main.py`). Needs an X display
(`DISPLAY=:0`). Load `data/kml/QAU Polygon.kml`, set start/goal, plan a path,
open Google Earth Pro at a location.

## 3. Build the reference database from REAL imagery — `./scripts/run_project.sh build-db`

This is what makes vision localization actually match. Three steps:

1. `build_satellite_ground_texture.py` — cleans `data/satellite/raw/qau_campus_satellite.jpg`
   into the Gazebo ground texture `qau_ground_plane/materials/textures/qau_satellite.png`
   (2048×1538, ~0.293 m/px, 600 m × 450.6 m footprint centred on the world origin).
2. `build_real_reference_tiles.py` — cuts 25 geo-referenced tiles (5×5) from that
   **same** texture, each matching the camera footprint (~277 m × 208 m at 240 m),
   with one tile centred exactly on the origin. Writes `simulation/database/images_real/`.
3. `build_reference_database.py` — extracts ORB + trains the BoVW vocabulary and
   writes `simulation/database/qau_campus.vnsdb`.

Because the reference tiles and the world texture are the **same** satellite
image, ORB geometric verification matches (unlike the old synthetic building
crops in `qau_campus.vnsdb.synthetic.bak`).

## 4. Vision-only world — `./scripts/run_project.sh loctest`

Brings up `localization_test.launch.py` headless: `gzserver` on the flat,
satellite-textured world + a static nadir camera at 240 m + `vns_node`. Watch the
log for:

```
[vns.vision.localizer] Localized: ref=qau_tile_r3_c2 conf=0.87 inliers=26 lat=33.746452 lon=73.137000
```

`/vns/vision_pose` publishes a **vision-derived** position (longitude is exact;
latitude lands within one tile row). This is the clean world designed to validate
vision — no PX4, no flight.

## 5. Full stack + mission + GNSS denial

Two terminals.

**Terminal A — bring up the stack:**
```bash
./scripts/run_project.sh sim
```
Wait for `[commander] Ready for takeoff!` (PX4 EKF2 needs ~60–90 s to converge;
trust that log line, not a stand-alone MAVSDK health check — `vns_node` already
owns MAVSDK port 14540).

**Terminal B — fly + toggle GNSS:**
```bash
./scripts/run_project.sh mission          # 70-wp QAU mission @240 m, on port 14550
./scripts/run_project.sh deny off         # gps_gate silences GPS -> vns_node: HEALTHY -> DENIED
./scripts/run_project.sh deny on          # DENIED -> HEALTHY
```

You'll see `GNSS state transition: HEALTHY -> DENIED` and the eval log record the
navigation-mode cascade `GPS -> DR -> FAILSAFE` while denied.

> **Vision during flight:** the full world (`uav_test_world`) has 3-D buildings and
> angled lighting, and the wide mission overflies the 600×450 m textured patch, so
> in-flight vision fixes are sparse there **by design** — `localization_test` is the
> world built to demonstrate vision. Real reference imagery covering the whole
> flight path (larger satellite crop) would extend vision coverage; that's the
> open follow-up (P4).

## 6. Validation dashboard — `./scripts/run_project.sh dashboard`

Builds `reports/dashboard/index.html` from the newest `logs/ground_truth_*.jsonl`
+ `logs/images/manifest_*.jsonl` + the `.vnsdb`. Run it after a `sim` session.

## 7. Cleanup — `./scripts/run_project.sh clean`

Kills any leftover `gzserver` / `px4` / `vns_node` / `mavsdk_server` and frees the
ports (11345, 14540, 14550). Run this if a launch failed to exit cleanly.

> **Disk note:** `logging.log_images: true` saves every camera frame to
> `logs/images/` (~4 GB per multi-minute run). `logs/` is git-ignored; clear it
> between long runs if disk gets tight.
