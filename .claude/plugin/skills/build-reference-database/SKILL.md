---
description: Build the qau_campus.vnsdb ORB feature database for visual navigation. Use when the user asks to build, rebuild, or regenerate the VNS reference database, or when database-related errors appear.
---

# Build VNS Reference Database

## Purpose

Build `qau_campus.vnsdb` — the ORB feature database used by the Visual Navigation System to localise the UAV against QAU Campus reference imagery.  Two scripts run in sequence: `capture_reference_images.py` produces the images, then `build_reference_database.py` extracts features and writes the binary database.

---

## Prerequisites (run before every session)

```bash
source /opt/ros/humble/setup.bash
export GAZEBO_MODEL_PATH=/home/hp/GPS-DENIED-SYSTEM/simulation/models
```

> Rules enforced by plugin hooks:
> - Use `python3` (never bare `python` — it is not installed)
> - Use `pip3 install` (never bare `pip install`)
> - Never set `GZ_SIM_RESOURCE_PATH` (wrong env var; use `GAZEBO_MODEL_PATH`)

---

## Step 1 — Install Dependencies (once)

```bash
pip3 install opencv-python pyyaml numpy
```

---

## Step 2 — Generate Synthetic Reference Images

```bash
cd /home/hp/GPS-DENIED-SYSTEM

python3 simulation/scripts/capture_reference_images.py \
    --synthetic \
    --config simulation/database/qau_reference_metadata.yaml \
    --output simulation/database/images \
    --width 640 \
    --height 480
```

**Produces:**
- `simulation/database/images/<point_id>.jpg` — one JPG per reference point
- `simulation/database/images/database_index.yaml` — image index consumed by Step 3

The `--synthetic` flag generates textured images offline (no live Gazebo required).  To capture from a running Gazebo simulation, omit `--synthetic` and ensure the camera topic `/vns_drone/downward_camera/image_raw` is publishing.

---

## Step 3 — Build the ORB Feature Database

```bash
cd /home/hp/GPS-DENIED-SYSTEM

python3 simulation/scripts/build_reference_database.py \
    --input  simulation/database/images/database_index.yaml \
    --output simulation/database/qau_campus.vnsdb \
    --max-features 500 \
    --algorithm ORB
```

**Produces:**
- `simulation/database/qau_campus.vnsdb` — serialised database (pickle)

---

## Step 4 — Verify Output

```bash
ls -lh /home/hp/GPS-DENIED-SYSTEM/simulation/database/qau_campus.vnsdb
python3 - <<'EOF'
import pickle, sys
with open("simulation/database/qau_campus.vnsdb", "rb") as f:
    db = pickle.load(f)
print(f"Version:  {db['version']}")
print(f"Entries:  {len(db['entries'])}")
print(f"Algorithm:{db['algorithm']}")
EOF
```

The file must exist, be non-empty (typically 1–20 MB), and `Entries` must be > 0.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `pip3 install opencv-python` |
| `ModuleNotFoundError: No module named 'yaml'` | PyYAML not installed | `pip3 install pyyaml` |
| `command not found: python` | bare `python` used | Replace with `python3` |
| `FileNotFoundError: qau_reference_metadata.yaml` | Wrong working directory | Run from `/home/hp/GPS-DENIED-SYSTEM` |
| `ValueError: No images found in index` | Step 2 not run first | Run capture step before build step |
| `cv2.error` during feature extraction | Corrupt or empty image | Re-run Step 2 to regenerate images |

---

## Reference Files

| File | Role |
|------|------|
| `simulation/database/qau_reference_metadata.yaml` | Reference point definitions (lat/lon/alt/heading) |
| `simulation/scripts/capture_reference_images.py` | Step 2 — image generation |
| `simulation/scripts/build_reference_database.py` | Step 3 — feature extraction and DB build |
| `simulation/database/qau_campus.vnsdb` | Output database |
| `simulation/database/images/database_index.yaml` | Intermediate image index |
| `simulation/config/simulation.yaml` | `database.path` must point to qau_campus.vnsdb |
