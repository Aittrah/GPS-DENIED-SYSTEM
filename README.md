# GNSS-Free Navigation System for UAVs

A desktop application for mission planning and GNSS-denied UAV navigation using visual localization.  
Built as Final Year Project at IIT, Quaid-i-Azam University Islamabad (2022–2026).

**Supervisor:** Dr. Bushra Almas  
**Team:** Muhammad Ahsan, Aittrah Sardar

---

## What This Project Does

This system navigates a UAV without GPS by comparing the drone camera feed against a pre-built
database of satellite imagery. The desktop app lets you:

- Load satellite maps and plan waypoint missions
- Build a visual reference database from satellite imagery
- Run GNSS-denied navigation using ORB feature matching and FAISS retrieval
- Monitor live position on an interactive map

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- ROS2 Humble *(only for simulation — not needed to run the desktop app)*

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Aittrah/GPS-DENIED-SYSTEM.git
cd GPS-DENIED-SYSTEM
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` is optional. If not installed, the FAISS retrieval backend
> is disabled but the app still runs using the SQLite-based database.

### 4. Run the Desktop App

```bash
python -m src.vns.desktop_app
```

Or:

```bash
python src/vns/desktop_app.py
```

The app opens a Tkinter window with:
- A map panel for loading satellite imagery and planning routes
- Waypoint management tools
- KML import/export
- Navigation status display

---

## Project Structure

```
GPS-DENIED-SYSTEM/
├── src/
│   └── vns/
│       ├── desktop_app.py             # Main entry point — run this
│       ├── map_widget.py              # Interactive map widget
│       ├── core/                      # Shared dataclasses and coordinate utils
│       │   ├── models.py              # Inter-module data contracts
│       │   └── geo_utils.py           # WGS84 <-> NED conversions
│       ├── database/                  # Visual reference databases
│       │   ├── reference_database.py  # SQLite-backed database (used by app)
│       │   ├── faiss_database.py      # FAISS vector-index retrieval
│       │   └── reference_db.py        # Safe-archive format database
│       ├── preprocessing/             # Image preprocessing pipeline
│       ├── perception/                # Feature extraction (ORB / DINOv2)
│       ├── navigation/                # Path planning and waypoint management
│       ├── kml/                       # KML file import/export
│       ├── map/                       # Map tile rendering
│       ├── mpvi/                      # Map-patch visual index
│       ├── state_estimation/          # Position state estimator
│       ├── control/                   # Flight control interface
│       ├── interfaces/                # MAVLink interface
│       ├── vision/                    # BoVW retrieval pipeline
│       ├── config/                    # Configuration management
│       ├── utils/                     # Shared utilities
│       └── validation/                # Accuracy reporting
├── tests/
│   ├── unit/                          # Unit tests (core, database, preprocessing)
│   └── fixtures/                      # Test images and helpers
├── simulation/                        # ROS2 / Gazebo simulation stack
├── data/                              # Runtime data (databases, patches, maps)
├── requirements.txt
└── pyproject.toml
```

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src/vns tests/
```

---

## Simulation (ROS2 / Gazebo)

Requires **ROS2 Humble** and **Gazebo Classic 11**.

```bash
source /opt/ros/humble/setup.bash
pip3 install "numpy<2"
# If pytest/import checks complain about a missing parser dependency:
# pip3 install lark
rosdep install --from-paths . --ignore-src -r -y --rosdistro humble
colcon build --packages-select vns --symlink-install
source install/setup.bash

# Preflight the ROS Python environment before starting the camera path
python3 simulation/scripts/check_ros2_python_env.py

# Deterministic visual-localization smoke test (no PX4)
ros2 launch vns localization_test.launch.py headless:=true

# Launch with GPS enabled
ros2 launch vns full_simulation.launch.py headless:=true

# Launch in GNSS-denied mode
ros2 launch vns full_simulation.launch.py headless:=true gps_enabled:=false
```

The simulation config now targets PX4 SITL's onboard-computer MAVSDK port
`udp://:14540`. Port `14550` remains reserved for QGroundControl / GCS traffic.

For the real Gazebo reference-image capture and database rebuild workflow, see
`docs/database_creation.md`. The committed `simulation/database/images/` and
`simulation/database/qau_campus.vnsdb` artifacts remain the explicit synthetic
baseline until you run the Gazebo capture flow locally.

---

## License

MIT License — see LICENSE file for details.
