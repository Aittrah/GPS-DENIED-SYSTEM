markdown# GNSS-Free Navigation System for UAVs

A desktop application for mission planning and GNSS-denied UAV navigation.
Built as Final Year Project at IIT, Quaid-i-Azam University Islamabad (2022–2026).

**Supervisor:** Dr. Bushra Almas  
**Team:** Muhammad Ahsan, Aittrah Sardar

---

## What This Project Does

- Python 3.10+
- ROS2 Humble (for simulation)
- Gazebo Classic 11 (for simulation)
- PX4 Autopilot (for simulation)

## Quick Start

### 1. Python Environment and Tests

```bash
# Clone the repository
git clone <repository-url>
cd GPS-DENIED-SYSTEM

# Ubuntu may require this before `python3 -m venv` works
sudo apt install python3-venv

# Create the project virtual environment
python3 -m venv .venv-run
source .venv-run/bin/activate

# Install Python runtime dependencies and developer tooling
pip3 install -r requirements.txt
pip3 install -e ".[dev]"

# Run the Python test suite
python3 -m pytest tests/ -q
```

### Run the App

```bash
cp simulation/config/simulation.yaml simulation/config/my_config.yaml
```

Key configuration sections:
- `camera`: Camera intrinsics and mount position
- `gnss`: Thresholds for GPS degradation detection
- `navigation`: Failsafe and blending parameters
- `database`: Reference image database path

## Simulation Setup

### Prerequisites

1. **Install ROS2 Humble**: Follow [ROS2 installation guide](https://docs.ros.org/en/humble/Installation.html)

2. **Install Gazebo Classic 11 integration**:
   ```bash
   sudo apt install gazebo ros-humble-gazebo-ros-pkgs
   ```

3. **Install PX4 SITL**:
   ```bash
   git clone https://github.com/PX4/PX4-Autopilot.git --recursive
   cd PX4-Autopilot
   bash ./Tools/setup/ubuntu.sh
   make px4_sitl none_iris
   ```

4. **Install QGroundControl** (optional):
   Download from [QGroundControl](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html)

5. **Install ROS package dependencies and build the package**:
   ```bash
   source /opt/ros/humble/setup.bash
   rosdep install --from-paths . --ignore-src -r -y --rosdistro humble
   colcon build --packages-select vns --symlink-install
   source install/setup.bash
   ```

ROS libraries such as `rclpy`, `cv_bridge`, and `gazebo_ros` come from ROS 2
packages via `apt`/`rosdep`, not from `pip3`.

### Running the Simulation


1. **Start the full simulation stack**:
   ```bash
   source /opt/ros/humble/setup.bash
   source install/setup.bash

   # Launch simulation with GPS enabled
   ros2 launch vns full_simulation.launch.py

   # Or launch with GPS disabled (GNSS-denied mode)
   ros2 launch vns full_simulation.launch.py gps_enabled:=false
   ```

2. **Headless mode** (for CI/testing):
   ```bash
   ros2 launch vns full_simulation.launch.py headless:=true
   ```

3. **Record data for analysis**:
   ```bash
   ros2 launch vns full_simulation.launch.py record_bag:=true
   ```

### Testing GNSS-Denied Scenarios

The current runtime behavior is implemented by `simulation/scripts/gps_gate_node.py`.
It provides a simple on/off GPS gate for simulation:

| Mode | How to trigger | Implemented behavior |
|------|----------------|----------------------|
| GPS enabled | Default launch, or `ros2 service call /vns/set_gps_enabled std_srvs/srv/SetBool "{data: true}"` | Relays `/vns_drone/gps_raw` to `/vns_drone/gps` |
| GPS denied | `ros2 launch vns full_simulation.launch.py gps_enabled:=false`, or `ros2 service call /vns/set_gps_enabled std_srvs/srv/SetBool "{data: false}"` | Suppresses `/vns_drone/gps` output entirely |

`simulation/config/gps_control.yaml` still contains named scenario profiles such as
`complete_denial`, `degraded_signal`, `intermittent`, and `gradual_degradation`,
but those profiles are not currently implemented by the runtime GPS gate and should
be treated as future work rather than available simulation modes.

## Basic Usage

### Creating a Reference Database

```python
from vns.database import ReferenceDatabase

# Create new database
db = ReferenceDatabase()

# Add geotagged images
db.add_image("path/to/image1.jpg")
db.add_image("path/to/image2.jpg")

# Save database
db.save("my_area.vnsdb")
```

### Running The Localization Pipeline

```python
from vns.config import ConfigManager
from vns.database import ReferenceDatabase
from vns.vision.localizer import VisualLocalizer

# Load configuration
config = ConfigManager.load("simulation/config/simulation.yaml")

# Load reference database
db = ReferenceDatabase.load("simulation/database/qau_campus.vnsdb")

# Initialize localizer
localizer = VisualLocalizer(config.data, db)

# Run localization on a camera frame
result = localizer.localize(frame, altitude=580.0, heading_deg=0.0)
```

## Demo

```bash
# Validate configuration file
vns --config simulation/config/simulation.yaml

# Build a BoVW-enabled reference database from an image index YAML
vns database build --input ./images/database_index.yaml --output area.vnsdb

# Inspect database contents
vns database inspect area.vnsdb

# Migrate a trusted legacy pickle database to the safe archive format
vns database migrate --input legacy.vnsdb --output safe.vnsdb
```

Existing pickle-era `.vnsdb` files, including older local copies of
`simulation/database/qau_campus.vnsdb`, must be migrated before they can be
loaded by default.

## Project Structure

```
vns/
├── src/vns/              # Main package
│   ├── core/             # Core data models and logic
│   ├── config/           # Configuration management
│   ├── database/         # Reference image database
│   ├── interfaces/       # MAVLink and ROS2 interfaces
│   ├── vision/           # Feature extraction and matching
│   ├── validation/       # Accuracy reporting
│   └── utils/            # Utility functions
├── simulation/           # Simulation resources
│   ├── config/           # Simulation configurations
│   ├── launch/           # ROS2 launch files
│   ├── models/           # Gazebo drone models
│   ├── worlds/           # Gazebo world files
│   └── scripts/          # Database preparation tools
├── tests/                # Test suite
└── resource/             # ROS 2 package resource marker
```

## Tests

```bash
python3 -m pytest tests/ -q
python3 -m pytest tests/ --cov=src/vns
```

### Code Quality

```bash
# Format code
make format

# Lint
make lint

# Type check
make typecheck

# Run all checks
make check
```

### Developer Tooling

The development workflow is centered on:
- Black for formatting
- Ruff for linting
- MyPy for type checking

## Hardware Deployment

For deploying on actual drone hardware:

1. **Companion Computer**: Raspberry Pi 4 (4GB+) or Jetson Nano
2. **Flight Controller**: Any PX4-compatible (Pixhawk recommended)
3. **Camera**: Global shutter camera, 60+ FPS recommended

See `docs/hardware_deployment.md` for detailed setup instructions.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gnss.degraded_hdop` | 5.0 | HDOP threshold for degraded status |
| `gnss.degraded_satellites` | 4 | Minimum satellites for normal status |
| `gnss.denied_timeout_seconds` | 2.0 | Seconds without signal before denied |
| `navigation.uncertainty_failsafe_threshold` | 10.0 | Meters of uncertainty to trigger failsafe |
| `matching.confidence_threshold` | 0.6 | Minimum match confidence (0-1) |

## License

MIT License - see LICENSE file for details.

## Documentation

- [How to Use the App](docs/HOW_TO_USE.md)
- [SRS Document](docs/SRS.md)
- [SDS Document](docs/SDS.md)
