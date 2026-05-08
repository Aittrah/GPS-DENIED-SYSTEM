# Visual Navigation System (VNS)

A GNSS-denied navigation system for autonomous drones that uses camera imagery matched against pre-indexed reference images to estimate position when GPS is unavailable or degraded.

## Features

- **Automatic GNSS Monitoring**: Detects GPS degradation and denial in real-time
- **Visual Position Estimation**: Matches camera frames against geotagged reference database
- **Seamless Mode Switching**: Smooth transitions between GPS and visual navigation
- **Simulation Support**: Full integration with PX4 SITL + Gazebo
- **Hardware Ready**: Same codebase for simulation and real drone deployment

## Requirements

- Python 3.10+
- ROS2 Humble (for simulation)
- Gazebo Harmonic (for simulation)
- PX4 Autopilot (for simulation)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd vns

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in development mode
make install-dev

# Or install manually:
pip install -e ".[dev]"
```

### 2. Verify Installation

```bash
# Run tests
make test

# Check code quality
make check
```

### 3. Configuration

Copy and customize the configuration file:

```bash
cp simulation/config/simulation.yaml config/my_config.yaml
```

Key configuration sections:
- `camera`: Camera intrinsics and mount position
- `gnss`: Thresholds for GPS degradation detection
- `navigation`: Failsafe and blending parameters
- `database`: Reference image database path

## Simulation Setup

### Prerequisites

1. **Install ROS2 Humble**: Follow [ROS2 installation guide](https://docs.ros.org/en/humble/Installation.html)

2. **Install Gazebo Harmonic**:
   ```bash
   sudo apt install ros-humble-ros-gz
   ```

3. **Install PX4 SITL**:
   ```bash
   git clone https://github.com/PX4/PX4-Autopilot.git --recursive
   cd PX4-Autopilot
   bash ./Tools/setup/ubuntu.sh
   make px4_sitl gz_x500
   ```

4. **Install QGroundControl** (optional):
   Download from [QGroundControl](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html)

### Running the Simulation


1. **Start the full simulation stack**:
   ```bash
   # Source ROS2
   source /opt/ros/humble/setup.bash
   
   # Launch simulation with GPS enabled
   ros2 launch simulation/launch/full_simulation.launch.py
   
   # Or launch with GPS disabled (GNSS-denied mode)
   ros2 launch simulation/launch/full_simulation.launch.py gps_enabled:=false
   ```

2. **Headless mode** (for CI/testing):
   ```bash
   ros2 launch simulation/launch/full_simulation.launch.py headless:=true
   ```

3. **Record data for analysis**:
   ```bash
   ros2 launch simulation/launch/full_simulation.launch.py record_bag:=true
   ```

### Testing GNSS-Denied Scenarios

The simulation supports several GPS denial scenarios configured in `simulation/config/gps_control.yaml`:

| Scenario | Description |
|----------|-------------|
| `complete_denial` | Total GPS loss |
| `degraded_signal` | High HDOP, few satellites |
| `intermittent` | Periodic GPS dropouts |
| `gradual_degradation` | Progressive signal loss |

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

### Running VNS Standalone

```python
from vns.core import VNS
from vns.config import ConfigManager

# Load configuration
config = ConfigManager().load("config/simulation.yaml")

# Initialize VNS
vns = VNS(config)

# Start processing
vns.start()
```

### Command Line Interface

```bash
# Run VNS with configuration file
vns --config config/simulation.yaml

# Build reference database from images
vns database build --input ./images --output area.vnsdb

# Inspect database contents
vns database inspect area.vnsdb
```

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
└── config/               # Configuration templates
```

## Development

### Running Tests

```bash
# All tests
make test

# Property-based tests only
make test-pbt

# With coverage
pytest tests/ --cov=src/vns
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

### Pre-commit Hooks

Pre-commit hooks are installed automatically with `make install-dev`. They run:
- Black (formatting)
- Ruff (linting)
- MyPy (type checking)

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check` to verify
5. Submit a pull request
