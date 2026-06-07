markdown# GNSS-Free Navigation System for UAVs

A desktop application for mission planning and GNSS-denied UAV navigation.
Built as Final Year Project at IIT, Quaid-i-Azam University Islamabad (2022–2026).

**Supervisor:** Dr. Bushra Almas  
**Team:** Muhammad Ahsan, Aittrah Sardar

---

## What This Project Does

This system enables UAVs to navigate without GPS by using:
- Satellite image matching (visual localization)
- Dead reckoning using IMU sensor data
- A* path planning with KML-based mission setup
- Google Earth Pro integration for area selection

---

## Project Structure
GPS-DENIED-SYSTEM/
├── src/vns/                    # Main Python package
│   ├── app/                    # Desktop application (UI)
│   ├── core/                   # Data models and coordinate utils
│   ├── preprocessing/          # Satellite and UAV image processing
│   ├── navigation/             # A* path planning
│   ├── kml/                    # KML file reader
│   ├── perception/             # Feature extraction (in progress)
│   ├── state_estimation/       # Sensor fusion (in progress)
│   └── simulation/             # Drone simulation (in progress)
├── tests/                      # Unit and integration tests
├── simulation/                 # Gazebo/PX4 simulation files
├── data/                       # Sample KML and satellite data
├── docs/                       # Documentation and demo video
└── assets/                     # Images for README

---

## Installation

### Requirements
- Python 3.10+
- Windows 10/11

### Install Dependencies

```bash
pip install opencv-python numpy Pillow
```

### Run the App

```bash
python src/vns/app/desktop_app.py
```

---

## How to Use

1. Enter start and end coordinates (latitude, longitude)
2. Click **Plot Points** to see them on the map
3. Click **Open in Google Earth Pro** to view the area
4. Draw a polygon in Google Earth and save it as KML
5. App auto-detects the KML file
6. Click **Run A* Path Planning** to generate waypoints
7. Export waypoints as CSV or JSON

See full guide: [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)

---

## Demo

> Demo video coming soon — see `docs/demo/`

---

## Modules Explained

| Module | What it does |
|--------|-------------|
| `app/` | Desktop GUI — mission planning interface |
| `core/` | GeoPoint, NED coordinates, UAV data models |
| `preprocessing/` | Resize, denoise, normalize satellite images |
| `navigation/` | A* algorithm, waypoint manager |
| `kml/` | Parse KML files from Google Earth |

---

## Tests

```bash
pytest tests/unit/ -v
```

---

## Documentation

- [How to Use the App](docs/HOW_TO_USE.md)
- [SRS Document](docs/SRS.md)
- [SDS Document](docs/SDS.md)
