# Hardware Deployment Guide: VNS

This guide details the physical installation, companion computer setup, and calibration procedures to deploy the Visual Navigation System (VNS) on actual autonomous drone hardware.

---

## 1. Hardware Requirements

To achieve real-time 30+ FPS visual navigation estimation, the following hardware is recommended:

| Component | Minimum Specification | Recommended |
|---|---|---|
| **Companion Computer** | Raspberry Pi 4 (4GB RAM) | Jetson Nano / Orin Nano |
| **Flight Controller** | FMUv5 (e.g. Pixhawk 4) | FMUv6X (e.g. Pixhawk 6C/6X) |
| **Camera** | USB Global Shutter (640x480, 60 FPS) | MIPI CSI Global Shutter (ar0144 / ar0234) |
| **Lens** | Wide-angle, low distortion (no fisheye) | 2.8mm focal length (M12 mount) |

> [!WARNING]
> Rolling shutter cameras are highly prone to **motion blur and jello effect** under quadcopter vibrations, which completely degrades ORB feature extraction. Always use a **Global Shutter** camera.

---

## 2. Camera Mounting

1. **Orientation:** Mount the camera pointing **straight down** (nadir view) perpendicular to the drone body center.
2. **Vibration Damping:** Use a carbon fiber plate with rubber vibration isolation balls to isolate the camera from high-frequency motor vibrations.
3. **Alignment:** Align the camera body axes with the drone body:
   - Camera $+X$ axis must point to the **Right** (matching drone $+Y$ / starboard).
   - Camera $+Y$ axis must point **Backward** (matching camera $V$ coordinates going down).
   - Camera mount parameters must be updated in `config/simulation.yaml` if physical rotation or off-center mounting is used.

---

## 3. Companion Computer Installation

On the companion computer (Ubuntu 22.04 LTS with ROS2 Humble installed):

1. **Clone and Install VNS:**
   ```bash
   git clone https://github.com/Aittrah/GPS-DENIED-SYSTEM.git
   cd GPS-DENIED-SYSTEM
   pip install -e ".[dev]"
   ```
2. **Camera Drivers:**
   Ensure `v4l-utils` is installed and you can capture raw frames:
   ```bash
   sudo apt install v4l-utils
   v4l2-ctl --list-devices
   ```

---

## 4. PX4 Flight Controller Configuration

To blend external vision estimates into PX4's EKF2 state estimator, configure the following parameters via QGroundControl:

- `EKF2_EV_CTRL`: Set to `Horizontal Position` and `Vertical Position` to enable external vision fusion.
- `EKF2_EV_DELAY`: Set to the vision processing latency (typically `50.0` ms for ORB features).
- `EKF2_EV_POS_X`, `_Y`, `_Z`: Enter the camera mount displacement relative to the flight controller center of gravity in meters.
- `MAV_1_CONFIG`: Set to the serial port connected to the companion computer (e.g., `TELEM2`).
- `MAV_1_MODE`: Set to `Onboard`.
- `SER_TEL2_BAUD`: Set to `921600` (baud rate for high speed MAVLink streaming).

---

## 5. System Calibration and Verification

1. **Intrinsics Calibration:** Run standard OpenCV chessboard calibration to determine `fx`, `fy`, `cx`, and `cy`. Update these parameters in `config.yaml`.
2. **MAVLink Connectivity Test:**
   Verify connection to the flight controller:
   ```bash
   vns --config config/simulation.yaml
   ```
3. **Safety Failsafe:** Set the Flight Controller failsafe parameter `COM_OBL_ACT` to `Return to Launch` (RTL) or `Land` in case external visual navigation drops out.
