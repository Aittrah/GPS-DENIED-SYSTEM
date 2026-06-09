#!/bin/bash
# HOOK 2: Block ros_gz_bridge / ros_gz_image — Gazebo Harmonic bridge packages.
# In Gazebo Classic, sensors are bridged by SDF plugins inside model.sdf, NOT
# by separate ros_gz_bridge nodes.
set -uo pipefail

TOOL_INPUT=$(cat)

MATCH=$(python3 -c "
import json, sys, re

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)

name = data.get('tool_name', '')
inp  = data.get('tool_input', {})

if name == 'Bash':
    content = inp.get('command', '')
elif name == 'Edit':
    content = inp.get('new_string', '')
elif name == 'Write':
    content = inp.get('content', '')
else:
    sys.exit(0)

if re.search(r'ros_gz_bridge|ros_gz_image', content):
    print('BLOCK')
" "$TOOL_INPUT" 2>/dev/null) || MATCH=""

if [ "$MATCH" = "BLOCK" ]; then
    cat >&2 <<'ERRMSG'
[VNS HOOK] BLOCKED: `ros_gz_bridge` and `ros_gz_image` are Gazebo Harmonic packages.
This project uses Gazebo Classic 11 with gazebo_ros_pkgs.

Sensors are bridged via <plugin> blocks INSIDE model.sdf — NOT as separate ROS 2 nodes.

Correct Gazebo Classic sensor bridge plugins (declare inside each <sensor> in model.sdf):

  Camera:        libgazebo_ros_camera.so
  IMU:           libgazebo_ros_imu_sensor.so
  GPS/NavSatFix: libgazebo_ros_gps_sensor.so
  Ground truth:  libgazebo_ros_p3d.so

Example plugin block:
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/vns_drone</namespace>
      <remapping>~/image_raw:=camera</remapping>
    </ros>
  </plugin>

See: /home/hp/GPS-DENIED-SYSTEM/simulation/models/uav_model/model.sdf
ERRMSG
    exit 2
fi

exit 0
