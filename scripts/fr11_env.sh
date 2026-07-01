#!/usr/bin/env bash

source /opt/ros/humble/setup.bash

export VNS_PYTHON=/usr/bin/python3

export PX4_AUTOPILOT=/home/hp/PX4-Autopilot
export PX4_DIR=/home/hp/PX4-Autopilot

export PATH="$PX4_AUTOPILOT/build/px4_sitl_default/bin:$PATH"

export GAZEBO_MODEL_PATH="/home/hp/GPS-DENIED-SYSTEM/simulation/models:$PX4_AUTOPILOT/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:${GAZEBO_MODEL_PATH:-}"

export GAZEBO_PLUGIN_PATH="$PX4_AUTOPILOT/build/px4_sitl_default/build_gazebo-classic:${GAZEBO_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="$PX4_AUTOPILOT/build/px4_sitl_default/build_gazebo-classic:${LD_LIBRARY_PATH:-}"

if [ -f /home/hp/GPS-DENIED-SYSTEM/install/setup.bash ]; then
  source /home/hp/GPS-DENIED-SYSTEM/install/setup.bash
fi

echo "FR-11 env loaded"
echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
echo "VNS_PYTHON=$VNS_PYTHON"
echo "PX4_AUTOPILOT=$PX4_AUTOPILOT"

export ROS_LOG_DIR=/home/hp/GPS-DENIED-SYSTEM/ros_logs
