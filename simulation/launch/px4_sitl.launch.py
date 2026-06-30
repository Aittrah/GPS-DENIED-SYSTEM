#!/usr/bin/env python3
"""
PX4 SITL Launch File (Gazebo Classic 11 + ROS 2 Humble)

Launches PX4 Software-In-The-Loop in "none" mode so it pairs with an
already-running Gazebo Classic simulation rather than the newer
`gz_<vehicle>` targets that require Gazebo Harmonic.

Usage:
    ros2 launch vns_simulation px4_sitl.launch.py

    # With specific vehicle airframe:
    ros2 launch vns_simulation px4_sitl.launch.py vehicle:=iris

Notes:
    - Gazebo is launched separately (see full_simulation.launch.py or
      UAV_simulation.launch.py). This file only starts PX4 + MAVLink.
    - PX4 -> ground listener UDP: 14550 (QGC) and 14540 (offboard / VNS).
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration

from vns.utils.paths import (
    compose_gazebo_model_path,
    compose_gazebo_resource_path,
    resolve_simulation_root,
)


def generate_launch_description():
    """Generate the launch description for PX4 SITL on Gazebo Classic."""

    simulation_root = resolve_simulation_root(Path(__file__).resolve())
    worlds_dir = simulation_root / 'worlds'
    models_dir = simulation_root / 'models'

    # ==================== Launch Arguments ====================

    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value=str(Path.home() / 'PX4-Autopilot'),
        description='Path to PX4-Autopilot directory'
    )

    vehicle_arg = DeclareLaunchArgument(
        'vehicle',
        default_value='iris',
        description='PX4 airframe (iris, standard_vtol, etc.). '
                    'Use Classic-compatible airframes only.'
    )

    instance_arg = DeclareLaunchArgument(
        'instance',
        default_value='0',
        description='PX4 instance number (for multi-vehicle setups)'
    )

    mavlink_udp_port_arg = DeclareLaunchArgument(
        'mavlink_udp_port',
        default_value='14540',
        description='MAVLink UDP port that VNS will connect to'
    )

    # ==================== Environment ====================

    px4_home = SetEnvironmentVariable(
        'PX4_HOME',
        LaunchConfiguration('px4_dir')
    )

    px4_sim_model = SetEnvironmentVariable(
        'PX4_SIM_MODEL',
        LaunchConfiguration('vehicle')
    )

    # Tell PX4 we are using an external Gazebo Classic instance.
    px4_sim_world = SetEnvironmentVariable(
        'PX4_SIM_WORLD',
        'none'
    )

    # This entrypoint starts PX4 only (Gazebo is launched separately), but we
    # compose the Gazebo asset paths here too — including the Gazebo share dir
    # (/usr/share/gazebo-11) — so a gzserver that inherits this environment can
    # resolve the camera sensor's shaders and base models consistently with the
    # other launch files.
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        compose_gazebo_model_path(models_dir),
    )
    gazebo_resource_path = SetEnvironmentVariable(
        'GAZEBO_RESOURCE_PATH',
        compose_gazebo_resource_path(worlds_dir),
    )
    gazebo_model_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    # ==================== PX4 SITL Process ====================
    #
    # `make px4_sitl none_iris` builds (if needed) and starts PX4 expecting
    # an already-running Gazebo Classic. Headless because Gazebo provides
    # its own GUI through gzclient.

    px4_sitl = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            'cd "$PX4_HOME" && HEADLESS=1 make px4_sitl none_${PX4_SIM_MODEL}'
        ],
        output='screen',
    )

    # ==================== MAVLink Router (optional) ====================
    #
    # PX4 SITL already opens UDP 14550 and 14540 by default, so a router is
    # only needed if you want >2 simultaneous ground listeners. Keep it
    # commented unless you have mavlink-routerd installed.
    #
    # mavlink_router = ExecuteProcess(
    #     cmd=[
    #         'mavlink-routerd',
    #         '-e', '127.0.0.1:14550',  # QGroundControl
    #         '-e', '127.0.0.1:14551',  # VNS
    #         '0.0.0.0:14540'
    #     ],
    #     output='screen'
    # )

    return LaunchDescription([
        # Arguments
        px4_dir_arg,
        vehicle_arg,
        instance_arg,
        mavlink_udp_port_arg,

        # Environment
        px4_home,
        px4_sim_model,
        px4_sim_world,
        gazebo_model_path,
        gazebo_resource_path,
        gazebo_model_db,

        # Delayed start so Gazebo (if launched in parallel) has time to come up
        TimerAction(period=5.0, actions=[px4_sitl]),

        # TimerAction(period=8.0, actions=[mavlink_router]),
    ])
