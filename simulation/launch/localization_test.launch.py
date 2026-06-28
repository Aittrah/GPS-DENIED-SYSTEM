#!/usr/bin/env python3
"""VNS localization smoke-test launch (Gazebo Classic 11 + ROS 2 Humble).

The deterministic, no-PX4 way to see the visual localizer publish a pose fix:

1. gzserver on `uav_localization_test.sdf` — a map-only world (clean textured
   ground, no buildings) with a STATIC downward camera parked at 240 m over the
   origin, so the camera always sees the pristine grid_center reference tile.
2. vns_node (run under .venv-run so cv_bridge does not segfault on numpy 2).

No PX4, no drone spawn, no flight: the camera is static in the world. The node
localizes every frame and publishes geometry_msgs/PoseStamped on
/vns/vision_pose. With no GPS, the GNSS monitor goes DENIED after a couple of
seconds and the blender uses the visual fix.

Verify:
    ros2 topic echo /vns/vision_pose --once
    # -> position y≈33.7470 (lat), x≈73.1370 (lon)

Usage:
    ros2 launch vns localization_test.launch.py
    ros2 launch vns localization_test.launch.py headless:=true
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Inputs are referenced from the SOURCE tree by absolute path so the launch
    # behaves identically whether invoked as `ros2 launch vns ...` (which runs
    # this file from the install/ share dir) or by absolute path, and always uses
    # the live repo files. The repo path is hardcoded throughout this project
    # (e.g. world material file:// URIs), so this is consistent.
    repo = Path('/home/hp/GPS-DENIED-SYSTEM')
    worlds_dir = repo / 'simulation' / 'worlds'
    models_dir = repo / 'simulation' / 'models'
    config_dir = repo / 'simulation' / 'config'
    database_dir = repo / 'simulation' / 'database'

    # vns_node runs under the .venv-run interpreter (numpy 1.26.x) so ROS Humble's
    # cv_bridge (built for numpy 1.x) does not segfault on the first camera frame.
    venv_python = str(repo / '.venv-run' / 'bin' / 'python3')

    # ==================== Arguments ====================

    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=str(worlds_dir / 'uav_localization_test.sdf'),
        description='Map-only world with the static downward camera'
    )
    vns_config_arg = DeclareLaunchArgument(
        'vns_config',
        default_value=str(config_dir / 'simulation.yaml'),
        description='Path to VNS configuration file'
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run gzserver only, skip the gzclient GUI'
    )

    # ==================== Environment ====================

    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        str(models_dir) + ':' + '${GAZEBO_MODEL_PATH}'
    )
    # Disable the online model database so gzserver does not block on startup.
    gazebo_model_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    # ==================== Gazebo Classic ====================

    gzserver = ExecuteProcess(
        cmd=[
            'gzserver',
            '--verbose',
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
            LaunchConfiguration('world_file'),
        ],
        output='screen',
    )
    gzclient = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless')),
    )

    # ==================== VNS Node ====================
    # Delayed so gzserver + the camera sensor are up before the node subscribes.
    vns_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='vns',
                executable='vns_node',
                name='vns_node',
                prefix=[venv_python],  # numpy/cv_bridge: see venv_python note above
                parameters=[{
                    'config_file': LaunchConfiguration('vns_config'),
                    'database_path': str(database_dir / 'qau_campus.vnsdb'),
                    'mavlink_enabled': False,
                    'simulation_mode': True,
                }],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        world_file_arg,
        vns_config_arg,
        headless_arg,
        gazebo_model_path,
        gazebo_model_db,
        gzserver,
        gzclient,
        vns_node,
    ])
