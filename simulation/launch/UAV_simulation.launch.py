#!/usr/bin/env python3
"""
VNS Simulation Launch File (Gazebo Classic 11 + ROS 2 Humble)

Slimmer launch entrypoint that brings up:
- Gazebo Classic (gzserver + gzclient) with the QAU Campus world
- Drone model spawned via gazebo_ros spawn_entity.py
- VNS node

PX4 SITL is intentionally NOT started here; use px4_sitl.launch.py for that,
or full_simulation.launch.py for the complete stack.

Usage:
    ros2 launch vns_simulation UAV_simulation.launch.py

    # GPS-denied:
    ros2 launch vns_simulation UAV_simulation.launch.py gps_enabled:=false
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
    """Generate the launch description for VNS simulation."""

    # Get package directories
    pkg_dir = Path(__file__).parent.parent
    worlds_dir = pkg_dir / 'worlds'
    models_dir = pkg_dir / 'models'
    config_dir = pkg_dir / 'config'

    drone_sdf = str(models_dir / 'iris_downward_cam' / 'model.sdf')

    # ==================== Launch Arguments ====================

    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled',
        default_value='true',
        description='Pass-through flag for the VNS node (sensor itself is '
                    'controlled inside the SDF / gps_control.yaml)'
    )

    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=str(worlds_dir / 'uav_test_world.sdf'),
        description='Path to Gazebo world file'
    )

    vns_config_arg = DeclareLaunchArgument(
        'vns_config',
        default_value=str(config_dir / 'simulation.yaml'),
        description='Path to VNS configuration file'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run gzserver only, skip gzclient GUI'
    )

    # ==================== Environment ====================

    # Gazebo Classic uses GAZEBO_MODEL_PATH (prepend, keep existing entries)
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        str(models_dir) + ':' + '${GAZEBO_MODEL_PATH}'
    )

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
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )

    # Spawn drone via spawn_entity.py (waits for /spawn_entity service from
    # libgazebo_ros_factory; the 4 s delay covers gzserver startup).
    spawn_drone = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_iris',
                arguments=[
                    '-entity', 'iris_downward_cam',
                    '-file', drone_sdf,
                    '-x', '0', '-y', '0', '-z', '1',
                ],
                output='screen',
            )
        ]
    )

    # NOTE: Bridges are NOT separate nodes in Gazebo Classic.
    # Camera / IMU / GPS / ground-truth topics are published directly by
    # the gazebo_ros plugins inside model.sdf. See full_simulation.launch.py
    # header comments for the exact plugin blocks to add to your SDF.

    # ==================== VNS Node ====================

    vns_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='vns',
                executable='vns_node',
                name='vns_node',
                parameters=[{
                    'config_file': LaunchConfiguration('vns_config'),
                    'gps_enabled': LaunchConfiguration('gps_enabled'),
                    'simulation_mode': True,
                }],
                remappings=[
                    ('camera/image_raw', '/vns_drone/camera'),
                    ('gps/fix', '/vns_drone/gps'),
                    ('imu/data', '/vns_drone/imu'),
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        gps_enabled_arg,
        world_file_arg,
        vns_config_arg,
        headless_arg,

        # Environment
        gazebo_model_path,

        # Gazebo
        gzserver,
        gzclient,
        spawn_drone,

        # VNS
        vns_node,
    ])
