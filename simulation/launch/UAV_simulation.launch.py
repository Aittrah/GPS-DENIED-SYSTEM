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
import tempfile

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vns.utils.camera_topic import (
    DEFAULT_CAMERA_TOPIC,
    render_sdf_with_camera_topic,
)


def _spawn_drone(context, *, drone_sdf: str):
    camera_topic = LaunchConfiguration('camera_topic').perform(context)
    resolved_sdf = drone_sdf

    if camera_topic != DEFAULT_CAMERA_TOPIC:
        rendered_sdf = render_sdf_with_camera_topic(
            Path(drone_sdf).read_text(encoding='utf-8'),
            camera_topic,
        )
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            suffix='.sdf',
            prefix='iris_downward_cam_',
            delete=False,
        ) as handle:
            handle.write(rendered_sdf)
            resolved_sdf = handle.name

    return [
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_iris',
            arguments=[
                '-entity', 'iris_downward_cam',
                '-file', resolved_sdf,
                '-x', '0', '-y', '0', '-z', '1',
            ],
            output='screen',
        )
    ]


def generate_launch_description():
    """Generate the launch description for VNS simulation."""

    # Get package directories
    pkg_dir = Path(__file__).parent.parent
    worlds_dir = pkg_dir / 'worlds'
    models_dir = pkg_dir / 'models'
    config_dir = pkg_dir / 'config'

    drone_sdf = str(models_dir / 'iris_downward_cam' / 'model.sdf')

    # vns_node runs under the .venv-run interpreter (numpy 1.26.x) so ROS Humble's
    # cv_bridge (built for numpy 1.x) does not segfault on the first camera frame.
    # The venv has --system-site-packages, so rclpy / ROS msgs resolve from the
    # sourced /opt/ros/humble overlay while numpy 1.26.x shadows user-site 2.x.
    # Absolute path: under `ros2 launch vns ...` this file runs from the install/
    # share dir, so the venv cannot be derived relative to __file__.
    venv_python = '/home/hp/GPS-DENIED-SYSTEM/.venv-run/bin/python3'

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

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value=DEFAULT_CAMERA_TOPIC,
        description='ROS image topic shared by the Gazebo camera and vns_node'
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

    # Disable the online model database so gzserver does not block on startup
    # trying to fetch models from models.gazebosim.org (slow/offline -> hang).
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
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )

    # Spawn drone via spawn_entity.py (waits for /spawn_entity service from
    # libgazebo_ros_factory; the 4 s delay covers gzserver startup).
    spawn_drone = TimerAction(
        period=4.0,
        actions=[
            OpaqueFunction(function=_spawn_drone, kwargs={'drone_sdf': drone_sdf})
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
                prefix=[venv_python],  # see venv_python note above (numpy/cv_bridge)
                parameters=[{
                    'config_file': LaunchConfiguration('vns_config'),
                    'camera_topic': LaunchConfiguration('camera_topic'),
                    'gps_enabled': LaunchConfiguration('gps_enabled'),
                    'simulation_mode': True,
                }],
                # camera_topic keeps the Gazebo publisher and node subscriber
                # aligned without ROS remap rules.
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        # Arguments
        gps_enabled_arg,
        world_file_arg,
        vns_config_arg,
        camera_topic_arg,
        headless_arg,

        # Environment
        gazebo_model_path,
        gazebo_model_db,

        # Gazebo
        gzserver,
        gzclient,
        spawn_drone,

        # VNS
        vns_node,
    ])
