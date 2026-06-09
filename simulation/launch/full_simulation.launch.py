#!/usr/bin/env python3
"""
Full VNS Simulation Launch File (Gazebo Classic 11 + ROS 2 Humble)

Launches the complete simulation stack:
1. Gazebo Classic with the QAU Campus world (gzserver/gzclient)
2. PX4 SITL connected via MAVLink UDP
3. ROS 2 <-> Gazebo bridges using gazebo_ros plugins (declared in the SDF)
4. VNS node

Usage:
    ros2 launch vns_simulation full_simulation.launch.py

    # GNSS-denied scenario:
    ros2 launch vns_simulation full_simulation.launch.py gps_enabled:=false

    # Headless (gzserver only, no GUI):
    ros2 launch vns_simulation full_simulation.launch.py headless:=true

Notes for Gazebo Classic:
    - There is no `gz sim` command and no `ros_gz_bridge`. Sensors are exposed
      to ROS 2 by adding <plugin> tags to the SDF model (libgazebo_ros_camera,
      libgazebo_ros_imu_sensor, libgazebo_ros_gps_sensor, libgazebo_ros_p3d).
    - Models are spawned via the `spawn_entity.py` script from gazebo_ros.
    - The model resource path is GAZEBO_MODEL_PATH, not GZ_SIM_RESOURCE_PATH.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the full simulation launch description."""

    # Get package directories
    pkg_dir = Path(__file__).parent.parent
    worlds_dir = pkg_dir / 'worlds'
    models_dir = pkg_dir / 'models'
    config_dir = pkg_dir / 'config'
    database_dir = pkg_dir / 'database'

    world_file = str(worlds_dir / 'vns_test_world.sdf')
    drone_sdf = str(models_dir / 'iris_downward_cam' / 'model.sdf')

    # ==================== Launch Arguments ====================

    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled',
        default_value='true',
        description='Enable GPS sensor bridge (the SDF still owns sensor on/off)'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run gzserver only (no gzclient GUI)'
    )

    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value=str(Path.home() / 'PX4-Autopilot'),
        description='Path to PX4-Autopilot directory'
    )

    qgc_arg = DeclareLaunchArgument(
        'launch_qgc',
        default_value='false',
        description='Launch QGroundControl'
    )

    rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz for visualization'
    )

    record_arg = DeclareLaunchArgument(
        'record_bag',
        default_value='false',
        description='Record ROS bag for analysis'
    )

    # ==================== Environment Setup ====================

    # Gazebo Classic uses GAZEBO_MODEL_PATH (not GZ_SIM_RESOURCE_PATH)
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        str(models_dir) + ':' + '${GAZEBO_MODEL_PATH}'
    )

    # ==================== Gazebo Classic Simulation ====================

    # gzserver (physics + sensors, always launched)
    gzserver = ExecuteProcess(
        cmd=[
            'gzserver',
            '--verbose',
            '-s', 'libgazebo_ros_init.so',     # /clock, ROS time
            '-s', 'libgazebo_ros_factory.so',  # spawn_entity service
            world_file,
        ],
        output='screen',
    )

    # gzclient (GUI) only if not headless
    gzclient = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )

    # Spawn the drone via gazebo_ros spawn_entity.py (waits for /spawn_entity service)
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

    # ==================== ROS-Gazebo Bridges ====================
    #
    # In Gazebo Classic the bridges are gazebo_ros plugins declared inside the
    # iris_downward_cam SDF (libgazebo_ros_camera, libgazebo_ros_imu_sensor,
    # libgazebo_ros_gps_sensor, libgazebo_ros_p3d). They publish directly on
    # ROS 2 topics under /vns_drone — no separate bridge nodes needed here.

    # ==================== PX4 SITL ====================
    #
    # In Gazebo Classic we use PX4's "none_iris" target: PX4 runs standalone
    # and talks to Gazebo through the existing model plugins / MAVLink UDP
    # rather than the newer gz_<vehicle> targets that need Gazebo Harmonic.

    px4_sitl = TimerAction(
        period=6.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash', '-c',
                    'cd "$0" && '
                    'HEADLESS=1 make px4_sitl none_iris',
                    LaunchConfiguration('px4_dir'),
                ],
                output='screen',
            )
        ]
    )

    # ==================== VNS Node ====================

    vns_node = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='vns',
                executable='vns_node',
                name='vns_node',
                parameters=[{
                    'config_file': str(config_dir / 'simulation.yaml'),
                    'database_path': str(database_dir / 'qau_campus.vnsdb'),
                    'simulation_mode': True,
                }],
                remappings=[
                    ('camera/image_raw', '/vns_drone/camera'),
                    ('gps/fix', '/vns_drone/gps'),
                    ('imu/data', '/vns_drone/imu'),
                    ('ground_truth', '/vns_drone/ground_truth'),
                ],
                output='screen'
            )
        ]
    )

    # ==================== Optional Tools ====================

    qgc = ExecuteProcess(
        cmd=['QGroundControl.AppImage'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_qgc'))
    )

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '/vns_drone/camera',
            '/vns_drone/gps',
            '/vns_drone/imu',
            '/vns_drone/ground_truth',
            '/vns/vision_pose',
            '/vns/status',
            '-o', 'vns_simulation_bag'
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record_bag'))
    )

    # ==================== Info ====================

    startup_info = LogInfo(
        msg='\n'
            '========================================\n'
            '  VNS Full Simulation Starting\n'
            '  Gazebo Classic 11 + ROS 2 Humble\n'
            '  Location: QAU Campus, Islamabad\n'
            '========================================\n'
    )

    return LaunchDescription([
        # Info
        startup_info,

        # Arguments
        gps_enabled_arg,
        headless_arg,
        px4_dir_arg,
        qgc_arg,
        rviz_arg,
        record_arg,

        # Environment
        gazebo_model_path,

        # Gazebo
        gzserver,
        gzclient,
        spawn_drone,

        # PX4
        px4_sitl,

        # VNS
        vns_node,

        # Optional tools
        qgc,
        bag_record,
    ])
