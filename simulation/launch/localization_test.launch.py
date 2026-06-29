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
from vns.utils.paths import (
    prepend_search_path,
    resolve_simulation_root,
    resolve_vns_python,
)


def generate_launch_description():
    simulation_root = resolve_simulation_root(Path(__file__).resolve())
    worlds_dir = simulation_root / 'worlds'
    models_dir = simulation_root / 'models'
    config_dir = simulation_root / 'config'
    database_dir = simulation_root / 'database'

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
    evaluation_logging_arg = DeclareLaunchArgument(
        'evaluation_logging',
        default_value='',
        description='Override evaluation logging; empty string uses config'
    )
    ground_truth_topic_arg = DeclareLaunchArgument(
        'ground_truth_topic',
        default_value='',
        description='Override ground-truth topic; empty string uses config'
    )

    vns_python_arg = DeclareLaunchArgument(
        'vns_python',
        default_value=resolve_vns_python(simulation_root),
        description='Python interpreter used for VNS ROS nodes'
    )

    # ==================== Environment ====================

    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        prepend_search_path(models_dir, 'GAZEBO_MODEL_PATH')
    )
    gazebo_resource_path = SetEnvironmentVariable(
        'GAZEBO_RESOURCE_PATH',
        prepend_search_path(worlds_dir, 'GAZEBO_RESOURCE_PATH')
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
                prefix=[LaunchConfiguration('vns_python')],
                parameters=[{
                    'config_file': LaunchConfiguration('vns_config'),
                    'database_path': str(database_dir / 'qau_campus.vnsdb'),
                    'ground_truth_topic': LaunchConfiguration('ground_truth_topic'),
                    'evaluation_logging': LaunchConfiguration('evaluation_logging'),
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
        evaluation_logging_arg,
        ground_truth_topic_arg,
        vns_python_arg,
        gazebo_model_path,
        gazebo_resource_path,
        gazebo_model_db,
        gzserver,
        gzclient,
        vns_node,
    ])
