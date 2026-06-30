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
from vns.utils.paths import (
    compose_gazebo_model_path,
    compose_gazebo_resource_path,
    resolve_simulation_root,
    resolve_vns_python,
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

    simulation_root = resolve_simulation_root(Path(__file__).resolve())
    worlds_dir = simulation_root / 'worlds'
    models_dir = simulation_root / 'models'
    config_dir = simulation_root / 'config'

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

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value=DEFAULT_CAMERA_TOPIC,
        description='ROS image topic shared by the Gazebo camera and vns_node'
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

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run gzserver only, skip gzclient GUI'
    )

    vns_python_arg = DeclareLaunchArgument(
        'vns_python',
        default_value=resolve_vns_python(simulation_root),
        description='Python interpreter used for VNS ROS nodes'
    )

    # ==================== Environment ====================

    # Gazebo Classic uses GAZEBO_MODEL_PATH (prepend, keep existing entries).
    # compose_gazebo_* also add the Gazebo share dir (/usr/share/gazebo-11) so
    # the camera sensor's shaders and base models resolve without sourcing
    # /usr/share/gazebo/setup.sh first.
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        compose_gazebo_model_path(models_dir),
    )

    gazebo_resource_path = SetEnvironmentVariable(
        'GAZEBO_RESOURCE_PATH',
        compose_gazebo_resource_path(worlds_dir),
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
                prefix=[LaunchConfiguration('vns_python')],
                parameters=[{
                    'config_file': LaunchConfiguration('vns_config'),
                    'camera_topic': LaunchConfiguration('camera_topic'),
                    'ground_truth_topic': LaunchConfiguration('ground_truth_topic'),
                    'evaluation_logging': LaunchConfiguration('evaluation_logging'),
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
        evaluation_logging_arg,
        ground_truth_topic_arg,
        headless_arg,
        vns_python_arg,

        # Environment
        gazebo_model_path,
        gazebo_resource_path,
        gazebo_model_db,

        # Gazebo
        gzserver,
        gzclient,
        spawn_drone,

        # VNS
        vns_node,
    ])
