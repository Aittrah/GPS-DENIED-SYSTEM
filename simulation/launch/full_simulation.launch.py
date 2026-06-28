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
import tempfile

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
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
    """Generate the full simulation launch description."""

    # Get package directories
    pkg_dir = Path(__file__).parent.parent
    worlds_dir = pkg_dir / 'worlds'
    models_dir = pkg_dir / 'models'
    config_dir = pkg_dir / 'config'
    # The reference database is NOT installed via data_files, so reference it from
    # the source tree by absolute path (works whether the launch runs from source
    # or from the install/ share dir under `ros2 launch vns ...`).
    database_dir = Path('/home/hp/GPS-DENIED-SYSTEM/simulation/database')

    world_file = str(worlds_dir / 'uav_test_world.sdf')
    drone_sdf = str(models_dir / 'iris_downward_cam' / 'model.sdf')

    # vns_node runs under the .venv-run interpreter (numpy 1.26.x) so ROS Humble's
    # cv_bridge (built for numpy 1.x) does not segfault on the first camera frame.
    # The venv was created with --system-site-packages, so rclpy and the ROS
    # message packages resolve from the sourced /opt/ros/humble overlay while the
    # venv's numpy 1.26.x shadows the user-site numpy 2.x. Absolute path: under
    # `ros2 launch vns ...` this file runs from the install/ share dir, so the
    # venv cannot be derived relative to __file__.
    venv_python = '/home/hp/GPS-DENIED-SYSTEM/.venv-run/bin/python3'

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

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value=DEFAULT_CAMERA_TOPIC,
        description='ROS image topic shared by the Gazebo camera and vns_node'
    )

    # ==================== Environment Setup ====================

    # Gazebo Classic uses GAZEBO_MODEL_PATH (not GZ_SIM_RESOURCE_PATH)
    gazebo_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        str(models_dir) + ':' + '${GAZEBO_MODEL_PATH}'
    )

    # Disable the online model database so gzserver does not block on startup
    # trying to fetch models from models.gazebosim.org (slow/offline -> hang).
    gazebo_model_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

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
            OpaqueFunction(function=_spawn_drone, kwargs={'drone_sdf': drone_sdf})
        ]
    )

    # ==================== ROS-Gazebo Bridges ====================
    #
    # Gazebo Classic plugins in the SDF publish sensor data on ROS 2 topics:
    #   camera, IMU, ground-truth -> /vns_drone/* (direct)
    #   GPS -> /vns_drone/gps_raw (gated by gps_gate_node below)

    # GPS gate: relays gps_raw -> gps when enabled, silences for denial.
    # Toggle at runtime:
    #   ros2 service call /vns/set_gps_enabled std_srvs/srv/SetBool "{data: false}"
    gps_gate = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    str(pkg_dir / 'scripts' / 'gps_gate_node.py'),
                    '--ros-args',
                    '-p', ['gps_enabled:=', LaunchConfiguration('gps_enabled')],
                ],
                output='screen',
            )
        ]
    )

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
                prefix=[venv_python],  # see venv_python note above (numpy/cv_bridge)
                parameters=[{
                    'config_file': str(config_dir / 'simulation.yaml'),
                    'database_path': str(database_dir / 'qau_campus.vnsdb'),
                    'camera_topic': LaunchConfiguration('camera_topic'),
                    'simulation_mode': True,
                }],
                # camera_topic keeps the Gazebo publisher and node subscriber
                # aligned without ROS remap rules.
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
            LaunchConfiguration('camera_topic'),
            '/vns_drone/gps_raw',
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
        camera_topic_arg,

        # Environment
        gazebo_model_path,
        gazebo_model_db,

        # Gazebo
        gzserver,
        gzclient,
        spawn_drone,

        # GPS gate (gps_raw -> gps, toggleable at runtime)
        gps_gate,

        # PX4
        px4_sitl,

        # VNS
        vns_node,

        # Optional tools
        qgc,
        bag_record,
    ])
