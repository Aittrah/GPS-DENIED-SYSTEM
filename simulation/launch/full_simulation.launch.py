#!/usr/bin/env python3
"""
Full VNS Simulation Launch File

Launches the complete simulation stack:
1. Gazebo with QAU Campus world
2. PX4 SITL
3. VNS node
4. All necessary bridges and tools

Usage:
    ros2 launch vns_simulation full_simulation.launch.py
    
    # Test GNSS-denied scenario:
    ros2 launch vns_simulation full_simulation.launch.py gps_enabled:=false
    
    # Headless mode for CI/testing:
    ros2 launch vns_simulation full_simulation.launch.py headless:=true
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    """Generate the full simulation launch description."""
    
    # Get package directories
    pkg_dir = Path(__file__).parent.parent
    launch_dir = pkg_dir / 'launch'
    worlds_dir = pkg_dir / 'worlds'
    models_dir = pkg_dir / 'models'
    config_dir = pkg_dir / 'config'
    database_dir = pkg_dir / 'database'
    
    # ==================== Launch Arguments ====================
    
    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled',
        default_value='true',
        description='Enable GPS sensor in simulation'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run simulation in headless mode'
    )
    
    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value='/opt/px4/PX4-Autopilot',
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
    
    gz_model_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        str(models_dir)
    )
    
    # ==================== Gazebo Simulation ====================
    
    gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', str(worlds_dir / 'vns_test_world.sdf')],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    
    gazebo_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', str(worlds_dir / 'vns_test_world.sdf')],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless'))
    )
    
    # Spawn drone after Gazebo starts
    spawn_drone = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz', 'service', '-s', '/world/qau_campus_world/create',
                    '--reqtype', 'gz.msgs.EntityFactory',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '5000',
                    '--req',
                    f'sdf_filename: "{models_dir}/vns_drone/model.sdf", '
                    'name: "vns_drone", '
                    'pose: {position: {x: 0, y: 0, z: 1}}'
                ],
                output='screen'
            )
        ]
    )
    
    # ==================== ROS-Gazebo Bridges ====================
    
    bridges = TimerAction(
        period=5.0,
        actions=[
            # Camera bridge
            Node(
                package='ros_gz_image',
                executable='image_bridge',
                name='camera_bridge',
                arguments=['/world/qau_campus_world/model/vns_drone/link/base_link/sensor/vns_camera/image'],
                remappings=[
                    ('/world/qau_campus_world/model/vns_drone/link/base_link/sensor/vns_camera/image',
                     '/vns_drone/camera')
                ],
                output='screen'
            ),
            
            # IMU bridge
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='imu_bridge',
                arguments=[
                    '/world/qau_campus_world/model/vns_drone/link/base_link/sensor/imu_sensor/imu'
                    '@sensor_msgs/msg/Imu@gz.msgs.IMU'
                ],
                remappings=[
                    ('/world/qau_campus_world/model/vns_drone/link/base_link/sensor/imu_sensor/imu',
                     '/vns_drone/imu')
                ],
                output='screen'
            ),
            
            # GPS bridge (conditional)
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='gps_bridge',
                arguments=[
                    '/world/qau_campus_world/model/vns_drone/link/base_link/sensor/gps_sensor/navsat'
                    '@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat'
                ],
                remappings=[
                    ('/world/qau_campus_world/model/vns_drone/link/base_link/sensor/gps_sensor/navsat',
                     '/vns_drone/gps')
                ],
                output='screen',
                condition=IfCondition(LaunchConfiguration('gps_enabled'))
            ),
            
            # Ground truth pose bridge
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='pose_bridge',
                arguments=[
                    '/model/vns_drone/pose@geometry_msgs/msg/PoseStamped@gz.msgs.Pose'
                ],
                remappings=[
                    ('/model/vns_drone/pose', '/vns_drone/ground_truth')
                ],
                output='screen'
            ),
        ]
    )
    
    # ==================== VNS Node ====================
    
    vns_node = TimerAction(
        period=8.0,
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
    
    # QGroundControl
    qgc = ExecuteProcess(
        cmd=['QGroundControl.AppImage'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_qgc'))
    )
    
    # ROS bag recording
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
    
    # ==================== Info Messages ====================
    
    startup_info = LogInfo(
        msg='\n'
            '========================================\n'
            '  VNS Full Simulation Starting\n'
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
        gz_model_path,
        
        # Gazebo
        gazebo_gui,
        gazebo_headless,
        spawn_drone,
        
        # Bridges
        bridges,
        
        # VNS
        vns_node,
        
        # Optional tools
        qgc,
        bag_record,
    ])