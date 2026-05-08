#!/usr/bin/env python3
"""
VNS Simulation Launch File

Launches the complete VNS simulation environment including:
- Gazebo with QAU Campus world
- PX4 SITL
- VNS node with simulation configuration
- Camera bridge for image topics

Usage:
    ros2 launch vns_simulation vns_simulation.launch.py
    
    # With GPS disabled:
    ros2 launch vns_simulation vns_simulation.launch.py gps_enabled:=false
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for VNS simulation."""
    
    # Get package directories
    pkg_dir = Path(__file__).parent.parent
    worlds_dir = pkg_dir / 'worlds'
    models_dir = pkg_dir / 'models'
    config_dir = pkg_dir / 'config'
    
    # Launch arguments
    gps_enabled_arg = DeclareLaunchArgument(
        'gps_enabled',
        default_value='true',
        description='Enable GPS sensor in simulation'
    )
    
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=str(worlds_dir / 'vns_test_world.sdf'),
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
        description='Run Gazebo in headless mode'
    )
    
    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value='/opt/px4',
        description='Path to PX4-Autopilot directory'
    )
    
    # Environment variables for Gazebo
    gazebo_model_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        str(models_dir)
    )
    
    # Gazebo simulation
    gazebo_cmd = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r',
            LaunchConfiguration('world_file'),
        ],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    
    gazebo_headless_cmd = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r', '-s',
            LaunchConfiguration('world_file'),
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless'))
    )
    
    # Spawn drone model
    spawn_drone = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/world/qau_campus_world/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req',
            f'sdf_filename: "{models_dir}/vns_drone/model.sdf", name: "vns_drone"'
        ],
        output='screen'
    )
    
    # ROS-Gazebo bridge for camera
    camera_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='camera_bridge',
        arguments=['/vns_drone/camera'],
        output='screen'
    )
    
    # ROS-Gazebo bridge for GPS (when enabled)
    gps_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gps_bridge',
        arguments=[
            '/vns_drone/gps@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat'
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('gps_enabled'))
    )
    
    # ROS-Gazebo bridge for IMU
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=[
            '/vns_drone/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'
        ],
        output='screen'
    )
    
    # VNS node
    vns_node = Node(
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
    
    return LaunchDescription([
        # Arguments
        gps_enabled_arg,
        world_file_arg,
        vns_config_arg,
        headless_arg,
        px4_dir_arg,
        
        # Environment
        gazebo_model_path,
        
        # Gazebo
        gazebo_cmd,
        gazebo_headless_cmd,
        spawn_drone,
        
        # Bridges
        camera_bridge,
        gps_bridge,
        imu_bridge,
        
        # VNS
        vns_node,
    ])