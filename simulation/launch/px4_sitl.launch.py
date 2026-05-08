#!/usr/bin/env python3
"""
PX4 SITL Launch File

Launches PX4 Software-In-The-Loop simulation for VNS testing.
This connects to Gazebo and provides MAVLink interface for VNS.

Usage:
    ros2 launch vns_simulation px4_sitl.launch.py
    
    # With specific vehicle:
    ros2 launch vns_simulation px4_sitl.launch.py vehicle:=iris
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, EnvironmentVariable


def generate_launch_description():
    """Generate the launch description for PX4 SITL."""
    
    # Launch arguments
    px4_dir_arg = DeclareLaunchArgument(
        'px4_dir',
        default_value='/opt/px4/PX4-Autopilot',
        description='Path to PX4-Autopilot directory'
    )
    
    vehicle_arg = DeclareLaunchArgument(
        'vehicle',
        default_value='x500',
        description='PX4 vehicle model (iris, x500, etc.)'
    )
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='qau_campus',
        description='Gazebo world name'
    )
    
    instance_arg = DeclareLaunchArgument(
        'instance',
        default_value='0',
        description='PX4 instance number'
    )
    
    mavlink_udp_port_arg = DeclareLaunchArgument(
        'mavlink_udp_port',
        default_value='14540',
        description='MAVLink UDP port'
    )
    
    # Environment variables for PX4
    px4_home = SetEnvironmentVariable(
        'PX4_HOME',
        LaunchConfiguration('px4_dir')
    )
    
    px4_sim_model = SetEnvironmentVariable(
        'PX4_SIM_MODEL',
        LaunchConfiguration('vehicle')
    )
    
    # PX4 SITL process
    px4_sitl = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'cd $PX4_HOME && make px4_sitl_default gz_' + 
            '$(echo $PX4_SIM_MODEL)'
        ],
        output='screen',
        shell=True
    )
    
    # Alternative: Direct PX4 execution (if already built)
    px4_direct = ExecuteProcess(
        cmd=[
            'bash', '-c',
            '''
            cd ${PX4_HOME} && \
            source Tools/simulation/gz/setup_gz.bash && \
            ./build/px4_sitl_default/bin/px4 \
                -i ${INSTANCE} \
                -d ./build/px4_sitl_default/etc
            '''
        ],
        output='screen',
        shell=True,
        additional_env={
            'INSTANCE': LaunchConfiguration('instance'),
        }
    )
    
    # MAVLink router for multiple connections
    mavlink_router = ExecuteProcess(
        cmd=[
            'mavlink-routerd',
            '-e', '127.0.0.1:14550',  # QGroundControl
            '-e', '127.0.0.1:14551',  # VNS
            '0.0.0.0:14540'           # PX4 SITL
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        px4_dir_arg,
        vehicle_arg,
        world_arg,
        instance_arg,
        mavlink_udp_port_arg,
        
        # Environment
        px4_home,
        px4_sim_model,
        
        # PX4 SITL (delayed start to allow Gazebo to initialize)
        TimerAction(
            period=5.0,
            actions=[px4_sitl]
        ),
        
        # MAVLink router (delayed start)
        TimerAction(
            period=8.0,
            actions=[mavlink_router]
        ),
    ])