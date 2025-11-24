#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, SetEnvironmentVariable

def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(
            name='WIDTH_ESTIMATION_PACKAGE_PATH',
            value='/home/marcus/trunk_width_ws/trunk_width_estimation'
        ),
        SetEnvironmentVariable(
            name='WIDTH_ESTIMATION_PACKAGE_DATA_PATH',
            value='/home/marcus/trunk_width_ws/width_estimation_analysis_data'
        ),

        # Run the Python script directly
        ExecuteProcess(
            cmd=[
                'python3',
                '/home/marcus/trunk_width_ws/trunk_width_estimation/scripts/ros2/ros_publisher_node.py'
            ],
            output='screen'
        ),

        # tree_template node
        Node(
            package='tree_template',
            executable='tree_template',
            output='screen'
        ),

        # trunk_to_template_position node
        Node(
            package='tree_template',
            executable='trunk_to_template_position',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        ),
    ])
