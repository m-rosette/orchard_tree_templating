#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([

        # Declare launch argument
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulated clock'
        ),

        SetEnvironmentVariable(
            name='WIDTH_ESTIMATION_PACKAGE_PATH',
            value='/home/marcus/trunk_width_ws/trunk_width_estimation'
        ),
        SetEnvironmentVariable(
            name='WIDTH_ESTIMATION_PACKAGE_DATA_PATH',
            value='/home/marcus/trunk_width_ws/width_estimation_package_data'
        ),

        # Pass use_sim_time to the Python script through env var
        SetEnvironmentVariable(
            name='USE_SIM_TIME',
            value=use_sim_time
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
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time
            }]
        ),

        # trunk_to_template_position node
        Node(
            package='tree_template',
            executable='trunk_to_template_position',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time
            }]
        ),

        # # row_prior_mapper node
        # Node(
        #     package='tree_template',
        #     executable='row_prior_mapper',
        #     output='screen',
        #     parameters=[{
        #         'use_sim_time': use_sim_time
        #     }]
        # ),

        # depth_image_to_pointcloud2 node
        Node(
            package='tree_template',
            executable='depth_image_to_pointcloud2',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time
            }]
        ),
    ])
