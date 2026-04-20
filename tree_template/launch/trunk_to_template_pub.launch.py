#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    default_params_file = os.path.join(
        get_package_share_directory("tree_template"),
        "config",
        "launch_config.yaml",
    )
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulated clock",
        ),

        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Path to a ROS2 params YAML file (launch_config.yaml)",
        ),

        # ----------------------------
        # Env vars for trunk_width_ws process
        # ----------------------------
        SetEnvironmentVariable(
            name="WIDTH_ESTIMATION_PACKAGE_PATH",
            value="/home/marcus/trunk_width_ws/trunk_width_estimation",
        ),
        SetEnvironmentVariable(
            name="WIDTH_ESTIMATION_PACKAGE_DATA_PATH",
            value="/home/marcus/trunk_width_ws/width_estimation_package_data",
        ),
        SetEnvironmentVariable(
            name="USE_SIM_TIME",
            value=use_sim_time,
        ),

        # Run the trunk_width_ws publisher script
        ExecuteProcess(
            cmd=[
                "python3",
                "/home/marcus/trunk_width_ws/trunk_width_estimation/scripts/ros2/ros_publisher_node.py",
            ],
            output="screen",
        ),

        # ----------------------------
        # tree_template nodes
        # ----------------------------
        Node(
            package="tree_template",
            executable="tree_template",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),

        Node(
            package="tree_template",
            executable="trunk_detection_relay",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),

        Node(
            package="tree_template",
            executable="slam_odom_correction_tf",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),

        # Node(
        #     package="tree_template",
        #     executable="depth_image_to_pointcloud2",
        #     output="screen",
        #     parameters=[params_file, {"use_sim_time": use_sim_time}],
        # ),

        # Odometry reframer node
        Node(
            package="tree_template",
            executable="odom_reframer_calibrated",
            name="odom_reframer",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),

        # Row fast slam node
        Node(
            package="tree_template",
            executable="row_fast_slam",
            name="row_fast_slam",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),

        # Save trunk locations from slam
        Node(
            package="tree_template",
            executable="save_best_particle_map",
            name="save_best_particle_map",
            output="screen",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
        ),
    ])
