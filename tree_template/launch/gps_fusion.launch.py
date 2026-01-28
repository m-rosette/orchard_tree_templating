import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_local",
            output="screen",
            parameters=[
                os.path.join(get_package_share_directory('tree_template'), 'config', 'ekf_local.yaml'),
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                # publish the local EKF output on a stable name
                ("/odometry/filtered", "/odometry/local"),
            ],
        ),

        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[
                os.path.join(get_package_share_directory('tree_template'), 'config', 'navsat.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),

        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_global",
            output="screen",
            parameters=[
                os.path.join(get_package_share_directory('tree_template'), 'config', 'ekf_global.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
