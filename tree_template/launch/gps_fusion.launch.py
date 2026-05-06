import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    pkg_config = os.path.join(
        get_package_share_directory('tree_template'), 'config', 'nav'
    )

    return LaunchDescription([
        # Local EKF: fuses wheel odom (/odometry/wheel from odom_reframer) + IMU
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_local",
            output="screen",
            parameters=[
                os.path.join(pkg_config, 'ekf_local.yaml'),
                {'use_sim_time': use_sim_time},
                {'publish_tf': True},
            ],
            remappings=[
                # Only need to redirect the EKF's output
                ("odometry/filtered", "/odometry/local"),
            ],
        ),

        # NavSat transform: GPS + IMU + local odom → /odometry/gps
        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[
                os.path.join(pkg_config, 'navsat.yaml'),
                {'use_sim_time': use_sim_time},
            ],
            remappings=[
                ("gps/fix", "/fix"),              # bag publishes /fix
                ("imu", "/imu/data"),             # bag publishes /imu/data
                ("odometry/filtered", "/odometry/local"),  # input: local EKF output
                ("odometry/gps", "/odometry/gps"),         # output: to global EKF
            ],
        ),

        # Global EKF: fuses /odometry/wheel + /odometry/gps → /odometry/global
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_global",
            output="screen",
            parameters=[
                os.path.join(pkg_config, 'ekf_global.yaml'),
                {'use_sim_time': use_sim_time},
                {'publish_tf': True},
            ],
            remappings=[
                ("odometry/filtered", "/odometry/global"),
            ],
        ),
        
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_imu",
            arguments=["--x", "0", "--y", "0", "--z", "0.025",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "amiga__base",
                    "--child-frame-id", "imu_link"],
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_gps",
            arguments=["--x", "-0.5", "--y", "0", "--z", "2.25",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "amiga__base",
                    "--child-frame-id", "reach_rs"],
        ),
    ])