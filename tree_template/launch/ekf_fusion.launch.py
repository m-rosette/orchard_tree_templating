import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Odometry fusion with robot_localization.
#
# Always:              ekf_local fuses the wheel odometry (/odometry/wheel from odom_reframer) and the IMU
#                      -> /odometry/local. ekf_local does not fuse GPS.
# use_gps_fusion:=true adds GPS (global) fusion -> /odometry/global and the map -> odom TF, in one of two modes:
#
# mode:=gps_heading    navsat reads /odometry/global, the heading from the GPS course (/gps/course_imu) and the fix
#   (default)          from /fix_rtk (RTK covariance). ekf_global fuses the wheel velocities, the IMU yaw rate, the
#                      GPS position and the GPS course heading (ekf_global_gps_heading.yaml). This mode needs
#                      odom_reframer twist_from_pose: true and use_now_stamp: false.
# mode:=legacy         navsat uses the IMU heading + yaw_offset and reads /odometry/local. ekf_global fuses the
#                      /odometry/local poses and /odometry/gps.
#
# robot_platform (warthog | amiga) selects the defaults for nav_config, gps_topic and imu_topic.
# This file does not publish the robot TF (see trunk_to_template_pub.launch.py publish_robot_description). nav_config is the robot_localization config dir (config/nav = 2023 Warthog,
# config/nav_2025 = 2025 Amiga, or an absolute path). gps_heading mode needs ekf_global_gps_heading.yaml,
# navsat_gps_heading.yaml and gps_heading_nodes.yaml in that dir (only config/nav has them).


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    robot_platform = LaunchConfiguration("robot_platform")
    use_gps_fusion = LaunchConfiguration("use_gps_fusion")
    mode = LaunchConfiguration("mode")
    gps_topic = LaunchConfiguration("gps_topic")
    imu_topic = LaunchConfiguration("imu_topic")

    pkg_share = get_package_share_directory("tree_template")
    pkg_config = os.path.join(pkg_share, "config")
    nav_config = PathJoinSubstitution([pkg_config, LaunchConfiguration("nav_config")])

    gps_heading_on = IfCondition(PythonExpression(
        ["'", use_gps_fusion, "' == 'true' and '", mode, "' == 'gps_heading'"]))
    legacy_on = IfCondition(PythonExpression(
        ["'", use_gps_fusion, "' == 'true' and '", mode, "' == 'legacy'"]))

    def per_platform(warthog, amiga):
        return PythonExpression(["'", warthog, "' if '", robot_platform, "' == 'warthog' else '", amiga, "'"])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("robot_platform", default_value="warthog",
                              description="warthog (2023 bags) | amiga (2025 bags): topic defaults"),
        DeclareLaunchArgument("use_gps_fusion", default_value="false",
                              description="Add GPS (global) fusion: navsat_transform + ekf_global"),
        DeclareLaunchArgument("mode", default_value="gps_heading",
                              description="GPS fusion mode: gps_heading | legacy (see header comment)"),
        DeclareLaunchArgument("nav_config", default_value=per_platform("nav", "nav_2025"),
                              description="robot_localization config dir under share/tree_template/config, "
                                          "or an absolute path"),
        DeclareLaunchArgument("gps_topic", default_value=per_platform("/fix", "/gps/pvt"),
                              description="GPS NavSatFix topic (navsat input in legacy mode)"),
        DeclareLaunchArgument("imu_topic", default_value=per_platform("/imu/data", "/oak0/imu"),
                              description="IMU topic (navsat input in legacy mode)"),

        # ----------------------------
        # Local EKF: wheel odom (/odometry/wheel from odom_reframer) + IMU -> /odometry/local
        # ----------------------------
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_local",
            output="screen",
            parameters=[
                PathJoinSubstitution([nav_config, "ekf_local.yaml"]),
                {"use_sim_time": use_sim_time},
                {"publish_tf": True},
            ],
            remappings=[("odometry/filtered", "/odometry/local")],
        ),

        # ----------------------------
        # GPS fusion, mode gps_heading
        # ----------------------------
        # /fix -> /fix_rtk with an RTK covariance
        Node(
            package="tree_template",
            executable="fix_cov_relay",
            name="fix_cov_relay",
            output="screen",
            parameters=[PathJoinSubstitution([nav_config, "gps_heading_nodes.yaml"]), {"use_sim_time": use_sim_time}],
            condition=gps_heading_on,
        ),
        # Heading from the GPS course -> /gps/course_imu
        Node(
            package="tree_template",
            executable="gps_course_heading",
            name="gps_course_heading",
            output="screen",
            parameters=[PathJoinSubstitution([nav_config, "gps_heading_nodes.yaml"]), {"use_sim_time": use_sim_time}],
            condition=gps_heading_on,
        ),
        # NavSat transform: /fix_rtk + GPS course heading + global odom -> /odometry/gps
        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[PathJoinSubstitution([nav_config, "navsat_gps_heading.yaml"]), {"use_sim_time": use_sim_time}],
            remappings=[
                ("gps/fix", "/fix_rtk"),
                ("imu", "/gps/course_imu"),
                ("odometry/filtered", "/odometry/global"),
                ("odometry/gps", "/odometry/gps"),
            ],
            condition=gps_heading_on,
        ),
        # Global EKF: wheel velocities + IMU yaw rate + GPS position + GPS course heading -> /odometry/global
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_global",
            output="screen",
            parameters=[
                PathJoinSubstitution([nav_config, "ekf_global_gps_heading.yaml"]),
                {"use_sim_time": use_sim_time},
                {"publish_tf": True},
            ],
            remappings=[("odometry/filtered", "/odometry/global")],
            condition=gps_heading_on,
        ),

        # ----------------------------
        # GPS fusion, mode legacy
        # ----------------------------
        # NavSat transform: GPS + IMU heading + local odom -> /odometry/gps
        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[PathJoinSubstitution([nav_config, "navsat.yaml"]), {"use_sim_time": use_sim_time}],
            remappings=[
                ("gps/fix", gps_topic),
                ("imu", imu_topic),
                ("odometry/filtered", "/odometry/local"),
                ("odometry/gps", "/odometry/gps"),
            ],
            condition=legacy_on,
        ),
        # Global EKF: /odometry/local + /odometry/gps -> /odometry/global
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_global",
            output="screen",
            parameters=[
                PathJoinSubstitution([nav_config, "ekf_global.yaml"]),
                {"use_sim_time": use_sim_time},
                {"publish_tf": True},
            ],
            remappings=[("odometry/filtered", "/odometry/global")],
            condition=legacy_on,
        ),
    ])
