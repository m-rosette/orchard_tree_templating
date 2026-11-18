#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, SetEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory

# Full tree_template SLAM pipeline.
#
# robot_platform:=warthog (default, 2023 bags) | amiga (2025 bags) selects the defaults for robot_config, params_file,
# nav_config, gps_topic, imu_topic, the detector topics and publish_robot_description. You can set each of these
# launch arguments directly.
#
# Robot description (harvest_hardware_description, apple-harvest repo), publish_robot_description:=true:
#   warthog  launch/warthog_state_publisher.launch.py (default on)
#   amiga    launch/robot_state_publisher.launch.py, the harvest URDF (default off: arm_control publishes it)
#
# Config files:
#   params_file    algorithm tuning (config/launch_config*.yaml)
#   robot_config   robot and machine values: frames, sensor topics, output paths (config/robot/*.yaml).
#                  The nodes load it after params_file, so its values win.
#
# Odometry:
#   use_ekf_fusion:=true (default)  local fusion only: ekf_local (wheel + IMU) -> /odometry/local, used by SLAM
#     use_gps_fusion:=true          adds GPS (global) fusion -> /odometry/global (gps_fusion_mode gps_heading | legacy)
#   use_ekf_fusion:=false           no robot_localization; SLAM uses /odometry/wheel (odom_reframer output)
#
# gps_fusion_mode:=gps_heading (default): GPS course heading in ekf_global, odom_reframer twist_from_pose and the GPS
# world-corrected map in save_best_particle_map. The launch turns on twist_from_pose and the world correction only
# while this GPS fusion runs (use_ekf_fusion and use_gps_fusion). gps_fusion_mode:=legacy turns all of them off.


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    robot_platform = LaunchConfiguration("robot_platform")
    params_file = LaunchConfiguration("params_file")
    robot_config = LaunchConfiguration("robot_config")
    use_ekf_fusion = LaunchConfiguration("use_ekf_fusion")
    use_gps_fusion = LaunchConfiguration("use_gps_fusion")
    gps_fusion_mode = LaunchConfiguration("gps_fusion_mode")
    publish_robot_description = LaunchConfiguration("publish_robot_description")

    pkg_share = get_package_share_directory("tree_template")
    config_dir = os.path.join(pkg_share, "config")
    ekf_fusion_launch = os.path.join(pkg_share, "launch", "ekf_fusion.launch.py")

    home = os.path.expanduser("~")
    default_width_estimation_path = os.path.join(home, "trunk_width_ws", "trunk_width_estimation")
    default_width_estimation_data_path = os.path.join(home, "trunk_width_ws", "width_estimation_package_data")
    default_detector_script = os.path.join(default_width_estimation_path, "scripts", "ros2", "ros_publisher_node.py")

    # Default value for each robot_platform
    def per_platform(warthog, amiga):
        return PythonExpression(["'", warthog, "' if '", robot_platform, "' == 'warthog' else '", amiga, "'"])

    # True only while the gps_heading global EKF is running
    gps_heading_active = PythonExpression([
        "'", use_ekf_fusion, "' == 'true' and '", use_gps_fusion, "' == 'true' and '",
        gps_fusion_mode, "' == 'gps_heading'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulated clock",
        ),

        DeclareLaunchArgument(
            "robot_platform",
            default_value="warthog",
            description="warthog (2023 Prosser bags) | amiga (2025 Prosser bags): config / topic / static TF defaults",
        ),

        DeclareLaunchArgument(
            "params_file",
            default_value=per_platform(os.path.join(config_dir, "launch_config.yaml"),
                                       os.path.join(config_dir, "launch_config_2025_with_polar_meas.yaml")),
            description="ROS2 params YAML with the algorithm tuning (launch_config.yaml)",
        ),

        DeclareLaunchArgument(
            "robot_config",
            default_value=per_platform(os.path.join(config_dir, "robot", "prosser_2023_warthog.yaml"),
                                       os.path.join(config_dir, "robot", "prosser_2025_amiga.yaml")),
            description="ROS2 params YAML with the robot / machine specifics (config/robot/*.yaml)",
        ),

        DeclareLaunchArgument(
            "use_ekf_fusion",
            default_value="true",
            description="Fuse wheel odometry + IMU with robot_localization ekf_local (local fusion only)",
        ),

        DeclareLaunchArgument(
            "use_gps_fusion",
            default_value="false",
            description="Also fuse GPS in ekf_global (-> /odometry/global). Needs use_ekf_fusion",
        ),

        DeclareLaunchArgument(
            "gps_fusion_mode",
            default_value="gps_heading",
            description="gps_heading (GPS course heading, world-corrected map) | legacy (IMU heading, all of that off)",
        ),

        DeclareLaunchArgument(
            "slam_odom_topic",
            default_value=PythonExpression(
                ["'/odometry/local' if '", use_ekf_fusion, "' == 'true' else '/odometry/wheel'"]),
            description="Odometry used by row_fast_slam (default: /odometry/local with the EKF, "
                        "/odometry/wheel without)",
        ),

        # Robot settings for ekf_fusion.launch.py (see that file)
        DeclareLaunchArgument("nav_config", default_value=per_platform("nav", "nav_2025"),
                              description="robot_localization config dir (nav = 2023 Warthog, nav_2025 = 2025 Amiga)"),
        DeclareLaunchArgument("gps_topic", default_value=per_platform("/fix", "/gps/pvt"),
                              description="GPS NavSatFix topic for navsat (legacy mode)"),
        DeclareLaunchArgument("imu_topic", default_value=per_platform("/imu/data", "/oak0/imu"),
                              description="IMU topic for navsat (legacy mode)"),

        # Optional nodes
        DeclareLaunchArgument("publish_robot_description", default_value=per_platform("true", "false"),
                              description="Start robot_state_publisher for robot_platform (harvest_hardware_description). "
                                          "false if the robot bringup already publishes it"),
        DeclareLaunchArgument("use_detector", default_value="true",
                              description="Start the external trunk detector (trunk_width_ws)"),
        DeclareLaunchArgument("use_slam_tf", default_value="false",
                              description="Run slam_odom_correction_tf (map->odom from SLAM; needs /odometry/global)"),
        DeclareLaunchArgument("use_trellis_from_extraction", default_value=per_platform("false", "true"),
                              description="Run trellis_from_rgbd_extraction (also publishes on fastslam_registry). "
                                          "Needs extraction_dir in the robot config"),

        DeclareLaunchArgument("width_estimation_package_path", default_value=default_width_estimation_path),
        DeclareLaunchArgument("width_estimation_package_data_path", default_value=default_width_estimation_data_path),
        DeclareLaunchArgument("detector_script", default_value=default_detector_script),
        DeclareLaunchArgument(
            "detector_rgb_topic",
            default_value=per_platform("/camera/color/image_raw", "/camera/base_camera/color/image_raw"),
            description="RGB image topic for the trunk detector",
        ),
        DeclareLaunchArgument(
            "detector_depth_topic",
            default_value=per_platform("/camera/aligned_depth_to_color/image_raw",
                                       "/camera/base_camera/aligned_depth_to_color/image_raw"),
            description="Depth image topic for the trunk detector (aligned to the RGB image)",
        ),

        # ----------------------------
        # Env vars for trunk_width_ws process
        # ----------------------------
        SetEnvironmentVariable(
            name="WIDTH_ESTIMATION_PACKAGE_PATH",
            value=LaunchConfiguration("width_estimation_package_path"),
        ),
        SetEnvironmentVariable(
            name="WIDTH_ESTIMATION_PACKAGE_DATA_PATH",
            value=LaunchConfiguration("width_estimation_package_data_path"),
        ),
        SetEnvironmentVariable(
            name="USE_SIM_TIME",
            value=use_sim_time,
        ),
        SetEnvironmentVariable(
            name="RGB_IMAGE_TOPIC",
            value=LaunchConfiguration("detector_rgb_topic"),
        ),
        SetEnvironmentVariable(
            name="DEPTH_IMAGE_TOPIC",
            value=LaunchConfiguration("detector_depth_topic"),
        ),

        # trunk_width_ws detector script (publishes TreeImageData; header.stamp = image time)
        ExecuteProcess(
            cmd=["python3", LaunchConfiguration("detector_script")],
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_detector")),
        ),

        # ----------------------------
        # Robot description for robot_platform (harvest_hardware_description)
        # ----------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("harvest_hardware_description"), "launch", "warthog_state_publisher.launch.py"])),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
            condition=IfCondition(PythonExpression(
                ["'", publish_robot_description, "' == 'true' and '", robot_platform, "' == 'warthog'"])),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("harvest_hardware_description"), "launch", "robot_state_publisher.launch.py"])),
            launch_arguments={"use_sim_time": use_sim_time, "use_fake_hardware": "true"}.items(),
            condition=IfCondition(PythonExpression(
                ["'", publish_robot_description, "' == 'true' and '", robot_platform, "' == 'amiga'"])),
        ),

        # ----------------------------
        # ekf_fusion launch (local EKF, optional GPS fusion)
        # ----------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(ekf_fusion_launch),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "robot_platform": robot_platform,
                "use_gps_fusion": use_gps_fusion,
                "mode": gps_fusion_mode,
                "nav_config": LaunchConfiguration("nav_config"),
                "gps_topic": LaunchConfiguration("gps_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
            }.items(),
            condition=IfCondition(use_ekf_fusion),
        ),

        # ----------------------------
        # tree_template nodes
        # ----------------------------
        # MoveIt trellis collision objects from the trunk registry
        Node(
            package="tree_template",
            executable="tree_template",
            name="tree_scene_node",
            output="screen",
            parameters=[params_file, robot_config, {"use_sim_time": use_sim_time}],
        ),

        # TreeImageData -> TrunkInfo measurements in the robot base frame
        Node(
            package="tree_template",
            executable="trunk_detection_relay",
            name="trunk_detection_relay",
            output="screen",
            parameters=[params_file, robot_config, {"use_sim_time": use_sim_time}],
        ),

        # SLAM map -> odom TF
        Node(
            package="tree_template",
            executable="slam_odom_correction_tf",
            name="slam_odom_correction_tf",
            output="screen",
            parameters=[params_file, robot_config, {"use_sim_time": use_sim_time}],
            condition=IfCondition(LaunchConfiguration("use_slam_tf")),
        ),

        # Odometry reframer (robot odometry -> /odometry/wheel)
        Node(
            package="tree_template",
            executable="odom_reframer_calibrated",
            name="odom_reframer",
            output="screen",
            parameters=[
                params_file,
                robot_config,
                {"twist_from_pose": gps_heading_active},  # only the gps_heading global EKF uses it
                {"use_sim_time": use_sim_time},
            ],
        ),

        # Row FastSLAM
        Node(
            package="tree_template",
            executable="row_fast_slam",
            name="row_fast_slam",
            output="screen",
            parameters=[
                params_file,
                robot_config,
                {"odom_topic": LaunchConfiguration("slam_odom_topic")},
                {"use_sim_time": use_sim_time},
            ],
        ),

        # Saves the SLAM map (and the GPS world-corrected map in gps_heading mode)
        Node(
            package="tree_template",
            executable="save_best_particle_map",
            name="save_best_particle_map",
            output="screen",
            parameters=[
                params_file,
                robot_config,
                {"world_correction.enable": gps_heading_active},
                {"use_sim_time": use_sim_time},
            ],
        ),

        # Trellis template map from an RGB-D extraction of a saved SLAM map
        Node(
            package="tree_template",
            executable="trellis_from_rgbd_extraction",
            name="trellis_from_rgbd_extraction",
            output="screen",
            parameters=[params_file, robot_config, {"use_sim_time": use_sim_time}],
            condition=IfCondition(LaunchConfiguration("use_trellis_from_extraction")),
        ),
    ])
