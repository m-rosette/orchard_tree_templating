import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# mode:=legacy      ekf_local + navsat (IMU heading + yaw_offset, reads /odometry/local) + ekf_global fusing
#                   /odometry/local poses + /odometry/gps. Previous behaviour (default).
# mode:=gps_heading ekf_local unchanged; navsat reads /odometry/global, heading from GPS course (/gps/course_imu),
#                   fix from /fix_rtk (realistic covariance); ekf_global fuses wheel velocities + IMU yaw rate +
#                   GPS position + GPS course heading (config/nav/ekf_global_gps_heading.yaml). Needs odom_reframer
#                   twist_from_pose: true and use_now_stamp: false. Validated on the 2023 Warthog bags: /odometry/global
#                   scale 0.996-0.999 vs RTK GPS; post-SLAM world correction (save_best_particle_map) -25% along-row error.


def _nodes(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration("use_sim_time")
    mode = LaunchConfiguration("mode").perform(context)
    if mode not in ("legacy", "gps_heading"):
        raise RuntimeError(f"gps_fusion.launch.py: unknown mode '{mode}' (legacy | gps_heading)")
    cfg = os.path.join(get_package_share_directory("tree_template"), "config", "nav")
    gps_heading = mode == "gps_heading"

    nodes = [
        # Local EKF: fuses wheel odom (/odometry/wheel from odom_reframer) + IMU (same in both modes)
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_local",
            output="screen",
            parameters=[
                os.path.join(cfg, "ekf_local.yaml"),
                {"use_sim_time": use_sim_time},
                {"publish_tf": True},
            ],
            remappings=[("odometry/filtered", "/odometry/local")],
        ),
        # NavSat transform: GPS + heading + odometry -> /odometry/gps
        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[
                os.path.join(cfg, "navsat_gps_heading.yaml" if gps_heading else "navsat.yaml"),
                {"use_sim_time": use_sim_time},
            ],
            remappings=[
                ("gps/fix", "/fix_rtk" if gps_heading else "/fix"),
                ("imu", "/gps/course_imu" if gps_heading else "/imu/data"),
                ("odometry/filtered", "/odometry/global" if gps_heading else "/odometry/local"),
                ("odometry/gps", "/odometry/gps"),
            ],
        ),
        # Global EKF -> /odometry/global (+ map->odom TF)
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_global",
            output="screen",
            parameters=[
                os.path.join(cfg, "ekf_global_gps_heading.yaml" if gps_heading else "ekf_global.yaml"),
                {"use_sim_time": use_sim_time},
                {"publish_tf": True},
            ],
            remappings=[("odometry/filtered", "/odometry/global")],
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
    ]
    if gps_heading:
        helper_params = [os.path.join(cfg, "gps_heading_nodes.yaml"), {"use_sim_time": use_sim_time}]
        nodes += [
            Node(package="tree_template", executable="fix_cov_relay", name="fix_cov_relay",
                 output="screen", parameters=helper_params),
            Node(package="tree_template", executable="gps_course_heading", name="gps_course_heading",
                 output="screen", parameters=helper_params),
        ]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("mode", default_value="legacy",
                              description="legacy | gps_heading (see header comment)"),
        OpaqueFunction(function=_nodes),
    ])
