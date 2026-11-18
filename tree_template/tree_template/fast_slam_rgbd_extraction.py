#!/usr/bin/env python3
"""
fast_slam_rgbd_extraction.py

Post-processing script: given a SLAM output YAML (with committed tree locations
and robot poses+timestamps) and a ROS2 bag (sqlite3 storage) containing
RealSense RGB-D data, this script finds — for each committed tree — the robot
pose where the tree is perpendicularly in front of the robot (broadside), then
extracts and saves the nearest RGB-D frame pair (color + depth) from the bag at
that timestamp.

Supports TWO camera streams simultaneously.  Each camera's frames are saved
into a named subdirectory under the tree folder.

Output structure:
    output_dir/
        tree_000/
            meta.yaml
            cam1/                        (single frame, buffer_radius=0)
                camera_info.yaml         color + depth intrinsics, depth->color extrinsics
                color.png
                depth.png                (16-bit, millimeters, depth camera pixels)
                depth_aligned.png        (16-bit, millimeters, registered to color.png)
                depth_colormap.png
            cam2/                        (5-frame buffer, --cam2_buffer_radius 2)
                camera_info.yaml
                frame_-2/color.png, depth.png, depth_aligned.png, depth_colormap.png
                ...
                frame_2/ ...
        tree_001/
            ...

meta.yaml has, for each saved frame: the color and depth header stamps, |color - depth| and sync_ok
(--max_color_depth_dt_s), the time from the broadside time, the odometry pose at the color stamp
(odom_pose_xyt, --odom_topic) and the SLAM robot_path pose (slam_path_pose_xyt). Buffer frames also have
base_delta_from_frame0_xyt: the robot motion from frame_0 in the base frame (--odom_to_base_yaw_deg).
The depth frame of each pair is the one with the header stamp nearest to the color frame.

The SLAM robot_path follows the best particle of each moment, so it can jump by more than 1 m between two
poses. Use the odometry for the motion between frames.

Broadside pose (--broadside_source odom, default): the odometry, put into the SLAM world frame by a trimmed
rigid fit to the robot_path near the tree (odom_alignment.py), at the time when the tree is straight to the side
(fwd = 0). meta.yaml keeps the robot_path result as slam_path_broadside and the fit as broadside_fit.

depth_aligned.png: depth registered to the color image (depth_registration.py, the librealsense align method).
The depth->color extrinsics come from --camN_depth_to_color (calibrate_depth_to_color output, see
config/camera_extrinsics/) or from the bag extrinsics topic. For a depth topic that is already aligned
(aligned_depth_to_color, or --camN_depth_aligned), depth_aligned.png is a copy of depth.png.

MEMORY MODEL
------------
This script queries the rosbag2 sqlite3 database DIRECTLY, one tree at a time.
For each tree and each (color/depth) topic, it runs a single SQL query
restricted to a small time window around the broadside timestamp, then STREAMS
the matching rows ONE AT A TIME from sqlite:

    for t_ns, data in cursor:        # one row fetched at a time, not buffered
        msg = deserialize_message(data, msg_type)
        ...keep only the single best-so-far candidate, discard the rest...

Only the single best-matching message per topic is ever converted to an image
array, and that array is written to disk and discarded before moving on to the
next camera/tree. Peak memory is therefore a small, constant number of images,
independent of bag size -- there is no "build a cache of all frames" step.

Clock mismatch handling:
    Image topics may carry header.stamp values in a different epoch than the
    SLAM YAML's robot_path timestamps (e.g. the bag was replayed/re-recorded
    later, so the bag's *receive* timestamps are in a different epoch than the
    image *header* timestamps, which retain the original recording's clock).

    --image_time_offset_s <value>
        The raw image header stamp expected at SLAM timestamp T is
        (T + image_time_offset_s). If image headers and SLAM timestamps are
        already in the same clock (common for re-processed bags), use 0.0.

    --auto_detect_clock_offset
        Computes offset = first_image_header_stamp - first_odom_header_stamp.
        Only useful when both topics share a clock domain at the start of the
        bag; for re-processed bags prefer --image_time_offset_s 0.0.

Usage:
    python fast_slam_rgbd_extraction.py \\
        --yaml slam_trunks_20260331_174636.yaml \\
        --bag  /path/to/bag_dir/ \\
        --output_dir rgbd_at_trees \\
        --cam1_color_topic /camera/mast_camera/color/image_raw \\
        --cam1_depth_topic /camera/mast_camera/recovered_depth/image_raw \\
        --cam2_color_topic /camera/base_camera/color/image_raw \\
        --cam2_depth_topic /camera/base_camera/depth/image_rect_raw \\
        --cam1_depth_to_color config/camera_extrinsics/prosser_2025_mast_camera.yaml \\
        --cam2_depth_to_color config/camera_extrinsics/prosser_2025_base_camera.yaml \\
        --cam2_buffer_radius 2 \\
        --image_time_offset_s 0.0 --max_time_delta_s 5.0
"""

import argparse
import os
import math
import sqlite3
import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tree_template.odom_alignment import broadside_from_odom, world_poses
from tree_template.trellis_geometry import pose_tf
from tree_template.depth_registration import (
    camera_info_to_dict,
    camera_matrix,
    check_no_distortion,
    extrinsics_from_realsense_msg,
    load_extrinsics_yaml,
    register_depth_to_color,
)


# ---------------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------------

class CameraConfig:
    """Holds topic names and output directory name for one camera."""

    def __init__(self, name: str, color_topic: str, depth_topic: str,
                 buffer_radius: int = 0, color_info_topic: str = None,
                 depth_info_topic: str = None, depth_to_color_yaml: str = None,
                 depth_is_aligned: bool = False, cli_prefix: str = "cam"):
        self.name          = name
        self.cli_prefix    = cli_prefix
        self.color_topic   = color_topic
        self.depth_topic   = depth_topic
        self.buffer_radius = buffer_radius  # 0 = single frame; N = save 2N+1 frames
        # camera_info topics. Default: camera_info next to the image topic.
        self.color_info_topic = color_info_topic or _sibling_info_topic(color_topic)
        self.depth_info_topic = depth_info_topic or _sibling_info_topic(depth_topic)
        self.depth_to_color_yaml = depth_to_color_yaml   # calibrate_depth_to_color output
        # True if the depth topic is already registered to the color image
        self.depth_is_aligned = depth_is_aligned or "aligned_depth_to_color" in depth_topic
        # Set in main(): color/depth camera_info dicts, depth->color R and t (mm), registration mode
        self.color_info = None
        self.depth_info = None
        self.R = None
        self.t_mm = None
        self.align_mode = "none"     # register | already_aligned | none
        self.align_source = None

    @property
    def topics(self):
        return [self.color_topic, self.depth_topic]

    def __repr__(self):
        return (f"CameraConfig(name={self.name!r}, "
                f"color={self.color_topic!r}, depth={self.depth_topic!r})")


def _sibling_info_topic(image_topic: str) -> str:
    """/a/b/image_raw -> /a/b/camera_info"""
    return image_topic.rsplit("/", 1)[0] + "/camera_info"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def last_pose_per_stamp(robot_poses):
    """
    Keep only the last row for each time stamp (robot_poses (N, 4): x, y, yaw, t; sorted by t).
    save_best_particle_map records /odom_slam_best: when the best particle changes after a measurement update,
    a second pose with the same stamp follows. The last one is the pose after the update.
    """
    t = robot_poses[:, 3]
    keep = np.append(np.diff(t) > 0, True)
    return robot_poses[keep]


def interp_robot_pose(robot_poses, t):
    """
    Return ([x, y, yaw], in_range) of the robot path (N, 4: x, y, yaw, t) at time t (SLAM clock).
    Linear interpolation; the yaw is unwrapped first. Outside the path, the nearest end pose and in_range=False.
    """
    ts = robot_poses[:, 3]
    in_range = bool(ts[0] <= t <= ts[-1])
    x = float(np.interp(t, ts, robot_poses[:, 0]))
    y = float(np.interp(t, ts, robot_poses[:, 1]))
    yaw = float(np.interp(t, ts, np.unwrap(robot_poses[:, 2])))
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return [x, y, yaw], in_range


def perpendicular_distance_along_path(robot_pos, robot_yaw, tree_pos):
    """
    Returns signed forward and lateral distances from the robot to the tree.

    Convention (matches SLAM frame):
      robot forward = +x at heading robot_yaw
      robot left    = +y

    Returns:
        fwd_dist: positive = tree is ahead of robot
        lat_dist: positive = tree is to the left
    """
    dx = tree_pos[0] - robot_pos[0]
    dy = tree_pos[1] - robot_pos[1]
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    fwd_dist =  cos_y * dx + sin_y * dy
    lat_dist = -sin_y * dx + cos_y * dy
    return fwd_dist, lat_dist


def find_broadside_pose_index(robot_poses, tree_xy):
    """
    Find the index in robot_poses where |fwd_dist| is minimised --
    i.e. the tree is most nearly perpendicular (broadside) to the robot.

    robot_poses: (N, 4) array [x, y, yaw, t]
    tree_xy:     (2,) array [x, y]
    """
    fwd_dists = np.zeros(len(robot_poses))
    lat_dists = np.zeros(len(robot_poses))
    for i, pose in enumerate(robot_poses):
        fwd_dists[i], lat_dists[i] = perpendicular_distance_along_path(
            pose[:2], pose[2], tree_xy
        )
    best_idx = int(np.argmin(np.abs(fwd_dists)))
    return best_idx, fwd_dists[best_idx], lat_dists[best_idx]


def _yaw_from_quat(q):
    return float(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("ZYX")[0])


def load_odom_series(con, topic):
    """
    Return (N, 4) [x, y, yaw, receive_s] for all nav_msgs/Odometry messages on topic, or None
    (the same columns as the SLAM robot_path, so interp_robot_pose works on it).
    The time is the bag receive time: robot odometry header stamps can use a different clock (the farm-ng
    /filter/state stamps do), and the receive time is in the host clock, like the image header stamps.
    """
    row = con.execute("SELECT id, type FROM topics WHERE name=?", (topic,)).fetchone()
    if row is None:
        return None
    tid, typ = row
    T = get_message(typ)
    out = []
    for t_ns, data in con.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
        m = deserialize_message(bytes(data), T)
        p = m.pose.pose
        out.append((p.position.x, p.position.y, _yaw_from_quat(p.orientation), t_ns * 1e-9))
    return np.array(out, dtype=np.float64) if out else None


def base_delta(pose0, pose1, odom_to_base_yaw):
    """
    Motion from pose0 to pose1 (odometry [x, y, yaw]) in the robot base frame at pose0: [dx, dy, dyaw].
    odom_to_base_yaw: yaw (rad) from the odometry child frame to the robot base frame.
    """
    # Points in base1 -> child1 -> odometry frame -> child0 -> base0 (skimage: A + B = A, then B).
    # c_b: base frame -> odometry child frame.
    c_b = pose_tf(0.0, 0.0, odom_to_base_yaw)
    rel = c_b + pose_tf(*pose1) + pose_tf(*pose0).inverse + c_b.inverse
    return [float(rel.translation[0]), float(rel.translation[1]), float(rel.rotation)]


# ---------------------------------------------------------------------------
# SQLite bag helpers
# ---------------------------------------------------------------------------

def _find_sqlite3_db(bag_path: str):
    """Return path to the .db3 file inside bag_path, or None if not found."""
    import glob
    if bag_path.endswith(".db3") and os.path.isfile(bag_path):
        return bag_path
    candidates = glob.glob(os.path.join(bag_path, "*.db3"))
    return candidates[0] if candidates else None


def _header_stamp_s(msg):
    """Return msg.header.stamp as float seconds."""
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def _topic_id_and_type_maps(con, topics):
    """
    Returns (topic_id_map, type_map) for the given topic name list.
    Topics not present in the bag are omitted (with a warning printed).
    """
    cur = con.cursor()
    cur.execute("SELECT id, name, type FROM topics")
    rows = cur.fetchall()
    name_to_id   = {name: tid for tid, name, _ in rows}
    name_to_type = {name: typ for _, name, typ in rows}

    topic_id_map = {}
    type_map = {}
    for t in topics:
        if t not in name_to_id:
            print(f"  [WARN] Topic '{t}' not found in bag.")
            continue
        topic_id_map[t] = name_to_id[t]
        type_map[t] = get_message(name_to_type[t])
    return topic_id_map, type_map


def _first_last_header_stamp(con, topic_id, msg_type):
    """
    Return ((h0_s, t0_ns), (h1_s, t1_ns)) for the first and last message of
    topic_id, where h is the image header.stamp (seconds) and t is the bag
    receive timestamp (ns). Returns None if the topic has < 2 distinct rows.
    """
    cur = con.cursor()
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp ASC LIMIT 1", (topic_id,))
    row0 = cur.fetchone()
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp DESC LIMIT 1", (topic_id,))
    row1 = cur.fetchone()
    if not row0 or not row1 or row0[0] == row1[0]:
        return None

    h0 = _header_stamp_s(deserialize_message(bytes(row0[1]), msg_type))
    h1 = _header_stamp_s(deserialize_message(bytes(row1[1]), msg_type))
    return (h0, row0[0]), (h1, row1[0])


def calibrate_header_to_receive(con, topic_id, msg_type):
    """
    Compute the linear mapping from image header.stamp (seconds) to bag
    receive timestamp (ns):

        t_receive_ns ~= t0_ns + (h_s - h0_s) * slope_ns_per_s

    slope ~= 1e9 for a real-time bag, < 1e9 for a slowed-down replay.
    Returns (h0_s, t0_ns, slope) or None if calibration isn't possible.
    """
    ends = _first_last_header_stamp(con, topic_id, msg_type)
    if ends is None:
        return None
    (h0, t0_ns), (h1, t1_ns) = ends
    if h1 == h0:
        return None
    slope = (t1_ns - t0_ns) / (h1 - h0)
    return h0, t0_ns, slope


def find_nearest_in_window(con, topic_id, msg_type,
                            t_min_ns, t_max_ns,
                            target_header_t, max_delta_s):
    """
    Stream rows of `topic_id` whose receive-timestamp is in
    [t_min_ns, t_max_ns], deserializing ONE AT A TIME and keeping only the
    single best-so-far candidate (smallest |header_stamp - target_header_t|).
    Every non-best candidate is dropped immediately -- it is never appended
    to any list, so it becomes garbage-collectable before the next row is
    even fetched from sqlite.

    Returns (best_msg, best_dt, best_t_ns, n_considered):
        best_msg  : the winning deserialized message, or None if nothing in
                    the window was within max_delta_s
        best_dt   : |header_stamp - target_header_t| for the winner (or for
                    the closest candidate found, even if outside tolerance),
                    or None if the window contained zero rows
        best_t_ns : bag receive timestamp (ns) of the best candidate, or None
        n_considered : number of rows streamed from sqlite for this query
    """
    cur = con.cursor()
    cur.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id = ? AND timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (topic_id, t_min_ns, t_max_ns),
    )

    best_msg, best_dt, best_t_ns, n = None, None, None, 0
    for t_ns, data in cur:  # one row at a time from the sqlite3 C API
        n += 1
        msg = deserialize_message(bytes(data), msg_type)
        dt = abs(_header_stamp_s(msg) - target_header_t)
        if best_dt is None or dt < best_dt:
            best_msg, best_dt, best_t_ns = msg, dt, t_ns
        # else: `msg` falls out of scope here and is discarded immediately.

    if best_msg is None or best_dt > max_delta_s:
        return None, best_dt, best_t_ns, n
    return best_msg, best_dt, best_t_ns, n


def fetch_adjacent_frames(con, topic_id, msg_type, center_receive_ns,
                           n_before, n_after):
    """
    Fetch up to n_before frames immediately before and n_after frames
    immediately after center_receive_ns in bag receive-time order.

    Returns (before, after):
        before : list of (receive_ns, deserialized msg), oldest-first (len <= n_before)
        after  : list of (receive_ns, deserialized msg), chronologically (len <= n_after)
    """
    cur = con.cursor()

    cur.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id=? AND timestamp < ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (topic_id, center_receive_ns, n_before),
    )
    rows_before = cur.fetchall()
    rows_before.reverse()  # oldest-first
    before = [(t_ns, deserialize_message(bytes(data), msg_type)) for t_ns, data in rows_before]

    cur.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id=? AND timestamp > ? "
        "ORDER BY timestamp ASC LIMIT ?",
        (topic_id, center_receive_ns, n_after),
    )
    after = [(t_ns, deserialize_message(bytes(data), msg_type)) for t_ns, data in cur.fetchall()]

    return before, after


# ---------------------------------------------------------------------------
# Image conversion / saving
# ---------------------------------------------------------------------------

def ros_image_to_cv2(msg):
    """Convert a sensor_msgs/Image to a numpy array."""
    enc    = msg.encoding
    h, w   = msg.height, msg.width

    if enc == "rgb8":
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif enc == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
    elif enc == "mono8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
    elif enc in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
    elif enc == "32FC1":
        data = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w))
        return (data * 1000.0).astype(np.uint16)  # metres -> millimetres
    else:
        raise ValueError(f"Unsupported image encoding: {enc}")


def save_camera_images(cam_dir, color_img, depth_img, depth_aligned_img=None):
    """
    Write color.png, depth.png (16-bit), and depth_colormap.png into cam_dir.
    If depth_aligned_img is not None, also write depth_aligned.png (16-bit, registered to color.png).
    """
    os.makedirs(cam_dir, exist_ok=True)
    cv2.imwrite(os.path.join(cam_dir, "color.png"), color_img)
    cv2.imwrite(os.path.join(cam_dir, "depth.png"), depth_img)  # 16-bit PNG
    if depth_aligned_img is not None:
        cv2.imwrite(os.path.join(cam_dir, "depth_aligned.png"), depth_aligned_img)

    depth_norm  = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(cam_dir, "depth_colormap.png"), depth_color)


# ---------------------------------------------------------------------------
# Camera calibration data and per-frame records
# ---------------------------------------------------------------------------

def _first_msg_on_topic(con, name):
    """First message on topic `name` (deserialized), or None if the topic is missing or empty."""
    row = con.execute("SELECT id, type FROM topics WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    tid, typ = row
    data = con.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1", (tid,)).fetchone()
    return deserialize_message(bytes(data[0]), get_message(typ)) if data else None


def setup_camera_alignment(con, cam):
    """
    Read the camera_info of cam from the bag and set how depth_aligned.png is made:
      already_aligned  the depth topic is registered to color (aligned_depth_to_color, or --camN_depth_aligned)
      register         register depth to color with the extrinsics from --camN_depth_to_color, or from the bag
                       extrinsics topic <ns>/extrinsics/depth_to_color if it has data
      none             no depth_aligned.png (a warning tells why)
    """
    cam.color_info = camera_info_to_dict(_first_msg_on_topic(con, cam.color_info_topic))
    cam.depth_info = camera_info_to_dict(_first_msg_on_topic(con, cam.depth_info_topic))
    if cam.color_info is None:
        print(f"  [WARN] {cam.name}: no camera_info on {cam.color_info_topic}")
    if cam.depth_info is None and not cam.depth_is_aligned:
        print(f"  [WARN] {cam.name}: no camera_info on {cam.depth_info_topic}")

    if cam.depth_is_aligned:
        cam.align_mode, cam.align_source = "already_aligned", cam.depth_topic
        return

    R = t_m = None
    if cam.depth_to_color_yaml:
        R, t_m, _ = load_extrinsics_yaml(cam.depth_to_color_yaml)
        cam.align_source = os.path.abspath(cam.depth_to_color_yaml)
    else:
        ns = cam.depth_topic.rsplit("/", 2)[0]       # /camera/x/depth/image_rect_raw -> /camera/x
        ext_topic = f"{ns}/extrinsics/depth_to_color"
        msg = _first_msg_on_topic(con, ext_topic)
        if msg is not None:
            R, t_m = extrinsics_from_realsense_msg(msg)
            cam.align_source = ext_topic

    reasons = []
    if R is None:
        reasons.append("no depth->color extrinsics (give --{}_depth_to_color)".format(cam.cli_prefix))
    if cam.color_info is None or cam.depth_info is None:
        reasons.append("camera_info missing")
    else:
        for which, info in (("color", cam.color_info), ("depth", cam.depth_info)):
            if not check_no_distortion(info):
                print(f"  [WARN] {cam.name}: {which} camera_info has distortion {info['d']}; "
                      f"the registration ignores it.")
    if reasons:
        cam.align_mode = "none"
        print(f"  [WARN] {cam.name}: depth_aligned.png not made: {'; '.join(reasons)}")
        return
    cam.align_mode = "register"
    cam.R = R
    cam.t_mm = np.asarray(t_m) * 1000.0


def write_camera_info_yaml(cam_dir, cam):
    """Write camera_info.yaml (color and depth intrinsics, depth->color extrinsics) into cam_dir."""
    data = {
        "color": cam.color_info,
        "depth": cam.depth_info,
        "depth_aligned": {
            "mode": cam.align_mode,
            "source": cam.align_source,
        },
    }
    if cam.align_mode == "register":
        data["depth_aligned"]["rotation"] = [float(x) for x in np.asarray(cam.R).reshape(-1)]
        data["depth_aligned"]["translation_m"] = [float(x) for x in np.asarray(cam.t_mm) / 1000.0]
    os.makedirs(cam_dir, exist_ok=True)
    with open(os.path.join(cam_dir, "camera_info.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


def save_frame(frame_dir, cam, color_msg, depth_msg, ctx):
    """
    Write the images of one color/depth pair and return its meta dict:
      color_stamp_s / depth_stamp_s   image header stamps (image clock)
      color_depth_dt_s                |depth stamp - color stamp|; sync_ok if <= --max_color_depth_dt_s
      t_from_broadside_s              color stamp - broadside time (image clock)
      world_pose_xyt                  robot base pose in the SLAM world frame at the color stamp: the odometry
                                      through the broadside fit of this tree (--broadside_source odom only)
      odom_pose_xyt                   --odom_topic pose at the color stamp (smooth; use it for motion between frames)
      slam_path_pose_xyt              SLAM robot_path pose at the color stamp (SLAM clock = stamp - clock offset).
                                      It follows the best particle of each moment and can jump by more than 1 m.
      depth_aligned                   True if depth_aligned.png was written
    """
    t_c = _header_stamp_s(color_msg)
    t_d = _header_stamp_s(depth_msg)
    color_img = ros_image_to_cv2(color_msg)
    depth_img = ros_image_to_cv2(depth_msg)

    aligned = None
    if cam.align_mode == "register":
        K_d, K_c = camera_matrix(cam.depth_info), camera_matrix(cam.color_info)
        aligned = register_depth_to_color(depth_img, K_d, K_c, cam.R, cam.t_mm, color_img.shape[:2])
    elif cam.align_mode == "already_aligned":
        if depth_img.shape[:2] == color_img.shape[:2]:
            aligned = depth_img
        else:
            print(f"  [WARN] {cam.name}: depth {depth_img.shape[:2]} declared aligned but color is "
                  f"{color_img.shape[:2]}; depth_aligned.png not written")

    save_camera_images(frame_dir, color_img, depth_img, aligned)

    pose, in_range = interp_robot_pose(ctx["robot_poses"], t_c - ctx["offset"])
    odom_pose = world_pose = None
    if ctx["odom"] is not None:
        t_rx = ctx["header_to_receive_s"](t_c)
        odom_pose, odom_ok = interp_robot_pose(ctx["odom"], t_rx)
        if ctx.get("world_fit") is not None:
            world_pose = world_poses(ctx["world_fit"], ctx["odom"], np.array([t_rx]), ctx["odom_to_base_yaw"])[0]
    dt_cd = abs(t_d - t_c)
    meta = {
        "saved": True,
        "color_stamp_s": t_c,
        "depth_stamp_s": t_d,
        "color_depth_dt_s": float(dt_cd),
        "sync_ok": bool(dt_cd <= ctx["max_sync_s"]),
        "t_from_broadside_s": float(t_c - ctx["raw_target"]),
        "slam_path_pose_xyt": [float(v) for v in pose],
        "depth_aligned": aligned is not None,
    }
    if not in_range:
        meta["slam_path_pose_note"] = "color stamp outside the robot path; nearest end pose"
    if world_pose is not None:
        meta["world_pose_xyt"] = [float(v) for v in world_pose]
    if odom_pose is not None:
        meta["odom_pose_xyt"] = [float(v) for v in odom_pose]
        if not odom_ok:
            meta["odom_pose_note"] = "color stamp outside the odometry; nearest end pose"
    if not meta["sync_ok"]:
        print(f"  [WARN] {cam.name} {os.path.basename(frame_dir)}: color/depth stamps "
              f"{dt_cd*1000:.0f} ms apart (> {ctx['max_sync_s']*1000:.0f} ms)")
    return meta


def nearest_depth_for_color(con, cam, topic_id_map, type_map, color_rx_ns, color_msg, max_delta_s, slope):
    """Depth frame with the header stamp nearest to the color stamp, from a receive-time window."""
    pad = int((max_delta_s + 0.5) * abs(slope))
    msg, dt, t_ns, n = find_nearest_in_window(
        con, topic_id_map[cam.depth_topic], type_map[cam.depth_topic],
        color_rx_ns - pad, color_rx_ns + pad, _header_stamp_s(color_msg), max_delta_s,
    )
    return msg, dt, t_ns, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yaml",       required=True,  help="SLAM output YAML file")
    parser.add_argument("--bag",        required=True,  help="ROS2 bag directory (sqlite3 storage)")
    parser.add_argument("--output_dir", default="/home/marcus/apple_harvest_ws/data/rgbd_at_trees_oct_2025_v3")

    # -- Camera 1 --------------------------------------------------------------
    cam1_grp = parser.add_argument_group("Camera 1")
    cam1_grp.add_argument(
        "--cam1_name",        default="mast_cam",
        help="Subdirectory name used for camera 1 images (default: mast_cam)")
    cam1_grp.add_argument(
        "--cam1_color_topic", default="/camera/mast_camera/color/image_raw",
        help="Color image topic for camera 1")
    cam1_grp.add_argument(
        "--cam1_depth_topic", default="/camera/mast_camera/recovered_depth/image_raw",
        help="Depth image topic for camera 1")

    # -- Camera 2 --------------------------------------------------------------
    cam2_grp = parser.add_argument_group("Camera 2")
    cam2_grp.add_argument(
        "--cam2_name",        default="base_cam",
        help="Subdirectory name used for camera 2 images (default: base_cam)")
    cam2_grp.add_argument(
        "--cam2_color_topic", default="/camera/base_camera/color/image_raw",
        help="Color image topic for camera 2")
    cam2_grp.add_argument(
        "--cam2_depth_topic", default="/camera/base_camera/depth/image_rect_raw",
        help="Depth image topic for camera 2")
    cam2_grp.add_argument(
        "--cam2_buffer_radius", type=int, default=0,
        help="Number of frames to extract on each side of the broadside frame "
             "for camera 2 (0 = single frame, 2 = 5-frame buffer). "
             "Frames are saved into frame_-N/ ... frame_0/ ... frame_N/ "
             "subdirectories under the camera folder.")

    # -- Camera calibration data (both cameras) ----------------------------------
    for n in ("cam1", "cam2"):
        g = parser.add_argument_group(f"{n} calibration")
        g.add_argument(f"--{n}_color_info_topic", default=None,
                       help="color camera_info topic (default: camera_info next to the color topic)")
        g.add_argument(f"--{n}_depth_info_topic", default=None,
                       help="depth camera_info topic (default: camera_info next to the depth topic)")
        g.add_argument(f"--{n}_depth_to_color", default=None,
                       help="depth->color extrinsics yaml from calibrate_depth_to_color "
                            "(config/camera_extrinsics/). Default: the bag extrinsics topic, if it has data")
        g.add_argument(f"--{n}_depth_aligned", action="store_true",
                       help="the depth topic is already registered to color (automatic for "
                            "aligned_depth_to_color topics)")

    # -- Timing / filtering ------------------------------------------------------
    parser.add_argument(
        "--odom_topic", default="/filter/state",
        help="Odometry topic (nav_msgs/Odometry). Gives odom_pose_xyt for each frame and the buffer frame motion "
             "(base_delta_from_frame0_xyt). Also used by --auto_detect_clock_offset")
    parser.add_argument(
        "--broadside_source", choices=["odom", "slam_path"], default="odom",
        help="odom (default): the broadside pose comes from --odom_topic, put into the SLAM world frame by a local "
             "trimmed fit to the SLAM robot_path. slam_path: the robot_path pose with the smallest |fwd| (it can "
             "be wrong by more than 1 m, because the path jumps when the best particle changes)")
    parser.add_argument(
        "--odom_time_offset_s", type=float, default=0.0,
        help="Odometry time (bag receive time) - SLAM robot_path time. 0 when SLAM ran on a replay of this bag "
             "with the bag clock")
    parser.add_argument(
        "--broadside_window_s", type=float, default=10.0,
        help="Half-width (s) of the robot_path window for the translation of the odometry -> world fit")
    parser.add_argument(
        "--broadside_rotation_window_s", type=float, default=60.0,
        help="Half-width (s) of the robot_path window for the rotation of the odometry -> world fit")
    parser.add_argument(
        "--odom_to_base_yaw_deg", type=float, default=0.0,
        help="Yaw from the odometry child frame to the robot base frame. 180 for the 2025 Amiga /filter/state "
             "(farm-ng forward = amiga__base -x; the same as odom_reframer rotate_pos_deg 180); 0 for the Warthog")
    parser.add_argument(
        "--max_time_delta_s", type=float, default=1.0,
        help="Max allowed time gap (s) between broadside pose and image frame")
    parser.add_argument(
        "--filter_pad_s", type=float, default=None,
        help="Half-width (s, in image-header-stamp space) of the SQL query "
             "window around each broadside time. Must be >= "
             "--max_time_delta_s (auto-bumped with a warning if not). "
             "Default: max_time_delta_s + 0.5")
    parser.add_argument(
        "--max_color_depth_dt_s", type=float, default=0.05,
        help="Frames with color and depth header stamps further apart than this get sync_ok: false in meta.yaml")
    parser.add_argument(
        "--perpendicular_window_m", type=float, default=None,
        help="Skip trees whose lateral distance exceeds this (metres)")

    clock_grp = parser.add_mutually_exclusive_group()
    clock_grp.add_argument(
        "--image_time_offset_s", type=float, default=None,
        help="Raw image header stamp expected at SLAM time T is "
             "(T + this value). Use 0.0 if image headers and SLAM "
             "timestamps already share a clock.",
    )
    clock_grp.add_argument(
        "--auto_detect_clock_offset", action="store_true",
        help="Compute offset = first_image_header_stamp - "
             "first_odom_header_stamp.",
    )

    args = parser.parse_args()

    cameras = [
        CameraConfig(args.cam1_name, args.cam1_color_topic, args.cam1_depth_topic,
                     color_info_topic=args.cam1_color_info_topic, depth_info_topic=args.cam1_depth_info_topic,
                     depth_to_color_yaml=args.cam1_depth_to_color, depth_is_aligned=args.cam1_depth_aligned,
                     cli_prefix="cam1"),
        CameraConfig(args.cam2_name, args.cam2_color_topic, args.cam2_depth_topic,
                     buffer_radius=args.cam2_buffer_radius,
                     color_info_topic=args.cam2_color_info_topic, depth_info_topic=args.cam2_depth_info_topic,
                     depth_to_color_yaml=args.cam2_depth_to_color, depth_is_aligned=args.cam2_depth_aligned,
                     cli_prefix="cam2"),
    ]

    print("Camera configuration:")
    for cam in cameras:
        print(f"  [{cam.name}]  color: {cam.color_topic}")
        print(f"  [{cam.name}]  depth: {cam.depth_topic}")

    # ------------------------------------------------------------------
    # 1. Load YAML, find broadside pose per tree
    # ------------------------------------------------------------------
    print(f"\nLoading YAML: {args.yaml}")
    with open(args.yaml, "r") as f:
        data = yaml.safe_load(f)

    trees       = data["trees"]
    robot_poses = np.array(data["robot_path"], dtype=np.float64)  # (N, 4): x,y,yaw,t

    print(f"  {len(trees)} committed trees, {len(robot_poses)} robot poses")
    robot_poses_interp = last_pose_per_stamp(robot_poses)
    n_dup = len(robot_poses) - len(robot_poses_interp)
    if n_dup:
        steps = np.hypot(*np.diff(robot_poses[:, :2], axis=0).T)[np.diff(robot_poses[:, 3]) <= 0]
        print(f"  [WARN] robot_path has {n_dup} poses with a repeated time stamp (best particle changed); "
              f"position step at these stamps up to {steps.max():.2f} m. Per-frame poses use the last pose "
              f"of each stamp.")
    print(f"  Robot path time range: [{robot_poses[0, 3]:.3f}, {robot_poses[-1, 3]:.3f}] s")

    broadside_info = {}
    for tree_id, tree_xy in trees.items():
        tree_xy = np.array(tree_xy, dtype=np.float64)
        idx, fwd, lat = find_broadside_pose_index(robot_poses, tree_xy)

        if args.perpendicular_window_m is not None and abs(lat) > args.perpendicular_window_m:
            print(f"  Tree {tree_id}: lat={lat:.2f} m exceeds window, skipping")
            continue

        pose = robot_poses[idx]
        broadside_info[tree_id] = {
            "pose_idx":    idx,
            "pose":        pose.tolist(),
            "fwd_dist_m":  float(fwd),
            "lat_dist_m":  float(lat),
            "timestamp_s": float(pose[3]),
            "tree_xy":     tree_xy.tolist(),
        }
        print(f"  Tree {tree_id:>3}: broadside t={pose[3]:.3f}s  "
              f"fwd={fwd:+.3f}m  lat={lat:+.3f}m  (idx={idx})")

    # ------------------------------------------------------------------
    # 2. Open the bag's sqlite3 database directly
    # ------------------------------------------------------------------
    db_path = _find_sqlite3_db(args.bag)
    if db_path is None:
        raise FileNotFoundError(
            f"No .db3 file found under '{args.bag}'. This script reads "
            f"rosbag2 sqlite3 storage directly; mcap bags are not supported."
        )
    print(f"\nOpening bag database: {db_path}")
    con = sqlite3.connect(db_path)

    all_image_topics = []
    for cam in cameras:
        all_image_topics.extend(cam.topics)
    all_image_topics = list(dict.fromkeys(all_image_topics))  # dedup, order kept

    topics_for_lookup = list(all_image_topics)
    if args.auto_detect_clock_offset:
        topics_for_lookup.append(args.odom_topic)

    topic_id_map, type_map = _topic_id_and_type_maps(con, topics_for_lookup)

    print("\nCamera calibration data:")
    for cam in cameras:
        setup_camera_alignment(con, cam)
        print(f"  [{cam.name}]  depth_aligned.png: {cam.align_mode}"
              + (f" ({cam.align_source})" if cam.align_source else ""))

    # ------------------------------------------------------------------
    # 3. Determine the clock offset
    #
    #    raw_image_header_stamp_at_broadside = broadside_t + image_time_offset_s
    # ------------------------------------------------------------------
    if args.image_time_offset_s is not None:
        offset = args.image_time_offset_s
        print(f"\n[Clock] Using manual offset: {offset:.3f} s")
    elif args.auto_detect_clock_offset:
        if args.odom_topic not in topic_id_map or all_image_topics[0] not in topic_id_map:
            raise RuntimeError("--auto_detect_clock_offset requires both the "
                                "odom topic and at least one image topic to be present in the bag.")
        img_ends  = _first_last_header_stamp(con, topic_id_map[all_image_topics[0]], type_map[all_image_topics[0]])
        odom_ends = _first_last_header_stamp(con, topic_id_map[args.odom_topic],     type_map[args.odom_topic])
        if img_ends is None or odom_ends is None:
            raise RuntimeError("Not enough messages to auto-detect clock offset.")
        t_img0  = img_ends[0][0]
        t_odom0 = odom_ends[0][0]
        offset = t_img0 - t_odom0
        print(f"\n[Clock] Auto-detected offset: {offset:.3f} s "
              f"(first image header={t_img0:.3f}s, first odom header={t_odom0:.3f}s)")
    else:
        offset = 0.0
        print("\n[Clock] No offset specified; assuming image headers and SLAM "
              "timestamps share a clock (offset=0.0).")

    # ------------------------------------------------------------------
    # 4. Calibrate header-stamp -> bag receive-time mapping
    #
    #    The SQL `timestamp` column is the bag receive time. For replayed or
    #    reprocessed bags, it can be in a different epoch than image header.stamp.
    # ------------------------------------------------------------------
    calibration = None
    cal_topic = None
    for t in all_image_topics:
        if t not in topic_id_map:
            continue
        calibration = calibrate_header_to_receive(con, topic_id_map[t], type_map[t])
        if calibration:
            cal_topic = t
            break

    if calibration is None:
        print("[WARN] Could not calibrate header-stamp -> receive-time mapping "
              "(need >=2 messages on at least one image topic). Falling back "
              "to scanning each topic's full timestamp range per query -- "
              "this will be slower.")
        h0_cal, t0_cal_ns, slope_cal = 0.0, 0, 1e9
        full_range = True
    else:
        h0_cal, t0_cal_ns, slope_cal = calibration
        full_range = False
        print(f"[time-map] Calibrated from '{cal_topic}':")
        print(f"  header t0={h0_cal:.3f}s  receive t0={t0_cal_ns*1e-9:.3f}s  "
              f"speed factor={slope_cal/1e9:.4f}x  "
              f"(1.0 = real-time, <1 = slower than real-time)")

    # Odometry for the per-frame poses. Image header stamps map to the bag receive time with the calibration.
    odom_series = load_odom_series(con, args.odom_topic)
    if odom_series is None:
        print(f"[WARN] No odometry on {args.odom_topic}: no odom_pose_xyt / base_delta_from_frame0_xyt in meta.yaml")
    else:
        print(f"[odom] {len(odom_series)} poses from {args.odom_topic}, "
              f"odom->base yaw {args.odom_to_base_yaw_deg:.1f} deg")

    # Image header stamp -> host time, for the odometry lookup (the odometry time is the bag receive time).
    # Header stamps in the host epoch (recorded bag: header = capture time, a few ms to 0.4 s before receive)
    # are used as they are: the receive time would add the transport delay of the image topic (0.37 s on the
    # 2025 mast camera). Only header stamps in a different epoch (replayed bag) go through the calibration.
    header_epoch_ok = calibration is not None and abs(t0_cal_ns * 1e-9 - h0_cal) < 5.0

    def header_to_receive_s(h_s):
        if header_epoch_ok:
            return h_s
        return (t0_cal_ns + (h_s - h0_cal) * slope_cal) * 1e-9

    print(f"[odom time] image header stamps {'used as host time' if header_epoch_ok else 'mapped with the receive-time calibration'}")

    # Broadside pose from the odometry (see odom_alignment.py)
    if args.broadside_source == "odom":
        if odom_series is None:
            raise RuntimeError(f"--broadside_source odom needs odometry on {args.odom_topic} "
                               f"(or use --broadside_source slam_path)")
        print(f"\n[broadside] from odometry (time offset {args.odom_time_offset_s:+.2f} s, "
              f"windows +/-{args.broadside_window_s:.0f} s / rotation +/-{args.broadside_rotation_window_s:.0f} s):")
        for tree_id, info in broadside_info.items():
            b = broadside_from_odom(
                robot_poses_interp, odom_series, np.array(info["tree_xy"]), info["timestamp_s"],
                args.odom_time_offset_s, math.radians(args.odom_to_base_yaw_deg),
                window_s=args.broadside_window_s, rotation_window_s=args.broadside_rotation_window_s,
            )
            info["slam_path_broadside"] = {
                "pose_xyt": [float(v) for v in info["pose"][:3]],
                "timestamp_s": info["timestamp_s"],
                "fwd_dist_m": info["fwd_dist_m"],
                "lat_dist_m": info["lat_dist_m"],
            }
            if b is None:
                print(f"  [WARN] Tree {tree_id}: odometry fit failed; using the robot_path broadside pose")
                info["broadside_source"] = "slam_path (odometry fit failed)"
                continue
            moved = math.hypot(b["pose_xyt"][0] - info["pose"][0], b["pose_xyt"][1] - info["pose"][1])
            print(f"  Tree {tree_id:>3}: t={b['t_slam']:.3f}s ({b['t_slam'] - info['timestamp_s']:+.2f}s)  "
                  f"fwd={b['fwd']:+.3f}m  lat={b['lat']:+.3f}m  moved {moved:.2f} m from the robot_path pose  "
                  f"fit rms {b['fit'].rms_inlier_m:.2f} m ({b['fit'].n_inliers}/{b['fit'].n_used})")
            info["_world_fit"] = b["fit"]
            info.update({
                "pose": b["pose_xyt"],
                "timestamp_s": b["t_slam"],
                "fwd_dist_m": b["fwd"],
                "lat_dist_m": b["lat"],
                "broadside_source": "odom",
                "broadside_fit": {
                    "odom_topic": args.odom_topic,
                    "odom_time_offset_s": args.odom_time_offset_s,
                    "window_s": args.broadside_window_s,
                    "rotation_window_s": args.broadside_rotation_window_s,
                    "rotation_deg": float(math.degrees(b["theta_fit"].theta)),
                    "translation_m": [float(v) for v in b["fit"].t],
                    "rms_inlier_m": b["fit"].rms_inlier_m,
                    "n_inliers": b["fit"].n_inliers,
                    "n_used": b["fit"].n_used,
                    "travel_in_window_m": b["fit"].span_m,
                },
            })

    # ------------------------------------------------------------------
    # 5. SQL query window half-width, in image-header-stamp seconds
    # ------------------------------------------------------------------
    pad_s = args.filter_pad_s
    if pad_s is None:
        pad_s = args.max_time_delta_s + 0.5
    if pad_s < args.max_time_delta_s:
        print(f"[WARN] --filter_pad_s ({pad_s:.2f}s) < --max_time_delta_s "
              f"({args.max_time_delta_s:.2f}s); bumping pad to match.")
        pad_s = args.max_time_delta_s

    buffer_ns = int(2e9)  # extra slack (receive-time ns) for linearisation error

    # ------------------------------------------------------------------
    # 6. Per-tree: query -> find nearest -> convert -> save -> discard
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    saved, skipped = 0, 0
    total_rows_considered = 0
    total_queries = 0

    print(f"\nExtracting {len(broadside_info)} trees "
          f"(query window: +/-{pad_s:.1f}s, max match delta: {args.max_time_delta_s:.1f}s)...")

    for tree_id, info in broadside_info.items():
        target_t   = info["timestamp_s"]
        raw_target = target_t + offset
        tree_dir   = os.path.join(args.output_dir, f"tree_{int(tree_id):03d}")

        if full_range:
            t_min_ns, t_max_ns = 0, 2**63 - 1
        else:
            center_ns = t0_cal_ns + (raw_target - h0_cal) * slope_cal
            pad_ns    = pad_s * abs(slope_cal)
            t_min_ns  = int(center_ns - pad_ns - buffer_ns)
            t_max_ns  = int(center_ns + pad_ns + buffer_ns)

        cam_meta   = {}
        any_cam_ok = False

        ctx = {
            "robot_poses": robot_poses_interp,
            "offset":      offset,
            "raw_target":  raw_target,
            "max_sync_s":  args.max_color_depth_dt_s,
            "odom":        odom_series,
            "header_to_receive_s": header_to_receive_s,
            "world_fit":   info.get("_world_fit"),
            "odom_to_base_yaw": math.radians(args.odom_to_base_yaw_deg),
        }

        for cam in cameras:
            cam_base = {
                "color_topic":   cam.color_topic,
                "depth_topic":   cam.depth_topic,
                "buffer_radius": cam.buffer_radius,
            }
            missing = [t for t in cam.topics if t not in topic_id_map]
            if missing:
                error = f"topic(s) not in bag: {missing}"
                color_msg = depth_msg = None
            else:
                # Color frame nearest to the broadside time, then the depth frame nearest to that color frame
                color_msg, dt_c, t_ns_color, n = find_nearest_in_window(
                    con, topic_id_map[cam.color_topic], type_map[cam.color_topic],
                    t_min_ns, t_max_ns, raw_target, args.max_time_delta_s,
                )
                total_rows_considered += n
                total_queries += 1
                error = None
                if color_msg is None:
                    dt_str = f"{dt_c:.3f}s" if dt_c is not None else "n/a"
                    error = (f"no color frame within {args.max_time_delta_s}s "
                             f"(nearest dt={dt_str}, {n} candidates in window)")
                else:
                    depth_msg, _, _, n = nearest_depth_for_color(
                        con, cam, topic_id_map, type_map, t_ns_color, color_msg,
                        args.max_time_delta_s, slope_cal)
                    total_rows_considered += n
                    total_queries += 1
                    if depth_msg is None:
                        error = f"no depth frame within {args.max_time_delta_s}s of the color frame"

            if error is not None:
                print(f"  [WARN] Tree {tree_id} / {cam.name}: {error}")
                cam_meta[cam.name] = dict(cam_base, saved=False, skip_reason=error)
                continue

            any_cam_ok = True
            cam_dir = os.path.join(tree_dir, cam.name)
            write_camera_info_yaml(cam_dir, cam)
            dt_c = abs(_header_stamp_s(color_msg) - raw_target)
            dt_d = abs(_header_stamp_s(depth_msg) - raw_target)

            if cam.buffer_radius == 0:
                # Single-frame mode (default)
                fm = save_frame(cam_dir, cam, color_msg, depth_msg, ctx)
                del color_msg, depth_msg
                print(f"  [OK]   Tree {tree_id} / {cam.name}: "
                      f"dt_color={dt_c:.3f}s  color-depth={fm['color_depth_dt_s']*1000:.0f}ms")
                cam_meta[cam.name] = dict(cam_base, dt_color_s=float(dt_c), dt_depth_s=float(dt_d), **fm)
            else:
                # Buffer mode: buffer_radius color frames on each side; each one gets the depth frame
                # with the nearest header stamp
                r = cam.buffer_radius
                before, after = fetch_adjacent_frames(
                    con, topic_id_map[cam.color_topic], type_map[cam.color_topic],
                    t_ns_color, r, r,
                )
                before = [None] * (r - len(before)) + before
                after  = after + [None] * (r - len(after))
                colors = before + [(t_ns_color, color_msg)] + after
                del before, after, color_msg

                frame_meta = {}
                n_buf_saved = 0
                for i, item in enumerate(colors):
                    frame_name = f"frame_{i - r}"
                    frame_dir  = os.path.join(cam_dir, frame_name)
                    if item is None:
                        frame_meta[frame_name] = {"saved": False,
                                                  "skip_reason": "insufficient adjacent frames in bag"}
                        print(f"  [WARN] Tree {tree_id} / {cam.name}/{frame_name}: "
                              f"missing frame at this position")
                        continue
                    c_rx, c_msg = item
                    if i == r:
                        d_msg = depth_msg
                    else:
                        d_msg, _, _, n = nearest_depth_for_color(
                            con, cam, topic_id_map, type_map, c_rx, c_msg, args.max_time_delta_s, slope_cal)
                        total_rows_considered += n
                        total_queries += 1
                    if d_msg is None:
                        frame_meta[frame_name] = {"saved": False, "skip_reason": "no depth frame for this color frame"}
                        continue
                    frame_meta[frame_name] = save_frame(frame_dir, cam, c_msg, d_msg, ctx)
                    n_buf_saved += 1
                    del c_msg, d_msg
                del colors, depth_msg

                frame_meta["frame_0"]["dt_color_s"] = float(dt_c)
                frame_meta["frame_0"]["dt_depth_s"] = float(dt_d)
                # Motion of each frame from frame_0, in the robot base frame at frame_0 (from the odometry)
                p0 = frame_meta["frame_0"].get("odom_pose_xyt")
                if p0 is not None:
                    for fm in frame_meta.values():
                        if fm.get("odom_pose_xyt") is not None:
                            fm["base_delta_from_frame0_xyt"] = [
                                float(v) for v in base_delta(p0, fm["odom_pose_xyt"],
                                                             math.radians(args.odom_to_base_yaw_deg))]

                print(f"  [OK]   Tree {tree_id} / {cam.name}: "
                      f"{n_buf_saved}/{2*r+1} buffer frames saved  (center dt_color={dt_c:.3f}s)")
                cam_meta[cam.name] = dict(cam_base, saved=n_buf_saved > 0, frames=frame_meta)

        if not any_cam_ok:
            print(f"  [SKIP] Tree {tree_id}: no camera produced valid frames at t={target_t:.3f}s")
            skipped += 1
            continue

        os.makedirs(tree_dir, exist_ok=True)
        meta = {
            "tree_id":                     int(tree_id),
            "tree_world_xy_m":             info["tree_xy"],
            "broadside_robot_pose_xyt":    info["pose"][:3],
            "broadside_timestamp_s":       info["timestamp_s"],
            "fwd_dist_to_tree_m":          info["fwd_dist_m"],
            "lat_dist_to_tree_m":          info["lat_dist_m"],
            "broadside_source":            info.get("broadside_source", "slam_path"),
            "image_time_offset_applied_s": offset,
            "slam_yaml":                   os.path.abspath(args.yaml),
            "odom_topic":                  args.odom_topic,
            "odom_to_base_yaw_deg":        args.odom_to_base_yaw_deg,
            "max_color_depth_dt_s":        args.max_color_depth_dt_s,
            "cameras":                     cam_meta,
        }
        for key in ("broadside_fit", "slam_path_broadside"):
            if key in info:
                meta[key] = info[key]
        with open(os.path.join(tree_dir, "meta.yaml"), "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        saved += 1

    con.close()
    print(f"\n[stats] {total_queries} queries, "
          f"{total_rows_considered} candidate rows streamed+discarded in total.")
    print(f"\nDone. Saved {saved} trees, skipped {skipped}.")
    print(f"Output: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()