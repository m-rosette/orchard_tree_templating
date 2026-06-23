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
                color.png
                depth.png                (16-bit, millimeters)
                depth_colormap.png
            cam2/                        (single frame, buffer_radius=0)
                color.png
                depth.png
                depth_colormap.png
            cam2/                        (5-frame buffer, --cam2_buffer_radius 2)
                frame_-2/color.png, depth.png, depth_colormap.png
                frame_-1/color.png, depth.png, depth_colormap.png
                frame_0/ color.png, depth.png, depth_colormap.png
                frame_1/ color.png, depth.png, depth_colormap.png
                frame_2/ color.png, depth.png, depth_colormap.png
        tree_001/
            ...

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
        --image_time_offset_s 0.0 --max_time_delta_s 5.0
"""

import argparse
import os
import math
import sqlite3
import numpy as np
import yaml
import cv2

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ---------------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------------

class CameraConfig:
    """Holds topic names and output directory name for one camera."""

    def __init__(self, name: str, color_topic: str, depth_topic: str,
                 buffer_radius: int = 0):
        self.name          = name
        self.color_topic   = color_topic
        self.depth_topic   = depth_topic
        self.buffer_radius = buffer_radius  # 0 = single frame; N = save 2N+1 frames

    @property
    def topics(self):
        return [self.color_topic, self.depth_topic]

    def __repr__(self):
        return (f"CameraConfig(name={self.name!r}, "
                f"color={self.color_topic!r}, depth={self.depth_topic!r})")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

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
        before : list of deserialized msgs, oldest-first (len <= n_before)
        after  : list of deserialized msgs, chronologically (len <= n_after)
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
    before = [deserialize_message(bytes(data), msg_type) for _, data in rows_before]

    cur.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id=? AND timestamp > ? "
        "ORDER BY timestamp ASC LIMIT ?",
        (topic_id, center_receive_ns, n_after),
    )
    after = [deserialize_message(bytes(data), msg_type) for _, data in cur.fetchall()]

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


def save_camera_images(cam_dir, color_img, depth_img):
    """Write color.png, depth.png (16-bit), and depth_colormap.png into cam_dir."""
    os.makedirs(cam_dir, exist_ok=True)
    cv2.imwrite(os.path.join(cam_dir, "color.png"), color_img)
    cv2.imwrite(os.path.join(cam_dir, "depth.png"), depth_img)  # 16-bit PNG

    depth_norm  = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(cam_dir, "depth_colormap.png"), depth_color)


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

    # -- Timing / filtering ------------------------------------------------------
    parser.add_argument(
        "--odom_topic", default="/filter/state",
        help="Odometry topic (only needed with --auto_detect_clock_offset)")
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
        CameraConfig(args.cam1_name, args.cam1_color_topic, args.cam1_depth_topic),
        CameraConfig(args.cam2_name, args.cam2_color_topic, args.cam2_depth_topic,
                     buffer_radius=args.cam2_buffer_radius),
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
    #    Needed because the SQL `timestamp` column is bag RECEIVE time, which
    #    may be in a completely different epoch than image header.stamp for
    #    replayed/reprocessed bags.
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

        for cam in cameras:
            results = {}  # role -> (msg_or_None, dt_or_None, t_ns_or_None, error_or_None)

            for role, topic in (("color", cam.color_topic), ("depth", cam.depth_topic)):
                if topic not in topic_id_map:
                    results[role] = (None, None, None, f"topic '{topic}' not in bag")
                    continue

                msg, dt, t_ns, n = find_nearest_in_window(
                    con, topic_id_map[topic], type_map[topic],
                    t_min_ns, t_max_ns, raw_target, args.max_time_delta_s,
                )
                total_rows_considered += n
                total_queries += 1

                if msg is None:
                    dt_str = f"{dt:.3f}s" if dt is not None else "n/a"
                    results[role] = (None, dt, None,
                        f"no frame within {args.max_time_delta_s}s "
                        f"(nearest dt={dt_str}, {n} candidates in window)")
                else:
                    results[role] = (msg, dt, t_ns, None)

            color_msg, dt_c, t_ns_color, err_c = results["color"]
            depth_msg, dt_d, t_ns_depth, err_d = results["depth"]
            error = err_c or err_d

            if error is not None:
                print(f"  [WARN] Tree {tree_id} / {cam.name}: {error}")
                cam_meta[cam.name] = {
                    "color_topic":   cam.color_topic,
                    "depth_topic":   cam.depth_topic,
                    "buffer_radius": cam.buffer_radius,
                    "saved":         False,
                    "skip_reason":   error,
                }
                continue

            any_cam_ok = True
            cam_dir = os.path.join(tree_dir, cam.name)

            if cam.buffer_radius == 0:
                # Single-frame mode (default, existing behavior)
                color_img = ros_image_to_cv2(color_msg)
                depth_img = ros_image_to_cv2(depth_msg)
                del color_msg, depth_msg
                save_camera_images(cam_dir, color_img, depth_img)
                del color_img, depth_img
                print(f"  [OK]   Tree {tree_id} / {cam.name}: "
                      f"dt_color={dt_c:.3f}s  dt_depth={dt_d:.3f}s")
                cam_meta[cam.name] = {
                    "color_topic": cam.color_topic,
                    "depth_topic": cam.depth_topic,
                    "buffer_radius": 0,
                    "dt_color_s":  float(dt_c),
                    "dt_depth_s":  float(dt_d),
                    "saved":       True,
                }
            else:
                # Multi-frame buffer mode: fetch radius frames on each side
                r = cam.buffer_radius
                before_colors, after_colors = fetch_adjacent_frames(
                    con, topic_id_map[cam.color_topic], type_map[cam.color_topic],
                    t_ns_color, r, r,
                )
                before_depths, after_depths = fetch_adjacent_frames(
                    con, topic_id_map[cam.depth_topic], type_map[cam.depth_topic],
                    t_ns_depth, r, r,
                )

                # Pad with None where the bag doesn't have enough adjacent frames
                before_colors = [None] * (r - len(before_colors)) + before_colors
                after_colors  = after_colors  + [None] * (r - len(after_colors))
                before_depths = [None] * (r - len(before_depths)) + before_depths
                after_depths  = after_depths  + [None] * (r - len(after_depths))

                all_colors = before_colors + [color_msg] + after_colors
                all_depths = before_depths + [depth_msg] + after_depths
                del color_msg, depth_msg, before_colors, after_colors
                del before_depths, after_depths

                frame_meta = {}
                n_buf_saved = 0
                for i, (c_msg, d_msg) in enumerate(zip(all_colors, all_depths)):
                    offset     = i - r
                    frame_name = f"frame_{offset}"
                    frame_dir  = os.path.join(cam_dir, frame_name)
                    if c_msg is not None and d_msg is not None:
                        color_img = ros_image_to_cv2(c_msg)
                        depth_img = ros_image_to_cv2(d_msg)
                        del c_msg, d_msg
                        save_camera_images(frame_dir, color_img, depth_img)
                        del color_img, depth_img
                        frame_meta[frame_name] = {"saved": True}
                        n_buf_saved += 1
                    else:
                        frame_meta[frame_name] = {
                            "saved": False,
                            "skip_reason": "insufficient adjacent frames in bag",
                        }
                        print(f"  [WARN] Tree {tree_id} / {cam.name}/{frame_name}: "
                              f"missing frame at this position")
                del all_colors, all_depths

                frame_meta["frame_0"]["dt_color_s"] = float(dt_c)
                frame_meta["frame_0"]["dt_depth_s"] = float(dt_d)

                print(f"  [OK]   Tree {tree_id} / {cam.name}: "
                      f"{n_buf_saved}/{2*r+1} buffer frames saved  "
                      f"(center dt_color={dt_c:.3f}s  dt_depth={dt_d:.3f}s)")
                cam_meta[cam.name] = {
                    "color_topic":   cam.color_topic,
                    "depth_topic":   cam.depth_topic,
                    "buffer_radius": r,
                    "saved":         n_buf_saved > 0,
                    "frames":        frame_meta,
                }

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
            "image_time_offset_applied_s": offset,
            "cameras":                     cam_meta,
        }
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