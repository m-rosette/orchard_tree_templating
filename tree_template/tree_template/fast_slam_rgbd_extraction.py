#!/usr/bin/env python3
"""
extract_rgbd_at_tree_broadside.py

Post-processing script: given a SLAM output YAML (with committed tree locations
and robot poses+timestamps) and a ROS2 bag file containing RealSense RGB-D data,
this script finds — for each committed tree — the robot pose where the tree is
perpendicularly in front of the robot (broadside), then extracts and saves the
nearest RGB-D frame pair (color + depth) from the bag at that timestamp.

Output structure:
    output_dir/
        tree_000/
            color.png
            depth.png          (16-bit, millimeters)
            depth_colormap.png (for quick visualization)
            meta.yaml          (tree id, tree world pos, robot pose, timestamp)
        tree_001/
            ...

Clock mismatch handling:
    If the image topics were stamped with Unix wall-clock time (e.g. ~1.697e9 s)
    but the odometry used ROS sim-time (e.g. seconds-since-bag-start ~2300 s),
    use one of:

      --image_time_offset_s <value>
            Manually supply the offset to SUBTRACT from image timestamps.
            Compute as: first_image_unix_stamp - first_odom_sim_stamp
            Example: --image_time_offset_s 1697583500.0

      --auto_detect_clock_offset
            Automatically compute the offset from the first message of each
            topic in the bag (requires --odom_topic to also be in the bag).
            Prints the detected value so you can hard-code it for future runs.

Usage:
    python extract_rgbd_at_tree_broadside.py \
        --yaml slam_trunks_20260331_174636.yaml \
        --bag  /path/to/your.bag \
        --output_dir rgbd_at_trees \
        [--color_topic /base_camera/color/image_raw] \
        [--depth_topic /base_camera/depth/image_rect_raw] \
        [--odom_topic  /odometry/filtered] \
        [--max_time_delta_s 0.5] \
        [--perpendicular_window_m 0.5] \
        [--image_time_offset_s 1697583500.0 | --auto_detect_clock_offset]
"""

import argparse
import os
import math
import numpy as np
import yaml
import cv2

# ROS2 bag reading
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


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
    Find the index in robot_poses where |fwd_dist| is minimised —
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
# Bag reading helpers
# ---------------------------------------------------------------------------

def build_topic_message_cache(bag_path, topics):
    """
    Read the bag and return:
        { topic: [(timestamp_s, msg), ...] }

    Timestamps come from header.stamp when available (more accurate than the
    bag receive timestamp), otherwise fall back to bag receive time.
    """
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    cache = {t: [] for t in topics}
    type_map = {}
    for topic in topics:
        if topic not in topic_types:
            print(f"[WARN] Topic '{topic}' not found in bag. "
                  f"Available: {sorted(topic_types.keys())}")
            continue
        type_map[topic] = get_message(topic_types[topic])

    print("Reading bag (this may take a moment)...")
    count = 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in type_map:
            msg = deserialize_message(data, type_map[topic])
            # Prefer header.stamp — reflects when data was captured, not
            # when the bag recorder received it.
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                s = msg.header.stamp
                t_s = float(s.sec) + float(s.nanosec) * 1e-9
            else:
                t_s = t_ns * 1e-9
            cache[topic].append((t_s, msg))
        count += 1
        if count % 5000 == 0:
            print(f"  ... read {count} messages")

    for topic, msgs in cache.items():
        if msgs:
            print(f"  '{topic}': {len(msgs)} msgs  "
                  f"t=[{msgs[0][0]:.3f}, {msgs[-1][0]:.3f}] s")
        else:
            print(f"  '{topic}': 0 messages")

    return cache


def detect_clock_offset(cache, image_topic, odom_topic):
    """
    Compute the offset to subtract from image timestamps to align them with
    odom timestamps:

        offset = first_image_stamp - first_odom_stamp

    Handles the common recording issue where the RealSense driver stamped with
    Unix wall-clock time (~1.697e9 s) while odom used ROS sim-time (~2300 s).

    Returns offset in seconds.  Apply as: corrected_t = raw_image_t - offset
    """
    img_msgs  = cache.get(image_topic, [])
    odom_msgs = cache.get(odom_topic,  [])

    if not img_msgs:
        raise RuntimeError(
            f"No messages cached for image topic '{image_topic}'. "
            "Cannot auto-detect clock offset."
        )
    if not odom_msgs:
        raise RuntimeError(
            f"No messages cached for odom topic '{odom_topic}'. "
            "Cannot auto-detect clock offset. "
            "Ensure --odom_topic is correct, or use --image_time_offset_s instead."
        )

    t_img  = img_msgs[0][0]
    t_odom = odom_msgs[0][0]
    offset = t_img - t_odom

    print(f"\n[Clock offset detection]")
    print(f"  First image stamp : {t_img:.6f} s")
    print(f"  First odom stamp  : {t_odom:.6f} s")
    print(f"  Detected offset   : {offset:.6f} s")
    print(f"  → subtracting {offset:.3f} s from all image timestamps")
    print(f"  Tip: hard-code with --image_time_offset_s {offset:.3f}\n")

    return offset


def apply_time_offset(cache, topics, offset):
    """Subtract offset (seconds) from all timestamps for the given topics."""
    for topic in topics:
        if topic in cache and cache[topic]:
            cache[topic] = [(t - offset, msg) for t, msg in cache[topic]]
    print(f"[Clock correction] Subtracted {offset:.3f} s from image timestamps.")


def find_nearest_message(msg_list, target_t, max_delta_s=0.5):
    """
    Binary-search the sorted msg_list for the message closest to target_t.
    Returns (msg, dt) or (None, None) if nothing is within max_delta_s.
    """
    if not msg_list:
        return None, None

    times = np.array([m[0] for m in msg_list])
    idx = int(np.searchsorted(times, target_t))

    candidates = []
    for i in [idx - 1, idx]:
        if 0 <= i < len(msg_list):
            dt = abs(msg_list[i][0] - target_t)
            candidates.append((dt, i))

    if not candidates:
        return None, None

    dt, best_i = min(candidates)
    if dt > max_delta_s:
        return None, None

    return msg_list[best_i][1], dt


# ---------------------------------------------------------------------------
# Image conversion helpers
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yaml",        required=True,  help="SLAM output YAML file")
    parser.add_argument("--bag",         required=True,  help="ROS2 bag file path")
    parser.add_argument("--output_dir",  default="/home/marcus/apple_harvest_ws/data/rgbd_at_trees_oct_2025_v3")
    parser.add_argument("--color_topic", default="/camera/mast_camera/color/image_raw")
    parser.add_argument("--depth_topic", default="/camera/mast_camera/recovered_depth/image_raw")
    parser.add_argument("--odom_topic",  default="/filter/state",
                        help="Only needed with --auto_detect_clock_offset")
    parser.add_argument("--max_time_delta_s", type=float, default=1.0,
                        help="Max allowed time gap (s) between broadside pose and image frame")
    parser.add_argument("--perpendicular_window_m", type=float, default=None,
                        help="Skip trees whose lateral distance exceeds this (metres)")

    clock_grp = parser.add_mutually_exclusive_group()
    clock_grp.add_argument(
        "--image_time_offset_s", type=float, default=None,
        help="Subtract this value (s) from all image timestamps. "
             "Compute as: first_image_stamp - first_odom_stamp",
    )
    clock_grp.add_argument(
        "--auto_detect_clock_offset", action="store_true",
        help="Auto-compute the clock offset from the first messages on the "
             "image and odom topics. Prints the value for future hard-coding.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load YAML
    # ------------------------------------------------------------------
    print(f"Loading YAML: {args.yaml}")
    with open(args.yaml, "r") as f:
        data = yaml.safe_load(f)

    trees       = data["trees"]       # dict: {id: [x, y]}
    robot_path  = data["robot_path"]  # list of [x, y, yaw, timestamp]
    robot_poses = np.array(robot_path, dtype=np.float64)  # (N, 4)

    print(f"  {len(trees)} committed trees, {len(robot_poses)} robot poses")
    print(f"  Robot path time range: [{robot_poses[0, 3]:.3f}, {robot_poses[-1, 3]:.3f}] s")

    # ------------------------------------------------------------------
    # 2. Find broadside pose per tree
    # ------------------------------------------------------------------
    broadside_info = {}

    for tree_id, tree_xy in trees.items():
        tree_xy = np.array(tree_xy, dtype=np.float64)
        idx, fwd, lat = find_broadside_pose_index(robot_poses, tree_xy)

        if args.perpendicular_window_m is not None and abs(lat) > args.perpendicular_window_m:
            print(f"  Tree {tree_id}: lat={lat:.2f} m exceeds window, skipping")
            continue

        pose = robot_poses[idx]
        ts   = float(pose[3])
        broadside_info[tree_id] = {
            "pose_idx":    idx,
            "pose":        pose.tolist(),
            "fwd_dist_m":  float(fwd),
            "lat_dist_m":  float(lat),
            "timestamp_s": ts,
            "tree_xy":     tree_xy.tolist(),
        }
        print(f"  Tree {tree_id:>3}: broadside t={ts:.3f}s  "
              f"fwd={fwd:+.3f}m  lat={lat:+.3f}m  (idx={idx})")

    # ------------------------------------------------------------------
    # 3. Read bag
    # ------------------------------------------------------------------
    topics_to_read = [args.color_topic, args.depth_topic]
    if args.auto_detect_clock_offset:
        topics_to_read.append(args.odom_topic)

    cache = build_topic_message_cache(args.bag, topics_to_read)

    # ------------------------------------------------------------------
    # 4. Apply clock offset
    # ------------------------------------------------------------------
    if args.auto_detect_clock_offset:
        offset = detect_clock_offset(cache, args.color_topic, args.odom_topic)
        apply_time_offset(cache, [args.color_topic, args.depth_topic], offset)

    elif args.image_time_offset_s is not None:
        print(f"\n[Clock correction] Applying manual offset: -{args.image_time_offset_s:.3f} s")
        apply_time_offset(cache, [args.color_topic, args.depth_topic], args.image_time_offset_s)
        for topic in [args.color_topic, args.depth_topic]:
            msgs = cache[topic]
            if msgs:
                print(f"  '{topic}' corrected range: [{msgs[0][0]:.3f}, {msgs[-1][0]:.3f}] s")

    else:
        print("\n[Clock] No offset applied. If no frames are matched, try "
              "--auto_detect_clock_offset or --image_time_offset_s.")

    # ------------------------------------------------------------------
    # 5. Extract and save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0
    skipped = 0

    for tree_id, info in broadside_info.items():
        target_t = info["timestamp_s"]

        color_msg, dt_c = find_nearest_message(
            cache[args.color_topic], target_t, args.max_time_delta_s)
        depth_msg, dt_d = find_nearest_message(
            cache[args.depth_topic], target_t, args.max_time_delta_s)

        if color_msg is None or depth_msg is None:
            print(f"  [SKIP] Tree {tree_id}: no frame within {args.max_time_delta_s}s "
                  f"of t={target_t:.3f}s  (dt_color={dt_c}, dt_depth={dt_d})")
            skipped += 1
            continue

        try:
            color_img = ros_image_to_cv2(color_msg)
            depth_img = ros_image_to_cv2(depth_msg)
        except Exception as e:
            print(f"  [SKIP] Tree {tree_id}: image conversion error: {e}")
            skipped += 1
            continue

        tree_dir = os.path.join(args.output_dir, f"tree_{int(tree_id):03d}")
        os.makedirs(tree_dir, exist_ok=True)

        cv2.imwrite(os.path.join(tree_dir, "color.png"), color_img)
        cv2.imwrite(os.path.join(tree_dir, "depth.png"), depth_img)  # 16-bit PNG

        depth_norm  = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        cv2.imwrite(os.path.join(tree_dir, "depth_colormap.png"), depth_color)

        applied_offset = (
            args.image_time_offset_s if args.image_time_offset_s is not None
            else ("auto-detected" if args.auto_detect_clock_offset else 0.0)
        )
        meta = {
            "tree_id":                     int(tree_id),
            "tree_world_xy_m":             info["tree_xy"],
            "broadside_robot_pose_xyt":    info["pose"][:3],
            "broadside_timestamp_s":       info["timestamp_s"],
            "fwd_dist_to_tree_m":          info["fwd_dist_m"],
            "lat_dist_to_tree_m":          info["lat_dist_m"],
            "dt_color_s":                  float(dt_c),
            "dt_depth_s":                  float(dt_d),
            "color_topic":                 args.color_topic,
            "depth_topic":                 args.depth_topic,
            "image_time_offset_applied_s": applied_offset,
        }
        with open(os.path.join(tree_dir, "meta.yaml"), "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        print(f"  [OK]   Tree {tree_id}: saved  dt_color={dt_c:.3f}s  dt_depth={dt_d:.3f}s")
        saved += 1

    print(f"\nDone. Saved {saved} trees, skipped {skipped}.")
    print(f"Output: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()