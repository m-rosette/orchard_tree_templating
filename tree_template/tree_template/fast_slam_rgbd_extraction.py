#!/usr/bin/env python3
"""
fast_slam_rgbd_extraction.py

For each tree of a SLAM map (save_best_particle_map yaml), finds the time when the tree is straight to the side of
the robot (broadside) and saves the RGB-D frames of two cameras at that time from a rosbag2 bag.

Output:
    output_dir/tree_000/
        meta.yaml
        <cam1_name>/                 single frame
            camera_info.yaml         color + depth intrinsics, depth->color extrinsics
            color.png, depth.png (uint16 mm), depth_aligned.png (registered to color.png), depth_colormap.png
        <cam2_name>/                 --cam2_buffer_radius N: frame_-N/ ... frame_N/ (consecutive color frames)
            camera_info.yaml
            frame_0/color.png, ...

meta.yaml, for each frame: color/depth header stamps, color_depth_dt_s and sync_ok (--max_color_depth_dt_s),
t_from_broadside_s, odom_pose_xyt (--odom_topic), world_pose_xyt (odometry in the SLAM world frame,
--broadside_source odom), slam_path_pose_xyt (SLAM robot_path; it jumps when the best particle changes).
Buffer frames also have base_delta_from_frame0_xyt (motion from frame_0 in the base frame, --odom_to_base_yaw_deg).
The depth frame of a pair is the one with the header stamp nearest to the color frame.

Broadside (--broadside_source odom): odometry put into the world frame by odom_alignment.broadside_from_odom.
meta.yaml also keeps the robot_path result (slam_path_broadside) and the fit (broadside_fit).

depth_aligned.png: depth_registration.register_depth_to_color with --camN_depth_to_color (calibrate_depth_to_color,
config/camera_extrinsics/) or the bag extrinsics topic. An aligned depth topic (aligned_depth_to_color,
--camN_depth_aligned) is copied.

Clocks: the image header stamp expected at SLAM time T is T + --image_time_offset_s. Image header stamps map to the
bag receive time with a linear fit from the first and last message of an image topic. The odometry time is its
receive time.

Usage:
    ros2 run tree_template fast_slam_rgbd_extraction \\
        --yaml slam_trunks_20260626_131635.yaml --bag /path/to/bag_dir --output_dir rgbd_at_trees \\
        --cam1_depth_topic /camera/mast_camera/recovered_depth/image_raw \\
        --cam1_depth_to_color config/camera_extrinsics/prosser_2025_mast_camera.yaml \\
        --cam2_depth_to_color config/camera_extrinsics/prosser_2025_base_camera.yaml \\
        --cam2_buffer_radius 2 --image_time_offset_s 2 --odom_to_base_yaw_deg 180
"""

import argparse
import collections
import math
import os

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

from tree_template.bag_reader import BagReader
from tree_template.depth_registration import (
    camera_info_to_dict,
    camera_matrix,
    distortion,
    extrinsics_from_realsense_msg,
    load_extrinsics_yaml,
    register_depth_to_color,
)
from tree_template.odom_alignment import broadside_from_odom, interp_pose, world_poses
from tree_template.trellis_geometry import pose_tf

BRIDGE = CvBridge()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class CameraConfig:
    """Topics, output folder name and depth registration of one camera."""

    def __init__(self, name, color_topic, depth_topic, buffer_radius=0, color_info_topic=None,
                 depth_info_topic=None, depth_to_color_yaml=None, depth_is_aligned=False, cli_prefix="cam"):
        self.name = name
        self.cli_prefix = cli_prefix
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.buffer_radius = buffer_radius          # 0: one frame; N: 2N+1 frames
        # Default camera_info topic: next to the image topic
        self.color_info_topic = color_info_topic or color_topic.rsplit("/", 1)[0] + "/camera_info"
        self.depth_info_topic = depth_info_topic or depth_topic.rsplit("/", 1)[0] + "/camera_info"
        self.depth_to_color_yaml = depth_to_color_yaml
        self.depth_is_aligned = depth_is_aligned or "aligned_depth_to_color" in depth_topic
        # Set by setup_camera_alignment()
        self.color_info = self.depth_info = None
        self.R = self.t_mm = None
        self.align_mode = "none"                    # register | already_aligned | none
        self.align_source = None


def setup_camera_alignment(bag: BagReader, cam: CameraConfig):
    """Read the camera_info of cam and set how depth_aligned.png is made (cam.align_mode)."""
    cam.color_info = camera_info_to_dict(bag.first(cam.color_info_topic)[0])
    cam.depth_info = camera_info_to_dict(bag.first(cam.depth_info_topic)[0])
    if cam.color_info is None:
        print(f"  [WARN] {cam.name}: no camera_info on {cam.color_info_topic}")
    if cam.depth_is_aligned:
        cam.align_mode, cam.align_source = "already_aligned", cam.depth_topic
        return

    R = t_m = None
    if cam.depth_to_color_yaml:
        R, t_m, _ = load_extrinsics_yaml(cam.depth_to_color_yaml)
        cam.align_source = os.path.abspath(cam.depth_to_color_yaml)
    else:
        ext_topic = cam.depth_topic.rsplit("/", 2)[0] + "/extrinsics/depth_to_color"
        msg = bag.first(ext_topic)[0]
        if msg is not None:
            R, t_m = extrinsics_from_realsense_msg(msg)
            cam.align_source = ext_topic

    reasons = []
    if R is None:
        reasons.append(f"no depth->color extrinsics (give --{cam.cli_prefix}_depth_to_color)")
    if cam.color_info is None or cam.depth_info is None:
        reasons.append("camera_info missing")
    if reasons:
        print(f"  [WARN] {cam.name}: no depth_aligned.png: {'; '.join(reasons)}")
        return
    cam.align_mode, cam.R, cam.t_mm = "register", R, np.asarray(t_m) * 1000.0


def write_camera_info_yaml(cam_dir: str, cam: CameraConfig):
    data = {"color": cam.color_info, "depth": cam.depth_info,
            "depth_aligned": {"mode": cam.align_mode, "source": cam.align_source}}
    if cam.align_mode == "register":
        data["depth_aligned"]["rotation"] = [float(x) for x in np.asarray(cam.R).reshape(-1)]
        data["depth_aligned"]["translation_m"] = [float(x) for x in cam.t_mm / 1000.0]
    os.makedirs(cam_dir, exist_ok=True)
    with open(os.path.join(cam_dir, "camera_info.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def depth_to_mm(msg) -> np.ndarray:
    """Depth image as uint16 mm (32FC1 metres are converted)."""
    d = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return (d * 1000.0).astype(np.uint16) if d.dtype == np.float32 else d


def save_camera_images(cam_dir, color_img, depth_img, depth_aligned_img=None):
    os.makedirs(cam_dir, exist_ok=True)
    cv2.imwrite(os.path.join(cam_dir, "color.png"), color_img)
    cv2.imwrite(os.path.join(cam_dir, "depth.png"), depth_img)
    if depth_aligned_img is not None:
        cv2.imwrite(os.path.join(cam_dir, "depth_aligned.png"), depth_aligned_img)
    depth_8u = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(cam_dir, "depth_colormap.png"), cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO))


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------

def last_pose_per_stamp(poses: np.ndarray) -> np.ndarray:
    """Rows [x, y, yaw, t] with only the last row of each stamp (the pose after a best-particle change)."""
    return poses[np.append(np.diff(poses[:, 3]) > 0, True)]


def tree_in_robot_frames(poses: np.ndarray, tree_xy: np.ndarray) -> np.ndarray:
    """(N, 2) [fwd, lat] of the tree in the robot frame of each pose [x, y, yaw, ...]."""
    d = np.column_stack([tree_xy[0] - poses[:, 0], tree_xy[1] - poses[:, 1], np.zeros(len(poses))])
    return Rotation.from_euler("z", -poses[:, 2]).apply(d)[:, :2]


def load_odom(bag: BagReader, topic: str):
    """(N, 4) [x, y, yaw, receive time s] of nav_msgs/Odometry messages on topic, or None."""
    rows = [(m.pose.pose, t) for _, m, t in bag.messages([topic])]
    if not rows:
        return None
    q = np.array([[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p, _ in rows])
    yaw = Rotation.from_quat(q).as_euler("ZYX")[:, 0]
    return np.array([[p.position.x, p.position.y, y, t * 1e-9] for (p, t), y in zip(rows, yaw)])


def base_delta(pose0, pose1, odom_to_base_yaw):
    """Motion [dx, dy, dyaw] from odometry pose0 to pose1, in the robot base frame at pose0."""
    c_b = pose_tf(0.0, 0.0, odom_to_base_yaw)                   # base frame -> odometry child frame
    rel = c_b + pose_tf(*pose1) + pose_tf(*pose0).inverse + c_b.inverse   # base1 -> base0
    return [float(rel.translation[0]), float(rel.translation[1]), float(rel.rotation)]


# ---------------------------------------------------------------------------
# Frames from the bag
# ---------------------------------------------------------------------------

def nearest_by_stamp(bag, topic, start_ns, end_ns, target_s, max_delta_s):
    """(message, receive ns) in [start_ns, end_ns] with the header stamp nearest to target_s, or (None, None)."""
    best, best_dt = (None, None), None
    for _, data, t in bag.raw([topic], start_ns, end_ns):
        msg = bag.deserialize(topic, data)
        dt = abs(stamp_s(msg) - target_s)
        if best_dt is None or dt < best_dt:
            best, best_dt = (msg, t), dt
    return best if best_dt is not None and best_dt <= max_delta_s else (None, None)


def neighbours(bag, topic, center_ns, n):
    """(before, after): up to n (receive ns, message) just before and just after center_ns, in receive order."""
    before = collections.deque(maxlen=n)
    for window in (int(2e9), int(10e9), None):
        before.clear()
        for _, data, t in bag.raw([topic], None if window is None else center_ns - window, center_ns - 1):
            before.append((t, data))
        if len(before) == n:
            break
    after = []
    for _, data, t in bag.raw([topic], center_ns + 1):
        after.append((t, data))
        if len(after) == n:
            break
    return ([(t, bag.deserialize(topic, d)) for t, d in before],
            [(t, bag.deserialize(topic, d)) for t, d in after])


def save_frame(frame_dir, cam, color_msg, depth_msg, ctx):
    """Write the images of one color/depth pair and return its meta.yaml entry."""
    t_c, t_d = stamp_s(color_msg), stamp_s(depth_msg)
    color_img = BRIDGE.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
    depth_img = depth_to_mm(depth_msg)

    aligned = None
    if cam.align_mode == "register":
        aligned = register_depth_to_color(
            depth_img, camera_matrix(cam.depth_info), camera_matrix(cam.color_info), cam.R, cam.t_mm,
            color_img.shape[:2], distortion(cam.depth_info), distortion(cam.color_info))
    elif cam.align_mode == "already_aligned":
        if depth_img.shape[:2] == color_img.shape[:2]:
            aligned = depth_img
        else:
            print(f"  [WARN] {cam.name}: depth {depth_img.shape[:2]} is not the color size {color_img.shape[:2]}")
    save_camera_images(frame_dir, color_img, depth_img, aligned)

    path = ctx["robot_poses"]
    t_slam = t_c - ctx["offset"]
    dt_cd = abs(t_d - t_c)
    meta = {
        "saved": True,
        "color_stamp_s": t_c,
        "depth_stamp_s": t_d,
        "color_depth_dt_s": float(dt_cd),
        "sync_ok": bool(dt_cd <= ctx["max_sync_s"]),
        "t_from_broadside_s": float(t_c - ctx["raw_target"]),
        "slam_path_pose_xyt": [float(v) for v in interp_pose(path, t_slam)[0]],
        "depth_aligned": aligned is not None,
    }
    if not path[0, 3] <= t_slam <= path[-1, 3]:
        meta["slam_path_pose_note"] = "color stamp outside the robot path; nearest end pose"
    odom = ctx["odom"]
    if odom is not None:
        t_rx = ctx["header_to_receive_s"](t_c)
        if ctx["world_fit"] is not None:
            wp = world_poses(ctx["world_fit"], odom, np.array([t_rx]), ctx["odom_to_base_yaw"])[0]
            meta["world_pose_xyt"] = [float(v) for v in wp]
        meta["odom_pose_xyt"] = [float(v) for v in interp_pose(odom, t_rx)[0]]
        if not odom[0, 3] <= t_rx <= odom[-1, 3]:
            meta["odom_pose_note"] = "color stamp outside the odometry; nearest end pose"
    if not meta["sync_ok"]:
        print(f"  [WARN] {cam.name} {os.path.basename(frame_dir)}: color/depth stamps {dt_cd * 1000:.0f} ms apart")
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", required=True, help="SLAM map yaml (save_best_particle_map)")
    p.add_argument("--bag", required=True, help="rosbag2 bag (sqlite3 or mcap)")
    p.add_argument("--output_dir", required=True)
    for n, name, color, depth in (
            ("cam1", "mast_cam", "/camera/mast_camera/color/image_raw", "/camera/mast_camera/recovered_depth/image_raw"),
            ("cam2", "base_cam", "/camera/base_camera/color/image_raw", "/camera/base_camera/depth/image_rect_raw")):
        g = p.add_argument_group(n)
        g.add_argument(f"--{n}_name", default=name, help="output folder name")
        g.add_argument(f"--{n}_color_topic", default=color)
        g.add_argument(f"--{n}_depth_topic", default=depth)
        g.add_argument(f"--{n}_color_info_topic", default=None, help="default: camera_info next to the color topic")
        g.add_argument(f"--{n}_depth_info_topic", default=None, help="default: camera_info next to the depth topic")
        g.add_argument(f"--{n}_depth_to_color", default=None,
                       help="depth->color extrinsics yaml (calibrate_depth_to_color). Default: bag extrinsics topic")
        g.add_argument(f"--{n}_depth_aligned", action="store_true", help="the depth topic is registered to color")
    p.add_argument("--cam2_buffer_radius", type=int, default=0, help="frames on each side of the broadside frame")

    p.add_argument("--odom_topic", default="/filter/state", help="nav_msgs/Odometry")
    p.add_argument("--odom_to_base_yaw_deg", type=float, default=0.0,
                   help="yaw of the base frame in the odometry child frame (180: 2025 Amiga /filter/state)")
    p.add_argument("--broadside_source", choices=["odom", "slam_path"], default="odom",
                   help="odom: odometry fitted to the robot_path; slam_path: robot_path pose with the smallest |fwd|")
    p.add_argument("--odom_time_offset_s", type=float, default=0.0, help="odometry time - SLAM time")
    p.add_argument("--broadside_window_s", type=float, default=10.0, help="half window, translation fit")
    p.add_argument("--broadside_rotation_window_s", type=float, default=60.0, help="half window, rotation fit")
    p.add_argument("--max_time_delta_s", type=float, default=1.0, help="max |color stamp - broadside time|")
    p.add_argument("--filter_pad_s", type=float, default=None,
                   help="half window (s) of the color frame search (default max_time_delta_s + 0.5)")
    p.add_argument("--max_color_depth_dt_s", type=float, default=0.05, help="sync_ok limit")
    p.add_argument("--perpendicular_window_m", type=float, default=None, help="skip trees with a larger |lat|")
    clock = p.add_mutually_exclusive_group()
    clock.add_argument("--image_time_offset_s", type=float, default=None, help="image header time - SLAM time")
    clock.add_argument("--auto_detect_clock_offset", action="store_true",
                       help="offset = first image header stamp - first odometry header stamp")
    return p.parse_args()


def main():
    args = parse_args()
    cameras = [
        CameraConfig(getattr(args, f"{n}_name"), getattr(args, f"{n}_color_topic"), getattr(args, f"{n}_depth_topic"),
                     buffer_radius=args.cam2_buffer_radius if n == "cam2" else 0,
                     color_info_topic=getattr(args, f"{n}_color_info_topic"),
                     depth_info_topic=getattr(args, f"{n}_depth_info_topic"),
                     depth_to_color_yaml=getattr(args, f"{n}_depth_to_color"),
                     depth_is_aligned=getattr(args, f"{n}_depth_aligned"), cli_prefix=n)
        for n in ("cam1", "cam2")]
    odom_to_base_yaw = math.radians(args.odom_to_base_yaw_deg)

    # --- SLAM map and the robot_path broadside of each tree -----------------
    with open(args.yaml, "r") as f:
        data = yaml.safe_load(f)
    robot_poses = np.array(data["robot_path"], dtype=np.float64)        # [x, y, yaw, t]
    path = last_pose_per_stamp(robot_poses)
    print(f"{len(data['trees'])} trees, {len(robot_poses)} robot path poses "
          f"({len(robot_poses) - len(path)} repeated stamps)")

    broadside_info = {}
    for tree_id, tree_xy in data["trees"].items():
        tree_xy = np.array(tree_xy, dtype=np.float64)
        fl = tree_in_robot_frames(robot_poses, tree_xy)
        idx = int(np.argmin(np.abs(fl[:, 0])))
        fwd, lat = fl[idx]
        if args.perpendicular_window_m is not None and abs(lat) > args.perpendicular_window_m:
            print(f"  Tree {tree_id}: |lat| {abs(lat):.2f} m > --perpendicular_window_m, skipped")
            continue
        pose = robot_poses[idx]
        broadside_info[tree_id] = {"pose": pose[:3].tolist(), "fwd_dist_m": float(fwd), "lat_dist_m": float(lat),
                                   "timestamp_s": float(pose[3]), "tree_xy": tree_xy.tolist()}

    # --- Bag ------------------------------------------------------------------
    bag = BagReader(args.bag)
    image_topics = list(dict.fromkeys(t for c in cameras for t in (c.color_topic, c.depth_topic)))
    for t in image_topics:
        if t not in bag.types:
            print(f"  [WARN] topic {t} is not in the bag")
    for cam in cameras:
        setup_camera_alignment(bag, cam)
        print(f"  [{cam.name}] depth_aligned.png: {cam.align_mode}" +
              (f" ({cam.align_source})" if cam.align_source else ""))

    # --- Clocks ---------------------------------------------------------------
    if args.image_time_offset_s is not None:
        offset = args.image_time_offset_s
    elif args.auto_detect_clock_offset:
        img0, odom0 = bag.first(image_topics[0])[0], bag.first(args.odom_topic)[0]
        if img0 is None or odom0 is None:
            raise RuntimeError("--auto_detect_clock_offset needs messages on the image and odometry topics")
        offset = stamp_s(img0) - stamp_s(odom0)
    else:
        offset = 0.0
    print(f"Image time offset: {offset:.3f} s")

    # Header stamp -> receive time: linear fit from the first and last message of an image topic
    calibration = None
    for t in image_topics:
        (m0, r0), (m1, r1) = bag.first(t), bag.last(t)
        if m0 is not None and m1 is not None and stamp_s(m1) != stamp_s(m0):
            h0 = stamp_s(m0)
            calibration = (h0, r0, (r1 - r0) / (stamp_s(m1) - h0))
            break
    if calibration is None:
        print("[WARN] no header -> receive time fit; the frame search reads the whole bag")
    h0_cal, t0_cal_ns, slope_cal = calibration or (0.0, 0, 1e9)
    # Odometry lookup: header stamps in the host epoch are used directly (the receive time adds the image topic
    # delay); header stamps in a different epoch (replayed bag) go through the fit.
    header_epoch_ok = calibration is not None and abs(t0_cal_ns * 1e-9 - h0_cal) < 5.0

    def header_to_receive_s(h_s):
        return h_s if header_epoch_ok else (t0_cal_ns + (h_s - h0_cal) * slope_cal) * 1e-9

    odom = load_odom(bag, args.odom_topic)
    if odom is None:
        print(f"[WARN] no odometry on {args.odom_topic}")

    # --- Broadside from the odometry -----------------------------------------
    if args.broadside_source == "odom":
        if odom is None:
            raise RuntimeError(f"--broadside_source odom needs odometry on {args.odom_topic}")
        for tree_id, info in broadside_info.items():
            info["slam_path_broadside"] = {"pose_xyt": info["pose"], "timestamp_s": info["timestamp_s"],
                                           "fwd_dist_m": info["fwd_dist_m"], "lat_dist_m": info["lat_dist_m"]}
            b = broadside_from_odom(path, odom, np.array(info["tree_xy"]), info["timestamp_s"],
                                    args.odom_time_offset_s, odom_to_base_yaw,
                                    window_s=args.broadside_window_s, rotation_window_s=args.broadside_rotation_window_s)
            if b is None:
                print(f"  [WARN] Tree {tree_id}: odometry fit failed; robot_path broadside pose used")
                info["broadside_source"] = "slam_path (odometry fit failed)"
                continue
            moved = math.hypot(b["pose_xyt"][0] - info["pose"][0], b["pose_xyt"][1] - info["pose"][1])
            print(f"  Tree {tree_id:>3}: t={b['t_slam']:.3f} s ({b['t_slam'] - info['timestamp_s']:+.2f} s) "
                  f"lat={b['lat']:+.3f} m, {moved:.2f} m from the robot_path pose, fit RMS {b['fit'].rms_inlier_m:.2f} m")
            info["world_fit"] = b["fit"]
            info.update(pose=b["pose_xyt"], timestamp_s=b["t_slam"], fwd_dist_m=b["fwd"], lat_dist_m=b["lat"],
                        broadside_source="odom", broadside_fit={
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
                        })

    # --- Frames ---------------------------------------------------------------
    pad_s = max(args.filter_pad_s if args.filter_pad_s is not None else args.max_time_delta_s + 0.5,
                args.max_time_delta_s)
    slack_ns = int(2e9)                       # error of the linear header -> receive time fit
    depth_pad_ns = int((args.max_time_delta_s + 0.5) * abs(slope_cal))
    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0

    for tree_id, info in broadside_info.items():
        raw_target = info["timestamp_s"] + offset
        tree_dir = os.path.join(args.output_dir, f"tree_{int(tree_id):03d}")
        if calibration is None:
            t_min, t_max = None, None
        else:
            center = t0_cal_ns + (raw_target - h0_cal) * slope_cal
            t_min = int(center - pad_s * abs(slope_cal) - slack_ns)
            t_max = int(center + pad_s * abs(slope_cal) + slack_ns)
        ctx = {"robot_poses": path, "offset": offset, "raw_target": raw_target, "max_sync_s": args.max_color_depth_dt_s,
               "odom": odom, "header_to_receive_s": header_to_receive_s, "world_fit": info.get("world_fit"),
               "odom_to_base_yaw": odom_to_base_yaw}

        def depth_for(color_msg, color_rx):
            return nearest_by_stamp(bag, cam.depth_topic, color_rx - depth_pad_ns, color_rx + depth_pad_ns,
                                    stamp_s(color_msg), args.max_time_delta_s)[0]

        cam_meta = {}
        for cam in cameras:
            base = {"color_topic": cam.color_topic, "depth_topic": cam.depth_topic, "buffer_radius": cam.buffer_radius}
            # Color frame nearest to the broadside time, then the depth frame nearest to that color frame
            color_msg, color_rx = nearest_by_stamp(bag, cam.color_topic, t_min, t_max, raw_target, args.max_time_delta_s)
            depth_msg = depth_for(color_msg, color_rx) if color_msg is not None else None
            if depth_msg is None:
                error = f"no {'color' if color_msg is None else 'depth'} frame within {args.max_time_delta_s} s"
                print(f"  [WARN] Tree {tree_id} / {cam.name}: {error}")
                cam_meta[cam.name] = dict(base, saved=False, skip_reason=error)
                continue

            cam_dir = os.path.join(tree_dir, cam.name)
            write_camera_info_yaml(cam_dir, cam)
            dt_c, dt_d = abs(stamp_s(color_msg) - raw_target), abs(stamp_s(depth_msg) - raw_target)
            r = cam.buffer_radius
            if r == 0:
                fm = save_frame(cam_dir, cam, color_msg, depth_msg, ctx)
                cam_meta[cam.name] = dict(base, dt_color_s=float(dt_c), dt_depth_s=float(dt_d), **fm)
                print(f"  [OK] Tree {tree_id} / {cam.name}: dt_color={dt_c:.3f} s")
                continue

            # Buffer: r color frames on each side; each one with the depth frame of the nearest header stamp
            before, after = neighbours(bag, cam.color_topic, color_rx, r)
            colors = [None] * (r - len(before)) + before + [(color_rx, color_msg)] + after + [None] * (r - len(after))
            frames = {}
            for i, item in enumerate(colors):
                name = f"frame_{i - r}"
                if item is None:
                    frames[name] = {"saved": False, "skip_reason": "insufficient adjacent frames in bag"}
                    continue
                c_rx, c_msg = item
                d_msg = depth_msg if i == r else depth_for(c_msg, c_rx)
                if d_msg is None:
                    frames[name] = {"saved": False, "skip_reason": "no depth frame for this color frame"}
                    continue
                frames[name] = save_frame(os.path.join(cam_dir, name), cam, c_msg, d_msg, ctx)
            frames["frame_0"].update(dt_color_s=float(dt_c), dt_depth_s=float(dt_d))
            p0 = frames["frame_0"].get("odom_pose_xyt")
            for fm in frames.values():
                if p0 is not None and fm.get("odom_pose_xyt") is not None:
                    fm["base_delta_from_frame0_xyt"] = base_delta(p0, fm["odom_pose_xyt"], odom_to_base_yaw)
            n_ok = sum(f["saved"] for f in frames.values())
            cam_meta[cam.name] = dict(base, saved=n_ok > 0, frames=frames)
            print(f"  [OK] Tree {tree_id} / {cam.name}: {n_ok}/{2 * r + 1} frames, dt_color={dt_c:.3f} s")

        if not any(c.get("saved") for c in cam_meta.values()):
            print(f"  [SKIP] Tree {tree_id}: no camera frames")
            continue
        meta = {
            "tree_id": int(tree_id),
            "tree_world_xy_m": info["tree_xy"],
            "broadside_robot_pose_xyt": info["pose"][:3],
            "broadside_timestamp_s": info["timestamp_s"],
            "fwd_dist_to_tree_m": info["fwd_dist_m"],
            "lat_dist_to_tree_m": info["lat_dist_m"],
            "broadside_source": info.get("broadside_source", "slam_path"),
            "image_time_offset_applied_s": offset,
            "slam_yaml": os.path.abspath(args.yaml),
            "odom_topic": args.odom_topic,
            "odom_to_base_yaw_deg": args.odom_to_base_yaw_deg,
            "max_color_depth_dt_s": args.max_color_depth_dt_s,
            "cameras": cam_meta,
        }
        for key in ("broadside_fit", "slam_path_broadside"):
            if key in info:
                meta[key] = info[key]
        os.makedirs(tree_dir, exist_ok=True)
        with open(os.path.join(tree_dir, "meta.yaml"), "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
        saved += 1

    print(f"Saved {saved} of {len(broadside_info)} trees in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
