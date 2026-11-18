#!/usr/bin/env python3
"""
calibrate_depth_to_color.py

Finds the depth->color extrinsics of a RealSense camera from a bag that has both the raw depth topic and the
driver's aligned_depth_to_color topic. Use it when the bag has no extrinsics topic data and no /tf_static.

The aligned image has the same header stamp as the raw depth frame it came from. The tool pairs them by stamp,
then finds the R, t for which depth_registration.register_depth_to_color(raw) agrees best with the aligned
image (coarse search on t_x, then Nelder-Mead on t and the rotation vector).

Usage:
    ros2 run tree_template calibrate_depth_to_color \\
        --bag /path/to/bag_dir --camera_ns /camera/base_camera \\
        --out config/camera_extrinsics/prosser_2025_base_camera.yaml

The output yaml goes to fast_slam_rgbd_extraction (--cam1_depth_to_color / --cam2_depth_to_color).
"""

import argparse
import datetime
import os
import sqlite3

import numpy as np
import yaml
from scipy.optimize import minimize

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tree_template.depth_registration import (
    camera_info_to_dict,
    camera_matrix,
    check_no_distortion,
    register_depth_to_color,
    rotation_from_rotvec,
)
from tree_template.fast_slam_rgbd_extraction import _find_sqlite3_db, ros_image_to_cv2

CLIP_MM = 100.0


def _stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _topics(con):
    return {name: (tid, typ) for tid, name, typ in con.execute("SELECT id, name, type FROM topics")}


def _first_msg(con, topics, name):
    if name not in topics:
        return None
    tid, typ = topics[name]
    row = con.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1", (tid,)).fetchone()
    return deserialize_message(bytes(row[0]), get_message(typ)) if row else None


def load_pairs(con, topics, depth_topic, aligned_topic, n_pairs, search_ns=int(0.5e9)):
    """Return up to n_pairs (raw_depth, aligned) uint16 image pairs with the same header stamp."""
    a_id, a_typ = topics[aligned_topic]
    d_id, d_typ = topics[depth_topic]
    A, D = get_message(a_typ), get_message(d_typ)
    rows = con.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (a_id,)).fetchall()
    if not rows:
        return []
    pick = np.unique(np.linspace(0, len(rows) - 1, min(n_pairs, len(rows))).round().astype(int))
    pairs = []
    for i in pick:
        t_rx = rows[i][0]
        (a_data,) = con.execute("SELECT data FROM messages WHERE topic_id=? AND timestamp=?", (a_id, t_rx)).fetchone()
        a_msg = deserialize_message(bytes(a_data), A)
        target = _stamp_ns(a_msg)
        match = None
        for (d_data,) in con.execute(
            "SELECT data FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ?",
            (d_id, t_rx - search_ns, t_rx + search_ns),
        ):
            d_msg = deserialize_message(bytes(d_data), D)
            if abs(_stamp_ns(d_msg) - target) < 1_000_000:   # 1 ms
                match = d_msg
                break
        if match is not None:
            pairs.append((ros_image_to_cv2(match), ros_image_to_cv2(a_msg)))
    return pairs


def cost(params, pairs, K_d, K_c):
    """Mean |registered - aligned| (mm, clipped) plus a penalty for pixels with data in only one of them."""
    t_mm = np.asarray(params[:3]) * 1000.0
    R = rotation_from_rotvec(params[3:6])
    total = 0.0
    for raw, ref in pairs:
        reg = register_depth_to_color(raw, K_d, K_c, R, t_mm, ref.shape[:2]).astype(np.float64)
        ref = ref.astype(np.float64)
        both = (reg > 0) & (ref > 0)
        one = (reg > 0) ^ (ref > 0)
        diff = np.minimum(np.abs(reg[both] - ref[both]), CLIP_MM)
        n = max(int(both.sum() + one.sum()), 1)
        total += (diff.sum() + CLIP_MM * one.sum()) / n
    return total / len(pairs)


def stats(params, pairs, K_d, K_c):
    t_mm = np.asarray(params[:3]) * 1000.0
    R = rotation_from_rotvec(params[3:6])
    diffs, agree, n_ref = [], 0, 0
    for raw, ref in pairs:
        reg = register_depth_to_color(raw, K_d, K_c, R, t_mm, ref.shape[:2]).astype(np.float64)
        both = (reg > 0) & (ref > 0)
        diffs.append(np.abs(reg[both] - ref[both]))
        agree += int(((reg > 0) == (ref > 0)).sum())
        n_ref += reg.size
    d = np.concatenate(diffs)
    return {
        "median_abs_mm": float(np.median(d)),
        "p95_abs_mm": float(np.percentile(d, 95)),
        "frac_within_10mm": float(np.mean(d <= 10.0)),
        "valid_mask_agreement": float(agree / n_ref),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="rosbag2 directory or .db3 file (sqlite3)")
    ap.add_argument("--camera_ns", required=True, help="for example /camera/base_camera")
    ap.add_argument("--depth_topic", default=None, help="default <camera_ns>/depth/image_rect_raw")
    ap.add_argument("--aligned_topic", default=None, help="default <camera_ns>/aligned_depth_to_color/image_raw")
    ap.add_argument("--depth_info_topic", default=None, help="default <camera_ns>/depth/camera_info")
    ap.add_argument("--color_info_topic", default=None, help="default <camera_ns>/color/camera_info")
    ap.add_argument("--n_pairs", type=int, default=8, help="number of image pairs for the fit")
    ap.add_argument("--out", required=True, help="output yaml")
    args = ap.parse_args()

    ns = args.camera_ns.rstrip("/")
    depth_topic = args.depth_topic or f"{ns}/depth/image_rect_raw"
    aligned_topic = args.aligned_topic or f"{ns}/aligned_depth_to_color/image_raw"
    depth_info_topic = args.depth_info_topic or f"{ns}/depth/camera_info"
    color_info_topic = args.color_info_topic or f"{ns}/color/camera_info"

    db = _find_sqlite3_db(args.bag)
    if db is None:
        raise FileNotFoundError(f"No .db3 file under {args.bag}")
    con = sqlite3.connect(db)
    topics = _topics(con)
    for t in (depth_topic, aligned_topic, depth_info_topic, color_info_topic):
        if t not in topics:
            raise RuntimeError(f"Topic {t} is not in the bag")

    d_info = camera_info_to_dict(_first_msg(con, topics, depth_info_topic))
    c_info = camera_info_to_dict(_first_msg(con, topics, color_info_topic))
    for name, info in (("depth", d_info), ("color", c_info)):
        if not check_no_distortion(info):
            print(f"[WARN] {name} camera_info has distortion {info['d']}; this tool ignores it.")
    K_d, K_c = camera_matrix(d_info), camera_matrix(c_info)

    pairs = load_pairs(con, topics, depth_topic, aligned_topic, args.n_pairs)
    con.close()
    if not pairs:
        raise RuntimeError("No raw depth / aligned depth pairs with the same header stamp")
    print(f"{len(pairs)} image pairs (raw {pairs[0][0].shape} -> aligned {pairs[0][1].shape})")

    # Coarse search on t_x (the baseline between the depth origin and the color sensor), R = I
    coarse = pairs[: min(3, len(pairs))]
    grid = np.arange(-0.060, 0.0601, 0.002)
    costs = [cost([tx, 0, 0, 0, 0, 0], coarse, K_d, K_c) for tx in grid]
    tx0 = float(grid[int(np.argmin(costs))])
    print(f"coarse t_x = {tx0:+.4f} m (cost {min(costs):.2f})")

    x0 = np.array([tx0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = minimize(
        cost, x0, args=(pairs, K_d, K_c), method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 1500,
                 "initial_simplex": x0 + np.vstack([np.zeros(6), np.diag([0.002] * 3 + [0.003] * 3)])},
    )
    p = res.x
    R = rotation_from_rotvec(p[3:6])
    s = stats(p, pairs, K_d, K_c)
    s0 = stats([0, 0, 0, 0, 0, 0], pairs, K_d, K_c)
    print(f"fit: t = {np.round(p[:3], 5).tolist()} m, rotvec = {np.round(np.degrees(p[3:6]), 3).tolist()} deg")
    print(f"     {s}")
    print(f"identity extrinsics for comparison: {s0}")

    out = {
        "camera_ns": ns,
        "depth_frame": d_info["frame_id"],
        "color_frame": c_info["frame_id"],
        "convention": "p_color = R @ p_depth + t",
        "rotation": [float(x) for x in R.reshape(-1)],
        "translation_m": [float(x) for x in p[:3]],
        "rotvec_rad": [float(x) for x in p[3:6]],
        "fit": dict(s, n_pairs=len(pairs), cost=float(res.fun), bag=os.path.abspath(args.bag),
                    depth_topic=depth_topic, aligned_topic=aligned_topic,
                    date=datetime.date.today().isoformat()),
        "depth_camera_info": d_info,
        "color_camera_info": c_info,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.dump(out, f, default_flow_style=None, sort_keys=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
