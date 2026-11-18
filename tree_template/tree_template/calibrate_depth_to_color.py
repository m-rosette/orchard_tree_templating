#!/usr/bin/env python3
"""
calibrate_depth_to_color.py

Finds the depth->color extrinsics of a RealSense camera from a bag with the raw depth topic and the driver's
aligned_depth_to_color topic (for bags without extrinsics data). An aligned image has the header stamp of its raw
depth frame. The tool pairs the two by stamp and finds the R, t for which register_depth_to_color(raw) agrees best
with the aligned image: grid search on t_x, then Nelder-Mead on t and the rotation vector.

Usage:
    ros2 run tree_template calibrate_depth_to_color --bag /path/to/bag_dir --camera_ns /camera/base_camera \
        --out config/camera_extrinsics/prosser_2025_base_camera.yaml

fast_slam_rgbd_extraction reads the output (--cam1_depth_to_color / --cam2_depth_to_color).
"""

import argparse
import datetime
import os

import numpy as np
import yaml
from scipy.optimize import minimize

from tree_template.bag_reader import BagReader
from tree_template.depth_registration import (
    camera_info_to_dict,
    camera_matrix,
    distortion,
    register_depth_to_color,
    rotation_from_rotvec,
)
from tree_template.fast_slam_rgbd_extraction import depth_to_mm, stamp_s

CLIP_MM = 100.0


def load_pairs(bag: BagReader, depth_topic: str, aligned_topic: str, n_pairs: int, search_ns: int = int(0.5e9)):
    """Up to n_pairs (raw depth, aligned depth) uint16 images with the same header stamp (1 ms)."""
    rx = [t for _, _, t in bag.raw([aligned_topic])]
    pairs = []
    for i in np.unique(np.linspace(0, len(rx) - 1, min(n_pairs, len(rx))).round().astype(int)):
        aligned, _ = bag.first(aligned_topic, rx[i])
        for _, raw, _ in bag.messages([depth_topic], rx[i] - search_ns, rx[i] + search_ns):
            if abs(stamp_s(raw) - stamp_s(aligned)) < 1e-3:
                pairs.append((depth_to_mm(raw), depth_to_mm(aligned)))
                break
    return pairs


def cost(params, pairs, cal):
    """Mean |registered - aligned| (mm, clipped at CLIP_MM); CLIP_MM for pixels with data in only one image."""
    t_mm = np.asarray(params[:3]) * 1000.0
    R = rotation_from_rotvec(params[3:6])
    total = 0.0
    for raw, ref in pairs:
        reg = register_depth_to_color(raw, cal[0], cal[1], R, t_mm, ref.shape[:2], cal[2], cal[3]).astype(np.float64)
        ref = ref.astype(np.float64)
        both = (reg > 0) & (ref > 0)
        one = (reg > 0) ^ (ref > 0)
        diff = np.minimum(np.abs(reg[both] - ref[both]), CLIP_MM)
        n = max(int(both.sum() + one.sum()), 1)
        total += (diff.sum() + CLIP_MM * one.sum()) / n
    return total / len(pairs)


def stats(params, pairs, cal):
    """Agreement of register_depth_to_color(raw) with the aligned images."""
    t_mm = np.asarray(params[:3]) * 1000.0
    R = rotation_from_rotvec(params[3:6])
    diffs, agree, n_ref = [], 0, 0
    for raw, ref in pairs:
        reg = register_depth_to_color(raw, cal[0], cal[1], R, t_mm, ref.shape[:2], cal[2], cal[3]).astype(np.float64)
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
    ap.add_argument("--bag", required=True, help="rosbag2 bag (sqlite3 or mcap)")
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

    bag = BagReader(args.bag)
    for t in (depth_topic, aligned_topic, depth_info_topic, color_info_topic):
        if t not in bag.types:
            raise RuntimeError(f"Topic {t} is not in the bag")
    d_info = camera_info_to_dict(bag.first(depth_info_topic)[0])
    c_info = camera_info_to_dict(bag.first(color_info_topic)[0])
    # (K_depth, K_color, D_depth, D_color)
    cal = (camera_matrix(d_info), camera_matrix(c_info), distortion(d_info), distortion(c_info))

    pairs = load_pairs(bag, depth_topic, aligned_topic, args.n_pairs)
    if not pairs:
        raise RuntimeError("No raw depth / aligned depth pairs with the same header stamp")
    print(f"{len(pairs)} image pairs (raw {pairs[0][0].shape} -> aligned {pairs[0][1].shape})")

    # Grid search on t_x (depth origin -> color sensor), R = I
    coarse = pairs[: min(3, len(pairs))]
    grid = np.arange(-0.060, 0.0601, 0.002)
    costs = [cost([tx, 0, 0, 0, 0, 0], coarse, cal) for tx in grid]
    tx0 = float(grid[int(np.argmin(costs))])
    print(f"coarse t_x = {tx0:+.4f} m (cost {min(costs):.2f})")

    x0 = np.array([tx0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = minimize(
        cost, x0, args=(pairs, cal), method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 1500,
                 "initial_simplex": x0 + np.vstack([np.zeros(6), np.diag([0.002] * 3 + [0.003] * 3)])},
    )
    p = res.x
    R = rotation_from_rotvec(p[3:6])
    s = stats(p, pairs, cal)
    s0 = stats([0, 0, 0, 0, 0, 0], pairs, cal)
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
