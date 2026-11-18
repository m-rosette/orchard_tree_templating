"""
trellis_geometry.py

Pure functions that put the trunks of a saved SLAM map into the local frame of one anchor tree.
This module does not import ROS, so tests and offline tools can use it.

Local frame:
    origin  the robot pose when the anchor tree was broadside (broadside_robot_pose_xyt)
    +X      along the row (the world positions are de-rotated by row_yaw)
    +Y      to the left of the robot

The anchor trunk is at (fwd_dist_to_tree_m, lat_dist_to_tree_m) from its meta.yaml when row_yaw is the
broadside yaw of the anchor. When row_yaw is the run-wide SLAM row yaw, the anchor is at that point rotated
by (broadside_yaw - row_yaw).
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def load_meta_yamls(extraction_dir: str) -> Dict[int, dict]:
    """Return the tree_*/meta.yaml files in extraction_dir as a dict keyed by integer tree_id."""
    paths = sorted(glob.glob(os.path.join(extraction_dir, "tree_*", "meta.yaml")))
    if not paths:
        raise FileNotFoundError(f"No tree_*/meta.yaml files found under: {extraction_dir}")

    tree_data: Dict[int, dict] = {}
    for p in paths:
        with open(p, "r") as f:
            meta = yaml.safe_load(f)
        if "tree_id" in meta:
            tid = int(meta["tree_id"])
        else:
            tid = int(Path(p).parent.name.split("_")[-1])   # for example "tree_008"
            meta["tree_id"] = tid
        tree_data[tid] = meta
    return tree_data


def extract_world_xy(meta: dict) -> np.ndarray:
    """Return the trunk [x, y] in the SLAM world frame (key tree_world_xy_m)."""
    wp = meta.get("tree_world_xy_m")
    if wp is None:
        raise KeyError(f"meta.yaml for tree {meta.get('tree_id')} missing 'tree_world_xy_m'")
    return np.array([float(wp[0]), float(wp[1])], dtype=np.float64)


def extract_world_z(meta: dict) -> float:
    """fast_slam_rgbd_extraction does not save a trunk z. All trunks are at ground level."""
    return 0.0


def find_slam_results_yaml(extraction_dir: str) -> str:
    """
    Return the slam_trunks_*.yaml file in extraction_dir (save_best_particle_map.py writes it).
    If there are more than one, return the newest name (the names contain the date and time).
    Return "" if there is none.
    """
    paths = sorted(glob.glob(os.path.join(extraction_dir, "slam_trunks_*.yaml")))
    return paths[-1] if paths else ""


def load_global_row_yaw(slam_results_yaml: str) -> Optional[float]:
    """
    Return row_yaw_rad from the yaml that save_best_particle_map.py writes.
    Return None if the path is empty or missing, or the field is missing, null or not finite.
    """
    if not slam_results_yaml:
        return None
    p = Path(slam_results_yaml).expanduser()
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        v = float(data["row_yaw_rad"])
    except Exception:
        return None
    return v if np.isfinite(v) else None


def anchor_transform(
    tree_data: Dict[int, dict],
    anchor_id: int,
    global_row_yaw: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Return the local frame of the anchor tree:

    robot_origin_xy  (2,) world-frame robot position when the anchor tree was broadside
    row_yaw          yaw (rad) that de-rotates the world positions, so the row is along local +X.
                     global_row_yaw (the run-wide SLAM estimate) if it is set, otherwise the broadside yaw
                     of the anchor tree.
    """
    if anchor_id not in tree_data:
        raise ValueError(
            f"Anchor tree id {anchor_id} not found. Available ids: {sorted(tree_data.keys())}"
        )
    xyt = tree_data[anchor_id].get("broadside_robot_pose_xyt", [0.0, 0.0, 0.0])
    robot_origin_xy = np.array([float(xyt[0]), float(xyt[1])], dtype=np.float64)
    broadside_yaw = float(xyt[2]) if len(xyt) >= 3 else 0.0
    row_yaw = float(global_row_yaw) if global_row_yaw is not None else broadside_yaw
    return robot_origin_xy, row_yaw


def transform_to_local(world_xy: np.ndarray, robot_origin_xy: np.ndarray, row_yaw: float) -> np.ndarray:
    """Return the local (x, y) of a world point: R(-row_yaw) @ (world_xy - robot_origin_xy)."""
    c, s = np.cos(-row_yaw), np.sin(-row_yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return rot @ (np.asarray(world_xy, dtype=np.float64) - robot_origin_xy)


def local_positions(
    tree_data: Dict[int, dict],
    anchor_id: int,
    global_row_yaw: Optional[float] = None,
) -> Tuple[List[int], np.ndarray, List[int]]:
    """
    Return the local (x, y) of all trees that have a world position, in tree_id order.

    ids      tree ids, one for each row of xy
    xy       (N, 2) local positions
    skipped  tree ids without tree_world_xy_m
    """
    robot_origin_xy, row_yaw = anchor_transform(tree_data, anchor_id, global_row_yaw)
    ids: List[int] = []
    rows: List[np.ndarray] = []
    skipped: List[int] = []
    for tid in sorted(tree_data.keys()):
        try:
            world_xy = extract_world_xy(tree_data[tid])
        except KeyError:
            skipped.append(tid)
            continue
        ids.append(tid)
        rows.append(transform_to_local(world_xy, robot_origin_xy, row_yaw))
    xy = np.vstack(rows) if rows else np.zeros((0, 2), dtype=np.float64)
    return ids, xy, skipped


def expected_anchor_local(meta: dict, row_yaw: float) -> np.ndarray:
    """
    Return the local (x, y) that the anchor trunk must have, from its meta.yaml:
    R(broadside_yaw - row_yaw) @ (fwd_dist_to_tree_m, lat_dist_to_tree_m).
    """
    broadside_yaw = float(meta["broadside_robot_pose_xyt"][2])
    fl = np.array([float(meta["fwd_dist_to_tree_m"]), float(meta["lat_dist_to_tree_m"])], dtype=np.float64)
    a = broadside_yaw - row_yaw
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64) @ fl


# =============================================================================
#  Map correction from base-camera trunk detections
# =============================================================================

def rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def frame_poses_local(
    meta: dict,
    cam: str,
    robot_origin_xy: np.ndarray,
    row_yaw: float,
) -> Dict[int, Tuple[np.ndarray, float, str]]:
    """
    Pose of the robot base frame in the local frame for each saved frame of camera cam (buffer frames, or the
    single frame as index 0): {frame index: (xy (2,), yaw, source)}.

    source "world_pose": from world_pose_xyt (fast_slam_rgbd_extraction --broadside_source odom).
    source "broadside":  the frame has no world pose (older extraction); the broadside pose is used.
    """
    bs = meta.get("broadside_robot_pose_xyt", [robot_origin_xy[0], robot_origin_xy[1], row_yaw])
    cam_meta = (meta.get("cameras") or {}).get(cam) or {}
    frames = cam_meta.get("frames")
    if frames is None:
        frames = {"frame_0": cam_meta}
    out = {}
    for name, fm in frames.items():
        if not fm or not fm.get("saved", False):
            continue
        idx = int(name.split("_")[-1])
        wp, src = fm.get("world_pose_xyt"), "world_pose"
        if wp is None:
            wp, src = bs, "broadside"
        xy = rot2(-row_yaw) @ (np.asarray(wp[:2], dtype=np.float64) - robot_origin_xy)
        out[idx] = (xy, float(wp[2]) - row_yaw, src)
    return out


def base_to_local(points_base: np.ndarray, frame_xy: np.ndarray, frame_yaw: float) -> np.ndarray:
    """(N, 2) points in the robot base frame of one image -> local frame (frame pose from frame_poses_local)."""
    return np.atleast_2d(points_base) @ rot2(frame_yaw).T + frame_xy


def associate(
    meas_local: np.ndarray,
    pred_local: np.ndarray,
    pred_ids: List[int],
    gate_m: float,
    meas_frames: Optional[List[int]] = None,
) -> List[Tuple[int, int, float]]:
    """
    Match measurements to predicted trunks, inside gate_m and on the same side of the robot (sign of y).
    In one image (same meas_frames value) each trunk takes at most one measurement and each measurement at most
    one trunk: the pairs are taken by increasing distance. Over different images a trunk can match many times.
    Return [(measurement index, tree id, distance)], sorted by measurement index.
    """
    meas_local = np.atleast_2d(meas_local)
    if meas_frames is None:
        meas_frames = list(range(len(meas_local)))     # each measurement its own image: nearest trunk
    cand = []
    for i, m in enumerate(meas_local):
        d = np.linalg.norm(pred_local - m, axis=1)
        ok = (np.sign(pred_local[:, 1]) == np.sign(m[1])) & (d <= gate_m)
        cand.extend((float(d[j]), i, j) for j in np.nonzero(ok)[0])
    used_meas, used_trunk, out = set(), set(), []
    for d, i, j in sorted(cand):
        if i in used_meas or (meas_frames[i], j) in used_trunk:
            continue
        used_meas.add(i)
        used_trunk.add((meas_frames[i], j))
        out.append((i, int(pred_ids[j]), d))
    return sorted(out)


def cluster_detections(
    meas_local: np.ndarray,
    meas_frames: List[int],
    eps_m: float = 0.10,
) -> List[List[int]]:
    """
    Group the detections of one object over several images. After the motion compensation, the detections of one
    trunk in different images agree to a few cm. A detection joins the nearest group whose centroid is within
    eps_m and that has no detection from the same image yet; otherwise it starts a new group.
    Return the groups as lists of detection indices.
    """
    meas_local = np.atleast_2d(meas_local)
    groups: List[List[int]] = []
    for i in np.argsort(meas_frames, kind="stable"):
        best, best_d = None, eps_m
        for g in groups:
            if any(meas_frames[k] == meas_frames[i] for k in g):
                continue
            d = float(np.linalg.norm(meas_local[g].mean(axis=0) - meas_local[i]))
            if d <= best_d:
                best, best_d = g, d
        if best is None:
            groups.append([int(i)])
        else:
            best.append(int(i))
    return [sorted(g) for g in groups]


def associate_groups(
    meas_local: np.ndarray,
    groups: List[List[int]],
    pred_local: np.ndarray,
    pred_ids: List[int],
    gate_m: float,
) -> List[Tuple[List[int], int, float]]:
    """
    One-to-one match of detection groups (cluster_detections) to predicted trunks, by increasing distance between
    the group centroid and the trunk, inside gate_m and on the same side of the robot (sign of y).
    Return [(group, tree id, distance)].
    """
    meas_local = np.atleast_2d(meas_local)
    cand = []
    for gi, g in enumerate(groups):
        c = meas_local[g].mean(axis=0)
        d = np.linalg.norm(pred_local - c, axis=1)
        ok = (np.sign(pred_local[:, 1]) == np.sign(c[1])) & (d <= gate_m)
        cand.extend((float(d[j]), gi, j) for j in np.nonzero(ok)[0])
    used_g, used_t, out = set(), set(), []
    for d, gi, j in sorted(cand):
        if gi in used_g or j in used_t:
            continue
        used_g.add(gi)
        used_t.add(j)
        out.append((groups[gi], int(pred_ids[j]), d))
    return out


def fit_map_correction(
    pred: np.ndarray,
    meas: np.ndarray,
    tree_ids: List[int],
    max_yaw_rad: float,
) -> Tuple[float, np.ndarray, bool]:
    """
    Rigid 2-D correction (yaw, t) with meas ~ R(yaw) @ pred + t, for matched pairs (pred[i], meas[i]) of the
    trunks tree_ids[i].

    One distinct trunk: translation only (yaw = 0), t = mean(meas - pred).
    Two or more distinct trunks: least-squares rotation + translation. The pairs are weighted so each trunk has
    the same total weight (a trunk seen in 5 frames does not count 5 times). If |yaw| > max_yaw_rad, the yaw is
    clamped and the translation is fitted again with the clamped yaw.

    Return (yaw, t, clamped).
    """
    pred = np.atleast_2d(pred)
    meas = np.atleast_2d(meas)
    ids = np.asarray(tree_ids)
    uniq = np.unique(ids)
    w = np.array([1.0 / np.sum(ids == i) for i in ids])
    w = w / w.sum()

    if len(uniq) < 2:
        return 0.0, np.sum(w[:, None] * (meas - pred), axis=0), False

    cp = np.sum(w[:, None] * pred, axis=0)
    cm = np.sum(w[:, None] * meas, axis=0)
    a, b = pred - cp, meas - cm
    s = np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    c = np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    yaw = float(np.arctan2(s, c))
    clamped = abs(yaw) > max_yaw_rad
    if clamped:
        yaw = float(np.clip(yaw, -max_yaw_rad, max_yaw_rad))
    t = cm - rot2(yaw) @ cp
    return yaw, t, clamped


def apply_map_correction(xy: np.ndarray, yaw: float, t: np.ndarray) -> np.ndarray:
    """Apply the correction to (N, 2) local positions."""
    return np.atleast_2d(xy) @ rot2(yaw).T + t
