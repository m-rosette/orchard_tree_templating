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
