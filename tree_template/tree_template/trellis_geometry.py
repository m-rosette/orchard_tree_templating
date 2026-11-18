"""
trellis_geometry.py

Pure functions that put the trunks of a saved SLAM map into the local frame of one anchor tree.
This module does not import ROS, so tests and offline tools can use it. 2-D poses and rigid transforms are
skimage.transform.EuclideanTransform objects (pose_tf).

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
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import linear_sum_assignment
from skimage.transform import EuclideanTransform


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


def pose_tf(x: float, y: float, yaw: float) -> EuclideanTransform:
    """2-D pose as a transform: points in the frame of the pose -> points in the parent frame."""
    return EuclideanTransform(rotation=float(yaw), translation=(float(x), float(y)))


def local_tf(robot_origin_xy: np.ndarray, row_yaw: float) -> EuclideanTransform:
    """World -> local frame (origin at robot_origin_xy, +X along the row)."""
    return pose_tf(robot_origin_xy[0], robot_origin_xy[1], row_yaw).inverse


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
    ids = [t for t in sorted(tree_data) if "tree_world_xy_m" in tree_data[t]]
    skipped = [t for t in sorted(tree_data) if t not in ids]
    world = np.array([extract_world_xy(tree_data[t]) for t in ids]).reshape(-1, 2)
    return ids, local_tf(robot_origin_xy, row_yaw)(world), skipped


def expected_anchor_local(meta: dict, row_yaw: float) -> np.ndarray:
    """
    Return the local (x, y) that the anchor trunk must have, from its meta.yaml:
    R(broadside_yaw - row_yaw) @ (fwd_dist_to_tree_m, lat_dist_to_tree_m).
    """
    broadside_yaw = float(meta["broadside_robot_pose_xyt"][2])
    fl = [[float(meta["fwd_dist_to_tree_m"]), float(meta["lat_dist_to_tree_m"])]]
    return EuclideanTransform(rotation=broadside_yaw - row_yaw)(fl)[0]


# =============================================================================
#  Map correction from base-camera trunk detections
# =============================================================================

def frame_poses_local(
    meta: dict,
    cam: str,
    robot_origin_xy: np.ndarray,
    row_yaw: float,
) -> Dict[int, Tuple[EuclideanTransform, str]]:
    """
    For each saved frame of camera cam (buffer frames, or the single frame as index 0): (transform robot base
    frame of that image -> local frame, source).

    source "world_pose": from world_pose_xyt (fast_slam_rgbd_extraction --broadside_source odom).
    source "broadside":  the frame has no world pose (older extraction); the broadside pose is used.
    """
    bs = meta.get("broadside_robot_pose_xyt", [robot_origin_xy[0], robot_origin_xy[1], row_yaw])
    cam_meta = (meta.get("cameras") or {}).get(cam) or {}
    frames = cam_meta.get("frames")
    if frames is None:
        frames = {"frame_0": cam_meta}
    to_local = local_tf(robot_origin_xy, row_yaw)
    out = {}
    for name, fm in frames.items():
        if not fm or not fm.get("saved", False):
            continue
        wp, src = fm.get("world_pose_xyt"), "world_pose"
        if wp is None:
            wp, src = bs, "broadside"
        out[int(name.split("_")[-1])] = (pose_tf(*wp[:3]) + to_local, src)
    return out


def cluster_detections(meas_local: np.ndarray, eps_m: float = 0.10) -> List[List[int]]:
    """
    Group the detections of one object over several images (complete linkage: every two detections of a group
    are within eps_m). After the motion compensation, the detections of one trunk agree to a few cm.
    Return the groups as lists of detection indices.
    """
    meas_local = np.atleast_2d(meas_local)
    if len(meas_local) == 1:
        return [[0]]
    labels = fclusterdata(meas_local, t=eps_m, criterion="distance", method="complete")
    return [np.nonzero(labels == lab)[0].tolist() for lab in np.unique(labels)]


def associate_groups(
    meas_local: np.ndarray,
    groups: List[List[int]],
    pred_local: np.ndarray,
    pred_ids: List[int],
    gate_m: float,
) -> List[Tuple[List[int], int, float]]:
    """
    One-to-one match of detection groups to predicted trunks with the smallest total centroid distance
    (linear_sum_assignment), only for pairs inside gate_m and on the same side of the robot (sign of y).
    Return [(group, tree id, distance)].
    """
    meas_local = np.atleast_2d(meas_local)
    centroids = np.array([meas_local[g].mean(axis=0) for g in groups])
    d = np.linalg.norm(centroids[:, None, :] - pred_local[None, :, :], axis=2)
    ok = (d <= gate_m) & (np.sign(centroids[:, None, 1]) == np.sign(pred_local[None, :, 1]))
    rows, cols = linear_sum_assignment(np.where(ok, d, gate_m * 1e3))
    return [(groups[r], int(pred_ids[c]), float(d[r, c])) for r, c in zip(rows, cols) if ok[r, c]]


def fit_map_correction(
    pred: np.ndarray,
    meas: np.ndarray,
    tree_ids: List[int],
    max_yaw_rad: float,
) -> Tuple[EuclideanTransform, bool]:
    """
    Rigid 2-D correction T with meas ~ T(pred), for matched pairs (pred[i], meas[i]) of the trunks tree_ids[i].
    Each trunk has the same weight: the fit uses the mean detection of each trunk.

    One distinct trunk: translation only. Two or more: least-squares rotation + translation
    (EuclideanTransform.estimate). If |yaw| > max_yaw_rad, the yaw is clamped and the translation is fitted
    again with the clamped yaw.

    Return (T, clamped).
    """
    ids = np.asarray(tree_ids)
    uniq = np.unique(ids)
    P = np.array([np.atleast_2d(pred)[ids == u][0] for u in uniq])
    M = np.array([np.atleast_2d(meas)[ids == u].mean(axis=0) for u in uniq])
    yaw = 0.0
    if len(uniq) >= 2:
        T = EuclideanTransform()
        T.estimate(P, M)
        yaw = float(T.rotation)
    clamped = abs(yaw) > max_yaw_rad
    yaw = float(np.clip(yaw, -max_yaw_rad, max_yaw_rad))
    R = EuclideanTransform(rotation=yaw)
    return EuclideanTransform(rotation=yaw, translation=np.mean(M - R(P), axis=0)), clamped
