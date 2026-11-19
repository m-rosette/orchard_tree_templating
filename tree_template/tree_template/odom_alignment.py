"""
odom_alignment.py

Puts the robot odometry into the SLAM world frame near one time, so the broadside pose of a tree comes from a
smooth trajectory. No ROS imports.

The SLAM robot_path (/odom_slam_best) is the pose of the best particle at each time and jumps when the best particle
changes. Near one time, a trimmed rigid 2-D fit maps the odometry onto the path and ignores the jumps.

Tree fit (tree_local_fit): the trunk detections go into the odometry frame and make odometry-frame trees
(detection_trees). These trees are matched to the SLAM map trees near one time, and the fit maps the odometry onto
the map trees. It does not use the robot_path. At the start of a row the robot_path can jump by 0.5-0.9 m
while the filter converges, and the path fit is then bad.

Arrays (rows [x, y, yaw, t]):
    path  SLAM robot_path: world frame, SLAM clock
    odom  odometry: odometry frame, odometry clock (t_odom = t_slam + time_offset; 0 for SLAM on a replay of the
          same bag with the bag clock)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.transform import EuclideanTransform

from tree_template.trellis_geometry import pose_tf


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def interp_pose(poses: np.ndarray, t) -> np.ndarray:
    """
    (K, 3) [x, y, yaw] of poses (rows [x, y, yaw, t], t increasing) at times t: linear, yaw unwrapped.
    Outside the time range: the first or last pose.
    """
    v = np.column_stack([poses[:, 0], poses[:, 1], np.unwrap(poses[:, 2])])
    f = interp1d(poses[:, 3], v, axis=0, bounds_error=False, fill_value=(v[0], v[-1]), assume_sorted=True)
    out = np.atleast_2d(f(np.atleast_1d(t)))
    out[:, 2] = _wrap(out[:, 2])
    return out


@dataclass
class LocalFit:
    tf: EuclideanTransform     # odometry frame -> world frame
    rms_inlier_m: float        # RMS residual of the inliers
    n_used: int
    n_inliers: int
    span_m: float              # odometry travel in the window

    @property
    def theta(self) -> float:
        return float(self.tf.rotation)

    @property
    def t(self) -> np.ndarray:
        return self.tf.translation


def robust_local_fit(
    path: np.ndarray,
    odom: np.ndarray,
    time_offset: float,
    t_center: float,
    window_s: float = 10.0,
    theta: Optional[float] = None,
    min_points: int = 8,
    iters: int = 4,
) -> Optional[LocalFit]:
    """
    Fit odometry -> world to the path poses within window_s of t_center. With theta: translation only.
    After each fit, poses with residual > max(3 * median residual, 5 cm) are dropped (iters times).
    Return None if fewer than min_points path poses have odometry.
    """
    m = np.abs(path[:, 3] - t_center) <= window_s
    t_odom = path[m, 3] + time_offset
    m_idx = np.nonzero(m)[0]
    cover = (t_odom >= odom[0, 3]) & (t_odom <= odom[-1, 3])
    m_idx, t_odom = m_idx[cover], t_odom[cover]
    if m_idx.size < min_points:
        return None
    A = interp_pose(odom, t_odom)[:, :2]
    B = path[m_idx, :2]

    def fit(k) -> EuclideanTransform:
        if theta is None:
            T = EuclideanTransform()
            T.estimate(A[k], B[k])
            return T
        R = EuclideanTransform(rotation=theta)
        return EuclideanTransform(rotation=theta, translation=np.mean(B[k] - R(A[k]), axis=0))

    keep = np.ones(len(A), dtype=bool)
    for _ in range(iters):
        r = np.linalg.norm(fit(keep)(A) - B, axis=1)
        new_keep = r <= max(3.0 * float(np.median(r[keep])), 0.05)
        if new_keep.sum() < min_points or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    T = fit(keep)
    r = np.linalg.norm(T(A) - B, axis=1)
    span = float(np.sum(np.linalg.norm(np.diff(A, axis=0), axis=1)))
    return LocalFit(T, float(np.sqrt(np.mean(r[keep] ** 2))), int(len(A)), int(keep.sum()), span)


def world_poses(fit: LocalFit, odom: np.ndarray, t_odom: np.ndarray, odom_to_base_yaw: float) -> np.ndarray:
    """
    (K, 3) world [x, y, yaw] of the robot base at odometry times t_odom.
    odom_to_base_yaw: yaw of the base frame in the odometry child frame.
    """
    p = interp_pose(odom, t_odom)
    return np.column_stack([fit.tf(p[:, :2]), _wrap(p[:, 2] + fit.theta + odom_to_base_yaw)])


def detection_trees(
    odom: np.ndarray,
    meas: np.ndarray,
    time_offset: float,
    odom_to_base_yaw: float,
    eps_m: float = 0.15,
    min_hits: int = 3,
) -> np.ndarray:
    """
    (K, 4) [x, y, t, n] odometry-frame trees from trunk detections.
    meas rows [x, y, t]: the trunk in the robot base frame (trunk_detection_relay TrunkInfo), image time on the SLAM
    clock. Each detection goes into the odometry frame with the odometry pose at its time. Detections closer than
    eps_m are one tree (single linkage). t is the median detection time of the tree, n the number of detections.
    Trees with fewer than min_hits detections are dropped.
    """
    meas = np.asarray(meas, dtype=np.float64).reshape(-1, 3)
    t_odom = meas[:, 2] + time_offset
    ok = (t_odom >= odom[0, 3]) & (t_odom <= odom[-1, 3])
    if ok.sum() < max(min_hits, 2):
        return np.zeros((0, 4))
    p = interp_pose(odom, t_odom[ok])
    yaw = p[:, 2] + odom_to_base_yaw
    c, s = np.cos(yaw), np.sin(yaw)
    mx, my = meas[ok, 0], meas[ok, 1]
    xy = np.column_stack([p[:, 0] + c * mx - s * my, p[:, 1] + s * mx + c * my])
    labels = fclusterdata(xy, eps_m, criterion="distance", method="single")
    out = []
    for lab in np.unique(labels):
        k = labels == lab
        if k.sum() >= min_hits:
            out.append([*np.median(xy[k], axis=0), float(np.median(meas[ok, 2][k])), float(k.sum())])
    out = np.array(out, dtype=np.float64).reshape(-1, 4)
    return out[np.argsort(out[:, 2])]


def _match(A: np.ndarray, B: np.ndarray, gate_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one matches (rows of A, rows of B) with distance <= gate_m (Hungarian)."""
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    i, j = linear_sum_assignment(np.where(D <= gate_m, D, 1e6))
    ok = D[i, j] <= gate_m
    return i[ok], j[ok]


def tree_local_fit(
    trees: np.ndarray,
    map_xy: np.ndarray,
    init: EuclideanTransform,
    t_center: float,
    window_s: float = 10.0,
    theta: Optional[float] = None,
    gate_m: float = 0.25,
    search_m: float = 0.5,
    step_m: float = 0.02,
    min_matches: int = 2,
    iters: int = 4,
) -> Optional[Tuple[LocalFit, dict]]:
    """
    Fit odometry -> world from the odometry-frame trees (detection_trees) with t within window_s of t_center (SLAM
    clock), matched one to one to the map trees map_xy (world frame). With theta: translation only.

    1. Translation search: shifts of init by up to +/- search_m (step step_m). Each tree within gate_m of a map
       tree adds 1 - (d / gate_m)^2 to the score of the shift, and the shift with the highest score wins. This removes
       an init error smaller than search_m. A shift of one tree spacing can match as many trees as the correct
       shift, but with larger distances, so the score uses the distances and not only the number of matches.
    2. Refit: one-to-one matches within gate_m, rigid fit (rotation and translation with 3 or more matches and no
       theta), iters times.
    Return (LocalFit, info) or None if fewer than min_matches trees match. info: n_trees, n_matched, the score of
    the best shift, and the best score of the shifts more than gate_m from it (a value near the best score means
    the match is ambiguous, for example a shift of one tree spacing).
    """
    sel = np.abs(trees[:, 2] - t_center) <= window_s
    A = trees[sel, :2]
    if len(A) < min_matches or len(map_xy) < min_matches:
        return None
    rot = float(init.rotation) if theta is None else float(theta)
    base = EuclideanTransform(rotation=rot, translation=np.asarray(init.translation, dtype=np.float64))
    A0 = base(A)
    kd = cKDTree(map_xy)
    grid = np.arange(-search_m, search_m + 1e-9, step_m)
    best, scores = None, []
    for dx in grid:
        for dy in grid:
            d, _ = kd.query(A0 + [dx, dy])
            score = float(np.sum(np.clip(1.0 - (d / gate_m) ** 2, 0.0, None)))
            scores.append((dx, dy, score))
            if best is None or score > best[0]:
                best = (score, dx, dy)
    score_best, bx, by = best
    other = [sc for dx, dy, sc in scores if np.hypot(dx - bx, dy - by) > gate_m]
    T = EuclideanTransform(rotation=rot, translation=base.translation + [bx, by])
    i = j = np.zeros(0, dtype=int)
    for _ in range(iters):
        i, j = _match(T(A), map_xy, gate_m)
        if len(i) < min_matches:
            return None
        if theta is None and len(i) >= 3:
            T_new = EuclideanTransform()
            T_new.estimate(A[i], map_xy[j])
        else:
            R = EuclideanTransform(rotation=rot)
            T_new = EuclideanTransform(rotation=rot, translation=np.mean(map_xy[j] - R(A[i]), axis=0))
        T = T_new
    i, j = _match(T(A), map_xy, gate_m)
    if len(i) < min_matches:
        return None
    r = np.linalg.norm(T(A[i]) - map_xy[j], axis=1)
    span = float(np.sum(np.linalg.norm(np.diff(A, axis=0), axis=1))) if len(A) > 1 else 0.0
    info = {"n_trees": int(len(A)), "n_matched": int(len(i)), "search_score": round(score_best, 3),
            "search_other_score": round(float(max(other)), 3) if other else 0.0}
    return LocalFit(T, float(np.sqrt(np.mean(r ** 2))), int(len(A)), int(len(i)), span), info


def broadside_from_fit(
    fit: LocalFit,
    odom: np.ndarray,
    tree_xy: np.ndarray,
    t_guess: float,
    time_offset: float,
    odom_to_base_yaw: float,
    search_s: float = 5.0,
):
    """
    Broadside pose of one tree with an odometry -> world fit: the world pose at the time within search_s of t_guess
    (SLAM clock) where the tree is straight to the side of the robot (fwd = 0).
    Return dict(pose_xyt, t_slam, fwd, lat), or None if there is no odometry in the search window.
    """
    sel = np.abs(odom[:, 3] - (t_guess + time_offset)) <= search_s
    t_odom = odom[sel, 3]
    if t_odom.size < 2:
        return None
    # fwd: x of the tree in the robot base frame at each odometry sample
    fwd = np.array([pose_tf(*p).inverse([tree_xy])[0, 0] for p in world_poses(fit, odom, t_odom, odom_to_base_yaw)])
    # Time of the sign change of fwd next to the smallest |fwd| (linear between the two samples)
    k = int(np.argmin(np.abs(fwd)))
    t_best = t_odom[k]
    for j in (k - 1, k):
        if 0 <= j < len(fwd) - 1 and fwd[j] * fwd[j + 1] <= 0 and fwd[j] != fwd[j + 1]:
            t_best = t_odom[j] + fwd[j] / (fwd[j] - fwd[j + 1]) * (t_odom[j + 1] - t_odom[j])
            break
    pose = world_poses(fit, odom, np.array([t_best]), odom_to_base_yaw)[0]
    fwd_b, lat_b = pose_tf(*pose).inverse([tree_xy])[0]
    return {"pose_xyt": [float(v) for v in pose], "t_slam": float(t_best - time_offset),
            "fwd": float(fwd_b), "lat": float(lat_b)}


def broadside_from_odom(
    path: np.ndarray,
    odom: np.ndarray,
    tree_xy: np.ndarray,
    t_guess: float,
    time_offset: float,
    odom_to_base_yaw: float,
    window_s: float = 10.0,
    rotation_window_s: float = 60.0,
    search_s: float = 5.0,
):
    """
    Broadside pose of one tree from the odometry fitted to the robot_path (broadside_from_fit).
    The fit has two stages: the rotation from +/- rotation_window_s (more travel), then the translation from
    +/- window_s.
    Return dict(pose_xyt, t_slam, fwd, lat, fit, theta_fit), or None if a fit fails.
    """
    rot = robust_local_fit(path, odom, time_offset, t_guess, rotation_window_s)
    if rot is None:
        return None
    fit = robust_local_fit(path, odom, time_offset, t_guess, window_s, theta=rot.theta)
    if fit is None:
        return None
    b = broadside_from_fit(fit, odom, tree_xy, t_guess, time_offset, odom_to_base_yaw, search_s)
    if b is None:
        return None
    b.update(fit=fit, theta_fit=rot)
    return b


def broadside_from_trees(
    path: np.ndarray,
    odom: np.ndarray,
    trees: np.ndarray,
    map_xy: np.ndarray,
    tree_xy: np.ndarray,
    t_guess: float,
    time_offset: float,
    odom_to_base_yaw: float,
    window_s: float = 10.0,
    rotation_window_s: float = 60.0,
    search_s: float = 5.0,
    gate_m: float = 0.25,
    search_m: float = 0.5,
):
    """
    Broadside pose of one tree from the odometry fitted to the map trees (tree_local_fit), not to the robot_path.
    The robot_path fit over +/- rotation_window_s gives only the start value. Then the rotation comes from the trees
    within +/- rotation_window_s, and the translation from the trees within +/- window_s.
    Return dict(pose_xyt, t_slam, fwd, lat, fit, theta_fit, tree_fit_info), or None if a fit fails.
    """
    init = robust_local_fit(path, odom, time_offset, t_guess, rotation_window_s)
    if init is None:
        return None
    r = tree_local_fit(trees, map_xy, init.tf, t_guess, rotation_window_s, gate_m=gate_m, search_m=search_m)
    if r is None:
        return None
    rot, _ = r
    r = tree_local_fit(trees, map_xy, rot.tf, t_guess, window_s, theta=rot.theta, gate_m=gate_m, search_m=search_m)
    if r is None:
        return None
    fit, info = r
    b = broadside_from_fit(fit, odom, tree_xy, t_guess, time_offset, odom_to_base_yaw, search_s)
    if b is None:
        return None
    b.update(fit=fit, theta_fit=rot, tree_fit_info=info)
    return b
