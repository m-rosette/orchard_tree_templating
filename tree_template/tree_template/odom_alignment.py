"""
odom_alignment.py

Puts the robot odometry into the SLAM world frame, near one time, so the broadside pose of a tree comes from a
smooth trajectory. This module does not import ROS.

Why: save_best_particle_map records /odom_slam_best, the pose of the best particle at each moment. When a different
particle becomes the best one, the pose jumps (more than 1 m in the saved runs). The odometry is smooth but drifts
and is in its own frame. Near one time, a rigid 2-D transform (rotation + translation) maps the odometry onto the
SLAM path. A trimmed fit ignores the jumps.

Arrays:
    path   (N, 4) [x, y, yaw, t]  SLAM robot_path (world frame, SLAM clock)
    odom   (M, 4) [x, y, yaw, t]  odometry (odometry frame, odometry clock: t_odom = t_slam + time_offset)

time_offset cannot be found from the path: the path scatter (about 0.5 m RMS in the saved runs) is larger than the
motion in the offset range. When SLAM ran on a bag replay with the bag clock, /odom_slam_best has the stamps of the
odometry it used, and with the odometry time = bag receive time the offset is 0.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def interp_xy(odom: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Odometry x, y at times t (linear)."""
    return np.stack([np.interp(t, odom[:, 3], odom[:, 0]), np.interp(t, odom[:, 3], odom[:, 1])], axis=1)


def interp_yaw(odom: np.ndarray, t) -> np.ndarray:
    return _wrap(np.interp(t, odom[:, 3], np.unwrap(odom[:, 2])))


def fit_rigid_2d(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    """Least-squares (theta, t) with B ~ R(theta) @ A + t (Kabsch in 2-D)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    a, b = A - ca, B - cb
    s = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    c = np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])
    theta = float(np.arctan2(s, c))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return theta, cb - R @ ca


def apply_rigid_2d(theta: float, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return P @ R.T + t


@dataclass
class LocalFit:
    theta: float               # rotation odometry -> world (rad)
    t: np.ndarray              # translation odometry -> world (m)
    rms_inlier_m: float        # RMS residual of the inliers
    n_used: int
    n_inliers: int
    span_m: float              # travel distance inside the window (a short span makes theta weak)


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
    Fit odometry -> world from the SLAM path poses in [t_center - window_s, t_center + window_s].
    If theta is given, fit only the translation (with that rotation).
    Trimmed: after each fit, drop the poses with residual > max(3 * median residual, 5 cm).
    Return None if there are too few path poses, or the odometry does not cover the window.
    """
    m = np.abs(path[:, 3] - t_center) <= window_s
    t_odom = path[m, 3] + time_offset
    m_idx = np.nonzero(m)[0]
    cover = (t_odom >= odom[0, 3]) & (t_odom <= odom[-1, 3])
    m_idx, t_odom = m_idx[cover], t_odom[cover]
    if m_idx.size < min_points:
        return None
    A = interp_xy(odom, t_odom)
    B = path[m_idx, :2]

    def fit(k):
        if theta is None:
            return fit_rigid_2d(A[k], B[k])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return theta, np.mean(B[k] - A[k] @ R.T, axis=0)

    keep = np.ones(len(A), dtype=bool)
    for _ in range(iters):
        th, t = fit(keep)
        r = np.linalg.norm(apply_rigid_2d(th, t, A) - B, axis=1)
        new_keep = r <= max(3.0 * float(np.median(r[keep])), 0.05)
        if new_keep.sum() < min_points or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    th, t = fit(keep)
    r = np.linalg.norm(apply_rigid_2d(th, t, A) - B, axis=1)
    span = float(np.sum(np.linalg.norm(np.diff(A, axis=0), axis=1)))
    return LocalFit(th, t, float(np.sqrt(np.mean(r[keep] ** 2))), int(len(A)), int(keep.sum()), span)


def world_poses(fit: LocalFit, odom: np.ndarray, t_odom: np.ndarray, odom_to_base_yaw: float) -> np.ndarray:
    """
    (K, 3) world [x, y, yaw] of the robot base at odometry times t_odom.
    yaw = odometry yaw + fit.theta + odom_to_base_yaw (the base frame yaw from the odometry child frame).
    """
    xy = apply_rigid_2d(fit.theta, fit.t, interp_xy(odom, t_odom))
    yaw = _wrap(interp_yaw(odom, t_odom) + fit.theta + odom_to_base_yaw)
    return np.column_stack([xy, yaw])


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
    Broadside pose of one tree from the odometry, then the time in [t_guess - search_s, t_guess + search_s]
    (SLAM clock) where the tree is straight to the side of the robot (fwd = 0).

    Two-stage fit around t_guess: the rotation from +/- rotation_window_s (the odometry yaw drifts slowly, and a
    short window has too little travel for a good rotation against the path noise), then the translation from
    +/- window_s.

    Return dict(pose_xyt, t_slam, fwd, lat, fit, theta_fit) or None if a fit fails.
    """
    rot = robust_local_fit(path, odom, time_offset, t_guess, rotation_window_s)
    if rot is None:
        return None
    fit = robust_local_fit(path, odom, time_offset, t_guess, window_s, theta=rot.theta)
    if fit is None:
        return None
    sel = np.abs(odom[:, 3] - (t_guess + time_offset)) <= search_s
    t_odom = odom[sel, 3]
    if t_odom.size < 2:
        return None
    P = world_poses(fit, odom, t_odom, odom_to_base_yaw)
    d = tree_xy[None, :] - P[:, :2]
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    fwd = c * d[:, 0] + s * d[:, 1]
    # fwd crosses 0 between two odometry samples: interpolate the time of the crossing
    k = int(np.argmin(np.abs(fwd)))
    t_best = t_odom[k]
    for j in (k - 1, k):
        if 0 <= j < len(fwd) - 1 and fwd[j] * fwd[j + 1] <= 0 and fwd[j] != fwd[j + 1]:
            a = fwd[j] / (fwd[j] - fwd[j + 1])
            t_best = t_odom[j] + a * (t_odom[j + 1] - t_odom[j])
            break
    pose = world_poses(fit, odom, np.array([t_best]), odom_to_base_yaw)[0]
    dx, dy = tree_xy - pose[:2]
    cy, sy = np.cos(pose[2]), np.sin(pose[2])
    return {
        "pose_xyt": [float(v) for v in pose],
        "t_slam": float(t_best - time_offset),
        "fwd": float(cy * dx + sy * dy),
        "lat": float(-sy * dx + cy * dy),
        "fit": fit,
        "theta_fit": rot,
    }
