"""
odom_alignment.py

Puts the robot odometry into the SLAM world frame near one time, so the broadside pose of a tree comes from a
smooth trajectory. No ROS imports.

The SLAM robot_path (/odom_slam_best) is the pose of the best particle at each time and jumps when the best particle
changes. Near one time, a trimmed rigid 2-D fit maps the odometry onto the path and ignores the jumps.

Arrays (rows [x, y, yaw, t]):
    path  SLAM robot_path: world frame, SLAM clock
    odom  odometry: odometry frame, odometry clock (t_odom = t_slam + time_offset; 0 for SLAM on a replay of the
          same bag with the bag clock)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
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
    Broadside pose of one tree: the world pose (from the odometry) at the time within search_s of t_guess (SLAM
    clock) where the tree is straight to the side of the robot (fwd = 0).
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
    return {
        "pose_xyt": [float(v) for v in pose],
        "t_slam": float(t_best - time_offset),
        "fwd": float(fwd_b),
        "lat": float(lat_b),
        "fit": fit,
        "theta_fit": rot,
    }
