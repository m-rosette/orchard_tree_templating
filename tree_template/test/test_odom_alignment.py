"""
Checks for odom_alignment (no ROS): a synthetic run with a known true trajectory.

Run: python3 -m pytest test/test_odom_alignment.py -v
"""

import numpy as np

from tree_template.odom_alignment import broadside_from_odom, robust_local_fit


def _run(seed=0, theta_odom=np.pi + 0.02, odom_to_base=np.pi):
    """
    True robot in the world: drives along a slightly curved row at 0.3 m/s, heading +X (base frame yaw).
    Odometry: the true trajectory in a different frame (rotation theta_odom, translation), 40 Hz.
    The odometry child frame is turned by odom_to_base from the base frame (as the farm-ng /filter/state).
    SLAM path: the true pose at 7 Hz + 0.4 m noise + 1.5 m jumps (best particle changes), with repeated stamps.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 60.0, 0.025)
    x = 0.3 * t
    y = 0.02 * np.sin(t / 10.0)
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    true = np.column_stack([x, y, yaw, t])

    c, s = np.cos(theta_odom), np.sin(theta_odom)
    R = np.array([[c, -s], [s, c]])
    oxy = (true[:, :2] - np.array([5.0, -2.0])) @ R            # world = R @ odom + (5, -2)
    odom = np.column_stack([oxy, yaw - theta_odom - odom_to_base, t])

    k = np.arange(0, len(t), 6)
    path = true[k].copy()
    path[:, :2] += rng.normal(0, 0.4, size=(len(k), 2))
    jumps = rng.choice(len(k), size=len(k) // 10, replace=False)
    path[jumps, :2] += rng.choice([-1.5, 1.5], size=(len(jumps), 2))
    return true, odom, path


def test_rotation_fit_ignores_jumps():
    true, odom, path = _run()
    f = robust_local_fit(path, odom, 0.0, 30.0, window_s=60.0)
    assert abs(np.degrees(f.theta) - np.degrees(np.pi + 0.02 - 2 * np.pi)) < 1.0
    assert f.n_inliers < f.n_used          # the jumps are rejected


def test_broadside_recovers_true_pose():
    true, odom, path = _run()
    tree = np.array([9.0, 1.8])            # true broadside at x = 9.0 m, t = 30 s
    b = broadside_from_odom(path, odom, tree, t_guess=28.0, time_offset=0.0, odom_to_base_yaw=np.pi)
    assert b is not None
    assert abs(b["t_slam"] - 30.0) < 0.5
    assert abs(b["fwd"]) < 1e-3
    assert abs(b["lat"] - 1.8) < 0.15
    true_pose = true[np.argmin(abs(true[:, 3] - b["t_slam"]))]
    assert np.hypot(*(np.array(b["pose_xyt"][:2]) - true_pose[:2])) < 0.15
    assert abs(np.degrees(b["pose_xyt"][2] - true_pose[2])) < 1.5


def test_broadside_none_without_coverage():
    true, odom, path = _run()
    assert broadside_from_odom(path, odom[:10], np.array([9.0, 1.8]), 30.0, 0.0, np.pi) is None
